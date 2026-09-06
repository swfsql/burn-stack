//! Character-level [TinyStories-GPT4-clean] corpus: a stream of single-character
//! tokens over a **case-folded ASCII** alphabet, one *story* per training item.
//!
//! [TinyStories-GPT4-clean]: https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean
//!
//! # Alphabet
//!
//! The dataset is documented (and verified by its cleaning pipeline) to contain
//! exactly 74 distinct ASCII characters — the 52 cased letters plus
//! ``\n !"$',-.0123456789:;?``. Case-folding the letters leaves [`ALPHABET`]:
//! 48 tokens, every one of which actually occurs. That is the whole vocabulary;
//! there is no `<unk>`, no `<bos>`, and no padding class
//! (`pad_vocab_size_multiple = 1`), so every logit the model emits is a
//! character it can legitimately produce.
//!
//! # Download
//!
//! The dataset is a single 673 MB parquet file (one column, `text`; one row per
//! story; 2,669 ZSTD row groups of 1,024 rows). It is downloaded **whole**, once,
//! exactly the way [`MnistDataset`](super::super::mnist::dataset::MnistDataset)
//! downloads its IDX files, and cached at
//! `~/.cache/burn-dataset/tinystories-gpt4-clean/`. Reading it is lazy, so only
//! the row groups up to the requested rows are ever decompressed.
//!
//! The stories that come out are normalized and cached again, as text, one file
//! per `(split, story count)`, records divided by [`STORY_SEPARATOR`] — so a
//! second run reads a few MB of text and never opens (or needs) the parquet at
//! all. A cache whose record count disagrees with its name was written by an
//! older separator and is rebuilt in place.
//!
//! Splits follow the dataset card's suggested row ranges (the rows are
//! pre-shuffled, so a contiguous range is already a random sample): rows
//! `0..10k` are test, `10k..20k` validation, and `20k..` training.
//!
//! # Items, windows and runs
//!
//! One **item is one story**, and nothing is spliced between two of them: a story
//! is a self-contained example, and no separator character stands in for its
//! boundary. Leading and trailing whitespace is stripped, so the first token of
//! an item is always a real symbol. What (if anything) marks the start is the
//! model's business, not the corpus's — see [`lm_output`](super::lm::lm_output)
//! for the one hook this side offers.
//!
//! A story (303–4,149 characters, median 724) is longer than one back-propagation
//! window, so it is walked in **windows** of `seq_len` tokens with the recurrent
//! state carried across them ([`lm::epoch_train`](super::lm::epoch_train)) — the
//! *run*. Its length comes from the data, capped by `run_len`; the frontier gate
//! is what ends it early.
//!
//! Every slot of a batch walks its own story, and stories differ in length, so a
//! batch is padded to a whole number of windows of its longest one.
//! [`TinyStoriesBatch::scored`] records how many positions of each slot are real,
//! and [`lm_output`](super::lm::lm_output) scores only those — padding never
//! reaches the loss or the accuracy.

use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use burn_dataset::{Dataset, DatasetError, network::downloader::download_file_as_bytes};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use std::path::PathBuf;
use std::sync::Arc;

// ===========================================================================
// Vocabulary
// ===========================================================================

/// Every character the corpus contains, after case-folding: 48 tokens, in ASCII
/// order. Token ids are indices into this string.
pub const ALPHABET: &str = "\n !\"$',-.0123456789:;?abcdefghijklmnopqrstuvwxyz";

/// Number of character classes (= the model's `vocab_size`).
pub const VOCAB_SIZE: usize = ALPHABET.len();

/// Token id reserved by [`Vocab`] for "not in the alphabet".
const NO_TOKEN: u8 = u8::MAX;

/// Record separator of the **text cache**: ASCII `RS` (0x1E), which is outside
/// [`ALPHABET`] and therefore removed from every story by [`normalize`] — so a
/// story *cannot* contain one, and splitting the file on it is exact.
///
/// A blank line would be the obvious choice and is the wrong one: the dataset
/// card allows `\n` as a paragraph separator without forbidding two in a row,
/// and 5 of the first 32,768 training stories do carry one. It is a property of
/// the cache file only — never encoded, so the model never sees it.
pub const STORY_SEPARATOR: &str = "\u{1e}";

/// Byte ↔ token-id tables for [`ALPHABET`], with `A-Z` folded onto `a-z`.
pub struct Vocab {
    /// `byte → token id`, [`NO_TOKEN`] for bytes outside the alphabet.
    to_id: [u8; 256],
    /// `token id → byte`.
    to_byte: [u8; VOCAB_SIZE],
}

/// The one vocabulary, built at compile time.
pub const VOCAB: Vocab = Vocab::new();

impl Vocab {
    /// Build the tables from [`ALPHABET`].
    pub const fn new() -> Self {
        let alphabet = ALPHABET.as_bytes();
        let mut to_id = [NO_TOKEN; 256];
        let mut to_byte = [0u8; VOCAB_SIZE];
        let mut i = 0;
        while i < alphabet.len() {
            let byte = alphabet[i];
            to_id[byte as usize] = i as u8;
            to_byte[i] = byte;
            // The upper-case half of the corpus folds onto the same token.
            if byte.is_ascii_lowercase() {
                to_id[byte.to_ascii_uppercase() as usize] = i as u8;
            }
            i += 1;
        }
        Self { to_id, to_byte }
    }

    /// Token id of `byte` (case-folded), or `None` when it is outside the
    /// alphabet.
    pub const fn token(&self, byte: u8) -> Option<u8> {
        match self.to_id[byte as usize] {
            NO_TOKEN => None,
            id => Some(id),
        }
    }

    /// The character a token id stands for.
    pub const fn character(&self, token: u8) -> char {
        self.to_byte[token as usize] as char
    }

    /// Encode `text`, silently dropping anything outside the alphabet (the
    /// cached corpus is normalized first, so this only bites on user prompts).
    pub fn encode(&self, text: &str) -> Vec<u8> {
        text.bytes().filter_map(|byte| self.token(byte)).collect()
    }

    /// Decode token ids back to text.
    pub fn decode(&self, tokens: &[u8]) -> String {
        tokens.iter().map(|&t| self.character(t)).collect()
    }
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Download + cache
// ===========================================================================

/// The Hugging Face dataset id.
const DATASET: &str = "karpathy/tinystories-gpt4-clean";

/// The dataset's single parquet file, as named in the repository.
const PARQUET: &str = "tinystories_gpt4_clean.parquet";

/// Which of the dataset card's suggested row ranges to read.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Split {
    /// Rows `20_000..` — the training stories.
    Train,
    /// Rows `10_000..20_000` — the validation stories.
    Valid,
    /// Rows `0..10_000` — the held-out test stories.
    Test,
}

impl Split {
    /// First row index of this split, per the dataset card.
    pub const fn offset(self) -> usize {
        match self {
            Split::Test => 0,
            Split::Valid => 10_000,
            Split::Train => 20_000,
        }
    }

    /// Number of rows the split has (unbounded for `Train`).
    pub const fn capacity(self) -> usize {
        match self {
            Split::Test | Split::Valid => 10_000,
            Split::Train => 2_712_634,
        }
    }

    /// Cache-file stem.
    pub const fn name(self) -> &'static str {
        match self {
            Split::Test => "test",
            Split::Valid => "valid",
            Split::Train => "train",
        }
    }
}

/// `~/.cache/burn-dataset/tinystories-gpt4-clean/`, created on demand.
fn cache_dir() -> PathBuf {
    let dir = dirs::home_dir()
        .expect("Could not get home directory")
        .join(".cache")
        .join("burn-dataset")
        .join("tinystories-gpt4-clean");
    std::fs::create_dir_all(&dir).expect("Failed to create the cache directory");
    dir
}

/// Case-fold `story`, drop the (vanishingly rare, ~5 per million) characters
/// outside [`ALPHABET`], and trim the surrounding whitespace — after which the
/// text is exactly the token stream, opening on a real symbol.
///
/// Dropping everything outside the alphabet is also what keeps
/// [`STORY_SEPARATOR`] unambiguous: it is one of those characters. Interior
/// newlines are left exactly as the corpus has them, blank lines included.
fn normalize(story: &str) -> String {
    story
        .bytes()
        .filter(|&byte| VOCAB.token(byte).is_some())
        .map(|byte| byte.to_ascii_lowercase() as char)
        .collect::<String>()
        .trim()
        .to_owned()
}

/// The cached parquet file, downloading it (whole, once) if it is not there yet.
fn parquet_file() -> PathBuf {
    let path = cache_dir().join(PARQUET);
    if path.exists() {
        return path;
    }
    let url = format!("https://huggingface.co/datasets/{DATASET}/resolve/main/{PARQUET}");
    println!("downloading {DATASET} ({PARQUET}, 673MB, cached afterwards)");
    let bytes = download_file_as_bytes(&url, PARQUET);
    // Write beside the target and rename, so an interrupted download cannot
    // leave a truncated file that later runs would happily try to read.
    let partial = path.with_extension("parquet.partial");
    std::fs::write(&partial, &bytes).expect("Failed to write the parquet cache");
    std::fs::rename(&partial, &path).expect("Failed to move the parquet cache into place");
    println!("cached {} bytes into {path:?}", bytes.len());
    path
}

/// Read `n_stories` rows of `split` out of the parquet file and normalize them.
///
/// The reader is lazy — one row group (1,024 rows) at a time — and is dropped at
/// the last one the request touches, so a request never decompresses past its own
/// end of the file. The splits' offsets are small (20k rows, i.e. 20 groups) next
/// to the file's 2,669, so what is skipped costs little.
fn read_parquet(split: Split, n_stories: usize) -> Vec<String> {
    assert!(
        n_stories <= split.capacity(),
        "{n_stories} stories exceed the {} split ({} rows)",
        split.name(),
        split.capacity(),
    );
    let file = std::fs::File::open(parquet_file()).expect("Failed to open the parquet cache");
    let reader = SerializedFileReader::new(file).expect("Failed to read the parquet file");
    println!(
        "reading {n_stories} {} stories from {PARQUET}",
        split.name(),
    );

    let stories: Vec<String> = reader
        .get_row_iter(None)
        .expect("Failed to iterate the parquet rows")
        .skip(split.offset())
        .take(n_stories)
        .map(|row| {
            let row = row.expect("Failed to read a parquet row");
            normalize(row.get_string(0).expect("the `text` column is a string"))
        })
        .collect();
    assert_eq!(
        stories.len(),
        n_stories,
        "the {} split ran out of rows",
        split.name(),
    );
    stories
}

/// The normalized stories of `split`, downloading and extracting them on the
/// first call and reading the text cache — `<split>-<n_stories>.txt`, the
/// stories joined by [`STORY_SEPARATOR`] — afterwards.
pub fn stories(split: Split, n_stories: usize) -> Vec<String> {
    let path = cache_dir().join(format!("{}-{n_stories}.txt", split.name()));
    if let Ok(cached) = std::fs::read_to_string(&path) {
        let stories: Vec<String> = cached.split(STORY_SEPARATOR).map(str::to_owned).collect();
        if stories.len() == n_stories {
            return stories;
        }
        // Not a corrupt file: a cache written when the separator was a blank
        // line, which a handful of stories carry inside them and which was
        // therefore splitting those in two. Rebuilding costs one pass over the
        // (already downloaded) parquet, so it beats asking for a manual delete.
        println!(
            "the text cache {path:?} holds {} records for {n_stories} stories \
             (an older separator); rebuilding it",
            stories.len(),
        );
    }
    let stories = read_parquet(split, n_stories);
    // The separator is the cache file's only structure, so a story carrying one
    // would make the file unreadable. `normalize` drops every byte outside the
    // alphabet, and the separator is one of them, so this cannot fire.
    assert!(
        !stories.iter().any(|s| s.contains(STORY_SEPARATOR)),
        "a story contains the cache separator",
    );
    std::fs::write(&path, stories.join(STORY_SEPARATOR)).expect("Failed to write the corpus cache");
    println!("cached {} stories into {path:?}", stories.len());
    stories
}

// ===========================================================================
// Dataset + batcher
// ===========================================================================

/// One training item: the token ids of one whole story.
#[derive(Clone, Debug)]
pub struct TinyStoriesItem {
    /// Token ids of the story, `[story_len]`.
    pub tokens: Vec<u8>,
}

/// The split's stories, one per item.
pub struct TinyStoriesDataset {
    /// One token-id vector per story (shared, so cloning the dataset is free).
    stories: Arc<Vec<Vec<u8>>>,
    /// Window length: the BPTT span of one forward.
    seq_len: usize,
}

impl TinyStoriesDataset {
    /// Load (downloading once) `n_stories` of `split`.
    ///
    /// `run_len` caps how many windows one story may spend: a longer story is
    /// truncated to that many, which is the only thing the cap does — the run
    /// length itself comes from the story.
    pub fn new(split: Split, n_stories: usize, seq_len: usize, run_len: usize) -> Self {
        assert!(run_len >= 1, "a run holds at least one window");
        // A story must have at least one token to predict from and one to
        // predict, i.e. one whole window plus its final target at most.
        let max_tokens = run_len.saturating_mul(seq_len).saturating_add(1);
        let stories: Vec<Vec<u8>> = stories(split, n_stories)
            .iter()
            .map(|story| {
                let mut tokens = VOCAB.encode(story);
                tokens.truncate(max_tokens);
                tokens
            })
            .filter(|tokens| tokens.len() >= 2)
            .collect();
        assert!(
            !stories.is_empty(),
            "the {} corpus holds no story long enough to score",
            split.name(),
        );
        Self {
            stories: Arc::new(stories),
            seq_len,
        }
    }

    /// Total number of characters across the split's stories.
    pub fn num_tokens(&self) -> usize {
        self.stories.iter().map(Vec::len).sum()
    }

    /// Windows the split holds if every story were walked alone — the number of
    /// optimizer steps an epoch takes at `batch_size = 1` when the frontier
    /// never stalls. A real batch runs the windows of its *longest* story, so it
    /// takes somewhat fewer steps over somewhat more padding.
    pub fn num_windows(&self) -> usize {
        self.stories
            .iter()
            .map(|tokens| (tokens.len() - 1).div_ceil(self.seq_len))
            .sum()
    }
}

impl Dataset<TinyStoriesItem> for TinyStoriesDataset {
    fn get(&self, index: usize) -> Result<TinyStoriesItem, DatasetError> {
        Ok(TinyStoriesItem {
            tokens: self.stories[index].clone(),
        })
    }

    fn len(&self) -> usize {
        self.stories.len()
    }
}

/// A batch of stories, padded to a whole number of windows of the longest one;
/// [`window`](Self::window) cuts one window out of it.
#[derive(Clone, Debug)]
pub struct TinyStoriesBatch {
    /// Input token ids, `[batch_size, num_windows · seq_len]`.
    pub inputs: Tensor<2, Int>,
    /// Next-character targets (the inputs shifted by one),
    /// `[batch_size, num_windows · seq_len]`.
    pub targets: Tensor<2, Int>,
    /// Real (non-padding) scored positions per batch slot: `story_len - 1` for
    /// the whole batch, and what is left of it for a [`window`](Self::window).
    pub scored: Vec<usize>,
    /// Window length this batch was padded against.
    pub seq_len: usize,
}

impl TinyStoriesBatch {
    /// Windows the batch spans — its longest story's, and the length of the run
    /// [`epoch_train`](super::lm::epoch_train) walks.
    pub fn num_windows(&self) -> usize {
        self.inputs.dims()[1] / self.seq_len
    }

    /// Window `w` of the batch: the `[batch_size, seq_len]` slice of both
    /// tensors, with [`scored`](Self::scored) narrowed to what each slot still
    /// has left inside it (`0` for a story that ended earlier).
    ///
    /// Every batch slot advances together, so window `w` continues window
    /// `w - 1` in all of them — which is what makes one carried cache valid for
    /// the whole batch.
    pub fn window(&self, w: usize) -> Self {
        let seq_len = self.seq_len;
        assert!(
            w < self.num_windows(),
            "window {w} is past the batch ({} windows)",
            self.num_windows(),
        );
        Self {
            inputs: self.inputs.clone().narrow(1, w * seq_len, seq_len),
            targets: self.targets.clone().narrow(1, w * seq_len, seq_len),
            scored: self
                .scored
                .iter()
                .map(|&n| n.saturating_sub(w * seq_len).min(seq_len))
                .collect(),
            seq_len,
        }
    }
}

/// Stacks [`TinyStoriesItem`]s into a [`TinyStoriesBatch`], padding them to a
/// whole number of `seq_len` windows.
#[derive(Clone)]
pub struct TinyStoriesBatcher {
    /// Window length the batch is padded against.
    seq_len: usize,
}

impl TinyStoriesBatcher {
    /// A batcher padding to whole windows of `seq_len` tokens.
    pub fn new(seq_len: usize) -> Self {
        Self { seq_len }
    }
}

impl Batcher<TinyStoriesItem, TinyStoriesBatch> for TinyStoriesBatcher {
    fn batch(&self, items: Vec<TinyStoriesItem>, device: &Device) -> TinyStoriesBatch {
        let batch_size = items.len();
        // One scored position per token but the first, which has no predecessor
        // to be predicted from. A model that opens the sequence with something of
        // its own can score that one too, out of the extra output positions
        // `lm_output` picks up; this side neither knows nor asks.
        let scored: Vec<usize> = items.iter().map(|item| item.tokens.len() - 1).collect();
        let windows = scored
            .iter()
            .map(|n| n.div_ceil(self.seq_len))
            .max()
            .expect("a batch holds at least one story");
        let padded = windows * self.seq_len;

        let mut inputs = Vec::with_capacity(batch_size * padded);
        let mut targets = Vec::with_capacity(batch_size * padded);
        for (item, &n) in items.iter().zip(&scored) {
            // Token 0 pads both sides; every padded position is dropped from the
            // loss by `scored`, so which token it is only matters for the state
            // of a slot whose own story is already over.
            inputs.extend(item.tokens[..n].iter().map(|&t| t as i32));
            targets.extend(item.tokens[1..].iter().map(|&t| t as i32));
            inputs.resize(inputs.len() + (padded - n), 0);
            targets.resize(targets.len() + (padded - n), 0);
        }
        let shape = [batch_size, padded];
        TinyStoriesBatch {
            inputs: Tensor::<1, Int>::from_ints(inputs.as_slice(), device).reshape(shape),
            targets: Tensor::<1, Int>::from_ints(targets.as_slice(), device).reshape(shape),
            scored,
            seq_len: self.seq_len,
        }
    }
}
