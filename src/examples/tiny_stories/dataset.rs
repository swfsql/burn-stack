//! Character-level [TinyStories-GPT4-clean] corpus: a stream of
//! single-character tokens over a **case-folded ASCII** alphabet, one *story*
//! per training item.
//!
//! [TinyStories-GPT4-clean]: https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean
//!
//! # Alphabet
//!
//! The dataset documentation states (and its cleaning pipeline verifies) that
//! it contains exactly 74 distinct ASCII characters: the 52 cased letters plus
//! ``\n !"$',-.0123456789:;?``. Case-folding the letters leaves [`ALPHABET`]:
//! 48 tokens, and every one of them occurs. That is the whole vocabulary. There
//! is no `<unk>`, no `<bos>`, and no padding class
//! (`pad_vocab_size_multiple = 1`). So every logit that the model emits is a
//! character that it can legitimately produce.
//!
//! # Download
//!
//! The dataset is a single 673 MB parquet file (one column, `text`, one row
//! per story, 2,669 ZSTD row groups of 1,024 rows). It is downloaded
//! **whole**, once, exactly as
//! [`MnistDataset`](super::super::mnist::dataset::MnistDataset) downloads its
//! IDX files, and cached at `~/.cache/burn-dataset/tinystories-gpt4-clean/`.
//! The read is lazy, so only the row groups up to the requested rows are
//! decompressed.
//!
//! The stories that come out are normalized and cached again, as text: one
//! file per `(split, story count)`, with the records divided by
//! [`STORY_SEPARATOR`]. So a second run reads a few MB of text, and never
//! opens (or needs) the parquet. A cache whose record count disagrees with its
//! name is rebuilt in place.
//!
//! The splits follow the row ranges that the dataset card suggests: rows
//! `0..10k` are test, `10k..20k` validation, and `20k..` training. The rows are
//! pre-shuffled, so a contiguous range is already a random sample.
//!
//! # Items, windows and runs
//!
//! One **item is one story**, and nothing is spliced between two stories. A
//! story is a self-contained example, and no separator character stands in for
//! its boundary. Leading and trailing whitespace is stripped, so the first
//! token of an item is always a real symbol. The model decides what (if
//! anything) marks the start, not the corpus. See
//! [`lm_output`](super::lm::lm_output) for the one hook that this side offers.
//!
//! A story (303–4,149 characters, median 724) is longer than one
//! back-propagation window. So it is walked in **windows** of `seq_len`
//! tokens, with the recurrent state carried across them
//! ([`lm::epoch_train`](super::lm::epoch_train)): this is the *run*. Its
//! length comes from the data, capped by `run_len`. The frontier gate can end
//! it early.
//!
//! Every slot of a batch walks its own story, and stories differ in length. So
//! a batch is padded to a whole number of windows of its longest story.
//! [`TinyStoriesBatch::scored`] records how many positions of each slot are
//! real, and [`lm_output`](super::lm::lm_output) scores only those. Padding
//! never reaches the loss or the accuracy.
//!
//! # Packed rows
//!
//! The padding of a batch of stories grows with the window. So a long window
//! wastes most of a batch on its short stories. A packed batch
//! ([`PackedStoriesDataset`], [`PackedStoriesBatcher`]) is one window of
//! `rows` rows. Each row holds whole stories, one after another, and each
//! story starts from a fresh state (see [`crate::utils::packing`]). The model
//! says where a story can start and how many opening slots it reserves
//! ([`PackLayout`]). The packer fills the rows first-fit, over a few open rows
//! ([`pack_rows`]). No state goes from one packed batch to the next. Only the
//! train split is packed.

#[cfg(test)]
mod tests;

use crate::utils::Packed;
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

/// Record separator of the **text cache**: ASCII `RS` (0x1E). It is outside
/// [`ALPHABET`], so `normalize` removes it from every story. So a story
/// *cannot* contain one, and a split of the file on it is exact.
///
/// A blank line would be the obvious choice, and it is the wrong one. The
/// dataset card allows `\n` as a paragraph separator, and does not forbid two
/// in a row. 5 of the first 32,768 training stories contain one. The separator
/// is a property of the cache file only. It is never encoded, so the model
/// never sees it.
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

    /// Encode `text`, and silently drop anything outside the alphabet. The
    /// cached corpus is normalized first, so this affects only user prompts.
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

    /// Number of rows in the split. For `Train`, these are the rows from its
    /// offset to the end of the file.
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

/// Case-fold `story`, drop the characters outside [`ALPHABET`] (very rare, ~5
/// per million), and trim the surrounding whitespace. The text is then exactly
/// the token stream, and it opens on a real symbol.
///
/// The drop of everything outside the alphabet also keeps [`STORY_SEPARATOR`]
/// unambiguous, because the separator is one of those characters. Interior
/// newlines stay exactly as the corpus has them, blank lines included.
fn normalize(story: &str) -> String {
    story
        .bytes()
        .filter(|&byte| VOCAB.token(byte).is_some())
        .map(|byte| byte.to_ascii_lowercase() as char)
        .collect::<String>()
        .trim()
        .to_owned()
}

/// The cached parquet file. Downloads it (whole, once) if it is not there.
fn parquet_file() -> PathBuf {
    let path = cache_dir().join(PARQUET);
    if path.exists() {
        return path;
    }
    let url = format!("https://huggingface.co/datasets/{DATASET}/resolve/main/{PARQUET}");
    println!("downloading {DATASET} ({PARQUET}, 673MB, cached afterwards)");
    let bytes = download_file_as_bytes(&url, PARQUET);
    // Write beside the target and rename, so an interrupted download cannot
    // leave a truncated file that later runs would try to read.
    let partial = path.with_extension("parquet.partial");
    std::fs::write(&partial, &bytes).expect("Failed to write the parquet cache");
    std::fs::rename(&partial, &path).expect("Failed to move the parquet cache into place");
    println!("cached {} bytes into {path:?}", bytes.len());
    path
}

/// Read `n_stories` rows of `split` out of the parquet file and normalize them.
///
/// The reader is lazy: one row group (1,024 rows) at a time. It is dropped at
/// the last group that the request touches, so a request never decompresses
/// past its own end. The offsets of the splits are small (20k rows, that is,
/// 20 groups) next to the 2,669 groups of the file, so the skipped rows cost
/// little.
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

/// The normalized stories of `split`. The first call downloads and extracts
/// them. Later calls read the text cache: `<split>-<n_stories>.txt`, the
/// stories joined by [`STORY_SEPARATOR`].
pub fn stories(split: Split, n_stories: usize) -> Vec<String> {
    let path = cache_dir().join(format!("{}-{n_stories}.txt", split.name()));
    if let Ok(cached) = std::fs::read_to_string(&path) {
        let stories: Vec<String> = cached.split(STORY_SEPARATOR).map(str::to_owned).collect();
        if stories.len() == n_stories {
            return stories;
        }
        // Not a corrupt file: a cache with a different separator. A blank line
        // is one, and a few stories contain one, so it split those stories in
        // two. A rebuild costs one pass over the (already downloaded) parquet,
        // so it beats a request for a manual delete.
        println!(
            "the text cache {path:?} holds {} records for {n_stories} stories \
             (a different separator). Rebuilding it.",
            stories.len(),
        );
    }
    let stories = read_parquet(split, n_stories);
    // The separator is the only structure of the cache file, so a story with
    // one would make the file unreadable. `normalize` drops every byte outside
    // the alphabet, and the separator is one of them. So this cannot fire.
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
    /// One token-id vector per story (shared, so a clone of the dataset is
    /// free).
    stories: Arc<Vec<Vec<u8>>>,
    /// Window length: the BPTT span of one forward.
    seq_len: usize,
}

impl TinyStoriesDataset {
    /// Load `n_stories` of `split` (download them once).
    ///
    /// `run_len` caps the number of windows that one story can use. A longer
    /// story is truncated to that many. The cap does nothing else: the run
    /// length itself comes from the story.
    pub fn new(split: Split, n_stories: usize, seq_len: usize, run_len: usize) -> Self {
        assert!(run_len >= 1, "a run holds at least one window");
        // Keep at most `run_len` whole windows plus the final target. A story
        // must have at least one token to predict from and one to predict.
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

    /// The windows that the split holds if every story is walked alone. This
    /// is the number of optimizer steps of an epoch at `batch_size = 1` when
    /// the frontier never stalls. A real batch runs the windows of its
    /// *longest* story, so it takes somewhat fewer steps over somewhat more
    /// padding.
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

/// A batch of stories, padded to a whole number of windows of the longest
/// story. [`window`](Self::window) cuts one window out of it.
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
    /// `Some` for a batch of packed rows (see [`PackedStoriesBatcher`]). Then
    /// the batch is one window, and a slot holds several stories.
    pub packed: Option<PackedRows>,
}

/// The layout of a batch of packed rows: where each story starts, and which
/// positions are scored.
#[derive(Clone, Debug)]
pub struct PackedRows {
    /// The resets and the opening slots of the stories, for the model.
    pub layout: Packed,
    /// `[batch_size, width]`: `true` at a scored position. These are the
    /// characters of each story after its first, plus its last opening slot
    /// (scored against its first character).
    pub score_bs: Tensor<2, Bool>,
}

impl TinyStoriesBatch {
    /// The batch, built by a dataloader worker on the host, moved to `device`
    /// by the thread that steps the model (see
    /// [`loader_device`](crate::examples::device::loader_device)).
    pub fn to_device(self, device: &Device) -> Self {
        use crate::examples::device::batch_int;
        Self {
            inputs: batch_int(self.inputs, device),
            targets: batch_int(self.targets, device),
            packed: self.packed.map(|packed| PackedRows {
                layout: packed.layout.to_device(device),
                score_bs: crate::utils::packing::bool_to_device(packed.score_bs, device),
            }),
            ..self
        }
    }

    /// The windows that the batch spans: those of its longest story. This is
    /// also the length of the run that [`epoch_train`](super::lm::epoch_train)
    /// walks.
    pub fn num_windows(&self) -> usize {
        self.inputs.dims()[1] / self.seq_len
    }

    /// Window `w` of the batch: the `[batch_size, seq_len]` slice of both
    /// tensors, with [`scored`](Self::scored) narrowed to what each slot still
    /// has left inside it (`0` for a story that ended earlier).
    ///
    /// Every batch slot advances together, so window `w` continues window
    /// `w - 1` in all of them. This makes one carried cache valid for the
    /// whole batch.
    pub fn window(&self, w: usize) -> Self {
        let seq_len = self.seq_len;
        assert!(
            w < self.num_windows(),
            "window {w} is past the batch ({} windows)",
            self.num_windows(),
        );
        // A packed batch is one window.
        if self.packed.is_some() {
            return self.clone();
        }
        Self {
            inputs: self.inputs.clone().narrow(1, w * seq_len, seq_len),
            targets: self.targets.clone().narrow(1, w * seq_len, seq_len),
            scored: self
                .scored
                .iter()
                .map(|&n| n.saturating_sub(w * seq_len).min(seq_len))
                .collect(),
            seq_len,
            packed: None,
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
        // One scored position per token except the first, which has no
        // predecessor to predict it from. A model that opens the sequence with
        // something of its own can also score that token, from the extra
        // output positions that `lm_output` reads. This side does not know and
        // does not ask.
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
            // Token 0 pads both sides. `scored` drops every padded position
            // from the loss, so the pad token matters only for the state of a
            // slot whose story is already over.
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
            packed: None,
        }
    }
}

// ===========================================================================
// Packed rows
// ===========================================================================

/// What the model asks of a packed row. The data side does not know why.
#[derive(Clone, Copy, Debug)]
pub struct PackLayout {
    /// A story starts only at a multiple of this position (the positions
    /// where the model accepts a reset, for example its chunk starts).
    pub align: usize,
    /// Opening slots in front of each story (the model puts its class latents
    /// there).
    pub lead: usize,
}

impl Default for PackLayout {
    /// A model with no constraint: a story can start anywhere, with no
    /// opening slots.
    fn default() -> Self {
        Self { align: 1, lead: 0 }
    }
}

impl PackLayout {
    /// Positions that a story of `len` tokens takes in a row: its opening
    /// slots, then every token but the last (the inputs).
    pub fn positions(&self, len: usize) -> usize {
        self.lead + len - 1
    }
}

/// Pack stories into rows of `width` positions, first-fit over `open_rows`
/// open rows. Returns the stories of each row, in row order.
///
/// `lens` holds the token count of each story, and `order` the order in which
/// to place them (a shuffle). A story goes into the first open row that has
/// room for it after the last story of that row (rounded up to
/// `layout.align`). When no open row has room, the oldest open row closes,
/// and a new row opens with the story. So a story moves ahead of its place in
/// `order` by at most `open_rows` rows. `open_rows = 1` keeps the order
/// exactly.
///
/// # Panics
/// If a story does not fit into an empty row.
pub fn pack_rows(
    lens: &[usize],
    order: &[usize],
    width: usize,
    layout: PackLayout,
    open_rows: usize,
) -> Vec<Vec<usize>> {
    assert!(open_rows >= 1, "the packer keeps at least one open row");
    assert!(layout.align >= 1, "a story starts at a multiple of at least 1");
    let mut rows = Vec::new();
    // Each open row: its stories and its first free (aligned) position.
    let mut open: std::collections::VecDeque<(Vec<usize>, usize)> = Default::default();
    for &story in order {
        let need = layout.positions(lens[story]);
        assert!(need <= width, "story {story} takes {need} positions, more than a row of {width}");
        let next = |used: usize| (used + need).next_multiple_of(layout.align);
        match open.iter_mut().find(|(_, used)| used + need <= width) {
            Some((stories, used)) => {
                stories.push(story);
                *used = next(*used);
            }
            None => {
                if open.len() == open_rows {
                    rows.push(open.pop_front().expect("an open row").0);
                }
                open.push_back((vec![story], next(0)));
            }
        }
    }
    rows.extend(open.into_iter().map(|(stories, _)| stories));
    rows
}

/// One packed row: the token ids of its stories, in row order.
#[derive(Clone, Debug)]
pub struct PackedItem {
    /// The stories of the row.
    pub stories: Vec<Vec<u8>>,
}

/// The train split as packed rows (see [`pack_rows`]). One item is one row.
pub struct PackedStoriesDataset {
    /// The stories, each truncated to fit into one row.
    stories: Arc<Vec<Vec<u8>>>,
    /// The stories of each row.
    rows: Vec<Vec<usize>>,
}

impl PackedStoriesDataset {
    /// Pack `n_stories` of `split` into rows of `width` positions, in the
    /// order of a shuffle with `seed`. A story too long for a row is
    /// truncated to fit.
    pub fn new(
        split: Split,
        n_stories: usize,
        width: usize,
        layout: PackLayout,
        open_rows: usize,
        seed: u64,
    ) -> Self {
        use rand::SeedableRng;
        use rand::seq::SliceRandom;
        assert!(width > layout.lead, "a row holds the opening slots and at least one token");
        // The inputs of a story are all its tokens but the last.
        let max_tokens = width - layout.lead + 1;
        let stories: Vec<Vec<u8>> = stories(split, n_stories)
            .iter()
            .map(|story| {
                let mut tokens = VOCAB.encode(story);
                tokens.truncate(max_tokens);
                tokens
            })
            .filter(|tokens| tokens.len() >= 2)
            .collect();
        let lens: Vec<usize> = stories.iter().map(Vec::len).collect();
        let mut order: Vec<usize> = (0..stories.len()).collect();
        order.shuffle(&mut rand_chacha::ChaCha8Rng::seed_from_u64(seed));
        let rows = pack_rows(&lens, &order, width, layout, open_rows);
        Self {
            stories: Arc::new(stories),
            rows,
        }
    }

    /// Total number of characters in the packed stories.
    pub fn num_tokens(&self) -> usize {
        self.stories.iter().map(Vec::len).sum()
    }

    /// The packed rows: the items of the dataset.
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }
}

impl Dataset<PackedItem> for PackedStoriesDataset {
    fn get(&self, index: usize) -> Result<PackedItem, DatasetError> {
        Ok(PackedItem {
            stories: self.rows[index].iter().map(|&s| self.stories[s].clone()).collect(),
        })
    }

    fn len(&self) -> usize {
        self.rows.len()
    }
}

/// Stacks [`PackedItem`]s into a one-window [`TinyStoriesBatch`] of
/// `[rows, width]`, with its [`PackedRows`].
///
/// A story takes `lead` opening slots, then its tokens but the last as
/// inputs. The next token is the target of each input. The last opening slot
/// is scored against the first token. The next story starts at the next
/// multiple of `align`. Token 0 fills the slots and the gaps, which are not
/// scored.
#[derive(Clone)]
pub struct PackedStoriesBatcher {
    /// Positions per row.
    width: usize,
    /// The constraints of the model.
    layout: PackLayout,
}

impl PackedStoriesBatcher {
    /// A batcher of rows of `width` positions.
    pub fn new(width: usize, layout: PackLayout) -> Self {
        Self { width, layout }
    }
}

impl Batcher<PackedItem, TinyStoriesBatch> for PackedStoriesBatcher {
    fn batch(&self, items: Vec<PackedItem>, device: &Device) -> TinyStoriesBatch {
        let (width, PackLayout { align, lead }) = (self.width, self.layout);
        let rows = items.len();
        let mut inputs = vec![0i32; rows * width];
        let mut targets = vec![0i32; rows * width];
        let mut score = vec![false; rows * width];
        let mut starts = vec![Vec::new(); rows];
        let mut scored = vec![0usize; rows];
        for (b, item) in items.iter().enumerate() {
            let row = b * width;
            let mut start = 0;
            for tokens in &item.stories {
                let n = tokens.len() - 1;
                assert!(start + lead + n <= width, "a packed row overflows its width");
                starts[b].push(start);
                let first = start + lead;
                if lead > 0 {
                    targets[row + first - 1] = tokens[0] as i32;
                    score[row + first - 1] = true;
                }
                for j in 0..n {
                    inputs[row + first + j] = tokens[j] as i32;
                    targets[row + first + j] = tokens[j + 1] as i32;
                    score[row + first + j] = true;
                }
                scored[b] += n + usize::from(lead > 0);
                start = (first + n).next_multiple_of(align);
            }
        }
        let shape = [rows, width];
        TinyStoriesBatch {
            inputs: Tensor::<1, Int>::from_ints(inputs.as_slice(), device).reshape(shape),
            targets: Tensor::<1, Int>::from_ints(targets.as_slice(), device).reshape(shape),
            scored,
            seq_len: width,
            packed: Some(PackedRows {
                layout: Packed::from_starts(&starts, lead, width, device),
                score_bs: Tensor::<1, Bool>::from_bool(
                    burn::tensor::TensorData::new(score, [rows * width]),
                    device,
                )
                .reshape(shape),
            }),
        }
    }
}
