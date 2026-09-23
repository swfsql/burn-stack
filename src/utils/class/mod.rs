//! A *class token* / *class latent* is a learnable embedding spliced into the
//! sequence: a register in the style of the transformer `[CLS]`, which the
//! model can read and write through. A container inserts them at its input
//! boundary:
//!
//! - a network at its input, for [`ClassToken`] (width = the input feature
//!   width),
//! - a layer in its working sequence, for [`ClassLatent`] (width = `d_model`).
//!
//! They make the sequence longer for everything downstream. A container can
//! carry any number of them. The markers below say *where* each one lands. A
//! single `Param<Tensor<2>>` of shape `[num_markers, width]` holds the
//! embeddings (row `i` ↔ marker `i`).
//!
//! # Placement
//!
//! Each marker names an index into the *original* sequence of length `L`:
//! `Start` 0, `Middle` `L/2`, `End` `L`, `Custom(k)` `k`. The markers are
//! inserted in the order of that index. At a shared index, the kind breaks
//! the tie (`Start` < `Middle` < `End` < `Custom`), then the `Vec` order.
//!
//! One rule places three of the four kinds: the marker is emitted immediately
//! **before** the user token at its index. `Start` (0) opens the sequence,
//! `Middle` (`L/2`) splits it, and `Custom(k)` precedes token `k`. `End` is
//! the only exception, because it has no token to precede. It **closes** the
//! sequence, after the last user token.
//!
//! `Custom` is uniform in `k`, so a `Custom(k ≥ L)` never lands: there is no
//! token `k` to precede. If the caller feeds tokens past the announced `L`,
//! it lands then, still *before* the next token. It never trails. An
//! open-ended stream (no hint) is also never closed, for the same reason that
//! `End` needs the length hint below.
//!
//! So `step` returns the output of the **last** token that it emitted (and
//! leaves the state after it). This is the user token, unless `End` follows
//! it: `End` is then the latest token of the sequence. The markers emitted
//! before the user token only leave their mark on the state.
//!
//! `prime` is the call that reads *those* back. It emits the markers that
//! wait for the next user token, without that token, so it needs no input
//! data. It returns the last of them (`None` when none were waiting). `prime`
//! is exactly the opening half of `step`. So `prime` followed by `step` emits
//! what that `step` alone would emit, in the same order. Seedless generation
//! is `prime` → sample → `step` → sample → … `End` is never primed. It
//! *closes* the sequence, so it belongs to the call that carries the last
//! user token. This is why it is the one marker that `step` (and `forward`)
//! already returns.
//!
//! # Streamed placement
//!
//! Placement is **streamed**, in the same way for `forward` (a chunk of the
//! sequence) and `step` (a single token). [`ClassCursors`] carries one
//! `full_len` hint (the length of the whole sequence that the call is part
//! of), plus one cursor per level. Each cursor records how much of the output
//! sequence of *that* level the earlier calls already emitted. From those:
//!
//! - A marker whose output position is behind the cursor was emitted by an
//!   earlier call, and is skipped. So `Start` fires only while the cursor is
//!   still at 0, and a resumed stream does not insert it again.
//! - `Middle`/`End` resolve only against the whole sequence, so they
//!   **panic** without a `full_len` hint. `Start`/`Custom` do not depend on
//!   the length, and work on an open-ended stream.
//! - A marker that lands exactly at the end of a chunk is emitted by that
//!   chunk only if the chunk *closes* the sequence. Otherwise it opens the
//!   next chunk. So a split of a sequence at any point leaves the placement
//!   unchanged.
//!
//! Without cursors (`None`), `forward` treats its argument as the whole
//! sequence (`full_len` = its length, cursors at 0), and `step` injects
//! nothing.

use crate::utils::Padding;
use burn::config::Config;
use burn::module::Param;
use burn::nn::Initializer;
use burn::prelude::*;

/// Position marker for a learnable class **token** inserted into the input
/// sequence of a *network* (embedding width = the network input width /
/// "d_input").
#[derive(Config, Debug)]
pub enum ClassToken {
    /// Prepend before the whole sequence (index 0).
    Start,
    /// Insert before the middle token of the original sequence (index `L/2`).
    /// Needs a [`ClassCursors::full_len`] hint.
    Middle,
    /// **Close** the sequence: appended after its last token (index `L`). It
    /// is the only marker that trails a token, not precedes it. So a closing
    /// `step` returns it. Needs a [`ClassCursors::full_len`] hint.
    End,
    /// Insert before the token `index` of the original sequence, for any
    /// `index`. So a marker at or past the end never lands (no such token).
    /// The exception: the caller feeds tokens past the announced length, and
    /// the marker then precedes the next one.
    Custom(usize),
}

/// Position marker for a learnable class **latent** inserted into the working
/// sequence of a *layer* (embedding width = `d_model`).
#[derive(Config, Debug)]
pub enum ClassLatent {
    /// Prepend before the whole sequence (index 0).
    Start,
    /// Insert before the middle token of the original sequence (index `L/2`).
    /// Needs a [`ClassCursors::full_len`] hint.
    Middle,
    /// **Close** the sequence: appended after its last token (index `L`). It
    /// is the only marker that trails a token, not precedes it. So a closing
    /// `step` returns it. Needs a [`ClassCursors::full_len`] hint.
    End,
    /// Insert before the token `index` of the original sequence, for any
    /// `index`. So a marker at or past the end never lands (no such token).
    /// The exception: the caller feeds tokens past the announced length, and
    /// the marker then precedes the next one.
    Custom(usize),
}

/// Shared behaviour of the [`ClassToken`] / [`ClassLatent`] position markers,
/// letting one generic helper place either kind.
pub trait ClassMarker: Clone {
    /// Insertion index measured against the *original* sequence length `orig_len`.
    fn insert_pos(&self, orig_len: usize) -> usize;
    /// Tie-break rank among markers sharing an index (`Start`<`Middle`<`End`<`Custom`).
    fn group_rank(&self) -> usize;
    /// Whether the position of this marker is defined only against the whole
    /// sequence (`Middle`/`End`). Its placement then needs a
    /// [`ClassCursor::full_len`] hint.
    fn needs_full_len(&self) -> bool;
    /// Whether this marker *closes* the sequence: it trails the last token,
    /// not precedes one. Only `End` does.
    fn closes_sequence(&self) -> bool;
}

macro_rules! impl_class_marker {
    ($ty:ty) => {
        impl ClassMarker for $ty {
            fn insert_pos(&self, orig_len: usize) -> usize {
                match self {
                    Self::Start => 0,
                    Self::Middle => orig_len / 2,
                    Self::End => orig_len,
                    Self::Custom(index) => *index,
                }
            }
            fn group_rank(&self) -> usize {
                match self {
                    Self::Start => 0,
                    Self::Middle => 1,
                    Self::End => 2,
                    Self::Custom(_) => 3,
                }
            }
            fn needs_full_len(&self) -> bool {
                matches!(self, Self::Middle | Self::End)
            }
            fn closes_sequence(&self) -> bool {
                matches!(self, Self::End)
            }
        }
    };
}
impl_class_marker!(ClassToken);
impl_class_marker!(ClassLatent);

/// Placement state of **one** class-marker level (the markers of one
/// container): how far into the *output* sequence of that level the previous
/// calls got, and the length that positions are measured against.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClassCursor {
    /// Output-sequence position reached so far: user tokens *and* already
    /// emitted class markers (0 ⇒ the sequence has not started).
    pub offset: usize,
    /// Length of the whole sequence that this level receives, without its own
    /// markers. `None` ⇒ an open-ended stream: `Start`/`Custom` still place
    /// exactly, `Middle`/`End` panic.
    pub full_len: Option<usize>,
}

impl ClassCursor {
    /// A cursor at `offset` measured against `full_len`.
    pub fn at(offset: usize, full_len: Option<usize>) -> Self {
        Self { offset, full_len }
    }

    /// A new cursor for a call that covers the entire sequence of length
    /// `len`. `forward` assumes this when it gets no cursors.
    pub fn whole(len: usize) -> Self {
        Self {
            offset: 0,
            full_len: Some(len),
        }
    }

    /// Whether a chunk of `chunk_len` tokens placed from this cursor is the
    /// entire sequence: nothing emitted before it, and nothing announced after.
    pub fn covers_whole(&self, chunk_len: usize) -> bool {
        self.offset == 0 && self.full_len == Some(chunk_len)
    }
}

/// Everything that a `forward` (chunk) or `step` (single token) call needs to
/// place the class tokens / class latents of a whole network: one full-length
/// hint, plus one cursor per class-marker level. Pass the **same** value to
/// every call of a sequence. Each call advances the cursors that it uses, so
/// the next call resumes exactly where this one stopped.
///
/// The levels nest. Each cursor counts the sequence that *its* level sees,
/// which already includes the markers that the levels below it spliced in:
///
/// ```text
/// network     LatentNetwork's own ClassTokens  (before `in_proj`)
/// stack       Layers'/BidiLayers' own ClassLatents
/// per_layer   one cursor per virtual layer, for that Layer's ClassLatents
/// ```
///
/// [`Self::full_len`] is the length of the user sequence given to the
/// outermost call. The lengths of the inner levels come from it.
///
/// To read a marker back out of a *chunked* `forward`:
///
/// 1. Its position in the whole output is `class_*_output_indices(full_len)`
///    of the container.
/// 2. Subtract the cursor of the level as it was *before* that call. The
///    result is its index inside the output of the chunk.
///
/// A `step` returns one token: the last token that it emitted. This is the
/// user token, unless an `End` marker trails it. `End` is then the true last
/// token of the sequence.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClassCursors {
    /// Total length of the user sequence that all the calls form together.
    /// `None` for an open-ended stream (then `Middle`/`End` markers panic).
    pub full_len: Option<usize>,
    /// Cursor of the [`ClassToken`]s of a network (not used by a bare layer
    /// stack).
    pub network: usize,
    /// Cursor of the [`ClassLatent`]s of a layer container.
    pub stack: usize,
    /// One cursor per **virtual** layer, for the per-layer [`ClassLatent`]s.
    /// When it is empty, the first call sizes it (with zeros).
    pub per_layer: Vec<usize>,
}

impl ClassCursors {
    /// Cursors at the start of a sequence of known total length. This form
    /// enables `Middle`/`End` markers, and a whole-sequence `forward` uses it
    /// when it gets no cursors.
    pub fn new(full_len: usize) -> Self {
        Self {
            full_len: Some(full_len),
            ..Default::default()
        }
    }

    /// Cursors at the start of an open-ended stream (unknown total length):
    /// `Start`/`Custom` markers place exactly, `Middle`/`End` panic.
    pub fn stream() -> Self {
        Self::default()
    }

    /// Size [`Self::per_layer`] for a stack of `n` virtual layers (idempotent).
    pub(crate) fn fit(&mut self, n: usize) {
        if self.per_layer.is_empty() {
            self.per_layer = vec![0; n];
        }
        assert_eq!(
            self.per_layer.len(),
            n,
            "one class-latent cursor per virtual layer"
        );
    }

    /// Enter the inner level. Its sequence is longer by the `markers` that
    /// this level splices in and that land ([`landing_count`]). Returns the
    /// previous hint, for [`Self::leave`].
    pub(crate) fn enter<M: ClassMarker>(&mut self, markers: &[M]) -> Option<usize> {
        let saved = self.full_len;
        self.full_len = saved.map(|l| l + landing_count(markers, l));
        saved
    }

    /// Leave the inner level, restoring the hint [`Self::enter`] returned.
    pub(crate) fn leave(&mut self, saved: Option<usize>) {
        self.full_len = saved;
    }
}

/// How many of `markers` land in a sequence of `len` tokens: all except a
/// `Custom` at or past its end, which has no token to precede. The level above
/// sees a sequence that is this much longer, and places its own `Middle`/`End`
/// against it.
pub fn landing_count<M: ClassMarker>(markers: &[M], len: usize) -> usize {
    markers
        .iter()
        .filter(|m| m.closes_sequence() || m.insert_pos(len) < len)
        .count()
}

/// Panic if the position of a marker needs the whole sequence length and none
/// is known. `Middle`/`End` cannot be placed from a chunk (or a single token).
pub fn assert_full_len_known<M: ClassMarker>(
    markers: &[M],
    full_len: Option<usize>,
    who: &str,
) {
    assert!(
        full_len.is_some() || !markers.iter().any(|m| m.needs_full_len()),
        "{who}: Middle/End class markers need a full-length hint (ClassCursors::new)"
    );
}

/// Which of `markers` fall inside the next `chunk_len` user tokens, and where.
///
/// Returns `(at, marker)` pairs in output order: insert `markers[marker]`
/// *before* the `at`-th token of the chunk (`at == chunk_len` ⇒ after the last
/// one). `cursor` is advanced past the whole chunk, its insertions included.
pub fn class_chunk_plan<M: ClassMarker>(
    markers: &[M],
    chunk_len: usize,
    cursor: &mut ClassCursor,
    who: &str,
) -> Vec<(usize, usize)> {
    class_plan(markers, chunk_len, cursor, false, who)
}

/// [`class_chunk_plan`] for a **prime**. The chunk carries no user token of
/// its own beyond the `chunk_len` tokens that a lower level handed up (`0` at
/// the level where the call enters). One more user token is still to come.
///
/// So it differs only on the trailing edge of the chunk. The markers that wait
/// there for that next token are emitted now (they precede it either way).
/// `End` trails the last token, not precedes one, so the call that carries
/// that token emits it. A cursor already at the announced end has no next
/// token, so the plan is then empty.
pub fn class_prime_plan<M: ClassMarker>(
    markers: &[M],
    chunk_len: usize,
    cursor: &mut ClassCursor,
    who: &str,
) -> Vec<(usize, usize)> {
    class_plan(markers, chunk_len, cursor, true, who)
}

/// The shared placement loop of [`class_chunk_plan`] / [`class_prime_plan`].
fn class_plan<M: ClassMarker>(
    markers: &[M],
    chunk_len: usize,
    cursor: &mut ClassCursor,
    prime: bool,
    who: &str,
) -> Vec<(usize, usize)> {
    if markers.is_empty() {
        cursor.offset += chunk_len;
        return Vec::new();
    }
    assert_full_len_known(markers, cursor.full_len, who);
    let positions = class_marker_output_indices(markers, cursor.full_len.unwrap_or(usize::MAX));

    // User tokens consumed after this chunk: the output positions behind the
    // cursor, minus the markers among them, plus this chunk. More tokens than
    // an announced `full_len` are allowed. Every marker inside that length is
    // already placed by then, so the extra tokens stream through.
    let start = cursor.offset;
    let consumed = start - positions.iter().filter(|&&p| p < start).count() + chunk_len;
    // Whether this chunk reaches the announced end, that is, carries the last
    // user token. This is the only place where an `End` can go. A prime
    // carries no token of its own, so it never closes anything.
    let closes = !prime && cursor.full_len == Some(consumed);
    // Whether a prime can flush the markers that wait at the end of the chunk.
    // They need a next token to precede: a further announced user token, or an
    // open-ended stream.
    let flush = prime && cursor.full_len != Some(consumed);

    let mut order: Vec<usize> = (0..markers.len()).collect();
    order.sort_by_key(|&i| positions[i]);

    let mut out = start; // running output position
    let mut at = 0usize; // chunk tokens placed before it
    let mut plan = Vec::new();
    for i in order {
        let p = positions[i];
        if p < start {
            continue; // emitted by an earlier call
        }
        let need = p - out; // user tokens preceding this marker
        if at + need > chunk_len {
            break; // the token that it precedes is in a later chunk, and so is the marker
        }
        if at + need == chunk_len {
            // Nothing is left in this chunk to precede. Only two cases belong
            // here: a closing `End` on the chunk that ends the sequence, or, on
            // a prime, the markers due before the next token. A `Custom` waits
            // for its token (for one at or past the end, forever).
            let closing = closes && markers[i].closes_sequence();
            let pending = flush && !markers[i].closes_sequence();
            if !closing && !pending {
                break;
            }
        }
        at += need;
        out = p + 1;
        plan.push((at, i));
    }
    cursor.offset = out + (chunk_len - at);
    plan
}

/// Splice the learnable class markers `emb` (`[k, width]`, row `i` ↔
/// `markers[i]`) that fall inside the chunk `x` (`[batch, chunk_len, width]`).
/// Returns the longer chunk, and advances `cursor` past it.
///
/// `markers` empty (or none of them landing in this chunk) ⇒ `x` unchanged.
pub fn insert_class_markers<M: ClassMarker>(
    x: Tensor<3>,
    markers: &[M],
    emb: Option<&Param<Tensor<2>>>,
    cursor: &mut ClassCursor,
    who: &str,
) -> Tensor<3> {
    insert_class_markers_padded(x, None, markers, emb, cursor, who).0
}

/// [`insert_class_markers`] over a padded batch: the rows are spliced exactly
/// as there, and `padding` (`None` ⇒ every row real) follows them (see
/// [`Padding::splice`]).
pub fn insert_class_markers_padded<M: ClassMarker>(
    x: Tensor<3>,
    padding: Option<Padding>,
    markers: &[M],
    emb: Option<&Param<Tensor<2>>>,
    cursor: &mut ClassCursor,
    who: &str,
) -> (Tensor<3>, Option<Padding>) {
    let [_batch, chunk_len, width] = x.dims();
    let whole = cursor.covers_whole(chunk_len);
    let plan = class_chunk_plan(markers, chunk_len, cursor, who);
    if plan.is_empty() {
        return (x, padding);
    }
    let padding = padding.map(|p| p.splice(&plan, markers, whole, who));
    (splice_class_rows(x, &plan, &class_emb_table(markers, emb, width)), padding)
}

/// The class-marker embedding table (`[markers.len(), width]`), checked against
/// the markers it places and the feature width it is spliced into. Only called
/// where a marker is about to be emitted, so the param is present.
pub fn class_emb_table<M: ClassMarker>(
    markers: &[M],
    emb: Option<&Param<Tensor<2>>>,
    width: usize,
) -> Tensor<2> {
    let emb = emb
        .expect("class-token markers present but no embedding param")
        .val();
    assert_eq!(
        emb.dims(),
        [markers.len(), width],
        "one embedding row per class marker"
    );
    emb
}

/// The tensor half of [`insert_class_markers`]: splice the rows a
/// [`class_chunk_plan`] selected into `x` along its **sequence axis 1**,
/// broadcasting each row over every other axis.
///
/// It is rank-generic, so the same placement lands in a plain
/// `[batch, sequence, width]` chunk and in the Multi-Gate residual streams
/// `[batch, sequence, n_stream, width]`. A class marker must enter *every*
/// stream, because the streams carry the residual.
pub fn splice_class_rows<const D: usize>(
    x: Tensor<D>,
    plan: &[(usize, usize)],
    emb: &Tensor<2>,
) -> Tensor<D> {
    if plan.is_empty() {
        return x;
    }
    let dims = x.dims();
    let chunk_len = dims[1];
    // One marker row broadcast to a single sequence position: `[1, ‥, 1, width]`
    // expanded over the batch (and, for the streams, the stream axis).
    let mut row_shape = [1usize; D];
    row_shape[D - 1] = dims[D - 1];
    let mut row_dims = dims;
    row_dims[1] = 1;
    let row = |i: usize| {
        emb.clone()
            .narrow(0, i, 1)
            .reshape(row_shape)
            .expand(row_dims)
    };

    let mut segments: Vec<Tensor<D>> = Vec::with_capacity(2 * plan.len() + 1);
    let mut taken = 0usize; // chunk tokens emitted so far
    for &(at, i) in plan {
        if at > taken {
            segments.push(x.clone().narrow(1, taken, at - taken));
            taken = at;
        }
        segments.push(row(i));
    }
    if taken < chunk_len {
        segments.push(x.narrow(1, taken, chunk_len - taken));
    }
    Tensor::cat(segments, 1)
}

/// Width of the class embeddings (`[num_markers, width]`). A `prime` sizes its
/// rows by it, because it has no token to read the width from. Only called
/// where a marker is about to be emitted, so the param is present.
pub fn class_emb_width(emb: Option<&Param<Tensor<2>>>) -> usize {
    emb.expect("class-token markers present but no embedding param")
        .val()
        .dims()[1]
}

/// The embedding row of marker `i` as one broadcast token (`[batch, width]`):
/// the `step` counterpart of a slice of [`insert_class_markers`].
pub fn class_row(
    emb: Option<&Param<Tensor<2>>>,
    i: usize,
    batch: usize,
    width: usize,
) -> Tensor<2> {
    emb.expect("class-token markers present but no embedding param")
        .val()
        .narrow(0, i, 1)
        .expand([batch, width])
}

/// The output-sequence position of each marker (in `Vec` order) for an input of
/// length `orig_len`, without materialising any tensor. It mirrors the
/// placement in [`insert_class_markers`], and is useful to read a class token
/// back out.
///
/// A marker that never lands (a `Custom` at or past the end, with no token to
/// precede) reports the position that it *would* take. This is
/// `>= orig_len + (number of markers that land)`, that is, past the emitted
/// sequence.
pub fn class_marker_output_indices<M: ClassMarker>(
    markers: &[M],
    orig_len: usize,
) -> Vec<usize> {
    let k = markers.len();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by_key(|&i| (markers[i].insert_pos(orig_len), markers[i].group_rank(), i));
    let mut cursor = 0usize;
    let mut out_len = 0usize;
    let mut out_index = vec![0usize; k];
    for &i in &order {
        let p = markers[i].insert_pos(orig_len).min(orig_len);
        if p > cursor {
            out_len += p - cursor;
            cursor = p;
        }
        out_index[i] = out_len;
        out_len += 1;
    }
    out_index
}

/// Build the embedding param for `n` class markers of the given `width`
/// (`None` when there are none: Burn has no zero-width tensors).
pub fn init_class_emb(n: usize, width: usize, device: &Device) -> Option<Param<Tensor<2>>> {
    (n > 0).then(|| {
        Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        }
        .init([n, width], device)
    })
}
