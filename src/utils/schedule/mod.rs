//! # Layer schedules: virtual layer → real layer
//!
//! A [`Layers`](crate::modules::Layers) stack can run `n_virtual_layers`
//! virtual layers over only `n_real_layers` real layers (e.g. 48 virtual
//! layers over 12 real layers). Each virtual layer keeps its own cache, and
//! shares the tied parameters of its real layer. A layer schedule
//! ([`Schedule`]) maps the index of a virtual layer to the index of the real
//! layer that it runs.
//!
//! For **bidirectional** stacks, [`BidiSchedule`] also interleaves the two
//! passes: even virtual indices run the straight pass (→), and odd indices
//! run the reversed pass (←).
//!
//! Each variant has a worked example of its virtual→real map.
//!
//! A [`GradHorizon`] sits on top of that map. It says which of the virtual
//! layers back-propagate. It counts them **for each real layer**, so that no
//! real layer stays untrained, however a layer schedule spreads it.
//!
//! [`Applications`] reads the map the other way (which application of its
//! real layer each virtual layer is), for the parameters that a layer unties.

/// The layer schedule of a unidirectional layer stack: the map from the index
/// of each virtual layer to the index of its real layer.
#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Schedule {
    /// Runs the real layers in order, and starts again at real layer 0 after
    /// the last one.
    ///
    /// # Example
    /// - virtual len = 8, real len = 3:\
    ///   `  →    →    →      →    →    →      →    →       `\
    ///   `(0⇒0, 1⇒1, 2⇒2), (3⇒0, 4⇒1, 5⇒2), (6⇒0, 7⇒1, ...)`
    #[default]
    Cyclic,
    /// Stretches the real layers over the virtual layers: each group of
    /// consecutive virtual layers runs one real layer, in order.
    ///
    /// # Example
    /// - virtual len = 8, real len = 3:\
    ///   `  →    →    →      →    →    →      →    →       `\
    ///   `(0⇒0, 1⇒0, 2⇒0), (3⇒1, 4⇒1, 5⇒1), (6⇒2, 7⇒2, ...)`
    Stretched,
    /// Takes the real layer of each virtual layer from the index vector.
    ///
    /// # Example
    /// - virtual len = 8, real len = 3, custom = `[0, 1, 2, 2, 1, 0, 0, 0]`:\
    ///   `  →    →    →    →    →    →    →    →       `\
    ///   `(0⇒0, 1⇒1, 2⇒2, 3⇒2, 4⇒1, 5⇒0, 6⇒0, 7⇒0, ...)`
    Custom(Vec<usize>),
}

impl Schedule {
    /// Map `virtual_idx` (in `0..virtual_len`) to a real layer index in
    /// `0..real_len` according to this layer schedule.
    pub fn real_idx(&self, virtual_idx: usize, virtual_len: usize, real_len: usize) -> usize {
        match self {
            Schedule::Cyclic => virtual_idx % real_len,
            Schedule::Stretched => (virtual_idx * real_len) / virtual_len,
            Schedule::Custom(map) => *map.get(virtual_idx).unwrap(),
        }
    }

    /// The [`Applications`] of this layer schedule over `virtual_len` virtual
    /// layers.
    pub fn applications(&self, virtual_len: usize, real_len: usize) -> Applications {
        Applications::new(
            (0..virtual_len).map(|i| self.real_idx(i, virtual_len, real_len)),
            real_len,
        )
    }
}

/// Which **application** of its real layer each virtual layer is: the
/// virtual→real map read the other way, which counts how often the virtual
/// layers below already ran the same real layer. It indexes an untied
/// parameter (see [`crate::utils::untied`]). [`GradHorizon::Depth`] counts
/// the same thing from the top.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Applications {
    /// Per virtual layer: how many virtual layers below it share its real layer.
    pub index: Vec<usize>,
    /// Per real layer: its number of applications. At least one, because a
    /// real layer that the layer schedule never reaches is still built.
    pub count: Vec<usize>,
}

impl Applications {
    /// From the real index of each virtual layer, bottom-up.
    pub fn new(real_of: impl IntoIterator<Item = usize>, real_len: usize) -> Self {
        let mut count = vec![0; real_len];
        let index = real_of
            .into_iter()
            .map(|real| {
                count[real] += 1;
                count[real] - 1
            })
            .collect();
        for c in &mut count {
            *c = (*c).max(1);
        }
        Self { index, count }
    }
}

/// Which (virtual) layers of a stack build an autodiff graph: the type of
/// [`Layers::grad_horizon`](crate::modules::Layers::grad_horizon).
///
/// The other layers run on the inner (non-autodiff) backend, and retain no
/// activation. A horizon is a **mask**, not one cut. The stack runs each
/// untracked span of the mask on the inner backend, and lifts back at the cut
/// that ends it. A mask can hold any number of untracked spans.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GradHorizon {
    /// Back-propagate the last `K` applications of **every real layer**.
    ///
    /// `K` counts for each *real layer*, not for the stack. So every real
    /// layer keeps a tracked application, whatever the layer schedule does
    /// with it. This is the point: a plain suffix of `K` virtual layers
    /// reaches every real layer only under [`Schedule::Cyclic`]. It would
    /// leave every [`Schedule::Stretched`] real layer except the topmost with
    /// no gradient.
    ///
    /// # Example
    /// - virtual len = 8, real len = 3, `Depth(1)` (`T` = tracked):
    ///   - [`Schedule::Cyclic`] (`0 1 2 0 1 2 0 1`): `. . . . . T T T`, a
    ///     single cut, at `virtual_len - K·real_len` on an even stack.
    ///   - [`Schedule::Stretched`] (`0 0 0 1 1 1 2 2`): `. . T . . T . T`, one
    ///     cut **for each real layer**, at the tail of each run.
    ///
    /// A stack **without** virtual layers applies each real layer once, so
    /// any `K >= 1` tracks all of it. [`Self::last`] states the suffix mask
    /// for that case. [`Schedule::Custom`] has no canonical run to take a tail
    /// of, so it takes a [`Self::Mask`] instead (`Depth` panics on it).
    Depth(usize),
    /// Explicit mask with one flag for each virtual layer, `true` =
    /// back-propagated. Its length must be the virtual-layer count of the
    /// stack.
    Mask(Vec<bool>),
}

impl GradHorizon {
    /// A [`Self::Mask`] that tracks the **last `k`** of `virtual_len` layers:
    /// the single-cut suffix horizon, written out. It is what [`Self::Depth`]
    /// gives under [`Schedule::Cyclic`] with `k = K·real_len`. It is also the
    /// natural horizon to cut a stack without virtual layers (`Depth` cannot
    /// cut it).
    pub fn last(k: usize, virtual_len: usize) -> Self {
        GradHorizon::Mask((0..virtual_len).map(|i| i + k >= virtual_len).collect())
    }

    /// Resolve to one `tracked` flag per virtual layer. `schedule` is the
    /// layer schedule of the stack, `None` when it runs no virtual layers
    /// (each real layer applied once, `virtual_len == real_len`).
    ///
    /// # Panics
    /// A [`Self::Mask`] of the wrong length, or a [`Self::Depth`] against a
    /// [`Schedule::Custom`].
    pub fn tracked(
        &self,
        schedule: Option<&Schedule>,
        virtual_len: usize,
        real_len: usize,
    ) -> Vec<bool> {
        match self {
            GradHorizon::Mask(mask) => {
                assert_eq!(
                    mask.len(),
                    virtual_len,
                    "GradHorizon::Mask needs one flag per virtual layer",
                );
                mask.clone()
            }
            GradHorizon::Depth(k) => {
                assert!(
                    !matches!(schedule, Some(Schedule::Custom(_))),
                    "GradHorizon::Depth is undefined for Schedule::Custom: a hand-written \
                     virtual→real map has no canonical run to take the last K applications \
                     of. State the cuts with GradHorizon::Mask.",
                );
                // Walk the stack downwards, and keep the `k` topmost
                // applications of each real layer: one contiguous suffix under
                // `Cyclic`, one tail per run under `Stretched`.
                let mut tracked = vec![false; virtual_len];
                let mut kept = vec![0usize; real_len];
                for i in (0..virtual_len).rev() {
                    let real = schedule.map_or(i, |s| s.real_idx(i, virtual_len, real_len));
                    if kept[real] < *k {
                        kept[real] += 1;
                        tracked[i] = true;
                    }
                }
                tracked
            }
        }
    }
}

/// The layer schedule of a bidirectional layer stack. It interleaves the
/// straight pass (→, even indices) and the reversed pass (←, odd indices).
#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum BidiSchedule {
    /// Even virtual layers run the straight pass (→) on the even real layers,
    /// and odd virtual layers run the reversed pass (←) on the odd real
    /// layers. Each pass cycles over its real layers.
    ///
    /// # Example
    /// - virtual len = 10, real len = 4:\
    ///   `   →    ←      →    ←        →    ←      →    ←        →    ←          `\
    ///   `[(0⇒0, 1⇒1), (2⇒2, 3⇒3)], [(4⇒0, 5⇒1), (6⇒2, 7⇒3)], [(8⇒0, 9⇒1), (...)]`
    #[default]
    StridedCyclic,
    /// Even virtual layers run the straight pass (→) on the even real layers,
    /// and odd virtual layers run the reversed pass (←) on the odd real
    /// layers. Each pass stretches its real layers over its virtual layers.
    ///
    /// # Example
    /// - virtual len = 10, real len = 4:\
    ///   `   →    ←      →    ←      →    ←        →    ←      →    ←          `\
    ///   `[(0⇒0, 1⇒1), (2⇒0, 3⇒1), (4⇒0, 5⇒1)], [(6⇒2, 7⇒3), (8⇒2, 9⇒3), (...)]`
    StridedStretched,
    /// Each bidirectional pair (the virtual layers `2i` and `2i + 1`) runs one
    /// real layer for its straight pass (→) and its reversed pass (←). The
    /// pairs cycle over the real layers.
    ///
    /// # Example
    /// - virtual len = 10, real len = 4:\
    ///   `   →    ←      →    ←      →    ←      →    ←        →    ←          `\
    ///   `[(0⇒0, 1⇒0), (2⇒1, 3⇒1), (4⇒2, 5⇒2), (6⇒3, 7⇒3)], [(8⇒0, 9⇒0), (...)]`
    SymmetricCyclic,
    /// Each bidirectional pair (the virtual layers `2i` and `2i + 1`) runs one
    /// real layer for its straight pass (→) and its reversed pass (←). The
    /// real layers stretch over the pairs.
    ///
    /// # Example
    /// - virtual len = 10, real len = 4:\
    ///   `   →    ←      →    ←       →    ←               →    ←        →    ←   `\
    ///   `[(0⇒0, 1⇒0), (2⇒0, 3⇒0)],[(4⇒1, 5⇒1), (...)], [(6⇒2, 7⇒2)], [(8⇒3, 9⇒3)]`
    SymmetricStretched,
    /// Takes the real layer of each virtual layer from the index vector.
    ///
    /// # Example
    /// - virtual len = 10, real len = 4, custom = `[0, 1, 2, 2, 1, 0, 0, 0, 3, 2]`:\
    ///   `   →    ←        →    ←        →    ←        →    ←        →    ←            `\
    ///   `[(0⇒0, 1⇒1)], [(2⇒2, 3⇒2)], [(4⇒1, 5⇒0)], [(6⇒0, 7⇒0)], [(8⇒3, 9⇒2)], [(...)]`
    Custom(Vec<usize>),
}

impl BidiSchedule {
    /// Map `virtual_idx` (in `0..virtual_len`) to a real layer index in
    /// `0..real_len`. An even `virtual_idx` runs the straight pass, and an odd
    /// one runs the reversed pass. The layer schedule cycles or stretches over
    /// the outer index `virtual_idx / 2`.
    pub fn real_idx(&self, virtual_idx: usize, virtual_len: usize, real_len: usize) -> usize {
        let virtual_outer_idx = virtual_idx / 2;
        let virtual_outer_len = virtual_len / 2;
        match self {
            BidiSchedule::StridedCyclic => {
                let odd_len = real_len / 2;
                let even_len = odd_len + real_len % 2;
                let is_even = virtual_idx.is_multiple_of(2);
                if is_even {
                    (virtual_outer_idx % even_len) * 2
                } else {
                    (virtual_outer_idx % odd_len) * 2 + 1
                }
            }
            BidiSchedule::StridedStretched => {
                let odd_len = real_len / 2;
                let even_len = odd_len + real_len % 2;
                let is_even = virtual_idx.is_multiple_of(2);
                if is_even {
                    ((virtual_outer_idx * even_len) / virtual_outer_len) * 2
                } else {
                    ((virtual_outer_idx * odd_len) / virtual_outer_len) * 2 + 1
                }
            }
            BidiSchedule::SymmetricCyclic => virtual_outer_idx % real_len,
            BidiSchedule::SymmetricStretched => (virtual_outer_idx * real_len) / virtual_outer_len,
            BidiSchedule::Custom(map) => *map.get(virtual_idx).unwrap(),
        }
    }

    /// The [`Applications`] of this layer schedule over `virtual_len` virtual
    /// layers, with both passes counted against the real layer that they
    /// index.
    pub fn applications(&self, virtual_len: usize, real_len: usize) -> Applications {
        Applications::new(
            (0..virtual_len).map(|i| self.real_idx(i, virtual_len, real_len)),
            real_len,
        )
    }
}

#[cfg(test)]
mod tests;
