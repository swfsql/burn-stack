//! # Virtual-layer → real-weight scheduling
//!
//! A [`Layers`](crate::modules::Layers) stack can run `n_virtual_layers`
//! logical passes over only `n_real_layers` weight sets (e.g. 48 logical from
//! 12 real). Each virtual layer keeps its own cache but shares parameters. A
//! [`Schedule`] maps a virtual layer index to the index of the real weights to
//! use.
//!
//! For **bidirectional** stacks, [`BidiSchedule`] also interleaves the two
//! directions: even virtual indices run the straight (→) pass, and odd
//! indices run the reverse (←) pass.
//!
//! Each variant has a worked virtual→real mapping example.
//!
//! A [`GradHorizon`] sits on top of that mapping. It says which of the
//! virtual layers back-propagate, counted **per real layer**, so that no
//! weight set is left untrained, however a schedule spreads it.
//!
//! [`Applications`] reads the map the other way (which application of its
//! real layer each virtual layer is), for the parameters that a layer unties.

/// How a unidirectional layer stack maps virtual layer indices to real
/// (weight-bearing) layer indices.
#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Schedule {
    /// Fills virtual positions by wrapping around the real schedule in a looping fashion.
    ///
    /// # Example
    /// - virtual len = 8, real len = 3:  
    ///   `  →    →    →      →    →    →      →    →       `  
    ///   `(0⇒0, 1⇒1, 2⇒2), (3⇒0, 4⇒1, 5⇒2), (6⇒0, 7⇒1, ...)`
    #[default]
    Cyclic,
    /// Fills virtual positions by stretching the real schedule.
    ///
    /// # Example
    /// - virtual len = 8, real len = 3:  
    ///   `  →    →    →      →    →    →      →    →       `  
    ///   `(0⇒0, 1⇒0, 2⇒0), (3⇒1, 4⇒1, 5⇒1), (6⇒2, 7⇒2, ...)`
    Stretched,
    /// Fills virtual positions by referring to the index vector.
    ///
    /// # Example
    /// - virtual len = 8, real len = 3, custom = `[0, 1, 2, 2, 1, 0, 0, 0]`:  
    ///   `  →    →    →    →    →    →    →    →       `  
    ///   `(0⇒0, 1⇒1, 2⇒2, 3⇒2, 4⇒1, 5⇒0, 6⇒0, 7⇒0, ...)`
    Custom(Vec<usize>),
}

impl Schedule {
    /// Map `virtual_idx` (in `0..virtual_len`) to a real layer index in
    /// `0..real_len` according to this schedule.
    pub fn real_idx(&self, virtual_idx: usize, virtual_len: usize, real_len: usize) -> usize {
        match self {
            Schedule::Cyclic => virtual_idx % real_len,
            Schedule::Stretched => (virtual_idx * real_len) / virtual_len,
            Schedule::Custom(map) => *map.get(virtual_idx).unwrap(),
        }
    }

    /// The [`Applications`] of this schedule over `virtual_len` layers.
    pub fn applications(&self, virtual_len: usize, real_len: usize) -> Applications {
        Applications::new(
            (0..virtual_len).map(|i| self.real_idx(i, virtual_len, real_len)),
            real_len,
        )
    }
}

/// Which **application** of its real layer each virtual layer is: the
/// virtual→real map read the other way, which counts how often a weight set
/// was already used. An untied parameter is indexed by it (see
/// [`crate::utils::untied`]). [`GradHorizon::Depth`] counts the same thing
/// from the top.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Applications {
    /// Per virtual layer: how many virtual layers below it share its real layer.
    pub index: Vec<usize>,
    /// Per real layer: its number of applications. At least one, because a
    /// real layer that the schedule never reaches is still built.
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
/// activation. A horizon is a **mask**, not one boundary. The stack cuts down
/// wherever the mask turns off, and lifts back wherever it turns on, as many
/// times as the mask says.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GradHorizon {
    /// Back-propagate the last `K` applications of **every real layer**.
    ///
    /// `K` counts per *weight set*, not per stack. So every real layer keeps a
    /// tracked application, whatever the schedule does with it. This is the
    /// point: a plain suffix of `K` virtual layers reaches every weight only
    /// under [`Schedule::Cyclic`]. It would leave every [`Schedule::Stretched`]
    /// real layer except the topmost with no gradient.
    ///
    /// # Example
    /// - virtual len = 8, real len = 3, `Depth(1)` (`T` = tracked):
    ///   - [`Schedule::Cyclic`] (`0 1 2 0 1 2 0 1`): `. . . . . T T T`, a
    ///     single cut, at `virtual_len - K·real_len` on an even stack.
    ///   - [`Schedule::Stretched`] (`0 0 0 1 1 1 2 2`): `. . T . . T . T`, one
    ///     cut **per real layer**, at the tail of each run.
    ///
    /// A stack **without** weight sharing applies each real layer once, so any
    /// `K >= 1` tracks all of it. [`Self::last`] states the suffix mask for
    /// that case. [`Schedule::Custom`] has no canonical run to take a tail of,
    /// so it takes a [`Self::Mask`] instead (`Depth` panics on it).
    Depth(usize),
    /// Explicit per-virtual-layer mask, `true` = back-propagated. Its length
    /// must be the virtual-layer count of the stack.
    Mask(Vec<bool>),
}

impl GradHorizon {
    /// A [`Self::Mask`] that tracks the **last `k`** of `virtual_len` layers:
    /// the single-cut suffix horizon, written out. It is what [`Self::Depth`]
    /// gives under [`Schedule::Cyclic`] with `k = K·real_len`. It is also the
    /// natural horizon to cut a stack that shares no weights (`Depth` cannot
    /// cut it).
    pub fn last(k: usize, virtual_len: usize) -> Self {
        GradHorizon::Mask((0..virtual_len).map(|i| i + k >= virtual_len).collect())
    }

    /// Resolve to one `tracked` flag per virtual layer. `schedule` is the
    /// schedule of the stack, `None` when it runs no virtual layers (each real
    /// layer applied once, `virtual_len == real_len`).
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

/// How a bidirectional layer stack maps virtual layer indices to real layer
/// indices, interleaving the straight (→, even indices) and reverse (←, odd
/// indices) directions.
#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum BidiSchedule {
    /// Use even virtual positions for straight-direction (→), and odd virtual positions for
    /// reverse-direction (←), wrapping around for each schedule.
    ///
    /// # Example
    /// - virtual len = 10, real len = 4:  
    ///   `   →    ←      →    ←        →    ←      →    ←        →    ←          `  
    ///   `[(0⇒0, 1⇒1), (2⇒2, 3⇒3)], [(4⇒0, 5⇒1), (6⇒2, 7⇒3)], [(8⇒0, 9⇒1), (...)]`
    #[default]
    StridedCyclic,
    /// Use even virtual positions for straight-direction (→), and odd virtual positions for
    /// reverse-direction (←), stretching for each schedule.
    ///
    /// # Example
    /// - virtual len = 10, real len = 4:  
    ///   `   →    ←      →    ←      →    ←        →    ←      →    ←          `  
    ///   `[(0⇒0, 1⇒1), (2⇒0, 3⇒1), (4⇒0, 5⇒1)], [(6⇒2, 7⇒3), (8⇒2, 9⇒3), (...)]`
    StridedStretched,
    /// Fills virtual positions by wrapping around the real schedule in a looping fashion,
    /// replicating between the straight (→) and reverse (←) directions.
    ///
    /// # Example
    /// - virtual len = 10, real len = 4:  
    ///   `   →    ←      →    ←      →    ←      →    ←        →    ←          `  
    ///   `[(0⇒0, 1⇒0), (2⇒1, 3⇒1), (4⇒2, 5⇒2), (6⇒3, 7⇒3)], [(8⇒0, 9⇒0), (...)]`
    SymmetricCyclic,
    /// Fills virtual positions by stretching the real schedule, replicating between
    /// the straight (→) and reverse (←) directions.
    ///
    /// # Example
    /// - virtual len = 10, real len = 4:  
    ///   `   →    ←      →    ←       →    ←               →    ←        →    ←   `  
    ///   `[(0⇒0, 1⇒0), (2⇒0, 3⇒0)],[(4⇒1, 5⇒1), (...)], [(6⇒2, 7⇒2)], [(8⇒3, 9⇒3)]`
    SymmetricStretched,
    /// Fills virtual positions by referring to the index vector.
    ///
    /// # Example
    /// - virtual len = 10, real len = 4, custom = `[0, 1, 2, 2, 1, 0, 0, 0, 3, 2]`:  
    ///   `   →    ←        →    ←        →    ←        →    ←        →    ←            `  
    ///   `[(0⇒0, 1⇒1)], [(2⇒2, 3⇒2)], [(4⇒1, 5⇒0)], [(6⇒0, 7⇒0)], [(8⇒3, 9⇒2)], [(...)]`
    Custom(Vec<usize>),
}

impl BidiSchedule {
    /// Map `virtual_idx` (in `0..virtual_len`) to a real layer index in
    /// `0..real_len`. Even/odd `virtual_idx` selects the straight/reverse
    /// direction. The schedule cycles or stretches over the outer index
    /// `virtual_idx / 2`.
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

    /// The [`Applications`] of this schedule over `virtual_len` layers, both
    /// directions counted against the one real layer they index.
    pub fn applications(&self, virtual_len: usize, real_len: usize) -> Applications {
        Applications::new(
            (0..virtual_len).map(|i| self.real_idx(i, virtual_len, real_len)),
            real_len,
        )
    }
}

#[cfg(test)]
mod tests;
