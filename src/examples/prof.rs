//! Opt-in wall-clock timers over the phases of a training step. They show where
//! the time of a step goes, and whether a phase grows over a run. An example
//! flag such as `--profile <N>` calls [`init`]. Then the module prints the mean
//! milliseconds of each phase once per `N` windows. With `sync`, it also syncs
//! the device after each phase, so a phase then includes its GPU execution,
//! not only its enqueueing. Every function does nothing until [`init`].
//!
//! Phases:
//!
//! - `fwd`: forward + loss (with [`extra`], also the forward of the extra pass).
//! - `bwd`: backward (with [`extra`], also the backward of the extra pass).
//! - `gap`: window end → the optimizer. These are the metric reads of the loop,
//!   that is its implicit sync.
//! - `opt`: the optimizer step.
//! - `out`: optimizer → next window (logging, next batch).
//!
//! The calls of one step, in order: [`window_start`], [`mark`] after the
//! forward, [`mark`] after the backward, [`window_end`], then [`opt_start`]
//! and [`opt_end`] around the optimizer step.

use burn::prelude::*;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

struct State {
    every: usize,
    sync: bool,
    n: usize,
    sums: [f64; 5],
    window_end: Option<Instant>,
    opt_end: Option<Instant>,
    device: Option<Device>,
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();

/// Start timing: print the means of the phases once per `every` windows,
/// and sync the device after each phase if `sync` is true.
pub fn init(every: usize, sync: bool) {
    let state = Mutex::new(State {
        every,
        sync,
        n: 0,
        sums: [0.0; 5],
        window_end: None,
        opt_end: None,
        device: None,
    });
    assert!(STATE.set(state).is_ok(), "the profiler starts once");
}

fn ms(from: Instant, to: Instant) -> f64 {
    (to - from).as_secs_f64() * 1e3
}

/// The start of a window (`t0`). It also ends the `out` phase.
pub fn window_start() -> Instant {
    let now = Instant::now();
    if let Some(state) = STATE.get() {
        let mut s = state.lock().unwrap();
        if let Some(end) = s.opt_end.take() {
            s.sums[4] += ms(end, now);
        }
    }
    now
}

/// The end of a phase. It syncs the device first when the profiler syncs.
pub fn mark(device: &Device) -> Instant {
    if let Some(state) = STATE.get()
        && state.lock().unwrap().sync
    {
        device.sync().expect("device sync");
    }
    Instant::now()
}

/// The end of a window: `fwd` is `t0..t1`, and `bwd` is `t1..t2`.
pub fn window_end(device: &Device, t0: Instant, t1: Instant, t2: Instant) {
    if let Some(state) = STATE.get() {
        let mut s = state.lock().unwrap();
        s.sums[0] += ms(t0, t1);
        s.sums[1] += ms(t1, t2);
        s.window_end = Some(Instant::now());
        s.device = Some(device.clone());
    }
}

/// An extra pass of the window (an auxiliary loss after the main backward):
/// add `t2..t3` to `fwd` and `t3..t4` to `bwd`.
pub fn extra(t2: Instant, t3: Instant, t4: Instant) {
    if let Some(state) = STATE.get() {
        let mut s = state.lock().unwrap();
        s.sums[0] += ms(t2, t3);
        s.sums[1] += ms(t3, t4);
    }
}

/// The start of the optimizer step (`t3`). It also ends the `gap` phase.
pub fn opt_start() -> Instant {
    let now = Instant::now();
    if let Some(state) = STATE.get() {
        let mut s = state.lock().unwrap();
        if let Some(end) = s.window_end.take() {
            s.sums[2] += ms(end, now);
        }
    }
    now
}

/// The end of the optimizer step. Once per `every` steps, print the `prof`
/// line.
pub fn opt_end(t3: Instant) {
    let Some(state) = STATE.get() else {
        return;
    };
    let mut s = state.lock().unwrap();
    if s.sync
        && let Some(device) = &s.device
    {
        device.sync().expect("device sync");
    }
    let now = Instant::now();
    s.sums[3] += ms(t3, now);
    s.opt_end = Some(now);
    s.n += 1;
    if s.n % s.every == 0 {
        let [f, b, g, o, u] = s.sums.map(|sum| sum / s.every as f64);
        // Live allocations: flat unless some launch shape varies. Each new
        // shape pins a cached metadata buffer (burn#5751), but the bytes
        // barely move. `reserved` is every page that the pools hold (the
        // high-water mark of the step, the CUDA context excluded).
        let allocs = match s.device.as_ref().and_then(Device::memory_pool_usage) {
            Some(usage) => format!(
                " allocs {} ({:.1} MB) reserved {:.1} MB",
                usage.number_allocs,
                usage.bytes_in_use as f64 / 1e6,
                usage.bytes_reserved as f64 / 1e6
            ),
            None => String::new(),
        };
        println!(
            "prof n={} fwd {f:.1} bwd {b:.1} gap {g:.1} opt {o:.1} out {u:.1} total {:.1} ms{allocs}",
            s.n,
            f + b + g + o + u
        );
        s.sums = [0.0; 5];
    }
}
