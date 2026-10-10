//! A cap on the device memory of a training run (an example flag such as
//! `--max-vram <MiB>`).
//!
//! After each backward (the peak of a step), [`check`] reads the bytes that the
//! memory pools reserve. Above the cap, the process stops with an error.
//!
//! # What the report misses
//!
//! The reserved bytes are the pages that the pools hold. The device holds
//! more:
//!
//! - **Dropped pages.** A pool drops a page on its own: a page that stays idle
//!   past its period, or a page that a growth of the pages left behind. A
//!   dropped page leaves the report at once, but the storage only queues it.
//!   The driver frees it at the next explicit cleanup
//!   (`Device::memory_cleanup`), and nothing else frees it. So between two
//!   cleanups, the queue holds memory that no report shows (~0.5 GiB on a
//!   model near 1M params).
//! - **The CUDA context and the loaded kernels** (one module per kernel
//!   shape). No pool holds them (~140 MiB).
//!
//! So with the cap, [`cleanup`] also runs on a schedule. The default runs
//! before windows 0, 1, 2, 4, 8, … and before each 256th window. With a period
//! of 1 (`--vram-cleanup-every 1`), the queue never outlives a window. Then
//! the reserved bytes after a backward are the pool memory of the device at
//! the peak of the step, apart from the pages dropped within that step. The
//! cost: each cleanup also gives back the empty pages, so the next window
//! allocates them again.

use burn::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// The largest reserved bytes that [`check`] read since the last
/// [`take_peak`].
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The largest reserved bytes that [`check`] read since the last call (the
/// high-water mark of the pools over those windows), and start again.
/// `None` when [`check`] read nothing.
pub fn take_peak() -> Option<u64> {
    Some(PEAK.swap(0, Ordering::Relaxed)).filter(|&peak| peak > 0)
}

/// Before training window `window` (counted from 0 in this process): clean up
/// the memory of `device`. With `every`, before each `every`-th window. Else
/// before windows 0, 1, 2, 4, 8, … and before each 256th window.
pub fn cleanup(device: &Device, window: u64, every: Option<u64>) {
    let due = match every {
        Some(every) => window.is_multiple_of(every),
        None => window.is_power_of_two() || window.is_multiple_of(256),
    };
    if due {
        device.memory_cleanup();
    }
}

/// After a backward: stop the process (exit code 3) when the memory pools of
/// `device` reserve more than `max_mib` MiB. A device with no pool report
/// passes. It also records the reserved bytes for [`take_peak`].
pub fn check(device: &Device, max_mib: u64) {
    if let Some(usage) = device.memory_pool_usage() {
        PEAK.fetch_max(usage.bytes_reserved, Ordering::Relaxed);
        let reserved_mib = usage.bytes_reserved >> 20;
        if reserved_mib > max_mib {
            eprintln!("VRAM LIMIT: the memory pools reserve {reserved_mib} MiB > --max-vram {max_mib} MiB; stopping");
            std::process::exit(3);
        }
    }
}
