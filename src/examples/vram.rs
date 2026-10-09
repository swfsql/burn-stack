//! A cap on the device memory of a training run (an example flag such as
//! `--max-vram <MiB>`).
//!
//! After each backward (the peak of a step), [`check`] reads the bytes that the
//! memory pools reserve. Above the cap, the process stops with an error.
//!
//! A pool only queues a page that it frees, and the device keeps a queued page
//! until an explicit cleanup. The report does not count the queue, so the
//! process can hold much more than the pools reserve (~0.5 GiB on a model near
//! 1M params). So with the cap, [`cleanup`] also runs before windows 0, 1, 2,
//! 4, 8, … and before each 256th window. Then the reserved bytes and the CUDA
//! context hold the process.

use burn::prelude::*;

/// Before training window `window` (counted from 0 in this process): clean up
/// the memory of `device` at windows 0, 1, 2, 4, 8, … and at each 256th
/// window.
pub fn cleanup(device: &Device, window: u64) {
    if window.is_power_of_two() || window.is_multiple_of(256) {
        device.memory_cleanup();
    }
}

/// After a backward: stop the process (exit code 3) when the memory pools of
/// `device` reserve more than `max_mib` MiB. A device with no pool report
/// passes.
pub fn check(device: &Device, max_mib: u64) {
    if let Some(usage) = device.memory_pool_usage() {
        let reserved_mib = usage.bytes_reserved >> 20;
        if reserved_mib > max_mib {
            eprintln!("VRAM LIMIT: the memory pools reserve {reserved_mib} MiB > --max-vram {max_mib} MiB; stopping");
            std::process::exit(3);
        }
    }
}
