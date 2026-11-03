# 01: One backend list, and a compile error for a one-sided backend

Status: needs-triage

Blocked by: None

From the architecture review of 2026-11-03. Line numbers are those of burn-mamba
commit `42ca3d0e` and burn-stack commit `12429e1`. The macro approach is not tried.

**Files:** `src/utils/backend_macros.rs:16-38`, `../burn-mamba/build.rs:14-20`,
`../burn-mamba/Cargo.toml:26-41`,
`../burn-mamba/src/mamba2/ssd/serial_recalculated/serial_recalculated.rs:83-90,391`.

**Problem:** One fact has 8 lists: 2 Cargo tables, 1 macro, `build.rs` and 4
`#[backend_extension]` attributes. The attributes say "Change them together". The
build shows no error when they differ. By reading, if `burn/x` or
`burn-stack/backend-x` is on and `burn-mamba/backend-x` is off, the op reaches
`unimplemented!` at run time (ADR-0002).

**Direction:** One burn-stack macro holds the list and wraps the extension trait in
the consumer. It also compares the features of both crates at compile time. Then
`build.rs` can go. `burn/x` alone stays undetected, because the cfgs of Burn are
private. Forwarding stays (ADR-0002).

**Not checked:** Whether `macro_rules!` can emit the attribute around a consumer
trait, and whether `unexpected_cfgs` already fires in this expansion.

Split this ticket by repo at triage. The deletion of the unused traits is in
`../burn-mamba/.scratch/arch-review-backlog/issues/05-unused-backend-traits.md`.
