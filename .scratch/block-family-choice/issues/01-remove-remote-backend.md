# 01: Remove the remote backend

Status: resolved

Blocked by: None (can start immediately)

Spec: `../burn-mamba/.scratch/block-family-choice/spec.md`. Run only one session
on the two repos.

**What to build:** The project does not support the remote backend. Remove the
`backend-remote` feature from `Cargo.toml`, and remove the `Remote` impl from
`impl_backend_ext_for_burn_backends!` (`src/utils/backend_macros.rs:27-28`).
Remove each other mention that you find with `rg -i "remote"`, in the code and
in the docs. Keep the mentions that mean a git remote.

**Done when:**

- `cargo check`, `cargo check --all-targets` and `cargo doc --no-deps` have no
  warnings.
- `cargo test --lib -- --test-threads=1` passes, with the same 90 tests.
