# Portable Burn tensor ops only, no custom kernels

burn-stack builds every container from the standard tensor ops of Burn. It
has no custom kernel. A container must run on each backend that its families
run on. So one code path serves CPU, WGPU, CUDA, Metal, LibTorch and the
other backends of Burn. We accept that this is slower than fused kernels.
Each family crate makes the same decision for its blocks (burn-mamba
ADR-0001).

## Consequences

- A custom *backward* is permitted. It recomputes the forward with the same
  tensor ops, so it is not a kernel. `utils/combined_grad.rs` and
  `utils/fprim.rs` hold the plumbing for it.
- A slow op gets a new formulation in tensor ops, not a kernel. For example,
  `prefix_sum` gives the values of `cumsum`, but its cost on cubecl does not
  grow quadratically with the length.
