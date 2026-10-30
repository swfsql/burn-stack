# Muon sees split projections, the model keeps them fused

A family fuses several independent linear maps into one `Linear`, for example
its `in_proj`. Muon orthogonalises a whole 2-D matrix. On a fused weight, that
couples maps that share only their allocation. So the model keeps its fused
weights and its single GEMM, and the optimizer splits them. A family gives
the column seams of each fused weight in a `ProjSpec`, with the widths that
the `split_into` of its forward uses. `Segmented` then steps each
sub-projection with its own optimizer.

The plan is an allowlist. Muon steps only the weights that a `ProjSpec`
names. Every other parameter stays on the fallback optimizer (AdamW or plain
SGD). These are:

- the biases, the norm gains, and each tensor of rank 1 or 3,
- the per-head scalar channels of a fused weight,
- the I/O weights: the embedding, the LM head, the in/out projections of a
  network and the class marker tables.

## Considered Options

- **Split the projections in the model**: one GEMM for each map, in the
  forward and in the backward.
- **Muon on the whole fused weight**: it couples independent maps.
- **A denylist**: the Muon of Burn asserts rank 2. A new parameter that the
  list does not name makes a run panic.
- **A Muon that steps a 3-D tensor as a stack of matrices**: no 3-D tensor
  of a block is a stack of linear maps. Each one is a per-head gain, a bias,
  an initial condition or a depthwise filter.

## Consequences

A `ProjSpec` matches its container and its weight as two separate path
substrings. So one plan covers plain, virtual-layer and bidirectional stacks,
hand-written models, and a block that is an `enum`.
