# burn-stack

Block-generic layer/network composition on [Burn](https://github.com/tracel-ai/burn/).

Everything that sits *around* a sequence-mixing block — and nothing that belongs
to any particular one:

```text
VocabNetwork<M>   embedding → Layers<M> → final RMSNorm → LM head → logits
LatentNetwork<M>  in_proj → Layers<M> → [norm_f] → out_proj   (continuous I/O)
BidiLayers<M>     paired straight (→) / reversed (←) stacks, merged per pair
Layers<M>         N (virtual) layers over R real weight sets
Layer<M>          Pre-LN residual:  y = x·residual_scale + Block(RMSNorm(x))
M: Block          the mixer core, supplied by you
```

Plus the pieces those need: multi-stream gated residuals (Multi-Gate),
`[CLS]`-style class tokens/latents with streamed placement, virtual-layer and LR
schedules, a truncated-BPTT gradient horizon, fp16-stable norms/activations,
losses, and Muon parameter groups over fused projection weights.

## Plugging a block in

Implement two traits and one cache trait:

```rust
impl Block for MyBlock {
    type Cache = MyCache;              // one layer's streaming state
    type Caches = MyCaches;            // one slot per virtual layer: CacheStack
    type Options = ();                 // per-call algorithm/chunk selector

    fn block_forward(&self, x: Tensor<3>, cache: Option<MyCache>, options: ())
        -> (Tensor<3>, MyCache) { … }  // chunked: training + prefill
    fn block_step(&self, x: Tensor<2>, cache: Option<MyCache>)
        -> (Tensor<2>, MyCache) { … }  // recurrent: decode
    fn zero_caches_3d(&self, x: &Tensor<3>, n: usize) -> MyCaches { … }
    fn zero_caches_2d(&self, x: &Tensor<2>, n: usize) -> MyCaches { … }
}

impl BlockConfig for MyBlockConfig {
    type Block = MyBlock;
    fn d_model(&self) -> usize { … }
    fn init_block(&self, device: &Device) -> MyBlock { … }
    fn muon_projections(&self) -> Vec<ProjSpec> { … }   // feature `optim`
}
```

`src/reference.rs` is a complete worked example (a gated EMA mixer) and is what
this crate's own test suite composes — so the containers are exercised without
depending on any real family.

## The contract

`forward()` from any cache equals `step()` unrolled from that same cache — on
outputs, on the final cache, and on gradients. The containers rely on it; your
block owes it. `prime()` is `step()` without a user token, emitting the class
markers waiting for the next one.

## Backends

Built on Burn's Dispatch backend, so the library types are not backend-generic:
the backend is a runtime `Device`, and autodiff and dtype are device
properties. Select one (or several) with the `backend-*` features; `flex` is the
default. There are no custom kernels — only portable Burn tensor ops.

## Users

- [`burn-mamba`](../burn-mamba) — Mamba-1/2/3 selective state space models.
