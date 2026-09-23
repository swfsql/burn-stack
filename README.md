# burn-stack

Block-generic layer/network composition on [Burn](https://github.com/tracel-ai/burn/).

It holds everything *around* a sequence-mixing block, and nothing that belongs
to a particular one:

```text
VocabNetwork<M>   embedding → Layers<M> → final RMSNorm → LM head → logits
LatentNetwork<M>  in_proj → Layers<M> → [norm_f] → out_proj   (continuous I/O)
BidiLayers<M>     paired straight (→) / reversed (←) stacks, merged per pair
Layers<M>         N (virtual) layers over R real weight sets
Layer<M>          Pre-LN sub-blocks: Block(RMSNorm(x)), then the optional
                  SwiGLU MLP over norm2. Layers owns the outer residual
M: Block          the mixer core, supplied by you
```

It also has the parts that those need:

- multi-stream gated residuals (Multi-Gate),
- `[CLS]`-style class tokens/latents with streamed placement,
- virtual-layer and LR schedules,
- a truncated-BPTT gradient horizon,
- fp16-stable norms/activations,
- losses,
- Muon parameter groups over fused projection weights.

## Plugging a block in

Implement two traits and one cache trait:

```rust
impl Block for MyBlock {
    type Cache = MyCache;              // the streaming state of one layer
    type Caches = MyCaches;            // one slot per virtual layer: CacheStack
    type Options = ();                 // per-call algorithm/chunk selector

    fn block_forward(&self, x: Tensor<3>, cache: Option<MyCache>, options: (),
        pad: Option<Tensor<2, Bool>>)  // `true` at right padding
        -> (Tensor<3>, MyCache) { … }  // chunked: training + prefill
    fn block_step(&self, x: Tensor<2>, cache: Option<MyCache>)
        -> (Tensor<2>, MyCache) { … }  // recurrent: decode
    fn zero_caches_3d(&self, x: &Tensor<3>, n_virtual: usize) -> MyCaches { … }
    fn zero_caches_2d(&self, x: &Tensor<2>, n_virtual: usize) -> MyCaches { … }
    fn untied_params(&self) -> Vec<UntiedParam> { … } // empty ⇒ all tied
}

impl BlockConfig for MyBlockConfig {
    type Block = MyBlock;
    fn d_model(&self) -> usize { … }
    fn init_block(&self, n_applications: usize, device: &Device) -> MyBlock { … }
    fn muon_projections(&self) -> Vec<ProjSpec> { … }   // feature `optim`
}
```

`src/reference.rs` is a complete worked example (a gated EMA mixer). The test
suite of this crate composes it, so the tests exercise the containers without
a dependency on a real family.

## The contract

- `forward()` from any cache equals `step()` unrolled from that same cache: on
  the outputs, on the final cache, and on the gradients. The containers rely
  on it, and your block must supply it.
- `pad` marks **right** padding per slot. A padded row is absent: the real
  outputs and the returned cache of each slot are those of that slot run
  alone.
- `prime()` is `step()` without a user token. It emits the class markers that
  wait for the next token.

## Backends

The crate uses the Dispatch backend of Burn, so the library types are not
backend-generic. The backend is a runtime `Device`, and autodiff and dtype are
device properties. Select one (or several) with the `backend-*` features.
`flex` is the default. There are no custom kernels, only portable Burn tensor
ops.

## Users

- [`burn-mamba`](https://github.com/swfsql/burn-mamba) — Mamba-1/2/3 selective state space models.
- [`burn-deltanet`](https://github.com/swfsql/burn-deltanet) — [Gated]-DeltaNet-1/2 models.
