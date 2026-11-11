# burn-stack

Block-generic layer/network composition on [Burn](https://github.com/tracel-ai/burn/).

It holds everything *around* a sequence-mixing block, and nothing that belongs
to a particular family:

```text
VocabNetwork<M>   embedding → Layers<M> → final RMSNorm → LM head → logits
LatentNetwork<M>  in_proj → Layers<M> → [norm_f] → out_proj   (continuous I/O)
BidiLayers<M>     paired straight (→) / reversed (←) stacks, merged per pair
Layers<M>         N virtual layers over R real layers
Layer<M>          Pre-LN sub-blocks: Block(RMSNorm(x)), then the optional
                  SwiGLU MLP over norm2. Layers owns the outer residual
M: Block          the block of a family, supplied by you
```

It also has the parts that those need:

- multi-stream gated residuals (Multi-Gate),
- class markers (class tokens and class latents) with streamed placement,
- layer schedules and LR schedules,
- a gradient horizon,
- fp16-stable norms/activations,
- losses,
- the Muon plan over fused projections.

## Plugging a block in

Implement two block traits and two cache traits. `CacheTensors` is the one
list of the fields of a cache. The backend hop of a cache follows from it.

```rust
impl Block for MyBlock {
    type Cache = MyCache;              // the cache of one layer
    type Caches = MyCaches;            // one slot per virtual layer: CacheStack
    type Options = ();                 // per-call algorithm/chunk selector

    fn block_forward(&self, x: Tensor<3>, cache: Option<MyCache>, options: (),
        pad: Option<Tensor<2, Bool>>)  // `true` at right padding
        -> (Tensor<3>, MyCache) { … }  // parallel: training + prefill
    fn block_step(&self, x: Tensor<2>, cache: Option<MyCache>)
        -> (Tensor<2>, MyCache) { … }  // recurrent: decode
    fn zero_cache(&self, batch: usize, device: &Device) -> MyCache { … }
    fn untied_params(&self) -> Vec<UntiedParam> { … } // empty ⇒ all tied
}

impl BlockConfig for MyBlockConfig {
    type Block = MyBlock;
    fn d_model(&self) -> usize { … }
    fn init_block(&self, n_applications: usize, device: &Device) -> MyBlock { … }
    fn muon_projections(&self) -> Vec<ProjSpec> { … }   // feature `optim`
}

impl CacheTensors for MyCache {        // every tensor, paired with `other`
    fn zip_tensors(self, other: Self, z: &mut impl TensorZip) -> Self { … }
}

impl CacheStack for MyCaches {
    type Cache = MyCache;
    fn slot_count(&self) -> usize { … }
    fn into_slots(self) -> Vec<Option<MyCache>> { … }
    fn from_slots(slots: Vec<Option<MyCache>>) -> Self { … }
}
```

`src/reference.rs` is a complete worked example (a gated EMA block). The test
suite of this crate composes it, so the tests exercise the containers without
a dependency on a real family.

## The contract

- `forward()` from any cache equals `step()` unrolled from that same cache: on
  the outputs, on the final cache, and on the gradients. The containers rely
  on it, and your block must supply it.
- `pad` marks **right** padding per row. A padded position is absent: the
  outputs of the real tokens and the returned cache of each row are those of
  that row run alone.
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

## Sources

- [H-Net](https://arxiv.org/abs/2507.07955): *Dynamic Chunking for End-to-End
  Hierarchical Sequence Modeling*. The official code is
  [`goombalab/hnet`](https://github.com/goombalab/hnet).
- [Multi-Gate Residuals](https://arxiv.org/abs/2605.23259).

## Box setup

The docs name directories next to this repo, for example `../papers/hnet/` and
`../burn/`. To create them on a new box, run:

```bash
scripts/setup-box.sh
```

The script gets each directory at the exact version that
[`docs/box.tsv`](docs/box.tsv) lists, because the docs cite `file:line`. It also
reads the list of each repo next to this repo. It does not change a directory
that exists. When a source is not available, the script gives a warning and
continues.
