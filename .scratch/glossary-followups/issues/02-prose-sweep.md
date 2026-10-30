# 02: Prose sweep: the core containers, README and CLAUDE.md

Status: ready-for-agent

Blocked by: 01

**What to build:** The prose of the core containers uses the terms of
`GLOSSARY.md`, and no term that it lists under _Avoid_. A reader finds one name
for each concept. Tickets 03, 04 and 05 do the same for other parts of the
crate, with the term map and the method of this ticket.

## Scope

- Doc comments and comments in the modules directory, except H-Net (ticket
  03). These are:
  - the layer, the layer stack, the networks and the shapes,
  - the bidirectional stacks, MGR and the MLP,
  - the caches, the model config and the block traits,
  - the norm, activation, loss and misc modules.
- The reference block and the crate root.
- README.md, and the CLAUDE.md that ticket 01 wrote.
- Panic, assert and error messages. Before you change one, find the tests that
  match its text (`should_panic(expected = …)`), and keep the matched part or
  update the test.
- Identifiers stay. List each identifier that conflicts with a glossary term
  under `## Comments` for a later decision.

## Term map

`GLOSSARY.md` is the term map: for each _Avoid_ term, use the term of its
entry. The table lists the cases that need a judgment.

From the burn-mamba grill:

| Avoid | Use |
|---|---|
| "slot" for one sequence of a batch ("every slot of a batch", "per batch slot") | **row** |
| "slot" for a batch length of the length-bucketed loader | **bucket** |
| "padded row", "real row", "pad row" (for a time position) | **padded position**, **real token**, **position** |
| "restart" as a synonym of reset | **reset** |
| "document", "story", "episode" for a sequence in a packed row | **segment** (in the tiny-stories dataset, "story" stays for the data) |
| "special token", "CLS", "register token", and a loose mix of the three class words | **class marker** (the general term), **class token** (a network inserts it, raw input space), **class latent** (a layer container inserts it, `d_model` space) |
| "weight set" | **real layer** |
| "mixer", "mixer core", "SSM core", "core block" | **block**. "A particular mixer" becomes "a particular family". |
| "variant", "version", "generation" for an implementation of a block | **family** |
| bare "step" for one recurrence step or one training step | "step" is only the step call. Write "training step" for the optimizer. |
| "KV cache", "memory" for what a block keeps between calls | **cache** |

From the burn-stack grill:

| Avoid | Use |
|---|---|
| "segment" for a part of a fused projection, or for a run of untracked layers | **sub-projection**, **untracked span** |
| "fused weight", "column block" | **fused projection**, **sub-projection** |
| "chunk" for the part of a sequence that one forward takes ("`forward` chunks", the prefill chunks) | **window**. "Chunk" stays for a chunk of a chunked scan and for an H-Net chunk. |
| bare "carry" for the gradient horizon | **straight-through carry**. "Carry-biased" stays for the MGR gates (the Highway Networks term). "Carry" stays as a plain verb ("carry the cache into the next window"). |
| "row" for a place on the time axis, "ghost row" | **position**, **ghost latent**. "Row" stays for one sequence of a batch, and for a row of a matrix. |
| "main network", "inner network", "encoder network", "decoder network", "level" in H-Net | **main stack**, **inner H-Net**, **encoder**, **decoder**, **stage**. "Level" stays for a marker level. |
| bare "schedule" | **layer schedule**, or "LR schedule". In prose, `Lr::Sequence` is "a chain of LR schedules", and one `LrSegment` is "one schedule of the chain". |
| bare "boundary" | **chunk boundary** (H-Net), **cut** (gradient horizon), **I/O weight** (Muon), "at its input" (class markers), **reset** (packed rows) |
| "truncated BPTT" for the gradient horizon | **gradient horizon**. "BPTT" stays for the training window of the examples. |
| "lane", "depth-stream" | **residual stream** |
| "forward" or "backward" for a direction of a bidirectional pair | **straight pass**, **reversed pass** |
| bare "graph" for a capture | **captured step**, **replay**. Bare "graph" stays for the autodiff graph. |
| "unshared parameter", "per-layer parameter" | **untied parameter** |

## Method

Use the Edit tool, site by site. Each site needs a judgment, so no scripted
replacement. Write in the style of the `asd-ste100` skill.

## Acceptance criteria

- [ ] For each _Avoid_ term, `rg` over the scope finds only the uses that the
      tables permit. Each other hit that you keep is listed with its reason
      under `## Comments`.
- [ ] `cargo check` and `cargo test --lib -- --test-threads=1` pass (flex).
- [ ] `cargo doc --no-deps` has no warnings.

## Comments
