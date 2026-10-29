# 02: Prose sweep for the glossary terms

Status: ready-for-agent

Blocked by: 01

**What to build:** All prose in the crate uses the terms of `GLOSSARY.md`, and
no term that it lists under _Avoid_. A reader finds one name for each concept.

## Scope

- Doc comments and comments in `src` (also `src/examples/`).
- Panic, assert and error messages. Before you change one, find the tests that
  match its text (`should_panic(expected = …)`), and keep the matched part or
  update the test.
- README.md and CLAUDE.md.
- Identifiers stay. List each identifier that conflicts with a glossary term
  under `## Comments` for a later decision.

## Term map (decided in the burn-mamba grill)

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

Add the terms that the burn-stack grill adds to the glossary.

## Method

Use the Edit tool, site by site. Each site needs a judgment, so no scripted
replacement. Write in the style of the `asd-ste100` skill.

## Acceptance criteria

- [ ] For each _Avoid_ term, `rg` over the scope finds only the uses that the
      table permits. Each other hit that you keep is listed with its reason
      under `## Comments`.
- [ ] `cargo check` and `cargo test --lib -- --test-threads=1` pass (flex).
- [ ] `cargo doc --no-deps` has no warnings.

## Comments
