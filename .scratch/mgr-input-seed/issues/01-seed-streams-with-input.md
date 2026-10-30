# 01: Seed every residual stream with the input

Status: needs-triage

Blocked by: `.scratch/glossary-followups/` 01 to 05 (the glossary changes)

**What to build:** A setting of Multi-Gate Residuals that starts every
residual stream as a copy of the stack input. This is an idea of the user, and
nobody has thought it through yet.

## Facts from the burn-stack grill

- Today, the residual streams do not start at zero. The stack input is stream
  1. Each of the first `n_stream − 1` layers appends its output as a new
  stream (the accumulation phase). The gated mixing starts after them.
- The MGR module header rejects a copy of the input in every stream. Identical
  streams get identical scores, gates and gradients, so they never separate,
  and MGR collapses to one stream.
- That is true only when every stream has the same gate bias. `init_bias_step`
  gives each stream its own initial bias. Then the gates differ at the first
  mix, and the streams separate.
- ADR-0006 gives the copy of the input across depth to the residual scheme,
  not to the gradient horizon. This setting is that job.

## Questions for triage

- Does the setting replace the accumulation phase, or come before it?
- Does it need a nonzero `init_bias_step`, and is that a panic or a default?
- Does it train as well as the accumulation phase? Which benchmark shows it?
- What does the straight-through carry do at a cut, when every stream started
  as the input?

## Comments
