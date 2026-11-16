# 17: One method for the `enter` and `leave` of the class cursors

Status: needs-triage

Blocked by: None

From the architecture review of 2026-11-16. Line numbers are those of commit
`b198fa5`.

**Files:** `src/modules/network.rs:108-112,213-215,236-238`, `src/utils/class/`
(`ClassCursors::enter`, `ClassCursors::leave`).

**Problem:** A network calls `class.enter(&self.class_tokens)` before it runs its
stack, and `class.leave(saved)` after it. The pair is at three sites of `network.rs`.
Each site keeps `saved` by hand. A site with no `leave` compiles, and the cursors of
the next call are then wrong.

**Direction:** One method of `ClassCursors` takes the class tokens and a closure. It
calls `enter`, runs the closure and calls `leave`. `enter` and `leave` become private
to the class module.

**Callers:** A grep of the non-test code of burn-stack and of burn-mamba (`src/` and
`examples/`) finds no other call of `enter` or `leave`. The tests were not read.
