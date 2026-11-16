# 12: The cache stack is one type of burn-stack

Status: needs-triage

Blocked by: None

From the architecture review of 2026-11-16. Line numbers are those of burn-stack
commit `b198fa5` and burn-mamba commit `7807d31b`. The review compiled nothing.

**Files:** `src/modules/cache.rs:44-66` (`CacheStack`), `src/modules/mod.rs:104-110`
(`Block::Caches`), `src/modules/layers.rs:424-450,597-629,799-808`,
`src/modules/bidi.rs:316-385`, `src/modules/hnet/cache.rs`,
`src/block_contract.rs:140,545`, `src/reference.rs:47-69`,
`../burn-mamba/src/family/adapters.rs`, `../burn-mamba/src/family/cache_traversal.rs`,
`../burn-mamba/src/family/block.rs` (`MambaCaches`), the `cache.rs` of each family.

**Problem:** Each family writes a struct around a `Vec`, a `CacheStack` impl and a
cache traversal that only forwards to the `Vec`. There are 7 such types: `RefCaches`,
`Mamba1Caches`, `Mamba2Caches`, `Mamba3DoubleSsdCaches`, `Mamba3SingleSsdCaches`,
`Mamba3Caches` and `MambaCaches`. The 5 `CacheStack` impls do not agree:

- 4 of them unwrap an empty slot in `from_slots`. `MambaCaches` keeps it.
- `Mamba3Caches::from_slots` guesses the pathway from slot 0.

The contract `cache_stack_moves_every_slot` tests only these impls. Deletion test:
with the 7 types deleted, no complexity comes back, because `modules/cache.rs` already
gives `Vec<C>` a cache traversal.

**Direction:** The containers hold `Vec<M::Cache>`. `Block::Caches` and `CacheStack`
go. The detach of a stack is one function over the cache traversal.

**Second step, same seam:** A slot is never empty. `block_forward` and `block_step`
take `Self::Cache`, and the container calls `zero_cache`. Then the rule "the zero
cache equals no cache" leaves the interface of `Block`, with its contract
`zero_cache_equals_no_cache`. `Layers::prime` with no caches is the one path that
starts from empty slots now (`layers.rs:622-627`). Decide this step at triage.

**Cost:** A break of the public API of both crates. The LM loops and the examples name
the cache stack types (`MambaCaches`, `HNetCaches`).

**Related:** `../burn-mamba/.scratch/arch-review-backlog/issues/18-pathway-option.md`
deletes the enum `Mamba3Caches` first, which makes this ticket smaller.

Split this ticket by repo at triage.
