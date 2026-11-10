# 01: Glossary: the terms of the vocab network

Status: needs-triage

Blocked by: None (can start immediately)

Spec: None (one ticket)

**What to build:** `GLOSSARY.md` gets an entry for each of three concepts of
the vocab network. The code and the docs of this repo use them, and so do the
docs of `../burn-mamba-example`. No glossary defines them.

1. **The flag channel.** Each token is a pair `(id, flag)`. The input adds
   `flag_embedding` to the embedding of a token whose flag is 1. The last
   column of the logits is the flag logit, from `flag_head`. The header of
   `VocabNetwork` in `src/modules/network.rs` describes it.
2. **The padded vocabulary.** The vocabulary size, rounded up to a multiple of
   `pad_vocab_size_multiple`. It is the row count of the embedding and the
   width of the vocab logits.
3. **The LM head that reuses the embedding** (`missing_lm_head: true`,
   `lm_head = None`). The code and the docs call it "tied". The glossary gives
   "tied" another meaning: a **Tied parameter** is a parameter that all
   applications of its real layer share. `docs/source-terms.md` already keeps
   "Tie Word Embedding" apart from **Tied parameter**. So this concept needs a
   name of its own, or an `_Avoid_` note on the word "tied".

A glossary review in a burn-mamba-example session found these gaps. That app
uses the three concepts:

- It drops the flag logit.
- It checks its padded vocabulary against each checkpoint.
- It builds its LM head from the embedding at load time.

Its glossary leaves these terms to burn-stack, because they are terms of the
containers.

## Questions for the triage

- The term for concept 3. Do `missing_lm_head` and the "tied/untied LM head"
  prose change to follow it?

## Acceptance criteria

- [ ] `GLOSSARY.md` defines the three concepts.
- [ ] Each identifier and each doc of this repo that names one of these
      concepts uses its glossary term, as the Docs rules of `CLAUDE.md` require.

## Comments
