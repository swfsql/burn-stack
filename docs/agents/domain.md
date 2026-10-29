# Domain Docs

This file tells the engineering skills how to use the domain documentation of
this repo when they explore the code.

The layout is single-context: one `GLOSSARY.md` and one `docs/adr/`, both at the
repo root.

## Before exploring, read these

- **`GLOSSARY.md`** at the repo root.
- **`docs/adr/`**: read the ADRs about the area that you will work in.

If one of these files does not exist, **continue and say nothing**. Do not
report that it is missing. Do not suggest that you create it in advance. The
`/domain-modeling` skill (through `/grill-with-docs` and
`/improve-codebase-architecture`) creates these files only when a term or a
decision is resolved.

## File structure

```
/
├── GLOSSARY.md
├── docs/adr/
│   ├── 0001-<decision-slug>.md
│   └── 0002-<decision-slug>.md
└── src/
```

## Use the glossary's vocabulary

When your output names a domain concept, use the term that `GLOSSARY.md`
defines. This applies to issue titles, refactor proposals, hypotheses and test
names. Do not use a synonym that the glossary tells you to avoid.

If the glossary does not have the concept, that is a signal:

- Possibly, you use a term that the project does not use. Think again.
- Or the glossary has a real gap. Write it down for `/domain-modeling`.

## Flag ADR conflicts

If your output contradicts an ADR, say so clearly. Do not override the ADR
silently:

> _Contradicts ADR-<NNNN> (<title>), but worth reopening because…_
