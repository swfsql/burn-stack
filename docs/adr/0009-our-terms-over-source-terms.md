# Our terms over the terms of the sources

The glossary is the authority for the words of this crate. We do not rename a
term to match a paper or a reference implementation. The sources disagree with
each other. "Network" in the H-Net paper is our layer stack. "Mixer" holds the β
gates of one layer in Multi-Gate Residuals, and it is our block in the official
Mamba code. So no source gives one vocabulary to follow. A reader of the papers
must learn our words, and `docs/source-terms.md` maps the source words to them.

## Consequences

- A rename needs one of two reasons. The term is wrong by our own definitions,
  or its risk of a wrong port is worth the cost.
- `_Avoid_` applies to our own words. A doc can use a source word in quotes or
  backticks when it cites that source.
