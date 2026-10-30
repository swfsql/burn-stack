# A model config is a block-free shape plus a block config

The builders of the containers (`LayersBuilder` and the others) hold the
block config `C`. The `#[derive(Config)]` of Burn cannot serialise them
unless serde also handles that generic. So `modules/shape.rs` splits the
block off. `NetworkShape` holds everything about a stack except its block, and
`LatentShape`, `VocabShape`, `BidiShape` and `HNetShape` build on the same
split. The serialisable config of a model is the pair `{ shape, block }`. A
family then states nothing about the architecture around its block, and each
setting of a stack has one declaration.

## Consequences

The saved configs of every consumer have this form. A change to a shape
struct changes all of them.
