# burn-stack

Block-agnostic containers for sequence models on Burn: layers, layer stacks,
networks and caches, with the training helpers around them.

## Language

### Sequences

**Sequence**:
The ordered tokens of one stream. The length of a sequence is a count of
tokens.

**Token**:
One element of a sequence.

**Position**:
One place on the time axis of a row. A position holds a token or padding.

### Blocks and layers

**Block**:
The part of a layer that mixes information across the tokens of a sequence.
_Avoid_: mixer, mixer core, SSM core, core block

**Family**:
One implementation of a block, with its config and its caches.
_Avoid_: variant, version, generation, mixer

**Layer**:
A Pre-LN residual around one block.

**Real layer**:
One set of layer weights.
_Avoid_: weight set

**Virtual layer**:
One pass of a layer stack. Each virtual layer runs one real layer.

**Application**:
One run of a real layer, counted for each real layer. A real layer has one
application for each virtual layer that runs it.

**Layer stack**:
The virtual layers of a network, in order, over a set of real layers.

**Network**:
A layer stack with an input map in front and an output map behind. A vocab
network maps token ids. A latent network maps vectors.

**Cache**:
Everything that a block keeps for one row from one call to the next.
_Avoid_: KV cache, memory

### Calls

**Forward**:
The parallel call that processes many tokens of each row.

**Step**:
The recurrent call that processes one token of each row.
_Avoid_: step (alone, for one recurrence step or for one training step)

**Prime**:
A step with no user token. It emits the class markers that wait for the next
token.

**Prefill**:
A forward that only fills the cache from a prompt.

**Decode**:
Generation by repeated steps.

### Class markers

**Class marker**:
A learnable embedding that the containers insert into a sequence.
_Avoid_: CLS, special token, register token

**Class token**:
A class marker that a network inserts. It is a point in the raw input space of
the network.

**Class latent**:
A class marker that a layer container inserts. It is a point in the `d_model`
space.

### Batches

**Row**:
One sequence of a batch (one index of the batch axis).
_Avoid_: slot

**Padding**:
The positions at the end of a row that hold no token.
_Avoid_: padded row, real row, pad row (for a position)

**Slot**:
One entry of a buffer that has a fixed size. Examples: a token position that a
packed row reserves for a class latent, or one entry of a FIFO in a cache.
_Avoid_: row (for one sequence of a batch), bucket

**Bucket**:
One batch length of a length-bucketed dataloader. All batches of a bucket have
the same shape.
_Avoid_: slot

### Packing

**Packed row**:
A row that holds several sequences, one after another.

**Segment**:
One sequence in a packed row.
_Avoid_: document, story, episode

**Reset**:
The mark on the first token of a segment.
_Avoid_: restart
