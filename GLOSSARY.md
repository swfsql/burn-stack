# burn-stack

Block-agnostic containers for sequence models on Burn: layers, layer stacks,
networks and caches, with the training helpers around them.

## Language

### Sequences

**Sequence**:
An ordered list of tokens that a model takes as one input. The length of a
sequence is a count of tokens.

**Token**:
One element of a sequence.

**Position**:
One place on the time axis of a row. A position holds a token or padding.
_Avoid_: row (for a place on the time axis)

**Chunk**:
A contiguous part of a sequence that a block or an H-Net takes as one unit. A
chunked scan uses chunks of one fixed length. The router of an H-Net sets the
length of each chunk.

**Chunk boundary**:
The position where a chunk starts.
_Avoid_: boundary (alone)

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

**Layer schedule**:
The map from each virtual layer of a layer stack to the real layer that it
runs.
_Avoid_: schedule (alone)

**Tied parameter**:
A parameter that all applications of its real layer share.

**Untied parameter**:
A parameter that a real layer holds once for each of its applications.
_Avoid_: unshared parameter, per-layer parameter

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

**Window**:
The part of a sequence that one forward takes. The cache carries the state
from one window to the next.
_Avoid_: chunk (for a window)

**Captured step**:
A step that the backend records once and then replays as one dispatch.
_Avoid_: graph (alone), CUDA graph

**Replay**:
One run of a captured step.

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

**Marker level**:
A container that inserts class markers: a network, a layer stack or a layer.
Each marker level has its own cursor.

**Cursor**:
The count of the output positions of a marker level that the earlier calls
already emitted.

**Length hint**:
The length of the whole sequence that a call is part of. The `Middle` and
`End` markers need it.

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

### Residuals

**Residual scheme**:
How a layer stack combines the output of each layer with its input: the
standard additive skip, or Multi-Gate Residuals.

**Residual stream**:
A vector at each position that a layer stack passes from one layer to the
next. A standard stack has one. Multi-Gate Residuals keep several.
_Avoid_: lane, depth-stream

**Multi-Gate Residuals (MGR)**:
The residual scheme that keeps several residual streams between layers and
mixes them with gates.

### Bidirectional stacks

**Bidirectional pair**:
A straight pass and a reversed pass over the same sequence, with their
outputs merged.

**Straight pass**:
The pass of a bidirectional pair that reads the sequence in order.
_Avoid_: forward (for a direction)

**Reversed pass**:
The pass of a bidirectional pair that reads the sequence in reverse order.
_Avoid_: backward (for a direction)

### Gradient horizon

**Gradient horizon**:
The virtual layers of a layer stack that back-propagate.
_Avoid_: truncated BPTT

**Tracked layer**:
A virtual layer inside the gradient horizon.

**Untracked layer**:
A virtual layer outside the gradient horizon. It runs with no gradient.

**Untracked span**:
A run of consecutive untracked layers.
_Avoid_: segment, untracked segment

**Cut**:
The place where an untracked span ends and the graph starts again.
_Avoid_: boundary

**Straight-through carry**:
What enters an untracked span, added back at its cut with value zero and an
identity gradient.
_Avoid_: carry (alone)

**Ghost latent**:
A value-zero copy of a class latent of an untracked span, in the
straight-through carry.
_Avoid_: ghost row

### Optimizer

**Fused projection**:
One linear weight that holds several independent linear maps side by side.
_Avoid_: fused weight

**Sub-projection**:
One of the independent linear maps in a fused projection.
_Avoid_: segment, column block

**Muon plan**:
The allowlist of the weights and sub-projections that Muon steps.

**Fallback optimizer**:
The optimizer of each parameter that the Muon plan does not name: AdamW or
plain SGD.

**I/O weight**:
A weight at the input or the output of a network. Examples: the embedding, the
LM head, the input and output projections, and the class marker tables.
_Avoid_: boundary weight

### H-Net

**H-Net**:
A model that runs layer stacks at several resolutions. Each stage passes one
position for each chunk to its inner H-Net.

**Stage**:
An encoder, a router and a decoder around an inner H-Net.
_Avoid_: level

**Encoder**:
The layer stack of a stage before its inner H-Net.
_Avoid_: encoder network

**Decoder**:
The layer stack of a stage after its inner H-Net.
_Avoid_: decoder network

**Router**:
The part of a stage that decides each chunk boundary.

**Main stack**:
The innermost layer stack of an H-Net. It runs on the most compressed
sequence.
_Avoid_: main network

**Inner H-Net**:
What a stage runs on its chunks: the later stages and the main stack. An H-Net
with no stage is its main stack alone.
_Avoid_: inner network
