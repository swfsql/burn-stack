# An untied parameter keeps its copies side by side on an existing axis

A real layer can hold a parameter once for each of its applications (an
untied parameter). The copies lie side by side on an axis that the parameter
already has. For example, a gain of `d_model = 4` values, untied over 3
applications, has the shape `[12]`. So the parameter keeps its rank, its type
and its path. Application `k` reads its own copy through `untied::view`. A
real layer with one application keeps its stock shape. So the default case
(no virtual layers, no untied parameter) has no extra cost.

## Considered Options

- **One parameter for each application** (`gamma.0`, `gamma.1`, …): a struct
  declares its fields in the code. The stack sets the count of applications
  later, when it builds the layer.
- **A new application axis** (shape `[3, 4]`): it is better organised. But
  Burn fixes the rank of a tensor in its type, and most untied parameters are
  in modules of Burn (`Linear`, `Conv1d`). So a family stores the stacked
  tensor beside the module, and it takes the slice of the application before
  each forward. Then the default case also has the axis and the slicing.
  `Segmented`, `ProjSpec` and `InitPolicy` also need to read the new axis.
  This option stays open as a low-priority proposal.

## Consequences

- `tile` copies one draw into every application. So an untied stack starts
  equal to the tied stack.
- A saved record holds the longer shape.
- `InitPolicy` redraws every value, so `retie` makes the copies equal again
  after it.
- `ProjSpec::tiled` lets Muon step each copy alone.
