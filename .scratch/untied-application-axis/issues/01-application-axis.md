# 01: Untied parameters on their own application axis

Status: needs-triage

Priority: low

Blocked by: `.scratch/glossary-followups/` 01 to 05 (the glossary changes)

**What to build:** An untied parameter keeps its copies on a new application
axis, with the shape `[applications, …]`. Today the copies lie side by side on
an axis that it already has. The user finds the new layout better organised. ADR-0008 records the
current layout and why the code does not use this one now.

## The idea from the burn-stack grill

Burn fixes the rank of a tensor in its type, and most untied parameters are in
modules of Burn (`Linear`, `Conv1d`). This layout does not need copies of those
modules. A family keeps the stacked tensor beside the module, and it takes the
slice of the application before each forward.

## Costs to weigh

- The default case (no virtual layers, no untied parameter) also gets the
  axis and the slicing, unless a design avoids them.
- `Segmented`, `ProjSpec` (`tiled`) and `InitPolicy` must read the new axis.
- Each family that unties a parameter changes (burn-mamba, burn-deltanet).
- Saved records change shape.

## Questions for triage

- Is there a layout that gives the default case no extra cost?
- If the answer is yes, a new ADR supersedes ADR-0008.

## Comments
