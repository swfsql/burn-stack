# Issue tracker: Local Markdown

The issues and specs of this repo are markdown files in `.scratch/`.

## Conventions

- Each feature has one directory: `.scratch/<feature-slug>/`.
- The spec is `.scratch/<feature-slug>/spec.md`.
- Each implementation issue is one file:
  `.scratch/<feature-slug>/issues/<NN>-<slug>.md`. The numbers start at `01`.
  Do not put all the tickets in one file.
- A `Status:` line near the top of each issue file records its triage state.
  `triage-labels.md` gives the role strings.
- Comments and the conversation history go at the bottom of the file, below a
  `## Comments` heading.

## When a skill says "publish to the issue tracker"

Create a new file in `.scratch/<feature-slug>/`. Create the directory if it does
not exist.

## When a skill says "fetch the relevant ticket"

Read the file at the given path. Usually, the user gives the path or the issue
number.

## Wayfinding operations

`/wayfinder` uses these. The **map** is one file, and each **child** ticket is
one more file.

- **Map**: `.scratch/<effort>/map.md`. It holds the Notes / Decisions-so-far /
  Fog body.
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`. The numbers start
  at `01`. The body holds the question. A `Type:` line records the ticket type
  (`research`/`prototype`/`grilling`/`task`). A `Status:` line records
  `claimed`/`resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked
  when each file on that line is `resolved`.
- **Frontier**: find the files in `.scratch/<effort>/issues/` that are open,
  unblocked and unclaimed. The lowest number goes first.
- **Claim**: set `Status: claimed` and save the file before you do other work.
- **Resolve**: add the answer below an `## Answer` heading and set
  `Status: resolved`. Then add a context pointer (gist + link) to the
  Decisions-so-far of `map.md`.
