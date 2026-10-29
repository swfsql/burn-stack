# Triage Labels

The skills use five canonical triage roles. This file maps each role to its
label string in the issue tracker of this repo. In the local markdown tracker,
the label string is the value of the `Status:` line of an issue file.

| Label in mattpocock/skills | Label in our tracker | Meaning                                         |
| -------------------------- | -------------------- | ----------------------------------------------- |
| `needs-triage`             | `needs-triage`       | The maintainer must evaluate this issue         |
| `needs-info`               | `needs-info`         | It waits for more information from the reporter |
| `ready-for-agent`          | `ready-for-agent`    | It is fully specified. An AFK agent can do it   |
| `ready-for-human`          | `ready-for-human`    | A human must implement it                       |
| `wontfix`                  | `wontfix`            | Nobody will work on it                          |

When a skill names a role (for example, "apply the AFK-ready triage label"),
use the label string of that role from this table.

To use a different vocabulary, edit the "Label in our tracker" column.
