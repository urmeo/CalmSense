# Outputs

All project outputs live here.

| Path | Contents |
| --- | --- |
| `results/` | Committed metrics, tables, historical results and provenance |
| `figures/` | Committed research plots, historical figures and `demo.gif` |
| `models/` | Shipped Random Forest and SHA-256 checksum |
| `dashboard/` | Committed results and signal modules used by the static dashboard |
| `generated/` | Ignored caches, plots, logs, site builds and synthetic runs |

[Experiment commands](../README.md#reproduce-experiments) refresh `results/` and `models/`;
local plots go to `generated/figures/`. Synthetic runs use `generated/demo/{results,figures,models}/`.
Committed `figures/` remain a separate snapshot. See [result provenance](results/README.md).
