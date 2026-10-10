"""Public benchmarks, run the way openodke runs: fetch, prepare, run three ways, score.

Every dataset module has the same functions:

- `fetch(dest)` downloads the official files; nothing is redistributed here;
- `prepare(root, ..., out)` writes a runnable directory — one text file per
  document, the dataset's ontology in openodke's form, its gold rows, and an
  `odke.json` run config (`paper=True` for ODKE+'s own grounder and gate);
- `run(prepared)` runs that config three ways — extraction alone, + grounding,
  + corroboration (`openodke.eval.ablation.ablate`) — and scores each row with
  the dataset's own metrics;
- `score(...)` those metrics on their own, for triples from any system.

A new dataset is one more module with these functions, and a line in `DATASETS`.
"""

from __future__ import annotations

from types import ModuleType

from openodke.eval.datasets import redocred, text2kgbench, trex

DATASETS: dict[str, ModuleType] = {
    "text2kgbench": text2kgbench,
    "redocred": redocred,
    "trex": trex,
}

__all__ = ["DATASETS", "redocred", "text2kgbench", "trex"]
