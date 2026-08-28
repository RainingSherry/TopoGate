"""V16.1 predictive topology gate.

This package is intentionally independent from V16.  It keeps topology out of
the Stage-A representation and evaluates each candidate edge on a count view
that was not used to construct the candidate graph.
"""

from .config import V16_1Config, load_config

__all__ = ["V16_1Config", "load_config", "fit_v16_1", "run_v16_1"]


def __getattr__(name: str):
    if name in {"fit_v16_1", "run_v16_1"}:
        from .run import fit_v16_1, run_v16_1

        return {"fit_v16_1": fit_v16_1, "run_v16_1": run_v16_1}[name]
    raise AttributeError(name)
