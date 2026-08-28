"""V16 predictive graph gate for sparse count clustering.

V16 is intentionally isolated from the earlier V-series.  The topology path
is an assignment-space readout driven by held-out count support; it never
changes the Stage-A encoder.
"""

from .config import V16Config, load_config

__all__ = ["V16Config", "load_config", "fit_v16", "run_v16"]


def __getattr__(name: str):
    if name in {"fit_v16", "run_v16"}:
        from .run import fit_v16, run_v16

        return {"fit_v16": fit_v16, "run_v16": run_v16}[name]
    raise AttributeError(name)
