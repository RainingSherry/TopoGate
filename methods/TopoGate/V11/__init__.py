"""TopoGate V11: dynamic trustworthy-topology deep clustering."""
from __future__ import annotations

from typing import Any

from .config import V11Config, load_config
from .model import TopoGateV11


def __getattr__(name: str) -> Any:
    if name in {"fit_v11", "run_v11"}:
        from . import run as _run

        value = getattr(_run, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = ["V11Config", "TopoGateV11", "fit_v11", "load_config", "run_v11"]
