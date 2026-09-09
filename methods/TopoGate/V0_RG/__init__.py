"""TopoGate V0_RG: label-free RG-NeighborMix-scMAE dataset adapter."""

from .config import V0_RGConfig, load_config


def fit_predict(*args, **kwargs):
    """Lazily import the training stack for lightweight config/manifest use."""
    from .trainer import fit_predict as _fit_predict

    return _fit_predict(*args, **kwargs)

__all__ = ["V0_RGConfig", "fit_predict", "load_config"]
