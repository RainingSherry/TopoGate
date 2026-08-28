"""Small unlabeled-fit toy worlds for the C2 mechanism contract.

The fixture constructor creates labels only so a test/audit can verify what
the synthetic world was designed to encode.  The corruption library never
receives those labels.  This is an apparatus sensitivity check, not clustering
evidence for the real panel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToyWorld:
    name: str
    x: np.ndarray
    labels_for_audit: np.ndarray
    signal_description: str


def make_world(
    name: str,
    *,
    n_per_class: int = 48,
    n_features: int = 24,
    seed: int = 20260818,
) -> ToyWorld:
    """Construct World S, V, or M with a known information role.

    ``World S`` uses disjoint support blocks with the same positive value
    distribution.  ``World V`` uses a shared support with class-dependent
    magnitudes.  ``World M`` combines both.  Every world has two balanced
    classes and strictly non-negative values so support tests are unambiguous.
    """

    if name not in {"S", "V", "M"}:
        raise ValueError(f"unknown toy world {name!r}; expected S, V or M")
    if n_per_class < 4 or n_features < 8 or n_features % 4:
        raise ValueError("toy dimensions must be >=4 and n_features divisible by four")

    rng = np.random.default_rng(seed)
    n = 2 * n_per_class
    x = np.zeros((n, n_features), dtype=np.float32)
    labels = np.repeat(np.arange(2, dtype=np.int64), n_per_class)
    support_blocks = (np.arange(0, n_features // 2), np.arange(n_features // 2, n_features))
    shared = np.arange(0, n_features // 2)

    if name in {"S", "M"}:
        for cls in range(2):
            rows = slice(cls * n_per_class, (cls + 1) * n_per_class)
            block = support_blocks[cls]
            # Same value law in both classes; only the active coordinates differ.
            x[rows, block] = rng.uniform(0.8, 1.2, size=(n_per_class, block.size)).astype(np.float32)
    else:
        x[:, shared] = rng.uniform(0.8, 1.2, size=(n, shared.size)).astype(np.float32)

    if name in {"V", "M"}:
        # Keep support fixed while changing magnitude.  The separation is
        # intentionally moderate so the fixture is not a trivial one-hot test.
        x[np.ix_(labels == 0, shared)] *= np.float32(0.8)
        x[np.ix_(labels == 1, shared)] *= np.float32(1.8)

    descriptions = {
        "S": "class information is encoded by support blocks; value law is shared",
        "V": "class information is encoded by values on a shared support",
        "M": "class information is encoded by both support blocks and values",
    }
    return ToyWorld(name, x, labels, descriptions[name])


def label_free_role_scores(x: np.ndarray) -> dict[str, float]:
    """Return label-free structural summaries used by the toy audit.

    These scores intentionally do not attempt to recover the class labels.  A
    separate test may compare them with ``labels_for_audit`` to verify the
    fixture design, but no training code should call that comparison.
    """

    matrix = np.asarray(x, dtype=np.float32)
    support = np.abs(matrix) > 1e-6
    prevalence = np.mean(support, axis=0)
    active_values = np.abs(matrix[support])
    return {
        "n": float(matrix.shape[0]),
        "d": float(matrix.shape[1]),
        "mean_nnz": float(np.mean(np.sum(support, axis=1))),
        "support_entropy": float(
            np.mean(
                -np.where(prevalence > 0, prevalence * np.log(prevalence), 0.0)
                - np.where(prevalence < 1, (1.0 - prevalence) * np.log(1.0 - prevalence), 0.0)
            )
        ),
        "value_std": float(np.std(active_values)) if active_values.size else 0.0,
    }


def audit_world_definitions() -> dict[str, object]:
    """Return deterministic checks that the three worlds have their intended roles."""

    worlds = {name: make_world(name) for name in ("S", "V", "M")}
    checks: dict[str, bool] = {}
    # Support separability: class-specific support signatures differ in S/M,
    # but not in V.  This is only a fixture-definition test.
    for name, world in worlds.items():
        support = np.abs(world.x) > 1e-6
        class_support = [support[world.labels_for_audit == cls].mean(axis=0) for cls in (0, 1)]
        support_gap = float(np.mean(np.abs(class_support[0] - class_support[1])))
        value = world.x.copy()
        active_value_means = [
            float(np.mean(value[world.labels_for_audit == cls][value[world.labels_for_audit == cls] > 0]))
            for cls in (0, 1)
        ]
        value_gap = float(abs(active_value_means[0] - active_value_means[1]))
        checks[f"{name}_support_signal"] = bool(support_gap > 0.1) if name in {"S", "M"} else bool(support_gap < 1e-6)
        checks[f"{name}_value_signal"] = bool(value_gap > 0.1) if name in {"V", "M"} else bool(value_gap < 0.25)
    return {
        "status": "completed_valid" if all(checks.values()) else "protocol_insensitive",
        "labels_used_for_fixture_audit_only": True,
        "fit_labels_allowed": False,
        "checks": checks,
        "worlds": {
            name: {"description": world.signal_description, "shape": list(world.x.shape)}
            for name, world in worlds.items()
        },
    }
