#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(parent for parent in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
            if (parent / "methods" / "TopoGate" / "V18_scmae_latent_gate").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.TopoGate.V18_scmae_latent_gate.run import main


if __name__ == "__main__":
    main()
