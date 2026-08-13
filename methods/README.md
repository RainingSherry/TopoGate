# methods

This directory contains the TopoGate implementations, the scMAE runtime
dependencies used by them, and small shared utilities. External baseline
source trees and dataset binaries are intentionally excluded from this public
snapshot.

Start with [`TopoGate/CORE_CODE_INDEX.md`](TopoGate/CORE_CODE_INDEX.md) for the
version boundaries and runnable entry points. Each version keeps its own
configuration, trainer, tests, and output contract.

The runners expect the caller to provide a real dataset path. They do not
download data or infer labels from this repository.
