"""V23 frozen masked-recovery response profiling.

The package initializer intentionally avoids importing torch so CLI entrypoints
can establish physical-GPU isolation before the runtime is loaded.
"""

__all__: list[str] = []
