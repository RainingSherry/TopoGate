"""V24-Q1 conditional incremental response protocol.

V24 intentionally reuses the audited V23 frozen probe for fit/profile while
keeping its synthetic construction and conditional evaluation independent.
"""

from .config import V24Q1Config

__all__ = ["V24Q1Config"]
