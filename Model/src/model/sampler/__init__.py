"""
Legacy compatibility: re-export sampling from model.sampling.
Prefer importing from `model.sampling` going forward.
"""
from ..sampling import *

__all__ = [name for name in dir() if not name.startswith('_')]

# Sampling modules for generation
