# -*- coding: utf-8 -*-

from .naive import naive_nsa

try:
    from .parallel import parallel_nsa
except Exception:
    parallel_nsa = None

__all__ = [
    'naive_nsa',
    'parallel_nsa',
]
