"""Import shim for the p4wrist binding (p4wrist/python/p4wrist.py).

Build the library once with:
    cmake -S p4wrist -B p4wrist/build && cmake --build p4wrist/build
"""
import sys
from pathlib import Path

_BINDING = Path(__file__).resolve().parent / 'p4wrist' / 'python'
if str(_BINDING) not in sys.path:
    sys.path.insert(0, str(_BINDING))

import p4wrist  # noqa: E402

__all__ = ['p4wrist']
