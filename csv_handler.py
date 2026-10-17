"""Compatibility alias; implementation lives in csv_engine/handler.py."""
import sys
from csv_engine import handler as _implementation
sys.modules[__name__] = _implementation
