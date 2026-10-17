"""Compatibility alias; implementation lives in csv_engine/analyst.py."""
import sys
from csv_engine import analyst as _implementation
sys.modules[__name__] = _implementation
