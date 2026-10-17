"""Compatibility alias; implementation lives in csv_engine/importer.py."""
import sys
from csv_engine import importer as _implementation
sys.modules[__name__] = _implementation
