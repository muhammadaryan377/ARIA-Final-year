"""Compatibility alias; application routes import csv_engine.analyst directly."""
import sys
from csv_engine import analyst as _implementation

# Populate the original module too: concurrent importers can retain its identity
# while sys.modules is being redirected to the canonical implementation.
globals().update({name: value for name, value in vars(_implementation).items()
                  if not name.startswith("__")})
sys.modules[__name__] = _implementation
