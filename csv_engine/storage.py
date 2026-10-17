from contextlib import contextmanager
from pathlib import Path
import sqlite3

class CsvAnalysisError(RuntimeError):
    def __init__(self, message, trace=None):
        super().__init__(message)
        self.trace = trace or {}


def _text(value, limit=500):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("Missing or oversized text field")
    return value.strip()


def _strings(value, limit=10):
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError("Expected a bounded list")
    return [_text(v) for v in value]


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


@contextmanager
def _connection(session):
    uri = session.get("db_uri", "")
    if not uri.startswith("sqlite:///"):
        raise CsvAnalysisError("Reconnect the CSV dataset before analyzing it.")
    path = Path(uri[len("sqlite:///"):]).resolve()
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=3)
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()


