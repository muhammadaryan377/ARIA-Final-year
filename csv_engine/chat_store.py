"""Durable, user- and dataset-scoped CSV conversations; no provider credentials."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from contextlib import closing
import sqlite3
from pathlib import Path


def dataset_key(fingerprint):
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()


class ChatStore:
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=10)) as conn, conn:
            conn.row_factory = sqlite3.Row
            conn.execute('''CREATE TABLE IF NOT EXISTS csv_chats (
                user_id INTEGER NOT NULL, dataset TEXT NOT NULL, conversation_id TEXT NOT NULL,
                payload TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY(user_id, dataset, conversation_id))''')
            yield conn

    def load(self, user_id, dataset, conversation_id=None):
        with self.connection() as conn:
            if conversation_id:
                row = conn.execute('SELECT conversation_id,payload,updated_at FROM csv_chats WHERE user_id=? AND dataset=? AND conversation_id=?',
                                   (user_id, dataset, conversation_id)).fetchone()
            else:
                row = conn.execute('SELECT conversation_id,payload,updated_at FROM csv_chats WHERE user_id=? AND dataset=? ORDER BY updated_at DESC LIMIT 1',
                                   (user_id, dataset)).fetchone()
        if row is None:
            return None
        return {'conversation_id': row['conversation_id'], 'updated_at': row['updated_at'], **json.loads(row['payload'])}

    def save(self, user_id, dataset, conversation_id, *, context=None, display=None, new=False):
        if not isinstance(conversation_id, str) or not 1 <= len(conversation_id) <= 100:
            raise ValueError('Invalid conversation identifier')
        with self.connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT payload FROM csv_chats WHERE user_id=? AND dataset=? AND conversation_id=?',
                               (user_id, dataset, conversation_id)).fetchone()
            payload = json.loads(row['payload']) if row and not new else {'history': [], 'turns': []}
            if context is not None:
                payload['history'] = (payload['history'] + [context])[-10:]
            if display is not None:
                payload['turns'] = (payload['turns'] + [display])[-10:]
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            # Keep the most recent complete turn rather than truncate its evidence.
            while len(encoded.encode()) > 4_000_000 and len(payload['turns']) > 1:
                payload['turns'].pop(0)
                payload['history'] = payload['history'][-len(payload['turns']):]
                encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            if len(encoded.encode()) > 4_000_000:
                raise ValueError('Conversation result is too large to persist')
            conn.execute('INSERT OR REPLACE INTO csv_chats VALUES(?,?,?,?,?)',
                         (user_id, dataset, conversation_id, encoded, datetime.now(timezone.utc).isoformat()))
            # Bounded retention: latest twenty conversations per user, across datasets.
            conn.execute('DELETE FROM csv_chats WHERE user_id=? AND rowid NOT IN (SELECT rowid FROM csv_chats WHERE user_id=? ORDER BY updated_at DESC LIMIT 20)', (user_id, user_id))

    def delete(self, user_id, dataset, conversation_id):
        with self.connection() as conn:
            return conn.execute('DELETE FROM csv_chats WHERE user_id=? AND dataset=? AND conversation_id=?',
                                (user_id, dataset, conversation_id)).rowcount > 0
