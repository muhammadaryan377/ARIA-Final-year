"""Detect leaked handles on every OS, even where open files can be unlinked."""
import contextlib
import io
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from csv_engine.chat_store import ChatStore
from csv_engine import evaluation


class ConnectionLifetimeTests(unittest.TestCase):
    def assertClosed(self, connection):
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute('SELECT 1')

    def test_chat_connection_commits_and_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(Path(tmp)/'chat.db')
            with store.connection() as connection:
                connection.execute("INSERT INTO csv_chats VALUES(1,'d','c','{}','now')")
            self.assertClosed(connection)
            self.assertIsNotNone(store.load(1,'d','c'))

    def test_chat_connection_rolls_back_and_closes_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ChatStore(Path(tmp)/'chat.db')
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                with store.connection() as connection:
                    connection.execute("INSERT INTO csv_chats VALUES(1,'d','c','{}','now')")
                    raise RuntimeError('interrupted')
            self.assertClosed(connection)
            self.assertIsNone(store.load(1,'d','c'))

    def test_evaluation_closes_every_sqlite_connection(self):
        original = sqlite3.connect
        connections = []
        def tracked(*args, **kwargs):
            connection = original(*args, **kwargs)
            connections.append(connection)
            return connection
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp)/'report.json'
            with patch('sqlite3.connect', side_effect=tracked), patch('sys.argv', ['evaluation','--output',str(report)]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluation.main(),0)
            self.assertTrue(report.exists())
            self.assertGreater(len(connections), 0)
            for connection in connections:
                self.assertClosed(connection)

if __name__=='__main__': unittest.main()
