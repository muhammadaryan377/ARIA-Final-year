"""Hermetic regression tests for authenticated ARIA HTML/CSV data exports."""

import csv
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.data_export_routes import MAX_CSV_ROWS, router
from core.deps import get_current_user


class DataExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        db_file = folder / "records.sqlite"
        with sqlite3.connect(db_file) as conn:
            conn.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, note TEXT)")
            conn.executemany("INSERT INTO records VALUES (?, ?)", [
                (1, "normal"),
                (2, "=1+1"),
                (3, " \t@SUM(1)"),
            ])

        schema_path = folder / "schema.json"
        schema_path.write_text(json.dumps({
            "tables": {
                "records": {
                    "columns": [
                        {"column": "id", "data_type": "INTEGER"},
                        {"column": "note", "data_type": "TEXT"},
                    ],
                    "primary_key": ["id"],
                    "row_count": 3,
                    "empty": False,
                },
            },
            "summary": {"total_db_tables": 1},
        }), encoding="utf-8")

        session = {
            "db": {".db": str(db_file)},
            "db_uri": f"sqlite:///{db_file}",
            "db_type": "sqlite",
            "dialect": "sqlite",
            "source_type": "semi-structured",
            "db_name": "<script>alert(1)</script>",
            "schema_path": schema_path,
        }
        session_patch = patch("api.data_export_routes.get_session", return_value=session)
        session_patch.start()
        self.addCleanup(session_patch.stop)
        data_routes_patch = patch("api.data_routes.get_session", return_value=session)
        data_routes_patch.start()
        self.addCleanup(data_routes_patch.stop)

        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app)

    def authenticate(self):
        self.app.dependency_overrides[get_current_user] = lambda: {"user_id": 123}
        self.addCleanup(self.app.dependency_overrides.clear)

    def test_export_requires_authentication(self):
        response = self.client.get("/api/data/export?format=html")
        self.assertEqual(response.status_code, 401)

    def test_csv_exports_rows_and_escapes_spreadsheet_formulas(self):
        self.authenticate()
        response = self.client.get("/api/data/export?format=csv&table=records")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("attachment", response.headers["content-disposition"])
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["x-aria-export-row-limit"], str(MAX_CSV_ROWS))
        data = list(csv.reader(io.StringIO(response.content.decode("utf-8-sig"))))
        self.assertEqual(data, [
            ["id", "note"],
            ["1", "normal"],
            ["2", "'=1+1"],
            ["3", "' \t@SUM(1)"],
        ])

    def test_table_whitelist_and_format_validation(self):
        self.authenticate()
        self.assertEqual(self.client.get("/api/data/export?format=csv").status_code, 400)
        self.assertEqual(self.client.get("/api/data/export?format=csv&table=sqlite_master").status_code, 400)
        self.assertEqual(self.client.get("/api/data/export?format=zip").status_code, 422)

    def test_html_report_contains_summary_and_escapes_source_strings(self):
        self.authenticate()
        response = self.client.get("/api/data/export?format=html")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("ARIA Data Report", response.text)
        self.assertIn("Data Quality Findings", response.text)
        self.assertIn('<div class="stat">Tables<b>1</b>', response.text)
        self.assertIn("records", response.text)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", response.text)
        self.assertNotIn("<script>alert(1)</script>", response.text)
        self.assertIn("default-src 'none'", response.headers["content-security-policy"])


if __name__ == "__main__":
    unittest.main()
