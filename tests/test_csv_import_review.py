"""Regression coverage for cold imports and semantic-review list contracts."""
import concurrent.futures
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
from csv_engine.verification import review_plan


class Run:
    def __init__(self, checks):
        self.checks = checks

    def json_call(self, stage, mode, prompt, payload, validate):
        return validate({'status': 'accepted', 'reason': 'Checked evidence', 'checks': self.checks})


class ImportReviewTests(unittest.TestCase):
    def test_review_accepts_nine_explicit_checks(self):
        result = review_plan(Run(['check'] * 9), {'queries': []}, 'overview', {})
        self.assertEqual(len(result['checks']), 9)
        self.assertFalse(result['independently_verified'])

    def test_invalid_checks_explain_repair_contract(self):
        for checks in (None, 'checked', {'metric': True}, [], ['check'] * 17):
            with self.subTest(checks=checks), self.assertRaisesRegex(ValueError, 'JSON array of 1 to 16 strings'):
                review_plan(Run(checks), {'queries': []}, 'overview', {})

    def test_compatibility_original_module_has_exports(self):
        # A caller retaining the original module object must also see exports.
        spec = importlib.util.spec_from_file_location('_csv_compat_test', Path('services/csv_llm_service.py'))
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            self.assertTrue(callable(module.is_csv_session))
            self.assertTrue(callable(module.cached_suggestions_csv))
        finally:
            sys.modules.pop('_csv_compat_test', None)

    def test_concurrent_cold_imports(self):
        code = '''
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
barrier = Barrier(8)
def load(i):
    barrier.wait()
    if i % 2:
        from services.csv_llm_service import is_csv_session, cached_suggestions_csv
    else:
        from csv_engine.analyst import is_csv_session, cached_suggestions_csv
    assert is_csv_session({'source_type': 'semi-structured', 'file_type': 'csv'})
    assert callable(cached_suggestions_csv)
with ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(load, range(8)))
'''
        subprocess.run([sys.executable, '-c', code], check=True, timeout=30)


if __name__ == '__main__':
    unittest.main()
