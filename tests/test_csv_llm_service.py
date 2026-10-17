"""Offline contract tests: real SQLite, scripted provider responses, no API keys."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import ast
import logging
from types import SimpleNamespace
from unittest.mock import Mock

from services.csv_llm_service import ask_csv, dashboard, CsvAnalysisError, _context, _execute


class Provider:
    provider = "cloud"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def model_for(self, role):
        return "test-" + role

    def chat(self, role, messages, **kwargs):
        self.calls.append((role, messages, kwargs))
        if not self.responses:
            raise RuntimeError("Unexpected provider invocation")
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, str) else json.dumps(value)


def query(sql='SELECT SUM(revenue) AS total_revenue, SUM(profit) AS total_profit FROM facts',
          measures=None, dimensions=None, chart="table", title="Totals"):
    return {"title": title, "sql": sql, "measures": measures or ["total_revenue", "total_profit"],
            "dimensions": dimensions or [], "chart": chart, "format": "number",
            "reason": "Use the two separate stored measures requested by the user."}


PLAN = {"status": "ready", "queries": [query()]}
STORY = {"answer": "Revenue is {{0:0:total_revenue}} and profit is {{0:0:total_profit}}.",
         "findings": [], "followups": ["Show revenue by city"]}


class CsvLlmTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'test.db'
        with sqlite3.connect(self.path) as conn:
            conn.execute('CREATE TABLE facts(city TEXT, quantity INTEGER, unit_price REAL, revenue REAL, profit REAL)')
            conn.executemany('INSERT INTO facts VALUES(?,?,?,?,?)',
                             [('A', 2, 100, 200, 60), ('B', 3, 100, 300, 90)])
        self.session = {"source_type": "semi-structured", "file_type": "csv", "db_name": "facts",
                        "db_uri": 'sqlite:///' + str(self.path), "provider_name": "cloud"}

    def tearDown(self):
        self.tmp.cleanup()

    def provider(self, *responses):
        p = Provider(*responses)
        self.session['provider'] = p
        return p

    def test_two_totals_are_not_rewritten_or_treated_as_categories(self):
        p = self.provider(PLAN, STORY)
        result = ask_csv(self.session, 7, 'Total revenue and profit?')
        self.assertEqual(result['data'], [{'total_revenue': 500.0, 'total_profit': 150.0}])
        self.assertEqual(result['answer'], 'Revenue is 500 and profit is 150.')
        self.assertEqual(result['dimensions'], [])
        self.assertEqual(result['comparison_period'], {})
        self.assertEqual(result['chart_spec']['chart_type'], 'table')
        self.assertFalse(result['verdict']['verified'])
        self.assertEqual(len(p.calls), 2)
        self.assertEqual(result['llm_trace']['provider_invocation_count'], 2)
        self.assertTrue(all(c['status'] == 'success' for c in result['llm_trace']['calls']))

    def test_unknown_meaning_requires_llm_clarification(self):
        p = self.provider({'status': 'clarification', 'clarification': 'Which metric do you mean?'})
        result = ask_csv(self.session, 7, 'How are we doing?')
        self.assertEqual(result['status'], 'clarification')
        self.assertIsNone(result['sql_used'])
        self.assertEqual(len(p.calls), 1)

    def test_unavailable_provider_does_not_return_template(self):
        self.provider(RuntimeError('secret-api-key-do-not-leak'))
        with self.assertRaises(CsvAnalysisError) as exc:
            ask_csv(self.session, 7, 'Revenue?')
        self.assertNotIn('secret-api-key', str(exc.exception))
        self.assertFalse(exc.exception.trace['fallback_used'])

    def test_invalid_json_has_one_repair(self):
        p = self.provider('not json', PLAN, STORY)
        result = ask_csv(self.session, 7, 'Totals?')
        self.assertEqual(len(p.calls), 3)
        self.assertEqual(result['llm_trace']['calls'][0]['status'], 'invalid_response')

    def test_repeated_invalid_response_fails_closed(self):
        p = self.provider('not json', 'not json')
        with self.assertRaises(CsvAnalysisError):
            ask_csv(self.session, 7, 'Totals?')
        self.assertEqual(len(p.calls), 2)

    def test_wrong_column_is_repaired_before_success(self):
        bad = {'status':'ready','queries':[query('SELECT SUM(missing) AS total_revenue FROM facts', ['total_revenue'])]}
        self.provider(bad, PLAN, STORY)
        result = ask_csv(self.session, 7, 'Totals?')
        self.assertEqual(result['data'][0]['total_profit'], 150)

    def test_narration_rejects_invented_numeric_claims(self):
        bad = {'answer':'Revenue is 9000.', 'findings':[], 'followups':[]}
        self.provider(PLAN, bad, bad)
        with self.assertRaises(CsvAnalysisError):
            ask_csv(self.session, 7, 'Totals?')

    def test_invalid_cell_reference_rejected(self):
        bad = {'answer':'Profit is {{0:5:profit}}.', 'findings':[], 'followups':[]}
        self.provider(PLAN, bad, bad)
        with self.assertRaises(CsvAnalysisError):
            ask_csv(self.session, 7, 'Totals?')

    def test_sql_sandbox(self):
        ctx = _context(self.session)
        for sql in ['DELETE FROM facts', 'SELECT * FROM sqlite_master',
                    "SELECT load_extension('evil') FROM facts", 'SELECT 999 AS profit',
                    'SELECT * FROM facts; DROP TABLE facts',
                    "ATTACH DATABASE '/tmp/other.db' AS other"]:
            with self.subTest(sql=sql), self.assertRaises((ValueError, sqlite3.Error)):
                _execute(self.session, ctx, sql)
        with sqlite3.connect(self.path) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM facts').fetchone()[0], 2)

    def test_result_bound(self):
        with sqlite3.connect(self.path) as conn:
            conn.executemany('INSERT INTO facts VALUES(?,?,?,?,?)', [('A',1,1,1,1)] * 201)
        with self.assertRaises(ValueError):
            _execute(self.session, _context(self.session), 'SELECT city FROM facts')

    def test_grouped_query_preserves_llm_decision(self):
        plan = {'status':'ready', 'queries':[query('SELECT city AS region, SUM(profit) AS earned FROM facts GROUP BY city ORDER BY earned DESC', ['earned'], ['region'], 'bar')]}
        self.provider(plan, {'answer':'{{0:0:region}} has profit {{0:0:earned}}.', 'findings':[], 'followups':[]})
        r = ask_csv(self.session, 7, 'Which city earns most?')
        self.assertEqual(r['data'][0], {'region':'B','earned':90.0})
        self.assertEqual(r['chart_spec']['x'], 'region')

    def test_dashboard_cache_is_explicit_and_does_not_call_provider(self):
        plan = {'status':'ready', 'queries':[query('SELECT SUM(profit) AS profit_total FROM facts', ['profit_total'], title='Total profit')]}
        p = self.provider(plan, {'answer':'Profit is {{0:0:profit_total}}.', 'findings':[], 'followups':[]})
        first = dashboard(self.session, 7)
        second = dashboard(self.session, 7)
        self.assertFalse(first['llm_trace']['cached'])
        self.assertTrue(second['llm_trace']['cached'])
        self.assertEqual(second['llm_trace']['calls_this_request'], 0)
        self.assertEqual(len(p.calls), 2)
        self.assertEqual(first['metrics'][0]['rawValue'], 150)
        self.assertEqual(first['metrics'][0]['format'], 'number')
        self.assertNotIn('change', first['metrics'][0])

    def test_schema_change_invalidates_dashboard_cache(self):
        plan = {'status':'ready', 'queries':[query('SELECT SUM(profit) AS profit_total FROM facts', ['profit_total'])]}
        story = {'answer':'Profit is {{0:0:profit_total}}.', 'findings':[], 'followups':[]}
        p = self.provider(plan, story, plan, story)
        dashboard(self.session, 7)
        with sqlite3.connect(self.path) as conn:
            conn.execute('INSERT INTO facts VALUES("C",1,1,1,20)')
        result = dashboard(self.session, 7)
        self.assertEqual(len(p.calls), 4)
        self.assertEqual(result['metrics'][0]['rawValue'], 170)

    def test_multi_metric_total_cannot_be_a_fake_dimension(self):
        bad = copy.deepcopy(PLAN)
        bad['queries'][0]['measures'] = ['total_revenue']
        bad['queries'][0]['dimensions'] = ['total_profit']
        self.provider(bad, bad)
        with self.assertRaises(CsvAnalysisError):
            ask_csv(self.session, 7, 'Totals?')

    def test_null_and_empty_are_not_zero(self):
        plan = {'status':'ready', 'queries':[query('SELECT SUM(profit) AS profit_total FROM facts WHERE quantity < 0', ['profit_total'])]}
        self.provider(plan, {'answer':'Profit is {{0:0:profit_total}}.', 'findings':[], 'followups':[]})
        result = ask_csv(self.session, 7, 'Profit for negative quantities?')
        self.assertIsNone(result['data'][0]['profit_total'])
        self.assertIn('unavailable (NULL)', result['answer'])

    def _ask_route(self):
        # Exercise the real route body without importing unrelated optional ML dependencies.
        source = Path(__file__).resolve().parents[1] / 'api/query_routes.py'
        fn = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'ask')
        fn.decorator_list = []
        fn.args.defaults = []
        for arg in fn.args.args:
            arg.annotation = None
        class HTTPError(Exception):
            def __init__(self, status_code, detail):
                self.status_code, self.detail = status_code, detail
        saved = Mock()
        auth = SimpleNamespace(add_history=Mock(return_value=42))
        namespace = {'get_session': lambda uid: self.session, 'require_data': lambda r: 'cloud',
                     'Path': Path, 'json': json, 'auth': auth, 'save_session': saved,
                     'HTTPException': HTTPError, 'log': logging.getLogger('csv_route_test')}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(source), 'exec'), namespace)
        self.session['processed_path'] = Path(self.tmp.name) / 'processed.json'
        self.session['dialect'] = 'sqlite'
        req = SimpleNamespace(goal='Total revenue and profit?', conversation_id='new-conversation',
                              metric=None, filters=None, time_range=None, grouping=None, previous_goal=None)
        return namespace['ask'], req, auth, saved, HTTPError

    def test_route_persists_executed_answer_and_resets_conversation_context(self):
        self.provider(PLAN, STORY)
        self.session.update(conversation_id='old', conversation_history=[{'goal':'unrelated'}])
        route, req, auth, saved, _ = self._ask_route()
        result = route(None, req, {'user_id':7})
        self.assertEqual(result['history_id'], 42)
        self.assertEqual(result['turn_number'], 1)
        self.assertEqual(self.session['conversation_history'][0]['goal'], req.goal)
        persisted = json.loads(self.session['processed_path'].read_text())
        self.assertEqual(persisted['data'][0]['total_profit'], 150)
        auth.add_history.assert_called_once()
        saved.assert_called_once()

    def test_route_provider_failure_has_trace_and_does_not_save_fake_history(self):
        self.provider(RuntimeError('offline'))
        route, req, auth, saved, error_type = self._ask_route()
        with self.assertRaises(error_type) as exc:
            route(None, req, {'user_id':7})
        self.assertEqual(exc.exception.status_code, 502)
        self.assertEqual(exc.exception.detail['llm_trace']['calls'][0]['status'], 'error')
        auth.add_history.assert_not_called()
        saved.assert_not_called()


if __name__ == '__main__':
    unittest.main()
