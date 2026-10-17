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

    def test_overview_executes_and_exposes_multiple_queries(self):
        grouped = query('SELECT city AS city, SUM(profit) AS profit FROM facts GROUP BY city', ['profit'], ['city'], 'bar')
        plan = {'status': 'ready', 'queries': [query(), grouped],
                'understanding': {'description': 'Sales records', 'assumptions': [], 'limitations': []}}
        p = self.provider(plan, {'answer': 'Revenue {{0:0:total_revenue}}. City {{1:0:city}} earned {{1:0:profit}}.', 'findings': [], 'followups': []})
        result = ask_csv(self.session, 7, 'can you tell me about my data something')
        self.assertEqual(len(result['analysis_results']), 2)
        self.assertEqual(len(result['evidence']), 2)
        self.assertEqual(result['analysis_results'][1]['data'][1]['profit'], 90)
        instructions = '\n'.join(m['content'] for m in p.calls[0][1] if m['role'] == 'system')
        self.assertIn('without asking the user to pick a metric', instructions)
        self.assertTrue(all(len(m['content']) <= 3000 for m in p.calls[0][1] if m['role'] == 'system'))

    def test_investigation_uses_initial_evidence(self):
        plan = {**PLAN, 'needs_investigation': True}
        extra = {'status': 'ready', 'queries': [query('SELECT city AS city, SUM(profit) AS profit FROM facts GROUP BY city', ['profit'], ['city'], 'bar')]}
        p = self.provider(plan, extra, {'answer': 'Total profit {{0:0:total_profit}}; {{1:1:city}} contributes {{1:1:profit}}.', 'findings': [], 'followups': []})
        result = ask_csv(self.session, 7, 'Explore profit distribution')
        self.assertEqual(len(p.calls), 3)
        self.assertEqual(result['llm_trace']['calls'][1]['stage'], 'investigation')
        self.assertEqual(len(result['analysis_results']), 2)
        self.assertIn('total_profit', ''.join(m['content'] for m in p.calls[1][1]))

    def test_investigation_can_stop_without_more_queries(self):
        self.provider({**PLAN, 'needs_investigation': True}, {'status': 'ready', 'queries': []}, STORY)
        result = ask_csv(self.session, 7, 'Explore totals')
        self.assertEqual(len(result['analysis_results']), 1)

    def test_profile_missing_duplicates_and_no_raw_text_samples(self):
        with sqlite3.connect(self.path) as conn:
            conn.executemany('INSERT INTO facts VALUES(?,?,?,?,?)', [('private@example.com', None, 10, 20, None)] * 2)
        ctx = _context(self.session)
        self.assertEqual(ctx['profile']['duplicate_rows'], 1)
        self.assertEqual(next(c for c in ctx['columns'] if c['name'] == 'profit')['null_count'], 2)
        self.assertNotIn('private@example.com', json.dumps(ctx))
        self.assertEqual(ctx['profile']['scope'], 'full dataset')

    def test_profile_change_clears_old_semantic_context(self):
        _context(self.session)
        self.session['csv_dataset_understanding'] = {'description': 'Old context'}
        with sqlite3.connect(self.path) as conn:
            conn.execute('INSERT INTO facts VALUES("C",1,1,1,1)')
        self.assertEqual(_context(self.session)['row_count'], 3)
        self.assertNotIn('csv_dataset_understanding', self.session)

    def test_empty_dataset_reports_count_not_invented_totals(self):
        with sqlite3.connect(self.path) as conn: conn.execute('DELETE FROM facts')
        self.provider({'status': 'ready', 'queries': [query('SELECT COUNT(*) AS records FROM facts', ['records'])]},
            {'answer': 'There are {{0:0:records}} records.', 'findings': [], 'followups': []})
        result = ask_csv(self.session, 7, 'Summarize my data')
        self.assertEqual(result['data'], [{'records': 0}])
        self.assertEqual(result['dataset_profile']['row_count'], 0)

    def test_dimension_only_dataset(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute('CREATE TABLE categories(tag TEXT)')
            conn.executemany('INSERT INTO categories VALUES(?)', [('red',), ('blue',), ('red',)])
        self.session['db_name'] = 'categories'
        plan = {'status': 'ready', 'queries': [query('SELECT tag AS tag, COUNT(*) AS records FROM categories GROUP BY tag ORDER BY records DESC', ['records'], ['tag'], 'bar')]}
        self.provider(plan, {'answer': '{{0:0:tag}} has {{0:0:records}} records.', 'findings': [], 'followups': []})
        result = ask_csv(self.session, 7, 'Describe this data')
        self.assertEqual(result['data'][0], {'tag': 'red', 'records': 2})

    def test_zero_denominator_stays_unavailable(self):
        self.provider({'status': 'ready', 'queries': [query('SELECT SUM(profit)/NULLIF(SUM(revenue-revenue),0) AS ratio FROM facts', ['ratio'])]},
            {'answer': 'Ratio is {{0:0:ratio}}.', 'findings': [], 'followups': []})
        self.assertIsNone(ask_csv(self.session, 7, 'Calculate ratio')['data'][0]['ratio'])

    def test_date_coverage_can_be_text_scalar_results(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute('ALTER TABLE facts ADD COLUMN recorded TEXT')
            conn.execute("UPDATE facts SET recorded='2026-10-17'")
        plan = {'status': 'ready', 'queries': [{'title': 'Coverage', 'reason': 'Observed date coverage',
            'sql': 'SELECT MIN(recorded) AS start, MAX(recorded) AS end FROM facts',
            'measures': [], 'dimensions': ['start', 'end'], 'chart': 'table', 'format': 'number'}]}
        self.provider(plan, {'answer': 'Coverage is {{0:0:start}} through {{0:0:end}}.', 'findings': [], 'followups': []})
        self.assertIn('2026-10-17', ask_csv(self.session, 7, 'What is the date range?')['answer'])

    def test_too_many_queries_and_duplicate_queries_rejected(self):
        for queries in ([query()] * 7, [query()] * 2):
            self.provider({'status': 'ready', 'queries': queries}, {'status': 'ready', 'queries': queries})
            with self.assertRaises(CsvAnalysisError): ask_csv(self.session, 7, 'Overview')

    def test_llm_suggestions_cached_and_invalidated_by_model(self):
        from services.csv_llm_service import suggestions_csv, cached_suggestions_csv
        questions = ['Describe the dataset', 'Total profit?', 'Profit by city?', 'Quantity distribution?']
        p = self.provider({'questions': questions})
        self.assertEqual(suggestions_csv(self.session, 7)['suggestions'], questions)
        self.assertEqual(suggestions_csv(self.session, 7)['source'], 'llm_cache')
        self.assertEqual(len(p.calls), 1)
        self.session['provider_models'] = {'cloud': {'semi': 'changed'}}
        self.assertEqual(cached_suggestions_csv(self.session), [])

    def test_clarification_is_kept_for_next_turn(self):
        self.provider({'status': 'clarification', 'clarification': 'Which field represents age?'})
        route, req, auth, saved, _ = self._ask_route()
        result = route(None, req, {'user_id': 7})
        self.assertEqual(self.session['conversation_history'][-1]['clarification'], result['message'])
        auth.add_history.assert_not_called()
        saved.assert_called_once()


    def test_repair_receives_rejected_candidate_and_trace_reason(self):
        p = self.provider('not json', PLAN, STORY)
        result = ask_csv(self.session, 7, 'Totals?')
        repair_context = ' '.join(m['content'] for m in p.calls[1][1])
        self.assertIn('not json', repair_context)
        self.assertIn('Expecting value', result['llm_trace']['calls'][0]['validation_error'])

    def test_multi_measure_chart_is_presented_as_table(self):
        plan = copy.deepcopy(PLAN)
        plan['queries'][0]['chart'] = 'bar'
        plan['queries'][0].pop('format')
        self.provider(plan, STORY)
        result = ask_csv(self.session, 7, 'Totals?')
        self.assertEqual(result['chart_spec']['chart_type'], 'table')
        self.assertEqual(result['data'][0]['total_profit'], 150)

    def test_dashboard_accepts_multi_measure_scalar_query(self):
        self.provider(PLAN, STORY)
        result = dashboard(self.session, 7)
        self.assertEqual([m['rawValue'] for m in result['metrics']], [500, 150])

    def test_narrator_gets_copyable_reference_catalogue(self):
        p = self.provider(PLAN, STORY)
        ask_csv(self.session, 7, 'Totals?')
        payload = ''.join(m['content'] for m in p.calls[1][1] if m['role'] == 'user')
        self.assertIn('evidence_cells', payload)
        self.assertIn('{{0:0:total_profit}}', payload)

    def test_narration_general_intro_with_grounded_findings(self):
        self.provider(PLAN, {'answer': 'Here is an overview of the stored measures.',
            'findings': ['Revenue {{0:0:total_revenue}}; profit {{0:0:total_profit}}.'], 'followups': []})
        result = ask_csv(self.session, 7, 'Overview')
        self.assertIn('500', result['findings'][0])


    def test_current_chat_followup_receives_prior_answer_and_values(self):
        p = self.provider(PLAN, STORY, {'question': 'Repeat total revenue and total profit', 'uses_history': True}, PLAN, STORY)
        route, req, _, _, _ = self._ask_route()
        route(None, req, {'user_id': 7})
        req.goal = 'What were those totals again?'
        result = route(None, req, {'user_id': 7})
        self.assertEqual(result['turn_number'], 2)
        messages = ''.join(m['content'] for m in p.calls[2][1] if m['role'] == 'user')
        self.assertIn('Revenue is 500 and profit is 150.', messages)
        self.assertIn('sample_results', messages)

    def test_new_chat_and_dataset_change_do_not_inherit_history(self):
        p = self.provider(PLAN, STORY, PLAN, STORY, PLAN, STORY)
        route, req, _, _, _ = self._ask_route()
        route(None, req, {'user_id': 7})
        req.conversation_id = 'another-chat'
        self.assertEqual(route(None, req, {'user_id': 7})['turn_number'], 1)
        req.conversation_id = 'new-conversation'
        with sqlite3.connect(self.path) as conn:
            conn.execute('INSERT INTO facts VALUES("C",1,1,1,1)')
        self.assertEqual(route(None, req, {'user_id': 7})['turn_number'], 1)

    def test_returning_to_existing_chat_restores_only_its_context(self):
        self.provider(PLAN, STORY, PLAN, STORY, {'question': 'Total revenue and profit', 'uses_history': False}, PLAN, STORY)
        route, req, _, _, _ = self._ask_route()
        route(None, req, {'user_id': 7})
        req.conversation_id = 'second-chat'
        route(None, req, {'user_id': 7})
        req.conversation_id = 'new-conversation'
        self.assertEqual(route(None, req, {'user_id': 7})['turn_number'], 2)


    def test_followup_resolution_preserves_winning_city_filter(self):
        history = [{'goal': 'Which city has highest revenue?', 'answer': 'B has highest revenue at 300.'}]
        filtered = {'status': 'ready', 'queries': [query("SELECT SUM(profit) AS profit FROM facts WHERE city='B'", ['profit'])]}
        p = self.provider({'question': 'What is total profit for city B?', 'uses_history': True}, filtered,
                          {'answer': 'Profit for B is {{0:0:profit}}.', 'findings': [], 'followups': []})
        result = ask_csv(self.session, 7, 'What is its profit?', history=history)
        self.assertEqual(result['data'], [{'profit': 90}])
        self.assertEqual(result['resolved_question'], 'What is total profit for city B?')
        self.assertEqual(result['llm_trace']['calls'][0]['stage'], 'question_resolution')
        plan_payload = ''.join(m['content'] for m in p.calls[1][1])
        self.assertIn('What is total profit for city B?', plan_payload)

    def test_unclear_reference_asks_instead_of_querying_all_rows(self):
        p = self.provider({'clarification': 'Do you mean A or B?'})
        result = ask_csv(self.session, 7, 'Its profit?', history=[{'answer': 'A and B were compared.'}])
        self.assertEqual(result['status'], 'clarification')
        self.assertEqual(len(p.calls), 1)


if __name__ == '__main__':
    unittest.main()
