"""Known-pattern tests use real SQLite and real statistical calculations."""
import json
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path
from csv_engine.statistics import run_tool, _bh
from csv_engine.profiling import _context
from csv_engine.analyst import ask_csv, dashboard
from test_csv_llm_service import Provider


class StatisticalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'patterns.db'
        self.session = {'db_name': 'facts', 'db_uri': 'sqlite:///' + str(self.path), 'provider_name': 'cloud'}
        rng = random.Random(41)
        with sqlite3.connect(self.path) as conn:
            conn.execute('CREATE TABLE facts(x REAL, y REAL, value REAL, segment TEXT, day TEXT)')
            conn.executemany('INSERT INTO facts VALUES(?,?,?,?,?)', [(i, 2*i+rng.random(), 10 if i<60 else 30,
                'A' if i<60 else 'B', f'2026-09-{i%30+1:02d}') for i in range(120)])
        self.context = _context(self.session)

    def tearDown(self): self.tmp.cleanup()

    def tool(self, name, **args):
        return run_tool(self.session, self.context, {'tool': name, 'arguments': args})

    def test_known_correlation_and_adjustment(self):
        r = self.tool('relationships', columns=['x','y','value'])
        self.assertEqual(len(r['data']), 3)
        pair = r['data'][0]
        self.assertGreater(pair['spearman_r'], .99)
        self.assertGreaterEqual(pair['adjusted_p'], pair['p_value'])
        self.assertEqual(pair['pair_count'], 120)
        self.assertEqual(_bh([.01,.04,.03]), [.03,.04,.04])

    def test_group_difference_effect(self):
        result = self.tool('group_comparison', columns=['value'], group_by='segment')
        a = next(r for r in result['data'] if r['group']=='A')
        self.assertEqual(a['median_difference'], -20)
        self.assertEqual(a['rank_biserial'], -1)
        self.assertEqual(a['group_count'], 60)

    def test_planted_outlier_and_source_row(self):
        with sqlite3.connect(self.path) as c: c.execute('UPDATE facts SET value=10000 WHERE rowid=5')
        r = self.tool('outliers', columns=['value'], group_by='segment')
        self.assertEqual(r['data'][0]['source_row'], 5)
        self.assertEqual(r['data'][0]['value'], 10000)
        self.assertEqual(r['data'][0]['group_median'], 10)

    def test_uploaded_columns_cannot_shadow_internal_row_identity(self):
        with sqlite3.connect(self.path) as c:
            c.execute('ALTER TABLE facts ADD COLUMN rowid TEXT')
            c.execute('ALTER TABLE facts ADD COLUMN __aria_row REAL')
            c.execute('UPDATE facts SET __aria_row=10, rowid="custom"')
            c.execute('UPDATE facts SET __aria_row=10000 WHERE _rowid_=5')
        self.session.pop('csv_dataset_profile', None)
        self.context = _context(self.session)
        result = self.tool('outliers', columns=['__aria_row'])
        self.assertEqual(result['data'][0]['source_row'], 5)
        self.assertEqual(result['data'][0]['value'], 10000)

    def test_constant_groups_do_not_force_anomalies(self):
        r = self.tool('outliers', columns=['value'], group_by='segment')
        self.assertEqual(r['status'], 'no_candidates')
        self.assertEqual(r['data'], [])

    def test_small_sample_and_missing_values(self):
        with sqlite3.connect(self.path) as c:
            c.execute('DELETE FROM facts WHERE rowid>5')
            c.execute('UPDATE facts SET value=NULL WHERE rowid=1')
        r = self.tool('distribution', columns=['value'])
        self.assertEqual(r['data'][0]['valid_count'], 4)
        self.assertEqual(r['data'][0]['excluded_count'], 1)
        self.assertEqual(self.tool('relationships', columns=['x','y'])['status'], 'insufficient_data')
        self.assertEqual(self.tool('outliers', columns=['value'])['status'], 'insufficient_data')

    def test_filters_preserve_entity_and_bind_values(self):
        r = self.tool('distribution', columns=['value'], filters={'segment':'A'})
        self.assertEqual(r['data'][0]['mean'], 10)
        malicious = self.tool('distribution', columns=['value'], filters={'segment': "A' OR 1=1 --"})
        self.assertEqual(malicious['sample_count'], 0)
        with self.assertRaises(ValueError): self.tool('distribution', columns=['x); DROP TABLE facts; --'])

    def test_planted_daily_jump(self):
        with sqlite3.connect(self.path) as c:
            c.execute('DELETE FROM facts')
            c.executemany('INSERT INTO facts VALUES(?,?,?,?,?)', [(i,i, 100 if i<15 else 300,'A',f'2026-09-{i+1:02d}') for i in range(30)])
        r = self.tool('time_changes', columns=['value'], date_column='day', aggregation='sum')
        self.assertEqual(len(r['data']),1)
        self.assertEqual(r['data'][0]['date'],'2026-09-16')
        self.assertEqual(r['data'][0]['change'],200)

    def test_bounded_sample_is_explicit(self):
        with sqlite3.connect(self.path) as c:
            c.executemany('INSERT INTO facts VALUES(?,?,?,?,?)', [(1,2,3,'A','2026-09-01')]*10000)
        r = self.tool('distribution', columns=['value'])
        self.assertTrue(r['sampled'])
        self.assertEqual(r['sample_count'],10000)
        self.assertIn('not a random sample',' '.join(r['limitations']))

    def test_tool_planning_and_grounded_explanation(self):
        tool = {'tool':'group_comparison','title':'Segment differences','reason':'Compare values across segments',
                'arguments':{'columns':['value'],'group_by':'segment'}}
        self.session['provider'] = Provider({'status':'ready','queries':[],'tools':[tool]},
            {'answer':'Group {{0:0:group}} has median {{0:0:median}}.', 'findings':[], 'followups':[]})
        result = ask_csv(self.session, 7, 'Find unusual segment differences')
        self.assertEqual(result['analysis_results'][0]['statistical_evidence']['tool'], 'group_comparison')
        self.assertEqual(result['llm_trace']['statistical_tool_count'], 1)
        self.assertFalse(result['verdict']['verified'])
        self.assertIsNone(result['sql_used'])
        json.dumps(result, allow_nan=False)

    def test_dashboard_retains_discoveries_without_fake_kpis(self):
        tool = {'tool':'distribution','title':'Distribution','reason':'Inspect variation', 'arguments':{'columns':['x']}}
        self.session['provider'] = Provider({'status':'ready','queries':[],'tools':[tool]},
            {'answer':'Median is {{0:0:median}}.', 'findings':[], 'followups':[]})
        result = dashboard(self.session, 7)
        self.assertEqual(result['metrics'], [])
        self.assertEqual(len(result['statistical_insights']), 1)

    def test_compatibility_aliases_preserve_implementation(self):
        import csv_handler
        import csv_orchestrator
        from csv_engine import handler, importer
        import services.csv_llm_service as legacy
        from csv_engine import analyst
        self.assertIs(csv_handler, handler)
        self.assertIs(csv_orchestrator, importer)
        self.assertIs(legacy, analyst)
        source = Path(self.tmp.name)/'test.csv'
        source.write_text('name,amount\nA,12\nB,14\n')
        profile = csv_handler.CsvSchemaExtractor(source).extract()
        self.assertEqual(profile['row_count'], 2)

if __name__ == '__main__': unittest.main()
