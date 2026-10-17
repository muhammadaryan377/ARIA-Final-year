"""Review, adaptive investigation, sampling and durable-chat regression tests."""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from csv_engine.analyst import ask_csv, CsvAnalysisError, _execute
from csv_engine.chat_store import ChatStore
from csv_engine.investigation import investigate
from csv_engine.verification import validate_evidence
from csv_engine.statistics import run_tool
from csv_engine.profiling import _context
from test_csv_llm_service import Provider, query, PLAN, STORY

ACCEPT = {'status':'accepted','reason':'Requested source columns and filter match.', 'checks':['source metrics','entity filter','grain']}
REVISE = {'status':'revise','reason':'Use the stored profit column, not revenue, for profit.', 'checks':['source metrics']}


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / 'source.db'
        with sqlite3.connect(self.db) as c:
            c.execute('CREATE TABLE facts(city TEXT, revenue REAL, profit REAL)')
            c.executemany('INSERT INTO facts VALUES(?,?,?)', [('A',200,60),('B',300,90)])
        self.session = {'db_uri':'sqlite:///'+str(self.db),'db_name':'facts','provider_name':'cloud'}

    def ask(self, *responses):
        self.session['provider'] = Provider(*responses)
        return ask_csv(self.session, 1, 'Total revenue and profit?')

    def test_real_review_stage_and_reproducibility(self):
        result = self.ask(PLAN, ACCEPT, STORY)
        self.assertEqual([c['stage'] for c in result['llm_trace']['calls']], ['question_plan','semantic_review','explanation'])
        self.assertEqual(result['llm_trace']['verification_query_count'],1)
        self.assertFalse(result['validation']['verified'])
        self.assertEqual(result['data'][0]['total_profit'],150)

    def test_wrong_metric_replanned_once_before_answer(self):
        wrong = {'status':'ready', 'queries':[query('SELECT SUM(revenue) AS total_revenue, SUM(revenue) AS total_profit FROM facts')]}
        result = self.ask(wrong, REVISE, PLAN, ACCEPT, STORY)
        self.assertEqual(result['data'][0]['total_profit'],150)
        self.assertEqual(result['llm_trace']['provider_invocation_count'],5)
        calls = self.session['provider'].calls
        self.assertIn('review_correction', ''.join(m['content'] for m in calls[2][1]))

    def test_repeated_review_rejection_cannot_return_success(self):
        with self.assertRaises(CsvAnalysisError): self.ask(PLAN, REVISE, PLAN, REVISE)
        self.assertEqual(len(self.session['provider'].calls),4)

    def test_invalid_review_receives_bounded_repair(self):
        r = self.ask(PLAN, {'status':'accepted'}, ACCEPT, STORY)
        review = [c for c in r['llm_trace']['calls'] if c['stage']=='semantic_review']
        self.assertEqual([c['status'] for c in review], ['invalid_response','success'])

    def test_changed_source_fails_reproducibility(self):
        calls = 0
        def changed(*args):
            nonlocal calls
            columns, rows = _execute(*args)
            calls += 1
            if calls > 1: rows[0]['total_profit'] += 1
            return columns, rows
        with patch('csv_engine.analyst._execute', side_effect=changed):
            with self.assertRaisesRegex(CsvAnalysisError, 'changed during validation'):
                self.ask(PLAN, ACCEPT)

    def test_second_investigation_round_uses_previous_results(self):
        initial = copy.deepcopy(PLAN); initial['needs_investigation'] = True
        extra1 = {'status':'ready','queries':[query('SELECT COUNT(*) AS n FROM facts',['n'])], 'needs_investigation':True}
        extra2 = {'status':'ready','queries':[query('SELECT AVG(profit) AS average_profit FROM facts',['average_profit'])]}
        result = self.ask(initial, extra1, extra2, ACCEPT, STORY)
        self.assertEqual(result['llm_trace']['investigation']['rounds'],2)
        self.assertEqual(len(result['analysis_results']),3)
        self.assertTrue(result['analysis_plan']['investigation_complete'])

    def test_adaptive_loop_stops_and_labels_budget(self):
        run = SimpleNamespace(deadline=time.monotonic()+100, trace={})
        plan = {'queries':[{}]*8,'needs_investigation':True}
        result = investigate(run,plan,lambda _: self.fail('Planner must not run'))
        self.assertFalse(result['investigation_complete'])
        self.assertEqual(run.trace['investigation']['stop_reason'],'result_budget')

    def test_no_additional_evidence_stops_loop(self):
        run = SimpleNamespace(deadline=time.monotonic()+100, trace={})
        plan = {'queries':[{}],'needs_investigation':True}
        investigate(run,plan,lambda _: {'status':'ready','queries':[], 'needs_investigation':True})
        self.assertEqual(run.trace['investigation']['rounds'],1)
        self.assertTrue(plan['investigation_complete'])

    def test_nonfinite_evidence_is_rejected(self):
        with self.assertRaises(CsvAnalysisError):
            validate_evidence({'queries':[{'columns':['x'],'data':[{'x':float('nan')}]}]})

    def test_reservoir_includes_late_population_and_repeats(self):
        with sqlite3.connect(self.db) as c:
            c.execute('DELETE FROM facts')
            c.executemany('INSERT INTO facts VALUES(?,?,?)', [('A',0,0)]*10000+[('B',100,100)]*10000)
        context = _context(self.session)
        call = {'tool':'distribution','arguments':{'columns':['profit']}}
        first = run_tool(self.session,context,call)
        second = run_tool(self.session,context,call)
        self.assertEqual(first,second)
        self.assertTrue(45 < first['data'][0]['mean'] < 55)
        self.assertEqual(first['matched_count'],20000)

    def test_sampled_time_analysis_is_not_misrepresented_as_full(self):
        with sqlite3.connect(self.db) as c:
            c.executemany('INSERT INTO facts VALUES(?,?,?)', [('2026-09-01',1,1)]*10001)
        with self.assertRaisesRegex(ValueError,'complete records'):
            run_tool(self.session,_context(self.session), {'tool':'time_changes','arguments':{'columns':['profit'],'date_column':'city','aggregation':'sum'}})

    def test_chat_survives_new_store_instance_and_scopes_user_dataset(self):
        path = Path(self.tmp.name)/'chat.db'
        ChatStore(path).save(1,'dataset-a','chat',context={'goal':'profit'},display={'q':'profit','r':{'answer':'test'}})
        store = ChatStore(path)
        self.assertEqual(store.load(1,'dataset-a','chat')['history'][0]['goal'],'profit')
        self.assertIsNone(store.load(2,'dataset-a','chat'))
        self.assertIsNone(store.load(1,'dataset-b','chat'))
        self.assertFalse(store.delete(2,'dataset-a','chat'))
        self.assertTrue(store.delete(1,'dataset-a','chat'))

    def test_new_empty_chat_is_restored_without_previous_context(self):
        store=ChatStore(Path(self.tmp.name)/'chat.db')
        store.save(1,'dataset','old',context={'goal':'old'},display={'q':'old'})
        store.save(1,'dataset','new',new=True)
        self.assertEqual(store.load(1,'dataset')['history'],[])
        self.assertEqual(store.load(1,'dataset')['conversation_id'],'new')

    def test_chat_retention_is_bounded(self):
        store=ChatStore(Path(self.tmp.name)/'chat.db')
        for i in range(25): store.save(1,'dataset',str(i),new=True)
        self.assertIsNone(store.load(1,'dataset','0'))
        for i in range(12): store.save(1,'dataset','24',context={'goal':str(i)},display={'q':str(i)})
        self.assertEqual(len(store.load(1,'dataset','24')['turns']),10)
        self.assertEqual(store.load(1,'dataset','24')['history'][0]['goal'],'2')

class ConversationRouteTests(unittest.TestCase):
    def setUp(self):
        import ast
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ChatStore(Path(self.tmp.name)/'chats.db')
        source = Path(__file__).resolve().parents[1]/'api/query_routes.py'
        wanted = {'restore_csv_conversation','new_csv_conversation','delete_csv_conversation'}
        functions = [n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in wanted]
        # Preserve body logic while avoiding unrelated optional ML imports.
        for fn in functions:
            fn.decorator_list = []
            fn.args.defaults = []
            for arg in fn.args.args: arg.annotation = None
        self.session = {'source_type':'semi-structured','file_type':'csv','db_uri':'sqlite:///fixture'}
        self.ns = {'get_session':lambda uid:self.session,
                   '_csv_chat_store':lambda session:(self.store,'dataset'),
                   'save_session':lambda *args:None, 'HTTPException':RuntimeError}
        exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),str(source),'exec'),self.ns)

    def test_restore_never_returns_another_users_chat(self):
        self.store.save(1,'dataset','private',context={'goal':'secret'},display={'q':'secret','r':{}})
        restored = self.ns['restore_csv_conversation'](None,'private',{'user_id':2})
        self.assertIsNone(restored['conversation'])
        own = self.ns['restore_csv_conversation'](None,'private',{'user_id':1})['conversation']
        self.assertEqual(own['turns'][0]['q'],'secret')
        self.assertNotIn('history',own)

    def test_new_conversation_has_no_old_turns(self):
        self.store.save(1,'dataset','old',context={'goal':'old'},display={'q':'old'})
        created = self.ns['new_csv_conversation'](None,{'user_id':1})
        latest = self.ns['restore_csv_conversation'](None,None,{'user_id':1})['conversation']
        self.assertEqual(created['conversation_id'],latest['conversation_id'])
        self.assertEqual(latest['turns'],[])

    def test_delete_is_scoped_and_clears_memory(self):
        self.store.save(1,'dataset','chat',new=True)
        self.session['csv_conversations'] = {'chat': {'history':[]}}
        self.assertFalse(self.ns['delete_csv_conversation']('chat',{'user_id':2})['deleted'])
        self.assertTrue(self.ns['delete_csv_conversation']('chat',{'user_id':1})['deleted'])
        self.assertNotIn('chat',self.session['csv_conversations'])

if __name__ == '__main__': unittest.main()
