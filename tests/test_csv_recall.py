import unittest
from csv_engine.conversation import _resolve_question, conversation_turn
from csv_engine.analyst import ask_csv
from unittest.mock import patch

class Run:
    def __init__(self, result): self.result = result
    def json_call(self, stage, role, prompt, payload, validate):
        self.payload = payload
        return validate(self.result)

class RecallTests(unittest.TestCase):
    history = [{'goal':'Can we separate city effects?', 'answer':'City and product are perfectly confounded. Their effects cannot be separated.'}]
    def test_exact_recall_preserves_prior_conclusion(self):
        result = _resolve_question(Run({'mode':'recall','turn_index':0,'quote':self.history[0]['answer']}), 'What limitation did we identify?', self.history)
        self.assertEqual(result['answer'], self.history[0]['answer'])

    def test_invented_or_changed_quote_rejected(self):
        for index, quote in [(0,'The main limitation is sample size.'),(1,self.history[0]['answer']),(True,self.history[0]['answer'])]:
            with self.subTest(index=index), self.assertRaises(ValueError):
                _resolve_question(Run({'mode':'recall','turn_index':index,'quote':quote}), 'What limitation?', self.history)

    def test_recall_never_runs_sql_or_narration(self):
        from test_csv_llm_service import Provider
        provider = Provider({'mode':'recall','turn_index':0,'quote':self.history[0]['answer']})
        session = {'provider':provider}
        with patch('csv_engine.analyst._context', return_value={}), patch('csv_engine.analyst._plan') as planner:
            result = ask_csv(session,7,'What was the main limitation?',self.history)
        planner.assert_not_called()
        self.assertEqual(len(provider.calls),1)
        self.assertEqual(result['analysis_source'],'csv_chat_history')
        self.assertEqual(result['data'],[])
        self.assertFalse(result['result_valid'])

    def test_empty_chat_cannot_recall_other_conversations(self):
        result = _resolve_question(Run(None),'Which city earlier?',[])
        self.assertFalse(result['uses_history'])
        self.assertNotIn('answer',result)

    def test_long_answer_keeps_late_limitation(self):
        answer = 'Context. ' * 200 + self.history[0]['answer']
        saved = conversation_turn('Investigation', {'answer':answer})
        self.assertIn(self.history[0]['answer'],saved['answer'])

if __name__ == '__main__': unittest.main()
