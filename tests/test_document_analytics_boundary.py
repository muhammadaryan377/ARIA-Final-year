import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]

def function(path, name, namespace):
    tree = ast.parse((ROOT/path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    node.args.defaults = []
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), namespace)
    return namespace[name]

@pytest.mark.parametrize('identity', [{'source_type':'document'}, {'db_type':'document'}, {'db_uri':'document://owned-id'}])
def test_document_never_reads_old_csv_or_opens_sql(identity):
    # No Path/SQL helpers supplied: touching either would fail this regression.
    assert function('services/intelligence_service.py','_load_schema_mapping',{})(identity) == {}
    result = function('services/intelligence_service.py','run_explore',{})(identity, {'metric':'revenue'})
    assert result['ok'] is False and result['reason'] == 'document_source'

def test_ask_uses_owned_document_and_preserves_conversation(monkeypatch):
    spec = importlib.util.spec_from_file_location('insight_rag.analytics_adapter',ROOT/'insight_rag/analytics_adapter.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    monkeypatch.setitem(sys.modules,'insight_rag.analytics_adapter',adapter)
    calls=[]
    result={'answer':'Evidence [S1].','sources':[{'source_id':'S1','page':2}],
            'conversation_id':'document-chat-1','evidence_status':'supported'}
    class RAG:
        def chat(self, question, **kwargs):
            calls.append((question,kwargs)); return result
    fake=ModuleType('api.insight_rag_routes');fake._capability=lambda uid:RAG()
    monkeypatch.setitem(sys.modules,'api.insight_rag_routes',fake)
    session={'source_type':'document','document_id':'owned-id','document_conversation_id':'document-chat-1'}
    saved=[]
    ask=function('api/query_routes.py','ask',dict(get_session=lambda uid:session,
        require_data=lambda r:object(),save_session=lambda *args:saved.append(args)))
    reply=ask(None,SimpleNamespace(goal='Summarize',conversation_id=None),{'user_id':1})
    assert calls[0][1] == {'conversation_id':'document-chat-1','document_ids':['owned-id']}
    assert reply['analysis_source']=='document_rag' and reply['sql_used'] is None
    assert reply['evidence']==result['sources'] and saved
