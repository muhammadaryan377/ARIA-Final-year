"""Dependency-light regressions for provider isolation and upload wiring."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('document_providers', ROOT / 'insight_rag/providers.py')
providers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(providers)

@pytest.mark.parametrize('name', ['local', 'cloud', 'deepseek', 'openai', 'anthropic', 'grok'])
def test_selected_model_is_used_for_every_rag_role(name):
    provider = SimpleNamespace(provider=name, models={'semi': 'selected-model'})
    providers.configure_rag_models(provider)
    assert all(provider.models[r] == 'selected-model' for r in providers.RAG_ROLES)

def test_explicit_role_override_survives():
    provider = SimpleNamespace(models={'semi': 'selected', 'rag_verify': 'reviewer'})
    providers.configure_rag_models(provider)
    assert provider.models['rag_verify'] == 'reviewer'
    assert provider.models['rag'] == 'selected'

def test_missing_model_fails_actionably():
    with pytest.raises(ValueError, match='Select a model'):
        providers.configure_rag_models(SimpleNamespace(models={}))

def test_upload_routes_share_ingestion_and_do_not_call_ocr():
    for path, function in [('api/connect_routes.py', 'connect_pdf'), ('api/insight_rag_routes.py', 'upload_pdf')]:
        tree = ast.parse((ROOT/path).read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        calls = [ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)]
        assert any('PDFIngestion' in c for c in calls)
        assert 'create_document_processor' not in calls
        assert not any(c.endswith('process_document') for c in calls)
