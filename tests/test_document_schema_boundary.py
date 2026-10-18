"""Execute route functions with isolated dependencies and real filesystem reads."""
import ast
import json
from pathlib import Path
import pytest

class HTTPError(Exception):
    def __init__(self, status_code, detail):
        self.status_code, self.detail = status_code, detail

def routes(session):
    path = Path(__file__).resolve().parents[1] / 'api/schema_routes.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_extract_schema_impl', 'get_schema')]
    for n in nodes:
        n.decorator_list = []
        n.args.defaults = []
        for arg in n.args.args:
            arg.annotation = None
        n.returns = None
    namespace = dict(get_session=lambda uid: session, require_data=lambda req: None,
                     Path=Path, json=json, HTTPException=HTTPError)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace

@pytest.mark.parametrize('identity', [{'source_type': 'document'}, {'db_type': 'document'}])
def test_pdf_does_not_read_stale_schema(tmp_path, identity):
    stale = tmp_path/'stale.json'
    stale.write_text('{"tables":{"old_csv":{}}}')
    session = {'source_type': 'semi-structured', 'schema_path': stale, **identity}
    funcs = routes(session)
    for result in (funcs['get_schema'](None, {'user_id': 1}), funcs['_extract_schema_impl'](None, {'user_id': 1}, 'document')):
        assert result['ok'] and result['applicable'] is False
        assert result['schema']['tables'] == {}
    stale.unlink()
    assert funcs['get_schema'](None, {'user_id': 1})['applicable'] is False

def test_missing_csv_schema_has_actionable_error(tmp_path):
    funcs = routes({'source_type': 'semi-structured', 'schema_path': tmp_path/'missing.json'})
    with pytest.raises(HTTPError) as error:
        funcs['_extract_schema_impl'](None, {'user_id': 1}, 'sqlite')
    assert error.value.status_code == 409
