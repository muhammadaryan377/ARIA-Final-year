"""Translate statistical tool output into the shared evidence/result contract."""
from .statistics import run_tool, TOOL_CATALOG
from .storage import _text


def execute_discovery(session, context, call):
    title = _text(call.get('title'), 200)
    reason = _text(call.get('reason'), 1000)
    result = run_tool(session, context, call)
    measures = [c for c in result['columns'] if any(isinstance(r[c], (int,float)) for r in result['data'])]
    dimensions = [c for c in result['columns'] if c not in measures]
    return {'title': title, 'reason': reason, 'sql': None, 'chart': 'table',
            'measures': measures, 'dimensions': dimensions, 'columns': result['columns'],
            'data': result['data'], 'statistical_evidence': {k: v for k,v in result.items() if k not in ('data', 'columns')}}
