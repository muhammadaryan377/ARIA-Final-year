"""Small synthetic acceptance benchmark. Live mode makes billable provider calls.

Offline mode verifies fixture ingestion and SQL oracles, NOT LLM accuracy.
"""
import argparse
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import time
from .importer import load_csv_to_sqlite

CASES = [
    {'id':'separate_totals','file':'sales.csv','question':'Total revenue and total profit across all records. Return aliases total_revenue and total_profit.',
     'sql':'SELECT SUM(revenue) AS total_revenue,SUM(profit) AS total_profit FROM facts',
     'expected':[{'total_revenue':285000,'total_profit':90000}]},
    {'id':'city_filter','file':'sales.csv','question':'Total profit for Islamabad only. Return alias total_profit.',
     'sql':"SELECT SUM(profit) AS total_profit FROM facts WHERE city='Islamabad'",'expected':[{'total_profit':45000}]},
    {'id':'weighted_margin','file':'sales.csv','question':'Overall profit margin as a percentage: total profit divided by total revenue times one hundred. Return alias margin_pct.',
     'sql':'SELECT 100.0*SUM(profit)/NULLIF(SUM(revenue),0) AS margin_pct FROM facts','expected':[{'margin_pct':100*90000/285000}]},
    {'id':'winner','file':'sales.csv','question':'Which city has highest total revenue? Return city and total_revenue.',
     'sql':'SELECT city,SUM(revenue) AS total_revenue FROM facts GROUP BY city ORDER BY total_revenue DESC LIMIT 1',
     'expected':[{'city':'Islamabad','total_revenue':150000}]},
    {'id':'winner_followup','file':'sales.csv','question':'What is its total profit? Return alias total_profit.', 'previous':'winner',
     'sql':"SELECT SUM(profit) AS total_profit FROM facts WHERE city='Islamabad'",'expected':[{'total_profit':45000}]},
    {'id':'null_average','file':'sensors.csv','question':'Average of nonmissing reading values. Return alias average_reading.',
     'sql':'SELECT AVG(reading) AS average_reading FROM facts','expected':[{'average_reading':20}]},
    {'id':'zero_denominator','file':'sensors.csv','question':'Sum of readings divided by sum of capacity; return NULL if capacity total is zero. Return alias ratio.',
     'sql':'SELECT SUM(reading)/NULLIF(SUM(capacity),0) AS ratio FROM facts','expected':[{'ratio':None}]},
    {'id':'categorical_counts','file':'categories.csv','question':'Count records by label, ordered by label. Return label and records.',
     'sql':'SELECT label,COUNT(*) AS records FROM facts GROUP BY label ORDER BY label','expected':[{'label':'blue','records':1},{'label':'red','records':2}]},
    {'id':'absent_field','file':'sales.csv','question':'What is the average customer age?', 'clarification':True},
]


def matches(actual, expected):
    if not isinstance(actual, list) or len(actual) != len(expected): return False
    def row_matches(a,b):
        if set(a) != set(b): return False
        for key, value in b.items():
            got = a[key]
            if isinstance(value,(int,float)):
                if not isinstance(got,(int,float)) or not math.isclose(got,value,rel_tol=1e-7,abs_tol=1e-7): return False
            elif got != value: return False
        return True
    remaining = list(actual)
    for target in expected:
        index = next((i for i,row in enumerate(remaining) if row_matches(row,target)), None)
        if index is None: return False
        remaining.pop(index)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Call selected provider; uses its environment API key and existing backend consent')
    parser.add_argument('--provider', default='deepseek')
    parser.add_argument('--model', help='Exact model identifier already supported by your selected provider')
    parser.add_argument('--output', default='data/csv_evaluation_report.json')
    args = parser.parse_args()
    provider = None
    if args.live:
        from llm_provider import create_provider
        provider = create_provider(provider=args.provider, models={r:args.model for r in ('semi','story')} if args.model else None)
    fixture_dir = Path(__file__).resolve().parents[1]/'tests/fixtures/csv_eval'
    report = {'mode':'live_model' if args.live else 'offline_sql_oracles',
              'scope':'Synthetic acceptance cases, not general accuracy or a real-world benchmark.', 'cases':[]}
    histories = {}
    from .analyst import ask_csv
    from .conversation import conversation_turn
    with tempfile.TemporaryDirectory() as tmp:
        sessions = {}
        for filename in sorted({c['file'] for c in CASES}):
            uri, table, count = load_csv_to_sqlite(fixture_dir/filename, Path(tmp)/(filename+'.db'), table_name='facts')
            sessions[filename] = {'db_uri':uri, 'db_name':table, 'provider_name':args.provider, 'provider':provider,
                                  'source_type':'semi-structured','file_type':'csv'}
        for case in CASES:
            started = time.monotonic()
            entry = {'id':case['id']}
            try:
                if args.live:
                    if case.get('previous') and case['previous'] not in histories:
                        entry.update(status='blocked', reason='Required preceding case failed')
                    else:
                        result = ask_csv(sessions[case['file']],0,case['question'],history=histories.get(case.get('previous'),[]))
                        valid = result['status']=='clarification' if case.get('clarification') else result['status']=='success' and any(matches(q['data'],case['expected']) for q in result.get('analysis_results',[]))
                        entry.update(status='passed' if valid else 'failed', result=result)
                        if valid: histories[case['id']] = [conversation_turn(case['question'], result)]
                elif case.get('clarification'):
                    entry.update(status='not_run', reason='Requires a live model to evaluate clarification')
                else:
                    path = sessions[case['file']]['db_uri'][len('sqlite:///'):]
                    with sqlite3.connect(path) as conn:
                        conn.row_factory = sqlite3.Row
                        rows = [dict(r) for r in conn.execute(case['sql'])]
                    entry.update(status='passed' if matches(rows,case['expected']) else 'failed', actual=rows)
            except Exception as exc:
                entry.update(status='error', error_type=type(exc).__name__)
            entry['duration_ms'] = round((time.monotonic()-started)*1000)
            report['cases'].append(entry)
            print(case['id']+': '+entry['status'])
    report['counts'] = {status:sum(c['status']==status for c in report['cases']) for status in ('passed','failed','error','blocked','not_run')}
    output = Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print('Report: '+str(output))
    return int(any(report['counts'][s] for s in ('failed','error','blocked')))

if __name__=='__main__': raise SystemExit(main())
