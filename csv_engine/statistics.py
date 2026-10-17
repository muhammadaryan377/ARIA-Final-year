"""Bounded, reproducible statistical tools. No model-generated Python execution.

All p-values are exploratory, not proof of causes. Multiple-comparison adjustment
covers the comparisons within one tool invocation, not all user investigations.
"""
from collections import defaultdict
from datetime import date
from itertools import combinations
import math
import statistics as st
import time

from .storage import _connection, _quote

MAX_SAMPLE = 10000
_ROW_KEY = object()
TOOL_CATALOG = {
    'distribution': {'columns': 'one to six numeric source columns'},
    'outliers': {'columns': 'one numeric source column', 'group_by': 'optional category column; compare within groups'},
    'relationships': {'columns': 'two to six numeric source columns; never identifiers'},
    'group_comparison': {'columns': 'one numeric source column', 'group_by': 'category column'},
    'time_changes': {'columns': 'one numeric source column', 'date_column': 'ISO date column', 'aggregation': 'sum or mean'},
}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _bh(values):
    """Benjamini-Hochberg adjusted p-values, preserving order."""
    result = [None] * len(values)
    ordered = sorted((p, i) for i, p in enumerate(values) if p is not None)
    running = 1.0
    for rank in range(len(ordered), 0, -1):
        p, i = ordered[rank - 1]
        running = min(running, p * len(ordered) / rank)
        result[i] = min(1.0, running)
    return result


def _quantiles(values):
    if len(values) < 2:
        return [values[0]] * 3
    return st.quantiles(values, n=4, method='inclusive')


def _sample(session, context, args):
    known = {c['name'] for c in context['columns']}
    cols = args.get('columns')
    if not isinstance(cols, list) or not 1 <= len(cols) <= 6 or any(not isinstance(c, str) or c not in known for c in cols) or len(set(cols)) != len(cols):
        raise ValueError('Statistical tools require one to six distinct existing column names')
    extra = [args.get('group_by'), args.get('date_column')]
    selected = list(dict.fromkeys(cols + [c for c in extra if c is not None]))
    if any(c not in known for c in selected):
        raise ValueError('Unknown group or date column')
    filters = args.get('filters', {})
    if not isinstance(filters, dict) or len(filters) > 6:
        raise ValueError('Tool filters must be up to six exact column/value pairs')
    clauses, params = [], []
    for col, value in filters.items():
        if col not in known or not (value is None or isinstance(value, (str, int, float))) or isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Invalid statistical tool filter')
        clauses.append(f'{_quote(col)} IS ?')
        params.append(value)
    where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
    lowered = {c.lower() for c in known}
    row_id = next((c for c in ('rowid', '_rowid_', 'oid') if c not in lowered), None)
    if row_id is None:
        raise ValueError('Source columns shadow every SQLite row identifier; rename rowid, _rowid_ or oid')
    alias = '__aria_row'
    while alias.lower() in lowered:
        alias += '_'
    sql = f'SELECT {row_id} AS {_quote(alias)}, ' + ', '.join(_quote(c) for c in selected) + f' FROM {_quote(context["table"])}{where} ORDER BY {row_id} LIMIT {MAX_SAMPLE + 1}'
    with _connection(session) as conn:
        deadline = time.monotonic() + 3
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        records = [dict(r) for r in conn.execute(sql, params).fetchall()]
    for record in records:
        record[_ROW_KEY] = record.pop(alias)
    truncated = len(records) > MAX_SAMPLE
    return records[:MAX_SAMPLE], truncated, sql, params


def run_tool(session, context, call):
    if not isinstance(call, dict) or call.get('tool') not in TOOL_CATALOG:
        raise ValueError('Unknown statistical tool')
    name = call['tool']
    args = call.get('arguments', {})
    if not isinstance(args, dict):
        raise ValueError('Tool arguments must be an object')
    rows, truncated, sql, params = _sample(session, context, args)
    cols = args['columns']
    limits = ['Exploratory evidence; association or an unusual value does not establish cause, error or fraud.']
    if truncated:
        limits.append('Uses the first 10000 matching rows in import order, not a random sample. Findings may not generalize to the full dataset.')
    if len(rows) < 20:
        limits.append('Small dataset: estimates and anomaly labels are unstable.')
    output = []
    status = 'complete'
    method = ''
    if name == 'distribution':
        method = 'Numeric-only counts, inclusive quartiles, sample standard deviation; no imputation.'
        for col in cols:
            values = [r[col] for r in rows if _number(r[col])]
            q1, median, q3 = _quantiles(values) if values else (None, None, None)
            output.append({'column': col, 'valid_count': len(values), 'excluded_count': len(rows)-len(values),
                'minimum': min(values) if values else None, 'q1': q1, 'median': median, 'q3': q3,
                'maximum': max(values) if values else None, 'mean': st.mean(values) if values else None,
                'stddev': st.stdev(values) if len(values) > 1 else None})
    elif name == 'outliers':
        if len(cols) != 1:
            raise ValueError('Outliers requires one numeric column')
        col = cols[0]; group = args.get('group_by')
        buckets = defaultdict(list)
        for r in rows:
            if _number(r[col]):
                buckets[r[group] if group else 'all records'].append((r[_ROW_KEY], r[col]))
        if len(buckets) > 20:
            raise ValueError('Too many anomaly groups; choose at most twenty groups using filters')
        method = 'Per-group median absolute deviation, modified z-score > 3.5. If MAD is zero use 1.5 IQR fences; if both are zero flag departures from the constant baseline. Minimum eight valid rows per group.'
        excluded = 0
        for label, items in buckets.items():
            if len(items) < 8:
                excluded += 1; continue
            values = [v for _, v in items]; med = st.median(values)
            mad = st.median(abs(v-med) for v in values)
            q1, _, q3 = _quantiles(values); iqr = q3-q1
            for row_id, value in items:
                score = abs(0.67448975 * (value-med)/mad) if mad else None
                flagged = score > 3.5 if score is not None else (value < q1-1.5*iqr or value > q3+1.5*iqr)
                if flagged:
                    output.append({'source_row': row_id, 'group': label, 'value': value, 'group_median': med,
                                   'group_count': len(values), 'modified_z': score, 'absolute_deviation': abs(value-med)})
        output.sort(key=lambda r: r['absolute_deviation'], reverse=True)
        total = len(output); output = output[:50]
        limits.append(f'{total} unusual rows flagged; at most fifty displayed. {excluded} small groups excluded. Source row is SQLite import rowid, not a physical CSV line number.')
        if not output: status = 'no_candidates' if any(len(items) >= 8 for items in buckets.values()) else 'insufficient_data'
    elif name == 'relationships':
        if len(cols) < 2: raise ValueError('Relationships requires at least two numeric columns')
        from scipy.stats import spearmanr
        method = 'Pairwise-complete Spearman rank correlation, minimum twenty pairs; BH adjustment across all tested pairs in this invocation.'
        for a, b in combinations(cols, 2):
            pairs = [(r[a], r[b]) for r in rows if _number(r[a]) and _number(r[b])]
            if len(pairs) < 20 or len({v[0] for v in pairs}) < 2 or len({v[1] for v in pairs}) < 2:
                continue
            rho, p = spearmanr([v[0] for v in pairs], [v[1] for v in pairs])
            output.append({'column_a': a, 'column_b': b, 'pair_count': len(pairs), 'excluded_count': len(rows)-len(pairs), 'spearman_r': float(rho), 'p_value': float(p)})
        for r, p in zip(output, _bh([r['p_value'] for r in output])): r['adjusted_p'] = p
        limits.append('P-values are approximate and assume independent observations. Formula-linked fields, trends, repeated subjects and confounding can create misleading associations. No adjustment across different tool calls.')
        if not output: status = 'insufficient_data'
    elif name == 'group_comparison':
        if len(cols) != 1 or not args.get('group_by'): raise ValueError('Group comparison requires one numeric column and group_by')
        from scipy.stats import mannwhitneyu
        col = cols[0]; group = args['group_by']; buckets = defaultdict(list)
        for r in rows:
            if r[group] is not None and _number(r[col]): buckets[r[group]].append(r[col])
        if len(buckets) > 20: raise ValueError('Group comparison supports up to twenty groups; choose a lower-cardinality column')
        method = 'Each group versus remaining records: mean/median difference, Mann-Whitney U, rank-biserial effect; minimum eight per side; BH adjustment within this invocation.'
        for label, values in buckets.items():
            others = [v for k, items in buckets.items() if k != label for v in items]
            if len(values) < 8 or len(others) < 8: continue
            u, p = mannwhitneyu(values, others, alternative='two-sided', method='asymptotic')
            output.append({'group': label, 'group_count': len(values), 'rest_count': len(others),
                'mean': st.mean(values), 'median': st.median(values), 'rest_median': st.median(others),
                'median_difference': st.median(values)-st.median(others),
                'rank_biserial': 2*float(u)/(len(values)*len(others))-1, 'p_value': float(p)})
        for r, p in zip(output, _bh([r['p_value'] for r in output])): r['adjusted_p'] = p
        limits.append('Exploratory dependent comparisons. Differences in distributions do not necessarily imply a median shift. Check group sizes, repeated observations and confounding; no causal claim.')
        if not output: status = 'insufficient_data'
    else:
        if len(cols) != 1 or not args.get('date_column') or args.get('aggregation') not in ('sum','mean'):
            raise ValueError('Time changes requires one measure, date_column, and explicit sum or mean aggregation')
        col = cols[0]; dt = args['date_column']; buckets = defaultdict(list); excluded = 0
        for r in rows:
            try:
                day = date.fromisoformat(str(r[dt])[:10])
                if not _number(r[col]): raise ValueError('not numeric')
                buckets[day].append(r[col])
            except (ValueError, TypeError): excluded += 1
        dates = sorted(buckets)
        method = 'Observed daily sum/mean; robust outlier detection on adjacent-day differences; minimum fourteen observed days. Missing days are not filled with zero.'
        if len(dates) < 14:
            status = 'insufficient_data'
        else:
            agg = sum if args['aggregation'] == 'sum' else st.mean
            values = [agg(buckets[d]) for d in dates]
            changes = [(i,values[i]-values[i-1]) for i in range(1,len(dates)) if (dates[i]-dates[i-1]).days == 1]
            if len(changes) < 10:
                status = 'insufficient_data'
            else:
                med = st.median(v for _,v in changes); mad = st.median(abs(v-med) for _,v in changes)
                for i, delta in changes:
                    flagged = abs(0.67448975*(delta-med)/mad) > 3.5 if mad else delta != med
                    if flagged:
                        output.append({'date': dates[i].isoformat(), 'previous_date': dates[i-1].isoformat(),
                            'value': values[i], 'previous_value': values[i-1], 'change': delta,
                            'change_pct': 100*delta/abs(values[i-1]) if values[i-1] else None,
                            'day_records': len(buckets[dates[i]])})
                if not output: status = 'no_candidates'
        output = output[:50]
        limits.append(f'{excluded} invalid-date/non-numeric records excluded. Seasonal effects and incomplete periods are not modeled; spikes are investigation candidates, not proven structural changes.')
    # Reject rather than serialize infinities or fabricated statistics.
    if any(isinstance(v,float) and not math.isfinite(v) for r in output for v in r.values()):
        raise ValueError('Statistical tool produced non-finite output')
    return {'tool': name, 'arguments': args, 'status': status, 'method': method, 'limitations': limits,
            'sample_count': len(rows), 'sampled': truncated, 'sample_sql': sql, 'sql_parameters': params,
            'data': output, 'columns': list(output[0]) if output else []}
