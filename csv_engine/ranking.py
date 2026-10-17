"""Transparent triage priority, never a confidence or significance score."""
def rank_findings(queries):
    candidates = []
    for index, query in enumerate(queries):
        ev = query.get('statistical_evidence')
        if not ev or ev['status'] != 'complete' or not query['data']:
            continue
        tool = ev['tool']
        column = {'relationships': 'spearman_r', 'group_comparison': 'rank_biserial',
                  'outliers': 'modified_z', 'time_changes': 'change_pct'}.get(tool)
        if not column:
            continue
        values = [abs(row[column]) for row in query['data'] if isinstance(row.get(column), (int, float))]
        # Constant-baseline departures have no valid modified-z score.
        strength = max(values) if values else None
        candidates.append({'query_index': index, 'title': query['title'], 'tool': tool,
                           'effect_measure': column, 'largest_absolute_effect': strength,
                           'sample_count': ev['sample_count'], 'sampled': ev['sampled'],
                           'reason': query['reason'],
                           'note': 'Sorted within each method by observed effect, not business importance, confidence or causality.'})
    candidates.sort(key=lambda r: (r['tool'], -(r['largest_absolute_effect'] or 0)))
    return candidates
