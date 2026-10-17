"""Bounded adaptive investigation shared by Ask and Dashboard."""
import time
from .constants import MAX_QUERIES


def investigate(run, plan, planner):
    rounds = 0
    stop = 'sufficient_evidence'
    while plan.get('needs_investigation'):
        if rounds >= 3:
            stop = 'round_budget'; break
        if len(plan['queries']) >= MAX_QUERIES:
            stop = 'result_budget'; break
        if run.deadline - time.monotonic() < 25:
            stop = 'time_reserved_for_review_and_answer'; break
        extra = planner(plan)
        rounds += 1
        if extra['status'] != 'ready':
            plan.setdefault('understanding', {}).setdefault('limitations', []).append(extra['clarification'])
            stop = 'clarification_needed'; break
        if not extra['queries']:
            stop = 'no_additional_evidence'; break
        plan['queries'].extend(extra['queries'])
        for key in ('assumptions', 'limitations'):
            target = plan.setdefault('understanding', {}).setdefault(key, [])
            target.extend(v for v in extra.get('understanding', {}).get(key, []) if v not in target)
        plan['needs_investigation'] = extra.get('needs_investigation', False)
    complete = not plan.get('needs_investigation') or stop == 'no_additional_evidence'
    plan['investigation_complete'] = complete
    plan['needs_investigation'] = False
    if not complete:
        plan.setdefault('understanding', {}).setdefault('limitations', []).append(
            'Further investigation was requested but stopped: ' + stop.replace('_', ' ') + '.')
    run.trace['investigation'] = {'rounds': rounds, 'stop_reason': stop, 'complete': complete}
    return plan
