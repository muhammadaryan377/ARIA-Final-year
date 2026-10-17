"""Regression coverage for the 2026-10-17 Aryan backend audit.

No credentials, external LLM, or external database required.
"""
from types import SimpleNamespace
from unittest.mock import Mock
import math

import pytest
from fastapi import HTTPException


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch, tmp_path):
    import core.config as cfg
    import core.ratelimit as rl
    for key in ('ARIA_REDIS_URL', 'ARIA_REQUIRE_SHARED_STORE', 'ARIA_WORKERS',
                'WEB_CONCURRENCY', 'UVICORN_WORKERS', 'ARIA_RATE_LIMIT_STORE'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('ARIA_ENV', 'development')
    monkeypatch.setenv('ARIA_RATE_LIMIT', 'on')
    monkeypatch.setattr(cfg, 'SESSION_STORE', {})
    monkeypatch.setattr(cfg, 'SESSION_STORE_PATH', tmp_path / 'sessions.json')
    monkeypatch.setattr(cfg, '_ensure_user_paths', lambda session, uid: session)
    monkeypatch.setattr(cfg, '_REDIS_CLIENT', None)
    monkeypatch.setattr(rl, '_redis_client', None)
    monkeypatch.setattr(rl, '_hits', __import__('collections').defaultdict(list))


def test_shared_login_rejection_is_not_swallowed(monkeypatch):
    from api.auth_routes import _check_login_rate_limit
    import core.ratelimit as rl
    monkeypatch.setattr(rl, '_redis_check_limit', Mock(side_effect=HTTPException(429, 'limit')))
    with pytest.raises(HTTPException) as exc:
        _check_login_rate_limit(SimpleNamespace(client=SimpleNamespace(host='audit-client')))
    assert exc.value.status_code == 429


@pytest.mark.parametrize('mode', ['explicit', 'production', 'configured'])
def test_failed_redis_never_silently_splits_limits(monkeypatch, mode):
    import core.ratelimit as rl
    monkeypatch.setattr(rl, '_redis_check_limit', lambda *a: False)
    if mode == 'explicit':
        monkeypatch.setenv('ARIA_REQUIRE_SHARED_STORE', '1')
    elif mode == 'production':
        monkeypatch.setenv('ARIA_ENV', 'production')
        monkeypatch.setenv('ARIA_WORKERS', '2')
    else:
        monkeypatch.setenv('ARIA_REDIS_URL', 'redis://unavailable')
    with pytest.raises(HTTPException) as exc:
        rl.check_limit('audit', 'client', 10, 60)
    assert exc.value.status_code == 503
    assert not rl._hits


def test_session_read_and_write_fail_closed(monkeypatch):
    import core.config as cfg
    monkeypatch.setenv('ARIA_REQUIRE_SHARED_STORE', '1')
    monkeypatch.setattr(cfg, 'get_redis_client', lambda: None)
    with pytest.raises(HTTPException):
        cfg.get_session(1)
    with pytest.raises(HTTPException):
        cfg.save_session(1, {'db_name': 'never-committed'})
    assert 1 not in cfg.SESSION_STORE


def test_redis_miss_does_not_resurrect_worker_snapshot(monkeypatch):
    import core.config as cfg
    monkeypatch.setenv('ARIA_REDIS_URL', 'redis://test')
    monkeypatch.setattr(cfg, 'get_redis_client', lambda: SimpleNamespace(get=lambda key: None))
    cfg.SESSION_STORE[1] = (10**12, {'db_name': 'stale'})
    assert cfg.get_session(1)['db_name'] is None


def test_provider_failure_keeps_selection_and_safe_error(monkeypatch):
    import core.config as cfg
    import llm_provider
    fake = Mock(side_effect=RuntimeError('secret-test-token'))
    monkeypatch.setattr(llm_provider, 'create_provider', fake)
    session = {'provider_name': 'deepseek', 'provider': None}
    cfg._ensure_provider(session)
    assert session['provider_name'] == 'deepseek'
    assert session['provider'] is None
    assert 'secret-test-token' not in session['_provider_error']
    assert fake.call_count == 1


def test_saved_card_cannot_override_metadata_or_hide_nested_sql(monkeypatch):
    import api.prescription_routes as routes
    session = {}
    monkeypatch.setattr(routes, 'get_session', lambda uid: session)
    monkeypatch.setattr(routes, 'save_session', lambda uid, sess: None)
    body = routes.SaveRequest(prescription={
        'id': 'attacker', 'created_at': 'yesterday', 'question': 'override',
        'title': '<script>bad</script>safe', 'unknown': 'discard',
        'details': {'sql': 'SELECT secret', 'nested': {'api_key': 'secret'}, 'value': float('nan')},
    }, question='Original question')
    first = routes.prescription_save(body, None, user={'user_id': 1})['saved'][0]
    second = routes.prescription_save(body, None, user={'user_id': 1})['saved'][0]
    assert first['id'].startswith('rxsv_') and first['id'] != second['id']
    assert first['created_at'] != 'yesterday'
    assert first['question'] == 'Original question'
    assert '<' not in first['title']
    assert 'unknown' not in first
    assert first['details'] == {'nested': {}, 'value': None}


def test_saved_card_size_bound():
    from api.prescription_routes import _sanitize_saved_prescription
    with pytest.raises(HTTPException) as exc:
        _sanitize_saved_prescription({'title': 'x' * 70000})
    assert exc.value.status_code == 413


def test_dimension_only_schema_has_no_fake_measure():
    from goal_agent.grounding import GroundingMixin
    class Agent(GroundingMixin):
        tables = {'catalog': {'columns': {'model_year': {}, 'item_id': {}}, 'primary_key': ['item_id']}}
        def _classify_column(self, table, col): return 'numeric'
        def _is_dimension_like(self, table, col): return True
    assert Agent()._pick_measure_column('catalog') is None


@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf, -1])
def test_invalid_error_never_becomes_accuracy(bad):
    from prediction.evaluate import accuracy_from_mape
    assert accuracy_from_mape(bad) is None


def test_mape_handles_length_mismatch_and_nonfinite():
    from prediction.evaluate import mape
    assert mape([10, 20], [9]) == pytest.approx(10)
    assert mape([0, 0], [3, 5]) is None
    assert mape([10], [math.nan]) is None


def test_tokens_are_distinct_and_expiry_required(monkeypatch, tmp_path):
    import auth
    import jwt
    monkeypatch.setattr(auth, 'DB_PATH', tmp_path / 'auth.db')
    a, b = auth.issue_token(1, 'audit'), auth.issue_token(1, 'audit')
    assert a != b
    auth.revoke_token(a)
    assert auth.verify_token(a) is None
    assert auth.verify_token(b)['user_id'] == 1
    missing_exp = jwt.encode({'sub': '1', 'iat': 0}, auth.SECRET, algorithm=auth.JWT_ALGO)
    assert auth.verify_token(missing_exp) is None


def test_progress_terminal_result_survives_reconnect(monkeypatch):
    import api.connect_routes as routes
    monkeypatch.setattr(routes, '_upload_progress', {})
    routes._set_progress(71, 'ready', 100, 'Ready', done=True, result={'rows': 12})
    assert routes._get_progress_snapshot(71, terminal=True)['result']['rows'] == 12
    assert routes._get_progress_snapshot(71, terminal=True)['done'] is True
    routes._upload_progress[71]['timestamp'] -= routes._PROGRESS_TTL_S + 1
    assert routes._get_progress_snapshot(71) is None


def test_evidence_grade_downgrades_when_contradiction_outweighs_support():
    from hypothesis_engine.evidence_ladder import normalize
    assert normalize('Strong evidence', 3, 5) == 'Contradictory evidence'
    assert normalize('Moderate evidence', 3, 3) == 'Moderate evidence'


@pytest.mark.parametrize('value', ['nan', 'inf', '-inf', float('nan')])
def test_kpi_nonfinite_values_are_missing(value):
    from services.kpi_service import _coerce_num
    assert _coerce_num(value) is None


def test_rag_uses_selected_users_deepseek_key(monkeypatch):
    import core.config as cfg
    import api.insight_rag_routes as routes
    monkeypatch.setattr(cfg, 'get_session', lambda uid: {'provider_name': 'deepseek', 'api_keys': {'deepseek': 'test-only-key'}})
    provider = SimpleNamespace(models={})
    factory = Mock(return_value=provider)
    monkeypatch.setattr(routes, 'create_provider', factory)
    monkeypatch.setattr(routes, 'InsightAgent', lambda **kw: kw)
    monkeypatch.setattr(routes, 'InsightPDFRAG', lambda **kw: kw)
    result = routes._capability(1)
    assert factory.call_args.kwargs['api_key'] == 'test-only-key'
    assert 'rag_scope' in factory.call_args.kwargs['models']
    assert result['user_id'] == 1


def test_rag_does_not_switch_provider_on_session_outage(monkeypatch):
    import core.config as cfg
    import api.insight_rag_routes as routes
    monkeypatch.setattr(cfg, 'get_session', Mock(side_effect=HTTPException(503, 'store offline')))
    with pytest.raises(HTTPException) as exc:
        routes._capability(1)
    assert exc.value.status_code == 503


def test_feedback_oversize_and_nonfinite_rejected():
    from api.rl_routes import FeedbackEvent, post_event
    with pytest.raises(HTTPException) as exc:
        post_event(FeedbackEvent(event='chart', state={'data': 'x'*8000}), None, {'user_id': 1})
    assert exc.value.status_code == 413
    with pytest.raises(HTTPException) as exc:
        post_event(FeedbackEvent(event='chart', reward={'score': math.nan}), None, {'user_id': 1})
    assert exc.value.status_code == 422


def test_auth_import_does_not_load_secret():
    import subprocess, sys
    code = "import auth; assert 'SECRET' not in auth.__dict__"
    subprocess.run([sys.executable, '-c', code], check=True)


def test_unavailable_keystore_does_not_return_ciphertext_as_password():
    import sys
    from core.secret_box import decrypt
    if sys.platform != 'win32':
        with pytest.raises(RuntimeError):
            decrypt('enc1:not-a-password')
        assert decrypt('legacy-plaintext') == 'legacy-plaintext'


def test_generic_measure_words_do_not_swallow_specific_metrics():
    from goal_agent.suggestions import SuggestionsMixin
    class Agent(SuggestionsMixin):
        _GENERIC_TOTAL_WORDS = {'total', 'amount', 'value', 'sales', 'revenue', 'sum'}
        tables = {'records': {}}
        def _natural_label(self, name): return name.replace('_', ' ')
    agent = Agent()
    assert agent._measure_noun('records', 'total_amount') == 'overall total value'
    assert agent._measure_noun('records', 'sales_growth') == 'sales growth'
