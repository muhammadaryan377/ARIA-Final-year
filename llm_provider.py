"""
ARIA LLM Provider
-----------------
Unified interface over two LLM backends so the agents can run either:

  * local  - Ollama (offline, private, data never leaves the machine)  [SLOW]
  * cloud  - Groq API (fast, hosted inference)                          [FAST]

Every agent talks to `chat()` and never needs to know which backend it is.

Privacy context shown to the user:
    local: slow but your data stays on your machine (fully secure/offline)
    cloud: fast, but your schema/queries are sent to the Groq API
"""

import json
import logging
import os
import re
import time
from contextvars import ContextVar
from urllib.parse import quote_plus

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Load .env centrally (in core.config) before any module-level os.getenv below.
import core.config  # noqa: E402,F401

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# DeepSeek (OpenAI-compatible) configuration
# ---------------------------------------------------------------------------

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_VISION_MODEL = os.getenv("DEEPSEEK_VISION_MODEL", DEEPSEEK_MODEL)
DEEPSEEK_THINKING = os.getenv("DEEPSEEK_THINKING", "disabled").strip().lower()
if DEEPSEEK_THINKING not in {"enabled", "disabled"}:
    DEEPSEEK_THINKING = "disabled"

# Control-plane RAG calls use tiny bounded outputs and must not spend that
# budget on hidden reasoning.
_CONTROL_ROLES = {"rag_scope", "rag_verify", "rag_rewrite", "rag_plan"}
_RETRY_AFTER_RE = re.compile(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)

# Per-user LLM budgets (DoS/cost containment). Normal use never trips these;
# bursts get a fail-closed 429 so one caller cannot loop the provider.
_LLM_BUCKET_MAX = int(os.getenv("ARIA_LLM_MAX_PER_MIN", "30"))
_LLM_BUCKET_WINDOW_S = int(os.getenv("ARIA_LLM_WINDOW_S", "60"))
_LLM_LOCAL_MINIMIZE_THRESHOLD = int(os.getenv("ARIA_LLM_LOCAL_MINIMIZE_CHARS", "24000"))
_llm_buckets: dict[str, list[float]] = {}


def _llm_rate_limit(key="default") -> None:
    """Enforce a per-user/per-role LLM call budget (fail-closed on exceed).

    Prefers the shared ``core.ratelimit.check_limit`` sliding window when
    available; otherwise falls back to a simple in-process token bucket with
    the same bounds. Exceeding the budget raises (429-style) instead of
    queueing, so bursts fail fast and never hang a request.
    """
    bucket_key = str(key or "default")
    try:
        from core.ratelimit import check_limit as _shared_check
        _shared_check("llm_provider", bucket_key,
                      _LLM_BUCKET_MAX, _LLM_BUCKET_WINDOW_S)
        return
    except ImportError:
        pass
    except Exception as exc:
        if getattr(exc, "status_code", None) in (429, 503):
            raise
    now = time.monotonic()
    hits = [t for t in _llm_buckets.get(bucket_key, []) if now - t < _LLM_BUCKET_WINDOW_S]
    if len(hits) >= _LLM_BUCKET_MAX:
        raise RuntimeError(
            f"LLM rate limit exceeded for '{bucket_key}' "
            f"({_LLM_BUCKET_MAX} calls per {_LLM_BUCKET_WINDOW_S}s). "
            "Slow down and try again shortly."
        )
    hits.append(now)
    _llm_buckets[bucket_key] = hits


# ---------------------------------------------------------------------------
# Per-question RAG LLM call budget (insight_rag RAG_MAX_LLM_CALLS_PER_QUERY)
# ---------------------------------------------------------------------------
# One user question must never loop the hosted provider: every RAG-scoped call
# is charged against a per-question budget and denied once the budget is spent.
# The counter is context-local (thread/task safe, never a process-global) and
# is reset exactly once per user question by the RAG request entry point
# (``insight_rag.scope_layer.chat``), so the summary retry that re-runs
# ``_chat`` spends the same budget as the first attempt.

class RagLLMBudgetExceeded(RuntimeError):
    """A RAG question asked for more LLM calls than its per-query budget.

    Raised *before* provider I/O. Every insight_rag call site already wraps its
    RAG-scoped LLM calls in ``except Exception`` and degrades to a fallback
    answer, refusal or clarification, so a budget denial takes the same
    least-disruptive path and never surfaces as a request error.
    """


_RAG_BUDGET_ACTIVE = ContextVar("aria_rag_llm_budget_active", default=False)
_RAG_BUDGET_LIMIT = ContextVar("aria_rag_llm_budget_limit", default=0)
_RAG_BUDGET_USED = ContextVar("aria_rag_llm_budget_used", default=0)


def is_rag_role(role) -> bool:
    """True only for RAG-scoped roles (``rag`` / ``rag_*``).

    Every other role (sql, story, suggest, semi, schema, vision — the
    prescription, prediction, copilot, goal and schema agents) is never
    counted, gated or slowed by the RAG budget.
    """
    name = str(role or "")
    return name == "rag" or name.startswith("rag_")


def begin_rag_llm_budget(max_calls: int):
    """Reset and activate the per-question RAG budget; pair with ``end_rag_llm_budget``.

    Returns the context token to hand back to :func:`end_rag_llm_budget`.
    """
    _RAG_BUDGET_USED.set(0)
    _RAG_BUDGET_LIMIT.set(int(max_calls))
    return _RAG_BUDGET_ACTIVE.set(True)


def end_rag_llm_budget(token) -> None:
    """Deactivate the budget and restore the previous context state."""
    _RAG_BUDGET_ACTIVE.reset(token)


def rag_llm_calls_used() -> int:
    """RAG-scoped calls charged so far in the active request (diagnostics/tests)."""
    return _RAG_BUDGET_USED.get()


def charge_rag_llm_call(role) -> None:
    """Charge one RAG-scoped LLM call; deny it once the per-question budget is spent."""
    if not _RAG_BUDGET_ACTIVE.get() or not is_rag_role(role):
        return
    used = _RAG_BUDGET_USED.get()
    limit = _RAG_BUDGET_LIMIT.get()
    if used >= limit:
        raise RagLLMBudgetExceeded(
            f"RAG per-question LLM budget exhausted ({limit} calls). "
            "Continuing on the existing degraded/fallback path."
        )
    _RAG_BUDGET_USED.set(used + 1)


def _content_chars(messages) -> int:
    """Total prompt chars (used to bound the local path like the cloud path)."""
    try:
        if isinstance(messages, str):
            return len(messages)
        total = 0
        for m in messages or []:
            if isinstance(m, dict):
                content = m.get("content")
                if isinstance(content, str):
                    total += len(content)
                elif isinstance(content, list):
                    for p in content:
                        if isinstance(p, dict) and isinstance(p.get("text"), str):
                            total += len(p["text"])
        return total
    except Exception:
        return 0

# OmniRoute gateway (OpenAI-compatible routing to many models).
OMNIROUTE_BASE_URL = os.getenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1").rstrip("/")

# Grok / xAI (OpenAI-compatible). Generic hosted provider: key from
# GROK_API_KEY (fallback XAI_API_KEY), endpoint override via GROK_BASE_URL,
# model override via GROK_MODEL (fallback XAI_MODEL). No dataset-specific
# defaults; any model id listed by the API can be used.
GROK_BASE_URL = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1").rstrip("/")
GROK_MODEL = os.getenv("GROK_MODEL", os.getenv("XAI_MODEL", "grok-3-mini")).strip() or "grok-3-mini"

# Cloud privacy controls (P1-11). Hosted inference needs explicit *backend*
# consent: ARIA_CLOUD_ALLOWED=0 is a kill-switch, and ARIA_CLOUD_CONSENT=1 is
# an opt-in. A frontend toggle or per-request flag alone is insufficient —
# the backend env must grant consent, otherwise cloud calls raise and Local
# Ollama remains fully usable offline.
_CLOUD_CONSENT_VALUES = {"1", "true", "yes", "on"}

def _cloud_allowed() -> bool:
    return os.getenv("ARIA_CLOUD_ALLOWED", "1").strip() not in {"0", "false", "no", "off"}


def _cloud_consent_granted() -> bool:
    return os.getenv("ARIA_CLOUD_CONSENT", "0").strip().lower() in _CLOUD_CONSENT_VALUES


def _require_cloud_consent(provider: str, consent=None):
    if provider == "local":
        return
    if not _cloud_allowed():
        raise RuntimeError(
            "Cloud processing is disabled by configuration (ARIA_CLOUD_ALLOWED=0). "
            "Switch to Local Ollama or enable cloud consent."
        )
    # Explicit per-call deny always wins.
    if consent is False:
        raise RuntimeError(
            "Cloud processing was explicitly denied for this request. "
            "Switch to Local Ollama or grant consent."
        )
    # Backend consent is mandatory; a frontend-supplied consent flag alone
    # never enables cloud (it is accepted only as an additional signal).
    if not _cloud_consent_granted():
        raise RuntimeError(
            "Cloud processing requires explicit backend consent (ARIA_CLOUD_CONSENT=1). "
            "Switch to Local Ollama or set backend consent."
        )


_SECRET_PATTERNS = [
    re.compile(r"(gsk_[A-Za-z0-9]{8,})"),
    re.compile(r"(xai-[A-Za-z0-9\-_]{8,})"),
    re.compile(r"(sk-[A-Za-z0-9\-_]{8,})"),
    re.compile(r"(hf_[A-Za-z0-9]{8,})"),
    re.compile(r"(AKIA[0-9A-Z]{16})"),
    re.compile(r"(?i)(aws_secret_access_key\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)([^\s,;'\"]+)"),
    re.compile(r"(?i)(password\s*[:=]\s*)([^\s,;'\"]+)"),
    re.compile(r"(?i)(passwd\s*[:=]\s*)([^\s,;'\"]+)"),
    re.compile(r"(?i)(secret\s*[:=]\s*)([^\s,;'\"]+)"),
    re.compile(r"(?i)(bearer\s+)([A-Za-z0-9\-._~+/=]+)"),
    re.compile(r"(?i)(authorization\s*:\s*(?:bearer\s+|basic\s+)?)([A-Za-z0-9\-._~+/=]{8,})"),
    re.compile(r"(eyJ[A-Za-z0-9\-_]{8,}\.[A-Za-z0-9\-_]{8,}\.[A-Za-z0-9\-_]{8,})"),
    re.compile(r"(?i)(mongodb(?:\+srv)?|postgresql|postgres|mysql|mssql|sqlserver|redis|sqlite):\/\/[^\s'\";]+"),
    re.compile(r"(?i)(server\s*=\s*[^;]+;\s*(?:uid|user id)\s*=\s*[^;]+;\s*(?:pwd|password)\s*=\s*)([^;]+)"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(groq_api_key|deepseek_api_key|openai_api_key|xai_api_key|grok_api_key\s*[:=]\s*)([^\s,;'\"]+)"),
]


def _redact_secrets(text: str) -> str:
    """Remove credentials/tokens/connection strings before cloud transmission."""
    if not isinstance(text, str):
        return text
    out = text
    for pat in _SECRET_PATTERNS:
        try:
            if pat.groups >= 2:
                out = pat.sub(lambda m: (m.group(1) + "[REDACTED]"), out)
            else:
                out = pat.sub("[REDACTED]", out)
        except Exception:
            continue
    return out


def _sanitize_for_log(value) -> str:
    """Redact secrets and truncate before any logging/persistence.

    Call on exception messages and payloads before they reach logs, traces,
    or stored diagnostics. Redaction runs first; truncation second.
    """
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        return "[unrenderable]"
    redacted = _redact_secrets(text)
    if len(redacted) > 500:
        redacted = redacted[:500] + "…[truncated]"
    return redacted


def _minimize_for_cloud(messages, max_chars: int = 4000):
    """Data-minimize cloud payloads: redact secrets, bound size.

    Keeps schema metadata and the tail (usually the actual question) while
    truncating over-long middle content. Local provider path never calls this.
    """
    if isinstance(messages, str):
        red = _redact_secrets(messages)
        if len(red) > max_chars:
            # Keep the tail (question usually lives at the end)
            red = "…[context truncated]\n" + red[-(max_chars - 30):]
        return red
    out = []
    for m in messages or []:
        if not isinstance(m, dict):
            out.append(m)
            continue
        content = m.get("content")
        if isinstance(content, str):
            red = _redact_secrets(content)
            if len(red) > max_chars:
                red = red[:max_chars] + "\n…[truncated for cloud minimization]"
            out.append({**m, "content": red})
        elif isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    t = _redact_secrets(p["text"])
                    parts.append({**p, "text": t[:max_chars]})
                else:
                    parts.append(p)
            out.append({**m, "content": parts})
        else:
            out.append(m)
    return out


def _prepare_role_messages(role, messages):
    """Preserve bounded RAG evidence and trailing questions across provider I/O.

    RAG builds its own context budget. The generic 4k minimizer must not cut
    evidence, JSON schemas, candidate answers, or the latest question silently.
    Secret redaction still applies; oversized RAG payloads fail explicitly.
    """
    if role not in {"rag", "rag_scope", "rag_verify", "rag_rewrite", "rag_plan"}:
        return _minimize_for_cloud(messages)
    limit = 64000
    if _content_chars(messages) > limit:
        raise ValueError("Document request exceeds the 64000-character provider context budget; narrow the document scope.")
    return _minimize_for_cloud(messages, max_chars=limit)

# ---------------------------------------------------------------------------
# Provider metadata
# ---------------------------------------------------------------------------

PROVIDERS = {
    "local": {
        "label": "Local LLM (Ollama)",
        "privacy": "Your data never leaves your machine. Fully offline & secure, but slower on commodity hardware.",
        "models": {
            "sql": "aria-sql-mistral",
            "story": "aria-story-mistral",
            "suggest": "aria-goal",
            "semi": "aria-goal",
            "schema": "aria-goal",
        },
    },
    "cloud": {
        "label": "Cloud LLM (Groq API)",
        "privacy": "Fast inference on Groq's LPU hardware, but your schema, queries and data samples are sent to the Groq API.",
        "models": {
            "sql": "qwen/qwen3.8-27b",
            "story": "qwen/qwen3.8-27b",
            "suggest": "qwen/qwen3.8-27b",
            "semi": "qwen/qwen3.8-27b",
            "schema": "qwen/qwen3.8-27b",
            "vision": "qwen/qwen3.8-27b",
        },
    },
    "omniroute": {
        "label": "OmniRoute",
        "privacy": "Aggregated model routing; schema and queries are sent to the OmniRoute API.",
        "models": {
            "free-best": "omniroute/free-best",
            "free-fast": "omniroute/free-fast",
            "free-chat": "omniroute/free-chat",
            "open": "omniroute/open",
        },
    },
    "deepseek": {
        "label": "Cloud LLM (DeepSeek API)",
        "privacy": "Hosted DeepSeek inference. Your schema, queries, selected PDF evidence, and data samples may be sent to the DeepSeek API.",
        "models": {
            "sql": DEEPSEEK_MODEL,
            "story": DEEPSEEK_MODEL,
            "suggest": DEEPSEEK_MODEL,
            "semi": DEEPSEEK_MODEL,
            "schema": DEEPSEEK_MODEL,
            "vision": DEEPSEEK_VISION_MODEL,
        },
    },
    "grok": {
        "label": "Cloud LLM (Grok xAI API)",
        "privacy": "Hosted xAI Grok inference (OpenAI-compatible). Your schema, queries, selected PDF evidence, and data samples may be sent to the xAI API.",
        "models": {
            "sql": GROK_MODEL,
            "story": GROK_MODEL,
            "suggest": GROK_MODEL,
            "semi": GROK_MODEL,
            "schema": GROK_MODEL,
        },
    },
}

# New providers deliberately require an account-supported model ID.
for _name, _label, _env in (("openai", "OpenAI / GPT", "OPENAI_MODEL"),
                             ("anthropic", "Anthropic / Claude", "ANTHROPIC_MODEL")):
    PROVIDERS[_name] = {"label": _label, "models": {
        role: os.getenv(_env, "") for role in ("sql", "story", "suggest", "semi", "schema", "vision")}}

GROQ_MODEL = PROVIDERS["cloud"]["models"]["sql"]

# Local Ollama models default to a small runtime context (often 4096 tokens),
# which the schema-reasoning prompt + long output can exceed. Raise it so large
# schemas are not rejected with an "exceeds the available context size" error.
LOCAL_NUM_CTX = 8192


class LLMProvider:
    """Unified chat interface: local (Ollama) or cloud (Groq)."""

    def __init__(self, provider="local", api_key=None, models=None, base_url=None):
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from {list(PROVIDERS)}")
        # Provider-selection layer enforcement: constructing a hosted
        # provider without backend consent fails closed (no network touched).
        if provider != "local":
            _require_cloud_consent(provider)
        self.provider = provider
        self.base_url = base_url
        self.models = dict(PROVIDERS[provider]["models"])
        if models:
            self.models.update(models)

        # Unified cloud client used by all hosted providers (Groq "cloud" and
        # DeepSeek "deepseek"). Kept as a single attribute so every cloud path
        # shares the same retry/error handling.
        self._cloud_client = None
        if provider == "cloud":
            if not GROQ_AVAILABLE:
                raise RuntimeError("groq package is not installed. Run: pip install groq")
            # max_retries=0: the SDK's internal retry/backoff on 429s can block for
            # minutes. We handle rate limits ourselves with a short, bounded retry.
            self._cloud_client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"), max_retries=0, base_url=base_url) if base_url else Groq(api_key=api_key or os.getenv("GROQ_API_KEY"), max_retries=0)
            if not self._cloud_client.api_key:
                raise RuntimeError("Missing GROQ_API_KEY. Set it in .env or pass api_key.")
        elif provider in ("openai", "anthropic"):
            from services.hosted_clients import make_client
            self._cloud_client = make_client(provider, api_key, base_url)
        elif provider == "deepseek":
            from openai import OpenAI
            self._cloud_client = OpenAI(
                api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
                base_url=base_url or DEEPSEEK_BASE_URL,
            )
            if not self._cloud_client.api_key:
                raise RuntimeError("Missing DEEPSEEK_API_KEY. Set it in .env or pass api_key.")
        elif provider == "omniroute":
            from openai import OpenAI
            self._cloud_client = OpenAI(
                api_key=api_key or os.getenv("OMNIROUTE_API_KEY"),
                base_url=base_url or OMNIROUTE_BASE_URL,
            )
            if not self._cloud_client.api_key:
                raise RuntimeError("Missing OMNIROUTE_API_KEY. Set it in .env or pass api_key.")
        elif provider == "grok":
            from openai import OpenAI
            self._cloud_client = OpenAI(
                api_key=api_key or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY"),
                base_url=base_url or GROK_BASE_URL,
            )
            if not self._cloud_client.api_key:
                raise RuntimeError("Missing GROK_API_KEY (or XAI_API_KEY). Set it in .env or pass api_key.")

    # -- model lookup ---------------------------------------------------

    def model_for(self, role):
        return self.models.get(role)

    # -- thinking control ------------------------------------------------

    def _thinking_body(self, role=None):
        """Return provider-specific ``extra_body`` for reasoning control.

        DeepSeek accepts a ``thinking`` switch: control-plane RAG calls use tiny
        bounded outputs and must not spend that budget on hidden reasoning, so
        they always disable it. General DeepSeek generation follows the env
        switch. Groq does not accept this field, so it gets an empty body.
        """
        if self.provider != "deepseek":
            return {}
        thinking = "disabled" if role in _CONTROL_ROLES else DEEPSEEK_THINKING
        return {"thinking": {"type": thinking}}

    def ping(self, timeout: int = 20) -> bool:
        """Validate credentials/connectivity without spending generation budget.

        Hosted providers: list models (proves key + reachability even when a
        model returns empty content for tiny probes). Local: tiny chat.
        Raises on failure with the underlying reason.
        """
        if self.provider != "local":
            _require_cloud_consent(self.provider)
        if self.provider == "local":
            out = self.chat("suggest",
                            messages=[{"role": "user", "content": "Reply with: OK"}],
                            temperature=0, num_predict=5, timeout=timeout)
            if not out or not str(out).strip():
                raise RuntimeError("local model returned no content")
            return True
        client = self._cloud_client
        if client is None:
            raise RuntimeError(f"No client initialised for provider '{self.provider}'")
        try:
            client.with_options(timeout=timeout).models.list()
        except TypeError:
            # Older SDKs without with_options/timeout override.
            client.models.list()
        return True

    def chat_structured(
        self,
        role,
        messages,
        *,
        json_schema,
        schema_name="structured_response",
        temperature=0.0,
        num_predict=900,
        timeout=None,
        reasoning_effort="low",
    ):
        """Return JSON-mode content from the hosted provider as a raw string.

        DeepSeek and Groq both support ``response_format={type: json_object}``.
        ARIA still performs its own Pydantic/schema validation afterwards, so
        provider JSON mode is a formatting hint, never a trust boundary.
        The JSON Schema is injected as a system instruction that the caller's
        own validation enforces.
        """
        charge_rag_llm_call(role)
        del reasoning_effort  # accepted for interface compatibility
        if self.provider == "local":
            raise RuntimeError("Structured output is available only on a hosted provider.")
        _require_cloud_consent(self.provider)
        if timeout is None:
            timeout = 20
        model = self.model_for(role)
        if not model:
            raise ValueError(f"No model configured for role '{role}'")
        if not isinstance(json_schema, dict) or json_schema.get("type") != "object":
            raise ValueError("json_schema must be a JSON Schema object definition.")

        schema_text = json.dumps(json_schema, ensure_ascii=False, separators=(",", ":"))
        structured_messages = [
            {
                "role": "system",
                "content": (
                    f"Return only one valid JSON object named {schema_name}. "
                    "It must match this JSON Schema exactly. Do not use markdown fences, "
                    "commentary, or extra keys. "
                    f"JSON Schema: {schema_text}"
                ),
            },
            *messages,
        ]
        extra_body = self._thinking_body(role)
        structured_messages = _prepare_role_messages(role, structured_messages)
        try:
            def _request():
                kwargs = {
                    "model": model,
                    "messages": structured_messages,
                    "temperature": temperature,
                    "max_tokens": num_predict,
                    "timeout": timeout,
                    "response_format": {"type": "json_object"},
                }
                if extra_body:
                    kwargs["extra_body"] = extra_body
                return self._cloud_client.chat.completions.create(**kwargs)

            completion = self._cloud_with_retry(_request)
            content = completion.choices[0].message.content
            if not content:
                raise RuntimeError("Structured LLM response was empty.")
            return content.strip()
        except TimeoutError:
            raise
        except Exception as exc:
            logging.warning("LLMProvider.%s structured chat failed (%s): %s",
                            self.provider, role, _sanitize_for_log(exc))
            raise

    # -- unified chat ---------------------------------------------------

    def chat(self, role, messages, temperature=0.1, num_predict=400, timeout=None,
             user_id=None):
        """Send messages to the active backend for a given role (sql/story/suggest/semi).

        `timeout` is in seconds. If the call exceeds it, a TimeoutError is raised so
        the caller can fall back to a fast template instead of waiting indefinitely.
        Defaults: 15s cloud, 30s local, 120s max for local story (never 900s).
        A per-user rate-limit budget is enforced before any provider I/O.
        RAG-scoped roles are additionally charged against the per-question
        RAG budget (see ``charge_rag_llm_call``).
        """
        charge_rag_llm_call(role)
        _llm_rate_limit(str(user_id or "default") + f":{role}")
        if timeout is None:
            if self.provider == "local" and role in ("story",):
                # Bounded local story budget: 120s max. An explicit timeout
                # passed by the caller always wins but is capped below.
                timeout = 120
            else:
                # Short, bounded timeouts so analytical queries respond quickly
                # rather than blocking indefinitely on a slow LLM response.
                timeout = 15 if self.provider != "local" else 30
        timeout = min(int(timeout), 120) if self.provider == "local" else int(timeout)
        if self.provider == "local" and _content_chars(messages) > _LLM_LOCAL_MINIMIZE_THRESHOLD:
            # Same data-minimization as the cloud path (4000 chars/msg) when
            # local context exceeds 24k chars: bounds memory/time on CPU.
            messages = _prepare_role_messages(role, messages)
        model = self.model_for(role)
        if not model:
            raise ValueError(f"No model configured for role '{role}'")

        try:
            if self.provider == "local":
                return self._chat_local(model, messages, temperature, num_predict, timeout)
            return self._chat_cloud(role, model, messages, temperature, num_predict, timeout)
        except TimeoutError:
            raise
        except Exception as exc:
            logging.warning("LLMProvider.%s chat failed (%s): %s",
                            self.provider, role, _sanitize_for_log(exc))
            raise

    def complete(self, role, prompt, temperature=0.1, num_predict=400, timeout=None,
                 user_id=None):
        """Generate a raw completion for `prompt`.

        Used for completion-only models (e.g. sqlcoder locally, which has no chat
        template and just echoes the prompt when driven through `chat()`). For the
        cloud backend the prompt is sent as a single user message.
        Local timeouts are capped at 120s (never 900s); a per-user rate-limit
        budget is enforced before any provider I/O. RAG-scoped roles are
        additionally charged against the per-question RAG budget.
        """
        charge_rag_llm_call(role)
        _llm_rate_limit(str(user_id or "default") + f":{role}")
        if timeout is None:
            if self.provider == "local" and role == "sql":
                timeout = 120
            else:
                timeout = 30 if self.provider != "local" else 120
        timeout = min(int(timeout), 120) if self.provider == "local" else int(timeout)
        if self.provider == "local" and isinstance(prompt, str) \
                and len(prompt) > _LLM_LOCAL_MINIMIZE_THRESHOLD:
            prompt = _minimize_for_cloud(prompt)
        model = self.model_for(role)
        if not model:
            raise ValueError(f"No model configured for role '{role}'")

        try:
            if self.provider == "local":
                return self._complete_local(model, prompt, temperature, num_predict, timeout)
            _require_cloud_consent(self.provider)
            prompt = _minimize_for_cloud(prompt)
            extra_body = self._thinking_body(role)
            def _request():
                kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": num_predict,
                    "timeout": timeout,
                }
                if extra_body:
                    kwargs["extra_body"] = extra_body
                return self._cloud_client.chat.completions.create(**kwargs)

            completion = self._cloud_with_retry(_request)
            content = completion.choices[0].message.content
            if not (content or "").strip():
                # Reasoning models intermittently return empty content at
                # temperature 0. One bounded retry with light sampling.
                def _retry_request():
                    kwargs = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": max(temperature, 0.3),
                        "max_tokens": num_predict,
                        "timeout": timeout,
                    }
                    if extra_body:
                        kwargs["extra_body"] = extra_body
                    return self._cloud_client.chat.completions.create(**kwargs)

                completion = self._cloud_with_retry(_retry_request)
                content = completion.choices[0].message.content
            return (content or "").strip()
        except TimeoutError:
            raise
        except Exception as exc:
            logging.warning("LLMProvider.%s complete failed (%s): %s",
                            self.provider, role, _sanitize_for_log(exc))
            raise

    def _complete_local(self, model, prompt, temperature, num_predict, timeout):
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("ollama package is not installed. Run: pip install ollama")
        client = ollama.Client(timeout=timeout)
        response = client.generate(
            model=model,
            prompt=prompt,
            keep_alive="30m",
            options={
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": LOCAL_NUM_CTX,
                "stop": ["<|im_end|>", "<|im_start|>", "```", "<|endoftext|>", "\n\nUser:"],
            },
        )
        return response["response"].strip()

    def _chat_local(self, model, messages, temperature, num_predict, timeout):
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("ollama package is not installed. Run: pip install ollama")
        client = ollama.Client(timeout=timeout)
        response = client.chat(
            model=model,
            messages=messages,
            keep_alive="30m",
            options={"temperature": temperature, "num_predict": num_predict, "num_ctx": LOCAL_NUM_CTX},
        )
        return response["message"]["content"].strip()

    def _chat_cloud(self, role, model, messages, temperature, num_predict, timeout):
        _require_cloud_consent(self.provider)
        messages = _prepare_role_messages(role, messages)
        extra_body = self._thinking_body(role)
        def _request():
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": num_predict,
                "timeout": timeout,
            }
            if extra_body:
                kwargs["extra_body"] = extra_body
            return self._cloud_client.chat.completions.create(**kwargs)

        completion = self._cloud_with_retry(_request)
        content = completion.choices[0].message.content
        if not (content or "").strip():
            # Same bounded empty-content retry as complete(): reasoning models
            # intermittently return nothing at temperature 0.
            def _retry_request():
                retry_kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": max(temperature, 0.3),
                    "max_tokens": num_predict,
                    "timeout": timeout,
                }
                if extra_body:
                    retry_kwargs["extra_body"] = extra_body
                return self._cloud_client.chat.completions.create(**retry_kwargs)

            completion = self._cloud_with_retry(_retry_request)
            content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("Cloud LLM response was empty.")
        return content.strip()

    def _retry_after_seconds(self, exc):
        """Extract a bounded retry-after hint from a rate-limit error.

        Looks at the response header first, then the error message text for a
        "try again in Ns" hint. Returns a float in seconds, or None.
        """
        response = getattr(exc, "response", None)
        if response is not None and hasattr(response, "headers"):
            try:
                retry_after = float(response.headers.get("retry-after", 0))
                if retry_after > 0:
                    return retry_after
            except Exception:
                pass
        message = str(exc)
        match = _RETRY_AFTER_RE.search(message)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                pass
        return None

    def _cloud_with_retry(self, fn, attempts=2):
        """Call `fn` (a hosted API request) with a bounded retry for rate limits.

        Both clients are configured/used with retries disabled so a 429 never
        blocks on the SDK's own long retry-after backoff. Here we wait only a
        short, capped delay and give up after a few attempts, raising the last
        error so callers fall back to templates instead of hanging on the request.
        NOTE: blocking sleep is intentional here; async routes offload the
        whole pipeline via asyncio.to_thread so the event loop never blocks.
        """
        last_exc = None
        for attempt in range(attempts):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                response = getattr(exc, "response", None)
                status = getattr(exc, "status_code", None)
                if status is None and response is not None:
                    status = getattr(response, "status_code", None)
                if status != 429 or attempt >= attempts - 1:
                    raise
                wait = 1.5
                retry_after = self._retry_after_seconds(exc)
                if retry_after is not None:
                    wait = min(retry_after + 0.5, 30.0)
                logging.warning("Rate limit (429); retrying in %.1fs (attempt %d/%d)",
                                wait, attempt + 1, attempts)
                time.sleep(wait)
        raise last_exc  # pragma: no cover

    def vision(self, prompt, images, temperature=0.1, num_predict=1500, timeout=60):
        """Send rendered page images (base64 PNG strings) plus a text prompt to a
        vision model. Available on a hosted backend with a vision model.

        Used to transcribe scanned/handwritten PDFs into structured records.
        """
        model = self.models.get("vision")
        if not model:
            raise RuntimeError(f"No vision model configured for provider '{self.provider}'.")
        if self.provider == "local":
            raise RuntimeError(
                "Handwritten/scanned PDF extraction requires a hosted provider "
                "with a vision model. Switch backend and retry."
            )
        _require_cloud_consent(self.provider)
        prompt = _minimize_for_cloud(prompt)
        content = [{"type": "text", "text": prompt}]
        for b64 in images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        try:
            return self._vision_once(prompt, content, model, temperature, num_predict, timeout)
        except TimeoutError:
            raise

    def _vision_once(self, prompt, content, model, temperature, num_predict, timeout):
        kwargs = {}
        if "json" in (prompt or "").lower() or model.startswith("qwen/"):
            kwargs["response_format"] = {"type": "json_object"}
        extra_body = self._thinking_body("vision")
        def _request():
            call_kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "temperature": temperature,
                "max_tokens": num_predict,
                "timeout": timeout,
            }
            if extra_body:
                call_kwargs["extra_body"] = extra_body
            call_kwargs.update(kwargs)
            return self._cloud_client.chat.completions.create(**call_kwargs)

        completion = self._cloud_with_retry(_request)
        result = completion.choices[0].message.content
        if not result:
            raise RuntimeError("Vision response was empty.")
        return result.strip()

    def __repr__(self):
        return f"LLMProvider(provider={self.provider}, models={self.models})"


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

_UNSET = object()

def create_provider(provider=_UNSET, api_key=None, models=None, base_url=None,
                    warn_on_default=False):
    """Build an LLMProvider.

    Policy: never silently switch backend. If a hosted provider ('cloud',
    'deepseek', 'omniroute' or 'grok') is requested but its dependencies/key are
    missing, raise so the caller/user decides (no silent downgrade to local). Switching provider
    mid-session only happens through an explicit user action (see
    /api/provider/switch) — never automatically.
    """
    if provider is _UNSET:
        if warn_on_default:
            logging.getLogger("aria.llm_provider").warning(
                "create_provider called without explicit provider – defaulting to local"
            )
        provider = "local"
    # Provider-selection layer enforcement (fail closed before any import,
    # client construction, or network I/O for hosted providers).
    if provider != "local":
        _require_cloud_consent(provider)
    if provider == "cloud":
        if not GROQ_AVAILABLE:
            raise RuntimeError(
                "The Cloud provider is unavailable because the 'groq' package is not "
                "installed (pip install groq). Stay on Local or fix this first."
            )
        llm = LLMProvider("cloud", api_key=api_key or os.getenv("GROQ_API_KEY"), models=models, base_url=base_url)
        if not llm._cloud_client.api_key:
            raise RuntimeError("Missing GROQ_API_KEY. Set it in .env or pass api_key.")
        return llm
    if provider == "deepseek":
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "The DeepSeek provider is unavailable because the 'openai' package is not "
                "installed (pip install openai). Choose another provider."
            )
        llm = LLMProvider("deepseek", api_key=api_key or os.getenv("DEEPSEEK_API_KEY"), models=models, base_url=base_url)
        if not llm._cloud_client.api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY. Set it in .env or pass api_key.")
        return llm
    if provider == "omniroute":
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "The OmniRoute provider is unavailable because the 'openai' package is not "
                "installed (pip install openai). Choose another provider."
            )
        llm = LLMProvider("omniroute", api_key=api_key or os.getenv("OMNIROUTE_API_KEY"), models=models, base_url=base_url)
        if not llm._cloud_client.api_key:
            raise RuntimeError("Missing OMNIROUTE_API_KEY. Set it in .env or pass api_key.")
        return llm
    if provider == "grok":
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "The Grok provider is unavailable because the 'openai' package is not "
                "installed (pip install openai). Choose another provider."
            )
        llm = LLMProvider("grok", api_key=api_key or os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY"), models=models, base_url=base_url)
        if not llm._cloud_client.api_key:
            raise RuntimeError("Missing GROK_API_KEY (or XAI_API_KEY). Set it in .env or pass api_key.")
        return llm
    if provider in ("openai", "anthropic"):
        return LLMProvider(provider, api_key=api_key, models=models, base_url=base_url)
    if provider != "local":
        raise ValueError(f"Unknown provider '{provider}'")
    return LLMProvider("local", models=models)
