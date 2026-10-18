"""Selected provider/model resolution shared by document chat entry points."""
RAG_ROLES = ("rag", "rag_scope", "rag_rewrite", "rag_verify", "rag_plan")

def configure_rag_models(provider):
    model = (provider.models.get("rag") or provider.models.get("semi")
             or provider.models.get("story") or provider.models.get("sql"))
    if not model and getattr(provider, "provider", None):
        from llm_provider import PROVIDERS
        defaults = PROVIDERS.get(provider.provider, {}).get("models", {})
        model = defaults.get("semi") or defaults.get("story") or defaults.get("sql")
    if not model:
        raise ValueError("Select a model for document chat in AI provider settings.")
    for role in RAG_ROLES:
        provider.models.setdefault(role, model)
    return provider

def provider_for_session(session):
    from core.config import resolve_provider_key
    from llm_provider import create_provider, PROVIDERS
    name = session.get("provider_name") or "local"
    if name not in PROVIDERS:
        raise ValueError("Selected AI provider is not supported.")
    provider = create_provider(
        name,
        api_key=None if name == "local" else resolve_provider_key(name, session.get("api_keys") or {}),
        models=(session.get("provider_models") or {}).get(name),
    )
    return configure_rag_models(provider)
