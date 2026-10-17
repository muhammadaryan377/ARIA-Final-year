"""Adapters preserving ARIA's shared retry, consent and prompt handling."""
import os
from types import SimpleNamespace


class ClaudeClient:
    def __init__(self, api_key, timeout=25):
        self.api_key = api_key
        if not api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY")
        self.timeout = timeout
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.models = SimpleNamespace(list=self.list_models)

    def with_options(self, timeout=25):
        return ClaudeClient(self.api_key, timeout)

    def _request(self, method, path, **kwargs):
        import httpx
        with httpx.Client(timeout=kwargs.pop("timeout", self.timeout), follow_redirects=False) as client:
            response = client.request(method, "https://api.anthropic.com/v1/" + path,
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, **kwargs)
            if response.status_code >= 300:
                # Never propagate a body which could contain user content or secrets.
                raise RuntimeError(f"Anthropic request failed (HTTP {response.status_code})")
            return response.json()

    def list_models(self):
        result = self._request("GET", "models", params={"limit": 100})
        return SimpleNamespace(data=[SimpleNamespace(id=m["id"]) for m in result["data"]])

    def create(self, **kwargs):
        system, messages = [], []
        for message in kwargs["messages"]:
            if message["role"] in ("system", "developer"):
                system.append(message["content"])
            else:
                if not isinstance(message["content"], str):
                    raise ValueError("Claude adapter currently supports text analysis only")
                messages.append({"role": message["role"], "content": message["content"]})
        body = {"model": kwargs["model"], "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 1024)}
        if system:
            body["system"] = "\n\n".join(system)
        # JSON shape is requested in ARIA's system prompt and validated downstream.
        result = self._request("POST", "messages", json=body, timeout=kwargs.get("timeout") or self.timeout)
        text = "".join(block.get("text", "") for block in result["content"] if block["type"] == "text")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class OpenAIClient:
    def __init__(self, client):
        self.client = client
        self.api_key = client.api_key
        self.models = client.models
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def with_options(self, **kwargs):
        return OpenAIClient(self.client.with_options(**kwargs))

    def create(self, **kwargs):
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens", 1024)
        # Some reasoning models do not accept a temperature override.
        kwargs.pop("temperature", None)
        return self.client.chat.completions.create(**kwargs)


def make_client(provider, api_key=None, base_url=None):
    if base_url:
        raise ValueError("OpenAI and Claude use their official endpoints; custom URLs are not supported")
    if provider == "anthropic":
        return ClaudeClient(api_key or os.getenv("ANTHROPIC_API_KEY"))
    from openai import OpenAI
    return OpenAIClient(OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"),
                               base_url="https://api.openai.com/v1", max_retries=0))
