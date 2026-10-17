# Startup provider and model selection

The startup picker supports Groq (`cloud`), DeepSeek, OpenAI (`openai`),
Anthropic (`anthropic`), xAI (`grok`) and local Ollama. Choose a provider,
enter its API key (or reuse its saved key), load the account's available models,
and choose a text/chat model ID. You can also enter an ID manually.
The list can include embedding or other non-chat models; those cannot be used here.
No model is advertised as the strongest and no cross-provider fallback is used.

Hosted use still requires `ARIA_CLOUD_CONSENT=1` in the backend environment.
Provider environment key alternatives are `GROQ_API_KEY`, `DEEPSEEK_API_KEY`,
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GROK_API_KEY` (or `XAI_API_KEY`).
OpenAI/Claude have no hardcoded model default: pick an account-supported ID,
or configure `OPENAI_MODEL` / `ANTHROPIC_MODEL` on the server.

Continue makes a real bounded generation request to validate the chosen model.
This consumes provider quota and may be billed. Model discovery does not save
credentials. Successful selection saves the key using existing encrypted session
storage, and saves model IDs for reconstruction and subsequent CSV/database/PDF
connections. Switching clears the CSV dashboard cache. Each hosted analysis role
uses the selected model; local retains its existing role-specific models.

OpenAI uses official Chat Completions with max_completion_tokens. Claude uses
Anthropic's native Messages API, with system messages separated and response
text mapped into ARIA's common interface. Claude's adapter currently supports
text analysis only, not image/PDF vision inputs. Existing schema validation,
read-only SQL execution, cloud consent, and result-cell grounding remain active.
The startup picker is updated; the older Settings provider form is separate.

Validation: adapter and switch regression tests use mocks (no keys/network),
plus existing CSV regression tests and frontend production build. Live provider
billing/model access must be verified with your own account. Stronger models do
not independently verify metric meaning or guarantee correct answers.
