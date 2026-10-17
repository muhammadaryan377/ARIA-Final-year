import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
from services.hosted_clients import ClaudeClient, OpenAIClient


class ProviderTests(unittest.TestCase):
    def test_openai_reasoning_parameters(self):
        raw = Mock(api_key="test")
        OpenAIClient(raw).create(model="example", messages=[], max_tokens=256, temperature=0)
        args = raw.chat.completions.create.call_args.kwargs
        self.assertEqual(args["max_completion_tokens"], 256)
        self.assertNotIn("temperature", args)
        self.assertNotIn("max_tokens", args)

    def test_claude_native_translation(self):
        client = ClaudeClient("test")
        client._request = Mock(return_value={"content": [{"type": "text", "text": '{"ok":true}'}]})
        result = client.create(model="test-model", messages=[{"role": "system", "content": "Return JSON"},
            {"role": "user", "content": "Hello"}], max_tokens=800, response_format={"type": "json_object"})
        body = client._request.call_args.kwargs["json"]
        self.assertEqual(body["system"], "Return JSON")
        self.assertEqual(body["messages"], [{"role": "user", "content": "Hello"}])
        self.assertNotIn("response_format", body)
        self.assertEqual(result.choices[0].message.content, '{"ok":true}')

    def test_claude_model_listing(self):
        client = ClaudeClient("test")
        client._request = Mock(return_value={"data": [{"id": "example-model"}]})
        self.assertEqual(client.models.list().data[0].id, "example-model")

    def switch_function(self, session, llm):
        source = ast.parse(Path("api/connect_routes.py").read_text())
        node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "switch_provider")
        node.decorator_list = []
        node.args.defaults = []
        for arg in node.args.args: arg.annotation = None
        class HTTPError(Exception):
            def __init__(self, status_code, detail): super().__init__(detail)
        create = Mock(return_value=llm)
        scope = dict(get_session=lambda uid: session, save_session=Mock(), create_provider=create,
            PROVIDER_NAMES={"openai", "local"}, HOSTED_PROVIDERS={"openai"}, HTTPException=HTTPError)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "switch", "exec"), scope)
        return scope["switch_provider"], create

    def test_switch_saves_model_and_invalidates_dashboard(self):
        session = {"api_keys": {"openai": "saved", "cloud": "different"}, "csv_llm_dashboard": {"old": True}}
        llm = Mock(models={"semi": "chosen"})
        llm.chat.return_value = "OK"
        switch, create = self.switch_function(session, llm)
        fake_config = NS(resolve_provider_key=lambda name, keys: keys.get(name))
        with patch.dict("sys.modules", {"core.config": fake_config}):
            switch(NS(provider="openai", api_key=None, base_url=None, model="chosen"), {"user_id": 7})
        self.assertEqual(create.call_args.kwargs["api_key"], "saved")
        self.assertTrue(all(m == "chosen" for m in create.call_args.kwargs["models"].values()))
        self.assertEqual(session["provider_models"]["openai"], {"semi": "chosen"})
        self.assertNotIn("csv_llm_dashboard", session)

    def test_failed_probe_preserves_existing_session(self):
        session = {"provider_name": "local"}
        llm = Mock(); llm.chat.side_effect = RuntimeError("bad model")
        switch, _ = self.switch_function(session, llm)
        with patch.dict("sys.modules", {"core.config": NS(resolve_provider_key=lambda *a: None)}):
            with self.assertRaisesRegex(Exception, "check failed"):
                switch(NS(provider="openai", api_key="test", base_url=None, model="bad"), {"user_id": 7})
        self.assertEqual(session, {"provider_name": "local"})

if __name__ == "__main__": unittest.main()
