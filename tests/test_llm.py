"""Tests for provider selection without making network requests."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.agent import build_tools
from src.llm import LLMConfig, get_default_llm_config, get_provider_presets
from src import tools as paper_tools


class LLMConfigTests(unittest.TestCase):
    def test_provider_presets_cover_remote_local_and_custom_endpoints(self):
        presets = get_provider_presets()

        self.assertEqual(
            {"deepseek", "openai", "qwen", "ollama", "custom"},
            set(presets),
        )
        self.assertFalse(presets["ollama"].requires_api_key)

    def test_deepseek_environment_remains_backward_compatible(self):
        env = {
            "DEEPSEEK_API_KEY": "deepseek-secret",
            "DEEPSEEK_BASE_URL": "https://deepseek.example/v1",
            "DEEPSEEK_MODEL": "deepseek-test",
        }
        with patch.dict(os.environ, env, clear=True):
            config = get_default_llm_config()

        self.assertEqual("deepseek", config.provider)
        self.assertEqual("deepseek-test", config.model)
        self.assertEqual("https://deepseek.example/v1", config.base_url)
        self.assertEqual("deepseek-secret", config.api_key)
        self.assertTrue(config.is_ready)

    def test_generic_environment_can_select_an_openai_compatible_provider(self):
        env = {
            "LLM_PROVIDER": "custom",
            "LLM_MODEL": "my-model",
            "LLM_BASE_URL": "http://127.0.0.1:9000/v1",
            "LLM_TEMPERATURE": "0.7",
        }
        with patch.dict(os.environ, env, clear=True):
            config = get_default_llm_config()

        self.assertEqual("custom", config.provider)
        self.assertEqual("my-model", config.model)
        self.assertEqual("http://127.0.0.1:9000/v1", config.base_url)
        self.assertEqual(0.7, config.temperature)
        self.assertTrue(config.is_ready)

    def test_remote_provider_requires_a_key_but_ollama_does_not(self):
        openai_config = LLMConfig(
            provider="openai",
            model="gpt-test",
            base_url="https://api.openai.com/v1",
        )
        ollama_config = LLMConfig(
            provider="ollama",
            model="qwen2.5:7b",
            base_url="http://localhost:11434/v1",
        )

        self.assertFalse(openai_config.is_ready)
        self.assertTrue(ollama_config.is_ready)
        self.assertEqual("local-no-key", ollama_config.effective_api_key)

    def test_agent_tool_schema_does_not_expose_internal_llm_config(self):
        config = LLMConfig(
            provider="ollama",
            model="qwen2.5:7b",
            base_url="http://localhost:11434/v1",
        )
        tools = build_tools(config)
        schemas = {item.name: item.args for item in tools}

        self.assertEqual(8, len(tools))
        self.assertNotIn("llm_config", schemas["summarize_papers"])
        self.assertNotIn("llm_config", schemas["deep_read"])
        self.assertNotIn("llm_config", schemas["chat_with_paper"])

    def test_all_ai_paper_tools_use_the_selected_model(self):
        config = LLMConfig(
            provider="custom",
            model="selected-model",
            base_url="http://127.0.0.1:9000/v1",
            temperature=0.7,
        )
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="generated answer"))
            ]
        )

        calls = [
            lambda: paper_tools.summarize_papers("1234.5678", llm_config=config),
            lambda: paper_tools.deep_read("1234.5678", llm_config=config),
            lambda: paper_tools.chat_with_paper(
                "1234.5678",
                "What is the contribution?",
                llm_config=config,
            ),
        ]
        with (
            patch.object(paper_tools, "_get_text", return_value="paper text"),
            patch.object(paper_tools, "create_openai_client", return_value=fake_client),
        ):
            for run_tool in calls:
                with self.subTest(tool=run_tool):
                    fake_client.chat.completions.create.reset_mock()
                    run_tool()
                    request = fake_client.chat.completions.create.call_args.kwargs
                    self.assertEqual("selected-model", request["model"])
                    self.assertEqual(0.7, request["temperature"])


if __name__ == "__main__":
    unittest.main()
