"""Unified configuration and adapters for OpenAI-compatible LLM providers."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI


load_dotenv()


@dataclass(frozen=True)
class ProviderPreset:
    """Defaults and credential conventions for a supported provider."""

    id: str
    label: str
    base_url: str
    default_model: str
    api_key_env: str
    base_url_env: str
    model_env: str
    requires_api_key: bool = True
    description: str = ""


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "deepseek": ProviderPreset(
        id="deepseek",
        label="DeepSeek",
        base_url="https://api.deepseek.com",
        default_model="deepseek-v4-pro",
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_BASE_URL",
        model_env="DEEPSEEK_MODEL",
        description="DeepSeek 官方 OpenAI-compatible 接口",
    ),
    "openai": ProviderPreset(
        id="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4.1-mini",
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        model_env="OPENAI_MODEL",
        description="OpenAI 官方 API",
    ),
    "qwen": ProviderPreset(
        id="qwen",
        label="Qwen / 通义千问",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        api_key_env="DASHSCOPE_API_KEY",
        base_url_env="QWEN_BASE_URL",
        model_env="QWEN_MODEL",
        description="阿里云百炼 OpenAI-compatible 接口",
    ),
    "ollama": ProviderPreset(
        id="ollama",
        label="Ollama（本地）",
        base_url="http://localhost:11434/v1",
        default_model="qwen2.5:7b",
        api_key_env="OLLAMA_API_KEY",
        base_url_env="OLLAMA_BASE_URL",
        model_env="OLLAMA_MODEL",
        requires_api_key=False,
        description="完全在本机运行的 Ollama OpenAI-compatible 接口",
    ),
    "custom": ProviderPreset(
        id="custom",
        label="自定义接口",
        base_url="http://localhost:8000/v1",
        default_model="",
        api_key_env="LLM_API_KEY",
        base_url_env="LLM_BASE_URL",
        model_env="LLM_MODEL",
        requires_api_key=False,
        description="任意 OpenAI-compatible 服务",
    ),
}


def get_provider_presets() -> dict[str, ProviderPreset]:
    """Return a copy so UI code cannot mutate global provider defaults."""

    return dict(PROVIDER_PRESETS)


def get_provider_preset(provider: str) -> ProviderPreset:
    """Resolve a provider id, falling back to DeepSeek for compatibility."""

    return PROVIDER_PRESETS.get(provider.strip().lower(), PROVIDER_PRESETS["deepseek"])


def _read_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


@dataclass(frozen=True)
class LLMConfig:
    """The single model contract used by the UI, agent, and paper tools."""

    provider: str
    model: str
    base_url: str
    api_key: str = ""
    temperature: float = 0.3

    @property
    def preset(self) -> ProviderPreset:
        return get_provider_preset(self.provider)

    @property
    def provider_label(self) -> str:
        return self.preset.label

    @property
    def effective_api_key(self) -> str:
        # OpenAI-compatible local servers still expect a non-empty client value.
        return self.api_key or "local-no-key"

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if self.provider not in PROVIDER_PRESETS:
            errors.append("未知的模型供应商。")
        if not self.model.strip():
            errors.append("模型名称不能为空。")
        if not self.base_url.strip():
            errors.append("API Base URL 不能为空。")
        if self.preset.requires_api_key and not self.api_key.strip():
            errors.append(f"{self.provider_label} 需要 API Key。")
        if not 0 <= self.temperature <= 2:
            errors.append("Temperature 必须在 0 到 2 之间。")
        return errors

    @property
    def is_ready(self) -> bool:
        return not self.validation_errors()

    def require_valid(self) -> "LLMConfig":
        errors = self.validation_errors()
        if errors:
            raise ValueError(" ".join(errors))
        return self


def get_default_llm_config() -> LLMConfig:
    """Build configuration from generic env vars with provider-specific fallbacks."""

    provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
    if provider not in PROVIDER_PRESETS:
        provider = "custom"
    preset = get_provider_preset(provider)

    try:
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    except ValueError:
        temperature = 0.3

    return LLMConfig(
        provider=provider,
        model=_read_env("LLM_MODEL", preset.model_env) or preset.default_model,
        base_url=_read_env("LLM_BASE_URL", preset.base_url_env) or preset.base_url,
        api_key=_read_env("LLM_API_KEY", preset.api_key_env),
        temperature=temperature,
    )


def create_chat_model(config: LLMConfig) -> ChatOpenAI:
    """Create the LangChain model used for Agent reasoning and tool calling."""

    config.require_valid()
    return ChatOpenAI(
        model=config.model,
        temperature=config.temperature,
        api_key=config.effective_api_key,
        base_url=config.base_url,
    )


def create_openai_client(config: LLMConfig) -> OpenAI:
    """Create the SDK client used by summarize, deep-read, and paper chat tools."""

    config.require_valid()
    return OpenAI(
        api_key=config.effective_api_key,
        base_url=config.base_url,
    )
