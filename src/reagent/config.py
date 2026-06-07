from __future__ import annotations

import copy
import os
import tomllib
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "llm": {
        "model": "",
        "reasoning_effort": "medium",
        "thinking_budget_tokens": 8192,
        "models": {"available": []},
    },
    "agent": {"max_turns": 50},
    "providers": {},
    "mcp": {"servers": {}},
    "skills": {"enabled": True, "paths": []},
}

_PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


class ConfigError(ValueError):
    """Raised when authored runtime config is invalid or incomplete."""


@dataclass(frozen=True)
class LLMModelsConfig:
    available: list[str]


@dataclass(frozen=True)
class LLMConfig:
    model: str
    reasoning_effort: str
    thinking_budget_tokens: int
    models: LLMModelsConfig


@dataclass(frozen=True)
class AgentConfig:
    max_turns: int


@dataclass(frozen=True)
class ProviderConfig:
    key: str | None = None
    base_url: str | None = None


@dataclass(frozen=True)
class MCPServerConfig:
    enabled: bool
    transport: str
    command: str | None
    args: list[str]
    env: dict[str, str]


@dataclass(frozen=True)
class MCPConfig:
    servers: dict[str, MCPServerConfig]


@dataclass(frozen=True)
class SkillsConfig:
    enabled: bool
    paths: list[str]


@dataclass(frozen=True)
class Config:
    llm: LLMConfig
    agent: AgentConfig
    providers: dict[str, ProviderConfig]
    mcp: MCPConfig
    skills: SkillsConfig


@dataclass(frozen=True)
class Layer:
    name: str
    path: Path | None
    data: dict[str, Any]


@dataclass(frozen=True)
class Layers:
    layers: list[Layer]
    merged: dict[str, Any]
    config: Config


def load(cwd: str | Path | None = None, env: Mapping[str, str] | None = None) -> Config:
    """Load and resolve runtime config for the current process."""
    layers = load_layers(cwd=cwd, env=env)
    if not layers.config.llm.model:
        raise ConfigError("Missing required config key llm.model; set it in config.toml or MODEL_ID.")
    return layers.config


def apply_provider_env(config: Config, env: MutableMapping[str, str] | None = None) -> None:
    """Expose configured provider keys through LiteLLM's standard env lookup."""
    target = os.environ if env is None else env
    for provider_name, provider in config.providers.items():
        env_name = _PROVIDER_KEY_ENV.get(provider_name)
        if env_name is not None and provider.key:
            target.setdefault(env_name, provider.key)


def load_layers(
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    extra_config_paths: Sequence[str | Path] = (),
) -> Layers:
    """Read config sources in precedence order without hiding their origins."""
    effective_env = os.environ if env is None else env
    base_dir = Path.cwd() if cwd is None else Path(cwd)

    layer_list = [Layer("defaults", None, copy.deepcopy(DEFAULTS))]
    layer_list.extend(_file_layers(base_dir, effective_env, extra_config_paths))
    if model_id := effective_env.get("MODEL_ID"):
        layer_list.append(Layer("env", None, {"llm": {"model": model_id}}))

    merged: dict[str, Any] = {}
    for layer in layer_list:
        _deep_merge(merged, layer.data)

    server_paths = _server_source_paths(layer_list)
    return Layers(layers=layer_list, merged=merged, config=_to_config(merged, server_paths=server_paths))


def _file_layers(
    cwd: Path,
    env: Mapping[str, str],
    extra_config_paths: Sequence[str | Path],
) -> list[Layer]:
    layers: list[Layer] = []
    reagent_home = Path(env.get("REAGENT_HOME", "~/.reagent")).expanduser()
    user_config = reagent_home / "config.toml"
    project_config = _project_config_path(cwd)

    if user_config.exists():
        layers.append(Layer("user", user_config, _read_config(user_config)))
    if project_config.exists():
        layers.append(Layer("project", project_config, _read_config(project_config)))
    for extra_path in extra_config_paths:
        path = Path(extra_path)
        if path.exists():
            layers.append(Layer("extra", path, _read_config(path)))
    return layers


def _project_config_path(cwd: Path) -> Path:
    root = _find_git_root(cwd)
    base_dir = root if root is not None else cwd
    return base_dir / ".reagent" / "config.toml"


def _find_git_root(cwd: Path) -> Path | None:
    current = cwd.resolve()
    for path in (current, *current.parents):
        if _is_git_root(path):
            return path
    return None


def _is_git_root(path: Path) -> bool:
    git_path = path / ".git"
    if git_path.is_file():
        return True
    return git_path.is_dir() and (git_path / "HEAD").exists()


def _read_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except ValueError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc

    _validate_authored_config(data, path)
    return data


def _validate_authored_config(data: dict[str, Any], path: Path) -> None:
    _require_table(data, str(path))
    _reject_unknown(data, {"llm", "agent", "providers", "mcp", "skills"}, path, "")

    if "llm" in data:
        _validate_llm(data["llm"], path)
    if "agent" in data:
        _validate_agent(data["agent"], path)
    if "providers" in data:
        _validate_providers(data["providers"], path)
    if "mcp" in data:
        _validate_mcp(data["mcp"], path)
    if "skills" in data:
        _validate_skills(data["skills"], path)


def _validate_llm(value: Any, path: Path) -> None:
    table = _require_table(value, _key(path, "llm"))
    _reject_unknown(table, {"model", "reasoning_effort", "thinking_budget_tokens", "models"}, path, "llm")
    _optional_string(table, "model", path, "llm.model")
    _optional_string(table, "reasoning_effort", path, "llm.reasoning_effort")
    _optional_positive_int(table, "thinking_budget_tokens", path, "llm.thinking_budget_tokens")
    if "models" in table:
        models = _require_table(table["models"], _key(path, "llm.models"))
        _reject_unknown(models, {"available"}, path, "llm.models")
        _optional_string_list(models, "available", path, "llm.models.available")


def _validate_agent(value: Any, path: Path) -> None:
    table = _require_table(value, _key(path, "agent"))
    _reject_unknown(table, {"max_turns"}, path, "agent")
    _optional_positive_int(table, "max_turns", path, "agent.max_turns")


def _validate_providers(value: Any, path: Path) -> None:
    providers = _require_table(value, _key(path, "providers"))
    for name, provider in providers.items():
        key_prefix = f"providers.{name}"
        table = _require_table(provider, _key(path, key_prefix))
        _reject_unknown(table, {"key", "base_url"}, path, key_prefix)
        _optional_string(table, "key", path, f"{key_prefix}.key")
        _optional_string(table, "base_url", path, f"{key_prefix}.base_url")


def _validate_mcp(value: Any, path: Path) -> None:
    table = _require_table(value, _key(path, "mcp"))
    _reject_unknown(table, {"servers"}, path, "mcp")
    if "servers" not in table:
        return

    servers = _require_table(table["servers"], _key(path, "mcp.servers"))
    for name, server in servers.items():
        key_prefix = f"mcp.servers.{name}"
        server_table = _require_table(server, _key(path, key_prefix))
        _reject_unknown(server_table, {"enabled", "transport", "command", "args", "env"}, path, key_prefix)
        _optional_bool(server_table, "enabled", path, f"{key_prefix}.enabled")
        _optional_string(server_table, "transport", path, f"{key_prefix}.transport")
        _optional_string(server_table, "command", path, f"{key_prefix}.command")
        _optional_string_list(server_table, "args", path, f"{key_prefix}.args")
        if "env" in server_table:
            env_table = _require_table(server_table["env"], _key(path, f"{key_prefix}.env"))
            for env_key, env_value in env_table.items():
                if not isinstance(env_value, str):
                    raise ConfigError(f"{path}: {key_prefix}.env.{env_key} must be a string")


def _validate_skills(value: Any, path: Path) -> None:
    table = _require_table(value, _key(path, "skills"))
    _reject_unknown(table, {"enabled", "paths"}, path, "skills")
    _optional_bool(table, "enabled", path, "skills.enabled")
    _optional_string_list(table, "paths", path, "skills.paths")


def _server_source_paths(layers: list[Layer]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for layer in layers:
        servers = layer.data.get("mcp", {}).get("servers", {})
        if not isinstance(servers, dict):
            continue
        for name in servers:
            paths[name] = str(layer.path) if layer.path is not None else "<merged>"
    return paths


def _to_config(data: dict[str, Any], server_paths: Mapping[str, str]) -> Config:
    llm = data["llm"]
    agent = data["agent"]
    providers = data["providers"]
    mcp = data["mcp"]
    skills = data["skills"]

    provider_configs = {
        name: ProviderConfig(key=value.get("key"), base_url=value.get("base_url")) for name, value in providers.items()
    }
    server_configs = {
        name: _to_mcp_server(name, value, server_paths.get(name, "<merged>")) for name, value in mcp["servers"].items()
    }
    return Config(
        llm=LLMConfig(
            model=llm["model"],
            reasoning_effort=llm["reasoning_effort"],
            thinking_budget_tokens=llm["thinking_budget_tokens"],
            models=LLMModelsConfig(available=list(llm["models"]["available"])),
        ),
        agent=AgentConfig(max_turns=agent["max_turns"]),
        providers=provider_configs,
        mcp=MCPConfig(servers=server_configs),
        skills=SkillsConfig(enabled=skills["enabled"], paths=list(skills["paths"])),
    )


def _to_mcp_server(name: str, data: dict[str, Any], path: str) -> MCPServerConfig:
    transport = data.get("transport", "stdio")
    command = data.get("command")
    if transport == "stdio" and not command:
        raise ConfigError(f"{path}: mcp.servers.{name}.command is required when transport is stdio")
    return MCPServerConfig(
        enabled=data.get("enabled", True),
        transport=transport,
        command=command,
        args=list(data.get("args", [])),
        env=dict(data.get("env", {})),
    )


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _reject_unknown(table: Mapping[str, Any], allowed: set[str], path: Path, prefix: str) -> None:
    for key in table:
        if key not in allowed:
            full_key = f"{prefix}.{key}" if prefix else key
            raise ConfigError(f"{path}: unknown config key {full_key}")


def _require_table(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a table")
    return value


def _optional_string(table: Mapping[str, Any], name: str, path: Path, key: str) -> None:
    if name in table and not isinstance(table[name], str):
        raise ConfigError(f"{path}: {key} must be a string")


def _optional_string_list(table: Mapping[str, Any], name: str, path: Path, key: str) -> None:
    if name not in table:
        return
    values = table[name]
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        raise ConfigError(f"{path}: {key} must be a list of strings")


def _optional_bool(table: Mapping[str, Any], name: str, path: Path, key: str) -> None:
    if name in table and not isinstance(table[name], bool):
        raise ConfigError(f"{path}: {key} must be a boolean")


def _optional_positive_int(table: Mapping[str, Any], name: str, path: Path, key: str) -> None:
    if name not in table:
        return
    value = table[name]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{path}: {key} must be a positive integer")


def _key(path: Path, key: str) -> str:
    return f"{path}: {key}"
