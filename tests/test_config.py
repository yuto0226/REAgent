from __future__ import annotations

from pathlib import Path

import pytest

from reagent.config import ConfigError, apply_provider_env, load, load_layers


def write_toml(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def mark_git_root(path: Path) -> None:
    write_toml(path / ".git" / "HEAD", "ref: refs/heads/main\n")


def test_load_uses_defaults_with_model_id_env(tmp_path):
    config = load(cwd=tmp_path, env={"MODEL_ID": "claude-sonnet-4"})

    assert config.llm.model == "claude-sonnet-4"
    assert config.llm.reasoning_effort == "medium"
    assert config.llm.thinking_budget_tokens == 8192
    assert config.llm.models.available == []
    assert config.agent.max_turns == 50
    assert config.skills.enabled is True
    assert config.skills.paths == []


def test_user_project_extra_and_env_layers_override_in_order(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    extra = tmp_path / "extra.toml"
    mark_git_root(project)
    write_toml(home / "config.toml", "[llm]\nmodel = 'from-home'\nreasoning_effort = 'low'\n")
    write_toml(project / ".reagent" / "config.toml", "[llm]\nmodel = 'from-project'\n")
    write_toml(extra, "[agent]\nmax_turns = 12\n")

    layers = load_layers(
        cwd=project,
        env={"REAGENT_HOME": str(home), "MODEL_ID": "from-env"},
        extra_config_paths=[extra],
    )

    assert [layer.name for layer in layers.layers] == [
        "defaults",
        "user",
        "project",
        "extra",
        "env",
    ]
    assert layers.config.llm.model == "from-env"
    assert layers.config.llm.reasoning_effort == "low"
    assert layers.config.agent.max_turns == 12


def test_project_config_is_found_from_git_root(tmp_path):
    repo = tmp_path / "repo"
    nested = repo / "a" / "b"
    write_toml(repo / ".git" / "HEAD", "ref: refs/heads/main\n")
    write_toml(repo / ".reagent" / "config.toml", "[llm]\nmodel = 'from-root'\n")
    nested.mkdir(parents=True)

    config = load(cwd=nested, env={})

    assert config.llm.model == "from-root"


def test_deep_merge_providers_and_mcp_servers(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    mark_git_root(project)
    write_toml(
        home / "config.toml",
        """
        [llm]
        model = "base-model"

        [providers.openai]
        key = "home-key"

        [mcp.servers.shell]
        command = "python"
        args = ["-m", "shell_server"]
        env = { TOKEN = "home" }
        """,
    )
    write_toml(
        project / ".reagent" / "config.toml",
        """
        [providers.openai]
        base_url = "https://example.test/v1"

        [mcp.servers.shell]
        enabled = false
        env = { MODE = "project" }

        [mcp.servers.fs]
        command = "fs-server"
        """,
    )

    config = load(cwd=project, env={"REAGENT_HOME": str(home)})

    assert config.providers["openai"].key == "home-key"
    assert config.providers["openai"].base_url == "https://example.test/v1"
    assert config.mcp.servers["shell"].command == "python"
    assert config.mcp.servers["shell"].args == ["-m", "shell_server"]
    assert config.mcp.servers["shell"].enabled is False
    assert config.mcp.servers["shell"].env == {"TOKEN": "home", "MODE": "project"}
    assert config.mcp.servers["fs"].enabled is True
    assert config.mcp.servers["fs"].transport == "stdio"
    assert config.mcp.servers["fs"].args == []
    assert config.mcp.servers["fs"].env == {}
    assert config.mcp.servers["fs"].headers == {}


def test_mcp_http_server_parses_url_and_headers(tmp_path):
    config_path = write_toml(
        tmp_path / "config.toml",
        """
        [mcp.servers.ida]
        transport = "http"
        url = "http://127.0.0.1:14542/mcp"
        headers = { Authorization = "Bearer abc" }
        """,
    )

    layers = load_layers(cwd=tmp_path, env={}, extra_config_paths=[config_path])

    server = layers.config.mcp.servers["ida"]
    assert server.transport == "http"
    assert server.url == "http://127.0.0.1:14542/mcp"
    assert server.headers == {"Authorization": "Bearer abc"}


def test_llm_models_available_lists_replace(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    mark_git_root(project)
    write_toml(
        home / "config.toml",
        """
        [llm]
        model = "chosen"

        [llm.models]
        available = ["a", "b"]
        """,
    )
    write_toml(
        project / ".reagent" / "config.toml",
        """
        [llm.models]
        available = ["c"]
        """,
    )

    config = load(cwd=project, env={"REAGENT_HOME": str(home)})

    assert config.llm.models.available == ["c"]


def test_load_requires_llm_model(tmp_path):
    with pytest.raises(ConfigError, match="llm.model"):
        load(cwd=tmp_path, env={})


def test_apply_provider_env_maps_known_keys_without_overwriting(tmp_path):
    home = tmp_path / "home"
    write_toml(
        home / "config.toml",
        """
        [providers.openai]
        key = "config-openai"

        [providers.anthropic]
        key = "config-anthropic"
        """,
    )
    config = load(
        cwd=tmp_path,
        env={"MODEL_ID": "model", "REAGENT_HOME": str(home)},
    )
    env = {"OPENAI_API_KEY": "existing"}

    apply_provider_env(config, env)

    assert env["OPENAI_API_KEY"] == "existing"
    assert env["ANTHROPIC_API_KEY"] == "config-anthropic"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[unknown]\nvalue = 1\n", "unknown"),
        ("[llm]\nmodel = 123\n", "llm.model"),
        ("[agent]\nmax_turns = 0\n", "agent.max_turns"),
        ("[llm.models]\navailable = ['ok', 3]\n", "llm.models.available"),
        ("[mcp.servers.bad]\ncommand = 'run'\nunexpected = true\n", "mcp.servers.bad.unexpected"),
        ("[mcp.servers.bad]\ntransport = 'stdio'\n", "mcp.servers.bad.command"),
        (
            "[mcp.servers.bad]\ntransport = 'http'\nurl = 'http://x'\nheaders = { Authorization = 1 }\n",
            "mcp.servers.bad.headers.Authorization",
        ),
    ],
)
def test_invalid_config_reports_path_and_key(tmp_path, content, message):
    config_path = write_toml(tmp_path / "config.toml", content)

    with pytest.raises(ConfigError) as exc_info:
        load_layers(cwd=tmp_path, env={}, extra_config_paths=[config_path])

    assert str(config_path) in str(exc_info.value)
    assert message in str(exc_info.value)
