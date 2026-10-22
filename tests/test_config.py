import pytest

from llm_gateway import ConfigError, GatewayConfig, Route, load_config


def test_route_parse_splits_on_first_colon_only():
    r = Route.parse("OpenRouter:meta-llama/llama-3.1-8b-instruct:free")
    assert (r.provider, r.model) == ("openrouter", "meta-llama/llama-3.1-8b-instruct:free")
    assert r.key == "openrouter:meta-llama/llama-3.1-8b-instruct:free"


@pytest.mark.parametrize("bad", ["deepseek", ":model", "provider:", ""])
def test_route_parse_rejects_malformed(bad):
    with pytest.raises(ConfigError):
        Route.parse(bad)


def test_from_dict_preserves_route_order_and_per_route_timeout():
    cfg = GatewayConfig.from_dict(
        {
            "gateway": {"timeout_s": 20},
            "jobs": {"Vision": ["deepseek:deepseek-chat", {"route": "gemini:flash", "timeout_s": 5}]},
        }
    )
    routes = cfg.routes_for("vision")
    assert [r.key for r in routes] == ["deepseek:deepseek-chat", "gemini:flash"]
    assert [r.timeout_s for r in routes] == [20.0, 5.0]


def test_unknown_job_falls_back_to_default_then_errors():
    cfg = GatewayConfig.from_dict({"jobs": {"default": ["a:x"]}})
    assert cfg.routes_for("anything")[0].key == "a:x"
    with pytest.raises(ConfigError, match="LLM_ROUTE_QC"):
        GatewayConfig().routes_for("qc")


def test_custom_provider_needs_base_url():
    with pytest.raises(ConfigError):
        GatewayConfig.from_dict({"providers": {"acme": {}}})
    cfg = GatewayConfig.from_dict({"providers": {"acme": {"base_url": "http://x"}}})
    assert cfg.providers["acme"].api_key_env == "ACME_API_KEY"


def test_env_route_override_replaces_chain():
    base = GatewayConfig.from_dict({"jobs": {"fast": ["a:x"]}})
    cfg = base.with_env({"LLM_ROUTE_FAST": "b:y, c:z", "LLM_ROUTE_QC": "d:w"})
    assert [r.key for r in cfg.routes_for("fast")] == ["b:y", "c:z"]
    assert [r.key for r in cfg.routes_for("qc")] == ["d:w"]
    assert [r.key for r in base.routes_for("fast")] == ["a:x"]  # original untouched


def test_legacy_flag_routes_every_job_to_legacy_route():
    base = GatewayConfig.from_dict({"jobs": {"fast": ["a:x"], "vision": ["b:y", "c:z"]}})
    cfg = base.with_env({"LLM_GATEWAY_LEGACY": "1", "LLM_GATEWAY_LEGACY_ROUTE": "openai:gpt-legacy"})
    for job in ("fast", "vision", "never-configured"):
        assert [r.key for r in cfg.routes_for(job)] == ["openai:gpt-legacy"]
    off = cfg.with_env({"LLM_GATEWAY_LEGACY": "0"})
    assert [r.key for r in off.routes_for("vision")] == ["b:y", "c:z"]


def test_legacy_flag_without_route_is_a_config_error():
    with pytest.raises(ConfigError):
        GatewayConfig().with_env({"LLM_GATEWAY_LEGACY": "1"})


def test_load_config_from_toml_plus_env(tmp_path):
    path = tmp_path / "gw.toml"
    path.write_text(
        '[gateway]\nlegacy_route = "openai:old"\ndeadline_s = 12\n'
        '[jobs]\nqc = ["deepseek:deepseek-chat"]\n'
        '[prices."deepseek:deepseek-chat"]\ninput = 1.0\noutput = 2.0\n'
    )
    cfg = load_config(path, env={"LLM_GATEWAY_LEDGER": "x.sqlite3"})
    assert cfg.deadline_s == 12 and cfg.ledger_path == "x.sqlite3"
    assert cfg.prices["deepseek:deepseek-chat"]["output"] == 2.0
    assert load_config(path, env={"LLM_GATEWAY_LEGACY": "yes"}).routes_for("qc")[0].key == "openai:old"


def test_example_config_parses():
    from pathlib import Path

    cfg = load_config(Path(__file__).parents[1] / "examples" / "llm-gateway.example.toml", env={})
    assert set(cfg.jobs) == {"fast", "qc", "prompt", "vision", "research"}
    assert cfg.routes_for("research")[0].timeout_s == 120
