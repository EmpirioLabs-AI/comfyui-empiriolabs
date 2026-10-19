from __future__ import annotations

import pytest

from empiriolabs_comfy import config
from fakes import TEST_KEY


def test_environment_variable_wins_over_config_file(tmp_path):
    path = tmp_path / "config.ini"
    path.write_text("[empiriolabs]\napi_key = sk-empiriolabs-from-file\n", encoding="utf-8")
    assert config.read_api_key({config.ENV_VAR: f"  {TEST_KEY}\n"}, path) == TEST_KEY


def test_config_file_is_used_when_the_environment_has_no_key(tmp_path):
    path = tmp_path / "config.ini"
    path.write_text("# comment\n[empiriolabs]\napi_key = sk-empiriolabs-from-file\n", encoding="utf-8")
    assert config.read_api_key({config.ENV_VAR: "   "}, path) == "sk-empiriolabs-from-file"


def test_config_file_with_a_byte_order_mark_is_read(tmp_path):
    path = tmp_path / "config.ini"
    path.write_bytes("[empiriolabs]\napi_key = sk-empiriolabs-bom\n".encode("utf-8-sig"))
    assert config.read_api_key({}, path) == "sk-empiriolabs-bom"


def test_missing_key_returns_none(tmp_path):
    assert config.read_api_key({}, tmp_path / "config.ini") is None
    empty = tmp_path / "empty.ini"
    empty.write_text("[empiriolabs]\napi_key =\n", encoding="utf-8")
    assert config.read_api_key({}, empty) is None


def test_unreadable_config_file_never_echoes_its_contents(tmp_path):
    path = tmp_path / "config.ini"
    path.write_text(f"{TEST_KEY}\n", encoding="utf-8")  # the key without a section header
    with pytest.raises(config.ApiKeyError) as excinfo:
        config.read_api_key({}, path)
    assert TEST_KEY not in str(excinfo.value)
    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__
    assert "[empiriolabs]" in str(excinfo.value)


def test_require_api_key_explains_both_options(monkeypatch, tmp_path):
    monkeypatch.delenv(config.ENV_VAR, raising=False)
    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "config.ini")
    with pytest.raises(config.ApiKeyError) as excinfo:
        config.require_api_key()
    message = str(excinfo.value)
    assert config.ENV_VAR in message
    assert "config.ini" in message
    assert "https://platform.empiriolabs.ai/dashboard/api-keys" in message


def test_require_api_key_reads_the_environment(monkeypatch):
    monkeypatch.setenv(config.ENV_VAR, TEST_KEY)
    assert config.require_api_key() == TEST_KEY


def test_config_file_lives_in_the_pack_folder():
    assert config.config_path().parent == config.PACK_DIR
    assert (config.PACK_DIR / "config.example.ini").is_file()
