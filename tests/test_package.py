"""Checks on the published files: registry metadata, copy rules and Comfy Registry review rules."""

from __future__ import annotations

import ast
import json
import re
import struct
from pathlib import Path

import pytest

from empiriolabs_comfy import __version__

PACK_DIR = Path(__file__).resolve().parent.parent
EM_DASH = "\u2014"
SOURCE_FILES = sorted([PACK_DIR / "__init__.py", *(PACK_DIR / "empiriolabs_comfy").glob("*.py")])
USER_FACING_FILES = [
    PACK_DIR / "README.md",
    PACK_DIR / "pyproject.toml",
    PACK_DIR / "config.example.ini",
    *sorted((PACK_DIR / "example_workflows").glob("*.json")),
    *SOURCE_FILES,
]
OUR_NODE_TYPES = {"EmpirioLabsGenerateImage", "EmpirioLabsGenerateVideo", "EmpirioLabsChat", "EmpirioLabsTextToSpeech"}


def _pyproject() -> str:
    return (PACK_DIR / "pyproject.toml").read_text(encoding="utf-8")


def _toml_value(key: str) -> str:
    match = re.search(rf'^{re.escape(key)}\s*=\s*"([^"]*)"', _pyproject(), re.MULTILINE)
    assert match, key
    return match.group(1)


def test_registry_metadata():
    assert _toml_value("name") == "empiriolabs"
    assert re.fullmatch(r"\d+\.\d+\.\d+", _toml_value("version"))
    assert _toml_value("PublisherId") == "empiriolabs"
    assert _toml_value("DisplayName") == "EmpirioLabs AI"
    assert _toml_value("Repository") == "https://github.com/EmpirioLabs-AI/comfyui-empiriolabs"
    assert _toml_value("requires-comfyui").startswith(">=")
    assert re.search(r"^dependencies = \[\]", _pyproject(), re.MULTILINE)


def test_icon_is_a_square_png_no_larger_than_400px():
    icon_url = _toml_value("Icon")
    prefix = "https://raw.githubusercontent.com/EmpirioLabs-AI/comfyui-empiriolabs/main/"
    assert icon_url.startswith(prefix)
    data = (PACK_DIR / icon_url[len(prefix):]).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert width == height <= 400


def test_version_comes_from_pyproject():
    assert __version__ == _toml_value("version")


@pytest.mark.parametrize("path", USER_FACING_FILES, ids=lambda p: p.name)
def test_no_em_dash_in_published_text(path):
    assert EM_DASH not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_source_avoids_what_the_registry_review_flags(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"eval", "exec", "compile", "__import__"}, f"{path.name}: {node.func.id}()"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                assert name.split(".")[0] not in {"subprocess", "pip", "ctypes", "pickle", "marshal"}, f"{path.name}: {name}"
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os":
            assert node.attr not in {"system", "popen", "spawnl", "execv", "execl"}, f"{path.name}: os.{node.attr}"


def test_api_host_is_fixed():
    text = (PACK_DIR / "empiriolabs_comfy" / "client.py").read_text(encoding="utf-8")
    assert 'API_BASE = "https://api.empiriolabs.ai"' in text
    # No setting or environment variable can point the requests at another host.
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "os" not in imported and "getenv" not in text


def test_ignore_files():
    gitignore = (PACK_DIR / ".gitignore").read_text(encoding="utf-8").split()
    assert "config.ini" in gitignore
    comfyignore = (PACK_DIR / ".comfyignore").read_text(encoding="utf-8").split()
    for entry in ("tests/", "scripts/", ".github/", "config.ini"):
        assert entry in comfyignore
    assert "example_workflows/" not in comfyignore


def test_publish_workflow():
    text = (PACK_DIR / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")
    assert "Comfy-Org/publish-node-action@main" in text
    assert "secrets.REGISTRY_ACCESS_TOKEN" in text
    assert '- "pyproject.toml"' in text
    assert "github.repository_owner == 'EmpirioLabs-AI'" in text


def _workflow_files():
    return sorted((PACK_DIR / "example_workflows").glob("*.json"))


def test_there_are_image_and_video_example_workflows():
    names = [p.name for p in _workflow_files()]
    assert any("image" in n for n in names) and any("video" in n for n in names)


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_example_workflows_use_our_nodes_and_hold_no_key(path):
    text = path.read_text(encoding="utf-8")
    workflow = json.loads(text)
    types = {node["type"] for node in workflow["nodes"]}
    assert types & OUR_NODE_TYPES
    assert not re.search(r"sk-empiriolabs-[A-Za-z0-9]", text)
    for node in workflow["nodes"]:
        if node["type"] in OUR_NODE_TYPES:
            for value in node.get("widgets_values", []):
                assert not (isinstance(value, str) and value.startswith("sk-"))
