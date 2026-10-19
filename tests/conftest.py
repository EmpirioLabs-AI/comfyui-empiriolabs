"""Shared test setup.

The node and media tests need ComfyUI's own `comfy_api` package, so they run inside a
ComfyUI Python environment: point COMFYUI_PATH at a ComfyUI checkout, or keep this pack
in ComfyUI/custom_nodes/ and it is found automatically. The client, catalog, key and
package tests run without ComfyUI.

No test talks to the network: every API call and download goes through FakeAPI.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
PACK_DIR = TESTS_DIR.parent


def _find_comfyui() -> Path | None:
    candidates = []
    if os.environ.get("COMFYUI_PATH"):
        candidates.append(Path(os.environ["COMFYUI_PATH"]))
    candidates.append(PACK_DIR.parent.parent)  # ComfyUI/custom_nodes/<this pack>
    for candidate in candidates:
        if (candidate / "comfy_api" / "latest" / "__init__.py").is_file():
            return candidate.resolve()
    return None


COMFYUI_DIR = _find_comfyui()
for path in (str(PACK_DIR), str(TESTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
if COMFYUI_DIR is not None and str(COMFYUI_DIR) not in sys.path:
    sys.path.append(str(COMFYUI_DIR))

from empiriolabs_comfy import catalog, client, config  # noqa: E402
from fakes import TEST_KEY, FakeAPI, FakeInterrupt, fake_catalog  # noqa: E402


@pytest.fixture
def live_catalog():
    """The fake catalog, loaded as if it came from /v1/models."""
    saved = catalog._state
    catalog._state = catalog._State()
    catalog._store(catalog.parse_catalog(fake_catalog()))
    catalog._state.startup_attempted = True
    yield catalog
    catalog._state = saved


@pytest.fixture
def no_catalog():
    """A catalog that never loaded, so the fallback is in use and no retry is due."""
    saved = catalog._state
    catalog._state = catalog._State()
    catalog._state.startup_attempted = True
    catalog._state.last_attempt = float("inf")
    yield catalog
    catalog._state = saved


@pytest.fixture
def api(monkeypatch):
    """FakeAPI in place of the HTTP layer, a test key in the environment, and instant sleeps."""
    fake = FakeAPI()
    monkeypatch.setattr(client, "_send", fake.send)
    monkeypatch.setattr(client, "_fetch", fake.fetch)
    monkeypatch.setattr(client, "_comfy_interrupted", fake.interrupted)

    def raise_fake_interrupt():
        raise FakeInterrupt()

    monkeypatch.setattr(client, "raise_interrupt", raise_fake_interrupt)

    async def fast_sleep(seconds):
        if fake.interrupted():
            raise client.Interrupted()
        await asyncio.sleep(0)

    monkeypatch.setattr(client, "sleep", fast_sleep)
    monkeypatch.setenv(config.ENV_VAR, TEST_KEY)
    return fake

