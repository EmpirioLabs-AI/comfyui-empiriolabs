"""Model lists for the node dropdowns, read from the public EmpirioLabs AI catalog.

Nothing here is a hand-kept model list. GET https://api.empiriolabs.ai/v1/models
(public, no key) is read once while ComfyUI starts, with a short timeout, and again
in the background every few hours. Each node lists the models whose catalog entry
names the endpoint the node calls in `supported_endpoints`.

If the catalog cannot be reached, each node offers the single FALLBACK_MODELS entry
so that it still loads and runs, and a background retry replaces it with the live
list. Refreshing the browser page shows the new list.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import Counter
from typing import Any

import aiohttp

from .client import API_BASE, USER_AGENT

CATALOG_URL = API_BASE + "/v1/models"
STARTUP_TIMEOUT = 6.0
REFRESH_TIMEOUT = 20.0
REFRESH_INTERVAL = 6 * 3600.0
RETRY_INTERVAL = 120.0

ENDPOINTS = {
    "image": "POST /v1/images/generations",
    "video": "POST /v1/videos/generations",
    "chat": "POST /v1/chat/completions",
    "speech": "POST /v1/audio/speech",
}

# FALLBACK ONLY. Used while the live catalog cannot be reached, so that each node still
# loads and runs with one model. It is not the model list: the catalog replaces it as soon
# as it loads. When the catalog lists these models they are also the default selections.
FALLBACK_MODELS = {
    "image": "flux-2-klein-4b",
    "video": "hunyuan-video-1-5",
    "chat": "qwen3-8-27b",
    "speech": "tts-2-flash",
}

# FALLBACK ONLY, for the aspect ratio and resolution dropdowns while the catalog is unreachable.
FALLBACK_OPTIONS = {
    ("image", "aspect_ratio"): ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
    ("image", "resolution"): ["1K", "2K", "4K"],
    ("video", "aspect_ratio"): ["16:9", "9:16", "1:1", "4:3", "3:4"],
    ("video", "resolution"): ["480p", "720p", "1080p"],
}

log = logging.getLogger(__name__)


class _State:
    def __init__(self) -> None:
        self.models: list[dict[str, Any]] | None = None  # None until the catalog has loaded once
        self.startup_attempted = False
        self.loaded_at = 0.0
        self.last_attempt = 0.0
        self.refreshing = False
        self.lock = threading.Lock()


_state = _State()


def parse_catalog(payload: Any) -> list[dict[str, Any]]:
    """The available entries of a /v1/models answer. Raises ValueError for anything else."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("unexpected catalog response")
    entries = [m for m in payload["data"] if isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"]]
    if not entries:
        raise ValueError("the catalog listed no models")
    return [m for m in entries if m.get("status") in (None, "available")]


async def fetch(timeout: float) -> list[dict[str, Any]]:
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.get(CATALOG_URL, headers=headers, allow_redirects=False) as response:
            if response.status != 200:
                raise ValueError(f"HTTP {response.status}")
            payload = await response.json(content_type=None)
    return parse_catalog(payload)


def _store(models: list[dict[str, Any]]) -> None:
    with _state.lock:
        _state.models = models
        _state.loaded_at = time.monotonic()


def _describe(exc: BaseException) -> str:
    return str(exc) or type(exc).__name__


async def ensure_loaded(timeout: float = STARTUP_TIMEOUT) -> None:
    """Load the catalog once at startup. Never raises: on failure the fallback stays in place
    and maybe_refresh() tries again later."""
    if _state.models is not None or _state.startup_attempted:
        return
    _state.startup_attempted = True
    _state.last_attempt = time.monotonic()
    try:
        models = await asyncio.wait_for(fetch(timeout), timeout + 1.0)
    except Exception as exc:  # noqa: BLE001 - any failure falls back
        log.warning(
            "EmpirioLabs AI: could not load the model catalog (%s). Each node lists one fallback model "
            "until the catalog loads.",
            _describe(exc),
        )
        return
    _store(models)
    log.info("EmpirioLabs AI: loaded %d models from the catalog.", len(models))


def _background_fetch() -> None:
    try:
        models = asyncio.run(fetch(REFRESH_TIMEOUT))
    except Exception as exc:  # noqa: BLE001
        log.debug("EmpirioLabs AI: catalog refresh failed (%s)", _describe(exc))
    else:
        _store(models)
    finally:
        with _state.lock:
            _state.refreshing = False


def maybe_refresh() -> None:
    """Start a background refresh when the list is stale or the startup load failed. Never blocks."""
    now = time.monotonic()
    with _state.lock:
        if _state.refreshing:
            return
        if _state.models is None:
            due = now - _state.last_attempt >= RETRY_INTERVAL
        else:
            due = now - _state.loaded_at >= REFRESH_INTERVAL
        if not due:
            return
        _state.refreshing = True
        _state.last_attempt = now
    threading.Thread(target=_background_fetch, name="empiriolabs-catalog", daemon=True).start()


def is_live() -> bool:
    return _state.models is not None


def models_for(kind: str) -> list[dict[str, Any]]:
    endpoint = ENDPOINTS[kind]
    return [m for m in (_state.models or []) if endpoint in (m.get("supported_endpoints") or [])]


def model_ids(kind: str) -> list[str]:
    ids = sorted({m["id"] for m in models_for(kind)})
    return ids or [FALLBACK_MODELS[kind]]


def default_model(kind: str) -> str:
    ids = model_ids(kind)
    preferred = FALLBACK_MODELS[kind]
    return preferred if preferred in ids else ids[0]


def find_model(model_id: str) -> dict[str, Any] | None:
    for model in _state.models or []:
        if model.get("id") == model_id:
            return model
    return None


def parameter(model_id: str, name: str) -> dict[str, Any] | None:
    model = find_model(model_id)
    if model is None:
        return None
    for spec in model.get("supported_parameters") or []:
        if isinstance(spec, dict) and spec.get("name") == name:
            return spec
    return None


def option_values(kind: str, name: str) -> list[str]:
    """Every value the catalog offers for `name` across this node's models, most common first."""
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for model in models_for(kind):
        for spec in model.get("supported_parameters") or []:
            if not isinstance(spec, dict) or spec.get("name") != name or not isinstance(spec.get("options"), list):
                continue
            for option in spec["options"]:
                if isinstance(option, bool) or not isinstance(option, (str, int, float)):
                    continue
                value = str(option)
                counts[value] += 1
                first_seen.setdefault(value, len(first_seen))
    values = sorted(counts, key=lambda value: (-counts[value], first_seen[value]))
    return values or list(FALLBACK_OPTIONS.get((kind, name), []))


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def seed_for(model_id: str, seed: int) -> int | None:
    """The seed to send, mapped into the model's range, or None when the model takes no seed."""
    spec = parameter(model_id, "seed")
    if spec is None:
        return None
    low = _as_int(spec.get("min"))
    high = _as_int(spec.get("max"))
    low = 0 if low is None else low
    high = 2**31 - 1 if high is None else high
    if high < low:
        return None
    return low + (int(seed) % (high - low + 1))
