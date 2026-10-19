from __future__ import annotations

import logging

import pytest

from empiriolabs_comfy import catalog
from fakes import fake_catalog, run


def test_parse_catalog_keeps_available_models_only():
    models = catalog.parse_catalog(fake_catalog())
    ids = {m["id"] for m in models}
    assert "retired-image" not in ids
    assert "flux-2-klein-4b" in ids


@pytest.mark.parametrize("payload", [None, [], {"data": None}, {"data": []}, {"data": [{"name": "no id"}]}])
def test_parse_catalog_rejects_unexpected_answers(payload):
    with pytest.raises(ValueError):
        catalog.parse_catalog(payload)


def test_models_are_listed_by_the_endpoint_each_node_calls(live_catalog):
    assert catalog.model_ids("image") == ["flux-2-klein-4b", "seedream-5-0-lite"]
    assert catalog.model_ids("video") == ["hunyuan-video-1-5", "pixverse-v6"]
    assert catalog.model_ids("chat") == ["qwen3-8-27b", "text-only-model"]
    assert catalog.model_ids("speech") == ["tts-2-flash"]
    assert catalog.is_live()


def test_defaults_prefer_the_fallback_model_when_listed(live_catalog):
    assert catalog.default_model("image") == "flux-2-klein-4b"
    catalog._state.models = [m for m in catalog._state.models if m["id"] != "flux-2-klein-4b"]
    assert catalog.default_model("image") == "seedream-5-0-lite"


def test_fallback_is_one_model_per_node(no_catalog):
    assert not catalog.is_live()
    for kind, model in catalog.FALLBACK_MODELS.items():
        assert catalog.model_ids(kind) == [model]
        assert catalog.default_model(kind) == model


def test_option_values_are_the_union_most_common_first(live_catalog):
    assert catalog.option_values("image", "aspect_ratio") == ["1:1", "16:9", "9:16", "4:3", "21:9"]
    assert catalog.option_values("image", "resolution") == ["2K", "4K"]
    assert catalog.option_values("video", "resolution") == ["720p", "480p", "1080p", "360p", "540p"]


def test_option_values_fall_back_when_the_catalog_is_missing(no_catalog):
    assert catalog.option_values("video", "aspect_ratio") == catalog.FALLBACK_OPTIONS[("video", "aspect_ratio")]
    assert catalog.option_values("chat", "aspect_ratio") == []


def test_seed_is_mapped_into_each_models_range(live_catalog):
    assert catalog.seed_for("flux-2-klein-4b", 70000) == 70000 % 65536
    assert catalog.seed_for("hunyuan-video-1-5", 123) == 123  # no range given: 0..2**31-1
    assert catalog.seed_for("seedream-5-0-lite", 5) is None  # takes no seed
    assert catalog.seed_for("unknown-model", 5) is None


def test_ensure_loaded_falls_back_quietly_when_the_catalog_is_unreachable(monkeypatch, caplog):
    monkeypatch.setattr(catalog, "_state", catalog._State())

    async def failing_fetch(timeout):
        raise OSError("network down")

    monkeypatch.setattr(catalog, "fetch", failing_fetch)
    with caplog.at_level(logging.WARNING):
        run(catalog.ensure_loaded())
    assert not catalog.is_live()
    assert "could not load the model catalog" in caplog.text

    calls = []

    async def counting_fetch(timeout):
        calls.append(timeout)
        return []

    monkeypatch.setattr(catalog, "fetch", counting_fetch)
    run(catalog.ensure_loaded())  # the startup attempt is made once
    assert calls == []


def test_ensure_loaded_stores_the_live_catalog(monkeypatch):
    monkeypatch.setattr(catalog, "_state", catalog._State())

    async def fetch(timeout):
        return catalog.parse_catalog(fake_catalog())

    monkeypatch.setattr(catalog, "fetch", fetch)
    run(catalog.ensure_loaded())
    assert catalog.is_live() and "qwen3-8-27b" in catalog.model_ids("chat")


def test_maybe_refresh_only_starts_a_refresh_when_due(monkeypatch, live_catalog):
    started = []

    class FakeThread:
        def __init__(self, target, name, daemon):
            started.append(name)

        def start(self):
            pass

    monkeypatch.setattr(catalog.threading, "Thread", FakeThread)
    catalog.maybe_refresh()
    assert started == []  # just loaded

    catalog._state.loaded_at -= catalog.REFRESH_INTERVAL + 1
    catalog.maybe_refresh()
    catalog.maybe_refresh()  # a refresh is already running
    assert started == ["empiriolabs-catalog"]


def test_maybe_refresh_retries_after_a_failed_startup(monkeypatch, no_catalog):
    started = []
    monkeypatch.setattr(
        catalog.threading, "Thread", lambda target, name, daemon: started.append(name) or type("T", (), {"start": lambda self: None})()
    )
    catalog.maybe_refresh()
    assert started == []  # not due yet
    catalog._state.last_attempt = 0.0
    catalog.maybe_refresh()
    assert started == ["empiriolabs-catalog"]


def test_background_fetch_stores_the_result(monkeypatch, no_catalog):
    async def fetch(timeout):
        return catalog.parse_catalog(fake_catalog())

    monkeypatch.setattr(catalog, "fetch", fetch)
    catalog._state.refreshing = True
    catalog._background_fetch()
    assert catalog.is_live() and not catalog._state.refreshing
