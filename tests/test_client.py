from __future__ import annotations

import asyncio
import base64

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from empiriolabs_comfy import client
from fakes import TEST_KEY, run

MEDIA = "https://media.empiriolabs.ai/files/a.webp?token=t&expires=1"


# --------------------------------------------------------------------------- API calls


def test_api_call_sends_the_key_only_as_a_bearer_header(api):
    api.add("GET", "/v1/jobs/j1", (200, {"job_id": "j1", "status": "processing"}))
    run(client.api_json("GET", "/v1/jobs/j1", api_key=TEST_KEY))
    call = api.calls[0]
    assert call.headers["Authorization"] == f"Bearer {TEST_KEY}"
    assert call.headers["User-Agent"].startswith("comfyui-empiriolabs/")
    assert "Content-Type" not in call.headers


def test_api_error_uses_the_api_message_and_request_id(api):
    api.add(
        "POST",
        "/v1/images/generations",
        (404, {"error": {"code": "model_not_found", "message": "Model 'x' is not available.", "request_id": "req_1"}}),
    )
    with pytest.raises(client.ApiError) as excinfo:
        run(client.api_json("POST", "/v1/images/generations", api_key=TEST_KEY, body={"model": "x"}))
    assert str(excinfo.value) == "EmpirioLabs AI returned HTTP 404: Model 'x' is not available. (request ID req_1)"
    assert excinfo.value.status == 404 and not excinfo.value.retryable


def test_api_error_never_repeats_the_key(api):
    api.add("POST", "/v1/chat/completions", (400, {"error": {"message": f"bad key {TEST_KEY}"}}))
    with pytest.raises(client.ApiError) as excinfo:
        run(client.api_json("POST", "/v1/chat/completions", api_key=TEST_KEY, body={}))
    assert TEST_KEY not in str(excinfo.value)


def test_unauthorized_points_at_the_key_settings(api):
    api.add("GET", "/v1/jobs/j1", (401, {"error": {"code": "invalid_api_key", "message": "Invalid API key."}}))
    with pytest.raises(client.ApiError) as excinfo:
        run(client.api_json("GET", "/v1/jobs/j1", api_key=TEST_KEY))
    assert "EMPIRIOLABS_API_KEY" in str(excinfo.value)


@pytest.mark.parametrize(
    "status,body,expected,retryable",
    [
        (402, b"", "The account balance is too low", False),
        (429, b"<html>busy</html>", "Too many requests", True),
        (503, b"oops", "The service had a problem", True),
        (400, b'{"detail": "prompt is required"}', "prompt is required", False),
    ],
)
def test_error_bodies_that_are_not_the_usual_shape(api, status, body, expected, retryable):
    api.add("POST", "/v1/videos/generations", (status, body, "text/plain"))
    with pytest.raises(client.ApiError) as excinfo:
        run(client.api_json("POST", "/v1/videos/generations", api_key=TEST_KEY, body={}))
    assert expected in str(excinfo.value)
    assert excinfo.value.retryable is retryable


def test_redirects_from_the_api_are_refused(api):
    api.add("GET", "/v1/jobs/j1", (302, b"", "text/html"))
    with pytest.raises(client.ApiError, match="unexpected redirect"):
        run(client.api_json("GET", "/v1/jobs/j1", api_key=TEST_KEY))


def test_network_failures_become_readable_errors(api):
    api.add("GET", "/v1/jobs/j1", aiohttp.ClientConnectionError("boom"))
    with pytest.raises(client.ApiError, match="Could not reach EmpirioLabs AI") as excinfo:
        run(client.api_json("GET", "/v1/jobs/j1", api_key=TEST_KEY))
    assert excinfo.value.retryable


def test_a_timeout_becomes_a_readable_error(api):
    api.add("GET", "/v1/jobs/j1", asyncio.TimeoutError())
    with pytest.raises(client.ApiError, match="did not answer in time") as excinfo:
        run(client.api_json("GET", "/v1/jobs/j1", api_key=TEST_KEY))
    assert excinfo.value.retryable


def test_a_timed_out_post_warns_that_it_may_still_be_billed(api):
    api.add("POST", "/v1/images/generations", asyncio.TimeoutError())
    with pytest.raises(client.ApiError, match="may still finish and be billed") as excinfo:
        run(client.api_json("POST", "/v1/images/generations", api_key=TEST_KEY, body={}))
    assert not excinfo.value.retryable
    assert client.USAGE_LOGS in str(excinfo.value)
    assert len(api.calls) == 1


def test_a_non_json_success_body_is_an_error(api):
    api.add("GET", "/v1/jobs/j1", (200, b"<html></html>", "text/html"))
    with pytest.raises(client.ApiError, match="could not read"):
        run(client.api_json("GET", "/v1/jobs/j1", api_key=TEST_KEY))


# --------------------------------------------------------------------------- jobs


def test_wait_for_job_polls_until_completed_and_reports_progress(api):
    api.add(
        "GET",
        "/v1/jobs/job_1",
        (200, {"job_id": "job_1", "status": "processing", "progress": 0.25}),
        (200, {"job_id": "job_1", "status": "processing", "progress": 60}),
        (200, {"job_id": "job_1", "status": "completed", "result": {"data": [{"url": MEDIA}]}}),
    )
    updates = []
    job = run(client.wait_for_job("job_1", api_key=TEST_KEY, timeout=60, interval=1, on_update=updates.append))
    assert client.job_status(job) == "completed"
    assert [u.progress for u in updates] == [0.25, 0.6, None]
    assert len(api.calls_to("GET", "/v1/jobs/job_1")) == 3


def test_wait_for_job_retries_transient_errors(api):
    api.add(
        "GET",
        "/v1/jobs/job_1",
        (503, {"error": {"message": "busy"}}),
        aiohttp.ClientConnectionError("reset"),
        (200, {"job_id": "job_1", "status": "succeeded", "result": {}}),
    )
    job = run(client.wait_for_job("job_1", api_key=TEST_KEY, timeout=60, interval=1))
    assert job["status"] == "succeeded"


def test_wait_for_job_gives_up_after_repeated_transient_errors(api):
    api.add("GET", "/v1/jobs/job_1", (503, {"error": {"message": "busy"}}))
    with pytest.raises(client.ApiError, match="busy"):
        run(client.wait_for_job("job_1", api_key=TEST_KEY, timeout=600, interval=1))
    assert len(api.calls_to("GET", "/v1/jobs/job_1")) == client.MAX_POLL_FAILURES + 1


def test_wait_for_job_stops_on_a_missing_job(api):
    api.add("GET", "/v1/jobs/job_1", (404, {"error": {"code": "not_found", "message": "Job 'job_1' not found"}}))
    with pytest.raises(client.ApiError, match="not found"):
        run(client.wait_for_job("job_1", api_key=TEST_KEY, timeout=60, interval=1))
    assert len(api.calls) == 1


def test_failed_job_reports_the_api_reason(api):
    api.add(
        "GET",
        "/v1/jobs/job_1",
        (200, {"job_id": "job_1", "status": "failed", "error": {"message": "The prompt was flagged by moderation."}}),
    )
    with pytest.raises(client.JobFailedError) as excinfo:
        run(client.wait_for_job("job_1", api_key=TEST_KEY, timeout=60, interval=1))
    assert str(excinfo.value) == "The job failed: The prompt was flagged by moderation. (job job_1)"


def test_cancelled_job_is_reported(api):
    api.add("GET", "/v1/jobs/job_1", (200, {"job_id": "job_1", "status": "canceled"}))
    with pytest.raises(client.JobFailedError, match="cancelled"):
        run(client.wait_for_job("job_1", api_key=TEST_KEY, timeout=60, interval=1))


def test_slow_job_times_out_with_the_job_id(api):
    api.add("GET", "/v1/jobs/job_1", (200, {"job_id": "job_1", "status": "processing"}))
    with pytest.raises(client.ApiError) as excinfo:
        run(client.wait_for_job("job_1", api_key=TEST_KEY, timeout=0, interval=1))
    message = str(excinfo.value)
    assert "job_1" in message and "still running" in message
    assert not api.calls_to("DELETE", "/v1/jobs/job_1")


def test_cancelling_the_run_cancels_the_job(api):
    api.add("GET", "/v1/jobs/job_1", (200, {"job_id": "job_1", "status": "processing"}))
    api.add("DELETE", "/v1/jobs/job_1", (200, {"job_id": "job_1", "status": "cancelled"}))
    api.interrupt_after_calls = 2
    with pytest.raises(client.Interrupted):
        run(client.wait_for_job("job_1", api_key=TEST_KEY, timeout=600, interval=1))
    assert len(api.calls_to("DELETE", "/v1/jobs/job_1")) == 1


def test_cancel_job_failures_are_ignored(api):
    api.add("DELETE", "/v1/jobs/job_1", aiohttp.ClientConnectionError("down"))
    run(client.cancel_job("job_1", api_key=TEST_KEY))


def test_job_ids_are_encoded_as_one_path_segment(api):
    api.add("GET", "/v1/jobs/a%2Fb%3Fc", (200, {"job_id": "a/b?c", "status": "completed", "result": {}}))
    run(client.wait_for_job("a/b?c", api_key=TEST_KEY, timeout=60, interval=1))
    assert api.calls[0].path == "/v1/jobs/a%2Fb%3Fc"


def test_job_envelope_detection():
    assert client.is_job_envelope({"job_id": "j", "status": "processing"})
    assert not client.is_job_envelope({"job_id": "j", "data": [{"url": MEDIA}]})
    assert not client.is_job_envelope({"id": "chatcmpl-1", "choices": []})
    assert not client.is_job_envelope({"job_id": ""})


# --------------------------------------------------------------------------- results


def test_media_refs_reads_urls_inline_data_and_nested_results():
    png = b"\x89PNG fake"
    encoded = base64.b64encode(png).decode()
    assert [r.url for r in client.media_refs({"data": [{"url": MEDIA}, {"url": MEDIA + "2"}]})] == [MEDIA, MEDIA + "2"]
    assert client.media_refs({"data": [{"b64_json": encoded}]})[0].data == png
    ref = client.media_refs({"data": [f"data:image/png;base64,{encoded}"]})[0]
    assert ref.data == png and ref.mime_type == "image/png"
    assert client.media_refs({"status": "completed", "result": {"data": [{"video_url": MEDIA}]}})[0].url == MEDIA
    assert client.media_refs({"data": [{"url": MEDIA, "content_type": "image/webp"}]})[0].mime_type == "image/webp"
    assert client.media_refs({"data": [{"url": "ftp://example.com/a"}]}) == []
    assert client.media_refs("nope") == []


# --------------------------------------------------------------------------- downloads


def test_download_refuses_links_that_are_not_https(api):
    with pytest.raises(client.ApiError, match="not https"):
        run(client.download("http://media.empiriolabs.ai/a.webp", kind="image", timeout=10))
    with pytest.raises(client.ApiError, match="not https"):
        run(client.download("file:///etc/passwd", kind="image", timeout=10))
    assert api.downloads == []


def test_download_retries_while_a_new_file_propagates(api):
    api.add_file(MEDIA, 404, 503, (b"bytes", "image/webp"))
    assert run(client.download(MEDIA, kind="image", timeout=10)) == (b"bytes", "image/webp")
    assert len(api.downloads) == 3


def test_download_gives_up_with_the_link_in_the_message(api):
    api.add_file(MEDIA, 403)
    with pytest.raises(client.ApiError) as excinfo:
        run(client.download(MEDIA, kind="video", timeout=10))
    assert "HTTP 403" in str(excinfo.value) and MEDIA in str(excinfo.value)
    assert len(api.downloads) == 1


def test_download_size_limit(api, monkeypatch):
    monkeypatch.setitem(client.MAX_DOWNLOAD_BYTES, "image", 4)
    api.add_file(MEDIA, (b"12345", "image/png"))
    with pytest.raises(client.ApiError, match="larger than"):
        run(client.download(MEDIA, kind="image", timeout=10))


def test_inline_results_are_not_downloaded(api):
    data, mime = run(client.load_ref(client.MediaRef(data=b"x", mime_type="audio/wav"), kind="audio", timeout=5))
    assert (data, mime) == (b"x", "audio/wav")
    assert api.downloads == []


# --------------------------------------------------------------------------- interruption


def test_interruptible_cancels_the_pending_request(monkeypatch):
    state = {"interrupted": False, "cancelled": False}
    monkeypatch.setattr(client, "_comfy_interrupted", lambda: state["interrupted"])

    async def slow():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    async def scenario():
        async def flip():
            await asyncio.sleep(0.05)
            state["interrupted"] = True

        asyncio.ensure_future(flip())
        await client._interruptible(slow())

    with pytest.raises(client.Interrupted):
        run(scenario())
    assert state["cancelled"]


def test_raise_interrupt_outside_comfyui_raises_interrupted(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_comfy(name, *args, **kwargs):
        if name.startswith("comfy"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_comfy)
    with pytest.raises(client.Interrupted):
        client.raise_interrupt()
    assert client._comfy_interrupted() is False


# --------------------------------------------------------------------------- the real HTTP layer, on a local server


@pytest.fixture
def local_server():
    """Starts an aiohttp app on 127.0.0.1. Returns the server and its base URL."""

    async def start(app):
        server = TestServer(app, host="127.0.0.1")
        await server.start_server()
        return server, str(server.make_url("")).rstrip("/")

    return start


def test_send_and_fetch_against_a_real_server(local_server):
    seen = {}

    async def api_handler(request):
        seen["api_auth"] = request.headers.get("Authorization")
        seen["api_body"] = await request.json()
        return web.json_response({"ok": True})

    async def file_handler(request):
        seen["file_auth"] = request.headers.get("Authorization")
        return web.Response(body=b"0123456789", content_type="video/mp4")

    async def redirect_handler(request):
        raise web.HTTPFound("/file")

    async def offsite_handler(request):
        raise web.HTTPFound("https://elsewhere.example.com/file")

    async def scenario():
        app = web.Application()
        app.router.add_post("/v1/thing", api_handler)
        app.router.add_get("/file", file_handler)
        app.router.add_get("/redirect", redirect_handler)
        app.router.add_get("/offsite", offsite_handler)
        runner, base = await local_server(app)
        local_only = lambda url: url.startswith(base)  # noqa: E731
        try:
            status, headers, raw = await client._send(
                "POST",
                base + "/v1/thing",
                headers={"Authorization": "Bearer k", "Content-Type": "application/json"},
                body=b'{"a": 1}',
                timeout=10,
            )
            assert status == 200 and raw == b'{"ok": true}' and headers["content-type"].startswith("application/json")

            body, content_type = await client._fetch(base + "/redirect", max_bytes=100, timeout=10, allowed=local_only)
            assert body == b"0123456789" and content_type == "video/mp4"

            with pytest.raises(client._DownloadTooLarge):
                await client._fetch(base + "/file", max_bytes=5, timeout=10, allowed=local_only)
            with pytest.raises(client._UnsafeLink):
                await client._fetch(base + "/offsite", max_bytes=100, timeout=10, allowed=local_only)
            with pytest.raises(client._UnsafeLink):
                await client._fetch(base + "/file", max_bytes=100, timeout=10)  # http is not https
            with pytest.raises(client._DownloadStatus) as excinfo:
                await client._fetch(base + "/missing", max_bytes=100, timeout=10, allowed=local_only)
            assert excinfo.value.status == 404
        finally:
            await runner.close()

    run(scenario())
    assert seen["api_auth"] == "Bearer k" and seen["api_body"] == {"a": 1}
    assert seen["file_auth"] is None


def test_send_does_not_follow_redirects(local_server):
    async def handler(request):
        raise web.HTTPFound("https://elsewhere.example.com/steal")

    async def scenario():
        app = web.Application()
        app.router.add_get("/v1/jobs/x", handler)
        runner, base = await local_server(app)
        try:
            status, headers, _ = await client._send(
                "GET", base + "/v1/jobs/x", headers={"Authorization": "Bearer k"}, body=None, timeout=10
            )
        finally:
            await runner.close()
        return status, headers

    status, headers = run(scenario())
    assert status == 302 and headers["location"] == "https://elsewhere.example.com/steal"
