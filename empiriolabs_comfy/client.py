"""Requests to the EmpirioLabs AI API and downloads of the files it returns.

Rules this module keeps:

* Requests go to one fixed host, https://api.empiriolabs.ai. There is no setting
  that points the nodes anywhere else.
* The API key is sent only to that host, never to a download link, and never
  appears in an error message.
* Only links that the API returned are downloaded, only over https, and with a
  size limit. A redirect to anything other than https is refused.
* A request that can cost money (a POST) is never sent twice automatically.
  Only reads (polling a job, downloading a file) are retried.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, NoReturn, TypeVar
from urllib.parse import quote, urljoin, urlsplit

import aiohttp

from . import __version__

API_BASE = "https://api.empiriolabs.ai"
USER_AGENT = f"comfyui-empiriolabs/{__version__}"
USAGE_LOGS = "https://platform.empiriolabs.ai/dashboard/usage-logs"

MIB = 1024 * 1024
MAX_DOWNLOAD_BYTES = {"image": 100 * MIB, "audio": 512 * MIB, "video": 2048 * MIB}
DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_RETRY_STATUS = frozenset({404, 408, 425, 429, 500, 502, 503, 504})
MAX_REDIRECTS = 5
MAX_POLL_FAILURES = 6
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

log = logging.getLogger(__name__)

T = TypeVar("T")


class EmpirioLabsError(Exception):
    """A failure whose message is safe to show in ComfyUI."""


class ApiError(EmpirioLabsError):
    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class JobFailedError(EmpirioLabsError):
    def __init__(self, message: str, *, job_id: str, status: str):
        super().__init__(message)
        self.job_id = job_id
        self.status = status


class Interrupted(Exception):
    """The user cancelled the run in ComfyUI. Turned into ComfyUI's own interrupt by raise_interrupt()."""


# --------------------------------------------------------------------------- interruption


def _comfy_interrupted() -> bool:
    """True once the user presses Cancel in ComfyUI. Always False outside ComfyUI."""
    try:
        import comfy.model_management as model_management
    except Exception:
        return False
    return bool(model_management.processing_interrupted())


def raise_interrupt() -> NoReturn:
    """Stop the node the same way ComfyUI's own nodes stop when the queue is cancelled."""
    try:
        import comfy.model_management as model_management
    except Exception:
        raise Interrupted() from None
    model_management.throw_exception_if_processing_interrupted()
    raise model_management.InterruptProcessingException()


async def _interruptible(awaitable: Awaitable[T]) -> T:
    """Await `awaitable`, giving up as soon as the user cancels the run."""
    task = asyncio.ensure_future(awaitable)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=0.25)
            if done:
                return task.result()
            if _comfy_interrupted():
                raise Interrupted()
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


async def sleep(seconds: float) -> None:
    """Sleep that ends early with Interrupted when the user cancels the run."""
    deadline = time.monotonic() + max(seconds, 0.0)
    while True:
        if _comfy_interrupted():
            raise Interrupted()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        await asyncio.sleep(min(0.25, remaining))


# --------------------------------------------------------------------------- HTTP


async def _send(
    method: str, url: str, *, headers: dict[str, str], body: bytes | None, timeout: float
) -> tuple[int, dict[str, str], bytes]:
    """One HTTP exchange with the API. Redirects are not followed, so the key cannot leave the API host."""
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.request(method, url, data=body, headers=headers, allow_redirects=False) as response:
            raw = await response.read()
            return response.status, {k.lower(): v for k, v in response.headers.items()}, raw


class _DownloadStatus(Exception):
    def __init__(self, status: int):
        super().__init__(status)
        self.status = status


class _DownloadTooLarge(Exception):
    pass


class _UnsafeLink(Exception):
    pass


def is_https_url(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return parts.scheme == "https" and bool(parts.hostname)


async def _fetch(
    url: str,
    *,
    max_bytes: int,
    timeout: float,
    allowed: Callable[[str], bool] = is_https_url,
) -> tuple[bytes, str]:
    """Download one file without credentials. Every redirect hop must pass `allowed`."""
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    current = url
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        for _ in range(MAX_REDIRECTS + 1):
            if not allowed(current):
                raise _UnsafeLink()
            async with session.get(current, headers={"User-Agent": USER_AGENT}, allow_redirects=False) as response:
                if response.status in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if not location:
                        raise _DownloadStatus(response.status)
                    current = urljoin(current, location)
                    continue
                if response.status >= 400:
                    raise _DownloadStatus(response.status)
                if response.content_length is not None and response.content_length > max_bytes:
                    raise _DownloadTooLarge()
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.content.iter_chunked(MIB):
                    total += len(chunk)
                    if total > max_bytes:
                        raise _DownloadTooLarge()
                    chunks.append(chunk)
                return b"".join(chunks), response.headers.get("Content-Type", "")
    raise _UnsafeLink()


def parse_json(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None


_DEFAULT_ERRORS = {
    400: "The request was not accepted.",
    401: "The API key was not accepted.",
    402: "The account balance is too low for this request.",
    403: "This API key cannot use this model or endpoint.",
    404: "Not found.",
    413: "The request is too large.",
    429: "Too many requests. Wait a moment and try again.",
}


def _error_message(status: int, payload: Any) -> str:
    message: str | None = None
    request_id: str | None = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") if isinstance(error.get("message"), str) else None
            request_id = error.get("request_id") if isinstance(error.get("request_id"), str) else None
        elif isinstance(error, str):
            message = error
        if not message:
            for key in ("message", "detail"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    message = value
                    break
        if not request_id and isinstance(payload.get("request_id"), str):
            request_id = payload["request_id"]
    if not message:
        message = _DEFAULT_ERRORS.get(status, "The service had a problem. Try again in a moment.")
    if status == 401:
        message = f"{message} Check the key in the EMPIRIOLABS_API_KEY environment variable or in config.ini."
    text = f"EmpirioLabs AI returned HTTP {status}: {message}"
    if request_id:
        text += f" (request ID {request_id})"
    return text


def _scrub(text: str, api_key: str) -> str:
    return text.replace(api_key, "(API key)") if api_key else text


async def api_call(
    method: str,
    path: str,
    *,
    api_key: str,
    body: dict[str, Any] | None = None,
    timeout: float = 120.0,
    accept: str = "application/json",
) -> tuple[int, dict[str, str], bytes]:
    """Send one request to the API and return (status, headers, body). Raises ApiError on failure."""
    url = API_BASE + path
    headers = {"Authorization": f"Bearer {api_key}", "Accept": accept, "User-Agent": USER_AGENT}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    try:
        status, response_headers, raw = await _interruptible(
            _send(method, url, headers=headers, body=data, timeout=timeout)
        )
    except asyncio.TimeoutError:
        if method == "POST":
            # The request may have been accepted, so the node never repeats it by itself.
            raise ApiError(
                "EmpirioLabs AI did not answer in time. The request may still finish and be billed, so check "
                f"{USAGE_LOGS} before you run the node again.",
            ) from None
        raise ApiError("EmpirioLabs AI did not answer in time. Try again.", retryable=True) from None
    except aiohttp.ClientSSLError:
        raise ApiError(
            "The secure connection to EmpirioLabs AI could not be verified. Check the system certificates "
            "and any proxy or antivirus that inspects HTTPS traffic.",
            retryable=True,
        ) from None
    except aiohttp.ClientError:
        raise ApiError(
            "Could not reach EmpirioLabs AI. Check the internet connection and try again.", retryable=True
        ) from None
    if 300 <= status < 400:
        raise ApiError(f"EmpirioLabs AI returned an unexpected redirect (HTTP {status}).", status=status)
    if status >= 400:
        message = _scrub(_error_message(status, parse_json(raw)), api_key)
        raise ApiError(message, status=status, retryable=status in RETRYABLE_STATUS)
    return status, response_headers, raw


async def api_json(
    method: str, path: str, *, api_key: str, body: dict[str, Any] | None = None, timeout: float = 120.0
) -> dict[str, Any]:
    _, _, raw = await api_call(method, path, api_key=api_key, body=body, timeout=timeout)
    payload = parse_json(raw)
    if not isinstance(payload, dict):
        raise ApiError("EmpirioLabs AI returned a response the node could not read.")
    return payload


# --------------------------------------------------------------------------- jobs

_SUCCESS = frozenset({"completed", "complete", "succeeded", "success"})
_FAILED = frozenset({"failed", "error", "expired", "abandoned"})
_CANCELLED = frozenset({"cancelled", "canceled"})


def job_status(job: dict[str, Any]) -> str:
    value = str(job.get("status") or "").strip().lower()
    if value in _SUCCESS:
        return "completed"
    if value in _FAILED:
        return "failed"
    if value in _CANCELLED:
        return "cancelled"
    return value or "processing"


def is_job_envelope(payload: Any) -> bool:
    """True for the {job_id, status} answer the API gives when a result is not ready yet."""
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("job_id"), str)
        and bool(payload["job_id"])
        and not isinstance(payload.get("data"), list)
        and "choices" not in payload
    )


def job_error_message(job: dict[str, Any]) -> str | None:
    error = job.get("error")
    if isinstance(error, str) and error:
        return error
    if isinstance(error, dict):
        if isinstance(error.get("message"), str) and error["message"]:
            return error["message"]
        nested = error.get("error")
        if isinstance(nested, dict) and isinstance(nested.get("message"), str):
            return nested["message"]
    return None


def job_progress(job: dict[str, Any]) -> float | None:
    value = job.get("progress")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value > 1:
        value = value / 100.0
    return min(max(float(value), 0.0), 1.0)


def job_result(job: dict[str, Any]) -> dict[str, Any]:
    result = job.get("result")
    return result if isinstance(result, dict) else job


@dataclass
class JobUpdate:
    job_id: str
    status: str
    progress: float | None
    elapsed: float


async def cancel_job(job_id: str, *, api_key: str) -> None:
    """Best-effort cancel. Called after the user cancels, so it must not check for interruption itself."""
    url = API_BASE + "/v1/jobs/" + quote(job_id, safe="")
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json", "User-Agent": USER_AGENT}
    try:
        await _send("DELETE", url, headers=headers, body=None, timeout=10.0)
    except Exception:  # noqa: BLE001 - cancelling is best effort
        log.debug("EmpirioLabs AI: could not cancel job %s", job_id)


async def wait_for_job(
    job_id: str,
    *,
    api_key: str,
    timeout: float,
    interval: float,
    on_update: Callable[[JobUpdate], None] | None = None,
) -> dict[str, Any]:
    """Poll GET /v1/jobs/{job_id} until the job finishes. Cancels the job if the user cancels the run."""
    path = "/v1/jobs/" + quote(job_id, safe="")
    start = time.monotonic()
    failures = 0
    try:
        while True:
            job: dict[str, Any] | None = None
            try:
                job = await api_json("GET", path, api_key=api_key, timeout=30.0)
                failures = 0
            except ApiError as exc:
                if not exc.retryable or failures >= MAX_POLL_FAILURES:
                    raise
                failures += 1
            if job is not None:
                status = job_status(job)
                if on_update is not None:
                    on_update(JobUpdate(job_id, status, job_progress(job), time.monotonic() - start))
                if status == "completed":
                    return job
                if status == "failed":
                    detail = job_error_message(job) or "No reason was given."
                    raise JobFailedError(f"The job failed: {detail} (job {job_id})", job_id=job_id, status=status)
                if status == "cancelled":
                    raise JobFailedError(f"The job was cancelled (job {job_id}).", job_id=job_id, status=status)
            if time.monotonic() - start >= timeout:
                minutes = max(int(timeout // 60), 1)
                raise ApiError(
                    f"Job {job_id} is still running after {minutes} minutes. It keeps running on "
                    "EmpirioLabs AI, and its result stays available for an hour after it finishes at "
                    f"{API_BASE}/v1/jobs/{job_id}."
                )
            await sleep(interval if failures == 0 else min(interval * (2**failures), 30.0))
    except Interrupted:
        await cancel_job(job_id, api_key=api_key)
        raise


# --------------------------------------------------------------------------- results and downloads

_URL_KEYS = ("url", "video_url", "audio_url", "image_url", "file_url", "output_url")
_B64_KEYS = ("b64_json", "b64", "base64", "base64_data", "audio_base64")
_LIST_KEYS = ("data", "images", "videos", "audios", "outputs", "output", "output_urls", "urls")


@dataclass
class MediaRef:
    url: str | None = None
    data: bytes | None = None
    mime_type: str | None = None


def _decode_base64(value: str) -> tuple[bytes, str | None] | None:
    mime_type = None
    if value.startswith("data:"):
        header, _, value = value.partition(",")
        mime_type = header[5:].split(";")[0] or None
    try:
        return base64.b64decode(value, validate=False), mime_type
    except (binascii.Error, ValueError):
        return None


def _is_web_url(value: Any) -> bool:
    return isinstance(value, str) and value.lower().startswith(("https://", "http://"))


def _ref_from_value(value: Any) -> MediaRef | None:
    if isinstance(value, str):
        if _is_web_url(value):
            return MediaRef(url=value)
        if value.startswith("data:"):
            decoded = _decode_base64(value)
            return MediaRef(data=decoded[0], mime_type=decoded[1]) if decoded else None
        return None
    if not isinstance(value, dict):
        return None
    mime_type = value.get("content_type") or value.get("mime_type")
    mime_type = mime_type if isinstance(mime_type, str) else None
    for key in _URL_KEYS:
        if _is_web_url(value.get(key)):
            return MediaRef(url=value[key], mime_type=mime_type)
    for key in _B64_KEYS:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate:
            decoded = _decode_base64(candidate)
            if decoded:
                return MediaRef(data=decoded[0], mime_type=mime_type or decoded[1])
    return None


def media_refs(payload: Any) -> list[MediaRef]:
    """Find the generated files in an API answer or a finished job, as links or inline data."""
    if not isinstance(payload, dict):
        return []
    for key in _LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            refs = [ref for ref in (_ref_from_value(item) for item in value) if ref is not None]
            if refs:
                return refs
        elif isinstance(value, dict):
            nested = media_refs(value)
            if nested:
                return nested
    single = _ref_from_value(payload)
    if single is not None:
        return [single]
    result = payload.get("result")
    if isinstance(result, dict):
        return media_refs(result)
    return []


async def download(url: str, *, kind: str, timeout: float) -> tuple[bytes, str]:
    """Download a file the API returned. Returns (bytes, content type)."""
    if not is_https_url(url):
        raise ApiError("EmpirioLabs AI returned a file link that is not https, so the node did not download it.")
    max_bytes = MAX_DOWNLOAD_BYTES[kind]
    attempt = 0
    while True:
        attempt += 1
        try:
            return await _interruptible(_fetch(url, max_bytes=max_bytes, timeout=timeout))
        except _DownloadStatus as exc:
            if exc.status in DOWNLOAD_RETRY_STATUS and attempt < DOWNLOAD_ATTEMPTS:
                await sleep(2 ** (attempt - 1))
                continue
            raise ApiError(
                f"The generated {kind} could not be downloaded (HTTP {exc.status}). The link stays valid "
                f"for 7 days, so you can also open it directly: {url}"
            ) from None
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt < DOWNLOAD_ATTEMPTS:
                await sleep(2 ** (attempt - 1))
                continue
            raise ApiError(
                f"The generated {kind} could not be downloaded. Check the internet connection and try again. "
                f"The link stays valid for 7 days: {url}"
            ) from None
        except _DownloadTooLarge:
            raise ApiError(
                f"The generated {kind} is larger than {max_bytes // MIB} MB, so the node did not load it. "
                f"Download it from the link instead: {url}"
            ) from None
        except _UnsafeLink:
            raise ApiError(
                f"The link for the generated {kind} redirected somewhere other than https, so the node "
                "stopped the download."
            ) from None


async def load_ref(ref: MediaRef, *, kind: str, timeout: float) -> tuple[bytes, str]:
    """Bytes and content type for a result, whether it arrived inline or as a link."""
    if ref.data is not None:
        return ref.data, ref.mime_type or ""
    if ref.url is None:
        raise ApiError(f"The response did not include the generated {kind}.")
    body, content_type = await download(ref.url, kind=kind, timeout=timeout)
    return body, content_type or ref.mime_type or ""
