"""Test doubles and sample files shared by the tests. Nothing here touches the network."""

from __future__ import annotations

import asyncio
import io
import json
from dataclasses import dataclass, field
from typing import Any

from empiriolabs_comfy import client

TEST_KEY = "sk-empiriolabs-test-0123456789abcdef"


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- catalog

FAKE_CATALOG = {
    "object": "list",
    "data": [
        {
            "id": "flux-2-klein-4b",
            "display_name": "FLUX.2 Klein 4B",
            "status": "available",
            "category": "image",
            "supported_endpoints": ["POST /v1/images/generations"],
            "input_modalities": ["text", "image"],
            "supported_parameters": [
                {"name": "prompt", "type": "string"},
                {"name": "aspect_ratio", "type": "enum", "options": ["1:1", "16:9", "9:16", "4:3"]},
                {"name": "seed", "type": "number", "min": 0, "max": 65535},
            ],
        },
        {
            "id": "seedream-5-0-lite",
            "status": "available",
            "category": "image",
            "supported_endpoints": ["POST /v1/images/generations"],
            "supported_parameters": [
                {"name": "aspect_ratio", "type": "enum", "options": ["1:1", "16:9", "21:9"]},
                {"name": "resolution", "type": "enum", "options": ["2K", "4K"]},
            ],
        },
        {
            "id": "retired-image",
            "status": "retired",
            "supported_endpoints": ["POST /v1/images/generations"],
        },
        {
            "id": "hunyuan-video-1-5",
            "status": "available",
            "category": "video",
            "supported_endpoints": ["POST /v1/videos/generations"],
            "supported_parameters": [
                {"name": "resolution", "type": "enum", "options": ["480p", "720p", "1080p"]},
                {"name": "aspect_ratio", "type": "enum", "options": ["16:9", "9:16", "1:1"]},
                {"name": "seed", "type": "number"},
            ],
        },
        {
            "id": "pixverse-v6",
            "status": "available",
            "supported_endpoints": ["POST /v1/videos/generations"],
            "supported_parameters": [{"name": "resolution", "type": "enum", "options": ["360p", "540p", "720p"]}],
        },
        {
            "id": "qwen3-8-27b",
            "status": "available",
            "category": "text",
            "supported_endpoints": ["POST /v1/chat/completions", "POST /v1/responses"],
            "input_modalities": ["text", "image", "video"],
            "supported_parameters": [{"name": "seed", "type": "number", "min": 0, "max": 2147483647}],
        },
        {
            "id": "text-only-model",
            "status": "available",
            "supported_endpoints": ["POST /v1/chat/completions"],
            "input_modalities": ["text"],
        },
        {
            "id": "tts-2-flash",
            "status": "available",
            "category": "audio",
            "supported_endpoints": ["POST /v1/audio/speech"],
        },
    ],
}


def fake_catalog() -> dict[str, Any]:
    return json.loads(json.dumps(FAKE_CATALOG))


# --------------------------------------------------------------------------- API


@dataclass
class Call:
    method: str
    path: str
    headers: dict[str, str]
    body: Any
    timeout: float


class FakeInterrupt(Exception):
    """Stands in for ComfyUI's InterruptProcessingException."""


@dataclass
class FakeAPI:
    """Scripted stand-in for the HTTP layer (client._send and client._fetch)."""

    routes: dict[tuple[str, str], list[Any]] = field(default_factory=dict)
    files: dict[str, list[Any]] = field(default_factory=dict)
    calls: list[Call] = field(default_factory=list)
    downloads: list[str] = field(default_factory=list)
    interrupt_after_calls: int | None = None

    def add(self, method: str, path: str, *responses: Any) -> None:
        """Responses are (status, json payload), (status, bytes, content type) or an exception.
        The last one repeats once the others are used up."""
        self.routes.setdefault((method, path), []).extend(responses)

    def add_file(self, url: str, *responses: Any) -> None:
        """Responses are (bytes, content type), an HTTP status int, or an exception."""
        self.files.setdefault(url, []).extend(responses)

    def calls_to(self, method: str, path: str) -> list[Call]:
        return [c for c in self.calls if c.method == method and c.path == path]

    async def send(self, method, url, *, headers, body, timeout):
        assert url.startswith(client.API_BASE + "/"), url
        path = url[len(client.API_BASE):]
        parsed = json.loads(body.decode("utf-8")) if body else None
        self.calls.append(Call(method, path, dict(headers), parsed, timeout))
        queue = self.routes.get((method, path))
        if not queue:
            raise AssertionError(f"unexpected request {method} {path}")
        response = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(response, BaseException):
            raise response
        if len(response) == 3:
            status, raw, content_type = response
            return status, {"content-type": content_type}, raw
        status, payload = response
        return status, {"content-type": "application/json"}, json.dumps(payload).encode("utf-8")

    async def fetch(self, url, *, max_bytes, timeout, allowed=None):
        self.downloads.append(url)
        queue = self.files.get(url)
        if not queue:
            raise AssertionError(f"unexpected download {url}")
        response = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, int):
            raise client._DownloadStatus(response)
        data, content_type = response
        if len(data) > max_bytes:
            raise client._DownloadTooLarge()
        return data, content_type

    def interrupted(self) -> bool:
        return self.interrupt_after_calls is not None and len(self.calls) >= self.interrupt_after_calls


# --------------------------------------------------------------------------- sample files


def png_bytes(width=32, height=24, color=(200, 30, 30), mode="RGB") -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new(mode, (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def webp_bytes(width=32, height=24, color=(30, 200, 30)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="WEBP")
    return buffer.getvalue()


def wav_bytes(seconds=0.25, rate=16000, channels=1) -> bytes:
    import math
    import struct
    import wave

    frames = int(seconds * rate)
    samples = []
    for index in range(frames):
        value = int(12000 * math.sin(2 * math.pi * 440 * index / rate))
        samples.extend([value] * channels)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return buffer.getvalue()


def mp4_bytes(frames=6, width=64, height=48) -> bytes:
    """A tiny MP4 made in memory with PyAV (MPEG-4 Part 2, which every PyAV build can encode)."""
    import av
    import numpy as np

    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            array = np.full((height, width, 3), (index * 30) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()
