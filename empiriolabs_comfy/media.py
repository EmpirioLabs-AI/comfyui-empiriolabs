"""Conversions between downloaded files and ComfyUI's IMAGE, AUDIO and VIDEO values.

Modelled on ComfyUI's own API-node helpers (comfy_api_nodes/util/conversions.py and
download_helpers.py), using only libraries ComfyUI already ships: Pillow, NumPy,
PyTorch and PyAV.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import av
import numpy as np
import torch
from PIL import Image, ImageOps

MAX_DATA_URI_PIXELS = 2048 * 2048


class MediaError(ValueError):
    """A downloaded file could not be turned into a ComfyUI value."""


def _open_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(data))
        image.load()
    except Exception:
        raise MediaError("The generated image could not be decoded.") from None
    return ImageOps.exif_transpose(image)


def _has_alpha(image: Image.Image) -> bool:
    return "A" in image.getbands() or "transparency" in image.info


def images_to_tensor(files: list[bytes]) -> torch.Tensor:
    """Decode images into one [B, H, W, C] float tensor in 0..1.

    Images keep an alpha channel only when one of them has transparency, as ComfyUI's
    own API nodes do. Images whose size differs from the first are fitted to it, so the
    batch can be stacked.
    """
    if not files:
        raise MediaError("No images were returned.")
    images = [_open_image(data) for data in files]
    mode = "RGBA" if any(_has_alpha(image) for image in images) else "RGB"
    converted = [image.convert(mode) for image in images]
    width, height = converted[0].size
    arrays = []
    for image in converted:
        if image.size != (width, height):
            image = ImageOps.fit(image, (width, height), Image.Resampling.LANCZOS)
        arrays.append(np.asarray(image, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(arrays, axis=0))


def image_to_data_uri(image: torch.Tensor, max_pixels: int = MAX_DATA_URI_PIXELS) -> str:
    """Encode one [H, W, C] image as a data URI for a chat message: JPEG, or PNG when it has alpha."""
    array = image.detach().to(torch.float32).clamp(0.0, 1.0).cpu().numpy()
    array = np.round(array * 255.0).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 2:
        pil = Image.fromarray(array, "L").convert("RGB")
    elif array.shape[-1] == 4:
        pil = Image.fromarray(array, "RGBA")
    else:
        pil = Image.fromarray(array[..., :3], "RGB")
    width, height = pil.size
    if width * height > max_pixels:
        scale = (max_pixels / float(width * height)) ** 0.5
        pil = pil.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    if pil.mode == "RGBA":
        pil.save(buffer, format="PNG")
        mime = "image/png"
    else:
        pil.save(buffer, format="JPEG", quality=95)
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def batch_to_data_uris(images: torch.Tensor) -> list[str]:
    if images.ndim == 3:
        images = images.unsqueeze(0)
    return [image_to_data_uri(images[index]) for index in range(images.shape[0])]


def audio_to_comfy(data: bytes) -> dict[str, Any]:
    """Decode any audio file PyAV can read into ComfyUI's AUDIO value."""
    try:
        with av.open(BytesIO(data)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise MediaError("The generated audio has no audio track.")
            sample_rate = int(stream.codec_context.sample_rate or stream.rate or 0)
            resampler = av.AudioResampler(format="fltp")
            chunks: list[np.ndarray] = []
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    chunks.append(converted.to_ndarray())
            for converted in resampler.resample(None):
                chunks.append(converted.to_ndarray())
    except MediaError:
        raise
    except Exception:
        raise MediaError("The generated audio could not be decoded.") from None
    if not chunks or sample_rate <= 0:
        raise MediaError("The generated audio is empty.")
    waveform = torch.from_numpy(np.concatenate(chunks, axis=1).astype(np.float32))
    return {"waveform": waveform.unsqueeze(0).contiguous(), "sample_rate": sample_rate}


def check_video(data: bytes) -> None:
    """Fail early with a clear message when the downloaded file is not a playable video."""
    try:
        with av.open(BytesIO(data)) as container:
            if not any(s.type == "video" for s in container.streams):
                raise MediaError("The generated file has no video track.")
    except MediaError:
        raise
    except Exception:
        raise MediaError("The generated video could not be read.") from None


def video_to_comfy(data: bytes) -> Any:
    """ComfyUI's VIDEO value for an MP4 (or other container) held in memory."""
    from comfy_api.latest import InputImpl

    check_video(data)
    return InputImpl.VideoFromFile(BytesIO(data))
