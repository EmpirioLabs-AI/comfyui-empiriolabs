from __future__ import annotations

import base64
import io

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("av")

from PIL import Image  # noqa: E402

from empiriolabs_comfy import media  # noqa: E402
from fakes import mp4_bytes, png_bytes, wav_bytes, webp_bytes  # noqa: E402


def test_images_become_one_rgb_batch():
    batch = media.images_to_tensor([png_bytes(40, 30), webp_bytes(40, 30)])
    assert tuple(batch.shape) == (2, 30, 40, 3)
    assert batch.dtype == torch.float32
    assert 0.0 <= float(batch.min()) and float(batch.max()) <= 1.0
    assert abs(float(batch[0, 0, 0, 0]) - 200 / 255) < 1e-6


def test_different_sizes_are_fitted_to_the_first_image():
    batch = media.images_to_tensor([png_bytes(40, 30), png_bytes(80, 20)])
    assert tuple(batch.shape) == (2, 30, 40, 3)


def test_transparency_is_kept_as_an_alpha_channel():
    batch = media.images_to_tensor([png_bytes(8, 8, (10, 20, 30, 0), mode="RGBA"), png_bytes(8, 8)])
    assert tuple(batch.shape) == (2, 8, 8, 4)
    assert float(batch[0, 0, 0, 3]) == 0.0 and float(batch[1, 0, 0, 3]) == 1.0


def test_undecodable_image_raises_a_clear_error():
    with pytest.raises(media.MediaError, match="could not be decoded"):
        media.images_to_tensor([b"not an image"])
    with pytest.raises(media.MediaError):
        media.images_to_tensor([])


def _decode_data_uri(uri):
    header, _, payload = uri.partition(",")
    return header, Image.open(io.BytesIO(base64.b64decode(payload)))


def test_image_to_data_uri_uses_jpeg_without_alpha():
    image = torch.zeros((16, 24, 3))
    image[..., 0] = 1.0
    header, decoded = _decode_data_uri(media.image_to_data_uri(image))
    assert header == "data:image/jpeg;base64"
    assert decoded.size == (24, 16)
    assert decoded.getpixel((5, 5))[0] > 240


def test_image_to_data_uri_keeps_alpha_as_png_and_downscales():
    image = torch.ones((40, 40, 4))
    header, decoded = _decode_data_uri(media.image_to_data_uri(image, max_pixels=400))
    assert header == "data:image/png;base64"
    assert decoded.mode == "RGBA"
    assert decoded.size[0] * decoded.size[1] <= 400


def test_batch_to_data_uris_sends_each_image():
    assert len(media.batch_to_data_uris(torch.zeros((3, 8, 8, 3)))) == 3
    assert len(media.batch_to_data_uris(torch.zeros((8, 8, 3)))) == 1
    assert media.batch_to_data_uris(torch.zeros((1, 8, 8, 1)))[0].startswith("data:image/jpeg")


def test_audio_is_decoded_to_comfyui_audio():
    audio = media.audio_to_comfy(wav_bytes(seconds=0.5, rate=16000, channels=2))
    waveform = audio["waveform"]
    assert audio["sample_rate"] == 16000
    assert tuple(waveform.shape[:2]) == (1, 2)
    assert abs(waveform.shape[2] - 8000) <= 1
    assert waveform.dtype == torch.float32
    assert 0.3 < float(waveform.abs().max()) < 0.4  # 12000 / 32768


def test_bad_audio_raises_a_clear_error():
    with pytest.raises(media.MediaError, match="could not be decoded"):
        media.audio_to_comfy(b"not audio")


def test_check_video_accepts_mp4_and_rejects_other_files():
    media.check_video(mp4_bytes())
    with pytest.raises(media.MediaError):
        media.check_video(wav_bytes())
    with pytest.raises(media.MediaError):
        media.check_video(b"garbage")


def test_video_becomes_a_comfyui_video():
    pytest.importorskip("comfy_api.latest")
    video = media.video_to_comfy(mp4_bytes(frames=6, width=64, height=48))
    assert video.get_dimensions() == (64, 48)
    components = video.get_components()
    assert tuple(components.images.shape[1:]) == (48, 64, 3)
