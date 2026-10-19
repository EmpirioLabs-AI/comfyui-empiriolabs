"""Every node path, with the HTTP layer replaced by FakeAPI."""

from __future__ import annotations

import base64
import json
import re

import pytest

pytest.importorskip("comfy_api.latest")
torch = pytest.importorskip("torch")

from comfy_api.latest import ComfyExtension  # noqa: E402

from empiriolabs_comfy import catalog, client, config, nodes  # noqa: E402
from fakes import FakeInterrupt, mp4_bytes, png_bytes, run, wav_bytes, webp_bytes  # noqa: E402

MEDIA = "https://media.empiriolabs.ai/files/2026/10/19/x"
IMAGE_URL = MEDIA + "/one.webp?token=a&expires=1"
IMAGE_URL_2 = MEDIA + "/two.png?token=b&expires=1"
VIDEO_URL = MEDIA + "/video.mp4?token=c&expires=1"
AUDIO_URL = MEDIA + "/speech.wav?token=d&expires=1"
EM_DASH = "\u2014"


def _inputs(node_cls):
    return {item.id: item for item in node_cls.GET_SCHEMA().inputs}


# --------------------------------------------------------------------------- schema


@pytest.mark.parametrize("node_cls", nodes.NODES)
def test_schema_names_and_branding(node_cls, live_catalog):
    schema = node_cls.GET_SCHEMA()
    assert schema.node_id.startswith("EmpirioLabs")
    assert schema.display_name.startswith("EmpirioLabs AI ")
    assert schema.category == "EmpirioLabs AI"
    assert not schema.is_api_node  # these nodes are billed by EmpirioLabs AI, not by Comfy credits
    info = node_cls.GET_NODE_INFO_V1()
    assert info["display_name"] == schema.display_name


@pytest.mark.parametrize("node_cls", nodes.NODES)
def test_no_node_takes_an_api_key_as_input(node_cls, live_catalog):
    credential = re.compile(r"(^|_)(api_?key|key|token|secret|password|auth|authorization|bearer|credentials?)($|_)")
    for input_id in _inputs(node_cls):
        assert not credential.search(input_id.lower()), input_id


@pytest.mark.parametrize("node_cls", nodes.NODES)
def test_user_facing_text_has_no_em_dash(node_cls, no_catalog):
    text = json.dumps(node_cls.GET_NODE_INFO_V1(), ensure_ascii=False)
    assert EM_DASH not in text


def test_model_dropdowns_come_from_the_catalog(live_catalog):
    image_model = _inputs(nodes.GenerateImage)["model"]
    assert image_model.options == ["flux-2-klein-4b", "seedream-5-0-lite"]
    assert image_model.default == "flux-2-klein-4b"
    assert _inputs(nodes.GenerateVideo)["model"].options == ["hunyuan-video-1-5", "pixverse-v6"]
    assert _inputs(nodes.Chat)["model"].options == ["qwen3-8-27b", "text-only-model"]
    assert _inputs(nodes.TextToSpeech)["model"].options == ["tts-2-flash"]
    aspect = _inputs(nodes.GenerateImage)["aspect_ratio"]
    assert aspect.options[0] == "default" and "21:9" in aspect.options and aspect.default == "default"
    assert "catalog could not be loaded" not in (image_model.tooltip or "")


def test_fallback_dropdown_is_labelled(no_catalog):
    model = _inputs(nodes.GenerateImage)["model"]
    assert model.options == [catalog.FALLBACK_MODELS["image"]]
    assert "could not be loaded" in model.tooltip


def test_saved_values_outside_the_current_list_still_validate(no_catalog):
    assert nodes.GenerateImage.validate_inputs(model="seedream-5-0-pro", aspect_ratio="21:9", resolution="4K") is True
    assert nodes.GenerateVideo.validate_inputs(model="kling-3-0-turbo", resolution="1080p", aspect_ratio=None) is True
    assert nodes.Chat.validate_inputs(model="  ") == "Choose a value from the list."


def test_extension_registers_the_four_nodes(monkeypatch):
    monkeypatch.setattr(catalog, "_state", catalog._State())

    async def fetch(timeout):
        raise OSError("offline")

    monkeypatch.setattr(catalog, "fetch", fetch)

    async def scenario():
        extension = await nodes.comfy_entrypoint()
        assert isinstance(extension, ComfyExtension)
        await extension.on_load()
        return await extension.get_node_list()

    node_list = run(scenario())
    assert [n.GET_SCHEMA().node_id for n in node_list] == [
        "EmpirioLabsGenerateImage",
        "EmpirioLabsGenerateVideo",
        "EmpirioLabsChat",
        "EmpirioLabsTextToSpeech",
    ]


# --------------------------------------------------------------------------- Generate Image


def _image(**overrides):
    args = dict(model="flux-2-klein-4b", prompt="a red bicycle", aspect_ratio="default", resolution="default",
                num_images=1, seed=0, extra_parameters="")
    args.update(overrides)
    return run(nodes.GenerateImage.execute(**args))


def test_generate_image_sync_answer(api, live_catalog):
    api.add("POST", "/v1/images/generations", (200, {"created": 1, "data": [{"url": IMAGE_URL, "content_type": "image/webp"}]}))
    api.add_file(IMAGE_URL, (webp_bytes(32, 24), "image/webp"))
    output = _image(seed=70000)
    images, url = output.result
    assert tuple(images.shape) == (1, 24, 32, 3)
    assert url == IMAGE_URL
    body = api.calls[0].body
    assert body == {
        "model": "flux-2-klein-4b",
        "prompt": "a red bicycle",
        "seed": 70000 % 65536,
        "response_format": "url",
        "sync": True,
    }
    assert api.downloads == [IMAGE_URL]


def test_generate_image_sends_the_chosen_options(api, live_catalog):
    api.add("POST", "/v1/images/generations", (200, {"data": [{"url": IMAGE_URL}]}))
    api.add_file(IMAGE_URL, (png_bytes(), "image/png"))
    _image(model="seedream-5-0-lite", aspect_ratio="21:9", resolution="4K", num_images=1, seed=9,
           extra_parameters='{"watermark": false, "resolution": "2K"}')
    body = api.calls[0].body
    assert body["aspect_ratio"] == "21:9"
    assert body["resolution"] == "2K"  # extra_parameters takes precedence
    assert body["watermark"] is False
    assert "seed" not in body  # this model takes no seed
    assert "num_images" not in body


def test_generate_image_falls_back_to_the_job(api, live_catalog):
    api.add("POST", "/v1/images/generations", (200, {"job_id": "img_1", "status": "processing"}))
    api.add(
        "GET",
        "/v1/jobs/img_1",
        (200, {"job_id": "img_1", "status": "processing"}),
        (200, {"job_id": "img_1", "status": "completed",
               "result": {"data": [{"url": IMAGE_URL}, {"url": IMAGE_URL_2}]}}),
    )
    api.add_file(IMAGE_URL, (webp_bytes(32, 24), "image/webp"))
    api.add_file(IMAGE_URL_2, (png_bytes(64, 64), "image/png"))
    images, url = _image(num_images=2).result
    assert tuple(images.shape) == (2, 24, 32, 3)
    assert url == IMAGE_URL
    assert api.calls[0].body["num_images"] == 2
    assert len(api.calls_to("GET", "/v1/jobs/img_1")) == 2


def test_generate_image_failed_job(api, live_catalog):
    api.add("POST", "/v1/images/generations", (200, {"job_id": "img_1", "status": "processing"}))
    api.add("GET", "/v1/jobs/img_1", (200, {"job_id": "img_1", "status": "failed", "error": {"message": "Content was blocked."}}))
    with pytest.raises(client.JobFailedError, match="Content was blocked"):
        _image()
    assert api.downloads == []


def test_generate_image_inline_base64(api, live_catalog):
    encoded = base64.b64encode(png_bytes(10, 10)).decode()
    api.add("POST", "/v1/images/generations", (200, {"data": [{"b64_json": encoded}]}))
    images, url = _image(extra_parameters='{"response_format": "b64_json"}').result
    assert tuple(images.shape) == (1, 10, 10, 3)
    assert url == ""
    assert api.calls[0].body["response_format"] == "b64_json"
    assert api.downloads == []


def test_generate_image_without_images_in_the_answer(api, live_catalog):
    api.add("POST", "/v1/images/generations", (200, {"data": []}))
    with pytest.raises(client.EmpirioLabsError, match="did not include any images"):
        _image()


def test_generate_image_api_error(api, live_catalog):
    api.add("POST", "/v1/images/generations", (402, {"error": {"message": "Insufficient credits.", "request_id": "req_9"}}))
    with pytest.raises(client.ApiError, match="Insufficient credits.*req_9"):
        _image()
    assert len(api.calls) == 1  # a POST is never retried


@pytest.mark.parametrize(
    "extra,message",
    [
        ("{not json", "not valid JSON"),
        ("[1, 2]", "must be a JSON object"),
        ('{"model": "other"}', "cannot set model"),
        ('{"sync": false, "prompt": "x"}', "cannot set sync, prompt"),
    ],
)
def test_generate_image_rejects_bad_extra_parameters(api, live_catalog, extra, message):
    with pytest.raises(client.EmpirioLabsError, match=message):
        _image(extra_parameters=extra)
    assert api.calls == []


def test_generate_image_needs_a_prompt(api, live_catalog):
    with pytest.raises(client.EmpirioLabsError, match="Enter a prompt"):
        _image(prompt="   ")
    assert api.calls == []


def test_missing_key_stops_before_any_request(api, live_catalog, monkeypatch, tmp_path):
    monkeypatch.delenv(config.ENV_VAR)
    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "config.ini")
    with pytest.raises(config.ApiKeyError, match="EMPIRIOLABS_API_KEY"):
        _image()
    assert api.calls == []


def test_cancelling_during_the_image_job_cancels_it(api, live_catalog):
    api.add("POST", "/v1/images/generations", (200, {"job_id": "img_1", "status": "processing"}))
    api.add("GET", "/v1/jobs/img_1", (200, {"job_id": "img_1", "status": "processing"}))
    api.add("DELETE", "/v1/jobs/img_1", (200, {"job_id": "img_1", "status": "cancelled"}))
    api.interrupt_after_calls = 2
    with pytest.raises(FakeInterrupt):
        _image()
    assert len(api.calls_to("DELETE", "/v1/jobs/img_1")) == 1


# --------------------------------------------------------------------------- Generate Video


def _video(**overrides):
    args = dict(model="hunyuan-video-1-5", prompt="waves at dusk", image_url="", duration=0, resolution="default",
                aspect_ratio="default", seed=0, extra_parameters="")
    args.update(overrides)
    return run(nodes.GenerateVideo.execute(**args))


def test_generate_video_polls_the_job_and_returns_a_video(api, live_catalog):
    api.add("POST", "/v1/videos/generations", (202, {"job_id": "vid_1", "status": "processing", "poll_url": "/v1/jobs/vid_1"}))
    api.add(
        "GET",
        "/v1/jobs/vid_1",
        (200, {"job_id": "vid_1", "status": "processing", "progress": 0.4}),
        (200, {"job_id": "vid_1", "status": "completed", "progress": 1.0,
               "result": {"data": [{"url": VIDEO_URL, "content_type": "video/mp4"}]}}),
    )
    api.add_file(VIDEO_URL, (mp4_bytes(), "video/mp4"))
    video, url = _video(image_url=" https://example.com/start.png ", duration=5, resolution="480p",
                        aspect_ratio="16:9", seed=42).result
    assert url == VIDEO_URL
    assert video.get_dimensions() == (64, 48)
    assert api.calls[0].body == {
        "model": "hunyuan-video-1-5",
        "prompt": "waves at dusk",
        "image": "https://example.com/start.png",
        "duration": 5,
        "resolution": "480p",
        "aspect_ratio": "16:9",
        "seed": 42,
    }
    assert len(api.calls_to("GET", "/v1/jobs/vid_1")) == 2


def test_generate_video_defaults_send_only_model_and_prompt(api, live_catalog):
    api.add("POST", "/v1/videos/generations", (202, {"job_id": "vid_1", "status": "processing"}))
    api.add("GET", "/v1/jobs/vid_1", (200, {"job_id": "vid_1", "status": "completed", "result": {"data": [{"url": VIDEO_URL}]}}))
    api.add_file(VIDEO_URL, (mp4_bytes(), "video/mp4"))
    _video(model="pixverse-v6", seed=5)
    assert api.calls[0].body == {"model": "pixverse-v6", "prompt": "waves at dusk"}


def test_generate_video_failed_job(api, live_catalog):
    api.add("POST", "/v1/videos/generations", (202, {"job_id": "vid_1", "status": "processing"}))
    api.add("GET", "/v1/jobs/vid_1", (200, {"job_id": "vid_1", "status": "failed", "error": {"message": "Duration not supported."}}))
    with pytest.raises(client.JobFailedError, match="Duration not supported.*vid_1"):
        _video()


def test_generate_video_cancel_cancels_the_job(api, live_catalog):
    api.add("POST", "/v1/videos/generations", (202, {"job_id": "vid_1", "status": "processing"}))
    api.add("GET", "/v1/jobs/vid_1", (200, {"job_id": "vid_1", "status": "processing", "progress": 0.1}))
    api.add("DELETE", "/v1/jobs/vid_1", (200, {"job_id": "vid_1", "status": "cancelled"}))
    api.interrupt_after_calls = 3
    with pytest.raises(FakeInterrupt):
        _video()
    assert len(api.calls_to("DELETE", "/v1/jobs/vid_1")) == 1


def test_generate_video_rejects_a_start_image_that_is_not_a_link(api, live_catalog):
    with pytest.raises(client.EmpirioLabsError, match="image_url must be"):
        _video(image_url="C:/images/start.png")
    assert api.calls == []


def test_generate_video_reports_a_job_without_a_video(api, live_catalog):
    api.add("POST", "/v1/videos/generations", (202, {"job_id": "vid_1", "status": "processing"}))
    api.add("GET", "/v1/jobs/vid_1", (200, {"job_id": "vid_1", "status": "completed", "result": {"data": []}}))
    with pytest.raises(client.EmpirioLabsError, match="did not include a video"):
        _video()


def test_generate_video_rejects_a_download_that_is_not_a_video(api, live_catalog):
    api.add("POST", "/v1/videos/generations", (202, {"job_id": "vid_1", "status": "processing"}))
    api.add("GET", "/v1/jobs/vid_1", (200, {"job_id": "vid_1", "status": "completed", "result": {"data": [{"url": VIDEO_URL}]}}))
    api.add_file(VIDEO_URL, (b"<html>expired</html>", "text/html"))
    with pytest.raises(ValueError, match="could not be read"):
        _video()


# --------------------------------------------------------------------------- Chat


def _chat(**overrides):
    args = dict(model="qwen3-8-27b", prompt="Write a prompt about a lighthouse.", system_prompt="", image=None,
                max_tokens=0, seed=0, extra_parameters="")
    args.update(overrides)
    return run(nodes.Chat.execute(**args))


def _reply(content, finish_reason="stop"):
    return (200, {"id": "chatcmpl-1", "choices": [{"index": 0, "finish_reason": finish_reason,
                                                    "message": {"role": "assistant", "content": content}}]})


def test_chat_returns_the_reply_text(api, live_catalog):
    api.add("POST", "/v1/chat/completions", _reply("\n\nA lighthouse at dawn.\n"))
    (text,) = _chat(system_prompt="Reply briefly.", max_tokens=200, seed=3).result
    assert text == "A lighthouse at dawn."
    assert api.calls[0].body == {
        "model": "qwen3-8-27b",
        "messages": [
            {"role": "system", "content": "Reply briefly."},
            {"role": "user", "content": "Write a prompt about a lighthouse."},
        ],
        "max_tokens": 200,
        "seed": 3,
        "stream": False,
    }


def test_chat_sends_images_as_data_uris(api, live_catalog):
    api.add("POST", "/v1/chat/completions", _reply("Red."))
    images = torch.zeros((2, 16, 16, 3))
    _chat(image=images, prompt="What color is this?")
    content = api.calls[0].body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "What color is this?"}
    assert len(content) == 3
    for part in content[1:]:
        assert part["type"] == "image_url"
        assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_chat_refuses_images_for_a_text_only_model(api, live_catalog):
    with pytest.raises(client.EmpirioLabsError, match="does not accept images"):
        _chat(model="text-only-model", image=torch.zeros((1, 8, 8, 3)))
    assert api.calls == []


def test_chat_joins_content_parts(api, live_catalog):
    api.add("POST", "/v1/chat/completions", _reply([{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}]))
    assert _chat().result == ("Part one. Part two.",)


def test_chat_explains_an_empty_reply_cut_off_by_max_tokens(api, live_catalog):
    api.add("POST", "/v1/chat/completions", _reply(None, finish_reason="length"))
    with pytest.raises(client.EmpirioLabsError, match="Raise max_tokens"):
        _chat(max_tokens=5)


def test_chat_refusal_is_reported(api, live_catalog):
    api.add("POST", "/v1/chat/completions", (200, {"choices": [{"message": {"content": "", "refusal": "I can't help with that."}}]}))
    with pytest.raises(client.EmpirioLabsError, match="declined to answer"):
        _chat()


def test_chat_extra_parameters_cannot_replace_the_messages(api, live_catalog):
    with pytest.raises(client.EmpirioLabsError, match="cannot set messages"):
        _chat(extra_parameters='{"messages": []}')
    api.add("POST", "/v1/chat/completions", _reply("ok"))
    _chat(extra_parameters='{"temperature": 0.2, "reasoning_effort": "low"}')
    body = api.calls[-1].body
    assert body["temperature"] == 0.2 and body["reasoning_effort"] == "low" and body["stream"] is False


def test_chat_job_envelope_is_polled(api, live_catalog):
    api.add("POST", "/v1/chat/completions", (200, {"job_id": "chat_1", "status": "processing"}))
    api.add("GET", "/v1/jobs/chat_1", (200, {"job_id": "chat_1", "status": "completed",
                                             "result": {"choices": [{"message": {"content": "Done."}}]}}))
    assert _chat().result == ("Done.",)


# --------------------------------------------------------------------------- Text to Speech


def _speech(**overrides):
    args = dict(model="tts-2-flash", text="Hello from ComfyUI.", voice="", extra_parameters="")
    args.update(overrides)
    return run(nodes.TextToSpeech.execute(**args))


def test_speech_downloads_the_audio_link(api, live_catalog):
    api.add("POST", "/v1/audio/speech", (200, {"data": [{"url": AUDIO_URL, "format": "wav", "duration": 0.5}]}))
    api.add_file(AUDIO_URL, (wav_bytes(seconds=0.5), "audio/wav"))
    (audio,) = _speech(voice=" Sarah ").result
    assert audio["sample_rate"] == 16000 and audio["waveform"].shape[0] == 1
    assert api.calls[0].body == {"model": "tts-2-flash", "input": "Hello from ComfyUI.", "voice": "Sarah"}


def test_speech_accepts_audio_bytes_in_the_response(api, live_catalog):
    api.add("POST", "/v1/audio/speech", (200, wav_bytes(), "audio/wav"))
    (audio,) = _speech().result
    assert audio["waveform"].shape[1] == 1
    assert api.downloads == []
    assert "voice" not in api.calls[0].body


def test_speech_job_envelope_is_polled(api, live_catalog):
    api.add("POST", "/v1/audio/speech", (200, {"job_id": "tts_1", "status": "processing"}))
    api.add("GET", "/v1/jobs/tts_1", (200, {"job_id": "tts_1", "status": "completed", "result": {"data": [{"url": AUDIO_URL}]}}))
    api.add_file(AUDIO_URL, (wav_bytes(), "audio/wav"))
    (audio,) = _speech(extra_parameters='{"output_format": "WAV"}').result
    assert audio["sample_rate"] == 16000
    assert api.calls[0].body["output_format"] == "WAV"


def test_speech_without_audio_in_the_answer(api, live_catalog):
    api.add("POST", "/v1/audio/speech", (200, {"data": []}))
    with pytest.raises(client.EmpirioLabsError, match="did not include any audio"):
        _speech()


def test_speech_needs_text(api, live_catalog):
    with pytest.raises(client.EmpirioLabsError, match="Enter the text to speak"):
        _speech(text="")


# --------------------------------------------------------------------------- status helpers


def test_job_reporter_updates_the_status_and_progress():
    lines, values = [], []

    class Status:
        def set(self, text):
            lines.append(text)

    class Progress:
        def update(self, value):
            values.append(value)

    report = nodes._job_reporter(Status(), Progress(), "Video job")
    report(client.JobUpdate("vid_1", "processing", 0.42, 12.0))
    report(client.JobUpdate("vid_1", "queued", None, 13.0))
    assert lines == ["Video job: processing, 42%\nJob ID: vid_1", "Video job: queued\nJob ID: vid_1"]
    assert values == [42]


def test_elapsed_time_format():
    assert nodes._format_elapsed(7.9) == "7s"
    assert nodes._format_elapsed(65) == "1m 05s"


@pytest.fixture
def node_text(monkeypatch):
    """Records the status lines the nodes would show in the ComfyUI interface."""
    lines = []
    monkeypatch.setattr(nodes, "_show_text", lambda node_cls, text: lines.append(text))
    return lines


def test_status_line_reports_success(api, live_catalog, node_text):
    api.add("POST", "/v1/images/generations", (200, {"data": [{"url": IMAGE_URL}]}))
    api.add_file(IMAGE_URL, (png_bytes(), "image/png"))
    _image()
    assert node_text[0].startswith("Generating the image\nElapsed: ")
    assert node_text[-1].startswith("1 image ready in ")


def test_status_line_reports_a_failure(api, live_catalog, node_text):
    api.add("POST", "/v1/images/generations", (500, {"error": {"message": "boom"}}))
    with pytest.raises(client.ApiError):
        _image()
    assert node_text[-1].startswith("Failed after ")


def test_status_line_reports_a_cancelled_run(api, live_catalog, node_text):
    api.add("POST", "/v1/videos/generations", (202, {"job_id": "vid_1", "status": "processing"}))
    api.add("GET", "/v1/jobs/vid_1", (200, {"job_id": "vid_1", "status": "processing"}))
    api.add("DELETE", "/v1/jobs/vid_1", (200, {"job_id": "vid_1", "status": "cancelled"}))
    api.interrupt_after_calls = 2
    with pytest.raises(FakeInterrupt):
        _video()
    assert "Video job: processing\nJob ID: vid_1" in "\n".join(node_text)
    assert node_text[-1].startswith("Cancelled after ")


def test_chat_reply_is_shown_on_the_node(api, live_catalog, node_text):
    api.add("POST", "/v1/chat/completions", _reply("Hello there."))
    _chat()
    assert node_text[-1] == "Hello there."
