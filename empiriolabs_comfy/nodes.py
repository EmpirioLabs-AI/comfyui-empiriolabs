"""The EmpirioLabs AI nodes, written against ComfyUI's node API (comfy_api.latest)."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import inspect
import json
import time
from collections.abc import Awaitable
from typing import Any, TypeVar

import torch
from comfy_api.latest import ComfyExtension, io

from . import catalog, client, config, media
from .client import EmpirioLabsError

CATEGORY = "EmpirioLabs AI"
DEFAULT_OPTION = "default"
SEED_MAX = 2**31 - 1

SYNC_IMAGE_TIMEOUT = 330.0  # the API holds a synchronous image request for about four minutes
IMAGE_JOB_TIMEOUT = 15 * 60.0
VIDEO_JOB_TIMEOUT = 60 * 60.0
SPEECH_TIMEOUT = 300.0
SPEECH_JOB_TIMEOUT = 15 * 60.0
CHAT_TIMEOUT = 600.0

MODELS_PAGE = "https://empiriolabs.ai/models"
EXTRA_TOOLTIP = (
    'Optional request fields as a JSON object, for example {"negative_prompt": "blurry"}. '
    f"Each model lists the fields it accepts on its page at {MODELS_PAGE}. "
    "Fields set here take precedence over the dropdowns above."
)
SEED_TOOLTIP = "Change the seed to run the node again. Models that accept a seed also receive it."
FALLBACK_NOTE = (
    " The model list could not be loaded from the EmpirioLabs AI catalog, so only one model is "
    "shown. Check the internet connection and refresh the page."
)

T = TypeVar("T")


# --------------------------------------------------------------------------- schema helpers


def _accepts(target: Any, name: str) -> bool:
    try:
        return name in inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False


# Newer ComfyUI releases can fold rarely used inputs away and match extra search words.
# Older releases do not take these arguments, so they are only passed where accepted.
_ADVANCED = {"advanced": True} if _accepts(io.String.Input, "advanced") else {}
_SCHEMA_FIELDS = {field.name for field in dataclasses.fields(io.Schema)}


def _schema(**kwargs: Any) -> io.Schema:
    aliases = kwargs.pop("search_aliases", None)
    if aliases and "search_aliases" in _SCHEMA_FIELDS:
        kwargs["search_aliases"] = aliases
    return io.Schema(**kwargs)


def _model_input(kind: str, tooltip: str) -> io.Combo.Input:
    catalog.maybe_refresh()
    if not catalog.is_live():
        tooltip += FALLBACK_NOTE
    return io.Combo.Input(
        "model", options=catalog.model_ids(kind), default=catalog.default_model(kind), tooltip=tooltip
    )


def _choice_input(input_id: str, kind: str, tooltip: str) -> io.Combo.Input:
    options = [DEFAULT_OPTION, *catalog.option_values(kind, input_id)]
    return io.Combo.Input(input_id, options=options, default=DEFAULT_OPTION, tooltip=tooltip)


def _seed_input() -> io.Int.Input:
    return io.Int.Input(
        "seed", default=0, min=0, max=SEED_MAX, control_after_generate=True, tooltip=SEED_TOOLTIP
    )


def _extra_input() -> io.String.Input:
    return io.String.Input(
        "extra_parameters", multiline=True, default="", optional=True, tooltip=EXTRA_TOOLTIP, **_ADVANCED
    )


def _accept_any_text(*values: Any) -> bool | str:
    # The dropdowns come from the live catalog, which can differ from the list a workflow was
    # saved with. Any non-empty value is accepted here and the API decides, so a saved workflow
    # still runs when the catalog could not be loaded; the API rejects ids it does not serve.
    for value in values:
        if value is not None and (not isinstance(value, str) or not value.strip()):
            return "Choose a value from the list."
    return True


# --------------------------------------------------------------------------- run helpers


def _api_key() -> str:
    return config.require_api_key()


def _require_text(value: Any, message: str) -> str:
    text = value if isinstance(value, str) else ""
    if not text.strip():
        raise EmpirioLabsError(message)
    return text


def _extra_parameters(text: Any, reserved: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise EmpirioLabsError(f"extra_parameters is not valid JSON ({exc}).") from None
    if not isinstance(value, dict):
        raise EmpirioLabsError('extra_parameters must be a JSON object, for example {"negative_prompt": "blurry"}.')
    clash = [key for key in value if key in reserved]
    if clash:
        raise EmpirioLabsError(
            f"extra_parameters cannot set {', '.join(clash)}, because the node sets {'it' if len(clash) == 1 else 'them'}."
        )
    return value


def _chosen(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text != DEFAULT_OPTION else None


def _format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60:02d}s"


def _show_text(node_cls: type[io.ComfyNode], text: str) -> None:
    """Show a status line on the node in the ComfyUI interface. Does nothing outside ComfyUI."""
    hidden = getattr(node_cls, "hidden", None)
    node_id = getattr(hidden, "unique_id", None) if hidden is not None else None
    if not node_id:
        return
    with contextlib.suppress(Exception):  # the status line is informational only
        from server import PromptServer

        PromptServer.instance.send_progress_text(text, node_id)


class _Progress:
    """ComfyUI's progress bar on the node, when one is available."""

    def __init__(self, total: int = 100) -> None:
        try:
            from comfy.utils import ProgressBar

            self._bar: Any = ProgressBar(total)
        except Exception:  # noqa: BLE001
            self._bar = None

    def update(self, value: float) -> None:
        if self._bar is not None:
            with contextlib.suppress(Exception):
                self._bar.update_absolute(int(value))


def _is_interrupt(exc_type: type[BaseException]) -> bool:
    return issubclass(exc_type, client.Interrupted) or exc_type.__name__ == "InterruptProcessingException"


class _Status:
    """Keeps a status line and the elapsed time on the node while it runs, then the outcome."""

    def __init__(self, node_cls: type[io.ComfyNode], text: str) -> None:
        self.node_cls = node_cls
        self.text = text
        self.start = time.monotonic()
        self.finished = False
        self._task: asyncio.Task | None = None

    def set(self, text: str) -> None:
        if not self.finished:
            self.text = text
            self._push()

    def finish(self, text: str) -> None:
        """Replace the running status with the final line."""
        self.finished = True
        self._stop_ticker()
        _show_text(self.node_cls, text)

    def elapsed(self) -> str:
        return _format_elapsed(time.monotonic() - self.start)

    def _push(self) -> None:
        _show_text(self.node_cls, f"{self.text}\nElapsed: {self.elapsed()}")

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            self._push()

    def _stop_ticker(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def __aenter__(self) -> _Status:
        self._push()
        self._task = asyncio.ensure_future(self._tick())
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, *rest: Any) -> None:
        self._stop_ticker()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if exc_type is not None and not self.finished:
            outcome = "Cancelled" if _is_interrupt(exc_type) else "Failed"
            _show_text(self.node_cls, f"{outcome} after {self.elapsed()}.")


async def _guard(awaitable: Awaitable[T]) -> T:
    """Run a node body and turn a cancelled run into ComfyUI's own interrupt."""
    try:
        return await awaitable
    except client.Interrupted:
        client.raise_interrupt()


def _job_reporter(status: _Status, progress: _Progress, label: str):
    def report(update: client.JobUpdate) -> None:
        line = f"{label}: {update.status}"
        if update.progress is not None:
            percent = int(round(update.progress * 100))
            line += f", {percent}%"
            progress.update(percent)
        status.set(f"{line}\nJob ID: {update.job_id}")

    return report


# --------------------------------------------------------------------------- Generate Image


class GenerateImage(io.ComfyNode):
    RESERVED = ("model", "prompt", "sync", "wait")

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _schema(
            node_id="EmpirioLabsGenerateImage",
            display_name="EmpirioLabs AI Generate Image",
            category=CATEGORY,
            description=(
                "Creates images from a text prompt with an image model from the EmpirioLabs AI catalog. "
                "Each image uses credits from your EmpirioLabs AI account."
            ),
            search_aliases=["text to image", "txt2img", "empiriolabs", "empirio"],
            inputs=[
                _model_input("image", f"Image model from the EmpirioLabs AI catalog. Prices and details: {MODELS_PAGE}"),
                io.String.Input("prompt", multiline=True, default="", tooltip="Description of the image to create."),
                _choice_input(
                    "aspect_ratio",
                    "image",
                    "Output aspect ratio. default uses the model's default. Supported values depend on the model.",
                ),
                _choice_input(
                    "resolution",
                    "image",
                    "Output resolution. default uses the model's default. Supported values depend on the model.",
                ),
                io.Int.Input(
                    "num_images", default=1, min=1, max=4, tooltip="Number of images to create. Each image is billed."
                ),
                _seed_input(),
                _extra_input(),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.String.Output(
                    display_name="url",
                    tooltip=(
                        "Link to the first image, valid for 7 days. Connect it to image_url on "
                        "EmpirioLabs AI Generate Video to animate the image."
                    ),
                ),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def validate_inputs(cls, model=None, aspect_ratio=None, resolution=None) -> bool | str:
        return _accept_any_text(model, aspect_ratio, resolution)

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        aspect_ratio: str = DEFAULT_OPTION,
        resolution: str = DEFAULT_OPTION,
        num_images: int = 1,
        seed: int = 0,
        extra_parameters: str | None = "",
    ) -> io.NodeOutput:
        return await _guard(
            cls._run(model, prompt, aspect_ratio, resolution, num_images, seed, extra_parameters)
        )

    @classmethod
    async def _run(cls, model, prompt, aspect_ratio, resolution, num_images, seed, extra_parameters) -> io.NodeOutput:
        api_key = _api_key()
        body: dict[str, Any] = {"model": model, "prompt": _require_text(prompt, "Enter a prompt.")}
        if _chosen(aspect_ratio):
            body["aspect_ratio"] = _chosen(aspect_ratio)
        if _chosen(resolution):
            body["resolution"] = _chosen(resolution)
        if num_images and int(num_images) > 1:
            body["num_images"] = int(num_images)
        model_seed = catalog.seed_for(model, seed)
        if model_seed is not None:
            body["seed"] = model_seed
        body.update(_extra_parameters(extra_parameters, cls.RESERVED))
        body.setdefault("response_format", "url")
        body["sync"] = True

        progress = _Progress()
        async with _Status(cls, "Generating the image") as status:
            payload = await client.api_json(
                "POST", "/v1/images/generations", api_key=api_key, body=body, timeout=SYNC_IMAGE_TIMEOUT
            )
            if client.is_job_envelope(payload):
                job = await client.wait_for_job(
                    payload["job_id"],
                    api_key=api_key,
                    timeout=IMAGE_JOB_TIMEOUT,
                    interval=3.0,
                    on_update=_job_reporter(status, progress, "Image job"),
                )
                payload = client.job_result(job)
            refs = client.media_refs(payload)
            if not refs:
                raise EmpirioLabsError("The response did not include any images. Try again, or try another prompt or model.")
            status.set("Downloading the image" if len(refs) == 1 else f"Downloading {len(refs)} images")
            files = [(await client.load_ref(ref, kind="image", timeout=180.0))[0] for ref in refs]
            images = media.images_to_tensor(files)
            progress.update(100)
            status.finish(f"{len(files)} image{'s' if len(files) != 1 else ''} ready in {status.elapsed()}.")
        url = next((ref.url for ref in refs if ref.url), "")
        return io.NodeOutput(images, url)


# --------------------------------------------------------------------------- Generate Video


class GenerateVideo(io.ComfyNode):
    RESERVED = ("model", "prompt")

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _schema(
            node_id="EmpirioLabsGenerateVideo",
            display_name="EmpirioLabs AI Generate Video",
            category=CATEGORY,
            description=(
                "Creates a video from a text prompt, or from a start image link, with a video model from the "
                "EmpirioLabs AI catalog. Video generation usually takes 1 to 15 minutes. Each video uses "
                "credits from your EmpirioLabs AI account."
            ),
            search_aliases=["text to video", "image to video", "t2v", "i2v", "empiriolabs", "empirio"],
            inputs=[
                _model_input("video", f"Video model from the EmpirioLabs AI catalog. Prices and details: {MODELS_PAGE}"),
                io.String.Input("prompt", multiline=True, default="", tooltip="Description of the video to create."),
                io.String.Input(
                    "image_url",
                    default="",
                    optional=True,
                    tooltip=(
                        "Optional public link to a start image for image-to-video. Connect the url output of "
                        "EmpirioLabs AI Generate Image here."
                    ),
                ),
                io.Int.Input(
                    "duration",
                    default=0,
                    min=0,
                    max=120,
                    tooltip="Video length in seconds. 0 uses the model's default. Supported lengths depend on the model.",
                ),
                _choice_input(
                    "resolution",
                    "video",
                    "Output resolution. default uses the model's default. Supported values depend on the model.",
                ),
                _choice_input(
                    "aspect_ratio",
                    "video",
                    "Output aspect ratio. default uses the model's default. Image-to-video usually follows the image.",
                ),
                _seed_input(),
                _extra_input(),
            ],
            outputs=[
                io.Video.Output(display_name="video"),
                io.String.Output(display_name="url", tooltip="Link to the video file, valid for 7 days."),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def validate_inputs(cls, model=None, resolution=None, aspect_ratio=None) -> bool | str:
        return _accept_any_text(model, resolution, aspect_ratio)

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        image_url: str | None = "",
        duration: int = 0,
        resolution: str = DEFAULT_OPTION,
        aspect_ratio: str = DEFAULT_OPTION,
        seed: int = 0,
        extra_parameters: str | None = "",
    ) -> io.NodeOutput:
        return await _guard(
            cls._run(model, prompt, image_url, duration, resolution, aspect_ratio, seed, extra_parameters)
        )

    @classmethod
    async def _run(
        cls, model, prompt, image_url, duration, resolution, aspect_ratio, seed, extra_parameters
    ) -> io.NodeOutput:
        api_key = _api_key()
        body: dict[str, Any] = {"model": model, "prompt": _require_text(prompt, "Enter a prompt.")}
        start_image = (image_url or "").strip() if isinstance(image_url, str) else ""
        if start_image:
            if not start_image.lower().startswith(("https://", "http://")):
                raise EmpirioLabsError("image_url must be a public http or https link to an image.")
            body["image"] = start_image
        if duration and int(duration) > 0:
            body["duration"] = int(duration)
        if _chosen(resolution):
            body["resolution"] = _chosen(resolution)
        if _chosen(aspect_ratio):
            body["aspect_ratio"] = _chosen(aspect_ratio)
        model_seed = catalog.seed_for(model, seed)
        if model_seed is not None:
            body["seed"] = model_seed
        body.update(_extra_parameters(extra_parameters, cls.RESERVED))

        progress = _Progress()
        async with _Status(cls, "Starting the video job") as status:
            started = await client.api_json("POST", "/v1/videos/generations", api_key=api_key, body=body, timeout=120.0)
            if client.is_job_envelope(started):
                job = await client.wait_for_job(
                    started["job_id"],
                    api_key=api_key,
                    timeout=VIDEO_JOB_TIMEOUT,
                    interval=5.0,
                    on_update=_job_reporter(status, progress, "Video job"),
                )
                payload = client.job_result(job)
            else:
                payload = started
            refs = client.media_refs(payload)
            if not refs:
                raise EmpirioLabsError("The finished job did not include a video. Try again, or try another model.")
            status.set("Downloading the video")
            data, _ = await client.load_ref(refs[0], kind="video", timeout=900.0)
            video = media.video_to_comfy(data)
            progress.update(100)
            status.finish(f"Video ready in {status.elapsed()}.")
        return io.NodeOutput(video, refs[0].url or "")


# --------------------------------------------------------------------------- Chat


def _reply_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise EmpirioLabsError("The response did not include a reply.")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, list):
        text = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    elif isinstance(content, str):
        text = content
    else:
        text = ""
    text = text.strip()
    if text:
        return text
    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal.strip():
        raise EmpirioLabsError(f"The model declined to answer: {refusal.strip()}")
    if choice.get("finish_reason") == "length":
        raise EmpirioLabsError(
            "The reply ran out of tokens before any text was written. Raise max_tokens, or set it to 0 to "
            "use the model's default."
        )
    raise EmpirioLabsError("The model returned an empty reply. Try again, or choose another model.")


class Chat(io.ComfyNode):
    RESERVED = ("model", "messages", "stream")

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _schema(
            node_id="EmpirioLabsChat",
            display_name="EmpirioLabs AI Chat",
            category=CATEGORY,
            description=(
                "Sends a prompt, and optionally images, to a chat model from the EmpirioLabs AI catalog and "
                "returns the reply as text."
            ),
            search_aliases=["llm", "text generation", "vision", "prompt", "empiriolabs", "empirio"],
            inputs=[
                _model_input("chat", f"Chat model from the EmpirioLabs AI catalog. Prices and details: {MODELS_PAGE}"),
                io.String.Input("prompt", multiline=True, default="", tooltip="The message to send to the model."),
                io.String.Input(
                    "system_prompt",
                    multiline=True,
                    default="",
                    optional=True,
                    tooltip="Optional instructions for the model, sent as the system message.",
                ),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip=(
                        "Optional image, or a batch of images, for models that accept image input. "
                        "The images are sent with the prompt."
                    ),
                ),
                io.Int.Input(
                    "max_tokens",
                    default=0,
                    min=0,
                    max=1_000_000,
                    tooltip="Maximum length of the reply in tokens. 0 uses the model's default.",
                ),
                _seed_input(),
                _extra_input(),
            ],
            outputs=[io.String.Output(display_name="text", tooltip="The model's reply.")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def validate_inputs(cls, model=None) -> bool | str:
        return _accept_any_text(model)

    @classmethod
    async def execute(
        cls,
        model: str,
        prompt: str,
        system_prompt: str | None = "",
        image: torch.Tensor | None = None,
        max_tokens: int = 0,
        seed: int = 0,
        extra_parameters: str | None = "",
    ) -> io.NodeOutput:
        return await _guard(cls._run(model, prompt, system_prompt, image, max_tokens, seed, extra_parameters))

    @classmethod
    async def _run(cls, model, prompt, system_prompt, image, max_tokens, seed, extra_parameters) -> io.NodeOutput:
        api_key = _api_key()
        prompt = _require_text(prompt, "Enter a prompt.")
        messages: list[dict[str, Any]] = []
        if isinstance(system_prompt, str) and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        if image is not None:
            entry = catalog.find_model(model)
            modalities = entry.get("input_modalities") if entry else None
            if isinstance(modalities, list) and "image" not in modalities:
                raise EmpirioLabsError(
                    f"{model} does not accept images. Choose a model that accepts image input, or disconnect the image."
                )
            content: Any = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": uri}} for uri in media.batch_to_data_uris(image)
            ]
        else:
            content = prompt
        messages.append({"role": "user", "content": content})

        body: dict[str, Any] = {"model": model, "messages": messages}
        if max_tokens and int(max_tokens) > 0:
            body["max_tokens"] = int(max_tokens)
        model_seed = catalog.seed_for(model, seed)
        if model_seed is not None:
            body["seed"] = model_seed
        body.update(_extra_parameters(extra_parameters, cls.RESERVED))
        body["stream"] = False

        async with _Status(cls, "Waiting for the reply") as status:
            payload = await client.api_json(
                "POST", "/v1/chat/completions", api_key=api_key, body=body, timeout=CHAT_TIMEOUT
            )
            if client.is_job_envelope(payload):
                job = await client.wait_for_job(
                    payload["job_id"],
                    api_key=api_key,
                    timeout=CHAT_TIMEOUT,
                    interval=2.0,
                    on_update=_job_reporter(status, _Progress(), "Chat job"),
                )
                payload = client.job_result(job)
            text = _reply_text(payload)
            status.finish(text)
        return io.NodeOutput(text)


# --------------------------------------------------------------------------- Text to Speech


class TextToSpeech(io.ComfyNode):
    RESERVED = ("model", "input")

    @classmethod
    def define_schema(cls) -> io.Schema:
        return _schema(
            node_id="EmpirioLabsTextToSpeech",
            display_name="EmpirioLabs AI Text to Speech",
            category=CATEGORY,
            description="Turns text into speech with a text to speech model from the EmpirioLabs AI catalog.",
            search_aliases=["tts", "speech", "voice", "audio", "empiriolabs", "empirio"],
            inputs=[
                _model_input(
                    "speech", f"Text to speech model from the EmpirioLabs AI catalog. Prices and details: {MODELS_PAGE}"
                ),
                io.String.Input("text", multiline=True, default="", tooltip="The text to speak."),
                io.String.Input(
                    "voice",
                    default="",
                    optional=True,
                    tooltip=(
                        "Voice for the selected model. Leave empty to use the model's default voice. "
                        f"Each model lists its voices on its page at {MODELS_PAGE}."
                    ),
                ),
                _extra_input(),
            ],
            outputs=[io.Audio.Output(display_name="audio")],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def validate_inputs(cls, model=None) -> bool | str:
        return _accept_any_text(model)

    @classmethod
    async def execute(
        cls, model: str, text: str, voice: str | None = "", extra_parameters: str | None = ""
    ) -> io.NodeOutput:
        return await _guard(cls._run(model, text, voice, extra_parameters))

    @classmethod
    async def _run(cls, model, text, voice, extra_parameters) -> io.NodeOutput:
        api_key = _api_key()
        body: dict[str, Any] = {"model": model, "input": _require_text(text, "Enter the text to speak.")}
        if isinstance(voice, str) and voice.strip():
            body["voice"] = voice.strip()
        body.update(_extra_parameters(extra_parameters, cls.RESERVED))

        async with _Status(cls, "Generating speech") as status:
            _, headers, raw = await client.api_call(
                "POST",
                "/v1/audio/speech",
                api_key=api_key,
                body=body,
                timeout=SPEECH_TIMEOUT,
                accept="application/json, audio/*",
            )
            content_type = headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type.startswith("audio/") or (content_type != "application/json" and raw[:1] not in (b"{", b"[")):
                data = raw
            else:
                payload = client.parse_json(raw)
                if not isinstance(payload, dict):
                    raise EmpirioLabsError("EmpirioLabs AI returned a response the node could not read.")
                if client.is_job_envelope(payload):
                    job = await client.wait_for_job(
                        payload["job_id"],
                        api_key=api_key,
                        timeout=SPEECH_JOB_TIMEOUT,
                        interval=3.0,
                        on_update=_job_reporter(status, _Progress(), "Speech job"),
                    )
                    payload = client.job_result(job)
                refs = client.media_refs(payload)
                if not refs:
                    raise EmpirioLabsError("The response did not include any audio. Try again, or try another voice or model.")
                status.set("Downloading the audio")
                data, _ = await client.load_ref(refs[0], kind="audio", timeout=SPEECH_TIMEOUT)
            audio = media.audio_to_comfy(data)
            status.finish(f"Speech ready in {status.elapsed()}.")
        return io.NodeOutput(audio)


# --------------------------------------------------------------------------- extension

NODES: list[type[io.ComfyNode]] = [GenerateImage, GenerateVideo, Chat, TextToSpeech]


class EmpirioLabsExtension(ComfyExtension):
    async def on_load(self) -> None:
        await catalog.ensure_loaded()

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        # Releases of ComfyUI that predate on_load() only call this method.
        await catalog.ensure_loaded()
        return list(NODES)


async def comfy_entrypoint() -> EmpirioLabsExtension:
    return EmpirioLabsExtension()
