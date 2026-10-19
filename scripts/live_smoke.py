"""Live smoke test against the real EmpirioLabs AI API.

Runs the node code itself, outside the ComfyUI server, so it needs a ComfyUI Python
environment: set COMFYUI_PATH to a ComfyUI checkout. The API key comes from the
EMPIRIOLABS_API_KEY environment variable (or config.ini), as in ComfyUI.

Without --spend it only reads the public model catalog, which is free. Each paid
step must be asked for, and runs one small request:

    python scripts/live_smoke.py                       # catalog only, free
    python scripts/live_smoke.py --spend --image       # one 1:1 image
    python scripts/live_smoke.py --spend --image --chat --speech
    python scripts/live_smoke.py --spend --video pixverse-v6 --video-duration 1 --video-resolution 360p

Outputs are written to smoke_output/, which git ignores.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import wave
from pathlib import Path
from urllib.parse import urlsplit

PACK_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PACK_DIR / "smoke_output"


def _setup_paths() -> None:
    comfyui = os.environ.get("COMFYUI_PATH")
    candidates = [Path(comfyui)] if comfyui else []
    candidates.append(PACK_DIR.parent.parent)
    for candidate in candidates:
        if (candidate / "comfy_api" / "latest" / "__init__.py").is_file():
            sys.path.append(str(candidate.resolve()))
            break
    else:
        raise SystemExit("Set COMFYUI_PATH to a ComfyUI checkout; the nodes need its comfy_api package.")
    sys.path.insert(0, str(PACK_DIR))


def _host(url: str) -> str:
    return urlsplit(url).hostname or "(none)" if url else "(inline)"


def _save_png(images, path: Path) -> None:
    import numpy as np
    from PIL import Image

    array = (images[0].clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def _save_wav(audio, path: Path) -> None:
    import numpy as np

    waveform = audio["waveform"][0].clamp(-1, 1).numpy()  # [channels, samples]
    pcm = (waveform.T * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(waveform.shape[0])
        handle.setsampwidth(2)
        handle.setframerate(audio["sample_rate"])
        handle.writeframes(pcm.tobytes())


async def main(args: argparse.Namespace) -> int:
    from empiriolabs_comfy import catalog, config, nodes

    started = time.monotonic()
    await catalog.ensure_loaded(timeout=20)
    if not catalog.is_live():
        print("FAIL catalog: could not load https://api.empiriolabs.ai/v1/models")
        return 1
    for kind in ("image", "video", "chat", "speech"):
        ids = catalog.model_ids(kind)
        print(f"catalog {kind}: {len(ids)} models, default {catalog.default_model(kind)}")
    print(f"catalog loaded in {time.monotonic() - started:.1f}s")

    if not args.spend:
        print("Paid steps skipped. Add --spend and the steps to run.")
        return 0
    config.require_api_key()
    OUTPUT_DIR.mkdir(exist_ok=True)
    images = None
    failures = 0

    async def step(name, coro):
        nonlocal failures
        t0 = time.monotonic()
        try:
            result = await coro
        except Exception as exc:  # noqa: BLE001 - report every failure and continue
            failures += 1
            print(f"FAIL {name} after {time.monotonic() - t0:.1f}s: {type(exc).__name__}: {exc}")
            return None
        print(f"ok   {name} in {time.monotonic() - t0:.1f}s")
        return result

    if args.image:
        output = await step(
            f"image ({args.image})",
            nodes.GenerateImage.execute(
                model=args.image,
                prompt="A small glass greenhouse on a cliff above a calm sea at sunrise, soft light",
                aspect_ratio="1:1",
                resolution="default",
                num_images=1,
                seed=7,
                extra_parameters="",
            ),
        )
        if output is not None:
            images, url = output.result
            _save_png(images, OUTPUT_DIR / "image.png")
            print(f"     image tensor {tuple(images.shape)}, link host {_host(url)}")

    if args.chat:
        output = await step(
            f"chat ({args.chat}{', with the image' if images is not None else ''})",
            nodes.Chat.execute(
                model=args.chat,
                prompt="Describe this image in one short sentence." if images is not None else "Say hello in five words.",
                system_prompt="",
                image=images,
                max_tokens=0,
                seed=1,
                extra_parameters="",
            ),
        )
        if output is not None:
            (text,) = output.result
            print(f"     reply: {text[:160]!r}")

    if args.speech:
        output = await step(
            f"speech ({args.speech})",
            nodes.TextToSpeech.execute(
                model=args.speech, text="Hello from ComfyUI.", voice="", extra_parameters=""
            ),
        )
        if output is not None:
            (audio,) = output.result
            _save_wav(audio, OUTPUT_DIR / "speech.wav")
            seconds = audio["waveform"].shape[-1] / audio["sample_rate"]
            print(f"     audio {tuple(audio['waveform'].shape)} at {audio['sample_rate']} Hz, {seconds:.2f}s")

    if args.video:
        output = await step(
            f"video ({args.video})",
            nodes.GenerateVideo.execute(
                model=args.video,
                prompt="Gentle waves roll onto a sandy beach at sunset",
                image_url="",
                duration=args.video_duration,
                resolution=args.video_resolution,
                aspect_ratio="16:9",
                seed=3,
                extra_parameters=args.video_extra,
            ),
        )
        if output is not None:
            video, url = output.result
            path = OUTPUT_DIR / "video.mp4"
            video.save_to(str(path))
            width, height = video.get_dimensions()
            print(f"     video {width}x{height}, {path.stat().st_size} bytes, link host {_host(url)}")

    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spend", action="store_true", help="allow paid requests")
    parser.add_argument("--image", nargs="?", const="flux-2-klein-4b", help="image model (default flux-2-klein-4b)")
    parser.add_argument("--chat", nargs="?", const="qwen3-8-27b", help="chat model (default qwen3-8-27b)")
    parser.add_argument("--speech", nargs="?", const="tts-2-flash", help="speech model (default tts-2-flash)")
    parser.add_argument("--video", help="video model, for example pixverse-v6")
    parser.add_argument("--video-duration", type=int, default=1)
    parser.add_argument("--video-resolution", default="360p")
    parser.add_argument("--video-extra", default="", help="extra_parameters JSON for the video request")
    _setup_paths()
    sys.exit(asyncio.run(main(parser.parse_args())))
