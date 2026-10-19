# EmpirioLabs AI nodes for ComfyUI

Generate images and videos, chat with text and vision models, and turn text into speech in ComfyUI with models from the [EmpirioLabs AI](https://empiriolabs.ai) catalog.

EmpirioLabs AI is a specialized AI inference and integration provider. It self-hosts open-source models on GPU infrastructure it manages, runs optimized endpoints for proprietary models, helps teams ship their own models to large audiences, and offers on-demand GPU Cloud instances and hosted AI agents. Its model catalog covers text, image, audio, video, transcription, embedding, reranking, research, and 3D models.

## Nodes

All nodes are in the **EmpirioLabs AI** category. Each model list loads from the EmpirioLabs AI model catalog when ComfyUI starts and shows only the models that support what the node does.

| Node | Outputs | What it does |
| --- | --- | --- |
| EmpirioLabs AI Generate Image | `images` (IMAGE), `url` (STRING) | Creates images from a prompt. Options: aspect ratio, resolution, number of images (1 to 4) and seed. `url` links to the first image. |
| EmpirioLabs AI Generate Video | `video` (VIDEO), `url` (STRING) | Creates a video from a prompt, or animates a start image given as a public link in `image_url`. Options: duration, resolution, aspect ratio and seed. The node shows the job status and progress while the video renders. |
| EmpirioLabs AI Chat | `text` (STRING) | Sends a prompt, an optional system prompt and optional images to a chat model and returns the reply. Images are sent with the prompt to models that accept image input. |
| EmpirioLabs AI Text to Speech | `audio` (AUDIO) | Turns text into speech. Leave `voice` empty for the model's default voice. |

Every node also has an optional `extra_parameters` field for any other request field a model accepts, written as a JSON object, for example `{"negative_prompt": "blurry"}`. Fields set there take precedence over the node's dropdowns. Each model lists the fields and voices it accepts on its page at [empiriolabs.ai/models](https://empiriolabs.ai/models).

To animate a generated image, connect the `url` output of Generate Image to the `image_url` input of Generate Video.

## Installation

The pack needs ComfyUI 0.3.48 or later and no extra Python packages.

**ComfyUI-Manager.** Once the pack is listed in the Comfy Registry, open **Manager**, choose **Custom Nodes Manager**, search for **EmpirioLabs AI**, install it and restart ComfyUI. With comfy-cli, run `comfy node install empiriolabs`.

**Git.** Clone the repository into ComfyUI's `custom_nodes` folder and restart ComfyUI:

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/EmpirioLabs-AI/comfyui-empiriolabs.git
```

## API key

1. Sign in or create an account at [platform.empiriolabs.ai](https://platform.empiriolabs.ai).
2. Open [API Keys](https://platform.empiriolabs.ai/dashboard/api-keys) in the dashboard and create a key. Keys start with `sk-empiriolabs-`.
3. Give the key to ComfyUI in one of these two ways.

**Environment variable.** Set `EMPIRIOLABS_API_KEY`, then restart ComfyUI:

```sh
# Windows (PowerShell or Command Prompt); open a new window afterwards
setx EMPIRIOLABS_API_KEY "sk-empiriolabs-..."

# macOS and Linux; add the line to ~/.zshrc or ~/.bashrc to keep it
export EMPIRIOLABS_API_KEY="sk-empiriolabs-..."
```

**Config file.** Copy `config.example.ini` to `config.ini` in the pack folder (`ComfyUI/custom_nodes/comfyui-empiriolabs`, or `ComfyUI/custom_nodes/empiriolabs` when installed from the registry) and paste the key after `api_key =`. The nodes read the file on every run, so no restart is needed. Git ignores `config.ini`, and it is never part of the published pack.

When both are set, the environment variable is used. The nodes never take the key as an input, because ComfyUI saves node inputs in workflow files and in the metadata of the images it saves.

## Example workflows

The pack adds two example workflows to ComfyUI's **Templates** panel, listed under the pack's name. You can also open them from the `example_workflows` folder.

- `empiriolabs_generate_image.json`: Generate Image, then Save Image.
- `empiriolabs_image_to_video.json`: Generate Image creates the first frame, Generate Video animates it, and Save Video saves the result.

## Usage notes

- **Credits.** Requests use the credits on your EmpirioLabs AI account. Prices are listed on the [pricing page](https://empiriolabs.ai/pricing). Each run with a new seed is a new request; set the seed's control after generate to `fixed` to reuse the last result while the inputs stay the same.
- **Waiting.** The node shows its status and the elapsed time. Video generation usually takes 1 to 15 minutes.
- **Cancelling.** Cancelling the run in ComfyUI stops the node and cancels the job it is waiting for. A generation that has already started may still be billed.
- **Generated file links expire 7 days after they are created.** Save the files you want to keep with ComfyUI's Save nodes.
- **Model lists.** The lists refresh in the background every few hours; reload the page to see new models. If the catalog cannot be reached when ComfyUI starts, each node lists one model until the catalog loads. A workflow saved with a model that is not in the current list still runs.
- **Errors.** The node shows the message returned by the API, such as an unsupported parameter for the selected model, with a request ID to quote to support.

## Data handling

The nodes send your prompts, inputs and API key only to the EmpirioLabs AI API at `https://api.empiriolabs.ai`, and download generated files only from the links the API returns. The pack does not store prompts, outputs or keys; ComfyUI saves the files you save with its own nodes. For how EmpirioLabs AI handles data, see the [privacy policy](https://empiriolabs.ai/privacy-policy).

## Development

The tests use a mocked API and never reach the network. Run them from a ComfyUI Python environment and point `COMFYUI_PATH` at the ComfyUI folder:

```sh
COMFYUI_PATH=/path/to/ComfyUI python -m pytest
```

`scripts/live_smoke.py` checks the pack against the live API. Without `--spend` it only reads the public model catalog; each paid step runs one small request:

```sh
COMFYUI_PATH=/path/to/ComfyUI EMPIRIOLABS_API_KEY=sk-empiriolabs-... python scripts/live_smoke.py --spend --image --chat --speech
```

Versions are published to the Comfy Registry by the [publish workflow](.github/workflows/publish.yml) when the version in `pyproject.toml` changes on `main`. It needs a Comfy Registry publishing key in the `REGISTRY_ACCESS_TOKEN` repository secret.

## Links

- Website: [empiriolabs.ai](https://empiriolabs.ai)
- Documentation: [docs.empiriolabs.ai](https://docs.empiriolabs.ai)
- API reference: [docs.empiriolabs.ai/api-reference-overview](https://docs.empiriolabs.ai/api-reference-overview)
- Models and prices: [empiriolabs.ai/models](https://empiriolabs.ai/models)
- Support: [support@empiriolabs.ai](mailto:support@empiriolabs.ai)

## License

[MIT](LICENSE)
