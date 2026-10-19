"""Where the nodes find the EmpirioLabs AI API key.

The key is never a node input. ComfyUI saves node inputs into workflow files and
into the metadata of the images it saves, so a key typed into a node would travel
with every shared workflow and picture. The key comes from the environment, or
from a config.ini file in this pack's folder that git ignores.
"""

from __future__ import annotations

import configparser
import os
from collections.abc import Mapping
from pathlib import Path

ENV_VAR = "EMPIRIOLABS_API_KEY"
CONFIG_FILE_NAME = "config.ini"
CONFIG_SECTION = "empiriolabs"
CONFIG_OPTION = "api_key"

PACK_DIR = Path(__file__).resolve().parent.parent

KEY_HELP = (
    f"Set the {ENV_VAR} environment variable before you start ComfyUI, or copy "
    f"config.example.ini to {CONFIG_FILE_NAME} in the EmpirioLabs AI node folder and paste "
    "your key there. Create a key at https://platform.empiriolabs.ai/dashboard/api-keys."
)


class ApiKeyError(RuntimeError):
    """No usable API key was found. The message never contains a key."""


def config_path() -> Path:
    return PACK_DIR / CONFIG_FILE_NAME


def read_api_key(environ: Mapping[str, str] | None = None, path: Path | None = None) -> str | None:
    """Return the API key from the environment or config.ini, or None when neither has one."""
    environ = os.environ if environ is None else environ
    value = (environ.get(ENV_VAR) or "").strip()
    if value:
        return value

    path = config_path() if path is None else path
    if not path.is_file():
        return None

    parser = configparser.ConfigParser(interpolation=None)
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            parser.read_file(handle)
    except (OSError, UnicodeDecodeError, configparser.Error):
        # The parser's own message quotes the offending line, which may be the key itself,
        # so it is never passed on.
        raise ApiKeyError(
            f"{CONFIG_FILE_NAME} in the EmpirioLabs AI node folder could not be read. It needs an "
            f"[{CONFIG_SECTION}] section with an {CONFIG_OPTION} line, as in config.example.ini."
        ) from None

    value = parser.get(CONFIG_SECTION, CONFIG_OPTION, fallback="").strip()
    return value or None


def require_api_key() -> str:
    key = read_api_key()
    if not key:
        raise ApiKeyError("No EmpirioLabs AI API key was found. " + KEY_HELP)
    return key
