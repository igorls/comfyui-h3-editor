"""H3 Take. Relative imports only: ComfyUI owns the top-level name `nodes`."""

import os

from . import editor_api
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

editor_api.install()

if os.path.exists(os.path.join(os.path.dirname(__file__), "dev.enable")):
    from .dev_reload import install

    install(__name__)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
