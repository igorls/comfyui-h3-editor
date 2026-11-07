"""H3 Take. Relative imports only: ComfyUI owns the top-level name `nodes`."""

import os

from . import editor_api
from .nodes import NODE_CLASS_MAPPINGS as TAKE_NODES
from .nodes import NODE_DISPLAY_NAME_MAPPINGS as TAKE_NAMES
from .preview import NODE_CLASS_MAPPINGS as PREVIEW_NODES
from .preview import NODE_DISPLAY_NAME_MAPPINGS as PREVIEW_NAMES

NODE_CLASS_MAPPINGS = {**TAKE_NODES, **PREVIEW_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**TAKE_NAMES, **PREVIEW_NAMES}
WEB_DIRECTORY = "./web"

editor_api.install()

if os.path.exists(os.path.join(os.path.dirname(__file__), "dev.enable")):
    from .dev_reload import install

    install(__name__)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
