"""Reload this pack in a running ComfyUI. Loaded only when a dev.enable file sits beside it.

POST /h3take/reload re-imports the pack and swaps its node classes. Models stay
loaded. It also resets ComfyUI's execution caches, because those hold instances
of the old classes. Refresh the browser afterwards to pick up changed inputs and JS.
"""

import importlib
import logging
import sys

MODULES = ("policy", "latent_writer", "prompting", "h3lora", "media", "ggml_dequant", "ggml_ops", "hyperflow", "project",
           "editor_api", "nodes", "preview")


def install(package):
    from aiohttp import web
    from server import PromptServer

    server = PromptServer.instance

    @server.routes.post("/h3take/reload")
    async def reload_pack(request):
        if server.prompt_queue.get_tasks_remaining() > 0:
            return web.json_response({"reloaded": False, "reason": "a prompt is running"}, status=409)
        core = sys.modules["nodes"]
        try:
            # A module new since ComfyUI started is imported for the first time.
            loaded = [importlib.reload(sys.modules[full]) if full in sys.modules else importlib.import_module(full)
                      for full in ("%s.%s" % (package, name) for name in MODULES)]
        except Exception as error:
            logging.exception("H3 Take reload failed")
            return web.json_response({"reloaded": False, "reason": repr(error)}, status=500)
        names = []
        for module in loaded:
            for node_id, cls in getattr(module, "NODE_CLASS_MAPPINGS", {}).items():
                old = core.NODE_CLASS_MAPPINGS.get(node_id)
                cls.RELATIVE_PYTHON_MODULE = getattr(old, "RELATIVE_PYTHON_MODULE", "custom_nodes.comfyui-h3-take")
                core.NODE_CLASS_MAPPINGS[node_id] = cls
                core.NODE_DISPLAY_NAME_MAPPINGS[node_id] = module.NODE_DISPLAY_NAME_MAPPINGS.get(node_id, node_id)
                names.append(node_id)
        server.prompt_queue.set_flag("unload_models", False)
        server.prompt_queue.set_flag("free_memory", True)
        return web.json_response({"reloaded": True, "nodes": names})
