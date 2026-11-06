"""Timeline editor. The widget draws the plan. This node only stores it and names the passes."""

import json

from policy import max_block_for_vram, split_span


DEFAULT = {
    "fps": 24,
    "vram_gb": 16,
    "hold": {"start": 124, "length": 124, "label": "Source"},
    "generates": [
        {"id": "before", "start": 0, "length": 124, "label": "Before"},
        {"id": "after", "start": 248, "length": 243, "label": "After"},
    ],
}


class H3Timeline:
    """A video-editor timeline for one H3 take. Does not load a model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline_data": ("STRING", {"multiline": True, "default": json.dumps(DEFAULT)}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("plan",)
    FUNCTION = "describe"
    CATEGORY = "minimax/h3"
    DESCRIPTION = (
        "Place a held source and the generate regions before and after it. "
        "Long regions split into sequential passes that fit the VRAM cap. "
        "This node does not sample."
    )

    def describe(self, timeline_data):
        try:
            data = json.loads(timeline_data or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("timeline data is not JSON") from exc
        gib = float(data.get("vram_gb") or 16)
        cap = max_block_for_vram(gib)
        lines = ["%s GB caps a pass at %s frames. One pass is in memory at a time." % (gib, cap)]
        hold = data.get("hold") or {}
        if hold:
            lines.append(
                "Hold source at frame %s for %s frames. Those frames are not resampled."
                % (hold.get("start", 0), hold.get("length", 0))
            )
        for gen in data.get("generates") or []:
            passes = split_span(int(gen.get("length") or 0), max_block=cap, overlap=39)
            lines.append(
                "%s: %s frames at %s become %s pass(es)."
                % (gen.get("label") or gen.get("id"), gen.get("length"), gen.get("start"), len(passes))
            )
            for block in passes:
                lines.append(
                    "  pass %s starts at %s, length %s, holds %s"
                    % (block["pass"], gen.get("start", 0) + block["start"], block["length"], block["hold"])
                )
        return ("\n".join(lines),)


NODE_CLASS_MAPPINGS = {"H3Timeline": H3Timeline}
NODE_DISPLAY_NAME_MAPPINGS = {"H3Timeline": "H3 Timeline"}
