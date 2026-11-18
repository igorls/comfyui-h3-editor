"""Experimental clip-time style guidance, independent of diffusion-step scheduling."""

import json
import math


def clean_markers(value, length=None):
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise ValueError('style markers must be JSON: [{"frame": 0, "style": "Watercolor"}, ...]')
    if not isinstance(value, list) or len(value) > 32:
        raise ValueError("style markers must be a list of up to 32 markers")
    rows = []
    for marker in value:
        if not isinstance(marker, dict):
            raise ValueError("each style marker needs frame and style")
        frame, style = marker.get("frame"), marker.get("style")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 or frame >= (length or 3600):
            raise ValueError("a style marker's frame must be inside the generated clip")
        if not isinstance(style, str) or len(style) > 12000:
            raise ValueError("a style marker needs style text of up to 12000 characters")
        if any(row["frame"] == frame for row in rows):
            raise ValueError("two style markers cannot share a frame")
        name = marker.get("name") or "Style"
        if not isinstance(name, str) or len(name) > 80:
            raise ValueError("a style marker's name must be text of up to 80 characters")
        rows.append({"frame": frame, "style": style.strip(), "name": name})
    if len({row["style"] for row in rows}) > 4:
        raise ValueError("a clip can blend up to four distinct styles")
    return sorted(rows, key=lambda row: row["frame"])


def schedule(value, length, default_style=""):
    rows = clean_markers(value, length)
    if rows and rows[0]["frame"] > 0:
        if len(rows) == 32:
            raise ValueError("a clip can have up to 32 style markers, including its starting look")
        rows.insert(0, {"frame": 0, "style": default_style, "name": "Project look"})
    if len({row["style"] for row in rows}) > 4:
        raise ValueError("a clip can blend up to four distinct styles, including its starting look")
    return rows


def weights(markers, frames):
    """Smooth weights at source-video frames. A marker is the destination of the preceding blend."""
    styles = list(dict.fromkeys(row["style"] for row in markers))
    result = [[0.0] * len(frames) for _ in styles]
    for index, frame in enumerate(frames):
        left = markers[0]
        right = None
        for marker in markers[1:]:
            if marker["frame"] > frame:
                right = marker
                break
            left = marker
        if right is None or frame <= left["frame"]:
            result[styles.index(left["style"])][index] = 1.0
        else:
            amount = (frame - left["frame"]) / (right["frame"] - left["frame"])
            amount = amount * amount * (3 - 2 * amount)
            result[styles.index(left["style"])][index] += 1 - amount
            result[styles.index(right["style"])][index] += amount
    return result


def latent_positions(steps, length):
    """H3's temporal grid: each group of five latents covers 1, 4, 4, 4, 4 video frames."""
    positions = []
    for step in range(steps):
        group, rest = divmod(step, 5)
        positions.append(min(length - 1, group * 17 + (0 if rest == 0 else 2.5 + (rest - 1) * 4)))
    if positions:
        positions[-1] = length - 1
    return positions


def make_guider(model, conditioning):
    import torch
    import comfy.samplers
    from comfy_extras.nodes_custom_sampler import Guider_Basic

    timeline = next((meta.get("h3_style_timeline") for _, meta in conditioning if meta.get("h3_style_timeline")), None)
    if not timeline:
        guider = Guider_Basic(model)
        guider.set_conds(conditioning)
        return guider

    class TemporalStyleGuider(comfy.samplers.CFGGuider):
        def __init__(self):
            super().__init__(model)
            self.keys = []
            # Metadata added downstream (guides, RefMods, masks) applies to every style, while each encoding keeps its token tags.
            shared = {key: value for key, value in conditioning[0][1].items()
                      if key not in ("h3_style_timeline", "minimax_token_tags", "pooled_output")}
            for index, variant in enumerate(timeline["conditioning"]):
                key = "positive" if index == 0 else "style_%s" % index
                rows = [[text, dict(meta, **shared)] for text, meta in variant]
                self.keys.append(key)
                self.inner_set_conds({key: rows})

        def predict_noise(self, x, timestep, model_options=None, seed=None):
            shapes = self.inner_model.latent_shapes
            if not shapes or len(shapes) != 2 or len(shapes[0]) != 5:
                raise ValueError("H3 Style Guider needs H3 picture-and-sound latents")
            shape = tuple(shapes[0])
            count = math.prod(shape[1:])
            if x.shape[-1] != count + math.prod(shapes[1][1:]):
                raise ValueError("the packed H3 latent does not match its picture and sound shapes")
            blend = weights(timeline["markers"], latent_positions(shape[2], timeline["length"]))
            result = None
            for index, key in enumerate(self.keys):
                # Separate evaluations preserve H3's batch-size-one requirement and bound peak memory.
                prediction = comfy.samplers.sampling_function(self.inner_model, x, timestep, None, self.conds[key], 1.0,
                                                              model_options=model_options or {}, seed=seed)
                if result is None:
                    result = prediction.clone()
                    result[..., :count] = 0
                temporal = prediction.new_tensor(blend[index]).view(1, 1, shape[2], 1, 1)
                result[..., :count].view(shape).add_(prediction[..., :count].reshape(shape) * temporal)
            # Audio is always the first style's prediction; a visual style change never crossfades different voices.
            return result

    return TemporalStyleGuider()
