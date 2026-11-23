"""Directional jog guides, defringing, global scale locking, and sprite atlases for independent editor takes."""

import math
import os

FRAMES = 39
WIDTH, HEIGHT = 256, 384
TILE, BODY = 64, 48
EXPORT_SIZES = (64, 128, 256)
MIN_TILE, MAX_TILE = 32, 512
MAX_ATLAS_PIXELS = 32 * 1024 * 1024
PIVOT = (32, 59)
MATTE_PREFIX = "background_removal:"
# The edge of a cell: soft keeps partly covered pixels, hard makes every pixel fully in or out (pixel art).
ALPHAS = ("soft", "hard")
# Optional depth evidence for cutouts: a depth node pack and its weights, when the user has installed both.
# Each pack: the node classes it must offer, the ComfyUI models folder its weights sit in, the weights file of
# each size (preferred first) with the name its loader takes, and the focal file every size needs.
DEPTH_PACKS = {
    "tp": {
        "nodes": ("TPDepthPro2Loader", "TPDepthPro2Video", "TPDepthPro2RawImage"),
        "folder": "depth_pro_2",
        "sizes": (("L", "depthpro2-l.safetensors"), ("H+", "depthpro2-h.safetensors"),
                  ("B", "depthpro2-b.safetensors"), ("S", "depthpro2-s.safetensors")),
        "focal": "depthpro2-focal.safetensors",
    },
}
DEPTH_PREFIX = "depth:"
# The flat colours a sprite can be drawn on. White suits a dark character; a character with white or pale
# parts cuts out cleanly only from a colour it does not wear.
BACKDROPS = {"white": (255, 255, 255), "green": (0, 177, 64), "blue": (0, 71, 187)}
# How far from its backdrop's colour a reference plate's pixels still count as backdrop when the plate is
# moved to another colour: plates are often enlarged, and a gap between the legs is then a pale blend.
PLATE_REACH = .09
BACKDROP_WORDS = {"white": "pure white", "green": "a flat solid chroma key green background",
                  "blue": "a flat solid chroma key blue background"}
# How far towards the backdrop (or, mirrored, the subject) a patch must lie for depth to decide about it.
DEPTH_CLEAR = .75

CANVASES = {
    "portrait": ((256, 384), (512, 768)),
    "square": ((512, 512), (512, 512)),
    "landscape": ((384, 256), (768, 512)),
}

VIEW_TAGS = ("front", "side", "back", "left")
VIEW_ALIASES = {
    "front": "front",
    "s": "front",
    "side": "side",
    "right": "side",
    "e": "side",
    "back": "back",
    "rear": "back",
    "n": "back",
    "left": "left",
    "w": "left",
}
DIRECTION_VIEWS = {
    "S": ("front",),
    "N": ("back",),
    "E": ("side", "front"),
    "W": ("left", "side", "front"),
    "SE": ("front", "side"),
    "SW": ("front", "left", "side"),
    "NE": ("back", "side"),
    "NW": ("back", "left", "side"),
}


def matte_model(available):
    for entry in sorted(available or ()):
        if entry.startswith(MATTE_PREFIX):
            name = entry[len(MATTE_PREFIX):]
            if name.replace("\\", "/").split("/")[-1].lower() == "birefnet.safetensors":
                return name
    return None


def depth_entry(pack, file):
    """How an installed depth weights file is listed among the available models."""
    return DEPTH_PREFIX + pack + ":" + file.replace("\\", "/")


def depth_model(available):
    """The installed depth model to read sprite frames with, as (pack, size), or None.

    Only one whose node pack and every weights file are already there: nothing is ever downloaded for it.
    """
    for pack, spec in DEPTH_PACKS.items():
        have = {entry[len(DEPTH_PREFIX) + len(pack) + 1:].lower() for entry in available or ()
                if entry.startswith(DEPTH_PREFIX + pack + ":")}
        if spec["focal"] in have:
            for size, file in spec["sizes"]:
                if file in have:
                    return pack, size
    return None


def depth_nodes(model, images, start):
    """Graph nodes that read a depth map for each frame: their ids start at `start`. Returns them and the
    link to the maps, an IMAGE batch."""
    pack, size = model
    a, b, c = str(start), str(start + 1), str(start + 2)
    if pack == "tp":
        return {
            a: {"class_type": "TPDepthPro2Loader",
                "inputs": {"model": size, "precision": "auto", "estimate_focal": True, "device": "auto"}},
            b: {"class_type": "TPDepthPro2Video",
                "inputs": {"model": [a, 0], "images": list(images), "mode": "buffered", "window": 16, "overlap": 4,
                           "focal_length_px": 0.0}},
            c: {"class_type": "TPDepthPro2RawImage", "inputs": {"depth": [b, 0], "encoding": "inverse_meters"}},
        }, [c, 0]
    raise ValueError("unknown depth node pack")


DIRECTIONS = {
    "S": "south, down the screen",
    "E": "east, screen right",
    "N": "north, up the screen, back visible",
    "W": "west, screen left",
    "NE": "northeast, diagonally up-right and away, rear three-quarter view",
    "NW": "northwest, diagonally up-left and away, rear three-quarter view",
    "SE": "southeast, diagonally down-right and toward the viewer, front three-quarter view",
    "SW": "southwest, diagonally down-left and toward the viewer, front three-quarter view",
}
CAMERAS = {
    "rpg": "45 degrees above the ground, visible top planes and foreshortened legs",
    "level": "at eye level",
    "overhead": "straight down from 90 degrees above, crown and shoulders visible",
}


def validate_frames(value):
    """Any legal H3 video length on the 5 + 17k grid with at least 22 frames (22, 39, 56, 73, ...)."""
    if type(value) is not int or value < 22 or (value - 5) % 17 != 0:
        raise ValueError("sprite frames must be a legal H3 length >= 22 on the 5 + 17k grid (22, 39, 56, 73, ...)")
    return value


def validate_tile_size(tile):
    """Validate an even sprite tile width x height in 32..512 (accepts an int or (w, h) pair)."""
    if type(tile) is int:
        w = h = tile
    elif isinstance(tile, (tuple, list)) and len(tile) == 2 and type(tile[0]) is int and type(tile[1]) is int:
        w, h = tile
    elif isinstance(tile, dict) and type(tile.get("w")) is int and type(tile.get("h")) is int:
        w, h = tile["w"], tile["h"]
    else:
        raise ValueError("tile size must be an even pixel size or (w, h) pair in %s..%s" % (MIN_TILE, MAX_TILE))
    if not (MIN_TILE <= w <= MAX_TILE and MIN_TILE <= h <= MAX_TILE) or w % 2 != 0 or h % 2 != 0:
        raise ValueError("tile dimensions must be even integers in %s..%s (got %sx%s)" % (MIN_TILE, MAX_TILE, w, h))
    return (w, h)


def canvas_size(config=None, quality="draft"):
    """Source (width, height) for a sprite canvas preset and render quality."""
    name = (config or {}).get("canvas", "portrait")
    if name not in CANVASES:
        raise ValueError("choose a sprite canvas of %s" % (", ".join(CANVASES),))
    draft, final = CANVASES[name]
    return draft if quality == "draft" else final


def normalize_view(view):
    """Normalize a cast picture view tag ('front', 'side', 'back', 'left', or compass S/E/N/W)."""
    if view is None:
        return None
    if not isinstance(view, str) or not view.strip():
        raise ValueError("view must be one of %s" % (", ".join(VIEW_TAGS),))
    key = view.strip().lower()
    if key not in VIEW_ALIASES:
        raise ValueError("view must be one of %s" % (", ".join(VIEW_TAGS),))
    return VIEW_ALIASES[key]


def filter_cast_pictures(pictures, direction):
    """Filter view-tagged cast pictures for a sprite compass direction so side refs don't pull rear views.

    When no picture carries a view tag, all pictures are kept unchanged for full backward compatibility.
    """
    pictures = list(pictures or ())
    if not pictures or direction not in DIRECTION_VIEWS:
        return pictures
    tagged = [p for p in pictures if isinstance(p, dict) and p.get("view")]
    if not tagged:
        return pictures
    desired = DIRECTION_VIEWS[direction]
    by_view = {}
    for pic in tagged:
        v = normalize_view(pic["view"])
        by_view.setdefault(v, []).append(pic)
    selected = []
    if direction in ("S", "N"):
        selected.extend(by_view.get(desired[0], []))
    elif direction == "E":
        selected.extend(by_view.get("side", []))
        if not selected:
            selected.extend(by_view.get("front", []))
    elif direction == "W":
        selected.extend(by_view.get("left", []))
        if not selected:
            selected.extend(by_view.get("side", []))
        if not selected:
            selected.extend(by_view.get("front", []))
    elif direction in ("SE", "SW"):
        selected.extend(by_view.get("front", []))
        side_key = "side" if direction == "SE" else ("left" if "left" in by_view else "side")
        selected.extend(by_view.get(side_key, []))
    elif direction in ("NE", "NW"):
        selected.extend(by_view.get("back", []))
        side_key = "side" if direction == "NE" else ("left" if "left" in by_view else "side")
        selected.extend(by_view.get(side_key, []))
    if not selected:
        return pictures
    # Deduplicate while preserving priority order.
    seen = set()
    out = []
    for pic in selected:
        ident = (pic.get("subfolder", ""), pic.get("file", ""))
        if ident not in seen:
            seen.add(ident)
            out.append(pic)
    return out


HEADING_VIEW = {"S": "front", "N": "back", "E": "side", "W": "left"}


def heading_matched(pictures, direction):
    """Is the first of these pictures one of the character facing this heading, as far as its view tag says.

    Untagged pictures count as matched, as they did before view tags. A diagonal has no view tag of its own,
    so a tagged picture never matches one.
    """
    pictures = [p for p in pictures or () if isinstance(p, dict)]
    if not pictures:
        return False
    if not any(p.get("view") for p in pictures):
        return True
    view = pictures[0].get("view")
    return bool(view) and normalize_view(view) == HEADING_VIEW.get(direction)


def still(image, backdrop="none", width=0, height=0):
    """A reference picture as RGB, standing on a backdrop colour, with the whole picture fitted inside a canvas.

    Dropping the transparency of a cutout leaves it on black, and a take drawn on white then fades from black
    to white: a transparent picture is laid on the backdrop. A picture already drawn on another flat colour
    (a white plate for a take on green) is cut out of it and moved, the regions it encloses included, since
    the model copies a sliver of the old backdrop between the legs onto every frame. A pinned frame of another
    shape than the canvas is cropped to fill it, which cuts the head and feet off a portrait picture on a
    square canvas; fitted, the rest is filled with the backdrop or the picture's own border colour.
    """
    import numpy as np
    from PIL import Image

    if backdrop != "none" and backdrop not in BACKDROPS:
        raise ValueError("backdrop must be none, %s" % ", ".join(BACKDROPS))
    colour = BACKDROPS.get(backdrop)
    if colour:
        if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
        else:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
            ground, _, flat = globals()["backdrop"](rgb)
            moved = flat and float(np.abs(ground - np.asarray(colour, dtype=np.float32) / 255).max()) > .08
            rgba = with_matte(image, pockets=True, reach=PLATE_REACH) if moved else None
        if rgba is not None:
            image = Image.new("RGBA", rgba.size, colour + (255,))
            image.alpha_composite(rgba)
    image = image.convert("RGB")
    if not (width and height) or image.size == (width, height):
        return image
    if colour:
        fill = colour
    else:
        rgb = np.asarray(image)
        fill = tuple(int(v) for v in np.median(np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1])), axis=0))
    scale = min(width / image.width, height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    canvas = Image.new("RGB", (width, height), fill)
    canvas.paste(image.resize(size, Image.Resampling.LANCZOS), ((width - size[0]) // 2, (height - size[1]) // 2))
    return canvas


def _clean_file_ref(ref, what):
    if (
        not isinstance(ref, dict)
        or not isinstance(ref.get("file"), str)
        or not ref["file"].strip()
        or not isinstance(ref.get("subfolder", ""), str)
    ):
        raise ValueError("%s needs a file and subfolder" % what)
    name, folder = ref["file"], ref.get("subfolder", "").replace("\\", "/")
    if (
        any(c in name for c in ("/", "\\", ":", "\x00"))
        or folder.startswith("/")
        or ":" in folder
        or "\x00" in folder
        or ".." in folder.split("/")
    ):
        raise ValueError("%s must sit inside the output folder" % what)
    return name, folder


def settings(value):
    if value is None:
        return None
    allowed = {"direction", "camera", "poses", "guide", "motion", "cutout", "alpha", "depth", "animation", "loop", "backdrop",
               "canvas", "frames", "tile_size", "mode", "refine", "self_pin", "style"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("unknown sprite setting")
    direction, camera, poses = value.get("direction", "S"), value.get("camera", "rpg"), value.get("poses", 7)
    if direction not in tuple(DIRECTIONS) or camera not in tuple(CAMERAS):
        raise ValueError("choose a compass direction and a level, rpg or overhead camera")
    if type(poses) is not int or not 4 <= poses <= 12:
        raise ValueError("a sprite cycle needs 4 to 12 poses at 12 fps")
    out = {"direction": direction, "camera": camera, "poses": poses}
    if "mode" in value:
        if value["mode"] != "prop":
            raise ValueError("sprite mode must be prop")
        if "motion" in value or value.get("guide") is not None:
            raise ValueError("static prop mode does not take a motion source or guide")
        out["mode"] = "prop"
    if "motion" in value:
        if value["motion"] not in ("prompt", "idle") or value.get("guide") is not None:
            raise ValueError("choose one sprite motion source")
        out["motion"] = value["motion"]
    if "animation" in value:
        name = value["animation"]
        if (out.get("motion") != "prompt" or not isinstance(name, str) or not name.strip()
                or len(name) > 64 or name != name.strip() or name.lower() in ("run", "idle", "custom", "custom motion")):
            raise ValueError("Choose a distinct name for the prompted sprite animation.")
        out["animation"] = name
    if "loop" in value:
        if value["loop"] is not True or "animation" not in out:
            raise ValueError("loop marks a named sprite animation as a repeating cycle")
        out["loop"] = True
    if "backdrop" in value:
        if value["backdrop"] not in BACKDROPS:
            raise ValueError("choose a sprite backdrop: %s" % ", ".join(BACKDROPS))
        if value["backdrop"] != "white":
            out["backdrop"] = value["backdrop"]
    if "cutout" in value:
        if value["cutout"] not in ("auto", "edge"):
            raise ValueError("choose auto or edge sprite cutout")
        out["cutout"] = value["cutout"]
    if "alpha" in value:
        if value["alpha"] not in ALPHAS:
            raise ValueError("choose soft or hard sprite alpha")
        out["alpha"] = value["alpha"]
    if "depth" in value:
        if value["depth"] not in ("off", "on"):
            raise ValueError("choose off or on for sprite depth")
        out["depth"] = value["depth"]
    if "canvas" in value:
        if value["canvas"] not in CANVASES:
            raise ValueError("choose a sprite canvas of %s" % (", ".join(CANVASES),))
        out["canvas"] = value["canvas"]
    if "frames" in value:
        out["frames"] = validate_frames(value["frames"])
    if "tile_size" in value:
        out["tile_size"] = list(validate_tile_size(value["tile_size"]))
    if "refine" in value:
        if type(value["refine"]) is not bool:
            raise ValueError("refine must be true or false")
        out["refine"] = value["refine"]
    if value.get("style") is not None:
        if not isinstance(value["style"], str) or len(value["style"].strip()) > 400:
            raise ValueError("sprite style must be text up to 400 characters")
        cleaned_style = value["style"].strip()
        if cleaned_style:
            out["style"] = cleaned_style
    if value.get("self_pin") is not None:
        sp = value["self_pin"]
        name, folder = _clean_file_ref(sp, "self_pin")
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            raise ValueError("self_pin must be an image file")
        stride = sp.get("stride", 2 * poses)
        if type(stride) is not int or not 4 <= stride <= 36:
            raise ValueError("self_pin stride must be an integer in 4..36")
        out["self_pin"] = {"file": name, "subfolder": folder, "stride": stride}
    if value.get("guide") is not None:
        g = value["guide"]
        if (
            not isinstance(g, dict)
            or not isinstance(g.get("file"), str)
            or not isinstance(g.get("subfolder", ""), str)
        ):
            raise ValueError("the sprite motion guide needs a video file and subfolder")
        name, folder = g["file"], g.get("subfolder", "").replace("\\", "/")
        if (
            not name.lower().endswith((".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"))
            or any(c in name for c in ("/", "\\", ":", "\x00"))
            or folder.startswith("/")
            or ":" in folder
            or "\x00" in folder
            or ".." in folder.split("/")
        ):
            raise ValueError("choose a motion guide video in the output folder")
        out["guide"] = {"file": name, "subfolder": folder}
    return out


def one_shot(config):
    """Is this a named gesture played once from start to finish (a wave, an attack), not a repeating cycle.

    A gesture is sampled across its whole length and pinned to the rest pose at both ends. A cycle (a walk)
    sampled that way starts and stops in every loop at an uneven speed: marked `loop`, it is cut like the
    built-in run, one period at a steady 12 poses a second.
    """
    return bool(config.get("animation")) and not config.get("loop")


def instruction(config):
    motion = (
        "running in place"
        if "motion" not in config and "guide" not in config
        else "performing the described movement in place"
    )
    poses = "Match the poses and foot contacts in <Video 1>, keeping the exact art style, colors, proportions and equipment. "
    if config.get("motion") == "prompt":
        if one_shot(config):
            poses = (
                "Perform the full gesture cleanly from start to finish and return to the initial pose, "
                "keeping the exact art style, colors, proportions and equipment. "
            )
        else:
            poses = "Loop the movement, keeping the exact art style, colors, proportions and equipment. "
    elif config.get("motion") == "idle":
        motion = "resting in its natural standing pose"
        poses = ("Make one subtle repeating idle gesture and return to the initial pose, "
                 "keeping the exact art style, colors, proportions and equipment. "
                 "Keep every ground contact planted and the body in the same spot. ")
    return (
        "Single full-body game character "
        + motion
        + " facing "
        + DIRECTIONS[config["direction"]]
        + ". Locked orthographic camera "
        + CAMERAS[config["camera"]]
        + ". "
        + poses
        + "Center the whole character on " + BACKDROP_WORDS[config.get("backdrop", "white")] + " with no floor or shadow."
    )


def jog(direction, camera, poses, frames=FRAMES, canvas="portrait"):
    """Project the same articulated stride onto a pitched orthographic camera in each facing."""
    import numpy as np
    from PIL import Image, ImageDraw

    validate_frames(frames)
    w, h = canvas_size({"canvas": canvas}, "draft")
    scale = min(w / float(WIDTH), h / float(HEIGHT))
    cx, cy = w / 2.0, h / 2.0
    pitch = math.radians({"rpg": 45, "level": 0, "overhead": 90}[camera])
    dx = int("E" in direction) - int("W" in direction)
    dz = int("S" in direction) - int("N" in direction)
    length = math.hypot(dx, dz)

    def project(j):
        f, height_val, r = j
        x, z = (f * dx - r * dz) / length, (f * dz + r * dx) / length
        return (
            round(cx + x * scale),
            round(cy + (-(height_val - 120) * math.cos(pitch) + z * math.sin(pitch)) * scale),
        )

    line_w = max(2, round(2 * scale))
    joint_r = max(1, round(1 * scale))
    cycle = []
    for frame in range(2 * poses):
        phase = math.pi * frame / poses
        bob = 4 * math.cos(2 * phase)
        neck = (6, 196 + bob, 0)
        limbs = []
        for side in (-1, 1):
            swing = side * math.cos(phase)
            lift = max(0, -side * math.sin(phase))
            hip, shoulder = (0, 118 + bob, 12 * side), (7 - swing * 5, 192 + bob, 24 * side)
            knee = (22 + swing * 22, 65 + lift * 26, 13 * side)
            foot = (swing * 50 - 4, 12 + lift * 53, 13 * side)
            elbow = (-swing * 29, 150 + bob, 28 * side)
            hand = (elbow[0] + 26, 148 + bob - swing * 12, 27 * side)
            colors = (
                ["#0055ff", "#00aaff", "#00ffff", "#00ff55", "#00ff00"]
                if side == 1
                else ["#ff5500", "#ffaa00", "#ffff00", "#aaff00", "#55ff00"]
            )
            limbs.extend(
                zip(
                    (neck, shoulder, elbow, neck, hip, knee),
                    (shoulder, elbow, hand, hip, knee, foot),
                    (colors[0], colors[1], colors[2], colors[3], colors[3], colors[4]),
                )
            )
        nose = (22, 226 + bob, 0)
        if dz >= 0 and camera != "overhead":
            limbs.append((neck, nose, "#0055ff"))
        for side in (-1, 1):
            if dz <= 0 and dx * side == -1:
                continue
            ear = (3, 231 + bob, 13 * side)
            limbs.append((neck, ear, "#aa00ff"))
        im = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(im)
        for a, b, color in limbs:
            xy = project(b)
            draw.line((project(a), xy), fill=color, width=line_w)
            draw.ellipse((xy[0] - joint_r, xy[1] - joint_r, xy[0] + joint_r, xy[1] + joint_r), fill=color)
        cycle.append(np.asarray(im))
    return np.stack([cycle[t % len(cycle)] for t in range(frames)])


def cutout(im):
    import numpy as np
    from scipy.ndimage import binary_propagation
    from PIL import Image

    rgb = np.asarray(im.convert("RGB"), dtype=np.int16)
    edges = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
    background = np.max(np.abs(rgb - np.median(edges, axis=0)), axis=2) < 36
    seed = np.zeros_like(background)
    seed[[0, -1]] = background[[0, -1]]
    seed[:, [0, -1]] = background[:, [0, -1]]
    im = im.convert("RGBA")
    im.putalpha(Image.fromarray((~binary_propagation(seed, mask=background)).astype("uint8") * 255))
    return im


def backdrop(rgb):
    """The colour a picture stands on, read from its border, and whether that border is one flat colour.

    `rgb` is a float array in 0..1. Returns (colour, noise, flat): a subject may cross the border, so the
    border counts as flat when most of it sits on the colour. A scene behind the subject is not flat.
    """
    import numpy as np

    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
    ground = np.median(border, axis=0)
    off = np.abs(border - ground).max(axis=1)
    near = off < .06
    noise = float(off[near].mean()) if near.any() else 1.0
    return ground, noise, bool(near.mean() >= .6)


def depth_side(depth, backdrop, subject):
    """Where a depth map puts each pixel between the subject (0) and the backdrop (1), or None when it
    cannot tell the two apart.

    `depth` is any monotonic depth picture (metres, inverse metres, a normalised map, near bright or near
    dark): it is read against the picture's own subject and backdrop, sampled where those are certain.
    """
    import numpy as np

    if not backdrop.any() or not subject.any():
        return None
    far, near = float(np.median(depth[backdrop])), float(np.median(depth[subject]))
    if not np.isfinite(far) or not np.isfinite(near) or abs(far - near) < .05 * max(abs(far), abs(near), 1e-6):
        return None
    with np.errstate(invalid="ignore"):
        side = ((near - depth) / (near - far)).clip(0, 1)
    return np.where(np.isfinite(depth), side, 1. if far > near else 0.).astype(np.float32)


def backdrop_key(rgb, ground, noise, subject=None, pockets=False, depth=None, reach=None):
    """How far each pixel is from being backdrop, 0..1, on a flat backdrop of any colour.

    Backdrop is the flat region of the backdrop's own colour that reaches the border. Colour alone cannot
    tell a white shirt from a white backdrop, so the rest takes evidence:

    - `subject`, a foreground matte (0..1). Where it calls a keyed patch foreground, a small patch is the
      matte's mistake (a lump of backdrop beside hair, a sliver between limbs) and goes; a large one is the
      subject's own backdrop-coloured surface (a white shirt) and stays.
    - `depth`, a depth map of the picture (see `depth_side`). It settles both by distance instead of size:
      a keyed patch at the subject's distance stays, whatever its size, and a region the subject encloses
      (between legs, under an arm) goes when it lies at the backdrop's distance. A depth map is soft at
      edges, so it decides only where it is clear (DEPTH_CLEAR); a patch in between is left to the rules
      that apply without it.
    - `pockets`, for pictures known to hold no surface of the backdrop's colour and with no depth map: also
      take the enclosed regions that are as flat as the backdrop and not a speck.

    The edge of what goes fades over the antialiased pixels around it. `reach` widens how far from the
    backdrop's colour a pixel still counts as backdrop, for a soft picture (an enlarged plate) whose gaps
    and edges are a blend of backdrop and subject rather than the backdrop's own colour.
    """
    import numpy as np
    from scipy import ndimage as ndi

    dist = np.abs(rgb - ground).max(axis=2)
    inner = max(.03, 4 * noise, reach or 0)
    flatness = max(.012, 2 * noise, .6 * (reach or 0))
    fade = 4 if reach else 2
    core = dist < inner
    seed = np.zeros_like(core)
    seed[[0, -1]] = core[[0, -1]]
    seed[:, [0, -1]] = core[:, [0, -1]]
    gone = ndi.binary_propagation(seed, mask=core)
    far = None
    if depth is not None:
        sure = ~ndi.binary_dilation(core, iterations=2)
        far = depth_side(depth, gone & (subject < .1) if subject is not None else gone,
                         sure & (subject > .9) if subject is not None else sure)
    if far is not None or pockets:
        labels, count = ndi.label(core & ~gone)
        if count:
            index = np.arange(1, count + 1)
            area = ndi.sum(np.ones_like(dist), labels, index)
            take = np.zeros(count + 1, dtype=bool)
            if pockets:
                take[1:] = (area >= max(8, core.size // 12000)) & (ndi.mean(dist, labels, index) <= flatness)
            if far is not None:
                side = ndi.median(far, labels, index)
                take[1:] = (take[1:] & (side > 1 - DEPTH_CLEAR)) | ((area >= 8) & (side >= DEPTH_CLEAR))
            gone |= take[labels]
    kept = np.zeros_like(core)
    if subject is not None or far is not None:
        solid = subject > .5 if subject is not None else far <= 1 - DEPTH_CLEAR
        labels, count = ndi.label(gone & solid)
        if count:
            index = np.arange(1, count + 1)
            keep = np.zeros(count + 1, dtype=bool)
            if subject is not None:
                keep[1:] = ndi.sum(np.ones_like(dist), labels, index) > max(64, .02 * solid.sum())
            if far is not None:
                side = ndi.median(far, labels, index)
                keep[1:] = (keep[1:] & (side < DEPTH_CLEAR)) | (side <= 1 - DEPTH_CLEAR)
            kept = keep[labels]
            gone &= ~kept
    # Video compression rings around a subject: the backdrop beside it is a little off its colour.
    gone = ndi.binary_dilation(gone, iterations=3, mask=(dist < inner + .07) & ~kept)
    key = np.ones_like(dist)
    rim = ndi.binary_dilation(gone, iterations=fade) & ~gone & ~kept
    key[rim] = ((dist[rim] - inner) / .1).clip(0, 1)
    key[gone] = 0
    return key


def with_matte(frame, mask=None, alpha_low=0.12, alpha_high=0.88, key=True, pockets=False, depth=None, reach=None):
    """Cut a subject out of its picture, as RGBA with the backdrop colour taken out of its edge pixels.

    `mask` is a foreground matte (0..1), or None to cut by the backdrop colour alone. On a flat backdrop of
    any colour the two are combined (see `backdrop_key`): the matte decides the subject, and the backdrop's
    own colour removes the patches of backdrop a matte keeps by mistake. `depth`, an optional depth map of
    the frame, tells those patches from the subject's own backdrop-coloured surfaces. On a scene (no flat
    border), or with `key` off, only the matte is used.
    """
    import numpy as np
    from PIL import Image

    rgb = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255
    ground, noise, flat = backdrop(rgb)
    if mask is None:
        alpha = coverage = np.ones(rgb.shape[:2] + (1,), dtype=np.float32)
    else:
        alpha = np.asarray(mask, dtype=np.float32)[..., None].clip(0, 1)
        if alpha.shape[:2] != rgb.shape[:2]:
            raise ValueError("sprite matte must match the source frame")
        span = max(float(alpha_high) - float(alpha_low), 1e-4)
        coverage = ((alpha - float(alpha_low)) / span).clip(0, 1)
    if mask is None or (key and flat):
        if depth is not None:
            depth = np.asarray(depth, dtype=np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
            if depth.shape != rgb.shape[:2]:
                depth = np.asarray(Image.fromarray(depth, "F").resize(rgb.shape[1::-1], Image.Resampling.BILINEAR))
        keyed = backdrop_key(rgb, ground, noise, None if mask is None else alpha[..., 0], pockets, depth, reach)[..., None]
        alpha, coverage = np.minimum(alpha, keyed), np.minimum(coverage, keyed)
    color = ((rgb - (1 - alpha) * ground) / np.maximum(alpha, .05)).clip(0, 1)
    return Image.fromarray((np.concatenate((color, coverage), axis=2) * 255).round().astype("uint8"))


def harden(im, threshold=128):
    """Every pixel fully in or fully out: the edge for pixel art, which has no half-transparent pixels."""
    im = im.convert("RGBA")
    im.putalpha(im.getchannel("A").point(lambda v: 255 if v >= threshold else 0))
    return im


def seam_score(frames, start, period):
    """Loop-closure jump from frame[start+period] back to frame[start], normalized by median frame step."""
    import numpy as np
    from PIL import Image

    if start < 0 or period < 1 or start + period >= len(frames):
        raise ValueError("seam window (%s, %s) does not fit in %s frames" % (start, period, len(frames)))
    sigs = []
    for frame in frames:
        rgba = frame.convert("RGBA").resize((64, 64), Image.Resampling.BOX)
        arr = np.asarray(rgba, dtype=np.float32)
        sigs.append(arr[:, :, :3].mean(2) * arr[:, :, 3] / 255.0)
    steps = [float(np.abs(sigs[i] - sigs[i + 1]).mean()) for i in range(start, start + period)]
    med = float(np.median(steps)) if steps else 0.0
    jump = float(np.abs(sigs[start] - sigs[start + period]).mean())
    return round(jump / max(med, 1e-6), 4)


def pick_self_pin_frame(frames, stride, preferred=None):
    """Pick a clean non-edge-touching stride-phase frame for deterministic 0 / stride / 2*stride self-pinning."""
    from PIL import Image

    total = len(frames)
    if total < stride + 1:
        raise ValueError("clip of %s frames is too short for stride %s self-pinning" % (total, stride))
    lo = min(6, total - 1)
    hi = min(total - 1, stride + 4)
    target = stride if preferred is None else max(0, min(total - 1, int(preferred)))
    candidates = sorted(range(lo, hi + 1), key=lambda idx: (abs(idx - target), idx))
    chosen = target
    for idx in candidates:
        im = frames[idx]
        rgba = im if im.mode == "RGBA" else cutout(im)
        bbox = rgba.getchannel("A").point(lambda v: 255 if v >= 128 else 0).getbbox()
        if bbox and bbox[0] > 1 and bbox[1] > 1 and bbox[2] < rgba.width - 1 and bbox[3] < rgba.height - 1:
            chosen = idx
            break
    frame = frames[chosen]
    # A cut-out pose stays a cutout: the take that pins it lays it on its own backdrop colour. Flattened on
    # white here, a pin moved to a green take kept a white rim, and the model drew that rim on every frame.
    picture = frame if frame.mode == "RGBA" else frame.convert("RGB")
    pin_indices = [i for i in (0, stride, 2 * stride) if i < total]
    return chosen, picture, pin_indices


def compute_global_layout(animation_boxes, tile_size=(TILE, TILE), body_ratio=BODY / TILE,
                          pivot_y_ratio=PIVOT[1] / TILE, margin=2, reference=None):
    """Compute one width- and height-clamped global scale & ground pivot shared across all animations of a character.

    Prevents full-action wide/overhead takes from shrinking the character relative to idle/run,
    while guaranteeing zero tile-edge clipping on every frame of every row.
    """
    import numpy as np

    tile_w, tile_h = validate_tile_size(tile_size)
    pivot_x = tile_w // 2
    pivot_y = int(round(tile_h * float(pivot_y_ratio)))
    per_anim = {}
    medians = []
    max_half_w = 0.0
    max_above = 0.0
    max_below = 0.0
    for name, boxes in animation_boxes.items():
        seen = [b for b in boxes if b is not None]
        if not seen:
            raise ValueError("animation %r has no visible silhouette frames" % (name,))
        h_med = float(np.median([b[3] - b[1] for b in seen]))
        cx = float(np.median([(b[0] + b[2]) / 2.0 for b in seen]))
        gy = float(np.percentile([b[3] for b in seen], 90))
        per_anim[name] = {"height": h_med, "center": cx, "ground": gy}
        medians.append(h_med)
        for b in seen:
            max_half_w = max(max_half_w, cx - b[0], b[2] - cx)
            max_above = max(max_above, gy - b[1])
            max_below = max(max_below, b[3] - gy)
    if not per_anim:
        raise ValueError("compute_global_layout needs at least one animation")
    ref_height = per_anim[reference]["height"] if (reference and reference in per_anim) else float(np.median(medians))
    scale_h = (tile_h * float(body_ratio)) / max(ref_height, 1e-6)
    avail_half_w = max(1.0, min(pivot_x, tile_w - pivot_x) - float(margin))
    avail_above = max(1.0, pivot_y - float(margin))
    avail_below = max(1.0, (tile_h - pivot_y) - float(margin))
    scale_w = avail_half_w / max(max_half_w, 1e-6)
    scale_top = avail_above / max(max_above, 1e-6)
    scale_bot = avail_below / max(max_below, 1e-6) if max_below > 0 else scale_h
    global_scale = min(scale_h, scale_w, scale_top, scale_bot)
    # Express the shared scale in equivalent reference-height units so strips() can reuse it seamlessly.
    effective_height = (tile_h * float(body_ratio)) / max(global_scale, 1e-6)
    for row in per_anim.values():
        row["scale"] = global_scale
        row["effective_height"] = effective_height
    return {
        "scale": global_scale,
        "reference_height": ref_height,
        "effective_height": effective_height,
        "tile": [tile_w, tile_h],
        "pivot": [pivot_x, pivot_y],
        "per_animation": per_anim,
    }


def action_indices(signatures, poses):
    """Sample one prompted action across its moving span, including the return and any pause inside it."""
    import numpy as np

    change = np.abs(np.diff(signatures, axis=0)).mean(axis=(1, 2))
    travel = np.r_[0.0, np.cumsum(change)]
    first, last = 0, len(signatures) - 1
    if travel[-1] > 0:
        # Discard small amounts of drift in the holds before and after the gesture.
        first = max(0, int(np.searchsorted(travel, .03 * travel[-1])) - 2)
        last = min(last, int(np.searchsorted(travel, .97 * travel[-1])) + 2)
    first = min(first, len(signatures) - poses)
    last = max(last, first + poses - 1)
    return np.linspace(first, last, poses).round().astype(int).tolist()


def playback_fps(layout, poses, idle, export_24fps=False):
    if export_24fps:
        return 24
    if "indices" in layout:
        indices = layout["indices"]
        return round(24 * (poses - 1) / (indices[-1] - indices[0]), 2)
    return 6 if idle else 12


def playback_indices(layout, poses, idle, export_24fps, total_frames=FRAMES):
    """Map output cells to saved movie frames, keeping the authored loop duration."""
    start = layout["start"]
    indices = layout.get("indices", [start + 2 * i for i in range(poses)])
    if (not isinstance(indices, list) or len(indices) != poses
            or any(type(i) is not int or not 0 <= i < total_frames for i in indices)
            or indices != sorted(set(indices)) or indices[0] != start):
        raise ValueError("Invalid sprite pose indices")
    cycle = indices + indices[-2:0:-1] if idle else indices
    if not export_24fps:
        return cycle
    if not idle and "indices" not in layout:
        return list(range(start, start + 2 * poses))
    count = round(len(cycle) * 24 / playback_fps(layout, poses, idle))
    if not idle:
        # One-shot gestures hold their final pose for the last authored cell.
        return [min(start + i, indices[-1]) for i in range(count)]
    # Follow both legs at the authored idle speed. Use actual in-between frames
    # and hold each source pose where the slower motion needs extra ticks.
    sequence = []
    for i in range(count):
        cell, remainder = divmod(i * len(cycle), count)
        left, right = cycle[cell], cycle[(cell + 1) % len(cycle)]
        sequence.append(left + (1 if right >= left else -1) * (abs(right - left) * remainder // count))
    return sequence


def _indexed_from_atlas(atlas):
    import numpy as np
    from PIL import Image

    rgba = np.asarray(atlas)
    visible = rgba[:, :, 3] >= 128
    pixels = rgba[:, :, :3][visible]
    if not len(pixels):
        raise ValueError("No sprite silhouette is visible in the selected cells")
    swatch = Image.fromarray(pixels.reshape(1, -1, 3))
    palette = swatch.quantize(colors=15, method=Image.Quantize.MEDIANCUT)
    cols = palette.getpalette()[:45]
    cols = (cols + [0] * 45)[:45]
    pal = Image.new("P", (1, 1))
    pal.putpalette(cols + cols[:3] * 241)
    idx = np.asarray(atlas.convert("RGB").quantize(palette=pal, dither=Image.Dither.NONE)).astype(np.uint16)
    indexed = Image.frombytes(
        "P", atlas.size, np.where(visible, np.where(idx < 15, idx, 0) + 1, 0).astype(np.uint8).tobytes()
    )
    indexed.putpalette([0, 0, 0] + cols + [0] * 720)
    indexed.info["transparency"] = 0
    return indexed


def _render_cells(selected_by_index, order, layout, tile_w, tile_h, ref_w, ref_h, global_scale=None, alpha="soft"):
    from PIL import Image

    if global_scale is not None:
        # global_scale is in output-tile pixels per reference-canvas pixel.
        scale_x = float(global_scale)
        scale_y = float(global_scale)
        px = tile_w / 2.0
        py = tile_h * (PIVOT[1] / float(TILE))
    else:
        scale_unit = BODY / layout["height"]
        scale_x = scale_unit * (tile_w / float(TILE))
        scale_y = scale_unit * (tile_h / float(TILE))
        px = PIVOT[0] * (tile_w / float(TILE))
        py = PIVOT[1] * (tile_h / float(TILE))
    atlas = Image.new("RGBA", (tile_w * len(order), tile_h))
    tiles = {}
    size = (tile_w, tile_h)
    for source_index, frame in selected_by_index.items():
        sx, sy = frame.width / float(ref_w), frame.height / float(ref_h)
        x = (layout["center"] - px / scale_x) * sx
        y = (layout["ground"] - py / scale_y) * sy
        box = (x, y, x + tile_w * sx / scale_x, y + tile_h * sy / scale_y)
        fine = frame.convert("RGBa").transform((4 * tile_w, 4 * tile_h), Image.Transform.EXTENT, box, Image.Resampling.BILINEAR)
        tile = fine.resize(size, Image.Resampling.LANCZOS).convert("RGBA")
        coverage = fine.getchannel("a").resize(size, Image.Resampling.BOX)
        tile.paste((0, 0, 0, 0), mask=coverage.point(lambda v: 0 if v else 255))
        if alpha == "hard":
            tile = harden(tile)
        tiles[source_index] = tile
    for column, index in enumerate(order):
        atlas.paste(tiles[index], (tile_w * column, 0))
    return atlas


def save_frame_cache(folder, frames_by_index):
    """Persist lossless defringed RGBA frames beside a take so re-exports never re-decode H.264 or re-run BiRefNet."""
    os.makedirs(folder, exist_ok=True)
    saved = {}
    for idx, frame in frames_by_index.items():
        name = "f%04d.png" % int(idx)
        path = os.path.join(folder, name)
        frame.convert("RGBA").save(path, format="PNG")
        saved[int(idx)] = name
    return saved


def load_frame_cache(folder, indices=None):
    """Load cached lossless RGBA frames.

    When indices is provided, returns {idx: Image} (or None if any requested frame is missing).
    When indices is None, returns all cached fNNNN.png frames as a sorted list (or None if empty).
    """
    from PIL import Image

    if not folder or not os.path.isdir(folder):
        return None
    if indices is None:
        files = sorted(f for f in os.listdir(folder) if f.startswith("f") and f.endswith(".png"))
        if not files:
            return None
        out = []
        for fname in files:
            with Image.open(os.path.join(folder, fname)) as im:
                out.append(im.convert("RGBA"))
        return out
    loaded = {}
    for idx in indices:
        path = os.path.join(folder, "f%04d.png" % int(idx))
        if not os.path.isfile(path):
            return None
        with Image.open(path) as im:
            loaded[int(idx)] = im.convert("RGBA")
    return loaded


def export_from_cache(folder, layout, poses, tile_size, idle=False, export_24fps=False,
                      total_frames=FRAMES, canvas="portrait", global_scale=None, alpha="soft"):
    """Re-slice and resize a sprite strip directly from lossless cached RGBA frames without touching the GPU."""
    tile_w, tile_h = validate_tile_size(tile_size)
    order = playback_indices(layout, poses, idle, export_24fps, total_frames=total_frames)
    indices = list(dict.fromkeys(order))
    cached = load_frame_cache(folder, indices)
    if cached is None:
        return None
    ref_w, ref_h = canvas_size({"canvas": canvas}, "draft")
    atlas = _render_cells(cached, order, layout, tile_w, tile_h, ref_w, ref_h, global_scale=global_scale, alpha=alpha)
    indexed = _indexed_from_atlas(atlas)
    return atlas, indexed


def strips(frames, poses, previous=None, masks_for=None, idle=False, cell_size=TILE, full_action=False,
           export_24fps=False, tile_size=None, canvas="portrait", global_scale=None, cache_dir=None,
           alpha="soft", depth_for=None):
    """Pick a loop once. Finish reuses the draft's phase, scale and ground contact in normalized source coordinates."""
    import numpy as np
    from PIL import Image

    total_frames = validate_frames(len(frames))
    if tile_size is not None:
        tile_w, tile_h = validate_tile_size(tile_size)
    else:
        if type(cell_size) is not int or cell_size not in EXPORT_SIZES:
            raise ValueError("choose a sprite cell size of 64, 128 or 256 px")
        tile_w = tile_h = cell_size
    ref_w, ref_h = canvas_size({"canvas": canvas}, "draft")
    keyed = [
        cutout(frame.convert("RGB").resize((ref_w, ref_h), Image.Resampling.LANCZOS)) for frame in frames
    ] if previous is None else []
    period = 2 * poses
    if previous is None:
        signatures = []
        boxes = []
        sig_w, sig_h = max(32, ref_w // 4), max(32, ref_h // 4)
        for frame in keyed:
            small = frame.resize((sig_w, sig_h), Image.Resampling.BOX)
            sig = np.asarray(small, dtype=np.float32)
            signatures.append(sig[:, :, :3].mean(2) * sig[:, :, 3] / 255)
            boxes.append(frame.getchannel("A").getbbox())

        def cost(start):
            win = boxes[start : start + period]
            edge = sum(
                b is None or b[0] <= 1 or b[1] <= 1 or b[2] >= ref_w - 1 or b[3] >= ref_h - 1 for b in win
            )
            return edge, float(np.abs(signatures[start] - signatures[start + period]).mean())

        if full_action:
            indices = action_indices(signatures, poses)
            start = indices[0]
            seen = [boxes[i] for i in indices if boxes[i]]
        else:
            start = min(range(6, total_frames - period), key=cost)
            seen = [b for b in boxes[start : start + period] if b]
        if not seen:
            raise ValueError("No sprite silhouette was visible against the background")
        height = float(np.median([b[3] - b[1] for b in seen]))
        center = float(np.median([(b[0] + b[2]) / 2 for b in seen]))
        ground = float(np.percentile([b[3] for b in seen], 90))
        layout = {"start": start, "height": height, "center": center, "ground": ground}
        if full_action and global_scale is None:
            # Give extended limbs room in every pose, using one crop for the whole take.
            left, right = min(b[0] for b in seen), max(b[2] for b in seen)
            layout["center"] = (left + right) / 2
            layout["height"] = max(height, (right - left) * BODY / (TILE - 2))
            layout["indices"] = indices
        elif full_action:
            layout["indices"] = indices
    else:
        layout = previous
        if not isinstance(layout, dict) or not all(
            k in layout for k in ("start", "height", "center", "ground")
        ):
            raise ValueError("Invalid sprite draft layout")
        start = layout["start"]
        if (
            type(start) is not int
            or not 0 <= start < (total_frames if "indices" in layout else total_frames - period)
            or not all(
                isinstance(layout[k], (int, float)) and math.isfinite(layout[k])
                for k in ("height", "center", "ground")
            )
            or layout["height"] <= 0
        ):
            raise ValueError("Invalid sprite draft layout")
    order = playback_indices(layout, poses, idle, export_24fps, total_frames=total_frames)
    indices = list(dict.fromkeys(order))
    if masks_for is None:
        depths = depth_for(indices) if depth_for is not None else [None] * len(indices)
        selected = [with_matte(frames[n], depth=d) for n, d in zip(indices, depths)]
    else:
        masks = masks_for(indices)
        if len(masks) != len(indices):
            raise ValueError("expected one foreground mask per sprite pose")
        depths = depth_for(indices) if depth_for is not None else [None] * len(indices)
        selected = [with_matte(frames[n], mask, depth=d) for n, mask, d in zip(indices, masks, depths)]
        if previous is None:
            # Re-measure silhouette height, horizontal center, and ground contact from the actual
            # foreground matte (BiRefNet) instead of the coarse background-key cutout, which can
            # fall back to the full canvas on tinted/lit backdrops and shrink the sprite to half size.
            ref_boxes = []
            for img in selected:
                sx, sy = ref_w / float(img.width), ref_h / float(img.height)
                bb = img.getchannel("A").point(lambda v: 255 if v >= 128 else 0).getbbox()
                if bb:
                    ref_boxes.append((bb[0] * sx, bb[1] * sy, bb[2] * sx, bb[3] * sy))
            if ref_boxes:
                height = float(np.median([b[3] - b[1] for b in ref_boxes]))
                center = float(np.median([(b[0] + b[2]) / 2.0 for b in ref_boxes]))
                ground = float(np.percentile([b[3] for b in ref_boxes], 90))
                layout["height"] = height
                layout["center"] = center
                layout["ground"] = ground
                if full_action and global_scale is None:
                    left, right = min(b[0] for b in ref_boxes), max(b[2] for b in ref_boxes)
                    layout["center"] = (left + right) / 2.0
                    layout["height"] = max(height, (right - left) * BODY / (TILE - 2))
    selected_by_index = dict(zip(indices, selected))
    if cache_dir is not None:
        save_frame_cache(cache_dir, selected_by_index)
    atlas = _render_cells(selected_by_index, order, layout, tile_w, tile_h, ref_w, ref_h, global_scale=global_scale, alpha=alpha)
    indexed = _indexed_from_atlas(atlas)
    return atlas, indexed, layout


def pack_atlas(rows, tile, style, include_webp=True, webp_quality=84):
    """Pack chosen strips, reflecting individual poses about their ground pivot. Supports square or non-square tiles and WebP."""
    import io
    import json
    import zipfile
    from pathlib import Path
    import numpy as np
    from PIL import Image, ImageOps

    tile_w, tile_h = validate_tile_size(tile)
    counts = [sheet["poses"] for _, _, sheet, _ in rows]
    columns, height = max(counts), len(rows)
    compact = columns * height * tile_w * tile_h > MAX_ATLAS_PIXELS
    if compact:
        cells = sum(counts)
        if cells * tile_w * tile_h > MAX_ATLAS_PIXELS:
            raise ValueError("This atlas is too large. Choose a smaller export size.")
        # Mixed 24 fps loops can waste most of each short row. Wrap their
        # cells and keep the exact location of every frame in the manifest.
        columns = 1 << (math.isqrt(cells - 1)).bit_length()
        height = math.ceil(cells / columns)
    atlas = Image.new("RGBA", (columns * tile_w, height * tile_h))
    animations = {}
    offset = 0
    for name, path, sheet, source in rows:
        count = sheet["poses"]
        frames = []
        spans = []
        with Image.open(path) as strip:
            if strip.size != (count * tile_w, tile_h):
                raise ValueError("The saved sprite strip does not match its take.")
            for index in range(count):
                row, col = divmod(offset + index, columns)
                x, y = col * tile_w, row * tile_h
                frame = strip.crop((index * tile_w, 0, (index + 1) * tile_w, tile_h)).convert("RGBA")
                if source["mirrored"]:
                    frame = ImageOps.mirror(frame)
                atlas.paste(frame, (x, y))
                alpha = np.asarray(frame.getchannel("A")) > 128
                occupied = np.flatnonzero(alpha.sum(axis=1) > 1)
                if len(occupied):
                    bottom = occupied[-1]
                    feet = np.flatnonzero(alpha[max(0, bottom - int(tile_h * .05)):bottom + 1].any(axis=0))
                    spans.append(int(feet[-1] - feet[0] + 1))
                frames.append(dict(x=x, y=y, w=tile_w, h=tile_h))
        offset += count if compact else columns
        pivot_x, pivot_y = sheet["pivot"]
        direction = name.rsplit("_", 1)[1]
        view = "side" if direction in ("E", "W") else "back" if "N" in direction else "front"
        animations[name] = dict(view=view, direction=direction, fps=sheet["fps"],
                                loop=dict(start=0, end=count - 1, mode="cycle"),
                                frame_size=dict(w=tile_w, h=tile_h), frames=frames,
                                anchor=dict(x=tile_w - pivot_x if source["mirrored"] else pivot_x, y=pivot_y,
                                            span=round(float(np.percentile(spans, 90)), 1) if spans else 0),
                                ground_speed=0, steps=[], waived=[],
                                **source)
    write_webp = bool(include_webp and style == "illustrated")
    if style == "indexed":
        atlas = atlas.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
    manifest = dict(format="animwright/1", name="Sprites", image="sprites.png", height=BODY * tile_h // TILE,
                    generated=dict(tool="H3 Editor", backend="h3", ai_generated=True, waived_gates=[]),
                    size=dict(w=atlas.width, h=atlas.height), animations=animations, rejected=[])
    if write_webp:
        manifest["image_webp"] = "sprites.webp"
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        with archive.open("sprites.png", "w") as image:
            atlas.save(image, format="PNG")
        if write_webp:
            with archive.open("sprites.webp", "w") as webp_file:
                atlas.save(webp_file, format="WEBP", quality=int(webp_quality), alpha_quality=90, method=6)
        archive.writestr("sprites.json", json.dumps(manifest, indent=2))
        template = (Path(__file__).parent / "editor" / "sprite-test.html").read_text(encoding="utf-8")
        # The browser can open the extracted ZIP from disk without a fetch for the manifest.
        prefix, slot = template.split('<script id="bundled-atlas" type="application/json">', 1)
        _, suffix = slot.split("</script>", 1)
        payload = json.dumps(manifest).replace("<", "\\u003c")
        archive.writestr("preview.html", prefix + '<script id="bundled-atlas" type="application/json">'
                         + payload + "</script>" + suffix)
    return data.getvalue()
