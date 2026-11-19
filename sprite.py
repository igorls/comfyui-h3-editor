"""Directional jog guides and small, transparent sprite strips for independent editor takes."""

import math

FRAMES = 39
WIDTH, HEIGHT = 256, 384
TILE, BODY = 64, 48
EXPORT_SIZES = (64, 128, 256)
PIVOT = (32, 59)
MATTE_PREFIX = "background_removal:"


def matte_model(available):
    for entry in sorted(available or ()):
        if entry.startswith(MATTE_PREFIX):
            name = entry[len(MATTE_PREFIX):]
            if name.replace("\\", "/").split("/")[-1].lower() == "birefnet.safetensors":
                return name
    return None
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


def settings(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {"direction", "camera", "poses", "guide", "motion", "cutout"}:
        raise ValueError("unknown sprite setting")
    direction, camera, poses = value.get("direction", "S"), value.get("camera", "rpg"), value.get("poses", 7)
    if direction not in tuple(DIRECTIONS) or camera not in tuple(CAMERAS):
        raise ValueError("choose a compass direction and a level, rpg or overhead camera")
    if type(poses) is not int or not 4 <= poses <= 12:
        raise ValueError("a sprite cycle needs 4 to 12 poses at 12 fps")
    out = {"direction": direction, "camera": camera, "poses": poses}
    if "motion" in value:
        if value["motion"] not in ("prompt", "idle") or value.get("guide") is not None:
            raise ValueError("choose one sprite motion source")
        out["motion"] = value["motion"]
    if "cutout" in value:
        if value["cutout"] not in ("auto", "edge"):
            raise ValueError("choose auto or edge sprite cutout")
        out["cutout"] = value["cutout"]
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


def instruction(config):
    motion = (
        "running in place"
        if "motion" not in config and "guide" not in config
        else "performing the described movement in place"
    )
    poses = "Match the poses and foot contacts in <Video 1>, keeping the same proportions and equipment. "
    if config.get("motion") == "prompt":
        poses = "Loop the movement with consistent proportions and equipment. "
    elif config.get("motion") == "idle":
        motion = "resting in its natural standing pose"
        poses = ("Make one subtle repeating idle gesture and return to the initial pose. "
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
        + "Center the whole character on pure white with no floor or shadow."
    )


def jog(direction, camera, poses):
    """Project the same articulated stride onto a pitched orthographic camera in each facing."""
    import numpy as np
    from PIL import Image, ImageDraw

    pitch = math.radians({"rpg": 45, "level": 0, "overhead": 90}[camera])
    dx = int("E" in direction) - int("W" in direction)
    dz = int("S" in direction) - int("N" in direction)
    length = math.hypot(dx, dz)

    def project(j):
        f, h, r = j
        x, z = (f * dx - r * dz) / length, (f * dz + r * dx) / length
        return (round(128 + x), round(192 - (h - 120) * math.cos(pitch) + z * math.sin(pitch)))

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
        im = Image.new("RGB", (WIDTH, HEIGHT))
        draw = ImageDraw.Draw(im)
        for a, b, color in limbs:
            xy = project(b)
            draw.line((project(a), xy), fill=color, width=2)
            draw.ellipse((xy[0] - 1, xy[1] - 1, xy[0] + 1, xy[1] + 1), fill=color)
        cycle.append(np.asarray(im))
    return np.stack([cycle[t % len(cycle)] for t in range(FRAMES)])


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


def with_matte(frame, mask):
    """Remove the estimated backdrop colour from partially covered foreground pixels."""
    import numpy as np
    from PIL import Image

    rgb = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
    ground = np.median(border, axis=0)
    alpha = np.asarray(mask, dtype=np.float32)[..., None].clip(0, 1)
    if alpha.shape[:2] != rgb.shape[:2]:
        raise ValueError("sprite matte must match the source frame")
    color = ((rgb - (1 - alpha) * ground) / np.maximum(alpha, .05)).clip(0, 1)
    coverage = ((alpha - .12) / .76).clip(0, 1)
    return Image.fromarray((np.concatenate((color, coverage), axis=2) * 255).round().astype("uint8"))


def strips(frames, poses, previous=None, masks_for=None, idle=False, cell_size=TILE):
    """Pick a loop once. Finish reuses the draft's phase, scale and ground contact in normalized source coordinates."""
    import numpy as np
    from PIL import Image

    if len(frames) != FRAMES:
        raise ValueError("Sprite Draft expects exactly 39 source frames")
    if type(cell_size) is not int or cell_size not in EXPORT_SIZES:
        raise ValueError("choose a sprite cell size of 64, 128 or 256 px")
    keyed = [
        cutout(frame.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)) for frame in frames
    ] if previous is None or (masks_for is None and cell_size == TILE) else []
    period = 2 * poses
    if previous is None:
        signatures = []
        boxes = []
        for frame in keyed:
            small = frame.resize((64, 96), Image.Resampling.BOX)
            sig = np.asarray(small, dtype=np.float32)
            signatures.append(sig[:, :, :3].mean(2) * sig[:, :, 3] / 255)
            boxes.append(frame.getchannel("A").getbbox())

        def cost(start):
            win = boxes[start : start + period]
            edge = sum(
                b is None or b[0] <= 1 or b[1] <= 1 or b[2] >= WIDTH - 1 or b[3] >= HEIGHT - 1 for b in win
            )
            return edge, float(np.abs(signatures[start] - signatures[start + period]).mean())

        start = min(range(6, FRAMES - period), key=cost)
        seen = [b for b in boxes[start : start + period] if b]
        if not seen:
            raise ValueError("No sprite silhouette was visible against the background")
        height = float(np.median([b[3] - b[1] for b in seen]))
        center = float(np.median([(b[0] + b[2]) / 2 for b in seen]))
        ground = float(np.percentile([b[3] for b in seen], 90))
        layout = {"start": start, "height": height, "center": center, "ground": ground}
    else:
        layout = previous
        if not isinstance(layout, dict) or not all(
            k in layout for k in ("start", "height", "center", "ground")
        ):
            raise ValueError("Invalid sprite draft layout")
        start = layout["start"]
        if (
            type(start) is not int
            or not 0 <= start < FRAMES - period
            or not all(
                isinstance(layout[k], (int, float)) and math.isfinite(layout[k])
                for k in ("height", "center", "ground")
            )
            or layout["height"] <= 0
        ):
            raise ValueError("Invalid sprite draft layout")
    scale = BODY / layout["height"]
    # Bake the return into the PNG. Left-to-right playback works in engines without a ping-pong mode.
    order = list(range(poses)) + (list(range(poses - 2, 0, -1)) if idle else [])
    atlas = Image.new("RGBA", (cell_size * len(order), cell_size))
    indices = [start + 2 * i for i in range(poses)]
    if masks_for is None:
        selected = [keyed[n] if cell_size == TILE else cutout(frames[n]) for n in indices]
    else:
        masks = masks_for(indices)
        if len(masks) != poses:
            raise ValueError("expected one foreground mask per sprite pose")
        selected = [with_matte(frames[n], mask) for n, mask in zip(indices, masks)]
    tiles = []
    for frame in selected:
        sx, sy = frame.width / WIDTH, frame.height / HEIGHT
        x = (layout["center"] - PIVOT[0] / scale) * sx
        y = (layout["ground"] - PIVOT[1] / scale) * sy
        box = (x, y, x + TILE * sx / scale, y + TILE * sy / scale)
        # The crop stays in source coordinates. Only its output sampling density changes.
        size = (cell_size, cell_size)
        fine = frame.convert("RGBa").transform((4 * cell_size, 4 * cell_size), Image.Transform.EXTENT, box, Image.Resampling.BILINEAR)
        tile = fine.resize(size, Image.Resampling.LANCZOS).convert("RGBA")
        coverage = fine.getchannel("a").resize(size, Image.Resampling.BOX)
        tile.paste((0, 0, 0, 0), mask=coverage.point(lambda v: 0 if v else 255))
        tiles.append(tile)
    for column, index in enumerate(order):
        atlas.paste(tiles[index], (cell_size * column, 0))
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
    return atlas, indexed, layout


def pack_atlas(rows, tile, style):
    """Pack chosen strips, reflecting individual poses about their ground pivot."""
    import io
    import json
    import zipfile
    from pathlib import Path
    import numpy as np
    from PIL import Image, ImageOps

    width = max(sheet["poses"] for _, _, sheet, _ in rows) * tile
    atlas = Image.new("RGBA", (width, len(rows) * tile))
    animations = {}
    for row, (name, path, sheet, source) in enumerate(rows):
        count = sheet["poses"]
        y = row * tile
        frames = []
        spans = []
        with Image.open(path) as strip:
            if strip.size != (count * tile, tile):
                raise ValueError("The saved sprite strip does not match its take.")
            for index in range(count):
                x = index * tile
                frame = strip.crop((x, 0, x + tile, tile)).convert("RGBA")
                if source["mirrored"]:
                    frame = ImageOps.mirror(frame)
                atlas.paste(frame, (x, y))
                alpha = np.asarray(frame.getchannel("A")) > 128
                occupied = np.flatnonzero(alpha.sum(axis=1) > 1)
                if len(occupied):
                    bottom = occupied[-1]
                    feet = np.flatnonzero(alpha[max(0, bottom - int(tile * .05)):bottom + 1].any(axis=0))
                    spans.append(int(feet[-1] - feet[0] + 1))
                frames.append(dict(x=x, y=y, w=tile, h=tile))
        pivot_x, pivot_y = sheet["pivot"]
        direction = name.rsplit("_", 1)[1]
        view = "side" if direction in ("E", "W") else "back" if "N" in direction else "front"
        animations[name] = dict(view=view, direction=direction, fps=sheet["fps"],
                                loop=dict(start=0, end=count - 1, mode="cycle"),
                                frame_size=dict(w=tile, h=tile), frames=frames,
                                anchor=dict(x=tile - pivot_x if source["mirrored"] else pivot_x, y=pivot_y,
                                            span=round(float(np.percentile(spans, 90)), 1) if spans else 0),
                                ground_speed=0, steps=[], waived=[],
                                **source)
    if style == "indexed":
        atlas = atlas.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
    manifest = dict(format="animwright/1", name="Sprites", image="sprites.png", height=BODY * tile // TILE,
                    generated=dict(tool="H3 Editor", backend="h3", ai_generated=True, waived_gates=[]),
                    size=dict(w=atlas.width, h=atlas.height), animations=animations, rejected=[])
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        with archive.open("sprites.png", "w") as image:
            atlas.save(image, format="PNG")
        archive.writestr("sprites.json", json.dumps(manifest, indent=2))
        template = (Path(__file__).parent / "editor" / "sprite-test.html").read_text(encoding="utf-8")
        # The browser can open the extracted ZIP from disk without a fetch for the manifest.
        prefix, slot = template.split('<script id="bundled-atlas" type="application/json">', 1)
        _, suffix = slot.split("</script>", 1)
        payload = json.dumps(manifest).replace("<", "\\u003c")
        archive.writestr("preview.html", prefix + '<script id="bundled-atlas" type="application/json">'
                         + payload + "</script>" + suffix)
    return data.getvalue()
