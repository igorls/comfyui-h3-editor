"""Directional jog guides and small, transparent sprite strips for independent editor takes."""

import math

FRAMES = 39
WIDTH, HEIGHT = 256, 384
TILE, BODY = 64, 48
PIVOT = (32, 59)
DIRECTIONS = {
    "S": "south, down the screen",
    "E": "east, screen right",
    "N": "north, up the screen, back visible",
    "W": "west, screen left",
}
CAMERAS = {
    "rpg": "45 degrees above the ground, visible top planes and foreshortened legs",
    "level": "at eye level",
    "overhead": "straight down from 90 degrees above, crown and shoulders visible",
}


def settings(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {"direction", "camera", "poses", "guide", "motion"}:
        raise ValueError("sprite takes direction, camera, poses and an optional guide video")
    direction, camera, poses = value.get("direction", "S"), value.get("camera", "rpg"), value.get("poses", 7)
    if direction not in tuple(DIRECTIONS) or camera not in tuple(CAMERAS):
        raise ValueError("choose a cardinal sprite direction and a level, rpg or overhead camera")
    if type(poses) is not int or not 4 <= poses <= 12:
        raise ValueError("a sprite cycle needs 4 to 12 poses at 12 fps")
    out = {"direction": direction, "camera": camera, "poses": poses}
    if "motion" in value:
        if value["motion"] != "prompt" or value.get("guide") is not None:
            raise ValueError("Use a guide video or motion from the prompt")
        out["motion"] = "prompt"
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

    def project(j):
        f, h, r = j
        x, z = {"S": (-r, f), "E": (f, r), "N": (r, -f), "W": (-f, -r)}[direction]
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
        if direction != "N" and camera != "overhead":
            limbs.append((neck, nose, "#0055ff"))
        for side in (-1, 1):
            if (direction, side) in (("E", -1), ("W", 1)):
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


def strips(frames, poses, previous=None):
    """Pick a loop once. Finish reuses the draft's phase, scale and ground contact in normalized source coordinates."""
    import numpy as np
    from PIL import Image

    if len(frames) != FRAMES:
        raise ValueError("Sprite Draft expects exactly 39 source frames")
    keyed = [
        cutout(frame.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)) for frame in frames
    ]
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
    atlas = Image.new("RGBA", (TILE * poses, TILE))
    for i in range(poses):
        # Inverse mapping avoids sizing each pose independently or allocating a huge intermediate on sparse masks.
        origin = (layout["center"] - PIVOT[0] / scale, layout["ground"] - PIVOT[1] / scale)
        box = (*origin, origin[0] + TILE / scale, origin[1] + TILE / scale)
        tile = keyed[start + 2 * i].transform(
            (TILE, TILE), Image.Transform.EXTENT, box, Image.Resampling.BICUBIC
        )
        atlas.paste(tile, (TILE * i, 0))
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
