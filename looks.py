"""Reusable prompt styles. The library is installation data, outside every project."""

import copy
import json
import os
import uuid


BUILTINS = (
    ("cinematic", "Cinematic", "Live-action, cinematic. Natural skin texture, motivated lighting, restrained color grading and believable camera movement."),
    ("documentary", "Documentary", "Observational documentary. Available light, natural colors, candid performances, subtle handheld camera and realistic textures."),
    ("anime", "Anime", "Hand-drawn anime. Clean expressive linework, cel shading, richly painted backgrounds and fluid character animation."),
    ("watercolor", "Watercolor", "Animated watercolor illustration. Translucent pigment, soft paper texture, gentle edges and a restrained pastel palette."),
    ("clay", "Clay animation", "Handcrafted clay animation. Tactile sculpted characters, miniature sets, soft studio lighting and small handmade imperfections."),
    ("noir", "Film noir", "Black-and-white film noir. Deep shadows, hard directional lighting, rich tonal contrast and subtle film grain."),
    ("product", "Product studio", "Premium product photography in motion. Clean studio lighting, precise material detail, controlled reflections and an uncluttered composition."),
    ("retro", "Vintage film", "Live-action on vintage 35 mm film. Warm muted colors, soft highlight rolloff, fine organic grain and gently imperfect optics."),
)


def builtins():
    return [{"id": "builtin_" + key, "name": name, "style": style, "builtin": True} for key, name, style in BUILTINS]


def read_saved(path):
    try:
        with open(path, encoding="utf-8") as file:
            rows = json.load(file)
    except FileNotFoundError:
        return []
    if not isinstance(rows, list) or any(not isinstance(row, dict) or not str(row.get("id", "")).startswith("look_")
                                        or not isinstance(row.get("name"), str) or not isinstance(row.get("style"), str) for row in rows):
        raise ValueError("the saved look library is invalid; restore looks.json from a backup")
    return rows


def library(path):
    return builtins() + read_saved(path)


def write_saved(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + "." + uuid.uuid4().hex + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(rows, file, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def save(path, name, style, look_id=None):
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 80:
        raise ValueError("give the look a name of 1 to 80 characters")
    if not isinstance(style, str) or not style.strip() or len(style) > 12000:
        raise ValueError("a look needs style text of 1 to 12000 characters")
    rows = copy.deepcopy(read_saved(path))
    if look_id and not any(row["id"] == look_id for row in rows):
        raise ValueError("that saved look is gone; save it as a new look")
    if any(row["name"].casefold() == name.strip().casefold() and row["id"] != look_id for row in rows):
        raise ValueError("a saved look already has that name; choose it to update it, or use another name")
    if not look_id and len(rows) >= 500:
        raise ValueError("the library holds up to 500 saved looks; remove one before adding another")
    entry = {"id": look_id or "look_" + uuid.uuid4().hex[:12], "name": name.strip(), "style": style.strip()}
    rows = [entry if row["id"] == look_id else row for row in rows] if look_id else rows + [entry]
    write_saved(path, rows)
    return entry


def remove(path, look_id):
    rows = read_saved(path)
    if not any(row["id"] == look_id for row in rows):
        raise ValueError("that saved look is gone; built-in looks cannot be removed")
    write_saved(path, [row for row in rows if row["id"] != look_id])
