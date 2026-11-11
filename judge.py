"""Rate a piece of video against a rubric, with Google's Gemini Decision model.

The model reads a video and answers typed questions with calibrated probabilities instead of prose. A rubric question
gives ordered levels, worst first, and comes back as an expected level (a number that can fall between two levels), the
probability of each level, and a confidence. Nothing is generated, so an answer is short, cheap and comparable from one
range of a cut to the next.

This module is the whole integration: a default rubric for generated video, the request, and the reading of the reply.
It needs the user's own Gemini API key and sends the video to Google, so the editor uses it only when asked to.
"""
import asyncio
import os

API = "https://generativelanguage.googleapis.com"
MODEL = "gemini-decision-1"
INLINE_LIMIT = 15 * 1024 * 1024        # larger files go through the Files API
RESOLUTIONS = ("low", "medium", "high")

# What to look for in a stretch of generated video. Most criteria are a check with two described ends, the bad one
# first: the answer is then how likely the good end is, which told a fresh shot from a degraded one far more clearly
# in testing than a five-level scale did (sharp face 0.77 against 0.51, same person 0.94 against 0.75). "match" is
# asked only when a cast picture is sent along. Cuts are not here on purpose: the model samples about a frame a
# second and rated a run of jump cuts as clean, so the editor adds its own measured row for them (see cuts.py).
# A criterion on lip sync was tried and dropped: it scored a screen recording as well as a talking presenter.
RUBRIC = [
    {"key": "same", "title": "Same person start to end",
     "instructions": "Compare the main person in the first second of the video with the same person in the last second.",
     "levels": ["By the end they look like a different, older or altered person", "They look like exactly the same person"]},
    {"key": "match", "title": "Matches the cast picture", "needs": "reference",
     "instructions": "Compare the person in the video with the reference picture of them. Judge the face: its shape, the eyes, "
                     "the apparent age and the skin tone.",
     "levels": ["The face no longer looks like the reference: older, heavier, darker or another person", "The face matches the reference picture"]},
    {"key": "sharp", "title": "Sharp face",
     "instructions": "Look at the main person's face. Is it sharp, with visible detail in the eyes, lashes and skin?",
     "levels": ["Soft, smeared or blurry face", "Sharp face with fine detail"]},
    {"key": "light", "title": "Natural light",
     "instructions": "Look at the overall lighting and contrast of the picture.",
     "levels": ["Harsh, crushed shadows or overcooked contrast", "Soft, natural lighting"]},
    {"key": "room", "title": "Background stays put",
     "instructions": "Watch the background from the first second to the last: screens, lamps, walls, shelves and objects.",
     "levels": ["Something in the background turns into something else", "The background stays exactly the same"]},
    {"key": "motion", "title": "Natural movement",
     "instructions": "Watch how people and objects move: faces, hands, hair, anything that changes shape.",
     "levels": ["Something warps, melts, flickers or moves in a way real footage would not", "Everything moves as real footage would"]},
    {"key": "attention", "title": "Holds attention",
     "instructions": "Rate how well the video holds a viewer's attention: variety of pictures, pacing, and whether something changes "
                     "often enough to stay interesting.",
     "levels": ["Static and dull", "Slow, little changes", "Fine, with flat stretches", "Engaging", "Hard to look away from"]},
]


class JudgeError(Exception):
    """A rating that could not be made, with a message fit to show the user."""


def clean_rubric(rubric):
    """A rubric as the caller wrote it, checked: a list of {key, instructions, levels, title?} with 2 to 10 levels each."""
    if rubric is None:
        return RUBRIC
    if not isinstance(rubric, list) or not 1 <= len(rubric) <= 20:
        raise ValueError("a rubric is a list of 1 to 20 criteria")
    out, seen = [], set()
    for row in rubric:
        if not isinstance(row, dict):
            raise ValueError("each criterion is an object with key, instructions and levels")
        key, levels = row.get("key"), row.get("levels")
        if not isinstance(key, str) or not key or len(key) > 64 or not all(ch.isalnum() or ch in "_-" for ch in key) or key in seen:
            raise ValueError("a criterion's key is a unique word of letters, numbers, _ and -")
        if not isinstance(row.get("instructions"), str) or not row["instructions"].strip():
            raise ValueError("the criterion %s needs instructions: what to rate" % key)
        if not isinstance(levels, list) or not 2 <= len(levels) <= 10 or not all(isinstance(x, str) and x.strip() for x in levels):
            raise ValueError("the criterion %s needs 2 to 10 levels, worst first, each a short description" % key)
        seen.add(key)
        out.append({"key": key, "title": str(row.get("title") or key)[:60], "instructions": row["instructions"].strip()[:2000],
                    "levels": [x.strip()[:300] for x in levels], **({"needs": "reference"} if row.get("needs") == "reference" else {})})
    return out


def questions(rubric):
    """The rubric as the API's questions: one score question per criterion."""
    return {row["key"]: {"type": "score", "instructions": row["instructions"], "criteria": row["levels"]} for row in rubric}


def read_answers(rubric, reply):
    """The API's reply as one row per criterion: its level as a number and as 0..1, the level it lands on, and how sure."""
    answers = reply.get("answers") or {}
    rows = []
    for row in rubric:
        got = answers.get(row["key"]) or {}
        if not isinstance(got.get("score"), (int, float)):
            rows.append({"key": row["key"], "title": row["title"], "error": "no answer came back for this criterion"})
            continue
        top = len(row["levels"]) - 1
        score = max(0.0, min(float(top), float(got["score"])))
        chances = got.get("probabilities") or {}
        rows.append({"key": row["key"], "title": row["title"], "score": round(score, 3), "of": top, "part": round(score / top, 4),
                     "level": row["levels"][int(round(score))], "levels": row["levels"],
                     "chances": [round(float(chances.get(str(n), 0.0)), 4) for n in range(top + 1)],
                     "confidence": round(float(got["confidence"]), 4) if isinstance(got.get("confidence"), (int, float)) else None})
    usage = reply.get("usage") or {}
    return {"rows": rows, "tokens": usage.get("total_input_tokens")}


def explain(status, body):
    """Why Google refused, in words for the user."""
    error = (body or {}).get("error") if isinstance(body, dict) else None
    message = (error or {}).get("message") if isinstance(error, dict) else None
    code = str((error or {}).get("code") or (error or {}).get("status") or "")
    if status in (401, 403) or "API key" in str(message):
        return "Google did not accept the Gemini API key. Check it in the editor's rating settings."
    if status == 429:
        return "Google's rate limit for this key was reached. Wait a minute, or rate a shorter range."
    if code in ("safety", "prohibited_content", "blocklist", "image_safety", "content_blocked"):
        return "Google declined to rate this video (%s)." % code
    return "Google answered %s: %s" % (status, str(message or body)[:300])


async def _json(response):
    try:
        return await response.json(content_type=None)
    except Exception:  # an error page that is not JSON
        return {"error": {"message": (await response.text())[:300]}}


async def upload(session, key, path):
    """Send a file to Google's Files API. Returns (uri, name); the file is deleted by remove(), or by Google after 48 hours."""
    size = os.path.getsize(path)
    headers = {"x-goog-api-key": key, "X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
               "X-Goog-Upload-Header-Content-Length": str(size), "X-Goog-Upload-Header-Content-Type": "video/mp4",
               "Content-Type": "application/json"}
    async with session.post(API + "/upload/v1beta/files", headers=headers, json={"file": {"display_name": "h3-editor range"}}) as response:
        if response.status >= 400:
            raise JudgeError(explain(response.status, await _json(response)))
        target = response.headers.get("x-goog-upload-url")
    if not target:
        raise JudgeError("Google did not say where to upload the video.")
    with open(path, "rb") as file:
        data = file.read()
    headers = {"Content-Length": str(size), "X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"}
    async with session.post(target, headers=headers, data=data) as response:
        body = await _json(response)
        if response.status >= 400:
            raise JudgeError(explain(response.status, body))
    file = body.get("file") or {}
    if not file.get("uri") or not file.get("name"):
        raise JudgeError("Google took the video but did not say where it put it.")
    # A video is processed before it can be read. Wait for that, when the service reports a state at all.
    for _ in range(90):
        if file.get("state") in (None, "ACTIVE"):
            break
        if file.get("state") == "FAILED":
            raise JudgeError("Google could not process the video.")
        await asyncio.sleep(2)
        async with session.get("%s/v1beta/%s" % (API, file["name"]), headers={"x-goog-api-key": key}) as response:
            file = dict(file, **(await _json(response)))
    return file["uri"], file["name"]


async def remove(session, key, name):
    """Delete an uploaded file. A failure here is not worth failing the rating for: Google deletes it in 48 hours anyway."""
    try:
        async with session.delete("%s/v1beta/%s" % (API, name), headers={"x-goog-api-key": key}):
            pass
    except Exception:
        pass


def picture_parts(pictures):
    """Reference pictures as input blocks: each a line of text saying who it is, then the picture itself."""
    import base64

    parts = []
    for label, path in pictures or []:
        kind = {".png": "image/png", ".webp": "image/webp"}.get(os.path.splitext(path)[1].lower(), "image/jpeg")
        with open(path, "rb") as file:
            data = base64.b64encode(file.read()).decode("ascii")
        parts.append({"type": "text", "text": "\nReference picture of %s:\n" % label})
        parts.append({"type": "image", "data": data, "mime_type": kind, "resolution": "medium"})
    if parts:
        parts.append({"type": "text", "text": "\nThe video to rate:\n"})
    return parts


async def rate(key, path, rubric=None, resolution="low", note="", pictures=None):
    """Rate the video at path against the rubric. Returns read_answers()'s rows. The upload is deleted afterwards.

    pictures is a list of (name, file path): reference pictures of the people in the video, for questions that compare.
    """
    import aiohttp

    if not key:
        raise JudgeError("No Gemini API key is set. Add one in the editor's rating settings.")
    rubric = clean_rubric(rubric)
    if not pictures:
        rubric = [row for row in rubric if row.get("needs") != "reference"]      # nothing to compare against
    if resolution not in RESOLUTIONS:
        raise ValueError("resolution is one of %s" % ", ".join(RESOLUTIONS))
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        uri, name = await upload(session, key, path)
        try:
            parts = picture_parts(pictures) + [{"type": "video", "uri": uri, "mime_type": "video/mp4", "resolution": resolution}]
            if note:
                parts.append({"type": "text", "text": "\n" + str(note)[:2000]})
            body = {"model": MODEL, "input": parts, "questions": questions(rubric)}
            async with session.post(API + "/v1beta/decisions", headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                                    json=body) as response:
                reply = await _json(response)
                if response.status >= 400:
                    raise JudgeError(explain(response.status, reply))
        finally:
            await remove(session, key, name)
    return read_answers(rubric, reply)
