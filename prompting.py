"""H3 prompts in the structure from MiniMax's writing guides. No Comfy import."""

DEFAULT_SOUND = "Quiet natural ambience fits the scene and continues throughout."


def dialogue_line(speaker, line, language="English"):
    """A spoken line. The identifying phrase sits outside <d>, the words inside it."""
    who = (speaker or "The speaker").strip()
    return "%s (S1) says: <d>[%s] %s</d>" % (who, language, line.strip())


def compose_prompt(style, action, camera="", speaker="", dialogue="", sound="", music="", first_frame=False):
    """A text or first-frame take: instruction line, then the three core fields."""
    parts = [(style or "Live-action, cinematic").strip().rstrip(",.") + ","]
    parts.append(action.strip())
    if camera.strip():
        parts.append(camera.strip())
    if dialogue.strip():
        parts.append(dialogue_line(speaker, dialogue))
    body = "integrated_multimodal_description: [Shot 1] " + " ".join(parts)
    fields = [
        body,
        "overall_soundscape: " + (sound.strip() or DEFAULT_SOUND),
        "non_diegetic_music: " + (music.strip() or "N/A"),
    ]
    if first_frame:
        fields.insert(0, "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.")
    return "\n\n".join(fields)
