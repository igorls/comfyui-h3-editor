"""H3 prompts in the structure from MiniMax's writing guides. No Comfy import."""

import re

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


def tag_subjects(text, subjects):
    """Turn @Name into <Subject N>. Longest names first, so @Old Tom wins over @Old."""
    order = sorted(enumerate(subjects, 1), key=lambda pair: -len(pair[1]["name"]))
    for number, subject in order:
        text = re.sub("@" + re.escape(subject["name"]), "<Subject %s>" % number, text, flags=re.IGNORECASE)
    return text


def compose_reference_prompt(style, action, subjects=(), first_frame=False, last_frame=False,
                             camera="", speaker="", dialogue="", sound="", music=""):
    """A take with references: the six sections of the full-reference guide.

    Picture numbers follow H3 Take: first frame, last frame, then references.
    subjects are dicts with a name and a description.
    """
    subjects = list(subjects)
    picture = 0
    first_tag = last_tag = None
    if first_frame:
        picture += 1
        first_tag = "<Picture %s>" % picture
    if last_frame:
        picture += 1
        last_tag = "<Picture %s>" % picture

    definitions = []
    retention = []
    if first_tag:
        definitions.append("%s is the first frame of [Shot 1]." % first_tag)
        retention.append("%s ([Shot 1] first frame): fully_preserved - the shot begins exactly on this frame." % first_tag)
    if last_tag:
        definitions.append("%s is the last frame of [Shot 1]." % last_tag)
        retention.append("%s ([Shot 1] last frame): fully_preserved - the shot ends exactly on this frame." % last_tag)
    for number, subject in enumerate(subjects, 1):
        picture += 1
        who = (subject.get("description") or subject["name"]).strip().rstrip(".")
        definitions.append("<Subject %s> is %s in <Picture %s>." % (number, who, picture))
        retention.append("<Subject %s> (appears in [Shot 1]): fully_preserved - their identity and appearance are retained." % number)

    tasks = []
    if first_tag or last_tag:
        tasks.append("keyframe completion")
    if subjects:
        tasks.append("reference generation")
    body = tag_subjects(action.strip(), subjects)
    missing = [
        "<Subject %s>" % number for number, _ in enumerate(subjects, 1)
        if "<Subject %s>" % number not in body
    ]
    lead = ""
    if missing:
        lead = " and ".join(missing) + (" appears in the shot. " if len(missing) == 1 else " appear in the shot. ")
    shot = lead + body
    if first_tag:
        shot = "The shot begins from %s. %s" % (first_tag, shot)
    if camera.strip():
        shot += " " + camera.strip()
    if dialogue.strip():
        shot += " " + dialogue_line(tag_subjects(speaker, subjects), dialogue)
    if last_tag:
        shot += " The shot ends on %s." % last_tag

    sentence = re.split(r"(?<=[.!?])\s", body, maxsplit=1)[0]
    look = (style or "Live-action, cinematic").strip().rstrip(",.")
    return "\n\n".join([
        "subject_definitions:\n" + "\n".join(definitions),
        "summary:\n[%s] %s" % (" + ".join(tasks), sentence),
        "retention_analysis:\n" + "\n".join(retention),
        "detailed_description:\nThe target video is in a %s style.\n[Shot 1] %s" % (look.lower(), shot),
        "overall_soundscape:\n" + (sound.strip() or DEFAULT_SOUND),
        "non_diegetic_music:\n" + (music.strip() or "N/A"),
    ])
