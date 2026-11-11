"""H3 prompts in the structure from MiniMax's writing guides. No Comfy import."""

import re

DEFAULT_SOUND = "Quiet natural ambience fits the scene and continues throughout."


VOICE = "<Audio 1>"


def dialogue_line(speaker, line, language="English", voice=False):
    """A spoken line. The identifying phrase sits outside <d>, the words inside it, in their own language.

    The speaker ID follows the identity itself, so "@Ana, off-screen" reads "<Subject 1> (S1), off-screen, says".
    """
    who = (speaker or "The speaker").strip()
    who = who[:1].upper() + who[1:]
    if "(S1)" not in who:
        tag = re.match(r"<Subject \d+>", who)
        who = tag.group(0) + " (S1)" + who[tag.end():] if tag else who + " (S1)"
    lead = who
    if voice:
        lead += ", using the voice timbre referenced from %s," % VOICE
    elif who.split("(S1)", 1)[1].strip():
        lead += ","
    return "%s says: <d>[%s] %s</d>" % (lead, language or "English", line.strip())


def compose_prompt(style, action, camera="", speaker="", dialogue="", sound="", music="", first_frame=False,
                   language="English"):
    """A text or first-frame take: instruction line, then the three core fields."""
    parts = [(style or "Live-action, cinematic").strip().rstrip(",.") + ","]
    parts.append(action.strip())
    if camera.strip():
        parts.append(camera.strip())
    if dialogue.strip():
        parts.append(dialogue_line(speaker, dialogue, language))
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
                             camera="", speaker="", dialogue="", sound="", music="", language="English", voice=None):
    """A take with references: the six sections of the full-reference guide.

    Picture numbers follow H3 Take: first frame, last frame, then references.
    subjects are dicts with a name and a description. voice says whose timbre <Audio 1> carries:
    {"subject": N} for a subject in the shot, or {"description": "..."} for an unseen voice.
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
        # A subject with several pictures names them all, in the order H3 Take is given them.
        tags = []
        for _ in range(max(1, int(subject.get("picture_count") or 1))):
            picture += 1
            tags.append("<Picture %s>" % picture)
        where = tags[0] if len(tags) == 1 else ", ".join(tags[:-1]) + " and " + tags[-1]
        who = (subject.get("description") or subject["name"]).strip().rstrip(".")
        definitions.append("<Subject %s> is %s in %s." % (number, who, where))
        retention.append("<Subject %s> (appears in [Shot 1]): fully_preserved - their identity and appearance are retained." % number)

    voiced = bool(voice) and bool(dialogue.strip())
    if voiced:
        owner = ("<Subject %s>" % voice["subject"]) if voice.get("subject") else (voice.get("description") or "the speaker").strip().rstrip(".")
        owner = owner if owner.startswith("<") else owner[:1].lower() + owner[1:]
        definitions.append("%s is the voice-timbre reference for %s (S1)." % (VOICE, owner))
        retention.append("%s: reference - the target speaker follows %s's voice timbre and delivery without copying the original signal." % (VOICE, VOICE))

    tasks = []
    if first_tag or last_tag:
        tasks.append("keyframe completion")
    if subjects:
        tasks.append("reference generation")
    if voiced:
        tasks.append("audio reference")
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
        shot += " " + dialogue_line(tag_subjects(speaker, subjects), dialogue, language, voice=voiced)
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
