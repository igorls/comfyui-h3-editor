---
name: h3-editor
description: Make videos with the H3 Editor, the timeline video editor for MiniMax H3 that runs inside the user's ComfyUI. Plan shots, generate them as drafts, review them, join them with generated extensions and bridges, add real footage and pictures, give characters and narrators consistent faces and voices, and export the cut. Use this skill whenever the user wants a video, ad, teaser, explainer, product demo, trailer, or scene made or edited with H3, MiniMax, ComfyUI, or "the editor", or hands you a script, storyboard, or shot list to turn into video, even if they never say "H3 Editor".
---

# H3 Editor

The H3 Editor is a timeline editor that lives inside the user's ComfyUI. Each clip is
rendered by MiniMax H3 as a video with sound. The timeline only points at those
renders, so moving, trimming, splitting, speed changes and transitions never render
anything again. A new render of a clip becomes another take; the clip keeps all of them.

You drive it over HTTP with `scripts/h3.py`, which needs only Python. The user can have
the editor open in ComfyUI while you work: your changes appear there as you make them,
and theirs merge with yours. Treat it as a shared cutting room, not a file you own.

Read `references/api.md` when you need an operation's exact fields, the summary format,
or the rules for linked clips. Everything else you need is here.

## Before you start

1. Run `python scripts/h3.py check`. If it cannot reach the editor, tell the user. The
   editor runs in their ComfyUI and renders on its GPU, so never start a ComfyUI of
   your own or render some other way. If ComfyUI is somewhere other than
   `http://127.0.0.1:8188`, set `H3_EDITOR_URL` or pass `--url`.
2. Pin down the brief: how long the cut runs, its shape (16:9, 9:16 or 1:1), the
   language people speak, who appears, the look, and any real material to use such as
   screen recordings, product shots or logos. Ask for what you cannot infer. A wrong
   guess about format or language costs a full set of renders.

## How a cut gets made

1. **Write the shot list first.** One line per shot: what we see and hear, how long it
   runs, who is in it, what is said. Shots of 2 to 8 seconds work best. H3 makes one
   continuous shot per render, so a cut is several clips.
2. **Create the project** with `python scripts/h3.py create "Morning teaser"`. It
   prints the project's folder name, here `Morning_teaser`, which every later command
   takes. Then set it up in one batch: `set_project` with the style, the format and the
   language, then the cast (see below).
3. **Add every clip in one batch** with `add_clip`, then render them all as drafts.
   Drafts are small and fast. They exist to judge the idea, the framing and the
   timing.
4. **Wait, then look at every draft.** `python scripts/h3.py wait PROJECT` blocks until
   nothing is rendering. `look PROJECT` then saves a sheet of 12 frames for every clip on
   the timeline, and prints the paths. Open each one. Check the subject, framing, motion,
   continuity with its neighbours, and anything that came out wrong. For faces, hands or
   lettering, `look PROJECT CLIP --detail` gives 4 large frames, and `--first`/`--last`
   narrow it to part of the take. A summary line that says `draft` only means the
   render finished, not that it is good.
5. **Fix what is wrong.** Rewrite the clip's action and render it again, which makes a
   new take. To render variations of a result you like, set its seed on the clip.
   `choose_take` switches between takes. Tell the user which shots you redid and why.
6. **Join shots that must flow.** An ordinary cut between clips is fine when the camera
   changes. When the action must continue, use the linked clips described below
   instead of hoping two renders match.
7. **Render finals only for the clips you keep**, after the user has seen the drafts if
   they are around. Say what it will cost first (see Costs). A final is a new render at
   a larger size. Even with the draft's seed set on the clip, it will not repeat the
   draft frame for frame: expect the same idea with different details, and look at the
   finals as carefully as the drafts. Export takes its size from the first clip on the
   timeline, so an all-draft cut exports at draft size.
8. **Export** and report back (see Reporting).

## Batches

Every edit goes through `ops`: a JSON list of operations, applied together or not at all.
Write it to a file, then run `python scripts/h3.py ops PROJECT shots.json`. Name a result
with `as`, and later operations in the same batch can use `"$name"` for its clip or
timeline item.

```json
[
  {"op": "set_project", "fields": {"style": "Live-action, cinematic, soft daylight", "format": "16:9", "language": "English"}},
  {"op": "add_clip", "as": "open", "fields": {"name": "Desk", "seconds": 4,
    "action": "A mug of coffee steams on a wooden desk by a rain-streaked window.",
    "camera": "Slow push in"}},
  {"op": "add_clip", "as": "phone", "fields": {"name": "Phone", "seconds": 3,
    "action": "A smartphone on the desk lights up with a new message. A hand picks it up."}},
  {"op": "render", "clip": "$open"},
  {"op": "render", "clip": "$phone"}
]
```

`render` makes a draft unless you add `"quality": "final"`. The reply lists what each
operation made, labelled with its `op` and `as`, and then the summary. Ids change as clips are
added and removed. The user may also be editing, so read `summary` again before a batch
that refers to existing items, rather than reusing ids from memory. If a batch is
refused, the error names the operation and the reason, and nothing changed.

## Writing a shot

The editor writes the actual H3 prompt from a clip's fields, in the structure MiniMax's
prompt guides ask for. Your job is the fields.

- **action** is the shot itself: what we see and hear, in the order it happens,
  in plain present tense. Be concrete about people, place, light and movement. Keep
  it to one continuous shot with no cuts inside it. Write it in English, whatever
  language people speak.
- **camera** is framing and movement: "Close-up, handheld", "Slow dolly out".
- **style** belongs to the project and goes into every clip, so do not repeat it.
- **dialogue** is the exact words spoken, in their own language. Set the project's
  `language` to match, or a clip's `language` when that line differs. **speaker** says
  who talks, for example "The barista, cheerful". Allow about 2.5 words per second of
  clip, or the line gets rushed or cut. H3 decides where in the shot a line falls.
  Phrases like "at the very end of the shot" help a little, but when a line must land
  on a precise moment, make that moment its own short shot, or record the line and
  use a track (below).
- **sound** is the soundscape, and **music** is music laid under the shot. Music is
  made per clip, so it changes at every cut. There is no music bed across the
  timeline yet. For a continuous score, leave music empty and tell the user to lay
  one over the export.
- H3 draws short words fairly well when you spell them out, for example "a sign with
  the word OPEN in large white capital letters". Check the detail sheet. It cannot
  reproduce logos, long text or a real app's interface. For product screens, logos and
  anything that must be exact, import the real recording or picture. Then generate the
  shots around it, or extend and bridge from it.

H3 renders set lengths, so `seconds` snaps to the nearest one. The reply and the
summary show the real frame counts at 24 fps.

| seconds asked | 2 | 3 | 4 | 4.5 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| frames rendered | 56 | 73 | 90 | 107 | 124 | 141 | 175 | 192 |
| seconds rendered | 2.33 | 3.04 | 3.75 | 4.46 | 5.17 | 5.88 | 7.29 | 8.00 |

## Cast: same face, same voice

Upload a picture with `python scripts/h3.py asset PROJECT face.png`, then use the
returned asset in `add_cast` with a name and a description. Choose who is in a clip with
its `cast` field, and write `@Name` in its action to place them. A cast member keeps the
same face and look from clip to clip.

One picture only shows what is in it. When the user has several pictures of a character
(front, three-quarter, profile, the whole figure), add the others as the member's
`pictures`: shots then keep details the main picture cannot show, such as hair length from
behind or what they wear.

A voice works the same way. Give a cast member a `voice` asset, or add a voice with no
picture, such as a narrator. Write `@Name` in a clip's speaker field, for example
`"@Narrator, off-screen"`. Every line they say then uses that voice's timbre. A cast
member needs a picture or a voice to exist, so get the voice first:

- upload a clean recording of it, 5 to 30 seconds, with `asset`; or
- make a voice sample: a clip of 6 seconds or more with `place: false`, where only
  this person speaks a few sentences, with `sound` set to "No sound effects, no room
  noise, no music, only the voice". Render it, then run `voice-from PROJECT CLIP`.
  It takes the whole sound of the clip, so anything else in it ends up in the voice.

Then `add_cast` with the voice asset and a description of how they sound, for example
"a warm, calm female narrator in her forties". Render the lines that use it after that.
A clip only uses the voice when it has dialogue. You cannot hear, so you cannot judge a
voice or a line. Tell the user which clips to listen to, rather than claiming they sound
right.

## Picture to a recorded track

When the sound already exists, hold it and have the picture generated to it, instead of
generating new speech. Examples are an approved narration, a voice-over from a speech
service, or an interview. This is how a talking presenter follows a fixed take, lips and
all.

1. Upload the sound with `asset`. MP3, WAV and the like all work.
2. Make the first window an ordinary `add_clip` with a `track` field: the asset plus
   `from`, the second of the sound where this clip's first frame sits, usually 0. Add
   `dialogue` with the words spoken in that window, which helps the mouth, and
   `language`. Use `cast` for the presenter's identity, and `first` to start exactly on
   an approved still. About 5 seconds per window works well.
3. Render it and look at it. Then `extend_after` from its timeline item. The new clip
   holds the next slice of the same track by itself, and keeps the cast. Give it an
   action and, in `fields`, the next words. Repeat until the track is covered. Each
   extension needs its source rendered first, so this is a loop of render, wait,
   extend. At the default 4.25 seconds, each one adds 102 new frames.
4. When a take finishes, the server replaces its sound with that exact slice of the
   track: sample aligned, level untouched. The cut plays the track straight through,
   with no drift at the joins.

Keep those items at 100% speed. If you trim a clip in the chain after its extension was
rendered, the extension shows `source changed` and needs another render. A cast voice is
ignored on a clip that holds a track, because the track already is the voice. To find the
words for each window, use word timings if the user has them. A window starting at `from`
covers about `from` to `from + seconds`.

## VFX clips: pictures drawn by code

Titles, captions, lower thirds, diagrams, a transition of your own, or an effect on a shot do not need the model. Write
them as a VFX clip: a small JavaScript module on the VFX lane, above the picture, that draws each frame and is handed the
picture below it. It renders in seconds, costs no GPU time and comes out the same every time.

1. `add_effect` with `start`, `frames` and `source`. `references/api.md` has the module's contract: `frame(env, k)` draws
   frame `k` on `env.canvas`, in 2D or with WebGL and three.js, and `env.below` is the shot under it.
2. Declare what the user may want to tune (a size, a colour, a word) as `meta.controls`, and read it from `env.params`.
   The user then gets sliders, and you change a value with `update_effect` and `params`, not by rewriting the code.
3. `render_effect`, then `wait`. An open editor page does the render. A clip left `waiting for an open editor` means
   none is open: ask the user to open the editor.
4. Read the summary. An `error` on the clip names the line; fix the code and render again. A clip `out of date` needs
   another render: its code, a control, its length or a shot under it changed.

Keep `frame` a pure function of `k`: no clock, no `Math.random()` (use `env.random(seed)`). Draw from `env.width` and
`env.height`, never fixed pixels, because the preview is smaller than the render.

## Linked clips: extend, bridge, merge

These generate new footage from frames already on the timeline. They hold 39 real
frames and their sound, so the join does not show. The source item must be rendered.

| op | Use it when |
|---|---|
| `extend_after` | The shot should carry on: what happens next, same camera |
| `extend_before` | You need what led up to a shot |
| `bridge` | Two rendered shots should connect through new footage. It is placed between them and pushes the later one along |
| `merge` | Two side-by-side shots should become one continuous move. It replaces their join and keeps the length of the scene |

A take keeps its first and last 39 frames as sampled, so a linked clip holds them with no
loss. That only works when the source item is not trimmed at the side being held, and when
the source was rendered at the same quality as the linked clip. Otherwise the frames are
read from the video, which darkens and softens the join slightly: avoid trimming the end of
a clip you are about to extend. Even so, a long single shot drifts. Past about 30 seconds
the room starts to change, and the model sometimes cuts to another framing on the first new
frame. Check every join with `cuts`, and render a low-scoring extension again with another seed.

Give each one an `action` describing the new part, plus `seconds` for extend and bridge.
Linked clips render like any other. When a source changes later, the summary marks the
linked clip `source changed`, and it needs another render.

## Footage and pictures

`python scripts/h3.py import PROJECT recording.mp4` turns a real video into footage. It
is fitted to the project's size at 24 fps, gets a sound track, and is placed at the end
of the timeline. Use `--fit contain`, the default, for screen recordings, so nothing is
cut off. Use `--fit cover` to fill the frame. `--from` and `--to` take part of a long
recording. A picture becomes a still clip of `--seconds` seconds. Footage trims, moves
and joins like a render and can be extended or bridged, but it is never rendered itself.
For a file already in ComfyUI's input folder, see the import route in the reference.

Music, narration or any other sound file goes under the cut on a lane of its own: upload it
with `?sound=1` and place it with `add_sound` (see "Extra sound" in the reference). The
export mixes it in. A sound that should drive a clip's picture is a track instead, as above.

## Costs

Renders queue on the user's GPU one at a time. Measured with the models loaded:

- a draft (640x352, 4 steps) takes 10 to 15 seconds per clip;
- a final (1344x768, 6 steps) is rendered by HyperFlow H3 Flash, then enlarged and
  redrawn for its last 3 steps at 1920x1088 (1088x1920 for 9:16), with a skin LoRA on
  both passes. That is about 50 seconds of GPU per second of footage. A 5 second shot
  takes around 4.5 minutes;
- on a GPU with less memory, or when the user wants speed over size, a project can render
  smaller: `set_project` `sizes`, such as `{"final": "large"}` for finals that are not
  enlarged (the reference lists the choices). Ask before changing it: it changes what the
  cut looks like;
- `{"final": "sharp"}` gives finals of the same size as the usual ones by another route: the
  picture is sampled once at 1344x768, enlarged two times with RelayVSR and scaled down. It is
  quicker (no second pass), edges such as lashes and hair come out crisper, and skin keeps the
  smoother look of the sampled size, where the usual final adds pores. It needs RelayVSR's two
  files. Which look is better is the user's taste: render one clip both ways and show them;
- a project can render on another checkpoint family instead (`set_project` `profile`; the
  reference lists them). In tests a 2 second 9:16 final took 1.5 to 2.5 minutes on
  HyperFlow, MiniMax H3 Turbo and the 20-step MiniMax H3 alike. Keep HyperFlow unless the
  user asks for another. Do not suggest PRISM for anyone who speaks, or for a shot made of
  extensions: its mouth follows the words less exactly, and its extensions drift. It suits
  short single shots;
- one clip can render on another family than the rest (`update_clip` `profile`). Changing
  family between renders loads the new models first. Send the renders in one batch: the
  editor queues the ones that share a model file together, so each model loads once;
- any other MiniMax H3 checkpoint the user has can be used too (`set_project` `checkpoints`:
  its file, the built-in family it is closest to, and its own steps, sampler and scheduler).
  Ask the user for the settings its author recommends rather than guessing;
- a project can carry LoRAs trained on MiniMax H3 (`set_project` `loras`, each a file and a
  strength), which every new render samples with, and a clip can carry its own (`loras` in
  its fields). Add one only when the user names it or asks for its look, and render a draft
  to check its effect;
- the first render after ComfyUI starts, or after a change of profile, also loads the
  models, about 20 GB each.

Draft freely. Before a batch of finals, tell the user how many clips and roughly how long
it will take. While renders run, there is nothing to poll: `wait` returns when they finish.

`look` saves its sheets under `h3-look/<project>` in the system's temporary folder unless
you pass `--out`. Keep downloads and batch files out of the skill's own folder.

## Working alongside the user

- They may be watching or editing in the editor. Your batches merge with their edits.
  Do not reverse a change they made unless they ask.
- Every change is a revision. `history` lists them, and `restore` goes back to one
  without losing takes rendered since. Use it to recover from a bad batch.
- Never delete a project, a clip with takes the user may want, or files in the output
  folder unless the user asks.

## Notes from the user

The user can point instead of describing: in the editor they stop on a frame, pin a
spot, box or draw on the picture, and write a note. The summary lists the open notes.
`python scripts/h3.py notes PROJECT` prints each one (clip, take, frame or range, where
it points, the text) and saves its frame with the drawing burned in. Open every picture:
the drawing is part of the note. Then:

1. fix the clip the note names, for example rewrite its action and render it again;
2. look at the new take at the same frame;
3. close the note with a line on what you changed:
   `{"op": "resolve_note", "note": "n1a2b3c4", "resolution": "Low sun through the awning now."}`.
   The user sees it in the editor.

A note marked as written on an earlier take means the clip has been rendered since: check
the new take before acting on it. A note you disagree with or cannot do stays open; say why
in chat instead of resolving it. You can also leave a note for the user with `add_note`,
for example on a shot you could not judge.

## Checking the cut

`python scripts/h3.py summary PROJECT` shows the timeline with timecodes, item and clip
ids, and each clip's state. Before you export, check:

- no clip is `not rendered`, `rendering` or `failed`. A failed clip's error says why;
- no gaps, because a gap exports as black and silence. `close_gap` without a frame
  closes all of them;
- nothing is `source changed`;
- when the user has a Gemini key set, `python scripts/h3.py rate PROJECT --from A --to B` rates a stretch against a
  rubric (same person from start to end, match with the cast picture, sharp face, light, background, movement, attention, cuts). It is a second opinion on what frames
  alone cannot tell you, such as whether a stretch drags. It sends that part of the video to Google, so run it on the
  ranges the user cares about, not as a routine step;
- no extension has drifted. On some models (the turbo merges, PRISM among them) each extension comes out harsher
  than the frames it carried on from: more shine, deeper lines on a face, and by the second or third extension the
  shot no longer looks like where it began. `python scripts/h3.py detail PROJECT` marks such a clip `DRIFT` with how
  far its texture sits from its shot's first take (10% or more). Do not ship it: render those lines as a clip of their
  own from a first frame, or render the whole shot on a model that holds it (HyperFlow stayed within 5% over three
  takes). Look at a face in the first and the last take of every chained shot before calling a cut done;
- no clip is soft. `python scripts/h3.py detail PROJECT` measures how sharp each clip is at the size of the cut and
  marks the ones far under the rest `SOFT`: a draft left among finals, or footage smaller than the cut. Render that
  clip as a final. For a finished file of any origin, `python detail.py FILE.mp4`, run from the H3 Editor pack's
  folder, lists the stretches that are much softer than the rest, with their times;
- for a 4K delivery, upscale last: `python scripts/h3.py render PROJECT CLIP --upscale` enlarges a clip's final two
  times (1920 x 1088 becomes 3840 x 2176) as a new take, in seconds. `--upscaler relayvsr` is fast and sharpens what is there; `--upscaler seedvr2`
  is many times slower and redraws fine texture, which looks more like a larger picture and can change a face. Ask the
  user which they want and show them one clip first. With RelayVSR, `--detail` (0 to 1, default 0.5) sets how much of
  the detail it adds is kept: look at a face after the first one, since high values add pores and shine
  that can make a person look older, and lower it if so. Do it once the cut is settled and every clip is
  a final: a clip that extends an upscaled clip would be generated from the enlarged picture. Upscale every clip of
  the cut: one left at the old size is enlarged plainly at export and looks softer than its neighbours (about 70% of
  their detail in `detail`, which is not low enough to be marked `SOFT`);
- no take changes picture inside, and no VFX clip pops in or out. The same `cuts` command lists both after the cuts:
  `inside: fade` or `inside: cut` means a generated take dissolves or cuts to another picture partway (the model
  sometimes ends a take that way): trim the clip to end before that frame, or render it again with another seed. A
  low score `into the start` or `out of the end` of a VFX clip means its first or last frame does not match the
  picture under it: render the VFX clip again if the clip under it changed, or move its edge to a cut;
- no take stops moving. The same list marks `inside: frozen` (the same frame held for a few frames or more: a stalled
  take) and `inside: still` (almost nothing moves for a second and a half or more), each with how long. A frozen
  stretch always needs fixing: trim the clip to end before it, or render it again with another seed. A still one
  is a mistake when the shot should move, such as someone talking; a shot meant to be still can stay. Takes made
  by the editor do not do this in normal use, so a mark here is worth a look. For a finished file of any origin,
  `python cuts.py --still FILE.mp4`, run from the H3 Editor pack's folder, lists the stretches that do not move;
  run it on your own edits too, since holding a clip on its last frame to fill a gap shows as frozen;
- no cut scores low. `python scripts/h3.py cuts PROJECT` scores each place where two
  rendered clips touch, 0 to 100, from the pictures. A `jump` (the same shot, the
  subject pops) under 60 needs fixing: extend the first clip and let the second
  continue from its last frame, or give the second a different framing;
- the total length matches the brief.

## Reporting

When you hand back, tell the user:

- the project name to open in H3 Editor, and where the export landed. Exports go to
  ComfyUI's output folder under `h3take_projects/<project>/exports/`;
- the shots, in order, with what each shows and whether it is a draft or a final;
- what you could not judge, for example how a line sounds. You can see frames but
  cannot hear the audio, so say so, and ask the user to listen;
- what is left, such as finals to render or shots to redo.
