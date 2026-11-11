# H3 Editor API reference

All routes are under `http://<comfyui>/h3take/api`. `{name}` is the project's folder
name, as `create` prints it: letters, numbers, `_` and `-`. Bodies and replies are JSON,
except the two uploads.

## Contents

- Routes
- The project model
- Operations
- Replies and errors
- The summary
- Lengths and timing

## Routes

| Method and path | Body | Reply |
|---|---|---|
| `POST /project/{name}/rename` | `{"name"}` | The name the project shows. Its folder, which is what `{name}` in every path means, stays as it was made. Returns `{"folder", "project"}` |
| `POST /project/{name}/duplicate` | optional `{"name"}` | A copy in a folder of its own, with its clips, takes and pictures, and no renders pending. The default name is the original's with `copy` after it. Returns `{"folder", "project"}`; 409 when the name is taken |
| `DELETE /project/{name}` | | Moves the project's folder to `h3take_projects/.trash`; nothing is erased. Refused with 409 while it is rendering. Do this only when the user asks. Returns `{"deleted", "trash"}` |
| `GET /projects` | | `{"projects": [folder, ...], "titles": {folder: name, ...}, "profiles": [{"id", "title", "hint", "available", "missing"}, ...], "finals": [note, ...], "loras": [file, ...], "realism": {"file", "strength", "installed", "range", "lora_range", "limit"}, "checkpoints": [file, ...], "samplers": [name, ...], "schedulers": [name, ...]}`. Each profile also carries its `steps`, `sampler` and `scheduler`. `finals` says what finals leave out because a file is missing, `loras` lists every file in `models/loras`, and `realism` describes the Realism Slider that finals run |
| `POST /projects` | `{"name"}` | `{"folder", "project"}`. Opens it if it exists |
| `GET /project/{name}` | | The whole project |
| `GET /project/{name}/summary` | | The summary, below |
| `GET /project/{name}/cuts` | | `{"cuts": [{"before", "after", "at", "kind", "score", "notes", ...}], "worst"}`: a score from 0 to 100 for each place where two rendered clips touch. `kind` is `continuous`, `new shot` or `jump`; a `jump` (the same shot, the subject pops) scores under 60 and wants an extension from the first clip's last frame or a different framing. Measured from the pictures, so it never changes for the same takes and trims |
| `POST /project/{name}/ops` | `{"ops": [...], "revision"?}` | `{"results", "revision", "summary"}` |
| `GET /project/{name}/wait?timeout=600` | | The summary, plus `timed_out`, once nothing renders. `after=N` waits for a revision past N instead. Timeout is at most 1800 |
| `POST /project/{name}/render` | `{"clip", "quality", "seed"?}` | `{"job", "revision", "project"}` |
| `POST /project/{name}/cancel` | `{"job"}` or `{"clip"}` | `{"cancelled", ...}` |
| `POST /project/{name}/import` | multipart `file` with fields, or JSON `{"source": {"folder": "input" or "output", "subfolder"?, "file"}, ...}` | `{"clip", "item", "frames", "revision", "project"}` |
| `POST /project/{name}/asset` | multipart `file`: PNG, JPEG or WebP, or WAV, MP3, FLAC, M4A, OGG, OPUS, AAC | `{"file", "subfolder"}`, an asset. With `?sound=1` on a sound, also `seconds`, its length, and `peaks`, as `add_sound` needs |
| `POST /project/{name}/grab` | `{"file", "subfolder", "frame"}`, a take's file and a frame of it | A picture asset |
| `POST /project/{name}/grab_sound` | `{"file", "subfolder", "first", "last"}` in take frames | A WAV asset, 30 s at most |
| `POST /project/{name}/export` | `{}`, or any of `scale` (1, 0.75 or 0.5 of the rendered size), `quality` (`best`, `high`, the default, or `small`) and `name` (the file's name instead of the project's) | `{"filename", "subfolder", "bytes", "plan"}` in ComfyUI's output folder. With `"plan": true` nothing is made: `{"plan": {"width", "height", "frames", "seconds", "clips", "drafts", "missing", "gaps"}}` says what an export would hold |
| `GET /project/{name}/history` | | `{"revisions": [{"revision", "time", "source", "clips", "frames"}]}`, newest first |
| `POST /project/{name}/restore` | `{"revision"}` | `{"revision", "summary"}` |
| `POST /sheet` | `{"filename", "subfolder", "frames"?, "first"?, "last"?, "count"?, "tile_width"?}` | `{"sheet", "picked", "frames"}`: a grid of frames saved beside the clip |

Import fields: `name`, `fit` (`contain` or `cover`), `seconds` for a picture (0.5 to
60, default 4), `start_seconds` and `end_seconds` to take part of a video, `start` as a
timeline frame, `place` (`false` keeps it off the timeline), and `bin`, the id of a bin
to file it in. Videos can be MP4, MOV,
WebM, MKV, M4V or AVI. An upload is one file, 4 GiB at most. Put anything larger in
ComfyUI's input folder and import it by name.

Download any output file with ComfyUI's own
`GET /view?filename=...&subfolder=...&type=output`.

`/sheet` picks `count` frames (default 12, at most 24) evenly from take frames `first` up
to `last` (default: all of it), each `tile_width` pixels wide (default 384). `frames` is
the take's frame count, which saves a probe. `picked` lists the frames shown, in reading
order.

## The project model

A **clip** is a shot: its text fields, its `takes` (renders, oldest first), and `active`,
the take in use. A **timeline item** places part of a clip's active take:

- `start` is the timeline frame where it begins;
- `in` and `out` are frames of the take: it shows frames `in` up to, but not including, `out`;
- `speed` runs from 0.25 to 4. The item fills `(out - in) / speed` frames;
- `transition` is `{"type": "dissolve" | "black" | "wipe", "frames"}`. It overlaps the
  item with the one before it.

One clip can sit on the timeline more than once, for example after a split. Clips never
overlap, except inside a transition. When an item grows into the next one, the next one
moves along. An **asset** is `{"file", "subfolder"}` in ComfyUI's output folder.

A clip with `"kind": "footage"` was imported. It has one take and never renders.

A **bin** is a folder in the editor's Clips panel: `{"id", "name"}` in the project's
`bins`. A clip's `bin` field names the bin it is filed in. Clips without one sit loose
below the bins. Bins only sort the panel, and never change a render or the timeline.

## Operations

Every operation is `{"op": ..., ...}`. Add `"as": "label"` to name its result. Later
operations in the batch can then write `"$label"`, and the clip or item is picked to fit
the field. `"$label.item"` and `"$label.clip"` choose explicitly. A clip's `cast` list can name cast members
added earlier in the batch the same way: `"cast": ["$her"]`, and its `bin` a bin: `"bin": "$broll"`.

### Clips

`add_clip`: `fields`, and beside them optional `start` (a timeline frame; the default is
the end) and `place` (default true). Returns `{"clip", "item"}`.

| field | |
|---|---|
| `name` | Shown in the editor |
| `action` | The shot, in English. A clip needs it to render |
| `camera` | Framing and movement |
| `speaker`, `dialogue` | Who speaks, and the exact words. `@Name` in `speaker` uses that cast member, and their voice if they have one |
| `language` | Language of this clip's lines. The default is the project's |
| `sound`, `music` | Soundscape, and music under the shot |
| `seconds` | 0.5 to 60. Snapped to a length H3 can render, see below |
| `cast` | Ids of the pictured cast members in this shot |
| `seed` | A whole number repeats a result; `null` gives a new one at each render |
| `first`, `last` | Picture assets the shot starts or ends on exactly. `null` removes one |
| `track` | A sound asset plus `from`, the second of it at the clip's first frame. The picture is generated to that slice, and the take keeps the exact sound. `null` removes it |
| `bin` | The id of the bin the clip is filed in. `null` takes it out |
| `loras` | LoRAs only this clip samples with, as `set_project` `loras` describes. `null` clears them |
| `profile` | The checkpoint family this clip renders on, one of the ids `set_project` takes. `null`, the default, is the project's. A take records the one that made it in `models.profile` |

`update_clip`: `clip`, `fields`. Changing `seconds` on a clip with no render resizes its
items, and later clips move with them. `remove_clip`: `clip`, together with all its
timeline items. Its render files stay on disk, so older revisions still play. A render still
running for it is thrown away when it finishes, so cancel it first. `place_clip`: `clip`,
optional `start`. Returns a new `item`.

### Extra sound

Music, narration or effects under the cut, on up to 4 lanes of their own beside the clips' own
sound. Upload the file with `POST /project/{name}/asset?sound=1`, which also returns its
`seconds`.

| op | Takes | Notes |
|---|---|---|
| `add_sound` | `asset`: `{"file", "subfolder"}`, `seconds`, optional `fields` | Returns `{"sound"}`. It starts at frame 0 on lane 0 and plays its whole length unless `fields` say otherwise |
| `update_sound` | `sound`, `fields` | |
| `remove_sound` | `sound` | The file stays in the project's assets |

`fields`: `name`; `start`, the timeline frame it begins at; `offset`, the second of the file it
starts from; `frames`, how long it plays in timeline frames, never past the end of the file;
`gain`, from 0 to 2, where 1 leaves it as it is; `lane`, 0 to 3. The summary lists them as
`sounds`. An export mixes each one in at its place and gain, and stops it where the cut ends.

### Bins

| op | Takes | Notes |
|---|---|---|
| `add_bin` | optional `fields`: `name` | The default name is `Bin 1`, `Bin 2` and so on. Returns `{"bin"}` |
| `update_bin` | `bin`, `fields`: `name` | An empty name keeps the old one |
| `remove_bin` | `bin` | Its clips stay, loose |

A clip extended, bridged or merged from a filed clip goes in the same bin.

### Timeline

| op | Takes | Notes |
|---|---|---|
| `update_item` | `item`, `fields`: `start`, `in`, `out`, `speed`, `transition`; optional `ripple` | `transition` can be `"none"`. Speed and transition always move the clips after it. With `ripple`, trimming does too |
| `remove_item` | `item`, optional `ripple` | Ripple closes the space it leaves |
| `split_item` | `item`, `frame` (a timeline frame inside it) | Returns `right`, the new second half |
| `close_gap` | optional `frame` | No frame closes every gap. Returns `removed` |
| `choose_take` | `clip`, `take` | Items keep their trim, cut short if the take is shorter |

To slip a clip, send `update_item` with `in` and `out` moved by the same number of frames:
the item keeps its `start` and length and shows another part of its take. Switching a
clip's take, or rendering it again, also keeps the item where it is; a shorter take pulls
`out` in.

### VFX clips: drawn by code

`add_effect`: `fields` with any of `name`, `start` (timeline frame), `frames`, `source` (the module's text), `params`
(control values) and `inputs` (pictures the code may draw: `{"logo": {"file", "subfolder"}}`, each an uploaded asset or
any picture in the output folder, such as a take's filmstrip; the code reads them as `env.inputs.logo`, an ImageBitmap). Returns `{"effect"}`. `update_effect`: `effect`, `fields`. `remove_effect`: `effect`. They sit on one
lane above the picture and never overlap each other. The summary lists them under `effects` with a `state`: `not rendered`,
`rendered` or `out of date`.

The module is JavaScript:

```js
export const meta = { controls: { size: { type: 'number', min: 0, max: 1, default: 0.5 } } };   // optional
export async function setup(env) {}                                                             // optional
export function frame(env, k) {}     // draw frame k; depend only on k, env.params and env.below
```

`env` holds `canvas` (take a `2d` or a WebGL context from it), `width`, `height`, `fps`, `frames`, `params`, `inputs`, `below` (the
picture under this frame as an ImageBitmap, or null where the lane below is empty) and `random(seed)`. A module that mentions
`THREE` gets three.js as `env.THREE`, and one that mentions `belowTexture` gets the picture below as a ready `THREE.Texture`.
Control types: `number`, `int`, `bool`, `enum` (with `options`), `color`, `text`. It runs in a sandbox: no network, no
imports, no access to the page.

`render_effect`: `effect`. A render bakes the clip over the picture below it into a take. The code runs in a browser, so
the request waits until an editor page is open on the project, and that page does it, usually in seconds. The summary
shows the clip as `waiting for an open editor to render it`, then `rendering`, then `rendered`; `wait` returns once no VFX
render is waiting either. If none is open, ask the user to open the editor, or open it yourself in a browser you control.
A failure is recorded on the clip as `error`, with the line of the module it came from. An export leaves out a VFX clip
with no take, and its plan counts `effects_missing` and `effects_stale`.

### Linked clips

`extend_after`, `extend_before`, `bridge`, `merge`: `item`, optional `seconds`, `action`,
and `fields` (any clip field). Each returns `{"clip", "item"}`, and the new clip still
needs a render.

- The source item must be rendered, with at least 39 frames on the timeline.
- `bridge` and `merge` join the item to the next one on the timeline, which must be rendered too.
- `merge` also needs the two items side by side at 100% speed, the first with at least
  71 frames and the second with at least 70. It trims 32 frames from the end of the
  first and 31 from the start of the second, and fills that 63-frame span with the new clip.
- A linked clip holds 39 real frames from each side it joins. The timeline shows only
  the new frames, so the hold is never seen twice.
- A first or last frame on a held side is ignored, because that side already has real
  frames.
- A linked clip keeps its source clip's `cast`.
- When the source clip holds a `track`, the linked clip holds the next slice of it.
  For `extend_after`, `bridge` and `merge` that is
  `from = source from + (source item out - 39) / 24`. For `extend_before` it is
  `from = source from + (source item in - (new length - 39)) / 24`, which is refused if
  it would fall before the start of the sound. The source item must be at 100% speed.
  A bridge lays the same track under the frames it holds from the next clip, so that
  clip should continue the track at that point.

### Project and cast

`set_project`: `fields`: `style` (applies to every clip), `format` (`16:9`, `9:16` or
`1:1`), `language` (for example `English` or `Portuguese`), and `profile`, the checkpoint
family new renders use (`hyperflow`, the default, `h3_turbo`, `h3`, `singularity`, `dasiwa`
`dasiwa_turbo`, `dasiwa_v5` or `eros`). `GET /projects` lists the profiles with whether each is installed;
pick only an `available` one. A new project starts on the first installed profile, and a
project that names none renders on it. The format sets new renders and imports, and export fits
every clip to it.

`set_project` also takes `sizes`, for a GPU with less memory or for speed:
`{"draft": "medium", "final": "large"}`. A draft is `small` (the usual) or `medium`. A final is
`full` (the usual: sampled large, then enlarged), `large`, `medium` or `small`, and only `full`
is enlarged. Leave a quality out, or send `null`, for the usual size. `GET /projects` lists
every choice per format under `sizes`, with `width`, `height` and the size it comes out at,
`out`. The summary's `sizes` says what the project renders at.

`set_project` also takes `loras`, the LoRAs every new render of the project samples with,
drafts and finals alike, and `realism`:

```json
{"op": "set_project", "fields": {
  "loras": [{"file": "film_look.safetensors", "strength": 0.8},
            {"file": "styles/grain.safetensors", "strength": -0.5, "on": false}],
  "realism": 1.0}}
```

- `loras` replaces the whole list: up to 8, stacked in the order given. `file` is a name
  from `loras` in `GET /projects`, `strength` runs from -2 to 2 (1 when left out), and
  `"on": false` keeps a LoRA in the list without rendering with it. Send `[]` to clear it.
  Use LoRAs trained on MiniMax H3: others have no layers to land on. A LoRA whose file is
  not installed is left out of renders.
- `start_sigma` on a LoRA, from 1 down to 0, is where in the schedule it joins. 1, the
  default, runs it on every step. A lower value leaves the first steps alone, so framing
  and pose stay as they would be without it; 0.75 is a good start for a look or skin LoRA.
- A clip takes `loras` too, in `add_clip` and `update_clip` `fields`, in the same shape: the
  LoRAs only that clip samples with, after the project's. A row for a file the project also
  lists takes that row's place for this clip, so `{"file": ..., "strength": 0.4}` changes its
  strength there and `{"file": ..., "on": false}` leaves it out. `null` clears the clip's list.
- `realism` is how strong the Realism Slider is on finals, from 0 to 3. It is 1.2 unless
  set, 0 turns it off, and `null` goes back to 1.2. Drafts never run it.

Takes already rendered do not change: render a clip again to see a new strength.

`set_project` also takes `checkpoints`, the project's own checkpoints: any MiniMax H3 file
in `models/diffusion_models`, rendered on a built-in family's settings with whatever it
changes.

```json
{"op": "set_project", "fields": {
  "checkpoints": [{"id": "mine", "title": "My fine-tune", "unet": "my_h3_ref2va.safetensors",
                   "base": "h3", "steps": {"draft": 8, "final": 16},
                   "sampler": "euler", "scheduler": "beta"}],
  "profile": "mine"}}
```

- `unet` is the ref2va file, a name from `checkpoints` in `GET /projects`. `unet_text` is
  an fl2va file for clips that are only a prompt; without it they render on `unet`.
- `base` is the built-in profile whose settings it starts from (`h3` when left out). It
  keeps that profile's LoRA, such as the Turbo LoRA of `h3_turbo`, and the way its finals
  are refined. `"lora": false` leaves that LoRA out, for a checkpoint with the turbo
  already merged in.
- `steps`, `sampler` and `scheduler` are optional and replace the base's. `GET /projects`
  lists the `samplers` and `schedulers` ComfyUI has. A HyperFlow file keeps its own
  schedule, so its `scheduler` is ignored.
- `id` is what `profile` names, on the project or on a clip. Give your own, or one is made.
  The list replaces the whole list, up to 12; a clip or project that names a checkpoint
  no longer listed renders on the project's model, or the first one installed.

`set_project` also takes `text_encoder`, `vae` and `audio_vae`: a file name as ComfyUI lists
it, or `null`. H3's text encoder and VAEs are published in several precisions (nvfp4, int8,
fp16, bf16), and a render takes the first one installed unless the project names one. A
named file that is not installed gives way to an installed one. `GET /projects` gives
`encoders` as `{"text_encoders": [file, ...], "vaes": [file, ...], "auto": {"text_encoder",
"vae", "audio_vae"}, "missing": [note, ...]}`: `auto` is what a render takes by itself, and
`missing` says which of the three no file was found for. A take records `text_encoder` and
`vae` in `models`.

Checkpoints the user saved as presets are shared by every project. `GET /presets` returns
`{"checkpoints": [...]}` in the same shape, and `GET /projects` carries them as `presets`.
A project renders on its own copy, so to use a preset, send its row in `set_project`
`checkpoints` with the same `id`. `PUT /presets` with `{"checkpoints": [...]}` replaces the
shared list, up to 64; change it only when the user asks.

`add_cast`: `fields`: `name`, `description`, and a picture asset (`file`, `subfolder`), a
`voice` asset, or both. Returns `{"cast"}`. The picture's keys sit directly in `fields`,
while the voice is its own object:

```json
{"op": "add_cast", "as": "ana", "fields": {"name": "Ana", "description": "a woman in her thirties, short dark hair, white shirt",
  "file": "3f1c.png", "subfolder": "h3take_projects/Teaser/assets",
  "voice": {"file": "9a0b.wav", "subfolder": "h3take_projects/Teaser/assets"}}}
```

A cast member can have more pictures beside the main one: other angles, other light, the
whole figure. Send them as `pictures`, a list of picture assets, in `add_cast` or
`update_cast`; the list replaces the one before, and `[]` clears it. A member takes up to 8
pictures, the main one included. H3 takes 9 pictures in one shot, pinned first and last
frames included: every member in the shot brings its main picture, and the room left is
shared out one more picture per member in turn. The summary gives each member's count as
`pictures`.

A cast member can also carry a RefMod, a reference saved with the ComfyUI-MiniMaxH3Mod
custom node pack: `"refmod": {"file": "hero_v1", "strength": 1.0}` in `add_cast` or
`update_cast`, and `null` removes it. `file` is a name from `refmods.files` in
`GET /projects`, and `strength` runs from 0 to 1. It works beside pictures or in place of
them: a member that is only a RefMod renders on the prompt-only model, with `@Name` written
out as the name and description. `GET /projects` gives `refmods` as
`{"installed", "files"}`; when the pack is not installed, or the file is missing, renders
leave the RefMod out. A take records the ones it used in `models.refmods`.

`make_refmod`: `cast`, optional `name` (the default is the member's name, in lower case with
underscores), optional `mode`: `compressed`, the default, which renders about as fast as no
reference, or `whole`, which keeps more of the pictures and makes drafts several times
slower. It queues the pack's own extract node on all of the member's pictures and sets the
member's `refmod` to the new file. Returns `{"cast", "refmod", "job"}`. The file takes a
few seconds to a minute to appear in `refmods.files`; renders leave it out until then. It is
refused when the pack is not installed, when the member has no picture, and when a RefMod of
that name already exists and is not this member's, unless `"replace": true` is sent.

`update_cast`: `cast`, `fields`: `name`,
`description`, `voice` (`null` removes it, unless it is all they have). `remove_cast`:
`cast`, which also takes them out of every clip.

The description is how H3 knows them. Describe a face and clothes for a picture, and how
they sound for a voice. A cast member without a picture is never placed in the shot, and
their description stands in for them in the prompt. Write `"@Name, off-screen"` as the
speaker when they should not be seen.

### Renders

`render`: `clip`, optional `quality` (`draft`, the default, or `final`), optional `seed`.
Returns `{"clip", "job", "take"}`. It is refused for footage, for a clip with no action, and for a clip already
rendering. Renders run after the edits in the same batch. `cancel`: `clip` or `job`.

The renders of one batch are queued grouped by the model file each one loads, so a batch that
mixes models (prompt-only clips and clips with pictures, or clips on different profiles) loads
each model once. Groups keep the order their first render was asked for, and a batch starts
with the model the queue already ends on. So send the renders you want together in one batch
rather than one call each.

| quality | 16:9 | 9:16 | 1:1 |
|---|---|---|---|
| draft | 640x352 | 352x640 | 480x480 |
| final | 1344x768 | 768x1344 | 1024x1024 |
| `medium`, when the project asks for it | 960x544 | 544x960 | 768x768 |

Steps depend on the project's profile: HyperFlow takes 4 for a draft and 6 for a final,
MiniMax H3 Turbo 6 and 6, the 20-step families (`h3`, `singularity`, `dasiwa`) 10 and 20,
and DaSiWa Turbo 8 and 8. A final's picture is then enlarged and its last steps redrawn at
the larger size, so its take is 1920x1088 for 16:9, 1088x1920 for 9:16 and 1440x1440 for 1:1.
The sound is the first pass's. A take records its picture size in `size` and what made it in
`models`: `profile`, `unet` and `checkpoint`, a turbo family's `profile_lora` and
`profile_lora_strength`, for a final `upscaler`, `lora` and `lora_strength`, and the
project's own LoRAs as `loras`: `[{"file", "strength"}, ...]`. Finals from before this record
`upscale` instead of `size`.

## Replies and errors

Each entry of `results` repeats its operation's `op` and `as`, then what it made.

Errors are JSON: `{"error": "..."}`. A refused batch adds `index`, counting from zero, and
changes nothing.

- **400**: a bad operation or field. The message says which and why.
- **404**: no such project, revision or job.
- **409**: you sent `revision` and the project has moved on. Read it again and retry.
- **502**: ComfyUI refused a render, usually a missing model. The edits and any renders
  before it were kept. `results` shows which.

A render that fails while running shows `failed` in the summary, with `error` on the
clip. The next render clears it.

When a clip holds a track, the finished take's sound is replaced by that slice of the
track, trimmed from the decoded original and encoded as AAC at 256 kb/s. The picture is
left as rendered. In the summary such a clip shows `"track": {"file", "from"}`.

## The summary

```json
{
 "name": "Morning teaser", "revision": 12, "style": "...", "format": "16:9",
 "language": "Portuguese", "frames": 508, "length": "00:21:04", "fps": 24,
 "cast": [{"id": "p1a2b3c4", "name": "Ana", "description": "...", "picture": true, "voice": true}],
 "bins": [{"id": "b1", "name": "Selects", "clips": 3}],
 "clips": [{"id": "c1", "name": "Desk", "state": "draft", "seconds": 4, "action": "...",
            "takes": 2, "take": "t9", "frames": 90, "take_seed": 123, "seed": 7, "quality": "draft",
            "link": "extend_after", "footage": true, "bin": "b1", "error": "..."}],
 "timeline": [{"item": "i1", "clip": "c1", "name": "Desk", "start": 0, "end": 90,
               "at": "00:00:00", "in": 0, "out": 90, "speed": 2, "transition": {...}}],
 "gaps": [{"start": 90, "end": 96}],
 "rendering": [{"job": "...", "clip": "c2", "quality": "draft"}]
}
```

`state` is one of `not rendered`, `rendering`, `failed`, `source changed`, `draft`,
`final`, or `footage`. `take_seed` is the seed the active take was made with. `seed`
appears only when the clip itself has one set, which repeats a result at every render. Timecodes are minutes:seconds:frames at 24 fps. Optional keys
appear only when they apply.

## Lengths and timing

Everything runs at 24 frames per second. H3 renders set lengths, so `seconds` snaps:

- an ordinary clip renders 5 + 17k frames, the nearest to `seconds` x 24. That is
  steps of about 0.7 s, and 22 frames at the least;
- `extend_after` and `extend_before` render 39 + 51k frames and show the 51k new
  ones, in steps of about 2.1 s;
- `bridge` renders 39 + 51k frames with k of 2 or more, and shows all but the 39 held
  at each end. A merge is the shortest bridge: 141 frames, 63 of them shown.

The timeline reserves that length before the render, so a finished take lands where the
timeline expects it.
