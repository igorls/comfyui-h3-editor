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
| `GET /project/{name}/archive` | `?keep=all\|cut\|text` | The project as one zip, `<name>.h3project.zip`: `project.json`, `manifest.json` (release, level, models it needs, files it points at in other projects) and the files under `files/`. `all` holds every take and export, `cut` the active takes of the clips on the timeline and of the VFX clips with footage, pictures and sounds, `text` no renders. The takes a level leaves out are left out of its `project.json` too. Without `keep` it answers `{"name", "release", "levels": {level: {"bytes", "files", "outside"}}}` |
| `POST /projects/open` | the zip as the body, `Content-Type: application/zip` | Opens a zip made by `archive` as a new project, under its old name or that name with a number. The body is written to disk as it arrives, so ComfyUI's upload size limit does not apply. Returns `{"folder", "name", "release", "keep", "missing_loras", "patches_off", "outside", "files"}`; `patches_off` names the model patches the zip had on, which are opened switched off; 400 with a reason for a zip that is not a project, holds unsafe paths or links, or comes from a newer release |
| `POST /project/{name}/duplicate` | optional `{"name"}` | A copy in a folder of its own, with its clips, takes and pictures, and no renders pending. The default name is the original's with `copy` after it. Returns `{"folder", "project"}`; 409 when the name is taken |
| `DELETE /project/{name}` | | Moves the project's folder to `h3take_projects/.trash`; nothing is erased. Refused with 409 while it is rendering. Do this only when the user asks. Returns `{"deleted", "trash"}` |
| `GET /projects` | | `{"projects": [folder, ...], "titles": {folder: name, ...}, "cards": {folder: {"name", "format", "clips", "frames", "picture", "rendering", "changed"}, ...}, "profiles": [{"id", "title", "hint", "available", "missing"}, ...], "finals": [note, ...], "loras": [file, ...], "patches": [{"node", "title", "category", "pack", "model", "controls"}, ...], "realism": {"file", "strength", "installed", "range", "lora_range", "limit"}, "checkpoints": [file, ...], "samplers": [name, ...], "schedulers": [name, ...]}`. Each profile also carries its `steps`, `sampler` and `scheduler`. `finals` says what finals leave out because a file is missing, `loras` lists every file in `models/loras`, `patches` lists the installed nodes a project can put on the model (see `set_project` `patches`), and `realism` describes the Realism Slider that finals run. `cards` is what a project list shows: `clips` and `frames` count what is on the timeline, `picture` is the `{"file", "subfolder"}` of the first rendered clip's poster or null, `rendering` is how many renders are queued, and `changed` is when the project file last changed, in seconds since 1970 |
| `POST /projects` | `{"name"}` | `{"folder", "project"}`. Opens it if it exists |
| `GET /models/setup` | | What the default profile still needs: `{"profile", "files": [{"id", "title", "file", "folder", "folder_path", "bytes", "for", "installed", "found", "partial", "license": {"name", "url"}, "source" or "page"}], "missing": [id, ...], "bytes", "manual": [id, ...], "ready", "short", "job"}`. `bytes` is the size of what is missing, `ready` says whether drafts render, `manual` names the files only the user can fetch (the Realism Slider on Civitai, behind a sign-in; `page` links it), `short` lists disks without room, and `job` is the download in progress or last run: `{"state", "error", "files": [{"id", "file", "bytes", "done", "state", "error"}], "bytes", "done", "rate"}`, null before the first |
| `POST /models/setup/download` | none | Starts downloading the missing files into ComfyUI's models folders, from the server's own list of Hugging Face files; returns the same as `GET /models/setup`. 409 while one runs, 507 when a disk is short. It is tens of GB: ask the user first |
| `POST /models/setup/cancel` | none | Stops it. What came is kept, and the next download carries on from it |
| `GET /project/{name}` | | The whole project |
| `GET /project/{name}/summary` | | The summary, below |
| `POST /project/{name}/ops` with `{"op": "render", "clip", "quality": "upscale", "upscaler"?, "detail"?}` | | Doubles the active final or imported footage into a new take. Its `upscaled` points to the source take. `GET /projects` lists each upscaler and its availability. Omitting `upscaler` picks the first installed of `relayvsr` and `seedvr2`. Both accept up to 240 frames. Experimental `vosr2` requires explicit selection and streams one frame at a time. RelayVSR alone accepts `detail` from 0 (plain enlargement) to 1 (sharpest), default 0.5. Drafts and already-upscaled takes cannot be upscaled. |
| `GET /project/{name}/detail` | | `{"level", "size", "clips": [{"item", "at", "fine", "part", "soft", "drift"?}]}`: how sharp each rendered clip on the timeline is at the size of the cut. A clip that extends another also has `drift`: `{"root", "change", "drifted", "limit"}`, its picture's texture against the first take of its shot (`root`, a timeline item), `drifted` when it is 10% or more away. `fine` is the measure, `part` is it against the cut's `level`, and `soft` is true under half: a draft among finals, or footage smaller than the cut. Measured from the pictures |
| `GET /project/{name}/cuts` | | `{"cuts": [{"before", "after", "at", "kind", "score", "notes", ...}], "effects": [{"effect", "edge", "at", "kind", "score", "notes"}], "inside": [{"item", "frame", "at", "kind", "score", "notes"}], "worst"}`. `effects` scores the cut into (`edge` `start`) and out of (`end`) each rendered VFX clip against the picture under it. Each cut also has `sound`: `{"kind", "score", "notes", "step", "colour", "click"}`, what the cut sounds like (`continuous`, `new sound`, `level jump`, `dropout` or `silent`; `step` is the change in loudness in dB, negative for quieter after). Each cut has `movement` too: `{"kind", "score", "notes", "before", "after"}`, how the camera moves on each side (`before` and `after` are pan x, pan y in widths a second and zoom in parts a second) and whether the move `carries on`, `stops`, `starts`, `reverses` or `changes`; `score` is null at a cut to a new shot. And `parts`: `{"kind", "score", "notes", "stops", "starts"}`, whether what moves inside the frame, beyond the camera's move, carries across the cut: `stops` and `starts` are the shares of the frame that moved on one side and are still on the other. `sound_inside` lists takes whose sound is cut off (`chop`), drops out, or ends far louder or quieter than it began (`swell`, `fade`). `arrivals` has one row for each bridge that ends on a pinned frame: `kind` `arrives` or `short of its pin`, with `level`, how much darker or lighter than its pin the take ends, in percent. `lips` has one row for each generated clip with words or a track whose lips could be measured (see lips.py): `kind` `in sync`, `early`, `late` or `no match`, `offset` in frames (negative when the lips come before the sound), `conf` (how clearly they match: under 1.5 is no match, 2 to 5 is usual for a good take), `score` and `notes`; `lips_missing` names the files the measure still needs in `models/syncnet`, empty when it runs. `scene` has one row for each generated clip whose background could be seen (see scene.py): `kind` `holds` or `changes`, `share` (the part of the background, the person left out, that is no longer what it was; 0.2 and more is `changes`), `frame` in the take and `at` on the timeline where it is worst, `since_shot` (true for an extension, held against the first frame its shot shows as well as against itself), `score` and `notes`; `scene_missing` names the file it still needs in `models/matting`. `scene_off` says why the check is off when the file is there but cannot be run (null when it runs). `worst` counts all of these. `inside` lists generated takes that change picture within the part their clip shows: `kind` `cut` or `fade`, `frame` in the take, `at` on the timeline; trim the clip before it or render again. `cuts`: a score from 0 to 100 for each place where two rendered clips touch. `kind` is `continuous`, `new shot` or `jump`; a `jump` (the same shot, the subject pops) scores under 60 and wants an extension from the first clip's last frame or a different framing. Measured from the pictures, so it never changes for the same takes and trims |
| `POST /project/{name}/rate` | `{"from", "to", "rubric"?, "resolution"?, "note"?}` | `{"rating": {"from", "to", "rows", "tokens"}}`: that range of the cut (timeline frames, up to 180 s) rated by Google's Gemini Decision model. Each row has `title`, `score` (a level, 0 is worst), `of` (the top level), `part` (0 to 1), `level` (its description) and `confidence`. A criterion with two levels is a check: `of` is 1 and `part` is how likely the good end is. The default rubric checks the same person from start to end, a match with the cast picture (sent along unless `"reference": false`), a sharp face, natural light, a stable background and natural movement, rates attention on five levels, and adds a `cuts` row measured by the editor itself (`measured: true`, no confidence), since the model samples about a frame a second and misses jump cuts; `rubric` replaces it with a list of `{key, title?, instructions, levels}`, levels worst first. Needs the user's Gemini key, which only they can set (409 if none); it sends the video to Google, so use it when the user wants a rating, not on every edit |
| `POST /project/{name}/ops` | `{"ops": [...], "revision"?}` | `{"results", "revision", "summary"}` |
| `GET /project/{name}/wait?timeout=600` | | The summary, plus `timed_out`, once nothing renders. `after=N` waits for a revision past N instead. Timeout is at most 1800 |
| `POST /project/{name}/render` | `{"clip", "quality", "seed"?}` | `{"job", "revision", "project"}` |
| `POST /project/{name}/cancel` | `{"job"}` or `{"clip"}` | `{"cancelled", ...}` |
| `POST /project/{name}/import` | multipart `file` with fields, or JSON `{"source": {"folder": "input" or "output", "subfolder"?, "file"}, ...}` | `{"clip", "item", "frames", "revision", "project"}` |
| `POST /project/{name}/asset` | multipart `file`: PNG, JPEG or WebP, or WAV, MP3, FLAC, M4A, OGG, OPUS, AAC | `{"file", "subfolder"}`, an asset. With `?sound=1` on a sound, also `seconds`, its length, and `peaks`, as `add_sound` needs |
| `POST /project/{name}/grab` | `{"file", "subfolder", "frame"}`, a take's file and a frame of it | A picture asset |
| `POST /project/{name}/grab_sound` | `{"file", "subfolder", "first", "last"}` in take frames | A WAV asset, 30 s at most |
| `POST /project/{name}/export` | `{}`, or any of `range` (timeline frames `{"from":24,"to":48}`, exclusive end), `scale` (1, 0.75 or 0.5 of the rendered size), `quality` (`best`, `high`, the default, or `small`) and `name` (the file's name instead of the project's) | `{"filename", "subfolder", "bytes", "plan"}` in ComfyUI's output folder. With `"plan": true` nothing is made: `{"plan": {"width", "height", "frames", "seconds", "clips", "drafts", "missing", "gaps"}}` says what an export would hold |
| `GET /project/{name}/history` | | `{"revisions": [{"revision", "time", "source", "clips", "frames"}]}`, newest first |
| `POST /project/{name}/restore` | `{"revision", "if_revision"?}` | `{"revision", "summary"}`. Optionally guard the current revision with `if_revision`. A mismatch returns 409. |
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

A clip with `"kind": "footage"` starts with the imported take. Upscaling creates a
second take that you can switch to or remove. At least one take must remain.

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
| `mid_guides` | Experimental interior guides for a generated shot. Array of `{"frame":34,"file":"pose.png","subfolder":"h3take_projects/film/assets"}`. Set `[]` to clear. |
| `track` | A sound asset plus `from`, the second of it at the clip's first frame. The picture is generated to that slice, and the take keeps the exact sound. `null` removes it |
| `bin` | The id of the bin the clip is filed in. `null` takes it out |
| `loras` | LoRAs only this clip samples with, as `set_project` `loras` describes. `null` clears them |
| `profile` | The checkpoint family this clip renders on, one of the ids `set_project` takes. `null`, the default, is the project's. A take records the one that made it in `models.profile` |

`update_clip`: `clip`, `fields`. Changing `seconds` on a clip with no render resizes its
items, and later clips move with them. `remove_clip`: `clip`, together with all its
timeline items. Its render files stay on disk, so older revisions still play. A render still
running for it is thrown away when it finishes, so cancel it first. `place_clip`: `clip`,
optional `start`. Returns a new `item`.

For an interior guide, upload a picture through `/asset` or capture a take frame through
`/grab`. Add its asset and destination frame to `fields.mid_guides` through `add_clip`
or `update_clip`. Updates replace the whole list. Read the current list and send its
project revision when editing one entry. The summary returns the guides sorted by frame.

Frames are zero-based in the source shot. On the timeline use
`min(item.out - 1, floor(item.in + (timelineFrame - item.start) * speed + 0.5))`
inside the placement, with speed defaulting to 1. Each frame must be unique and strictly
between the first and last generated frames of an ordinary shot (1 through length − 2).
Linked shots also exclude held frames. Up to eight pictures fit, or seven for a bridge that
already pins its full arrival hold. Shots sharing a source clip share its guides.
Generating a draft or final saves the authored guides with that take. The model steers
toward the pictures and may hold or jump near one. Review motion on both sides.

### Extra sound

Music, narration or effects under the cut, on up to 4 lanes of their own beside the clips' own
sound. Upload the file with `POST /project/{name}/asset?sound=1`, which also returns its
`seconds`.

| op | Takes | Notes |
|---|---|---|
| `add_sound` | `asset`: `{"file", "subfolder"}`, `seconds`, optional `fields` | Returns `{"sound"}`. It starts at frame 0 on lane 0 and plays its whole length unless `fields` say otherwise |
| `update_sound` | `sound`, `fields` | |
| `set_audio_track` | `lane`, `fields`: `gain`, `gain_points` | Lane 0 carries picture audio. Lane 1 corresponds to Sound 2, and so on. Gain 1 is unity. Track levels range from 0 to 2. Points use absolute timeline frames and hold their first and last levels outside the keyed span. Send `{"gain":1,"gain_points":[]}` to reset the track. |
| `remove_sound` | `sound` | The file stays in the project's assets |

`fields`: `name`; `start`, the timeline frame it begins at; `offset`, the second of the file it
starts from; `frames`, how long it plays in timeline frames, never past the end of the file;
`gain`, from 0 to 2, where 1 leaves it as it is; `lane`, 0 to 3. The summary lists them as
`sounds`. An export mixes each one in at its place and gain, and stops it where the cut ends.

`fade_in` and `fade_out` are lengths in whole timeline frames (0 disables that fade).
Each is limited during playback to the placed sound's length. Overlapping fades multiply.
`gain_points` is a list of up to 64 `{"frame": 24, "gain": 0.5}` pairs. The frame is
relative to the **start of the source file**, at 24 fps, so moving or trimming the
sound preserves its automation. A point may lie at either file boundary; each
unmarked file boundary defaults to full level. Gains from 0 to 1 multiply the
sound's overall volume and its fades. Levels interpolate linearly. To duck music,
place four points at the start of the ramp down, start and end of the spoken line,
and end of the ramp up. Send the whole list with `update_sound`, or `[]` to clear
it. The summary includes the fades and points when set.

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
| `update_item` | `item`, `fields`: `start`, `in`, `out`, `speed`, `transition`, `gain`, `crop`, `fade_in`, `fade_out`, `gain_points`; optional `ripple` | `transition` can be `"none"`. Speed and transition always move the clips after it. With `ripple`, trimming does too. `gain` scales the clip's own audio from 0 to 2. Fades count timeline frames from the placement edges. Gain points use `{"frame": source_frame, "gain": 0.0–1.0}` and interpolate between frames. The viewer and export apply both. |
| `remove_item` | `item`, optional `ripple` | Ripple closes the space it leaves |
| `split_item` | `item`, `frame` (a timeline frame inside it) | Returns `right`, the new second half |
| `close_gap` | optional `frame` | No frame closes every gap. Returns `removed` |
| `choose_take` | `clip`, `take` | Items keep their trim, cut short if the take is shorter |
| `remove_takes` | `clip`, `takes` (one or more take IDs), optional `allow_empty` | Read IDs from the clip in the project. The newest remaining take replaces a removed active take. To clear even the last take from a generated shot, set `allow_empty: true`. Its prompt and timeline edits survive for the next render. Imported footage needs one take. Returns `clip`, `removed` and `active`. Other clips using the media, files on disk, notes and history are preserved. Supply the current `revision` on the batch to guard the selection. |

To slip a clip, send `update_item` with `in` and `out` moved by the same number of frames:
the item keeps its `start` and length and shows another part of its take. Switching a
clip's take, or rendering it again, also keeps the item where it is; a shorter take pulls
`out` in.

To crop a placement, set `update_item.fields.crop` to a rectangle, for example
`{"x":0.5,"y":0.25,"w":0.5,"h":0.5}`. Coordinates are fractions of the
original take measured from its top left. A rectangle from a frame note uses
the same coordinates. The chosen region fills the aspect ratio of the cut.
The summary includes the rectangle on the timeline item. Set it to `null` to reset.

To export a marked range, pass `"range":{"from":24,"to":48}` to the export
route, including for a plan. The 24 exported frames are timeline frames 24
through 47. Bounds must be whole frames inside the cut. The output keeps the
cut's size, transitions, rendered VFX and mixed sound. Omit `range` for the whole cut.

### VFX clips: drawn by code

`add_effect`: `fields` with any of `name`, `start` (timeline frame), `frames`, `source` (the module's text), `params`
(control values) and `inputs` (pictures the code may draw: `{"logo": {"file", "subfolder"}}`, each an uploaded asset or
any picture in the output folder, such as a take's filmstrip; the code reads them as `env.inputs.logo`, an ImageBitmap)
and `data` (any JSON the code should read as `env.data`, such as word timings for captions; up to 2 MB). Returns `{"effect"}`. `update_effect`: `effect`, `fields`. `remove_effect`: `effect`. They sit on one
lane above the picture and never overlap each other. The summary lists them under `effects` with a `state`: `not rendered`,
`rendered` or `out of date`.

The module is JavaScript:

```js
export const meta = { controls: { size: { type: 'number', min: 0, max: 1, default: 0.5 } } };   // optional
export async function setup(env) {}                                                             // optional
export function frame(env, k) {}     // draw frame k; depend only on k, env.params and env.below
```

`env` holds `canvas` (an OffscreenCanvas: take a `2d` or a WebGL context from it), `width`, `height`, `fps`, `frames`, `params`,
`inputs`, `data`, `below` (the
picture under this frame as an ImageBitmap, or null where the lane below is empty) and `random(seed)`. A module that mentions
`THREE` gets three.js as `env.THREE`, and one that mentions `belowTexture` gets the picture below as a ready `THREE.Texture`.
Control types: `number`, `int`, `bool`, `enum` (with `options`), `color`, `text`. It runs in a worker inside a sandbox: there is
no `document` or `window`, no network and no imports. A module that takes over 20 seconds on a frame is stopped.

`render_effect`: `effect`. A render bakes the clip over the picture below it into a take. The code runs in a browser, so
the request waits until an editor page is open on the project, and that page does it, usually in seconds. The summary
shows the clip as `waiting for an open editor to render it`, then `rendering`, then `rendered`; `wait` returns once no VFX
render is waiting either. If none is open, ask the user to open the editor, or open it yourself in a browser you control.
A failure is recorded on the clip as `error`, with the line of the module it came from. An export leaves out a VFX clip
with no take, and its plan counts `effects_missing` and `effects_stale`.

### Linked clips

`extend_after`, `extend_before`, `bridge`, `merge`: `item`, optional `seconds`, `action`,
and `fields` (any clip field). `bridge` also takes `arrive`. Left out, `seconds` is 4.25,
and 6.375 for a bridge (not with `"arrive": "hold"`): a move between two framings needs
that room, and shorter ones tend to dissolve or cut. Each returns `{"clip", "item"}`, and the new clip still
needs a render.

- The source item must be rendered, with at least 39 frames on the timeline.
- `bridge` and `merge` join the item to the next one on the timeline, which must be rendered too.
- A `bridge` lands on a short run of the next clip: it holds that clip's first 22 frames,
  picture and sound, at its own end, and the timeline cuts 9 frames inside them, so the
  next item's `in` moves forward by 9 (and onto the latent's grid first, by 16 frames at
  most). The link says `"arrive": "short", "end_hold": 22`, and the next item needs 22
  frames on the timeline. `"arrive": "hold"` holds 39 frames of the next clip instead:
  H3 usually cuts to a run that long, so use it only into a clip that starts in
  mid-movement. `"arrive": "pin"` pins the next clip's first frame alone as the bridge's
  last and leaves the next item untrimmed; the take lands there without that clip's
  movement or sound. After rendering, read the cuts report: `sound`, `movement` and
  `parts` on the cut into the next clip, and `inside` for a bridge that cut or faded
  inside itself, which wants another seed.
- `merge` also needs the two items side by side at 100% speed, the first with at least
  71 frames and the second with at least 70. It replaces the last 32 frames of the
  first and the first 31 of the second with 63 new ones, so the scene keeps its length.
- A linked clip holds 39 real frames from each side it joins, and both files then show
  them. The timeline cuts from one to the other 9 frames inside them, where the two
  match best: the source item's `out` moves back 9 frames (or its `in` forward, at the
  side a clip arrives at), and the linked item plays those 9 itself, from `in` 30. The
  scene is as long as it would be with the cut at the edge, and nothing is seen twice.
  The link records this as `handover: 9`. Links made before have none and cut at the edge.
- `extend_after`, `extend_before` and `bridge` hold those frames as the latents the
  source was sampled in, which needs them to start on a multiple of 17 frames. A whole
  clip already does. On a trimmed one, the `out` it carries on from moves back, and the
  `in` it arrives at moves forward, by 16 frames at most, and the items after it close
  up. Read the new `in` and `out` from the summary. Footage keeps its cut, and `merge`
  holds its frames from the video.
- A first or last frame on a held side is ignored, because that side already has real
  frames.
- A linked clip keeps its source clip's `cast`.
- When the source clip holds a `track`, the linked clip holds the next slice of it.
  For `extend_after`, `bridge` and `merge` that is
  `from = source from + (end of the held frames - 39) / 24`. For `extend_before` it is
  `from = source from + (start of the held frames - (new length - 39)) / 24`, which is
  refused if it would fall before the start of the sound. The held frames end 9 frames
  after the source item's `out`, and start 9 before its `in`. The source item must be
  at 100% speed.
  A bridge that holds lays the same track under the frames it holds from the next clip,
  so that clip should continue the track at that point.

### Project and cast

`set_project`: `fields`: `style` (applies to every clip), `format` (a positive integer
ratio between `1:8` and `8:1`, including `3:2`, `4:3`, `7:5` and `21:9`),
`language` (for example `English` or `Portuguese`), and `profile`, the checkpoint
family new renders use (`hyperflow`, the default, `h3_turbo`, `h3`, `singularity`, `dasiwa`
`dasiwa_turbo`, `dasiwa_v5` or `eros`). `GET /projects` lists the profiles with whether each is installed;
pick only an `available` one. A new project starts on the first installed profile, and a
project that names none renders on it. The format sets new renders and imports.
Export fits every clip to the widest rendered take on the timeline.

`set_project` also takes `sizes`, for a GPU with less memory or for speed:
`{"draft": "medium", "final": "large"}`. A draft is `small` (the usual) or `medium`. A final is
`full` (the usual: sampled large, then enlarged by redrawing its last steps at the larger size),
`sharp` (sampled large once, enlarged two times with RelayVSR and scaled down to the same size as
`full`; it needs RelayVSR's two files), `large`, `medium` or `small`, which come out as sampled. Either can also be a size of the
project's own, written `"1024x576"`: each side a multiple of 32 from 256 to 2048, any shape,
and it comes out as sampled. A size off that grid is refused. Leave a quality out, or send `null`, for the usual size. `GET /projects` lists
every choice per format under `sizes`, with `width`, `height` and the size it comes out at,
`out`, and under `own` what a size of the project's own may be (`step`, `min`, `max`) with
one to start from, `first`. The summary's `sizes` says what the project renders at.

A size can also be `{"longest_side": 1280}` or `{"megapixels": 0.8}`. The editor
resolves the project ratio and target onto its 32-pixel grid and saves the resulting
width and height. Send `format` and `sizes` in one `set_project` to use the new ratio.
`GET /sizes?format=4:3&megapixels=0.786432` previews this as `"size":[1024,768]`,
with the normalized `format` and its preset `choices`. Use either target or omit
both for the preset list alone.

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
  and pose stay as they would be without it; 0.93 is a good start for a look or skin LoRA.
  H3 spends most of a render near 1: 0.93 is about halfway through, 0.75 four fifths.
- A clip takes `loras` too, in `add_clip` and `update_clip` `fields`, in the same shape: the
  LoRAs only that clip samples with, after the project's. A row for a file the project also
  lists takes that row's place for this clip, so `{"file": ..., "strength": 0.4}` changes its
  strength there and `{"file": ..., "on": false}` leaves it out. `null` clears the clip's list.
- `realism` is how strong the Realism Slider is on finals, from 0 to 3. It is 1.2 unless
  set, 0 turns it off, and `null` goes back to 1.2. Drafts never run it. Extensions, bridges
  and merges run half of it, since they carry on frames that went through it already.

Takes already rendered do not change: render a clip again to see a new strength.

`set_project` also takes `patches`, the model patches every new render of the project samples
with. A patch is a node of any installed pack that takes the model and plain values and returns
the model: attention and memory patches, caches, compile nodes and the like.

```json
{"op": "set_project", "fields": {
  "patches": [{"node": "SomeAttentionPatch", "inputs": {"chunks": 8}},
              {"node": "SomeStepCache", "on": false}]}}
```

- `patches` replaces the whole list: up to 8. Each is put on the model in the order given,
  after every LoRA, on drafts and on both passes of a final. `"on": false` keeps one in the
  list without rendering with it. Send `[]` to clear the list. It is set for the project;
  a clip cannot have its own.
- `node` is a node id from `patches` in `GET /projects`. That list holds every installed node
  that returns exactly one MODEL, requires exactly one MODEL, and whose other required
  inputs are numbers, switches, text or menus. The ones whose name or category mentions H3,
  MiniMax or attention come first. Each row has the node's `title`, `category`, the `pack` it
  comes from, sometimes `about`, and its `controls`: `{"name", "kind", "default"}` with `kind`
  one of `int`, `float`, `bool`, `text` or `choice`, plus `min`, `max` and `step` for a
  number, `options` for a choice, and the node's own `tooltip`. The pack's own nodes, plain
  LoRA loaders and the model shift every render already ends with are not listed.
- `inputs` names only what differs from the node's defaults: `{name: value}`, with the names
  of `controls`. A render is refused, with the input's name, when a name is not one of the
  node's or a value is outside its range or options. Nothing is checked when the list is set.
- A patch whose node is not installed on this machine stays in the project and is left out
  of renders. The summary lists the project's patches under `patches`.
- A project opened from a zip has every patch switched off, since whoever made the zip chose
  the nodes and their values. The reply to `POST /projects/open` names them in `patches_off`.
  Tell the user which ones they are, and switch one on only when the user asks for it.
- A patch takes up to 64 inputs, and a text value up to 4000 characters.
- The editor wires a patch as its node's definition says and knows nothing else about it. It
  cannot tell whether a patch helps or harms on this model, this GPU or this ComfyUI: render
  the same clip with the same `seed` with it and without, and compare how long the renders
  took and what `h3.py cuts`, `h3.py detail` and `h3.py look` say about the result.

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
`vae` in `models`. `text_encoders` also lists any Qwen3-VL 4B or 8B that has a projection
in `models/clip_projections`: it renders through that projection with about a third of the
32B's memory, and knows less (named people, speech in languages other than English). Use it
only when the user asks, or when the 32B does not fit. `encoders.projections` maps each such
encoder to the projections that fit it, the default first; `set_project` `text_projection`
picks one (a file name, or `null` for the default), and a take records it as `text_projection`.

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

**The cast catalog.** Cast members can be kept once per ComfyUI install, for any project to
bring in. `GET /projects` gives `catalog`, a list of entries: `id`, `version`, `name`,
`description`, and the picture, `pictures`, `voice` and `refmod` as a cast member has them,
pointing at the catalog's own copies. A project's copy carries `catalog`: `{"id", "version",
"synced"}`, and the summary's `cast` rows show the `catalog` id.

- `save_cast_to_catalog`: `cast`. Copies the member's files into the catalog and adds an
  entry, or, for a member that came from the catalog, writes over its entry and raises its
  version. Returns `{"cast", "catalog", "version"}`.
- `add_cast_from_catalog`: `catalog`, the entry's id, and `as` to name the new member for
  later operations of the batch. Copies the entry's files into the project. Returns the same.
- `update_cast_from_catalog`: `cast`. Brings a linked member up to the catalog's version,
  replacing changes made to it in the project.
- `POST /cast-catalog/spread` with `{"id"}` brings every project's copy of an entry up to
  date, except copies changed in their project; it returns `updated` and `kept`.
  `POST /cast-catalog/remove` with `{"id"}` takes an entry and its files out of the catalog;
  projects keep their copies. `GET /cast-catalog` lists the entries.

Change a catalog member only when the user asks: a newer catalog version changes what the
project renders next, and the user may have changed the copy on purpose.

The description is how H3 knows them. Describe a face and clothes for a picture, and how
they sound for a voice. A cast member without a picture is never placed in the shot, and
their description stands in for them in the prompt. Write `"@Name, off-screen"` as the
speaker when they should not be seen.

### Notes

Notes are what a reviewer pinned to a frame. Each one belongs to a clip and the take it was
written on, and counts frames inside that take, so it stays on its picture when the clip moves.

- `add_note`: `fields`: `clip`, `frame` (a frame of the take), `text`, optional `take` (the
  active one by default), `range` (`{"from", "to"}`, frames of the take; the note then starts
  at `from`), `shape`, `author`. `shape` is `{"type": "pin", "x", "y"}`,
  `{"type": "rect", "x", "y", "w", "h"}` or `{"type": "stroke", "paths": [[[x, y], ...], ...]}`,
  in fractions of the frame from its top left. Returns `{"note"}`.
- `resolve_note`: `note`, optional `resolution`, a line on what was done. `reopen_note` and
  `remove_note` take `note`.

The editor saves a still of each note it writes, the frame with the drawing on it, under
`still` (`{"file", "subfolder"}` in the output folder).

### Renders

`render`: `clip`, optional `quality` (`draft`, the default, `final`, or `finish`), optional `seed`.
`finish` makes a final from the clip's active take, which must be a draft: its saved latents are enlarged and
refined at the refine size in place of a new first pass, with the draft's seed, so the final keeps the draft's
framing, motion and sound. The new take records `finished`, the draft's id. It is refused when the active take is
not a draft, when the clip's length changed since the draft, and when the project's final size is Sharp; a
draft with no latents saved beside it fails at render time and needs rendering again.
The latent upscaler is required. Full finishes at the refine size; Large, Medium, Small and custom final sizes
finish directly at the selected size, without a new first pass.
Returns `{"clip", "job", "take"}`. It is refused for footage, for a clip with no action, and for a clip already
rendering. Renders run after the edits in the same batch. `cancel`: `clip` or `job`.

`POST /project/<name>/cancel` also accepts `{"all": true}` and optional `quality: "final"` or `"draft"`.
It deletes matching queued jobs before interrupting a matching running job and returns `cancelled` as a list.
Other projects' jobs are untouched.

To enlarge the active final (or imported footage) with experimental VOSR2, use
`{"op":"render","clip":"<id>","quality":"upscale","upscaler":"vosr2"}`.
See its `upscalers` row in `GET /projects`: `installed` checks the optional
wrapper's registered nodes and their interface, `weights_cached` checks the
on-disk bundle, and `setup` explains the next step. The wrapper can fetch its
roughly 7 GB weights on first use. An explicit VOSR2 selection processes frames
at 2× with a fixed seed and copies the original audio. `wait` and the project
summary report completion or failure. Inspect motion for flicker.

A clip can have `style_markers: [{"frame": 0, "name": "Pencil", "style": "Monochrome pencil drawing"}, ...]`.
Frames are relative to the generated take (before timeline trim and speed). Up to 32 markers and four distinct
styles are supported. Markers blend across clip time and add model evaluations; see [style transitions](../../../docs/style-timeline.md).

`GET /looks` lists the installation-wide style library. `POST /looks` saves `{name, style}`; send an existing
custom look's `id` to update it. `DELETE /looks/<id>` removes a custom entry. Built-ins are read-only and project
style text is always an independent copy. `POST /spectrogram` accepts `{filename, subfolder}` and returns a
cached PNG and audio duration under `spectrogram`, or null for a file without audio.

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
project's own LoRAs as `loras`: `[{"file", "strength"}, ...]`, and the model patches it ran with as
`patches`: `[{"node", "inputs"}, ...]`, each with every value its node got, defaults included. Finals from before this record
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
 "notes": [{"note": "n1", "status": "open", "clip": "c1", "name": "Desk", "take": "t9", "frame": 40,
            "range": null, "at": "00:01:16", "timeline_frame": 40, "text": "...",
            "where": "a box from 10% to 40% across and 20% to 60% down", "earlier_take": true,
            "still": {"file", "subfolder"}, "resolution": "..."}],
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
  ones and the 9 held frames beside them, in steps of about 2.1 s;
- `bridge` renders 39 + 51k frames and shows them from frame 30 to 13 frames before the
  end: all but 30 of the 39 it holds of the first clip, and 9 of the 22 it holds of the
  next. With `"arrive": "hold"` it renders 39 + 51k frames with k of 2 or more, and shows
  all but 30 of the 39 held at each end. With `"arrive": "pin"` it shows up to the frame
  before its last. A merge is the shortest holding bridge: 141 frames,
  81 of them shown, 63 of them new.

The timeline reserves that length before the render, so a finished take lands where the
timeline expects it.

### Sprite Draft

For an independent generated shot, pass `sprite` in `add_clip` or `update_clip` fields. For example, `{"sprite":{"direction":"E","camera":"rpg","poses":7}}` uses the built-in jog. Direction is S, E, N or W. Camera is level, rpg (45 degrees) or overhead (90 degrees). Choose 4–12 poses. Use `"motion":"idle"` for a planted idle, or `"motion":"prompt"` to drive movement entirely from the shot action. The optional `cutout` field accepts `auto` (default) or `edge`. Auto uses ComfyUI’s installed `background_removal/birefnet.safetensors` when available. The background key also works without weights. For another movement, add `guide: {"file":"movement.mp4","subfolder":"..."}` using an imported output video at 24 fps with at least 39 frames. Set `sprite` to null to return to normal shot timing. Use the existing cast pictures and shot action for character identity.

The default sprite model is the first installed of HyperFlow H3 Flash and H3 Turbo, then the project profile. An explicit shot profile or pinned project checkpoint still applies. Render a draft for 39 frames at 256 × 384. The completed take’s `sprite` object gives `illustrated` and `indexed` PNG filenames in the take’s subfolder, tile size, pivot, playback fps, pose count and extraction layout. Play cells from left to right at the reported `fps`, 6 for idles and 12 for other motion. An idle strip bakes in the return without doubling the endpoints. Seven sampled poses produce twelve PNG cells. `source_poses` counts the sampled poses and `poses` counts the exported cells. The output `cutout` field identifies the method used for that take. Each 64 × 64 cell targets a 48-pixel median character height with a shared ground pivot at (32, 59). The indexed strip uses up to 15 opaque colors and a transparent palette entry. Finish the chosen draft through the usual `quality: "finish"` route to refine at 512 × 768 using its saved latent and extraction layout.
