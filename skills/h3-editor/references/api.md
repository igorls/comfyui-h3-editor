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
| `GET /projects` | | `{"projects": [folder, ...]}` |
| `POST /projects` | `{"name"}` | `{"folder", "project"}`. Opens it if it exists |
| `GET /project/{name}` | | The whole project |
| `GET /project/{name}/summary` | | The summary, below |
| `POST /project/{name}/ops` | `{"ops": [...], "revision"?}` | `{"results", "revision", "summary"}` |
| `GET /project/{name}/wait?timeout=600` | | The summary, plus `timed_out`, once nothing renders. `after=N` waits for a revision past N instead. Timeout is at most 1800 |
| `POST /project/{name}/render` | `{"clip", "quality", "seed"?}` | `{"job", "revision", "project"}` |
| `POST /project/{name}/cancel` | `{"job"}` or `{"clip"}` | `{"cancelled", ...}` |
| `POST /project/{name}/import` | multipart `file` with fields, or JSON `{"source": {"folder": "input" or "output", "subfolder"?, "file"}, ...}` | `{"clip", "item", "frames", "revision", "project"}` |
| `POST /project/{name}/asset` | multipart `file`: PNG, JPEG or WebP, or WAV, MP3, FLAC, M4A, OGG, OPUS, AAC | `{"file", "subfolder"}`, an asset |
| `POST /project/{name}/grab` | `{"file", "subfolder", "frame"}`, a take's file and a frame of it | A picture asset |
| `POST /project/{name}/grab_sound` | `{"file", "subfolder", "first", "last"}` in take frames | A WAV asset, 30 s at most |
| `POST /project/{name}/export` | `{}` | `{"filename", "subfolder"}` in ComfyUI's output folder |
| `GET /project/{name}/history` | | `{"revisions": [{"revision", "time", "source", "clips", "frames"}]}`, newest first |
| `POST /project/{name}/restore` | `{"revision"}` | `{"revision", "summary"}` |
| `POST /sheet` | `{"filename", "subfolder", "frames"?, "first"?, "last"?, "count"?, "tile_width"?}` | `{"sheet", "picked", "frames"}`: a grid of frames saved beside the clip |

Import fields: `name`, `fit` (`contain` or `cover`), `seconds` for a picture (0.5 to
60, default 4), `start_seconds` and `end_seconds` to take part of a video, `start` as a
timeline frame, and `place` (`false` keeps it off the timeline). Videos can be MP4, MOV,
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

## Operations

Every operation is `{"op": ..., ...}`. Add `"as": "label"` to name its result. Later
operations in the batch can then write `"$label"`, and the clip or item is picked to fit
the field. `"$label.item"` and `"$label.clip"` choose explicitly. A clip's `cast` list can name cast members
added earlier in the batch the same way: `"cast": ["$her"]`.

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

`update_clip`: `clip`, `fields`. Changing `seconds` on a clip with no render resizes its
items, and later clips move with them. `remove_clip`: `clip`, together with all its
timeline items. `place_clip`: `clip`, optional `start`. Returns a new `item`.

### Timeline

| op | Takes | Notes |
|---|---|---|
| `update_item` | `item`, `fields`: `start`, `in`, `out`, `speed`, `transition`; optional `ripple` | `transition` can be `"none"`. Speed and transition always move the clips after it. With `ripple`, trimming does too |
| `remove_item` | `item`, optional `ripple` | Ripple closes the space it leaves |
| `split_item` | `item`, `frame` (a timeline frame inside it) | Returns `right`, the new second half |
| `close_gap` | optional `frame` | No frame closes every gap. Returns `removed` |
| `choose_take` | `clip`, `take` | Items keep their trim, cut short if the take is shorter |

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
`1:1`), and `language` (for example `English` or `Portuguese`). The format sets new
renders and imports, and export fits every clip to it.

`add_cast`: `fields`: `name`, `description`, and a picture asset (`file`, `subfolder`), a
`voice` asset, or both. Returns `{"cast"}`. The picture's keys sit directly in `fields`,
while the voice is its own object:

```json
{"op": "add_cast", "as": "ana", "fields": {"name": "Ana", "description": "a woman in her thirties, short dark hair, white shirt",
  "file": "3f1c.png", "subfolder": "h3take_projects/Teaser/assets",
  "voice": {"file": "9a0b.wav", "subfolder": "h3take_projects/Teaser/assets"}}}
``` `update_cast`: `cast`, `fields`: `name`,
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

| quality | 16:9 | 9:16 | 1:1 | steps |
|---|---|---|---|---|
| draft | 640x352 | 352x640 | 480x480 | 4 |
| final | 1344x768 | 768x1344 | 1024x1024 | 8 |

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
 "clips": [{"id": "c1", "name": "Desk", "state": "draft", "seconds": 4, "action": "...",
            "takes": 2, "take": "t9", "frames": 90, "take_seed": 123, "seed": 7, "quality": "draft",
            "link": "extend_after", "footage": true, "error": "..."}],
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
