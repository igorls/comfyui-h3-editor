# H3 Take

An editor and a ComfyUI node for MiniMax H3. The node uses ComfyUI's own H3
nodes and needs no other custom node pack. The editor renders with HyperFlow H3
Flash and refines finals at a larger size, which needs the models and packs
listed under Install.

## Editor

Open H3 Editor from the ComfyUI sidebar. It is a full-screen timeline editor.

- Add a clip, describe it, and render a draft or a final. Each render goes
  through the ComfyUI queue, and the player shows its live preview.
- Rendered clips stay on disk. Moving, trimming, or splitting them on the
  timeline never renders anything again.
- A new render of a clip becomes another take. Choose the take to use in the
  inspector.
- Select a clip and press Extend forward or Extend back to generate what happens
  next or before. Bridge to next generates what connects it to the following
  clip. These clips hold 39 real frames and their sound from the clips they
  join, so the cut does not show.
- If you change a clip that another one joins, the joined clip is marked. Render
  it again to pick up the new frames.
- Add pictures to Cast to keep a person or place the same across clips, and
  choose who is in each clip. Write @Name in What happens to place them.
- Give a clip a first or last frame by uploading one, or press Use playhead
  frame to take the frame under the playhead.
- Merge with next replaces the join between two side-by-side clips with a
  generated crossing, and the scene keeps its length.
- Speed and transitions belong to the timeline, so they never touch a render.
  Select a clip to set them. The player previews both.
- Each take keeps its seed and the prompt that made it. Keep a seed to render
  variations of one result, or restore a take's prompt.
- While a clip samples, the player loops the whole clip with its sound, updated
  at each step, so you can cancel a bad take early. This needs
  `taeh3.safetensors` in `models/vae_approx`. Without it you get a rough still.
- Export joins the timeline with ffmpeg and does not use the GPU. A gap becomes
  black and silence.
- Renders are queued by the server, which adds each finished take to the
  project. Closing or refreshing the editor never loses one.
- Import brings in real video or a picture. It is fitted to the project's size
  at 24 fps and becomes footage, which you can trim, extend, and bridge from
  like a render.
- Set the project's format (16:9, 9:16, or 1:1) and the language people speak
  in the Scene panel. A clip can speak another language.
- Give a cast member a voice, uploaded or taken from a clip on the timeline, or
  add a voice with no picture for a narrator. Write @Name in Who speaks and
  their lines use that voice.
- Picture to a track: a clip can hold a recorded sound, such as an approved
  narration, and its picture is generated to it. Extend forward carries on with
  the next part of the same track, so a long talking performance can be built
  from short windows. Each take keeps the exact sound.
- Two editors, or an editor and an agent, can work on one project at once.
  Changes are merged, and every change is a revision you can go back to.

Projects live in `h3take_projects` in ComfyUI's output folder.

| Key | Does |
|---|---|
| Space | Play or pause |
| Left, Right | Move one frame. Hold Shift for one second |
| Up, Down | Jump to the previous or next edit |
| Home, End | Go to the start or end |
| S | Split at the playhead |
| Delete | Remove a clip, or close a selected gap |
| Shift+Delete | Remove a clip and close the space |
| [ , ] | Flip between takes of the selected clip |
| Ctrl+wheel, \\ | Zoom at the pointer, fit the whole scene |
| Ctrl+Z, Ctrl+Shift+Z | Undo, redo |

The editor writes each prompt in the structure from MiniMax's H3 prompt guides.

## Drive the editor from code

Everything the editor does is an HTTP call, so an agent can build and render a
cut without a browser. Routes are under `/h3take/api`, and `{name}` is the
project's folder name.

For agents there is a skill in [skills/h3-editor](skills/h3-editor/SKILL.md). It
teaches the whole workflow, from shot list to export, and comes with
`scripts/h3.py`, a client that needs only Python. Copy the folder into an
agent's skills folder to install it.

| Route | Does |
|---|---|
| `GET /projects`, `POST /projects` | List projects, or create one from `{"name"}` |
| `GET /project/{name}/summary` | The cut in brief: clips, their state, the timeline with timecodes, gaps, renders running |
| `POST /project/{name}/ops` | A batch of edits and renders. The edits apply together or not at all |
| `GET /project/{name}/wait` | Waits until nothing is rendering, then returns the summary. `?timeout=` in seconds |
| `POST /project/{name}/import` | A video or picture, as an upload or as `{"source": {"folder": "input", "file"}}` |
| `POST /project/{name}/asset` | A picture or sound file to use as a cast picture, a frame, or a voice |
| `POST /project/{name}/grab_sound` | Part of a clip's sound as a voice: `{"file", "subfolder", "first", "last"}` |
| `POST /project/{name}/export` | Joins the timeline into one video |
| `GET /project/{name}/history`, `POST /project/{name}/restore` | Revisions, and going back to one |
| `POST /sheet` | Twelve frames from across a clip in one picture, to judge a take at a glance |

An operation is an object with an `op`. `as` names its result, so later
operations in the batch can write `"$name"` for the clip or item it made.

| op | Takes |
|---|---|
| `add_clip` | `fields`: name, action, camera, speaker, dialogue, sound, music, seconds, cast, seed, first, last, language, track. Optional `start` |
| `update_clip`, `remove_clip`, `place_clip` | `clip`, and `fields` or `start` |
| `update_item` | `item`, `fields`: start, in, out, speed, transition. `ripple` moves the clips after it |
| `remove_item`, `split_item`, `close_gap` | `item` with `ripple`, `item` with `frame`, or an optional `frame` |
| `extend_after`, `extend_before`, `bridge`, `merge` | `item`, optional `seconds`, `action`, `fields` |
| `choose_take` | `clip`, `take` |
| `set_project` | `fields`: style, format, language |
| `add_cast`, `update_cast`, `remove_cast` | A picture asset, a `voice` asset, or both, with name and description |
| `render`, `cancel` | `clip`, with `quality` draft or final |

```json
{"ops": [
  {"op": "add_clip", "as": "open", "fields": {"name": "Desk", "seconds": 4,
   "action": "A mug of coffee steams on a desk by a window."}},
  {"op": "render", "clip": "$open"}
]}
```

A refused batch answers with the operation that failed and why, and changes
nothing.

## Use the nodes in ComfyUI

There are two ways in.

- **H3 Editor** in the sidebar builds and runs every graph for you. Use it to
  write shots, render them, and cut them together.
- **Templates** in ComfyUI list two ready workflows under comfyui-h3-take:
  *H3 Take - text to video* and *H3 Take - extend a clip*. Open one, write the
  shot in H3 Take, and press Run.

The template graph is: load the model, text encoder, and the two H3 VAEs; set
`MiniMaxH3SigmaShift` to 12 and 3; pass the model through **H3 Live Preview**;
connect **H3 Take** to the guider and sampler; then decode picture and sound and
save the video. Each template carries a note with the settings that matter.

| Node | What it does |
|---|---|
| H3 Take | One shot: the prompt, pinned frames, references, held video or sound |
| H3 Live Preview | Shows the whole clip moving, with sound, at every sampling step |
| H3 Load Clip | Reads frames and sound from a clip in the output folder |
| H3 Load Still | Reads a picture from the output folder |
| H3 Load Sound | Reads a sound from the output folder, such as a cast voice |
| H3 LoRA | Applies an H3 LoRA. On HyperFlow H3 Flash it renumbers the LoRA's blocks to fit |
| H3 Refine Latent | Prepares an upscaled H3 latent for a few more steps, keeping held frames and the sound |

## H3 Take node

One H3 shot in a graph. Sampling, the checkpoint, and `MiniMaxH3SigmaShift` stay
on the graph. The editor builds the same graph for each clip.

### What it does

| You want | Set |
|---|---|
| A shot from text | `prompt` only |
| Start on a still | `anchor=start`, connect `start_image` |
| End on a still | `anchor=end`, connect `end_image` |
| Both ends | `anchor=both` |
| Stills along the shot | connect `mid_image_*`, list their frames in `mid_frames` |
| A picture the model may use, not a frame it must hit | `ref_image_*` |
| A reference clip, with or without its sound | `ref_video_*`, `ref_video_audio_*` |
| A voice or sound the model may use | `ref_audio_*` |
| Change a clip in place | `video_mode=v2v`, source length equals `length`, set `strength` |
| New frames after the clip | `video_mode=extend_after` |
| New frames before the clip | `video_mode=extend_before` |
| New frames on both sides | `video_mode=extend_both` |
| Keep the clip's sound over the held frames | `audio_hold=span`, connect `source_audio` |
| Generate picture to an existing track | `audio_hold=full`, connect `source_audio` |
| New sound on the same picture | `video_mode=v2v`, `strength=hold`, `audio_hold=off` |
| Regenerate one area of a held clip | connect `region`, white regenerates |

## Checkpoint

Set `checkpoint` to match the model on the graph. H3 ships two kinds and they
read the prompt differently. `ref2va` takes references. `fl2va` takes a prompt
and pinned frames, and refuses references.

## Prompt labels

The `labels` output lists what each number means in this take.

- Picture 1 is the start pin, when there is one
- The end pin is the next picture
- Soft references come after the pins
- Mid pins are guides and take no picture number
- Audio numbers count video soundtracks first, then `ref_audio_*`

A source video is not a picture number.

## Video is a held span

`strength` says how far the held video may move. The generated frames, before
or after that span, are always free.

| Strength | Result |
|---|---|
| `hold` | The source stays |
| `light` | A tint of the prompt, same shot |
| `medium` | The prompt changes what is in the shot, framing and pose stay |
| `heavy` | The prompt leads, the shot is still recognizable |
| `regenerate` | Nothing of the source stays |

These were measured on FastH3 Omni NVFP4 at 8 steps. Another checkpoint or step
count may need `custom`, which reads the raw mask value from `keep`. The range
is not linear. Below about 0.95 the source wins outright.

`extend_after` holds the source tail at frame 0. `extend_before` holds the
source head at the end. `extend_both` holds the middle and leaves at least
five new frames on each side. `align=av` snaps the hold to 39, 90, 141, …
so the audio clock matches. `align=video` only obeys the frame grid.

The node writes the source into the latent and sets a nested noise mask, so
connect its `latent` output to the sampler. `video_feather` softens the edge
where held video meets generated video. `audio_feather` does the same for
sound, and 40 ticks is one second. Source video must be 24 fps.

## What a pin still refuses

A pin is one still. Passing a clip to those sockets is not video-to-video.
Connect the clip as `source_video` and set `video_mode`.

## Known limits

Mid pins are not ready. In a test on FastH3 Omni NVFP4 a still pinned at frame
68 froze for about nine frames, with a hard jump in and out. Start pins, end
pins, and held video do not do this.

The `fl2va` route has not been run on a real checkpoint.

## Install

Clone this folder into ComfyUI `custom_nodes/` and restart ComfyUI. The
editor loads with the node. It does not download a model.

The editor's renders need:

- the ComfyUI-HyperFlow-Flash and ComfyUI-GGUF node packs;
- HyperFlow H3 Flash in `models/diffusion_models/hyperflow`:
  `minimax_h3_ref2va_hyperflow_flash_14b-Q8_0.gguf` renders clips with cast
  pictures, first or last frames, or voices, and
  `minimax_h3_fl2va_hyperflow_flash_14b-Q8_0.gguf` renders clips that are only
  a prompt;
- the H3 text encoder and the video and audio VAEs, as for ComfyUI's own H3
  workflows;
- for finals, a ComfyUI recent enough to have the MiniMax H3 latent upscale
  node, with `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors` in
  `models/latent_upscale_models`, and the MMH3 Realism Slider LoRA
  (Civitai model 3014078) as `MMH3-RealismSlider-V1.safetensors` in
  `models/loras`.

A draft renders at the project's draft size in 4 steps. A final renders at the
final size in 6 steps, which is H3's own size of about a megapixel. The learned
latent upscaler then enlarges the picture, and HyperFlow redraws its last 3
steps at that size. A 9:16 final comes out at 1088x1920, a 16:9 one at
1920x1088 and a 1:1 one at 1440x1440. The sound is the first pass's, and held
frames stay as they are. The Realism Slider runs at 1.5 on both passes, for a
matte, less made-up skin. HyperFlow H3 Flash prunes some of H3's blocks: its
ref2va dropped block 9 and its fl2va blocks 6, 21 and 22. The H3 LoRA node
moves an H3 LoRA's later blocks down to match, so they land on the layers they
were trained for. A LoRA attaches only where HyperFlow keeps a layer whole, and
ComfyUI skips the rest with a warning: on fl2va the slider's last `fc2` is one. Export takes the size of the largest picture in the cut. A project can name
other models under `models`. A `.gguf` unet loads as HyperFlow, and any other
unet loads with ComfyUI's plain loader and scheduler.

## Develop

`python -m unittest discover -s tests` runs from the repo root. Writer tests
need torch and ComfyUI, and skip without them.

`tools/deploy.sh` copies the repo into a running container, runs the tests
there, and reloads the pack in place, so models stay loaded. Refresh the browser
afterwards. `--restart` restarts ComfyUI instead. `tools/run_take.py` queues a
real take.
`tools/compare_takes.py` measures how closely a span matches its source.
