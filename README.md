# H3 Take

An editor and a ComfyUI node for MiniMax H3. It uses ComfyUI's own H3 nodes and
needs no other custom node pack.

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
- Export joins the timeline with ffmpeg and does not use the GPU. A gap becomes
  black and silence.
- Renders belong to the project, so closing or refreshing the editor does not
  lose one.

Projects live in `h3take_projects` in ComfyUI's output folder.

| Key | Does |
|---|---|
| Space | Play or pause |
| Left, Right | Move one frame. Hold Shift for one second |
| S | Split at the playhead |
| Delete | Remove from the timeline |
| Ctrl+Z, Ctrl+Shift+Z | Undo, redo |

The editor writes each prompt in the structure from MiniMax's H3 prompt guides.

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

## Develop

`python -m unittest discover -s tests` runs from the repo root. Writer tests
need torch and ComfyUI, and skip without them.

`tools/deploy.sh` copies the repo into a running container, runs the tests
there, and reloads the pack in place, so models stay loaded. Refresh the browser
afterwards. `--restart` restarts ComfyUI instead. `tools/run_take.py` queues a
real take.
`tools/compare_takes.py` measures how closely a span matches its source.
