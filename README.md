# H3 Take

One ComfyUI node for a MiniMax H3 shot. It calls ComfyUI's own H3 nodes and
needs no other custom node pack.

Sampling, the checkpoint, and `MiniMaxH3SigmaShift` stay on the graph.
Hiding those is how a pack takes control away.

## What it does

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

## Timeline

`H3 Timeline` is the editor. Place a held source and the generate regions
before and after it. Drag a block to move it. Drag an edge to trim. A region
longer than the VRAM cap becomes sequential passes. Under 24 GB the cap is
124 frames, under 48 GB it is 243, otherwise 362. Later passes hold the last
39 frames of the one before them. The node describes the passes. It does not
run them yet.

## Install

Clone this folder into ComfyUI `custom_nodes/` and restart ComfyUI. The
timeline script loads with the node. It does not download a model.

## Develop

`python -m unittest discover -s tests` runs from the repo root. Writer tests
need torch and ComfyUI, and skip without them.

`tools/deploy.sh` copies the repo into a running container, runs the tests
there, and restarts ComfyUI. `tools/run_take.py` queues a real take.
`tools/compare_takes.py` measures how closely a span matches its source.
