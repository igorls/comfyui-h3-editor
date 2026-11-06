# H3 Take

One ComfyUI node for a MiniMax H3 take. It calls ComfyUI's own
`MiniMaxH3ReferenceToVideo` and `MiniMaxH3AddGuide` for stills. It does not
import Context Loop, Motion Context, or DaSiWa Director.

Sampling, the checkpoint, and `MiniMaxH3SigmaShift` stay on the graph.
Hiding those is how a pack takes control away.

## What it does

| You want | Set |
|---|---|
| Start on a still | `anchor=start`, connect `start_image` |
| End on a still | `anchor=end`, connect `end_image` |
| Both ends | `anchor=both` |
| A picture the model may use, not a frame it must hit | `ref_image_1` … `ref_image_3` |
| A voice timbre | `voice` plus the H3 audio VAE |
| Change a clip in place | `video_mode=v2v`, source length equals `length`, set `keep` |
| New frames after the clip | `video_mode=extend_after` |
| New frames before the clip | `video_mode=extend_before` |
| New frames on both sides | `video_mode=extend_both` |

Picture numbers are fixed, so the prompt cannot drift:

- Picture 1 is the start pin, when there is one
- The end pin is the next picture
- Soft references come after the pins
- Voice is always `<Audio 1>`, timbre only

A source video is not a picture number. The `labels` output says where it sits.

## Video is a held span

`keep` 0 freezes the held frames. `keep` 1 regenerates them. The generated
frames, before or after that span, are always free. That is the control a
guide clip does not have.

`extend_after` holds the source tail at frame 0. `extend_before` holds the
source head at the end. `extend_both` holds the middle and leaves at least
five new frames on each side. `align=av` snaps the hold to 39, 90, 141, …
so the audio clock matches. `align=video` only obeys the frame grid.

The latent mask writer is not in this folder yet. The placement is. A queued
video mode fails loud instead of pinning the clip with AddGuide.

## What a pin still refuses

A start or end pin is one still. Passing a clip to those sockets is not
video-to-video. Connect the clip as `source_video` and set `video_mode`.

## Timeline

`H3 Timeline` is the editor. Place a held source and the generate regions
before and after it. Drag a block to move it. Drag an edge to trim. A region
longer than the VRAM cap becomes sequential passes. Under 24 GB the cap is
124 frames, under 48 GB it is 243, otherwise 362. Later passes hold the last
39 frames of the one before them. The node does not sample.

## Install

Clone this folder into ComfyUI `custom_nodes/` and restart ComfyUI. The
timeline script loads with the node. It does not download a model.
