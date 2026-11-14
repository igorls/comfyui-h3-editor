# The nodes

H3 Editor builds its graphs from ComfyUI's own MiniMax H3 nodes and the nodes
below. You can use them in your own workflows too.

| Node | What it does |
|---|---|
| H3 Take | One shot: the prompt, pinned frames, references, held video or sound |
| H3 Live Preview | Shows the whole clip moving, with sound, at every sampling step |
| H3 Load Clip | Reads frames and sound from a clip in the output folder, and the saved latent of those frames when there is one |
| H3 Save Hold | Saves the first and last 39 frames of a sampled take as latents beside its video, for clips that extend it |
| H3 Load Still | Reads a picture from the output folder |
| H3 Load Sound | Reads a sound from the output folder, such as a cast voice |
| H3 LoRA | Applies an H3 LoRA, wherever it can land on the model; see [models](models.md#loras) |
| H3 Refine Latent | Prepares an upscaled H3 latent for a few more steps, keeping held frames and the sound |
| H3 HyperFlow Loader | Loads a HyperFlow H3 Flash checkpoint: a GGUF file in any quantization, or an INT8 safetensors file |
| H3 HyperFlow Sigmas | The 4, 6 or 8-step schedule a HyperFlow checkpoint samples, for euler without CFG |
| H3 Small Text Encoder | A Qwen3-VL 4B or 8B in place of H3's 32B text encoder, through a ClipProj projection; see [models](models.md#a-smaller-text-encoder) |

## Templates

ComfyUI's template browser lists two ready workflows under this pack:
*H3 Take - text to video* and *H3 Take - extend a clip*. Open one, write the
shot in H3 Take, and press Run.

Both render on HyperFlow H3 Flash INT8. H3 HyperFlow Loader names the fl2va
file, which takes a prompt and pinned frames, and H3 HyperFlow Sigmas gives it
6 steps. To use reference pictures, choose the ref2va file in the loader and
set `checkpoint` to ref2va on H3 Take. [Models](models.md) lists the files and
where they go.

The template graph loads the model, the text encoder and the two H3 VAEs, sets
`MiniMaxH3SigmaShift` to 12 and 3, passes the model through H3 Live Preview,
connects H3 Take to the guider and sampler, then decodes picture and sound and
saves the video. Each template carries a note with the settings that matter.

## H3 Take

One H3 shot in a graph. Sampling, the checkpoint and `MiniMaxH3SigmaShift` stay
on the graph. The editor builds the same graph for each clip.

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

### Checkpoint

Set `checkpoint` to match the model on the graph. H3 comes in two kinds, and
they read the prompt differently. `ref2va` takes references. `fl2va` takes a
prompt and pinned frames, and refuses references.

### Prompt labels

The `labels` output lists what each number means in this take.

- Picture 1 is the start pin, when there is one.
- The end pin is the next picture.
- Soft references come after the pins.
- Mid pins are guides and take no picture number.
- Audio numbers count video soundtracks first, then `ref_audio_*`.

A source video is not a picture number.

### Held video

`strength` says how far the held video may move. The generated frames, before
or after that span, are always free.

| Strength | Result |
|---|---|
| `hold` | The source stays |
| `light` | A tint of the prompt, same shot |
| `medium` | The prompt changes what is in the shot; framing and pose stay |
| `heavy` | The prompt leads; the shot is still recognizable |
| `regenerate` | Nothing of the source stays |

These were measured on an earlier H3 checkpoint at 8 steps. Another checkpoint
or step count may need `custom`, which reads the raw mask value from `keep`.
The range is not linear: below about 0.95 the source wins outright.

`extend_after` holds the source's tail at frame 0. `extend_before` holds its
head at the end. `extend_both` holds the middle and leaves at least five new
frames on each side. `align=av` snaps the hold to 39, 90, 141 and so on, so the
audio clock matches. `align=video` only obeys the frame grid.

The node writes the source into the latent and sets a nested noise mask, so
connect its `latent` output to the sampler. `video_feather` softens the edge
where held video meets generated video. `audio_feather` does the same for
sound, where 40 ticks are one second. Source video must be 24 fps.

A pin is one still. Passing a clip to those sockets is not video-to-video:
connect the clip as `source_video` and set `video_mode`.

### Known limits

Mid pins are not ready. In a test on an earlier checkpoint, a still pinned at
frame 68 froze for about nine frames, with a hard jump in and out. Start pins,
end pins and held video do not do this.
