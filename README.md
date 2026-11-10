# H3 Editor

A timeline video editor for MiniMax H3, inside ComfyUI. Write your shots,
render them on your own GPU, and cut them together, picture and sound.

**[Project site](https://igorls.github.io/comfyui-h3-editor/)**: watch a cut
made with it, shot by shot.

![H3 Editor: the clips, the viewer, the project's checkpoint and LoRA sliders, and a four-shot cut with its sound on the timeline](docs/images/editor.jpg)

## What you can do

- **Write shots, not graphs.** Describe what happens and what is said. The
  editor builds the ComfyUI graph, queues it, and shows the clip moving, with
  sound, while it samples.
- **Keep people and places the same.** Add their pictures to the cast and
  write @Name in a shot. Give them a voice, or add a narrator who is only a voice.
- **Join shots without a visible cut.** Extend a clip forward or back, bridge
  two clips, or merge a join. A new clip carries on from real frames and sound
  of the clips it joins.
- **Edit as in any editor.** Trim, split, move, change speed, add transitions,
  flip between takes, file clips in bins, bring in your own footage and
  pictures, and export one video.
- **Try fast, finish large.** Drafts are small and quick. Finals are refined at
  up to 1920 pixels, with a realism LoRA for natural skin.
- **Pick the model, per project or per clip.** HyperFlow H3 Flash, MiniMax H3,
  its Turbo LoRA, Singularity and DaSiWa are built in, and a clip can render on
  a different one than the rest of the project. Any other H3 checkpoint works
  too: pick its file, set its steps, sampler and scheduler, and save it as a
  preset for every project.
- **Bring your LoRAs.** Add LoRAs trained on MiniMax H3 to a project and set each
  one's strength with a slider. Every render of the project uses them.
- **Generate to a recorded voice.** Give a clip a recorded track, such as an
  approved narration, and its picture is generated to it.
- **Let an agent make the video.** Everything the editor does is an HTTP call,
  and an agent skill teaches the whole workflow.

It needs no other custom node pack.

## Quick start

### 1. Install

You need a recent ComfyUI with its built-in MiniMax H3 support: 0.43 or newer.
The pack is tested on 0.44. In ComfyUI's `custom_nodes` folder:

```bash
git clone https://github.com/igorls/comfyui-h3-editor
```

```bash
pip install -r comfyui-h3-editor/requirements.txt
```

On the Windows portable build, run the second step as
`..\python_embeded\python.exe -m pip install -r comfyui-h3-editor\requirements.txt`.
The requirements bring ffmpeg, for importing and exporting video, and the gguf
reader for HyperFlow's GGUF models.

Restart ComfyUI. **H3 Editor** appears in the sidebar.

### 2. Download the models

The pack does not download models. These are the ones a first video needs,
about 54 GB:

| File | Folder | Size | Where to get it |
|---|---|---|---|
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders` | 15.7 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3), `text_encoders/` |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae` | 5.2 GB | Comfy-Org/MiniMax-H3, `vae/` |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae` | 0.6 GB | Comfy-Org/MiniMax-H3, `vae/` |
| `minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors` | `models/diffusion_models/hyperflow` | 16.4 GB | [VideoRebirthAI/hyperflow-h3-flash](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash), `ref2va_int8_convrot_comfyui/` |
| `minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors` | `models/diffusion_models/hyperflow` | 16.0 GB | VideoRebirthAI/hyperflow-h3-flash, `fl2va_int8_convrot_comfyui/` |

HyperFlow's model page recommends these INT8 files on RTX GPUs with 24 GB or
more. On a smaller GPU, take its GGUF files in their place, from Q8_0 at 15.2 GB
down to Q3_K_M at 7.2 GB, anywhere in `models/diffusion_models`; see
[models and quality](docs/models.md#hyperflows-other-files).

These are optional, and worth having:

| File | Folder | Size | Where to get it | For |
|---|---|---|---|---|
| `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors` | `models/latent_upscale_models` | 0.7 GB | [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) | Finals at up to 1920 pixels |
| `MMH3-RealismSlider-V1.safetensors` | `models/loras` | 3 MB | Civitai model 3014078 | Natural skin in finals |
| `taeh3.safetensors` | `models/vae_approx` | 23 MB | [OzzyGT/taeh3](https://huggingface.co/OzzyGT/taeh3) | The moving live preview |

Other H3 checkpoints work as well; see [models and quality](docs/models.md).

### 3. Make your first video

1. Open **H3 Editor** from the ComfyUI sidebar. It starts you on a first project.
2. In the **Scene** panel, choose the format: 16:9, 9:16 or 1:1.
3. Press **New clip**, write what happens in **What happens**, and press
   **Render draft**. The viewer shows the clip taking shape.
4. When the draft lands, press Space to play it. Not right? Change the text and
   render again: each render becomes another take.
5. Press **Extend forward** to carry the shot on, or add another clip.
6. Press **Render final** for the takes you keep, then **Export**. The video is
   saved in ComfyUI's output folder, under
   `h3take_projects/<project>/exports`.

## Using the editor

**Clips and takes.** Every render of a clip is a take with its own seed and the
prompt that made it. Flip between takes with `[` and `]`, keep a seed to render
variations of one result, or restore a take's prompt. Rendered clips stay on
disk: moving, trimming or splitting them never renders anything again.

**Cast and voices.** Add pictures to **Cast** and choose who is in each clip.
Write @Name in **What happens** to place them, and in **Who speaks** to use
their voice. A voice can be uploaded, or taken from a clip on the timeline.

**Joining shots.** Select a clip on the timeline and press **Extend forward** or
**Extend back** to generate what happens next or before. **Bridge to next**
generates what connects it to the following clip, and **Merge with next**
replaces the join between two clips with a generated crossing. These clips hold
39 real frames and their sound from the clips they join, so the cut does not
show. If a clip they join changes, they are marked: render them again.

**First and last frames.** Give a clip an exact first or last frame by
uploading a picture, or press **Use playhead frame**.

**Picture to a track.** A clip can hold a recorded sound, and its picture is
generated to it. Extend forward carries on with the next part of the same
track, so a long talking performance can be built from short shots.

**Your own footage.** **Import** brings in a video or a picture, fitted to the
project's format. You can trim it, and extend or bridge from it like a render.

**The Clips panel.** Click a clip to play it on its own, whether or not it is on
the timeline; click the timeline to go back to the cut. File clips in bins with
the folder button and drag clips onto them. Delete a clip with the trash button
in its panel, or with Delete in the Clips panel.

**Projects.** Projects live in `h3take_projects` in ComfyUI's output folder.
Renders run on the server, so closing the editor never loses one. Every change
is a revision you can go back to, and two editors, or an editor and an agent,
can work on one project at once.

| Key | Does |
|---|---|
| Space | Play or pause |
| Left, Right | Move one frame. Hold Shift for one second |
| Up, Down | Jump to the previous or next edit |
| Home, End | Go to the start or end |
| S | Split at the playhead |
| Delete | On the timeline, remove a clip from it, or close a selected gap. In the Clips panel, delete the clip or bin |
| Shift+Delete | Remove a clip from the timeline and close the space |
| [ , ] | Flip between takes of the selected clip |
| Ctrl+wheel, \\ | Zoom at the pointer, fit the whole scene |
| Ctrl+Z, Ctrl+Shift+Z | Undo, redo |

Press ? in the editor for this list.

## Going further

- **[Models and quality](docs/models.md):** every checkpoint the editor can
  render with, how finals are refined, and how LoRAs land.
- **[The nodes](docs/nodes.md):** H3 Take and the other nodes, for building
  your own ComfyUI workflows.
- **[For agents](skills/h3-editor/SKILL.md):** a skill that teaches an agent to
  plan shots, render, review and export through the editor's HTTP API, with
  `scripts/h3.py`, a client that needs only Python. Copy `skills/h3-editor` into
  an agent's skills folder to install it. The full API is in
  [references/api.md](skills/h3-editor/references/api.md).

## Troubleshooting

| You see | Do |
|---|---|
| A model in the Model menu says "not installed" | Its hint names the missing file. Put it in the folder from the tables above, then reload the editor |
| "ffmpeg is missing" | Install the requirements, step 1 above, and restart ComfyUI |
| "HyperFlow H3 Flash GGUF files need the gguf Python package" | The same |
| Finals come out at 768x1344 or 1344x768 | Add the latent upscaler. The Model hint says so too |
| The live preview is a still | Add `taeh3.safetensors` |
| ComfyUI runs out of memory | Try one of HyperFlow's GGUF files, which come in smaller sizes. So far the editor has been tested on one 96 GB GPU only |

## Develop

`python -m unittest discover -s tests` runs from the repo root. The writer,
LoRA and HyperFlow tests need torch and ComfyUI, with `PYTHONPATH` set to
ComfyUI's folder, and the GGUF tests need gguf too. They skip without them.

`tools/deploy.sh` copies the repo into a running container, runs the tests
there, and reloads the pack in place, so models stay loaded. Refresh the browser
afterwards. `--restart` restarts ComfyUI instead. `tools/run_take.py` queues a
real take, and `tools/compare_takes.py` measures how closely a span matches its
source.

## Credits and license

GNU General Public License v3.0, in [LICENSE](LICENSE). Parts are adapted from
ComfyUI-HyperFlow-Flash, ComfyUI-GGUF and ComfyUI-MiniMax-H3-Turbo, and keep
their own licenses; [NOTICE](NOTICE) lists them. Model files come under their
own licenses, which this pack's license does not change.
