# Models and quality

## What every render needs

| File | Folder | Size | Where to get it |
|---|---|---|---|
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders` | 15.7 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3), `text_encoders/` |
| `minimax_h3_video_vae_fp16.safetensors` | `models/vae` | 5.2 GB | Comfy-Org/MiniMax-H3, `vae/` |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae` | 0.6 GB | Comfy-Org/MiniMax-H3, `vae/` |

Then the model files of at least one checkpoint profile, below.

## Checkpoint profiles

A project renders with one profile, picked in the editor's **Model** menu and
stored in the project as `profile`. The menu marks the profiles whose files are
not installed. A new project starts on the first installed profile in this
order, so a fresh install renders with whatever it has.

A clip can render with a different profile than its project: select the clip
and pick one under **Model** in the panel on the right. Its next render uses it,
and takes keep the profile that made them. ComfyUI loads the other profile's
models when the queue reaches such a clip, which takes a while. **Render all
drafts** queues the clips that share a model one after another, so each model
is loaded once.

| Profile | Files | Steps, draft and final |
|---|---|---|
| HyperFlow H3 Flash, the default | `minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors` (16.4 GB, in `ref2va_int8_convrot_comfyui/`) and `minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors` (16.0 GB, in `fl2va_int8_convrot_comfyui/`) from [VideoRebirthAI/hyperflow-h3-flash](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash), in `models/diffusion_models/hyperflow` | 4 and 6, euler |
| MiniMax H3 Turbo | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` and `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (21 GB each) from Comfy-Org/MiniMax-H3, in `models/diffusion_models`, and the Turbo v4 LoRA `minimax_h3_turbo_v4_step600_ema.safetensors` from [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora), in `models/loras` | 6 and 6, euler |
| MiniMax H3 | the same two base files, without the LoRA | 10 and 20, res_multistep |
| Singularity v2 | `Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors` | 10 and 20, res_multistep |
| DaSiWa Hybrid v4 | its int8 ref2va file from Civitai model 2877206 | 10 and 20, res_multistep |
| DaSiWa Hybrid v4 Turbo | its turbo int8 ref2va file from the same model | 8 and 8, euler |
| DaSiWa GXS v5 Turbo | `dasiwa_minimax_h3_ref2va_v5xs_pruned_graviton_turbo_int8_row-wise_convrot_runtime_mixed.safetensors`, or another file of the same release | 6 and 8, euler |
| 10Eros Max beta 7 | `10Eros_Max_h3_8stepTURBO_beta7_fix1-native_int8_protected_edges_HQ.safetensors` from [TenStrip/10Eros-Max](https://huggingface.co/TenStrip/10Eros-Max), or another beta 7 file | 6 and 8, euler |
| PRISM (Singularity x 10Eros Max), for short single shots; speech and extensions hold better on HyperFlow | `PRISM_DMAD_PDMD_int8.safetensors` (18.9 GiB) from [OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM](https://huggingface.co/OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM), or another `PRISM_DMAD_PDMD` file | 6 and 8, euler on the Beta57 schedule |

H3 comes in two kinds. A family with an fl2va model renders prompt-only clips
on it and everything else on its ref2va model. A family with only ref2va renders
every clip on it.

### HyperFlow's other files

The default files are INT8 safetensors, which ComfyUI runs on its own INT8
kernels and which load without the gguf package. HyperFlow's model page
recommends them on RTX GPUs with 24 GB or more, and its GGUF files on smaller
ones. Keep their file names: the editor tells a HyperFlow safetensors file from
a plain H3 one by the `hyperflow` in its name.

The GGUF files come in quantizations from Q3_K_M (7.2 GB) to Q8_0, in the same
repository's `ref2va_gguf/` and `fl2va_gguf/` folders:

| File | Size |
|---|---|
| `minimax_h3_ref2va_hyperflow_flash_14b-Q8_0.gguf` | 15.2 GB |
| `minimax_h3_fl2va_hyperflow_flash_14b-Q8_0.gguf` | 14.9 GB |

Any of them works, anywhere in `models/diffusion_models`, in place of the INT8
files. The smaller ones need less memory and give up some detail. They need the
gguf package, which the pack's requirements install.

When several HyperFlow files are installed, the profile takes the INT8 file
first, then the Q8_0 file in `models/diffusion_models/hyperflow`, then the first
other file in name order. To render with another one, name it in the project, as
[below](#a-projects-own-models).

## Drafts and finals

A draft renders small: 640x352 for 16:9, 352x640 for 9:16, 480x480 for 1:1. Use
drafts to try things.

A final renders at H3's own size of about a megapixel. Its picture is then
enlarged by a learned latent upscaler, and the model redraws its last steps at
the larger size: HyperFlow its last 3 of 6, the 20-step families their last 3 of
20, and the turbo families a fixed tail from sigma 0.675. A 9:16 final comes out
at 1088x1920, a 16:9 one at 1920x1088 and a 1:1 one at 1440x1440. The sound is
the first pass's, and held frames stay as they are.

Finals also run the MMH3 Realism Slider LoRA at 1.2 (`start_sigma = 0.75`) on
both passes, for natural, un-airbrushed skin without drifting early composition
steps. Both extras are optional:

| File | Folder | Size | Where to get it | Without it |
|---|---|---|---|---|
| `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors` | `models/latent_upscale_models` | 0.7 GB | [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler), `minimax_h3_latent_upscaler_3d_conv_v1/` | Finals stay at H3's own size |
| `MMH3-Realism-V2.safetensors` | `models/loras` | 71 MB | [Civitai model 3041489](https://civitai.com/models/3041489) (falls back to `MMH3-RealismSlider-V1.safetensors` if only V1 is installed) | Finals render without it |

The editor says which ones are missing, under **Model** in the Project tab.

### Other sizes

A project can render at other sizes, with **Draft size** and **Final size** in
the Project tab. A smaller size needs less GPU memory and renders faster.

| Size | 16:9 | 9:16 | 1:1 | For |
|---|---|---|---|---|
| Small | 640x352 | 352x640 | 480x480 | Drafts, as usual, and finals |
| Medium | 960x544 | 544x960 | 768x768 | Drafts and finals |
| Large | 1344x768 | 768x1344 | 1024x1024 | Finals, without the enlarging pass |
| Full | 1920x1088 | 1088x1920 | 1440x1440 | Finals, as usual |
| Sharp | 1920x1088 | 1088x1920 | 1440x1440 | Finals, enlarged with RelayVSR instead |

A final at Large, Medium or Small is not enlarged, so it loads no upscaler and
samples once. The Realism Slider still runs on it. Takes already rendered keep
their size.

A Sharp final samples once at Large, is enlarged two times by RelayVSR (see
below) and is scaled down to the Full size. It skips the second sampling pass,
so it is quicker, edges come out crisper and skin keeps the smoother look of
the sampled size. It holds every enlarged frame in system memory, so a long
clip needs a lot of it: an eight-second clip can need more than 48 GB.

## Upscalers

**Upscale 2x**, on a clip whose take is a final, makes a new take at twice its
size. You pick the upscaler among the ones whose two files are installed:

| Upscaler | Files | Folder | Where to get them | What it does |
|---|---|---|---|---|
| RelayVSR | `relayvsr_x2_lightx2v_int8_convrot.safetensors`, `relayvsr_flash_vae_x2_fp32.safetensors` | `models/diffusion_models`, `models/vae` | [Comfy-Org/RelayVSR](https://huggingface.co/Comfy-Org/RelayVSR) | Seconds a clip, about 16 GB of GPU memory. Sharpens the edges that are there. Also what a Sharp final uses |
| SeedVR2 | `seedvr2_3b_int8_convrot.safetensors`, `seedvr2_ema_vae_fp16.safetensors` | `models/diffusion_models`, `models/vae` | [Comfy-Org/SeedVR2](https://huggingface.co/Comfy-Org/SeedVR2) | Minutes a clip and far more GPU memory. Redraws fine texture, and can change a face a little |

Files in a subfolder of those folders are found too.

Steps belong to the model, not the project: to render with other steps, add a
checkpoint with the same file under [your own checkpoints](#your-own-checkpoints).

Export joins the cut at the size of its largest picture.

## Live preview

While a clip samples, the player loops the whole clip, updated at every step.
Its sound is off at first, since it is rough until the last steps: the speaker
button in the corner of the viewer turns it on, and the editor remembers the
choice. That needs `taeh3.safetensors` (23 MB) from
[OzzyGT/taeh3](https://huggingface.co/OzzyGT/taeh3) in `models/vae_approx`.
Without it the preview is a rough still.

## LoRAs

The H3 LoRA node applies a LoRA trained on MiniMax H3.

- On int8 and fp8 weights it adds the LoRA at run time, so small updates are not
  rounded away.
- On the pruned base files it adds the LoRA's time-conditioning part at run time
  too, from a small bundled grid. A LoRA trained on a pruned base file already
  fits it, and loads as it is.
- A layer sized for another model is left out, and ComfyUI's log says how many.
- HyperFlow H3 Flash prunes some of H3's blocks: its ref2va model dropped block 9
  and its fl2va model blocks 6, 21 and 22. The node moves a LoRA's later blocks
  down to match, so they land on the layers they were trained for, and attaches
  at run time on HyperFlow's factorized low-rank and split-QKV layers too.
- Optional `start_sigma` and `end_sigma` inputs let a LoRA run only on part of
  the schedule, such as `start_sigma = 0.68` to keep early composition steps
  untouched and apply the LoRA on detail steps.
- Several H3 LoRA nodes in a row stack.

In the editor, a project has its own LoRAs. The Project tab of the panel on the
right has a LoRAs section: search `models/loras` to add one, drag its slider
to set its strength from -2 to 2, untick it to render without it, or remove it.
Every new render of the project samples with the ones that are ticked, in the
order listed, on drafts and finals alike. The same section has the Realism
Slider's strength on finals, from 0 to 3; it starts at 1.2.

Each LoRA has a button that says when it joins the render, **Starts at once** by
default. It opens a bar that stands for the render, from its first step on the
left to its last on the right, lit where the LoRA works. **At once** runs it on
every step. **After framing** (0.75) leaves the first steps to the model alone,
so the framing and pose stay as they would be without the LoRA. **Details
only** (0.4) lets it join for the last steps. Drag the bar or type a start for
anything between. The button then shows the start, for example **Starts at
0.75**.

A clip can have LoRAs of its own: in its Shot tab, open **Model and LoRAs** and
press **Edit** beside LoRAs. Only that clip samples with them, after the project's. Adding one the
project also has gives it another strength in that clip, and unticking it
there leaves it out of that clip.

The list you pick a LoRA from has a star beside each file. Starred files list
first, in every project on this browser.

## Your own checkpoints

Any other MiniMax H3 checkpoint, such as a community fine-tune, renders in the
editor too. **Models and their settings** in the Project tab of the panel on the
right opens the Models dialog, which lists the built-in models, the
project's own checkpoints and your presets. **Add a checkpoint** there picks a
file from `models/diffusion_models`; then set

- the built-in model it is closest to. It starts from that model's settings and
  the way its finals are refined;
- whether to add that model's LoRA, when it has one. Untick it for a checkpoint
  that already has the turbo merged in, so the Turbo LoRA is not applied twice;
- steps for drafts and for finals, the sampler and the scheduler, where its
  author recommends others;
- a second file for clips that are only a prompt, if the checkpoint comes with
  an fl2va model. Without one, those clips render on the same file.

**Save as a preset** keeps a checkpoint for your other projects: it then shows
in the Models dialog of each of them, and **Add to project** there copies it
into that project. A project renders on its own copy, so changing or
deleting a preset leaves the projects that already use it as they are. Presets
live in `checkpoints.json` in the `h3take_projects` folder of ComfyUI's output.

The checkpoint then appears in the **Model** menu, for the project or for one
clip. It is stored in the project as `checkpoints`. A file with hyperflow in its
name, and any `.gguf`, loads as HyperFlow and samples on HyperFlow's schedule.

## A smaller text encoder

H3's text encoder is a 32B Qwen3-VL, 15.7 GB in its smallest file. A Qwen3-VL 4B or 8B
can stand in for it through a projection: a small file, fitted once against the 32B,
that turns the small model's output into what H3 reads. The 4B with its projection
takes about 4.6 GB. This is the method of
[ComfyUI-ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj), built into the pack,
so that node is not needed.

| File | Folder | Size | Where to get it |
|---|---|---|---|
| `qwen3vl_4b_int8_convrot.safetensors`, or another precision of the 4B | `models/text_encoders` | 4.5 GB | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2), `text_encoders/` |
| `mmh3-4b-ClipProj-v3.1.safetensors` | `models/clip_projections` | 26 MB | [NicoLab28/ClipProj-MiniMax-H3](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3) |

For an 8B encoder, take `mmh3-8b-ClipProj-v3.1.safetensors` instead. Each projection
works with its own size only. The encoder must be a Qwen3-VL: a text-only Qwen3 of
the same size loads but ignores the prompt.

With both installed, the 4B or 8B shows in the **Text encoder** menu, marked as
going through a projection: in the Project tab, press **Edit** beside Model files,
or open Models and their settings. Pick it there. With more than one projection
for its size, a **Projection** menu picks between them; by itself the render takes
`v3.1`, then `v3.1-mlp`, then the older ones. The `-mlp` files (about 500 MB) add a
small network on top of the matrix. Without a projection, a project that names a
small encoder renders with the 32B.

What it costs: the small model knows less than the 32B. Named people and speech in
languages other than English suffer most; ClipProj's own measurements are in its
README. In a test of two draft shots, a prompt alone and a cast member speaking, the
4B followed the prompt and kept the face as the 32B did, but said "breakfast" not
quite right where the 32B said it cleanly. The bf16 4B and the `v3.1-mlp` projection
said it no better, so the small `v3.1` stays the default. Use the small encoder when
the 32B does not fit, not to save time, and listen to every line it speaks.

It saves memory only where the encoder is the problem. ComfyUI already moves the 32B
off the card before sampling, so this helps most when the 32B does not fit at all, or
when memory outside the card is short.

The node behind it, **H3 Small Text Encoder**, works in your own workflows too:
connect its `clip` wherever the 32B's goes.

## A project's own models

A project can still name its own files under `models` in its project file, which
wins over its profile: `unet` for the ref2va file and `unet_text` for the fl2va
one. A project that names only `unet` renders every clip on it. A `.gguf` model,
or a `.safetensors` one with `hyperflow` in its file name, loads as HyperFlow.
Any other model loads with ComfyUI's plain loader and scheduler.

## Licenses

Model files come under their own licenses, which this pack's license does not
change. Read each model's page before you use it. HyperFlow H3 Flash, for
example, is a derivative of MiniMax H3 under the MiniMax H3 Community License
Agreement, which limits where it may be used. ClipProj's projections are derived
from the activations of both Qwen3-VL and MiniMax H3, and their author says their
legal status is unclear.
