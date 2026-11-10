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
models when the queue reaches such a clip, which takes a while.

| Profile | Files | Steps, draft and final |
|---|---|---|
| HyperFlow H3 Flash, the default | `minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors` (16.4 GB, in `ref2va_int8_convrot_comfyui/`) and `minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors` (16.0 GB, in `fl2va_int8_convrot_comfyui/`) from [VideoRebirthAI/hyperflow-h3-flash](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash), in `models/diffusion_models/hyperflow` | 4 and 6, euler |
| MiniMax H3 Turbo | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` and `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (21 GB each) from Comfy-Org/MiniMax-H3, in `models/diffusion_models`, and the Turbo v4 LoRA `minimax_h3_turbo_v4_step600_ema.safetensors` from [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora), in `models/loras` | 6 and 6, euler |
| MiniMax H3 | the same two base files, without the LoRA | 10 and 20, res_multistep |
| Singularity v2 | `Minimax-h3_Singularity_ref2va_v2_pruned_int8.safetensors` | 10 and 20, res_multistep |
| DaSiWa Hybrid v4 | its int8 ref2va file from Civitai model 2877206 | 10 and 20, res_multistep |
| DaSiWa Hybrid v4 Turbo | its turbo int8 ref2va file from the same model | 8 and 8, euler |

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

Finals also run the MMH3 Realism Slider LoRA at 1.5 on both passes, for a matte,
less made-up skin. Both extras are optional:

| File | Folder | Size | Where to get it | Without it |
|---|---|---|---|---|
| `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors` | `models/latent_upscale_models` | 0.7 GB | [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler), `minimax_h3_latent_upscaler_3d_conv_v1/` | Finals stay at H3's own size |
| `MMH3-RealismSlider-V1.safetensors` | `models/loras` | 3 MB | Civitai model 3014078 (needs a Civitai login) | Finals render without it |

The editor says which ones are missing, under **Model** in the Scene panel.

Export joins the cut at the size of its largest picture.

## Live preview

While a clip samples, the player loops the whole clip with its sound, updated at
every step. That needs `taeh3.safetensors` (23 MB) from
[OzzyGT/taeh3](https://huggingface.co/OzzyGT/taeh3) in `models/vae_approx`.
Without it the preview is a rough still.

## LoRAs

The H3 LoRA node applies a LoRA trained on MiniMax H3.

- On int8 and fp8 weights it adds the LoRA at run time, so small updates are not
  rounded away.
- On the pruned base files it adds the LoRA's time-conditioning part at run time
  too, from a small bundled grid.
- HyperFlow H3 Flash prunes some of H3's blocks: its ref2va model dropped block 9
  and its fl2va model blocks 6, 21 and 22. The node moves a LoRA's later blocks
  down to match, so they land on the layers they were trained for. A LoRA
  attaches only where HyperFlow keeps a layer whole, and ComfyUI skips the rest
  with a warning: on fl2va, the Realism Slider's last `fc2` is one.
- Several H3 LoRA nodes in a row stack.

In the editor, a project has its own LoRAs. With no clip selected, the panel on
the right has a LoRAs section: search `models/loras` to add one, drag its slider
to set its strength from -2 to 2, untick it to render without it, or remove it.
Every new render of the project samples with the ones that are ticked, in the
order listed, on drafts and finals alike. The same section has the Realism
Slider's strength on finals, from 0 to 3; it starts at 1.5.

## Your own checkpoints

Any other MiniMax H3 checkpoint, such as a community fine-tune, renders in the
editor too. With no clip selected, the panel on the right has **Your
checkpoints**: pick the file from `models/diffusion_models`, then set

- the built-in model it is closest to. It starts from that model's settings and
  keeps its LoRA, if it has one, and the way its finals are refined;
- steps for drafts and for finals, the sampler and the scheduler, where its
  author recommends others;
- a second file for clips that are only a prompt, if the checkpoint comes with
  an fl2va model. Without one, those clips render on the same file.

**Save as a preset for every project** keeps a checkpoint for your other
projects: it then shows under **Model** in each of them, and picking it there
copies it into that project. A project renders on its own copy, so changing or
deleting a preset leaves the projects that already use it as they are. Presets
live in `checkpoints.json` in the `h3take_projects` folder of ComfyUI's output.

The checkpoint then appears in the **Model** menu, for the project or for one
clip. It is stored in the project as `checkpoints`. A file with hyperflow in its
name, and any `.gguf`, loads as HyperFlow and samples on HyperFlow's schedule.

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
Agreement, which limits where it may be used.
