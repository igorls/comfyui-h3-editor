# Models and quality

## Get the models from the editor

On a ComfyUI that has no H3 model files yet, the editor opens **Get the models**
when it starts. It names the files the default model, HyperFlow H3 Flash, needs
to render, with their sizes and licenses: about 54.6 GB in all. **Download**
fetches them from Hugging Face and puts each one in its folder under ComfyUI's
`models`, under the names in the tables below. It downloads the latent upscaler
for finals too.

- Each file comes in under its name with `.part` added, and gets its own name
  only once its size and sha256 match the file published in its repository. A
  file that does not match is deleted.
- A file already in place is never written to or replaced. Another precision of
  the same model counts as in place, such as the int8 text encoder or a HyperFlow
  GGUF: the panel checks what you have the same way a render picks its file.
- **Stop** keeps what came, and **Download** carries on from there. So does a
  broken connection, a few times, on its own. Closing the panel does not stop it.
- It checks first that the disks have room, with 1 GB to spare.
- It fetches only the files of its own list, from the commit of each repository
  that list names: nothing in a request can point it at another address.

Each model comes under its own license, linked beside it in the panel; read them
before you download (see [licenses](#licenses)). The MMH3 Realism Slider LoRA
is on Civitai, which asks you to sign in to download it, so the panel links to
its page: put the file in `models/loras` yourself. Finals render without it
until it is there.

If you close the panel, the Project tab has **Get the models** under Model for
as long as a file is missing. Other checkpoints, upscalers, the live preview and
the small text encoder are not on the list: get them as their sections below say.

Agents and scripts can use the same routes: `GET /h3take/api/models/setup` says
what is missing and how large it is, and the download in progress;
`POST /h3take/api/models/setup/download` starts it, and
`POST /h3take/api/models/setup/cancel` stops it. Neither takes a body.

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

Finals also run the MMH3 Realism Slider LoRA at 1.2 (`start_sigma = 0.93`) on
both passes, for natural, un-airbrushed skin without drifting early composition
steps: with HyperFlow's six steps it leaves the first two to the model alone.
Extensions, bridges and merges run it at half that strength, 0.6: they hold
frames of a final that went through it already and carry that look on. At full
strength on top, a chorus sung as a head clip and four extensions came out 46%
harsher by the fifth clip, with deeper lines on the face each time; with none,
the skin turned glossy and waxy along the chain. Half keeps its texture without
piling it up. Both extras are optional:

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

**Your own size**, the last choice in both menus, takes any width and height
from 256 to 2048 pixels, for a draft or a final. Each side is a multiple of 32,
which H3's latent needs, and what you type is moved to the nearest one: 1280 by
720 becomes 1280 by 736. The editor starts you at 1024x576 (576x1024 for a
vertical project), about 0.6 megapixels, between Medium and Large. A final at a
size of your own comes out as sampled, like Large. The size can have another
shape than the project's format, such as 1344x576 for a wider picture: the
editor says so under the boxes, and the cut and the export take the shape of
the largest picture on the timeline, with every other clip filling it. H3 was
trained near one megapixel, so far above or below that the picture can suffer.

Every render is 24 frames a second, whatever its size: H3 makes picture and
sound together at that rate, and it has no setting for another.

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
every step. **After framing** (0.93) leaves the first steps to the model alone,
so the framing and pose stay as they would be without the LoRA. **Details
only** (0.7) lets it join for the last steps. Drag the bar or type a start for
anything between. The button then shows the start, for example **Starts at
0.93**. The numbers sit close to 1 because H3 spends most of a render there:
0.93 is about halfway through, and 0.75 already four fifths.

A clip can have LoRAs of its own: in its Shot tab, open **Model and LoRAs** and
press **Edit** beside LoRAs. Only that clip samples with them, after the project's. Adding one the
project also has gives it another strength in that clip, and unticking it
there leaves it out of that clip.

The list you pick a LoRA from has a star beside each file. Starred files list
first, in every project on this browser.

## Model patches

Other packs ship nodes that change how the model runs without being a model
themselves: sparse or low-memory attention, a cache that skips steps, a
compile node. In a workflow you would put one between the loader and the
sampler. The editor can do the same for every render of a project.

The Project tab of the panel on the right has a **Model patches** section.
**Add a patch** opens a list of the nodes installed in your ComfyUI that take
one model and plain values and return one model; the ones whose name or
category mentions H3, MiniMax or attention come first, and typing searches by
name, pack and category. A patch you add gets a row with its node's own
settings, drawn from the node's definition: numbers with their range, switches,
menus and text. The button under its name shows what you changed from the
node's defaults and folds the settings away; **Reset** goes back to the
defaults. Turn on the explanations (the question mark at the top of the panel)
to read what the node says about itself and about each setting.

Every new render puts the ticked patches on the model in the order listed,
after the LoRAs, on drafts and on both passes of a final. The arrows move a
patch up or down, the tick box renders without it, and the bin removes it. A
project takes up to 8. A patch whose pack is not installed shows **Not
installed** and is left out of renders, so a project still opens and renders
on a machine without that pack. A take records the patches it was made with.

A project opened from a zip comes with its patches unticked, whoever made it.
A patch runs a node of one of your packs with the settings the project gives
it, so look at what each one is before you tick it.

The editor does not know what a patch does, and no pack is needed for the
editor itself. It cannot tell you whether a patch helps or harms: some speed a
render up at a cost to the picture, some only save memory, and some do not
work with every checkpoint. Render the same clip with the same seed with the
patch and without it, compare how long each took, and look at both. Compare
at the size you mean to use it: a patch that costs a fixed time at every step
can make a short draft slower and a long final much faster, and the tick box
turns it off for the drafts. For clips that join, `h3.py cuts` scores the cuts
of each version. If a render is
refused, the message names the patch and the setting that its node does not
accept.

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

## Lip sync check

The editor can say whether the lips of a clip follow its sound: every clip with
words or a track, over the part the timeline shows. It uses SyncNet, the
network lip-sync research measures with, and lists a clip under **Checks** when
its lips run ahead of the sound, behind it, or do not follow it at all. It needs
two files in `models/syncnet`, which the pack does not download for you:

| File | Size | Where to get it | Terms |
|---|---|---|---|
| `syncnet_v2.model` | 54 MB | `https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model`, from the Visual Geometry Group, University of Oxford | Offered for research purposes under Creative Commons Attribution 4.0 |
| `face_detection_yunet_2023mar.onnx` | 0.2 MB | [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT |

Without them nothing is measured and nothing is listed. It runs on the CPU, one
clip at a time, so the GPU stays free for renders, and each take is measured
once.

What it reports, per clip: the offset in frames between the lips and the sound
(negative when the lips come first), and how clearly they match. Talking video
of real people reads about 6 to 9. In tests, every HyperFlow take of a singer
held to a song read in sync, at 2 to 5; a cut whose picture and sound had slipped
apart by a third of a second read 8 frames early. Within a frame or two is in
sync. When a clip is listed, render it again on another seed and keep the take
that reads in sync; if a whole run of clips is off by the same amount, look at
how they were cut together before you render anything. The measure is a judge,
not a fix: it sees the face, not what the words are. A face turned away, a
microphone over the mouth, or a face too small in the frame gives a low
confidence too.

## Place check

A generated clip can hold the face and still lose the room: a keyboard behind the
singer turns into a monitor, a wall into a window. The editor checks the
background of every generated clip on the timeline, the person taken out, and
lists under **Checks** a clip where it changes, with the moment. An extension is
also held against the first frame of the shot it continues, so a chain that has
drifted into another room is caught where it shows. It needs one file in
`models/matting`, which the pack does not download for you:

| File | Size | Where to get it | Terms |
|---|---|---|---|
| `modnet.onnx` | 26 MB | `onnx/model.onnx` from [Xenova/modnet](https://huggingface.co/Xenova/modnet) on Hugging Face, saved as `modnet.onnx`. MODNet is by Ke et al., [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet) | Apache 2.0 |

Without it nothing is measured and nothing is listed. OpenCV 5 runs it on the
CPU, one clip at a time; a 20 s take takes about 4 s. OpenCV 4 cannot read
this file: there the check uses `onnxruntime` when it is installed
(`pip install onnxruntime` in ComfyUI's Python), and otherwise stays off and
says so in the cuts report.

How it works: every second, the background two seconds earlier and at the start
of the shot is compared with the background now, at the scale of objects rather
than grain. The camera's move is solved on the background alone, so a pan or a
push is followed rather than counted as change. What it reports is the share of
the background that is no longer what it was. In tests, a held camera on an
unchanged room read 0% over 20 and 30 s, and so did a slow push from a wide
shot into a close-up; a 20 s take sampled in independent windows, whose room
changed under the singer, read 50%. 20% and more is listed.

What it cannot see: anything on the person, such as clothes that change, and
too little background (an extreme close-up, or a face that fills the frame) is
not judged. While the camera zooms in or out, a moment is compared only with
frames framed within 8% of it, so a shot that pushes and pulls all the way
through is barely judged. A camera that travels far, such as walking through a
house, shows a new background by design, and may be listed.

## Long takes

A shot that has to stay one continuous moment, such as a singer through a whole
chorus, is best made as one clip up to about 20 s rather than a clip and
extensions. Nothing in it is conditioned on a finished clip, so nothing piles up:
in tests a 20.75 s take held the room, the skin and the lips from first frame to
last, where a chain of the same length drifted (see Place check). The first pass
draws the whole shot at once. A final longer than 10.4 s refines in context
windows of 124 frames that share 39, which holds the picture the first pass made:
on a 20.75 s take at 1088 x 1920 the result matched a whole-take refine to the
pore, and the final took 15 minutes instead of 23. Refined whole, that take held
50 GB of GPU memory at its peak. This needs ComfyUI's `MiniMaxH3ContextWindows` node; a
ComfyUI without it refines the whole take at once.

## Licenses

Model files come under their own licenses, which this pack's license does not
change. Read each model's page before you use it. HyperFlow H3 Flash, for
example, is a derivative of MiniMax H3 under the MiniMax H3 Community License
Agreement, which limits where it may be used. ClipProj's projections are derived
from the activations of both Qwen3-VL and MiniMax H3, and their author says their
legal status is unclear.
