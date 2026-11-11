# Changelog

What changed in each release of H3 Editor, newest first. Versions follow
[semantic versioning](https://semver.org): while the pack is at 0.x, a new minor
version can change how projects or the API behave, and the entry says so.
Each release is a tag on GitHub, `v` followed by the version.

## 0.3.0, 2026-11-11

### Added

- **Several pictures per cast member.** A cast member takes up to 8 pictures:
  other angles, other light, the whole figure. A shot uses up to 9 pictures in
  all, shared between its cast, and names each member's pictures in the prompt.
  For agents, `add_cast` and `update_cast` take `pictures`.
- **A choice of text encoder and video VAE.** The editor now renders with
  whichever precision of H3's text encoder and VAEs is installed (nvfp4, int8,
  fp16 or bf16), instead of needing one exact file of each. With more than one
  installed, the Project section has a menu for each. For agents, `set_project`
  takes `text_encoder`, `vae` and `audio_vae`.
- **RefMods on cast members, when the RefMod pack is installed.** A cast member
  can carry a RefMod made with the ComfyUI-MiniMaxH3Mod custom node pack, beside
  its pictures or in place of them, with a strength slider. The editor only
  loads them: without that pack nothing changes. For agents, `add_cast` and
  `update_cast` take `refmod`.
- **Realism Slider V2 and sigma gating on `H3Lora`.** Finals now default to
  `MMH3-Realism-V2.safetensors` at strength `1.2` with `start_sigma = 0.75`
  (automatically falling back to `MMH3-RealismSlider-V1.safetensors` if only V1
  is installed). `H3Lora` adds `start_sigma` and `end_sigma` inputs so LoRAs can
  target specific noise windows.
- **A New project dialog.** New project asks for the name and the format in a
  dialog of the editor's own, in place of the browser's prompt box, which does
  not open in every browser.
- **An export dialog.** Export now opens a dialog that says what the video will
  hold (length, clips, drafts, gaps) and lets you set the file name, the size
  and the quality, and download the result. For agents, `POST
  /project/{name}/export` takes `scale`, `quality` and `name`, and `plan` to
  ask what an export would hold.
- **Draft and Final on the timeline.** Each clip on the timeline says whether
  its take is a draft or a final.
- **A LoRA picker that copes with hundreds of files.** Add a LoRA opens a
  dialog with a search box and the whole list.
- **Resizable side panels.** Drag the edge of the clips panel or the inspector
  to make it wider or narrower; double-click the edge to reset it. The editor
  remembers the widths in your browser.

### Fixed

- `H3Lora` now applies runtime float deltas on INT8 fused SwiGLU `mlp.fc2`
  layers and HyperFlow's factorized `LowRankLinear` / `SplitQKV` modules instead
  of rounding small deltas away or skipping factorized weights.
- Playing the cut could stick at a cut for a second or two, flashing between
  the two clips. It happened where the next clip starts on a frame whose time
  is not an exact fraction of a second, such as frame 31.
- The next clip now starts a moment early and silent, so it is already running
  at the cut. Before, each cut held the picture and the sound for about a tenth
  of a second.
- The viewer went empty when the cut played to its end. It now holds the last
  frame.
- Dragging along the timeline ruler selected text all over the page.
- A LoRA file already renumbered for one HyperFlow model was renumbered a second
  time by `H3Lora` and landed on the wrong layers. The node now reads the file's
  `hyperflow.dropped_blocks` metadata, leaves such a file as it is on the model
  it was made for, and says so in the log when it is loaded on another model.
- Renders failed with "Required input is missing" on H3 Live Preview when
  another custom node pack registered a node with the same name. The editor
  now renders with ids only this pack uses.

### Upgrading

Restart ComfyUI after updating: this release registers new node ids, which a
running server does not pick up. Projects made with 0.2.0 open as they are.
Finals now use `MMH3-Realism-V2.safetensors` when it is in `models/loras`, and
fall back to `MMH3-RealismSlider-V1.safetensors` when only that one is there.

## 0.2.0, 2026-11-10

### Added

- **A project's own LoRAs.** Add LoRAs trained on MiniMax H3 to a project and
  set each one's strength, from -2 to 2, with a slider. Every render of the
  project uses the ones that are switched on, stacked in order. The Realism
  Slider's strength on finals is a slider too.
- **A model per clip.** A clip can render on a different model than the rest of
  its project. Takes record the model, steps and sampler that made them.
- **Your own checkpoints.** Render with any MiniMax H3 checkpoint file: pick
  the built-in model it is closest to, and set its steps, sampler and
  scheduler. It then shows in the Model menus.
- **Checkpoint presets.** Save a checkpoint once and every project offers it.
- For agents: `set_project` takes `loras`, `realism` and `checkpoints`,
  `update_clip` takes `profile`, and `GET /presets` and `PUT /presets` read and
  replace the presets. `GET /projects` also lists the LoRA and checkpoint
  files, the samplers and the schedulers.

### Fixed

- The H3 LoRA node skipped every layer of LoRAs whose keys are written in the
  trainers' `lora_unet_` form, so those LoRAs had no effect. They now load.
- In the editor, the first edit typed into a panel after a save could be lost.

### Upgrading

Restart ComfyUI after updating: the presets use new routes on its server.
Projects made with 0.1.0 open as they are.

## 0.1.0, 2026-11-10

The first public version: the timeline editor for MiniMax H3 inside ComfyUI,
the H3 Take node and the pack's other nodes, HyperFlow H3 Flash as the default
model with five more built in, finals refined at up to 1920 pixels, cast and
voices, extensions and bridges, picture generated to a recorded track, and the
agent skill with its HTTP API.
