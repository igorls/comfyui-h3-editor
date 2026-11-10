# Changelog

What changed in each release of H3 Editor, newest first. Versions follow
[semantic versioning](https://semver.org): while the pack is at 0.x, a new minor
version can change how projects or the API behave, and the entry says so.
Each release is a tag on GitHub, `v` followed by the version.

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
