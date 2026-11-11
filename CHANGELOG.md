# Changelog

What changed in each release of H3 Editor, newest first. Versions follow
[semantic versioning](https://semver.org): while the pack is at 0.x, a new minor
version can change how projects or the API behave, and the entry says so.
Each release is a tag on GitHub, `v` followed by the version.

## Unreleased

### Added

- **Cut scores.** Where two rendered clips touch on the timeline, the editor
  measures the cut from the pictures on both sides and gives it a score from 0
  to 100. A cut that carries on smoothly or changes to a new shot scores high.
  A jump cut, the same shot with the subject popped to another pose, scores
  low and gets a mark on the timeline with the reason. The same takes and trims
  always give the same score. For agents: `GET /project/{name}/cuts`.
- **Rate a range with Gemini.** Mark a range of the cut (I and O at the
  playhead, or Shift-drag on the ruler) and press Rate: that part is exported
  small, sent to Google's Gemini Decision model with your own API key, and
  comes back rated against a rubric. Six are checks, each read as how likely
  the good answer is: the same person from start to end, matching the cast
  picture (which is sent along), a sharp face, natural light, a background
  that stays put, and natural movement. One is a scale: how well it holds
  attention. A last row, smooth cuts, is the editor's own cut measurement for
  that range, because the model sees about a frame a second and cannot notice
  a jump cut. Each row has a bar and the description of where it
  landed. Ratings are kept with the project, so a range can be
  compared with how it scored before. This is optional and off until a key is
  saved: the key is entered in the Rate dialog and kept with ComfyUI, never in
  a project, and the video is deleted at Google once rated. For agents:
  `POST /project/{name}/rate`, with a rubric of their own if they want one.
- **VFX clips: clips drawn by code.** A new VFX lane above the picture holds
  clips whose picture is drawn by a small JavaScript module instead of the
  model. The module draws each frame on a canvas, in 2D or with WebGL and
  three.js, and gets the picture below it, so it can put a title over a shot,
  build a transition across a cut, or composite with the shot itself. The
  viewer draws it live as you scrub, type or move one of the controls the code
  declares. Render bakes every frame over the exact picture below into a take,
  in seconds and with no GPU model, and the export lays that take over the cut.
  A clip can also be handed named pictures to draw, such as a logo or a
  screenshot, and data to read, such as the word timings of a narration for
  captions. A take is marked out of date when the code, a control, a
  picture, the length or anything under the clip changes. The code runs in a worker inside a sandboxed frame: it has
  no page to navigate, no network and no access to the editor, and one that
  hangs is stopped. three.js 0.186 ships with the pack. For agents:
  `add_effect`, `update_effect`, `remove_effect` and `render_effect`. The
  code runs in a browser, so a render an agent asks for is done by an open
  editor page, and waits until there is one.
- **Slip.** A clip that shows only part of its take can show another part
  without moving or changing length, so a shot cut to a music cue can be
  auditioned against a longer take with the rest of the edit untouched.
  Alt-drag the clip, press `,` and `.`, or drag the new window under Part of
  the take in the Timeline tab, which also says how many frames are unused
  before and after. A clip with more take on either side has a dotted edge.
  For agents: `update_item` with `in` and `out` moved by the same amount.

### Changed

- Imported footage and pictures are kept at the size the project's finals
  come out at, 1920x1088 for 16:9, where they used to be made 1344x768. A
  1080p recording now stays as sharp as the shots beside it.

### Fixed

- **Extensions and bridges no longer lose quality at each join.** A linked
  clip used to read the frames it holds from its source's video file and
  encode them again. Each time, the picture came back 1.5 to 3% darker and
  softer, so a chain of extensions stepped darker at every join and the face
  drifted within half a minute. A take now also saves its first and last 39
  frames as the latents it was sampled in, in a `.hold.safetensors` file of
  about 1 MB beside its video, and a clip that extends or bridges it holds
  those as they are. Measured on one shot extended ten times to 64 seconds:
  the brightness step at the joins is gone, and the face keeps 75% or more of
  its detail where it used to fall to 33%. Takes rendered before this have no
  such file and are held from their video as before, so render a source again
  to get the benefit. A source trimmed at the held end is also held from its
  video.
- The viewer changed the picture's height between a draft and a final, because
  the two sizes have slightly different proportions (640x352 and 1920x1088).
  The viewer now shows the cut in one shape, the one the export uses, and
  every clip fills it.

## 0.4.1, 2026-11-11

### Added

- **Draft size and Final size**, in the Project tab under Format. A draft can
  render at a medium size, 960x544 for 16:9, instead of the small one. A final
  can skip the enlarging pass and stay at 1344x768, or render at the medium or
  the small size. A smaller size needs less GPU memory and renders faster. A
  project that picks none renders as before. For agents: `set_project` takes
  `sizes`, and `GET /projects` lists the choices.

### Fixed

- A LoRA trained on one of the pruned H3 base files failed every render on a
  model built on them, such as MiniMax H3 or MiniMax H3 Turbo, with "mat1 and
  mat2 shapes cannot be multiplied (32x8 and 2688x1)". The H3 LoRA node took
  its time-conditioning layers for ones trained on a full H3. They now load
  like the LoRA's other layers.
- A LoRA layer sized for another model is left out, and the log says how many,
  instead of failing the render.

### Upgrading

Restart ComfyUI after updating. Projects made with 0.4.0 open as they are.

## 0.4.0, 2026-11-11

### Changed

- **The panel on the right has tabs.** A clip's settings are split into Shot,
  Takes and Timeline, with the render buttons under each, and the project's
  settings (look, format, model, LoRAs) are always one click away in a Project
  tab instead of showing only when nothing is selected. Esc selects nothing.
- **A tighter, quieter interface.** Controls are about 30 pixels tall and text
  is 13 pixels, so more of the cut and of a clip's settings fit on screen.
  The timeline's edit tools are icons with a tip that names them and their
  key, and Extend back, Extend forward, Bridge to next and Merge with next
  are in one **Generate** menu that says what each does. Menus, sliders and
  dialogs are drawn in the editor's own style.
- **The Clips and Cast headers stay on one line in a narrow panel.** New bin
  and Import are icons, and Add voice, Add reference and Add RefMod are in one
  **Add** menu.
- **The sound follows its picture while you drag or trim a clip**, instead of
  jumping there when you let go.
- **A project opens showing its whole cut**, and the panels and the timeline
  give way when the window gets smaller, so the viewer keeps its room.
- **Messages stack and can be closed.** Deleting a clip, a bin or a take, or
  removing a cast member, shows an Undo button in its message.
- **The header says what is rendering.** It names the clip, how far it is and
  how many are waiting; press it to go to that clip.
- **The clock shows minutes and seconds first**, with the frames smaller after
  them.

### Added

- **Extra sound lanes.** **Add sound** lays music, narration or any other sound
  file under the cut, on up to four lanes below the clips' own sound. Move it,
  trim it, change its lane and set its volume; it plays in step with the cut
  and the export mixes it in. For agents: the `add_sound`, `update_sound` and
  `remove_sound` operations.
- **Rename, duplicate and delete a project.** A menu beside the project list
  renames the open project, copies it with its clips, takes and pictures into
  a folder of its own, or deletes it. A deleted project is moved to a `.trash`
  folder inside `h3take_projects` and never erased. For agents: `POST
  /project/{name}/rename`, `POST /project/{name}/duplicate` and `DELETE
  /project/{name}`.
- **Scrub a take before choosing it.** Run the pointer across a take's picture
  in the Takes tab to look through it, as in the Clips panel.
- **Drag a cast member into the text.** Drop a cast card on What happens, or
  on any other field of a clip, and its @Name lands there. A member with a
  picture or a RefMod also joins that clip's cast.
- **Make a RefMod from a cast member.** With the ComfyUI-MiniMaxH3Mod pack
  installed, a cast member's RefMod section has **Make a RefMod from these
  pictures**. It saves one from all the member's pictures and the member
  carries it from then on. For agents: the `make_refmod` operation.
- **LoRAs per clip.** A clip's Shot tab has **LoRAs for this clip**: LoRAs only
  that clip samples with, after the project's. Adding one the project also has
  changes its strength for that clip, or leaves it out. For agents: `loras` in
  a clip's `fields`.
- **When a LoRA joins.** Each LoRA has a **Start** slider: at 1 it runs on
  every step, lower leaves the first steps alone so framing and pose stay
  put. For agents: `start_sigma` on a LoRA.
- **Favorite LoRAs.** Star a file in the LoRA list and it lists first from then
  on.
- **Batches of renders are grouped by model.** Render all drafts, and a batch
  of renders sent by an agent, queue the clips that load the same model file
  one after another, so a project that mixes models, or prompt-only clips with
  clips that have pictures, loads each model once instead of going back and
  forth.
- **A bridge button in an empty space.** A gap between two rendered clips on
  the timeline shows a plus: press it to generate the clip that connects them.
- **Timeline clips work from the keyboard.** Tab reaches each clip and Enter
  selects it, and a screen reader hears its name, state and place.
- **Two more built-in models.** DaSiWa GXS v5 Turbo and 10Eros Max beta 7, both
  at 6 steps for drafts and 8 for finals. Each shows under **Model** once its
  file is in `models/diffusion_models`.
- **A Models dialog.** **Models and their settings** in the Project tab lists
  the built-in models, the project's own checkpoints and your presets in one
  place: pick the one the project renders with, add a checkpoint file, and edit
  its steps, sampler and scheduler there. It replaces the **Your checkpoints**
  section.
- **A checkpoint can leave its family's LoRA out.** A checkpoint set up on
  MiniMax H3 Turbo's settings has a tick box for the Turbo LoRA, for files that
  already have the turbo merged in. For agents: `"lora": false` on a row of
  `checkpoints`.

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
