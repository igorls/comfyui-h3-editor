# Changelog

What changed in each release of H3 Editor, newest first. Versions follow
[semantic versioning](https://semver.org): while the pack is at 0.x, a new minor
version can change how projects or the API behave, and the entry says so.
Each release is a tag on GitHub, `v` followed by the version.

## Unreleased

### Added

- **A smaller text encoder.** A Qwen3-VL 4B or 8B can stand in for H3's 32B
  text encoder through a projection from ClipProj, built into the pack: about
  4.6 GB instead of 15.7 GB with the 4B. Put the encoder in
  `models/text_encoders` and its projection in `models/clip_projections`, then
  pick it under **Text encoder** in the Project tab, and under **Projection**
  when more than one fits it. It knows less than the
  32B, mostly about named people and speech in languages other than English.
  The **H3 Small Text Encoder** node does the same in your own workflows. See
  [models and quality](docs/models.md#a-smaller-text-encoder).

## 0.6.1, 2026-11-13

### Fixed

- **Opening a project zip cannot fill the disk.** The upload is refused
  before it starts when its size will not fit twice (the zip and the project
  it opens into) with 2 GB to spare, and stops at that size when the browser
  does not say how big it is. The answer says how much room there is.

## 0.6.0, 2026-11-13

### Added

- **Notes on frames.** Stop on a frame and press **N**, or **Note** under the
  viewer, then pin a spot, drag a box or draw on the picture and write what
  should change. With a range marked by I and O, the note covers the range.
  A note stays on its frame when the clip moves or is trimmed, and says when
  the clip has a newer take than the one it was written on. Notes are marked
  on the ruler and listed in the Project tab, where they are resolved,
  reopened or deleted. For agents: the summary lists them, `h3.py notes`
  saves each one's frame with the drawing on it, and the operations
  `add_note`, `resolve_note`, `reopen_note` and `remove_note` work with them.
- **Project zips.** **Download as zip** in the project menu packs the project
  into one file to open on another machine, a cloud GPU or Colab, with
  **Open a project zip**. Three levels, each with its size shown first:
  everything (every take and export), what the cut uses (the takes the
  timeline shows, footage, pictures and sounds), or no renders (the shots,
  cast, pictures, sounds and footage, to render again there). Opening makes a
  new project and never overwrites one, checks every file in the zip before
  writing it, and says which LoRAs the project needs that the machine lacks.
  Uploads of any size work: the zip goes to disk as it arrives. For agents:
  `h3.py pack` and `h3.py unpack`, and the routes `GET
  /project/{name}/archive` and `POST /projects/open`. Thanks to a user who
  built this into a RunPod template and suggested it.
- **Checks in the header.** Everything the checks find is listed in one place:
  a chip at the top says how many things there are to fix and to check, and
  opens a list of them, worst first. It holds low cut scores, takes that cut,
  fade, freeze or go still inside, Soft and Drift clips, VFX clips that pop in
  or out, and your ratings under 70%. Each entry wears the same badge as its
  mark on the timeline, says where it is, and goes there when clicked.
- **Frozen and Still marks.** The cut now checks where a take stops moving. A
  red **Frozen** mark sits on a stretch where the same frame is held (a take
  that stalls), and an amber **Still** mark where almost nothing moves for a
  second and a half or more. A bar under the clip shows how long each lasts;
  clicking the mark goes to it. Measured on 75 rendered takes, none was marked,
  while a held frame is found to the frame. For agents, `h3.py cuts` lists
  them as `inside: frozen` and `inside: still`, and `python cuts.py --still
  FILE.mp4` checks any finished file, where a clip held on its last frame to
  fill a gap shows as frozen.
- **The writing model sees pictures.** Rewrite and Write shots now send the
  cast's own pictures with every request, each named, so the model sees who
  it writes about. The dialog shows them, and takes more: drop, paste or add a
  picture of a place or a product, or the frame the viewer shows. It works with
  every provider, ComfyUI's own Generate Text node included; a model that
  cannot see writes from the text alone. An endpoint that refuses pictures is
  asked again without them, and the dialog says so. Tested with qwen3.5 9B in
  ComfyUI and a 27B model behind an OpenAI-compatible server.
- **Right-click menus.** A right click on a clip on the timeline, a clip in
  the list, a VFX clip, a sound, a cast member or the viewer offers what can be
  done to it: render, rewrite, extend, bridge, merge, split, show its takes,
  remove, use the frame as a first or last frame, write a note. What cannot be
  done right now stays in the menu, greyed, with the reason. The menu key or
  Shift+F10 opens it for the focused clip.
- **Drop a picture on First frame or Last frame** to use it, or to replace the
  one there.
- **The version in the header.** The editor shows which H3 Editor it is next
  to New project, so a screenshot says which release it came from, and
  `h3.py check` prints it too.

### Changed

- **A cleaner side panel.** Long explanations are hidden once you know the
  editor: the **?** at the end of the panel's tabs shows them again, and
  each one is also the tip of the setting it explains. Short lines that say
  what is the case stay. Draft and Final size sit side by side, Models and
  their settings is a link beside Model, and Write shots and Set up a
  language model share a row.
- **Frames and takes show whole.** First and Last frame, and the takes in the
  Takes tab, take the project's shape and show the whole picture, where a 9:16
  picture used to be cropped to 16:9.
- **A real model picker for writing.** In Writing with a language model, the
  Model field opens a list in the editor's own style: List models fills and
  opens it, typing narrows it, the arrow keys and Enter pick, and the model in
  use is ticked. You can still type any name. Changing provider clears a model
  name that belongs to the other kind of provider.
- **The viewer shows where the picture ends.** Around a picture of another
  shape than the viewer, such as 9:16 in a wide window, the screen is hatched
  instead of black, so black bars no longer look like part of the video.
- **The live preview starts muted.** While a clip samples, its looping preview
  no longer plays sound until you turn it on with the speaker button in the
  corner of the viewer. The editor remembers the choice.

### Fixed

- **No black flash at a cut while playing.** At a cut, and at every extension,
  the next clip's video could still be seeking to its first frame when it was
  shown, and the viewer went black for a moment. The clip that ends now stays
  on screen until the next one has drawn its first frame, the next clip starts
  running ahead of the cut more reliably, and the clip after it is not loaded
  into the video that is still covering the cut.
- **Dancing is not a dissolve.** The check for a take that dissolves into
  another picture marked big moves (a dance, a camera move) as dissolves.
  It now also asks that the frames in between look like a mix of the two
  pictures, as a dissolve's do. Fades to black are found as before.
- A project with no notes printed "null" in the Project tab.
- A cast member clicked while a VFX clip was selected did not open.
- The VFX clip's Render and Remove buttons touched the panel's edge, and
  "frames" fell to a line of its own after Transition in.

## 0.5.0, 2026-11-13

### Added

- **Writing with a language model.** **Write shots**, in the Project tab,
  takes a description of the video and adds its shots as new clips. **Rewrite
  with a language model**, in a clip's Shot tab, writes that clip's text again
  from what you ask for. Both show what the model wrote before anything
  changes. Set the model up once, in the Project tab. The usual choice is a
  text model run by this ComfyUI with its own Generate Text node, so nothing
  leaves the machine: Comfy recommends `qwen3.5_9b_w4a8_gptq.safetensors`
  (6.8 GB, from Comfy-Org/MiniMax-H3-Image), which writes a shot in a few
  seconds once it is loaded. Or any OpenAI-compatible endpoint, Anthropic, Gemini or Ollama,
  called from the browser with text only. The settings, any key included,
  stay in the browser, never in a project. H3's own text encoder cannot do
  this: it keeps only the layers H3 reads, not the part that writes.
- **Cut scores.** Where two rendered clips touch on the timeline, the editor
  measures the cut from the pictures on both sides and gives it a score from 0
  to 100. A cut that carries on smoothly or changes to a new shot scores high.
  A jump cut, the same shot with the subject popped to another pose, scores
  low and gets a mark on the timeline with the reason. The same takes and trims
  always give the same score. For agents: `GET /project/{name}/cuts`.
  The cut into and out of a rendered VFX clip is scored the same way, against
  the picture under it. A generated take that should be one shot is checked
  inside as well: where it cuts or fades to another picture, the timeline gets
  a mark at that frame. `python cuts.py --scan film.mp4` scores the cuts of a
  finished file, and reports a few frames of another picture left between two
  cuts as a stray frame.
- **Upscale a final two times.** **Upscale 2x**, on a clip whose take is a
  final, enlarges it with a video upscaler ComfyUI runs natively: a
  1920 x 1088 final becomes 3840 x 2176, as a new take with the same sound.
  You choose the upscaler, among the ones whose two model files are in
  ComfyUI's models folder. RelayVSR (from Comfy-Org/RelayVSR) takes seconds
  and about 16 GB of GPU memory, and sharpens the edges that are there; a
  second menu sets how much of that is kept, since all of it brings out pores
  and shine that can make a face look older. SeedVR2 (from Comfy-Org/SeedVR2)
  takes minutes and far more memory, and redraws fine texture. Clips of up to
  ten seconds at a time. For agents: the `render` operation with
  `"quality": "upscale"`, and optionally `"upscaler"` and `"detail"`.
- **PRISM.** A new model in the Project tab: the PRISM merge of Singularity
  v2 and 10Eros Max (`PRISM_DMAD_PDMD_int8.safetensors`, from
  OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM on Hugging Face), 8
  steps for finals on Euler with the Beta57 schedule. The editor makes that
  schedule with ComfyUI's own node, so a project's own checkpoint can name
  `beta57` as its scheduler on any install. PRISM is for short single shots:
  in tests a speaking presenter's mouth followed the words less exactly than
  on HyperFlow, some areas came out over-sharpened, and extensions drifted
  (the second extension of a shot was 13 to 19% harsher than its first take,
  where HyperFlow stayed within 5%). HyperFlow stays the default.
- **Drift marks.** A shot made of a take and its extensions can drift: on
  some models every extension comes out harsher than the frames it carried
  on from, with more shine and deeper lines on a face. The editor now
  measures each extension against the first take of its shot and puts a
  `Drift` mark on the timeline when it has moved 10% or more, with what to do
  about it. For agents: `drift` on each clip of `GET /project/{name}/detail`.
- **Sharp finals.** A new **Final size**, Sharp, reaches the full size by
  another route. The picture is sampled once at H3's own size, enlarged two
  times with RelayVSR and scaled down to the full size (1920 x 1088 for 16:9).
  It is quicker than Full, which redraws the last steps at the larger size.
  Edges come out crisper, and skin keeps the smoother look of the sampled
  size. It needs RelayVSR's two files. For agents: `set_project` with
  `"sizes": {"final": "sharp"}`.
- **Soft clips.** The editor measures how sharp each rendered clip is at the
  size of the cut, and marks a clip `Soft` on the timeline when it sits far
  under the rest: a draft left among finals, or footage smaller than the cut.
  For agents: `GET /project/{name}/detail`. For any finished file,
  `python detail.py film.mp4` lists the stretches that are much softer than
  the rest, with their times. The same pictures always give the same numbers.
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
- **A Colab notebook.** `colab/h3_editor.ipynb` installs ComfyUI and the
  editor on a Colab GPU, downloads the models from Hugging Face, brings a
  project in and takes it home again, for finals a smaller GPU cannot fit.
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

### Upgrading

Restart ComfyUI after updating: this release adds routes that are registered
at startup. Projects made with 0.4.1 open as they are and render as before.
Upscaling and Sharp finals need RelayVSR's two files (or SeedVR2's for that
upscaler), and PRISM its own file; docs/models.md says where each comes from.

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
