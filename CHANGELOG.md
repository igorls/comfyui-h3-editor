# Changelog

What changed in each release of H3 Editor, newest first. Versions follow
[semantic versioning](https://semver.org): while the pack is at 0.x, a new minor
version can change how projects or the API behave, and the entry says so.
Each release is a tag on GitHub, `v` followed by the version.

## Unreleased

### Added

- **Get the models from the editor.** On a ComfyUI without H3's model files,
  the editor opens a panel that names the files HyperFlow H3 Flash needs to
  render, with each one's size and license, about 54.6 GB in all, and
  downloads them from Hugging Face into ComfyUI's models folders with one
  button, with progress for each file and for the whole. It downloads the
  latent upscaler for finals too. A file comes in under a `.part` name and is
  renamed only once its sha256 matches the published file; a file already in
  place, or another precision of it, is never replaced; a stopped download
  carries on where it stopped. The Realism Slider LoRA is on Civitai, which
  asks for a sign-in, so the panel links to it. The Project tab has **Get the
  models** while a file is missing. Agents use `GET /h3take/api/models/setup`
  and `POST /h3take/api/models/setup/download` and `.../cancel`.
- **A volume for each clip.** The **Timeline** tab of a clip has a **Volume**
  slider beside its speed and transition: 1 is as rendered, 0 is silent, and
  up to 2 is louder. It belongs to the clip's place on the timeline, so the
  same clip placed twice can differ, and both halves of a split keep it. The
  export applies it; the timeline shows it next to the clip's name. Above 1 it
  is louder in the export only, since the viewer tops out at 1. Agents send
  `gain` in `update_item`.
- **A volume for the viewer.** A speaker button and a slider beside the play
  controls set how loud the editor plays: the cut, a clip on its own, added
  sounds and the preview of a clip that is still rendering. It changes what
  you hear and nothing in the export, and the editor remembers it.
- **A take whose sound runs out is named.** When the sound of a take stops
  half a second or more before its picture does, as when a clip runs past the
  end of the song or voice it was made to, **Checks** says it runs on after
  its sound ends, and from which frame. It used to call that a take that ends
  much quieter than it begins.

- **A lip sync check.** Every clip with words or a track gets a measure of
  whether its lips follow its sound, over the part the timeline shows, and
  **Checks** lists the clips whose lips run ahead of the sound, behind it, or
  do not follow it at all, with the offset. It uses SyncNet, the network
  lip-sync research measures with, written into the pack from its MIT code; its
  weights and a small face detector go in `models/syncnet` and are not
  downloaded for you (see the models page). On a sung chorus it agreed with
  SyncNet's own code clip by clip, and found the cut where picture and sound
  had slipped apart by a third of a second. Agents get `lips` in the cuts
  report and a `lips:` line in `h3.py cuts`.
- **A place check.** A generated clip can keep the face and lose the room: a
  keyboard behind the singer becomes a monitor. Every generated clip's
  background, the person taken out, is compared second by second with itself
  two seconds earlier and with the start of its shot, following the camera's
  move, and **Checks** lists a clip where 20% or more of it is no longer what
  it was, with the moment. An extension is held against the first frame of its
  shot, so a chain that has drifted into another room is caught too. It needs
  MODNet's portrait matting file in `models/matting`, not downloaded for you
  (see the models page). On held 20 and 30 s takes and on a slow push into a
  close-up it read 0%; on a take whose room changed under the singer 50%.
  While the camera zooms, it compares only frames framed alike. Agents get `scene` in the cuts report.
- **Room on the card for a local writing model.** The writer's settings have
  **Free ComfyUI's GPU memory before each request**, for the OpenAI-compatible
  and Ollama providers, off by default. For a model server on the same card,
  such as llama.cpp, llama-swap, LM Studio or Ollama, ComfyUI unloads its
  models before each rewrite, and the next render loads them again. It is
  skipped while a render runs or waits, and the result says what happened.
  ([#6](https://github.com/igorls/comfyui-h3-editor/issues/6))

### Changed

- **Long finals refine in windows.** A final longer than 10.4 s draws its first
  pass whole, as before, and redraws its last steps at the final size in
  context windows of 124 frames, which holds the picture the first pass made.
  A 20.75 s sung take came out the same as one refined whole, to the pore, in
  15 minutes instead of 23 on the same GPU, with the room, skin and lips held
  from first frame to last. A ComfyUI without the
  core `MiniMaxH3ContextWindows` node refines the whole take as before.

- **The writing model keeps sung words out of the action.** When someone
  sings, the words go in the dialogue and the action describes the singing.
  In tests where H3 made the singing itself, quoting the lyric in the action
  gave a looser lip sync in 5 of 6 seeds.

### Fixed

- **Long shots made of extensions keep their look.** A take held its source's
  frames frozen, copied them as they were and added its own small lean on top,
  and over a chain the leans added up: colour, contrast and skin texture crept
  further with each extension, shinier and more orange on a face. Extensions,
  bridges and merges now let the held frames far from the join be partly
  redrawn, falling to nothing at the join, so the cut into the new take is as
  exact as before. On HyperFlow at final size, a head clip and four extensions
  came out 4, 16, 28 and 31% harsher than the head; now -3, 3, 6 and 3%, with
  the joins scoring the same. H3 Take has the setting as `context_noise`.
- **A chain of final extensions no longer grows harsher with each one.** An
  extension, bridge or merge held frames of a final that had been through the
  Realism Slider, then ran the slider again, adding its texture on top. On a
  chorus sung as a head clip and four extensions, the extensions came out 5,
  18, 33 and 46% harsher than the head, with crazed skin and deeper lines on
  the face by the end. Linked clips now run the slider at half the project's
  strength (0.6 at the usual 1.2): with it off the skin turned glossy and waxy
  along the chain, and at half it keeps its texture without piling it up. With
  the held frames' far end redrawn as well (above), the same chain stays within
  1% of the head by the third clip and 12% by the fifth. Holding the refined
  latents in the refine pass, new in 0.8.0, was ruled out: encoding the source
  video there instead gave the same drift.
- **An extension, bridge or merge keeps its source clip's model and LoRAs.**
  It used to take the project's, so extending a clip set to another model,
  such as MiniMax H3 Turbo in a HyperFlow project, changed the look where the
  two met. Clips made before keep what they have.
- **No more `Pin error.` warnings while HyperFlow streams from system
  memory.** On a card too small for HyperFlow, ComfyUI keeps part of it in
  system memory and asked to pin each of those layers, where they sat in
  ComfyUI's read-only mapping of the model file. CUDA cannot pin that memory,
  on Windows or Linux, so every layer printed the warning and was copied
  unpinned anyway. HyperFlow now skips those attempts and pins the rest as
  before. Each failed attempt also waited on the GPU, so steps can come out a
  little faster ([#4](https://github.com/igorls/comfyui-h3-editor/issues/4)).
- **The cut plays without micro jumps.** The viewer switched to the next clip
  on the page's own tick, after the browser had already shown the outgoing
  clip's next frame: at most cuts it showed a frame from past the out-point,
  then the incoming clip's frame from before its in-point, and some cuts froze
  for a moment. A cut is now timed from the last frame the outgoing clip put
  on screen, and the incoming clip stays hidden until it shows its in-point.
  On a 43 s reel of 14 clips, every one of its 1031 frames now plays in order,
  where 9 frames from outside the cut used to show.
- **A paused frame is the frame you asked for.** A frame's start time, a
  multiple of 1/24 s, is stored a hair short of itself, and a video held there
  showed the frame before, so stepping frame by frame was often one behind.
  A paused video now sits in the middle of its frame.

## 0.9.1, 2026-11-15

### Added

- **A size of your own.** **Draft size** and **Final size** in the Project tab
  end with **Your own size**: any width and height from 256 to 2048 pixels,
  each a multiple of 32, which H3's latent needs. What you type is moved to
  the nearest one, and the editor shows the megapixels. It starts at 1024x576
  (576x1024 for a vertical project), about 0.6 megapixels. A final at such a
  size comes out as sampled, with no enlarging pass. The shape can differ from
  the project's format, such as 1344x576: the cut and the export take the
  shape of the largest picture on the timeline. Agents write the size as
  `"1024x576"` in `set_project` `sizes`.

## 0.9.0, 2026-11-15

### Added

- **Model patches.** A project can put nodes from any installed pack on the
  model of every render: sparse or low-memory attention, a step cache, a
  compile node, and whatever else takes one model and plain values and returns
  one model. The Project tab has a **Model patches** section: **Add a patch**
  searches the installed nodes, with the ones for H3, MiniMax or attention
  first, and each patch gets a row with its node's own settings, drawn from
  the node's definition. Patches can be reordered, unticked and removed, and a
  project takes up to 8. They sit after the LoRAs, on drafts and on both
  passes of a final, and a take records the ones it was made with. A patch
  whose pack is not installed is kept in the project and left out of renders.
  A project opened from a zip has its patches switched off until you tick them.
  Agents use `set_project` `patches`, and `GET /projects` lists the installed
  nodes with their inputs under `patches`. The editor names no other pack and
  needs none. It cannot know whether a patch helps or harms: render a clip
  with it and without, and compare the render time and `h3.py cuts`.
  Measured with one, Veda Sparse Attention at 90% sparsity on MiniMax H3 Turbo:
  a 3.8 second draft took more than twice as long, a 3.8 second final was
  about 1.3 times faster, and an 8 second final of a bridge 2.3 times faster
  in one run, with the same scores at both of its cuts. All three 3.8 second
  finals made with it have a small jump 5 frames before their end, which the
  editor's checks report as a change of picture inside the take; the finals
  made without it and the drafts do not. That node accepts plain MiniMax H3
  checkpoints and refuses HyperFlow H3 Flash.

### Fixed

- **A project zip can no longer bring a revision history.** A zip made by hand
  could carry files that landed in the new project's history, its previews or
  its project file, and restoring such a revision brought back whatever it
  held. Opening a zip now passes over everything that **Download as zip** never
  packs, and the project starts with one revision. Zips made by the editor
  open as before.
- **Zip paths that Windows reads as another file are refused:** a name with a
  colon, a control character, one of `< > " | ? *`, or a device name such as
  `NUL`. The editor writes no such names.

## 0.8.0, 2026-11-15

### Added

- **Finals join finals without a decode in between.** A final is sampled
  twice: at H3's own size, then enlarged and refined at up to 1920 pixels. A
  final that extends or bridges another final held its source's latents in
  the first pass only; for the refine pass it read the source's video and
  encoded it again at the large size. A refined final now saves that pass's
  latents too, and the next final holds them in its own refine pass. The file
  beside a 5 second final grows to about 22 MB. Finals rendered before this
  are held from their video in the refine pass as before, so render a source
  again to get the benefit.

- **The cast catalog.** Save a cast member with **Save** beside **Catalog**, and
  bring them into any project with **Add**, **From the catalog**. A project gets
  its own copy of their pictures and voice, linked to the catalog. After a
  change is saved to the catalog, other projects show **Newer in the
  catalog** and an **Update** button, and the catalog can update every project
  at once, leaving alone copies changed in their project. Agents use
  `save_cast_to_catalog`, `add_cast_from_catalog` and
  `update_cast_from_catalog`.

- **Joins hold a take's own latents from any cut, with its sound.** A take
  now saves all of its latents, picture and sound, in its `.hold.safetensors`
  file (about 1.6 MB for a 5 second draft), where it used to save the first
  and last 39 frames of the picture. A clip that extends or bridges it holds
  them as they are from any cut on a multiple of 17 frames, the grid of H3's
  picture latent, and not only from the take's two ends. The held sound comes
  from the latents too when the cut is also on a multiple of 3 frames, as the
  end of every extension is; otherwise it is encoded from the file as before.
  Measured on a chain of three extensions, one of them from a trimmed clip:
  the held picture is the source's latents exactly at all three joins, and the
  held sound is at the join where the grids meet. Takes rendered before this
  keep their old file and are held from their two ends as before.
- **Joins cut where the two clips match.** A clip that extends or bridges
  another holds 39 of its frames, so both files show them. The timeline used
  to cut from one file to the other at the edge of those frames. That is the
  worst place for it: a video decoder draws the last frames of a file with
  nothing after them and the first with nothing before them, so they never
  quite match the same frames in the other file, and the held sound is
  softened over a fifth of a second next to the new material, so the two waves
  have drifted apart by then. New extensions, bridges and merges cut 9 frames
  inside the held frames: the source stops 9 frames earlier and the new clip
  plays those 9 itself, so the scene is as long as before. Measured on 16
  joins, the picture at a file's end was 1.5 dB under the rest on its last
  frame (5 dB at worst) and level from the fourth; at a file's start it was 6
  to 9 dB under on the first frame and level from the eighth. On a test chain
  the step in the sound at the cut fell from 7 to 22 times the steps around
  it to under 3. Joins made before keep their cut. Render a linked
  clip again after trimming its source and it is joined at the new cut.
- **The cuts report listens.** Every cut now has a sound score beside its
  picture score (`listen.py`): the level on each side as loudness after ITU-R
  BS.1770, the step between them against how much the sound moves by itself
  around the cut, how much the spectrum changes shape, and whether the wave
  clicks. A cut where the sound falls 36 dB in one frame scores 0 as a
  `level jump`, a cut to another place with the level held is a `new sound`,
  and sound into silence is a `dropout`. Inside a take it reports a sound
  that is cut off (a fast fall that stays down; sounds that start at once
  are left alone), a hole in the sound, and a take that ends 20 dB or more
  louder or quieter than it began. `GET /project/{name}/cuts` carries these
  as `sound` on each cut and as `sound_inside`, `h3.py cuts` prints them,
  and `worst` counts them.
- **The report reads the camera's move at a cut.** From the optical flow of
  the whole frame (Lucas and Kanade's method, solved for one pan and one zoom
  per frame, with moving subjects weighted down) it says how the camera
  moves on each side of a cut, in words (pushing in, pulling back, panning,
  tilting, still), and whether the move carries on, stops, starts or
  reverses. Inside one shot that is scored, against how much the move varies
  by itself around the cut, as a handheld shot does; at a cut to a new shot
  it is only described. It is `movement` on each cut. It reads the camera,
  not what moves inside the frame.
- **The report sees what stops moving at a cut.** With the camera's own
  move taken out, it looks at what is left in each of 72 parts of the frame:
  rain behind a window, a wave, a beam of light. Something that moves on one
  side of a cut and is still on the other is reported with where it is
  ("what moves in the upper middle of the frame before the cut is still
  after it"), and scored inside one shot. It is `parts` on each cut.
- **The report says whether a bridge arrived.** For a bridge that ends on a
  pinned frame, `arrivals` says whether the take got there: a take still
  fading up when it ends lands darker than its pin, and a jump at the
  hand-over means the frames before the pin did not lead to it.
- **A measure for joins.** `python joins.py PROJECT_FOLDER` compares the two
  copies of the frames every join holds, the source's and the new take's:
  whether the latents are the same, how alike the decoded pictures are frame
  by frame, how much darker or softer the new take's copy is, and whether the
  sound carries on or clicks at the cut. It reads the files and changes
  nothing.
- **A project list you can see.** The project menu in the top bar is now a
  list of the editor's own: each project with a frame of its first rendered
  clip, its shape, how many clips are on its timeline, how long it runs and
  when it last changed. The open project leads and the rest follow by last
  change. Type to narrow the list by name, move with the arrow keys and open
  with Enter. `GET /projects` returns the same facts for each project under
  `cards`.

### Changed

- **A bridge lands on the next clip.** A bridge used to hold the next clip's
  first 39 frames at its end, as it holds the first clip's last 39 at its
  start. H3 treats a run that long from another shot as another shot: in 8 of
  9 test pairs it followed the prompt and then cut to the next clip a few
  frames early, so the bridge was an extension with a cut at its end.
  **Bridge to next** now holds a short run of the next clip, 22 frames with
  their sound, and the timeline cuts 9 frames inside it. That is enough to
  say where things are and how they move, and too little for H3 to take for
  another shot. On two test pairs over three seeds each, the cut into the
  next clip scored 97 to 99 of 100 for the picture and 78 to 97 for the
  sound, and what moves in the frame carried across it every time. One
  pinned frame, tried first, scored 71 to 97 and 0 to 90: the take landed on
  it dark and silent, with the sound dropping up to 36 dB at the cut. A new
  bridge is 6.4 seconds unless you set its seconds, because a move between
  two framings needs the room. What happens inside the bridge still depends
  on the seed: it may push in, dissolve or cut on the way, so look at it and
  render it again when it does not move as you asked. The held sound is
  encoded from the next clip's file, not taken from its latents, and can
  step a little at the cut. **Bridge into its movement**, new in the
  Generate menu, holds 39 frames of the next clip as before. Agents pass
  `"arrive": "hold"` to `bridge` for that, or `"arrive": "pin"` for one
  pinned frame, which leaves the next clip untrimmed. Bridges made before
  keep their way of arriving.
- **LoRA starts follow where the render's steps are.** H3 samples with a
  shift that keeps most of a render near the start of the scale: a start of
  0.75 was already four fifths of the way in. **After framing** is now 0.93,
  about halfway, and **Details only** 0.7. A final's Realism Slider joins at
  0.93 instead of 0.75: of HyperFlow's six steps it now works on the last
  four, where it used to work on the last three. A start you set yourself
  stays as it is.
- **Extending from a trimmed clip moves the cut onto the latent's grid.**
  Extend forward, Extend back and Bridge move the end they carry on from back,
  and the start they arrive at forward, by 16 frames at most (two thirds of a
  second), so the new clip can hold the source's latents. The clips after it
  close up. A whole clip is already on the grid and is not touched, and
  imported footage keeps its cut. Merge still holds its frames from the video.

### Fixed

- The side panel's tabs no longer make the panel scroll sideways at its usual
  width.
- Space plays again after a choice from a menu followed by a scrub of the
  timeline. The menu kept the keyboard, so Space opened it instead.

## 0.7.0, 2026-11-13

### Added

- **A smaller text encoder.** A Qwen3-VL 4B or 8B can stand in for H3's 32B
  text encoder through a projection from ClipProj, built into the pack: about
  4.6 GB instead of 15.7 GB with the 4B. Put the encoder in
  `models/text_encoders` and its projection in `models/clip_projections`, then
  pick it under **Text encoder** in the Models dialog (**Model files** in the
  Project tab), and under **Projection** when more than one fits it. It knows less than the
  32B, mostly about named people and speech in languages other than English.
  The **H3 Small Text Encoder** node does the same in your own workflows. See
  [models and quality](docs/models.md#a-smaller-text-encoder).

### Changed

- **A shorter side panel.** The Project tab starts with a line on what the
  project holds, then Look, Format (with Language beside it and both sizes on
  one row) and Model. The text encoder, video VAE and projection moved into the
  Models dialog, and the panel says which files render. In the Shot tab, Camera
  and Seconds sit right under What happens, Rewrite is a small button in its
  label, and the first and last frames take one row of small pictures with
  icon buttons. A clip's model and LoRAs share one section, and its LoRAs are
  set in a dialog. A cast member's pictures, voice and RefMod are edited in a
  dialog, with the panel keeping the picture, the description and the name.
- **Labels beside short fields,** icons on the tabs, sections and small
  actions, and the tabs drop their words in a narrow panel.
- **Sections stay as you left them.** A folding section you open or close
  stays that way in this browser, and the Timeline tab shows whenever the clip
  is on the timeline, however it was selected. Delete take is now a trash
  button at the end of the Takes row.
- **A LoRA's start reads as a timeline.** The bar runs from the render's first
  step to its last, lit where the LoRA works, with **At once**, **After
  framing** and **Details only** for the usual starts. The button that opens
  it says **Starts at once** or the start, instead of **All steps**.

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
