const FPS = 24;
const $ = (id) => document.getElementById(id);
// Kept between visits, because ComfyUI only sends progress to the client that queued the render.
const clientId = (() => {
  try {
    const saved = localStorage.getItem("h3take.client");
    if (saved) return saved;
    const made = "h3editor-" + Math.random().toString(36).slice(2);
    localStorage.setItem("h3take.client", made);
    return made;
  } catch (_) {
    return "h3editor-" + Math.random().toString(36).slice(2);
  }
})();
const STAGES = { 10: "Reading the prompt", 15: "Sampling", 16: "Decoding the picture", 17: "Decoding the sound", 18: "Assembling", 19: "Saving" };

const state = {
  folder: null,
  project: null,
  clip: null,
  item: null,
  person: null,
  gap: null,
  playhead: 0,
  playing: false,
  zoom: 3,
  jobs: new Map(),
  running: null,
  edits: 0,
  savedEdits: 0,
  saving: null,
  quiet: false,
  profiles: [],
  finals: [],
  loras: [],
  realism: null,
  checkpoints: [],
  presets: [],
  refmods: { installed: false, files: [] },
  encoders: { text_encoders: [], vaes: [], auto: {}, missing: [] },
  samplers: [],
  schedulers: [],
  bin: null,
  tab: "shot",
  folded: new Set(),
  source: null,
  sourceFrame: 0,
};
let undo = [];
let redo = [];
let saveTimer = 0;

const px = (frames) => frames * state.zoom;
const uid = (prefix) => prefix + Math.random().toString(36).slice(2, 8);
const clipOf = (id) => state.project.clips.find((c) => c.id === id);
const binOf = (id) => (id && (state.project.bins || []).find((b) => b.id === id)) || null;
// A bin is selected only while no clip or cast member is.
const selectedBin = () => (!state.clip && !state.person && binOf(state.bin)) || null;
// The clip the viewer shows on its own, picked in the Clips panel. Any other selection ends it.
const viewing = () => (state.source && state.source === state.clip && clipOf(state.source)) || null;
const sourceEnd = () => activeTake(viewing())?.frames || 0;
const here = () => (viewing() ? state.sourceFrame : state.playhead);
const viewEnd = () => (viewing() ? sourceEnd() : sceneEnd());
const itemLength = (item) => Math.max(1, Math.round((item.out - item.in) / (item.speed || 1)));
const itemEnd = (item) => item.start + itemLength(item);
const viewUrl = (file, subfolder) => "/view?type=output&filename=" + encodeURIComponent(file) + "&subfolder=" + encodeURIComponent(subfolder || "");
const sorted = () => [...state.project.timeline].sort((a, b) => a.start - b.start);
const sceneEnd = () => state.project.timeline.reduce((end, item) => Math.max(end, itemEnd(item)), 0);

function activeTake(clip) {
  const takes = clip?.takes || [];
  return takes.find((t) => t.id === clip.active) || takes[takes.length - 1] || null;
}

function jobFor(clipId) {
  for (const job of state.jobs.values()) if (job.clip === clipId) return job;
  return null;
}

function timecode(frames) {
  const f = Math.max(0, Math.round(frames));
  const two = (n) => String(n).padStart(2, "0");
  return two(Math.floor(f / FPS / 60)) + ":" + two(Math.floor(f / FPS) % 60) + ":" + two(f % FPS);
}

// Messages stack, newest at the bottom, and each one can be closed. `action` adds a button, such as Undo.
function toast(message, bad = false, action = null) {
  const stack = $("toast");
  const el = document.createElement("div");
  el.className = "toast" + (bad ? " bad" : "");
  el.setAttribute("role", bad ? "alert" : "status");
  const text = document.createElement("span");
  text.textContent = message;
  el.append(text);
  let timer = 0;
  const close = () => {
    clearTimeout(timer);
    el.remove();
  };
  if (action) {
    const act = document.createElement("button");
    act.className = "toast-action";
    act.textContent = action.label;
    act.addEventListener("click", () => {
      close();
      action.run();
    });
    el.append(act);
  }
  const dismiss = document.createElement("button");
  dismiss.className = "toast-close";
  dismiss.setAttribute("aria-label", "Dismiss");
  dismiss.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 7l10 10M17 7L7 17"/></svg>';
  dismiss.addEventListener("click", close);
  el.append(dismiss);
  // A message stays while the pointer is on it, so it can be read and acted on.
  const wait = () => (timer = setTimeout(close, bad ? 8000 : action ? 6000 : 4000));
  el.addEventListener("pointerenter", () => clearTimeout(timer));
  el.addEventListener("pointerleave", wait);
  // The same message again takes the place of the one already showing.
  for (const shown of [...stack.children]) {
    if (shown.firstElementChild.textContent === message) shown.remove();
  }
  while (stack.children.length >= 4) stack.firstElementChild.remove();
  stack.append(el);
  wait();
}

const undoAction = { label: "Undo", run: () => restore(undo, redo) };

async function api(path, options) {
  const response = await fetch("/h3take/api" + path, options);
  if (!response.ok) {
    const text = await response.text();
    let message = text;
    try { message = JSON.parse(text).error || text; } catch (_) {}
    throw new Error(message || "The request failed.");
  }
  return response.json();
}

const postJson = (path, body, method = "POST") =>
  api(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

// The server owns the project. The editor sends its copy with the revision it started from,
// and the server merges it with whatever else changed: finished renders, an agent, another window.
function save() {
  state.edits++;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(flush, 300);
}

const unsaved = () => state.edits !== state.savedEdits || state.saving !== null;

async function flush() {
  clearTimeout(saveTimer);
  while (state.saving) await state.saving;
  if (state.edits === state.savedEdits) return;
  const sent = state.edits;
  const folder = state.folder;
  state.saving = (async () => {
    try {
      const result = await postJson("/project/" + folder + "?client=" + clientId, state.project, "PUT");
      state.savedEdits = sent;
      if (folder === state.folder && state.edits === sent) adopt(result.project);
    } catch (error) {
      toast(error.message, true);
    } finally {
      state.saving = null;
    }
  })();
  await state.saving;
}

// Take the server's copy as the editor's own. Only safe when nothing local is waiting to be saved.
function adopt(project, { external = false } = {}) {
  const before = state.jobs;
  const same = JSON.stringify({ ...project, revision: 0 }) === JSON.stringify({ ...state.project, revision: 0 });
  if (same) {
    // Nothing changed but the revision. The panel's fields still write to the objects they
    // were built on, so those stay in place: swapping them would lose what is typed next.
    state.project.revision = project.revision;
    return;
  }
  state.project = project;
  syncJobs();
  if (!clipOf(state.clip)) state.clip = null;
  if (!state.project.timeline.some((i) => i.id === state.item)) state.item = null;
  if (state.person && !(state.project.cast || []).some((p) => p.id === state.person)) state.person = null;
  if (!binOf(state.bin)) state.bin = null;
  if (external) {
    undo = [];
    redo = [];
  }
  for (const [id, job] of before) {
    if (state.jobs.has(id)) continue;
    const clip = clipOf(job.clip);
    if (!clip) continue;
    if (clip.takes.some((t) => t.id === job.take)) toast(clip.name + " is ready.");
    else if (clip.error && !job.reported) toast(clip.name + ": " + clip.error, true);
  }
  renderAll();
  ensureStrips();
}

async function refreshProject(external) {
  if (!state.folder || unsaved()) return;
  const folder = state.folder;
  const project = await api("/project/" + folder);
  if (folder === state.folder && !unsaved() && (project.revision || 0) > (state.project.revision || 0)) adopt(project, { external });
}

// Call before a change that undo should be able to reverse.
function checkpoint() {
  if (state.quiet) return;
  undo.push(JSON.stringify(state.project));
  if (undo.length > 100) undo.shift();
  redo = [];
}

function restore(from, to) {
  if (!from.length) return;
  to.push(JSON.stringify(state.project));
  // An undo is a new edit on top of the latest revision, not a step back to an old one.
  const revision = state.project.revision;
  state.project = JSON.parse(from.pop());
  state.project.revision = revision;
  if (!clipOf(state.clip)) state.clip = null;
  state.item = null;
  save();
  renderAll();
}

// Clips never overlap on one track. A clip dropped onto another pushes the later ones along.
function settle(moved) {
  const items = sorted();
  if (moved) {
    items.sort((a, b) => a.start - b.start || (a === moved ? -1 : b === moved ? 1 : 0));
  }
  let end = 0;
  for (const item of items) {
    const floor = end - (item.transition?.frames || 0);
    if (item.start < floor) item.start = floor;
    end = Math.max(end, itemEnd(item));
  }
}

function addClip() {
  checkpoint();
  const number = state.project.clips.length + 1;
  const clip = { id: uid("c"), name: "Clip " + number, action: "", camera: "", speaker: "", dialogue: "", sound: "", music: "", seconds: 5, takes: [], active: null };
  const bin = selectedBin();
  if (bin) clip.bin = bin.id;
  state.project.clips.push(clip);
  placeOnTimeline(clip, sceneEnd());
  state.clip = clip.id;
  state.person = null;
  state.item = state.project.timeline.at(-1).id;
  state.tab = "shot";
  save();
  renderAll();
  document.querySelector("#inspector textarea.big")?.focus();
}

// Mirrors snap_frames and planned_frames in project.py: a clip is as long as its render will be.
function snapFrames(seconds) {
  return 5 + 17 * Math.max(1, Math.round((seconds * FPS - 5) / 17));
}

function plannedFrames(clip) {
  const take = activeTake(clip);
  return take ? take.frames : clip.link ? linkedFrames(clip.link, clip.seconds) : snapFrames(clip.seconds);
}

// An unrendered clip follows its seconds on the timeline, and the clips after it move along.
function resizePlanned(clip) {
  if (clip.takes.length) return;
  const [first, last] = linkedRange(clip.link, plannedFrames(clip));
  for (const item of state.project.timeline) {
    if (item.clip !== clip.id) continue;
    const end = itemEnd(item);
    item.in = first;
    item.out = last;
    shiftFrom(end, itemEnd(item) - end, item);
  }
  settle(null);
  renderTimeline();
}

function placeOnTimeline(clip, start) {
  const [first, last] = linkedRange(clip.link, plannedFrames(clip));
  const item = { id: uid("i"), clip: clip.id, start, in: first, out: last };
  state.project.timeline.push(item);
  settle(item);
  return item;
}

const HOLD = 39;
const JOINT = 51;

// Mirrors linked_length in project.py.
function linkedFrames(link, seconds) {
  const frames = seconds * FPS;
  if (link.mode === "bridge") return link.hold + JOINT * Math.max(2, Math.round((frames + link.hold) / JOINT));
  return link.hold + JOINT * Math.max(1, Math.round(frames / JOINT));
}

// The part of a linked render the timeline shows. Held frames are already on screen in the source clip.
function linkedRange(link, frames) {
  if (!link) return [0, frames];
  if (link.mode === "extend_after") return [link.hold, frames];
  if (link.mode === "extend_before") return [0, frames - link.hold];
  return [link.hold, frames - link.hold];
}

function sideOf(item) {
  const take = activeTake(clipOf(item.clip));
  return { clip: item.clip, item: item.id, file: take.file, subfolder: take.subfolder, first: item.in, last: item.out };
}

function linkSides(link) {
  return link ? [link.a, link.b].filter(Boolean) : [];
}

// A linked clip was generated from specific frames. It no longer joins cleanly once those change.
function sideChanged(side) {
  const item = state.project.timeline.find((i) => i.id === side.item);
  const take = activeTake(clipOf(side.clip));
  if (!item || !take) return false;
  return take.file !== side.file || item.in !== side.first || item.out !== side.last;
}

// Only a join that still exists can show, so a clip that moved away from its source is left alone.
function touches(clip, side) {
  const source = state.project.timeline.find((i) => i.id === side.item);
  if (!source) return false;
  return state.project.timeline.some((item) => item.clip === clip.id && (itemEnd(item) === source.start || itemEnd(source) === item.start));
}

function isStale(clip) {
  return clip.takes.length > 0 && linkSides(clip.link).some((side) => sideChanged(side) && touches(clip, side));
}

function refreshLink(clip) {
  for (const side of linkSides(clip.link)) {
    const item = state.project.timeline.find((i) => i.id === side.item);
    if (item && activeTake(clipOf(item.clip))) Object.assign(side, sideOf(item));
  }
}

function followingItem(item) {
  return sorted().find((other) => other !== item && other.start >= itemEnd(item)) || null;
}

function selectedRendered() {
  const item = state.project.timeline.find((i) => i.id === state.item);
  return item && activeTake(clipOf(item.clip)) ? item : null;
}

function bridgeable(item, next) {
  return Boolean(item && next && activeTake(clipOf(item.clip)) && activeTake(clipOf(next.clip)) && itemLength(item) >= HOLD && itemLength(next) >= HOLD);
}

function updateTools() {
  const item = selectedRendered();
  const next = item && followingItem(item);
  const holdable = item && itemLength(item) >= HOLD;
  $("extendBack").disabled = !holdable;
  $("extendForward").disabled = !holdable;
  $("bridge").disabled = !bridgeable(item, next);
  $("merge").disabled = !(item && next && activeTake(clipOf(next.clip)) && next.start === itemEnd(item));
  $("remove").disabled = !(state.item || state.gap);
  $("ripple").disabled = !state.item;
  $("closeGap").disabled = !state.gap;
}

function shiftFrom(frame, by, skip = null) {
  for (const other of state.project.timeline) {
    if (other !== skip && other.start >= frame) other.start += by;
  }
}

// Mirrors continued_track in project.py: a linked clip holds the next slice of its source's track,
// so one long performance carries on across clips. Returns false, with a toast, when it cannot.
function continuedTrack(source, item, mode, frames, out = item.out) {
  if (!source.track) return null;
  if (speedOf(item) !== 1) {
    toast("Set " + source.name + " to 100% speed to continue its track.", true);
    return false;
  }
  const from = mode === "extend_before"
    ? source.track.from + (item.in - (frames - HOLD)) / FPS
    : source.track.from + (out - HOLD) / FPS;
  if (from < -1e-6) {
    toast("The track starts " + (-from).toFixed(2) + " s too late for this extension.", true);
    return false;
  }
  return { ...source.track, from: Math.round(Math.max(0, from) * 1e6) / 1e6 };
}

function addLinked(mode) {
  const item = selectedRendered();
  if (!item) return;
  const source = clipOf(item.clip);
  const next = followingItem(item);
  const seconds = 4.25;
  const link = { mode, hold: HOLD, a: sideOf(item) };
  if (mode === "bridge") link.b = sideOf(next);
  const track = continuedTrack(source, item, mode, linkedFrames(link, seconds));
  if (track === false) return;
  checkpoint();
  const titles = { extend_after: "After ", extend_before: "Before ", bridge: "Bridge from " };
  const clip = {
    id: uid("c"), name: titles[mode] + source.name, action: "", camera: "", speaker: "", dialogue: "",
    sound: "", music: "", seconds, takes: [], active: null, link,
  };
  if (source.cast?.length) clip.cast = [...source.cast];
  if (binOf(source.bin)) clip.bin = source.bin;
  if (track) clip.track = track;
  state.project.clips.push(clip);
  const [first, last] = linkedRange(link, linkedFrames(link, seconds));
  const length = last - first;
  let start = itemEnd(item);
  if (mode === "extend_before") {
    start = item.start;
    shiftFrom(item.start, length);
  } else if (mode === "extend_after") {
    shiftFrom(start, length);
  } else {
    shiftFrom(next.start, start + length - next.start);
  }
  const placed = { id: uid("i"), clip: clip.id, start, in: first, out: last };
  state.project.timeline.push(placed);
  state.clip = clip.id;
  state.person = null;
  state.item = placed.id;
  state.tab = "shot";
  save();
  renderAll();
  document.querySelector("#inspector textarea.big")?.focus();
}

const assetUrl = (asset) => viewUrl(asset.file, asset.subfolder);

async function uploadPicture(file) {
  const form = new FormData();
  form.append("file", file);
  return api("/project/" + state.folder + "/asset", { method: "POST", body: form });
}

// Real footage and pictures become clips on the server, sized and timed like the renders.
async function importFootage() {
  const file = await pickFile($("importFile"));
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  form.append("client", clientId);
  const bin = selectedBin();
  if (bin) form.append("bin", bin.id);
  toast("Importing " + file.name);
  try {
    await flush();
    const result = await api("/project/" + state.folder + "/import", { method: "POST", body: form });
    if (!unsaved()) adopt(result.project);
    selectClip(result.clip, result.item);
    toast(file.name + " is on the timeline.");
  } catch (error) {
    toast(error.message, true);
  }
}

// A voice with no picture, such as a narrator. Its lines use the voice; it never appears in the shot.
async function addVoiceMember() {
  const file = await pickFile($("voiceFile"));
  if (!file) return;
  try {
    const voice = await uploadPicture(file);
    checkpoint();
    state.project.cast = state.project.cast || [];
    const person = { id: uid("p"), name: "Narrator" + (state.project.cast.some((p) => p.name === "Narrator") ? " " + (state.project.cast.length + 1) : ""), description: "", voice };
    state.project.cast.push(person);
    state.person = person.id;
    state.clip = null;
    state.item = null;
    save();
    renderAll();
  } catch (error) {
    toast(error.message, true);
  }
}

function setVoice(person, voice) {
  checkpoint();
  if (voice) person.voice = voice;
  else delete person.voice;
  save();
  renderAll();
}

async function grabVoice(person) {
  const item = itemAt(state.playhead);
  const take = activeTake(clipOf(item?.clip));
  if (!take) {
    toast("Move the playhead over a clip where they speak, then press Use the clip under the playhead.");
    return;
  }
  try {
    const last = Math.min(item.out, item.in + 30 * FPS);
    setVoice(person, await postJson("/project/" + state.folder + "/grab_sound", { file: take.file, subfolder: take.subfolder, first: item.in, last }));
    toast(person.name + " now speaks with the voice from " + clipOf(item.clip).name + ".");
  } catch (error) {
    toast(error.message, true);
  }
}

// A clip can hold a slice of a recorded track, such as a narration, and have its picture made to it.
function trackSection(clip) {
  const box = document.createElement("div");
  box.className = "voice";
  const ways = document.createElement("div");
  ways.className = "ways";
  const button = (text, run) => {
    const element = document.createElement("button");
    element.className = "quiet";
    element.textContent = text;
    element.addEventListener("click", run);
    ways.append(element);
  };
  const hint = document.createElement("p");
  hint.className = "hint";
  if (clip.track) {
    const player = document.createElement("audio");
    player.controls = true;
    player.preload = "none";
    player.src = assetUrl(clip.track) + "#t=" + clip.track.from;
    box.append(player, field("Starts this many seconds into the track", clip.track.from, (v) => {
      if (Number(v) >= 0) clip.track.from = Number(v);
    }, { type: "number" }));
    button("Remove track", () => {
      checkpoint();
      delete clip.track;
      save();
      renderInspector();
    });
    hint.textContent = "The picture is made to this sound, and the take keeps it exactly. Extend forward to carry on with the next part of the track.";
  } else {
    button("Use a sound file", async () => {
      const file = await pickFile($("voiceFile"));
      if (!file) return;
      try {
        const asset = await uploadPicture(file);
        checkpoint();
        clip.track = { ...asset, from: 0 };
        save();
        renderInspector();
      } catch (error) {
        toast(error.message, true);
      }
    });
    hint.textContent = "Hold a recorded track, such as a narration, and the picture is made to it, lips and all.";
  }
  box.append(ways, hint);
  return [box];
}

function voiceSection(person) {
  const box = document.createElement("div");
  box.className = "voice";
  const ways = document.createElement("div");
  ways.className = "ways";
  const button = (text, run) => {
    const element = document.createElement("button");
    element.className = "quiet";
    element.textContent = text;
    element.addEventListener("click", run);
    ways.append(element);
  };
  if (person.voice) {
    const player = document.createElement("audio");
    player.controls = true;
    player.preload = "none";
    player.src = assetUrl(person.voice);
    box.append(player);
    if (person.file) button("Remove voice", () => setVoice(person, null));
  }
  button(person.voice ? "Replace with a file" : "Upload a voice", async () => {
    const file = await pickFile($("voiceFile"));
    if (!file) return;
    try {
      setVoice(person, await uploadPicture(file));
    } catch (error) {
      toast(error.message, true);
    }
  });
  button("Use the clip under the playhead", () => grabVoice(person));
  const hint = document.createElement("p");
  hint.className = "hint";
  hint.textContent = person.voice
    ? "Write @" + person.name + " in Who speaks, and their lines use this voice. Up to 30 seconds of it is used."
    : "Give " + person.name + " a voice so every line they say sounds like the same person.";
  box.append(ways, hint);
  return box;
}

function pickFile(input) {
  return new Promise((resolve) => {
    input.value = "";
    input.onchange = () => resolve(input.files[0] || null);
    input.click();
  });
}

async function addCastMember() {
  const file = await pickFile($("castFile"));
  if (!file) return;
  try {
    const asset = await uploadPicture(file);
    checkpoint();
    state.project.cast = state.project.cast || [];
    const person = { id: uid("p"), name: "Person " + (state.project.cast.length + 1), description: "", ...asset };
    state.project.cast.push(person);
    state.person = person.id;
    state.clip = null;
    state.item = null;
    save();
    renderAll();
  } catch (error) {
    toast(error.message, true);
  }
}

function setClipFrame(clip, side, asset) {
  checkpoint();
  if (asset) clip[side] = asset;
  else delete clip[side];
  save();
  renderInspector();
}

async function uploadFrame(clip, side) {
  const file = await pickFile($("frameFile"));
  if (!file) return;
  try {
    setClipFrame(clip, side, await uploadPicture(file));
  } catch (error) {
    toast(error.message, true);
  }
}

async function grabFrame(clip, side) {
  // The frame the viewer shows: of the clip on its own, or of the cut at the playhead.
  const alone = viewing();
  const item = alone ? null : itemAt(state.playhead);
  const take = activeTake(alone || clipOf(item?.clip));
  if (!take) {
    toast("Move the playhead over a rendered clip, then press Use playhead frame.");
    return;
  }
  try {
    const frame = alone
      ? Math.min(take.frames - 1, Math.round(state.sourceFrame))
      : Math.min(take.frames - 1, Math.round(item.in + (state.playhead - item.start) * speedOf(item)));
    setClipFrame(clip, side, await postJson("/project/" + state.folder + "/grab", { file: take.file, subfolder: take.subfolder, frame }));
  } catch (error) {
    toast(error.message, true);
  }
}

// Mirrors frame_of in project.py: a held end already has its real frames.
function frameUsable(clip, side) {
  const mode = clip.link?.mode;
  if (!mode) return true;
  return mode !== "bridge" && (side === "first") !== (mode === "extend_after");
}

function frameSlot(clip, side, label) {
  const slot = document.createElement("div");
  slot.className = "slot";
  const title = document.createElement("h3");
  title.textContent = label;
  const picture = document.createElement("div");
  picture.className = "picture";
  const ways = document.createElement("div");
  ways.className = "ways";
  const button = (text, run) => {
    const element = document.createElement("button");
    element.className = "quiet";
    element.textContent = text;
    element.addEventListener("click", run);
    ways.append(element);
  };
  if (clip[side]) {
    const image = document.createElement("img");
    image.alt = label;
    image.src = assetUrl(clip[side]);
    picture.append(image);
    button("Remove", () => setClipFrame(clip, side, null));
  } else {
    picture.textContent = "None";
    button("Use playhead frame", () => grabFrame(clip, side));
    button("Upload", () => uploadFrame(clip, side));
  }
  slot.append(title, picture, ways);
  return slot;
}

function castPicker(clip) {
  const title = document.createElement("h3");
  title.textContent = "Cast in this clip";
  const picks = document.createElement("div");
  picks.className = "picks";
  const pictured = state.project.cast.filter((p) => p.file || p.refmod);
  for (const person of pictured) {
    const on = (clip.cast || []).includes(person.id);
    const pick = document.createElement("button");
    pick.className = "pick" + (on ? " on" : "");
    pick.setAttribute("aria-pressed", String(on));
    // A member that is only a RefMod has no picture to show.
    const image = document.createElement(person.file ? "img" : "span");
    if (person.file) {
      image.alt = "";
      image.src = assetUrl(person);
    } else {
      image.className = "mod-mark";
      image.textContent = "RefMod";
    }
    const name = document.createElement("span");
    name.textContent = person.name;
    pick.append(image, name);
    pick.addEventListener("click", () => {
      checkpoint();
      clip.cast = on ? clip.cast.filter((id) => id !== person.id) : [...(clip.cast || []), person.id];
      save();
      renderInspector();
    });
    picks.append(pick);
  }
  const hint = document.createElement("p");
  hint.className = "hint";
  hint.textContent = "Write @" + pictured[0].name + " in What happens to place them in the shot.";
  return [title, picks, hint];
}

function renderCast() {
  const list = $("cast");
  const cast = state.project.cast || [];
  if (!cast.length) {
    list.innerHTML = '<p class="empty">Add a picture of a person or place to keep them the same across clips, or a voice for a narrator.</p>';
    list.style.display = "block";
    return;
  }
  list.style.display = "";
  list.replaceChildren(...cast.map((person) => {
    const button = document.createElement("button");
    button.className = "person" + (person.id === state.person ? " on" : "") + (person.voice && person.file ? " voiced" : "");
    if (person.voice) button.title = person.file ? person.name + " has a voice" : person.name + " is a voice";
    let image;
    if (person.file) {
      image = document.createElement("img");
      image.alt = "";
      image.src = assetUrl(person);
    } else if (person.refmod) {
      image = document.createElement("span");
      image.className = "mod-mark";
      image.textContent = "RefMod";
    } else {
      image = document.createElement("span");
      image.className = "voice-mark";
    }
    const name = document.createElement("span");
    name.textContent = person.name;
    button.append(image, name);
    button.addEventListener("click", () => {
      setSource(null);
      state.person = person.id;
      state.clip = null;
      state.item = null;
      state.bin = null;
      renderAll();
    });
    return button;
  }));
}

const CAST_PICTURES = 8;

// A RefMod is a reference saved with the ComfyUI-MiniMaxH3Mod pack. A cast member can carry one,
// beside its pictures or in place of them, and shots with that member render with it.
function refmodSection(person) {
  const current = person.refmod?.file || "";
  const known = state.refmods.files.includes(current);
  const options = [["", "None"], ...state.refmods.files.map((name) => [name, name])];
  if (current && !known) options.splice(1, 0, [current, current + " (not found)"]);
  const pick = choice("RefMod file", options, current, (value) => {
    if (value) person.refmod = { file: value, strength: person.refmod?.strength ?? 1 };
    else if (person.file || person.voice) delete person.refmod;
    else toast(person.name + " is only a RefMod. Remove the cast member instead.", true);
    renderCast();
  });
  const parts = [pick];
  if (person.refmod) {
    const strength = typeof person.refmod.strength === "number" ? person.refmod.strength : 1;
    parts.push(strengthSlider("RefMod strength", person.name + " RefMod strength", strength, [0, 1], 1,
      (v) => { if (person.refmod) person.refmod.strength = v; }));
  }
  const hint = document.createElement("p");
  hint.className = "hint";
  if (!state.refmods.installed) hint.textContent = "Renders leave this RefMod out: the ComfyUI-MiniMaxH3Mod pack is not installed.";
  else if (current && !known) hint.textContent = "Renders leave it out until " + current + " is in models/refmods.";
  else if (!state.refmods.files.length) hint.textContent = "No RefMods in models/refmods yet. Make one with the pack's Create H3 RefMod node.";
  else if (person.refmod && !person.file) hint.textContent = "With no picture, shots render on the prompt-only model, so describe them above. A whole RefMod can make drafts several times slower.";
  else hint.textContent = "A reference saved with the ComfyUI-MiniMaxH3Mod pack, added to every shot with " + person.name + ". A whole RefMod can make drafts several times slower.";
  parts.push(hint);
  return parts;
}

function addRefmodMember() {
  const file = state.refmods.files[0];
  if (!file) return;
  checkpoint();
  state.project.cast = state.project.cast || [];
  const person = { id: uid("p"), name: "Person " + (state.project.cast.length + 1), description: "", refmod: { file, strength: 1 } };
  state.project.cast.push(person);
  state.person = person.id;
  state.clip = null;
  state.item = null;
  save();
  renderAll();
}

// More pictures of the same cast member: other angles, other light, the whole figure.
function morePictures(person) {
  const box = document.createElement("div");
  box.className = "more-pictures";
  const others = (person.pictures || []).filter((p) => p && p.file);
  const row = document.createElement("div");
  row.className = "picture-row";
  for (const asset of others) {
    const cell = document.createElement("div");
    cell.className = "picture";
    const image = document.createElement("img");
    image.src = assetUrl(asset);
    image.alt = "Another picture of " + person.name;
    const remove = document.createElement("button");
    remove.className = "picture-remove";
    remove.textContent = "\u00d7";
    remove.title = "Remove this picture";
    remove.setAttribute("aria-label", "Remove this picture of " + person.name);
    remove.addEventListener("click", () => {
      checkpoint();
      person.pictures = person.pictures.filter((p) => p !== asset);
      if (!person.pictures.length) delete person.pictures;
      save();
      renderInspector();
    });
    cell.append(image, remove);
    row.append(cell);
  }
  if (others.length < CAST_PICTURES - 1) {
    const add = document.createElement("button");
    add.className = "quiet picture-add";
    add.textContent = others.length ? "Add" : "Add more pictures";
    add.title = "Add another picture of " + person.name;
    add.addEventListener("click", async () => {
      const file = await pickFile($("castFile"));
      if (!file) return;
      try {
        const asset = await uploadPicture(file);
        checkpoint();
        person.pictures = [...(person.pictures || []), asset];
        save();
        renderInspector();
      } catch (error) {
        toast(error.message, true);
      }
    });
    row.append(add);
  }
  const hint = document.createElement("p");
  hint.className = "hint";
  hint.textContent = others.length
    ? (others.length + 1) + " of " + CAST_PICTURES + " pictures. A shot takes 9 pictures in all, shared between its cast."
    : "Other angles and the whole figure help a shot keep them the same.";
  box.append(row, hint);
  return box;
}

function renderPerson(panel, person) {
  // The same frame as a clip or a bin: a head with the name, then one section for each thing.
  const head = document.createElement("div");
  head.className = "inspector-head";
  const row = document.createElement("div");
  row.className = "head-row";
  const title = document.createElement("h2");
  title.className = "head-title";
  title.textContent = person.name;
  row.append(title, trashButton("Remove " + person.name + " from the cast", () => {
    checkpoint();
    state.project.cast = state.project.cast.filter((c) => c !== person);
    for (const clip of state.project.clips) clip.cast = (clip.cast || []).filter((id) => id !== person.id);
    state.person = null;
    save();
    renderAll();
    toast("Removed " + person.name + " from the cast.", false, undoAction);
  }));
  const line = document.createElement("p");
  line.className = "status";
  const has = [];
  if (person.file) has.push(1 + (person.pictures || []).length === 1 ? "1 picture" : 1 + (person.pictures || []).length + " pictures");
  if (person.voice) has.push("a voice");
  if (person.refmod) has.push("a RefMod");
  const used = state.project.clips.filter((c) => (c.cast || []).includes(person.id)).length;
  line.textContent = has.join(", ") + (used ? ". In " + used + (used === 1 ? " clip" : " clips") : ". Not in a clip yet");
  head.append(row, line);
  const parts = [head];
  if (person.file) {
    const image = document.createElement("img");
    image.className = "portrait";
    image.alt = person.name;
    image.src = assetUrl(person);
    parts.push(section("Pictures", [image, morePictures(person)]));
  }
  parts.push(
    section("About", [
      field("Name", person.name, (v) => {
        person.name = v.trim() || person.name;
        title.textContent = person.name;
        renderCast();
      }),
      person.file || person.refmod
        ? field("What they look like", person.description, (v) => (person.description = v), { big: true })
        : field("How they sound", person.description, (v) => (person.description = v), { area: true, placeholder: "A warm, calm female narrator" }),
    ]),
    section("Voice", [voiceSection(person)]),
  );
  if (state.refmods.installed || person.refmod) parts.push(section("RefMod", refmodSection(person)));
  panel.replaceChildren(...parts);
}

const MERGE_FRAMES = 63;

// Replaces the end of one clip and the start of the next with a generated crossing.
// The scene keeps its length, because the new clip fills exactly what was trimmed away.
function mergeWithNext() {
  const item = selectedRendered();
  const next = item && followingItem(item);
  if (!item || !next || !activeTake(clipOf(next.clip))) return;
  if (next.start !== itemEnd(item)) {
    toast("Put the two clips side by side to merge them.", true);
    return;
  }
  const away = Math.ceil(MERGE_FRAMES / 2);
  const rest = MERGE_FRAMES - away;
  if (speedOf(item) !== 1 || speedOf(next) !== 1) {
    toast("Set both clips to 100% speed to merge them.", true);
    return;
  }
  if (item.out - item.in < HOLD + away || next.out - next.in < HOLD + rest) {
    toast("Both clips need at least " + (HOLD + away) + " frames on the timeline to merge.", true);
    return;
  }
  const track = continuedTrack(clipOf(item.clip), item, "bridge", 0, item.out - away);
  if (track === false) return;
  checkpoint();
  item.out -= away;
  next.in += rest;
  next.start += rest;
  const source = clipOf(item.clip);
  const clip = {
    id: uid("c"), name: "Merge from " + source.name,
    action: "One continuous shot that moves naturally from the opening frames into the closing frames.",
    camera: "", speaker: "", dialogue: "", sound: "", music: "", seconds: 2.625, takes: [], active: null,
    link: { mode: "bridge", hold: HOLD, a: sideOf(item), b: sideOf(next) },
  };
  if (source.cast?.length) clip.cast = [...source.cast];
  if (binOf(source.bin)) clip.bin = source.bin;
  if (track) clip.track = track;
  state.project.clips.push(clip);
  const [first, last] = linkedRange(clip.link, linkedFrames(clip.link, clip.seconds));
  const placed = { id: uid("i"), clip: clip.id, start: itemEnd(item), in: first, out: last };
  state.project.timeline.push(placed);
  state.clip = clip.id;
  state.person = null;
  state.item = placed.id;
  save();
  renderAll();
}

/* Timeline editing */

function gapAt(frame) {
  let end = 0;
  for (const item of sorted()) {
    if (frame >= end && frame < item.start) return { start: end, end: item.start };
    end = Math.max(end, itemEnd(item));
  }
  return null;
}

function closeGap() {
  if (!state.gap) return;
  checkpoint();
  const shift = state.gap.end - state.gap.start;
  for (const item of state.project.timeline) {
    if (item.start >= state.gap.end) item.start -= shift;
  }
  state.gap = null;
  save();
  renderTimeline();
  showFrame();
}

function rippleDelete() {
  const item = state.project.timeline.find((i) => i.id === state.item);
  if (!item) {
    toast("Select a clip on the timeline first.");
    return;
  }
  checkpoint();
  const end = itemEnd(item);
  const shift = itemLength(item) - (item.transition?.frames || 0);
  state.project.timeline = state.project.timeline.filter((i) => i !== item);
  for (const other of state.project.timeline) {
    if (other.start >= end - (other.transition?.frames || 0)) other.start -= shift;
  }
  state.item = null;
  save();
  renderTimeline();
  showFrame();
}

function editPoints() {
  const points = new Set([0]);
  for (const item of state.project.timeline) {
    points.add(item.start);
    points.add(itemEnd(item));
  }
  return [...points].sort((a, b) => a - b);
}

function jumpEdit(step) {
  setSource(null);
  const points = editPoints();
  const target = step > 0 ? points.find((p) => p > state.playhead) : points.filter((p) => p < state.playhead).at(-1);
  if (target !== undefined) seek(target);
}

function setZoom(value, anchor = null) {
  const scroller = $("tlScroll");
  const frame = anchor === null ? null : (scroller.scrollLeft + anchor) / state.zoom;
  state.zoom = Math.min(12, Math.max(0.1, value));
  $("zoom").value = String(state.zoom);
  renderTimeline();
  if (frame !== null) scroller.scrollLeft = frame * state.zoom - anchor;
}

function fitTimeline() {
  setZoom(($("tlScroll").clientWidth - 40) / Math.max(sceneEnd(), FPS));
  $("tlScroll").scrollLeft = 0;
}

function selectClip(clipId, itemId = null) {
  if (itemId || clipId !== state.source) setSource(null);
  state.clip = clipId;
  state.item = itemId;
  state.gap = null;
  state.person = null;
  state.bin = null;
  renderCast();
  renderBin();
  renderInspector();
  renderTimeline();
  showFrame();
}

function clipStatus(clip) {
  const job = jobFor(clip.id);
  if (job) {
    if (state.running !== job) return { kind: "live", text: "Queued" };
    return { kind: "live", text: job.max ? "Rendering " + Math.round((job.progress || 0) * 100) + "%" : "Rendering" };
  }
  if (clip.error) return { kind: "danger", text: "Render failed" };
  if (!clip.takes.length) return { kind: "muted", text: "Not rendered" };
  if (clip.kind === "footage") return { kind: "", text: "Footage" };
  if (isStale(clip)) return { kind: "danger", text: "Source changed" };
  const take = activeTake(clip);
  const quality = take.quality === "final" ? "Final" : "Draft";
  return { kind: "", text: clip.takes.length > 1 ? quality + ", take " + (clip.takes.indexOf(take) + 1) + " of " + clip.takes.length : quality };
}

function clipSeconds(clip) {
  const take = activeTake(clip);
  return (take ? take.frames / FPS : clip.seconds).toFixed(1) + " s";
}

// Run the pointer across a thumbnail to scrub through the clip, frame strip permitting.
function skim(thumb, take) {
  const poster = "url(" + viewUrl(take.poster, take.subfolder) + ")";
  thumb.style.backgroundImage = poster;
  if (!take.strip) return;
  const sprite = "url(" + viewUrl(take.strip.sprite, take.subfolder) + ")";
  const line = document.createElement("span");
  line.className = "skim";
  line.hidden = true;
  thumb.append(line);
  thumb.addEventListener("pointermove", (event) => {
    const rect = thumb.getBoundingClientRect();
    const share = Math.min(0.999, Math.max(0, (event.clientX - rect.left) / rect.width));
    const { count, w, h } = take.strip;
    const index = Math.floor(share * count);
    // Each frame covers the thumbnail like the poster does, cropped to fit and never squeezed.
    const scale = w && h ? Math.max(rect.width / w, rect.height / h) : 0;
    const tileW = scale ? w * scale : rect.width;
    const tileH = scale ? h * scale : rect.height;
    thumb.style.backgroundImage = sprite;
    thumb.style.backgroundSize = count * tileW + "px " + tileH + "px";
    thumb.style.backgroundPosition = (rect.width - tileW) / 2 - index * tileW + "px " + (rect.height - tileH) / 2 + "px";
    line.hidden = false;
    line.style.left = share * 100 + "%";
  });
  thumb.addEventListener("pointerleave", () => {
    thumb.style.backgroundImage = poster;
    thumb.style.backgroundSize = "";
    thumb.style.backgroundPosition = "";
    line.hidden = true;
  });
}

// Bins are folders in the Clips panel. Folding one is a per-viewer convenience, kept in this browser only.
function loadFolded() {
  try {
    state.folded = new Set(JSON.parse(localStorage.getItem("h3take.folded." + state.folder) || "[]"));
  } catch (_) {
    state.folded = new Set();
  }
}

function saveFolded() {
  try { localStorage.setItem("h3take.folded." + state.folder, JSON.stringify([...state.folded])); } catch (_) {}
}

function focusInBin(selector) {
  $("clips").querySelector(selector)?.focus({ preventScroll: true });
}

function toggleBin(binId) {
  if (state.folded.has(binId)) state.folded.delete(binId);
  else state.folded.add(binId);
  saveFolded();
  renderBin();
  focusInBin('.bin-group[data-bin="' + CSS.escape(binId) + '"] .fold');
}

function selectBin(binId) {
  setSource(null);
  state.bin = binId;
  state.clip = null;
  state.item = null;
  state.person = null;
  state.gap = null;
  renderAll();
  focusInBin('.bin-group[data-bin="' + CSS.escape(binId) + '"] .bin-name');
}

function addBin() {
  checkpoint();
  state.project.bins = state.project.bins || [];
  let number = state.project.bins.length + 1;
  while (state.project.bins.some((b) => b.name === "Bin " + number)) number++;
  const bin = { id: uid("b"), name: "Bin " + number };
  state.project.bins.push(bin);
  save();
  selectBin(bin.id);
  const name = $("inspector").querySelector('[data-key="name"]');
  if (!name) return;
  // Naming a new bin is part of making it, so one undo takes back both.
  state.quiet = true;
  name.focus();
  state.quiet = false;
  name.select();
}

function deleteBin(bin) {
  checkpoint();
  const freed = state.project.clips.filter((c) => c.bin === bin.id);
  for (const clip of freed) delete clip.bin;
  state.project.bins = state.project.bins.filter((b) => b !== bin);
  if (state.bin === bin.id) state.bin = null;
  state.folded.delete(bin.id);
  saveFolded();
  save();
  renderAll();
  const kept = freed.length === 1 ? " Its clip stays in the list." : freed.length ? " Its " + freed.length + " clips stay in the list." : "";
  toast("Deleted " + bin.name + "." + kept, false, undoAction);
}

function fileClip(clip, binId) {
  checkpoint();
  if (binId) clip.bin = binId;
  else delete clip.bin;
  save();
  renderBin();
  if (selectedBin()) renderInspector();
}

// Deletes the clip from the project, with its places on the timeline. Its render files stay on disk.
function deleteClip(clip) {
  if (!clip) return;
  if (jobFor(clip.id)) {
    toast(clip.name + " is rendering. Cancel its render, then delete it.", true);
    return;
  }
  checkpoint();
  if (state.source === clip.id) setSource(null);
  const placed = state.project.timeline.some((i) => i.clip === clip.id);
  state.project.clips = state.project.clips.filter((c) => c !== clip);
  state.project.timeline = state.project.timeline.filter((i) => i.clip !== clip.id);
  if (state.clip === clip.id) state.clip = null;
  if (!state.project.timeline.some((i) => i.id === state.item)) state.item = null;
  save();
  renderAll();
  toast("Deleted " + clip.name + (placed ? " and took it off the timeline" : "") + ".", false, undoAction);
}

function trashButton(label, onClick) {
  const button = document.createElement("button");
  button.className = "icon delete";
  button.setAttribute("aria-label", label);
  button.title = label;
  button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.5 7h15"/><path d="M9.5 7V4.8h5V7"/><path d="M6.5 7l.8 11.6a1.6 1.6 0 0 0 1.6 1.4h6.2a1.6 1.6 0 0 0 1.6-1.4L17.5 7"/><path d="M10.2 10.8v5.6M13.8 10.8v5.6"/></svg>';
  button.addEventListener("click", onClick);
  return button;
}

function renderBin() {
  const list = $("clips");
  const clips = state.project.clips;
  const bins = state.project.bins || [];
  $("clipCount").textContent = clips.length ? String(clips.length) : "";
  if (!clips.length && !bins.length) {
    list.innerHTML = '<p class="empty">Add a clip, describe the shot, and render a draft.</p>';
    return;
  }
  const filed = new Set(bins.map((b) => b.id));
  list.replaceChildren(
    ...bins.map((bin) => binGroup(bin, clips.filter((c) => c.bin === bin.id))),
    ...clips.filter((c) => !filed.has(c.bin)).map(clipRow),
  );
  list.querySelector(".clip.on, .bin-head.on")?.scrollIntoView({ block: "nearest" });
}

function binGroup(bin, inside) {
  const open = !state.folded.has(bin.id);
  const group = document.createElement("div");
  group.className = "bin-group";
  group.dataset.bin = bin.id;
  const head = document.createElement("div");
  head.className = "bin-head" + (selectedBin() === bin ? " on" : "") + (!open && inside.some((c) => c.id === state.clip) ? " holds" : "");
  const fold = document.createElement("button");
  fold.className = "fold";
  fold.setAttribute("aria-expanded", String(open));
  fold.setAttribute("aria-label", (open ? "Fold " : "Unfold ") + bin.name);
  fold.title = open ? "Fold the bin" : "Unfold the bin";
  fold.addEventListener("click", () => toggleBin(bin.id));
  const name = document.createElement("button");
  name.className = "bin-name";
  name.addEventListener("click", () => selectBin(bin.id));
  name.innerHTML = '<svg class="folder" viewBox="0 0 24 24" aria-hidden="true"><path d="M3.5 7.2a1.7 1.7 0 0 1 1.7-1.7h3.9l2 2.1h7.7a1.7 1.7 0 0 1 1.7 1.7v7.5a1.7 1.7 0 0 1-1.7 1.7H5.2a1.7 1.7 0 0 1-1.7-1.7z"/></svg>';
  const label = document.createElement("span");
  label.className = "label";
  label.textContent = bin.name;
  const count = document.createElement("span");
  count.className = "count";
  count.textContent = String(inside.length);
  name.append(label, count);
  head.append(fold, name);
  group.append(head);
  if (open) {
    const body = document.createElement("div");
    body.className = "bin-clips";
    if (inside.length) body.append(...inside.map(clipRow));
    else {
      const empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "Drag clips here.";
      body.append(empty);
    }
    group.append(body);
  }
  return group;
}

let draggedClip = null;

function clipRow(clip) {
  const take = activeTake(clip);
  const status = clipStatus(clip);
  const button = document.createElement("button");
  button.className = "clip" + (clip.id === state.clip ? " on" : "");
  button.setAttribute("aria-pressed", String(clip.id === state.clip));
  button.dataset.clip = clip.id;
  button.draggable = true;
  button.addEventListener("click", () => {
    setSource(clip.id);
    selectClip(clip.id);
    focusInBin('.clip[data-clip="' + CSS.escape(clip.id) + '"]');
  });
  button.addEventListener("dragstart", (event) => {
    draggedClip = clip.id;
    event.dataTransfer.setData("text/h3clip", clip.id);
  });
  button.addEventListener("dragend", () => {
    draggedClip = null;
    showDrop(undefined);
  });
  const thumb = document.createElement("span");
  thumb.className = "thumb" + (take ? "" : " none");
  if (take) skim(thumb, take);
  const text = document.createElement("span");
  text.className = "text";
  const title = document.createElement("span");
  title.className = "title";
  title.textContent = clip.name;
  const meta = document.createElement("span");
  meta.className = "meta";
  const length = document.createElement("span");
  length.textContent = clipSeconds(clip);
  const state_ = document.createElement("span");
  state_.className = "state " + status.kind;
  state_.dataset.status = clip.id;
  state_.textContent = status.text;
  meta.append(length, state_);
  text.append(title, meta);
  button.append(thumb, text);
  return button;
}

// Where a clip dragged over the Clips panel would land: a bin, or null for the loose list.
function dropBin(event) {
  return event.target.closest?.(".bin-group")?.dataset.bin || null;
}

function showDrop(target) {
  const list = $("clips");
  list.classList.toggle("drop", target === null);
  for (const group of list.querySelectorAll(".bin-group")) group.classList.toggle("drop", group.dataset.bin === target);
}

function field(label, value, onInput, { big = false, area = false, type = "text", placeholder = "" } = {}) {
  const wrap = document.createElement("label");
  wrap.className = "field";
  const title = document.createElement("span");
  title.textContent = label;
  const input = document.createElement(area || big ? "textarea" : "input");
  if (big) input.className = "big";
  if (!area && !big) input.type = type;
  if (placeholder) input.placeholder = placeholder;
  input.dataset.key = label;
  input.value = value ?? "";
  input.addEventListener("focus", () => checkpoint(), { once: true });
  input.addEventListener("input", () => {
    onInput(input.value);
    save();
  });
  wrap.append(title, input);
  return wrap;
}

function section(title, children) {
  const box = document.createElement("section");
  box.className = "section";
  const head = document.createElement("h3");
  head.textContent = title;
  box.append(head, ...children);
  return box;
}

function disclosure(title, open, note, children) {
  const box = document.createElement("details");
  box.className = "section";
  box.open = open;
  const summary = document.createElement("summary");
  summary.textContent = title;
  if (note) {
    const filled = document.createElement("span");
    filled.className = "filled";
    filled.textContent = note;
    summary.append(filled);
  }
  box.append(summary, ...children);
  return box;
}

const fileName = (file) => file.replaceAll("\\", "/").split("/").pop().replace(/\.(safetensors|gguf|ckpt|pt|pth)$/i, "");
const isCheckpoint = (row) => row && typeof row.id === "string" && typeof row.unet === "string" && row.unet !== "";

// A checkpoint of the user's own, shaped like the built-in profiles the server lists.
function asProfile(row) {
  const base = state.profiles.find((p) => p.id === (row.base || "h3"));
  const missing = [row.unet, row.unet_text].filter((f) => f && !state.checkpoints.some((c) => sameFile(c, f)));
  return {
    id: row.id, title: row.title || fileName(row.unet), own: true, available: !missing.length, missing,
    hint: "Your own checkpoint, on the settings of " + (base?.title || "MiniMax H3") + ".",
  };
}

const ownProfiles = () => (state.project?.checkpoints || []).filter(isCheckpoint).map(asProfile);
const allProfiles = () => [...state.profiles, ...ownProfiles()];
// The presets this project has no copy of yet. They are offered wherever a model is picked.
const sparePresets = () => state.presets.filter((row) => isCheckpoint(row) && !(state.project?.checkpoints || []).some((r) => r && r.id === row.id));
const menuProfiles = () => [...allProfiles(), ...sparePresets().map(asProfile)];

// A project renders on its own copy of a preset, so picking one brings the copy in.
function adoptPreset(id) {
  const preset = sparePresets().find((row) => row.id === id);
  if (preset) state.project.checkpoints = [...(state.project.checkpoints || []), structuredClone(preset)];
}

// What a checkpoint says, in one order, to tell a project's copy from the preset it came from.
const checkpointText = (row) => JSON.stringify([row.title || "", row.unet, row.unet_text || "", row.base || "h3",
  row.steps?.draft ?? null, row.steps?.final ?? null, row.sampler || "", row.scheduler || "", row.lora !== false]);

async function savePresets(rows, done) {
  try {
    state.presets = (await postJson("/presets", { checkpoints: rows }, "PUT")).checkpoints;
    toast(done);
  } catch (error) {
    toast(error.message, true);
  }
  renderInspector();
}
const profileOf = (id) => allProfiles().find((p) => p.id === id);
// The model the project renders with: the one it names, or the first one installed, as the server does.
const projectProfile = () => profileOf(state.project.profile) || state.profiles.find((p) => p.available) || state.profiles[0];

function choice(label, options, value, onChange) {
  const wrap = document.createElement("label");
  wrap.className = "choice";
  const title = document.createElement("span");
  title.textContent = label;
  const select = document.createElement("select");
  select.dataset.key = label;
  for (const [id, text] of options) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = text;
    option.selected = id === value;
    select.append(option);
  }
  select.addEventListener("change", () => {
    checkpoint();
    onChange(select.value);
    save();
    renderInspector();
    if ($("models").open) renderModels();
  });
  wrap.append(title, select);
  return wrap;
}

// H3's text encoder and video VAE come in several precisions. A project renders with the first
// one installed unless it names another here.
function encoderChoices() {
  const rows = [["Text encoder", "clip", state.encoders.text_encoders, state.encoders.auto.text_encoder],
    ["Video VAE", "vae", state.encoders.vaes, state.encoders.auto.vae]];
  const out = [];
  for (const [label, key, files, auto] of rows) {
    const current = state.project.models?.[key] || "";
    // Nothing to choose between, and nothing chosen: no menu.
    if (files.length < 2 && !current) continue;
    const options = [["", "Automatic" + (auto ? ": " + fileName(auto) : "")], ...files.map((f) => [f, fileName(f)])];
    if (current && !files.some((f) => sameFile(f, current))) options.push([current, fileName(current) + " (not installed)"]);
    out.push(choice(label, options, current, (value) => {
      const models = { ...(state.project.models || {}) };
      if (value) models[key] = value;
      else delete models[key];
      if (Object.keys(models).length) state.project.models = models;
      else delete state.project.models;
    }));
  }
  if (out.length) {
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent = "The int8 and nvfp4 files are smaller and load faster; bf16 and fp16 are the full precision. New renders use the choice.";
    out.push(hint);
  }
  return out;
}

// The checkpoint being edited in the Models dialog, or null while it shows the list.
let modelsEditing = null;

// What a model renders with, in one line: its file, steps, sampler, scheduler and any LoRA it brings.
function modelSummary(row) {
  const custom = isCheckpoint(row);
  const base = custom ? state.profiles.find((p) => p.id === (row.base || "h3")) || {} : row;
  const steps = (quality) => (custom && row.steps?.[quality]) || base.steps?.[quality] || "?";
  const lora = custom && row.lora === false ? null : base.lora;
  const parts = [
    steps("draft") + " and " + steps("final") + " steps",
    ((custom && row.sampler) || base.sampler || "") + " / " + ((custom && row.scheduler) || base.scheduler || "simple"),
  ];
  if (lora) parts.push("adds " + fileName(lora));
  parts.push(fileName(row.unet || ""));
  return parts.filter(Boolean).join(", ");
}

function openModels(editing = null) {
  modelsEditing = editing;
  renderModels();
  if (!$("models").open) $("models").showModal();
}

// Every model the project can render with, in one place: the built-in ones, the presets kept
// for every project, and this project's own. A checkpoint of the user's is edited here too.
function renderModels() {
  const body = $("modelsBody");
  const foot = $("modelsFoot");
  const rows = (state.project.checkpoints || []).filter(isCheckpoint);
  const live = (id) => (state.project.checkpoints || []).find((r) => r && r.id === id) || {};
  const redraw = () => {
    renderInspector();
    renderModels();
  };
  const button = (text, run, kind = "quiet small") => {
    const made = document.createElement("button");
    made.className = kind;
    made.textContent = text;
    made.addEventListener("click", run);
    return made;
  };
  const use = (id) => {
    checkpoint();
    adoptPreset(id);
    state.project.profile = id;
    save();
    redraw();
  };
  const editing = rows.find((r) => r.id === modelsEditing);
  if (!editing) modelsEditing = null;

  if (editing) {
    const row = editing;
    const base = state.profiles.find((p) => p.id === (row.base || "h3")) || state.profiles[0] || { title: "MiniMax H3", steps: {} };
    const shown = ownProfiles().find((p) => p.id === row.id);
    const files = (current) => {
      const list = state.checkpoints.map((f) => [f, f]);
      if (current && !state.checkpoints.some((f) => sameFile(f, current))) list.unshift([current, current + " (not installed)"]);
      return list;
    };
    const text = (label, value, onInput, options) => {
      const made = field(label, value, onInput, options);
      // The name and the numbers show in the list and the panel, so both are drawn again once the field is left.
      made.querySelector("input").addEventListener("change", redraw);
      return made;
    };
    const steps = (quality, label) => text(label, row.steps?.[quality] ?? "", (v) => {
      const target = live(row.id);
      const count = Number(v);
      const kept = { ...(target.steps || {}) };
      if (v.trim() !== "" && Number.isInteger(count) && count >= 1 && count <= 100) kept[quality] = count;
      else delete kept[quality];
      if (Object.keys(kept).length) target.steps = kept;
      else delete target.steps;
    }, { type: "number", placeholder: String(base.steps?.[quality] ?? "") });
    const pair = document.createElement("div");
    pair.className = "halves";
    pair.append(steps("draft", "Draft steps"), steps("final", "Final steps"));
    const set = (key) => (value) => {
      if (value) live(row.id)[key] = value;
      else delete live(row.id)[key];
    };
    const back = button("All models", () => openModels(null), "quiet small");
    const title = document.createElement("h3");
    title.className = "model-editing";
    title.textContent = row.title || fileName(row.unet);
    const children = [
      back, title,
      text("Name", row.title || "", (v) => (live(row.id).title = v), { placeholder: fileName(row.unet) }),
      choice("Model file", files(row.unet), row.unet, set("unet")),
      choice("Model file for clips that are only a prompt", [["", "The same file"], ...files(row.unet_text)], row.unet_text || "", set("unet_text")),
      choice("Settings from", state.profiles.map((p) => [p.id, p.title + ": " + modelSummary(p).split(", ").slice(1).join(", ")]), row.base || "h3", set("base")),
    ];
    if (base.lora) {
      // A family such as MiniMax H3 Turbo samples with a LoRA. A checkpoint with that built in must not get it twice.
      const keep = document.createElement("label");
      keep.className = "check";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = row.lora !== false;
      box.dataset.key = "Family LoRA";
      box.addEventListener("change", () => {
        checkpoint();
        if (box.checked) delete live(row.id).lora;
        else live(row.id).lora = false;
        save();
        redraw();
      });
      const label = document.createElement("span");
      label.textContent = "Add " + base.title + "'s LoRA, " + fileName(base.lora);
      keep.append(box, label);
      const why = document.createElement("p");
      why.className = "hint";
      why.textContent = "Untick it for a checkpoint that already has its turbo built in.";
      children.push(keep, why);
    }
    children.push(pair, choice("Sampler", [["", "As " + base.title + ": " + (base.sampler || "")], ...state.samplers.map((n) => [n, n])], row.sampler || "", set("sampler")));
    if (/hyperflow/i.test(fileName(row.unet)) || /\.gguf$/i.test(row.unet)) {
      const own = document.createElement("p");
      own.className = "hint";
      own.textContent = "A HyperFlow file samples on its own schedule, so there is no scheduler to pick.";
      children.push(own);
    } else {
      children.push(choice("Scheduler", [["", "As " + base.title + ": " + (base.scheduler || "simple")], ...state.schedulers.map((n) => [n, n])], row.scheduler || "", set("scheduler")));
    }
    const about = document.createElement("p");
    about.className = "hint" + (shown && !shown.available ? " bad" : "");
    about.textContent = shown && !shown.available
      ? "Not installed. Needs " + shown.missing.join(" and ") + " in models/diffusion_models."
      : "Renders with: " + modelSummary(row) + ". Empty steps follow " + base.title + ", and so does the way finals are refined.";
    children.push(about);
    body.replaceChildren(...children);

    const inUse = state.project.profile === row.id;
    const preset = state.presets.find((r) => r.id === row.id);
    const same = preset && checkpointText(preset) === checkpointText(row);
    const keep = button(!preset ? "Save as a preset" : same ? "Saved as a preset" : "Update the preset", async () => {
      await flush();
      const mine = structuredClone(live(row.id));
      await savePresets(preset ? state.presets.map((r) => (r.id === mine.id ? mine : r)) : [...state.presets, mine],
        preset ? "Updated the preset." : "Saved. Every project now offers it.");
      renderModels();
    }, "quiet");
    keep.disabled = Boolean(same);
    keep.title = "A preset is offered in every project";
    const go = button(inUse ? "The project renders with it" : "Render the project with it", () => use(row.id), "primary");
    go.disabled = inUse || (shown && !shown.available);
    const spacer = document.createElement("span");
    spacer.className = "spacer";
    foot.replaceChildren(trashButton("Remove " + (row.title || fileName(row.unet)) + " from this project", () => {
      checkpoint();
      state.project.checkpoints = state.project.checkpoints.filter((r) => !r || r.id !== row.id);
      // What rendered on it goes back to the first model installed, as the server would pick.
      if (inUse) state.project.profile = (state.profiles.find((p) => p.available) || state.profiles[0])?.id;
      modelsEditing = null;
      save();
      redraw();
    }), keep, spacer, go);
    return;
  }

  const current = projectProfile()?.id;
  const line = (row, status, actions) => {
    const item = document.createElement("div");
    item.className = "model-row" + (row.id === current ? " on" : "");
    const main = document.createElement("div");
    main.className = "model-main";
    const name = document.createElement("b");
    name.textContent = row.title || fileName(row.unet);
    const sub = document.createElement("p");
    sub.className = "model-sub" + (status.bad ? " bad" : "");
    sub.textContent = status.text;
    sub.title = status.text;
    main.append(name, sub);
    const side = document.createElement("div");
    side.className = "model-actions";
    if (row.id === current) {
      const badge = document.createElement("span");
      badge.className = "model-badge";
      badge.textContent = "In use";
      side.append(badge);
    }
    side.append(...actions);
    item.append(main, side);
    return item;
  };
  const group = (title, items, note) => {
    const box = document.createElement("div");
    box.className = "model-group";
    const head = document.createElement("h3");
    head.textContent = title;
    box.append(head);
    if (note) {
      const hint = document.createElement("p");
      hint.className = "hint";
      hint.textContent = note;
      box.append(hint);
    }
    box.append(...items);
    return box;
  };
  const builtIn = state.profiles.map((p) => line(p,
    p.available ? { text: modelSummary(p) } : { bad: true, text: "Not installed. Needs " + p.missing.join(" and ") + "." },
    p.available && p.id !== current ? [button("Use", () => use(p.id))] : []));
  const mine = rows.map((row) => {
    const shown = ownProfiles().find((p) => p.id === row.id);
    const ok = !shown || shown.available;
    return line(row, ok ? { text: modelSummary(row) } : { bad: true, text: "Not installed. Needs " + shown.missing.join(" and ") + "." }, [
      ...(ok && row.id !== current ? [button("Use", () => use(row.id))] : []),
      button("Edit", () => openModels(row.id)),
    ]);
  });
  const spare = sparePresets().map((row) => line(row, { text: modelSummary(row) }, [
    button("Add to project", () => {
      checkpoint();
      adoptPreset(row.id);
      save();
      redraw();
    }),
    trashButton("Delete the preset " + (row.title || fileName(row.unet)), async () => {
      await savePresets(state.presets.filter((r) => r.id !== row.id), "Deleted the preset. Projects that use it keep their copy.");
      renderModels();
    }),
  ]));
  const groups = [group("Built in", builtIn)];
  if (mine.length) groups.push(group("In this project", mine));
  if (spare.length) groups.push(group("Your presets", spare, "Kept for every project. Adding one copies it into this project."));
  body.replaceChildren(...groups);

  const add = button("Add a checkpoint", async () => {
    const file = await pickFile2("Add a checkpoint", state.checkpoints, "From models/diffusion_models. Pick a MiniMax H3 checkpoint, such as a community fine-tune.");
    if (!file) return;
    checkpoint();
    const id = "m" + crypto.randomUUID().replaceAll("-", "").slice(0, 8);
    state.project.checkpoints = [...(state.project.checkpoints || []), { id, title: fileName(file), unet: file, base: "h3" }];
    save();
    renderInspector();
    openModels(id);
  }, "quiet");
  const note = document.createElement("span");
  note.className = "model-note";
  note.textContent = "Any MiniMax H3 checkpoint works: add its file, then set what it renders with.";
  foot.replaceChildren(add, note);
}

// A clip renders on the project's model unless it picks its own here.
function clipModel(clip) {
  const wrap = document.createElement("label");
  wrap.className = "choice";
  const title = document.createElement("span");
  title.textContent = "This clip renders with";
  const select = document.createElement("select");
  select.dataset.key = "Clip model";
  const own = profileOf(clip.profile);
  const usual = document.createElement("option");
  usual.value = "";
  usual.textContent = "The project's model" + (projectProfile() ? ": " + projectProfile().title : "");
  usual.selected = !own;
  select.append(usual);
  for (const profile of menuProfiles()) {
    const option = document.createElement("option");
    option.value = profile.id;
    option.textContent = profile.available ? profile.title : profile.title + " (not installed)";
    option.disabled = !profile.available && profile.id !== own?.id;
    option.selected = profile.id === own?.id;
    select.append(option);
  }
  const hint = document.createElement("p");
  hint.className = "hint";
  const describe = () => {
    const picked = profileOf(select.value);
    if (!picked) hint.textContent = "Pick another model to render only this clip with it. Takes keep the model that made them.";
    else if (!picked.available) hint.textContent = "Not installed. Needs " + picked.missing.join(" and ") + ".";
    else hint.textContent = picked.hint + " The next render of this clip uses it. Switching between models loads the new one first, which takes a while.";
  };
  describe();
  select.addEventListener("change", () => {
    checkpoint();
    const live = clipOf(clip.id);
    adoptPreset(select.value);
    if (select.value) live.profile = select.value;
    else delete live.profile;
    save();
    renderInspector();
  });
  wrap.append(title, select);
  return [wrap, hint];
}

// A strength slider with its number beside it. The fill runs from zero to the thumb, so a
// negative strength reads as one. Dragging saves without a re-render, so the thumb is never lost.
function strengthSlider(key, label, value, [min, max], usual, onChange) {
  const wrap = document.createElement("div");
  wrap.className = "strength";
  const range = document.createElement("input");
  range.type = "range";
  range.dataset.key = key;
  range.setAttribute("aria-label", label);
  range.title = "Double-click for " + usual.toFixed(2);
  const amount = document.createElement("input");
  amount.type = "number";
  amount.className = "amount";
  amount.dataset.key = key + " amount";
  amount.setAttribute("aria-label", label + ", as a number");
  for (const input of [range, amount]) {
    input.min = min;
    input.max = max;
    input.addEventListener("focus", () => checkpoint(), { once: true });
  }
  range.step = 0.05;
  amount.step = 0.05;
  const show = (v, from) => {
    if (from !== range) range.value = v;
    if (from !== amount) amount.value = v.toFixed(2);
    const at = (x) => (Math.min(max, Math.max(min, x)) - min) / (max - min);
    range.style.setProperty("--from", Math.min(at(0), at(v)));
    range.style.setProperty("--to", Math.max(at(0), at(v)));
  };
  const set = (raw, from) => {
    const v = Math.round(Math.min(max, Math.max(min, Number(raw) || 0)) * 100) / 100;
    show(v, from);
    onChange(v);
    save();
  };
  show(value, null);
  range.addEventListener("input", () => set(range.value, range));
  range.addEventListener("dblclick", () => set(usual, null));
  amount.addEventListener("input", () => { if (amount.value !== "" && amount.validity.valid) set(amount.value, amount); });
  amount.addEventListener("change", () => set(amount.value, null));
  wrap.append(range, amount);
  return wrap;
}

// A searchable list in a dialog, for folders that hold hundreds of files. Resolves to the file picked, or null.
function pickFile2(title, files, hint) {
  const dialog = $("picker");
  const search = $("pickerSearch");
  const list = $("pickerList");
  $("pickerTitle").textContent = title;
  $("pickerHint").textContent = hint;
  search.value = "";
  search.placeholder = "Search " + files.length + (files.length === 1 ? " file" : " files");
  let active = 0;
  let shown = [];
  return new Promise((resolve) => {
    const finish = (file) => {
      search.removeEventListener("input", draw);
      search.removeEventListener("keydown", keys);
      dialog.removeEventListener("close", closed);
      $("pickerClose").removeEventListener("click", cancel);
      if (dialog.open) dialog.close();
      resolve(file);
    };
    const cancel = () => finish(null);
    const closed = () => finish(null);
    function draw(event) {
      if (event) active = 0;
      // Every word typed must be somewhere in the name, in any order.
      const words = search.value.toLowerCase().split(/\s+/).filter(Boolean);
      shown = files.filter((f) => words.every((w) => f.toLowerCase().includes(w)));
      active = Math.min(active, Math.max(0, shown.length - 1));
      $("pickerCount").textContent = words.length ? shown.length + " of " + files.length : "";
      if (!shown.length) {
        const none = document.createElement("p");
        none.className = "empty";
        none.textContent = "Nothing matches. Try fewer letters.";
        list.replaceChildren(none);
        return;
      }
      list.replaceChildren(...shown.map((file, index) => {
        const item = document.createElement("button");
        item.className = "picker-item" + (index === active ? " on" : "");
        item.setAttribute("role", "option");
        item.setAttribute("aria-selected", String(index === active));
        item.title = file;
        const name = document.createElement("b");
        name.textContent = loraName(file);
        item.append(name);
        const folder = file.replaceAll("\\", "/").split("/").slice(0, -1).join("/");
        if (folder) {
          const where = document.createElement("span");
          where.textContent = folder;
          item.append(where);
        }
        item.addEventListener("click", () => finish(file));
        return item;
      }));
      list.children[active]?.scrollIntoView({ block: "nearest" });
    }
    function keys(event) {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        active = Math.max(0, Math.min(shown.length - 1, active + (event.key === "ArrowDown" ? 1 : -1)));
        draw();
      } else if (event.key === "Enter" && shown[active]) {
        event.preventDefault();
        finish(shown[active]);
      }
    }
    search.addEventListener("input", draw);
    search.addEventListener("keydown", keys);
    dialog.addEventListener("close", closed);
    $("pickerClose").addEventListener("click", cancel);
    draw();
    dialog.showModal();
    search.focus();
  });
}

const loraName = (file) => file.replaceAll("\\", "/").split("/").pop().replace(/\.(safetensors|ckpt|pt|pth)$/i, "");
const sameFile = (a, b) => a.replaceAll("\\", "/") === b.replaceAll("\\", "/");

// The project's LoRAs: every render samples with them, each at its own strength.
function loraSection() {
  const rows = (state.project.loras || []).filter((row) => row && typeof row.file === "string");
  const limits = state.realism || { lora_range: [-2, 2], range: [0, 3], limit: 8 };
  const change = (edit) => {
    checkpoint();
    edit();
    save();
    renderInspector();
  };
  const children = [];
  const list = document.createElement("div");
  list.className = "loras";
  if (state.realism?.installed) {
    const usual = state.realism.strength;
    // It can be switched off like any other LoRA: off is a strength of zero, and finals render without it.
    const on = state.project.realism !== 0;
    const row = document.createElement("div");
    row.className = "lora" + (on ? "" : " off");
    const head = document.createElement("div");
    head.className = "lora-head";
    const name = document.createElement("label");
    name.className = "lora-name";
    name.title = state.realism.file;
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = on;
    box.dataset.key = "Realism on";
    box.addEventListener("change", () => change(() => {
      if (box.checked) delete state.project.realism;
      else state.project.realism = 0;
    }));
    const label = document.createElement("span");
    label.textContent = "Realism Slider";
    name.append(box, label);
    const note = document.createElement("span");
    note.className = "lora-note";
    note.textContent = on ? "Finals only" : "Off";
    const reset = document.createElement("button");
    reset.className = "quiet small";
    reset.textContent = "Reset";
    reset.title = "Back to " + usual.toFixed(2);
    reset.hidden = !on || typeof state.project.realism !== "number" || state.project.realism === usual;
    reset.addEventListener("click", () => change(() => delete state.project.realism));
    head.append(name, note, reset);
    row.append(head, strengthSlider("Realism", "Realism Slider strength on finals", state.project.realism ?? usual, limits.range, usual, (v) => {
      if (v === usual) delete state.project.realism;
      else state.project.realism = v;
      reset.hidden = v === usual || v === 0;
      // Dragging it to zero switches it off, and dragging it up switches it on again.
      box.checked = v !== 0;
      row.classList.toggle("off", v === 0);
      note.textContent = v === 0 ? "Off" : "Finals only";
    }));
    list.append(row);
  }
  for (const entry of rows) {
    // A save swaps the project for the server's copy, so a control finds its row again when it changes it.
    const live = () => (state.project.loras || []).find((r) => r && r.id === entry.id) || {};
    const row = document.createElement("div");
    const on = entry.on !== false;
    const found = state.loras.some((f) => sameFile(f, entry.file));
    row.className = "lora" + (on ? "" : " off");
    const head = document.createElement("div");
    head.className = "lora-head";
    const toggle = document.createElement("label");
    toggle.className = "lora-name";
    toggle.title = entry.file;
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = on;
    box.dataset.key = "LoRA " + entry.id + " on";
    box.addEventListener("change", () => change(() => (live().on = box.checked)));
    const name = document.createElement("span");
    name.textContent = loraName(entry.file);
    toggle.append(box, name);
    head.append(toggle);
    if (!found) {
      const note = document.createElement("span");
      note.className = "lora-note missing";
      note.textContent = "Not installed";
      note.title = "Renders leave it out until " + entry.file + " is in models/loras.";
      head.append(note);
    }
    head.append(trashButton("Remove " + loraName(entry.file), () => change(() => {
      state.project.loras = state.project.loras.filter((r) => r !== live());
    })));
    const strength = typeof entry.strength === "number" ? entry.strength : 1;
    row.append(head, strengthSlider("LoRA " + entry.id, loraName(entry.file) + " strength", strength, limits.lora_range, 1, (v) => (live().strength = v)));
    list.append(row);
  }
  if (list.children.length) children.push(list);
  const free = state.loras.filter((f) => !rows.some((r) => sameFile(r.file, f)) && !(state.realism && sameFile(state.realism.file, f)));
  if (rows.length >= limits.limit) {
    const full = document.createElement("p");
    full.className = "hint";
    full.textContent = "A project takes up to " + limits.limit + " LoRAs. Remove one to add another.";
    children.push(full);
  } else if (free.length) {
    const add = document.createElement("button");
    add.className = "quiet wide";
    add.textContent = "Add a LoRA";
    add.dataset.key = "Add a LoRA";
    add.addEventListener("click", async () => {
      const file = await pickFile2("Add a LoRA", free, "From models/loras. Only LoRAs trained on MiniMax H3 have an effect.");
      if (!file) return;
      change(() => {
        state.project.loras = [...(state.project.loras || []), { id: "l" + crypto.randomUUID().replaceAll("-", "").slice(0, 8), file, strength: 1, on: true }];
      });
    });
    children.push(add);
  }
  const hint = document.createElement("p");
  hint.className = "hint";
  hint.textContent = state.loras.length
    ? "Every new render samples with these, drafts and finals alike. They only work if they were trained on MiniMax H3."
    : "Put LoRAs trained on MiniMax H3 in models/loras, then reload this page to add them here.";
  children.push(hint);
  return section("LoRAs", children);
}

function renderScene(panel, selected = false) {
  const head = document.createElement("div");
  head.className = "inspector-head";
  const title = document.createElement("h2");
  title.textContent = "Project";
  const line = document.createElement("p");
  line.className = "status";
  line.textContent = selected ? "Settings for every clip in this project." : "Select a clip to edit it. Press ? for shortcuts.";
  head.append(title, line);
  const rendered = state.project.clips.filter((c) => c.takes.length).length;
  const facts = document.createElement("dl");
  facts.className = "facts";
  for (const [name, value] of [
    ["Length", timecode(sceneEnd())],
    ["Clips", String(state.project.clips.length)],
    ["Rendered", rendered + " of " + state.project.clips.length],
    ["Cast", String((state.project.cast || []).length)],
  ]) {
    const dt = document.createElement("dt");
    dt.textContent = name;
    const dd = document.createElement("dd");
    dd.textContent = value;
    facts.append(dt, dd);
  }
  const shape = document.createElement("label");
  shape.className = "choice";
  const shapeTitle = document.createElement("span");
  shapeTitle.textContent = "Format";
  const shapes = document.createElement("select");
  shapes.dataset.key = "Format";
  for (const [value, label] of [["16:9", "Wide, 16:9"], ["9:16", "Vertical, 9:16"], ["1:1", "Square, 1:1"]]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    option.selected = (state.project.format || "16:9") === value;
    shapes.append(option);
  }
  shapes.addEventListener("change", () => {
    checkpoint();
    state.project.format = shapes.value;
    save();
  });
  shape.append(shapeTitle, shapes);
  const formatHint = document.createElement("p");
  formatHint.className = "hint";
  formatHint.textContent = "New renders and imports use this shape. Export fits every clip to it.";
  const model = document.createElement("label");
  model.className = "choice";
  const modelTitle = document.createElement("span");
  modelTitle.textContent = "Model";
  const models = document.createElement("select");
  models.dataset.key = "Model";
  // A project that names no model renders with the first one installed, as the server does.
  const current = projectProfile()?.id || "hyperflow";
  for (const profile of menuProfiles()) {
    const option = document.createElement("option");
    option.value = profile.id;
    option.textContent = profile.available ? profile.title : profile.title + " (not installed)";
    option.disabled = !profile.available && profile.id !== current;
    option.selected = profile.id === current;
    models.append(option);
  }
  const modelHint = document.createElement("p");
  modelHint.className = "hint";
  const describeModel = () => {
    const profile = profileOf(models.value);
    if (!profile) modelHint.textContent = "";
    else if (profile.available) modelHint.textContent = [profile.hint + " New renders use it.", ...state.finals, ...state.encoders.missing].join(" ");
    else modelHint.textContent = "Not installed. Needs " + profile.missing.join(" and ") + ".";
  };
  describeModel();
  models.addEventListener("change", () => {
    checkpoint();
    adoptPreset(models.value);
    state.project.profile = models.value;
    describeModel();
    save();
    // The panel also lists the project's checkpoints and which one is in use.
    renderInspector();
  });
  model.append(modelTitle, models);
  // Every model, with its settings, and the user's own checkpoints, live in one dialog.
  const manageModels = document.createElement("button");
  manageModels.className = "quiet wide";
  manageModels.textContent = "Models and their settings";
  manageModels.dataset.key = "Models";
  manageModels.addEventListener("click", () => openModels(null));
  panel.replaceChildren(
    head,
    section("Look", [field("Style for every clip", state.project.style, (v) => (state.project.style = v), { area: true })]),
    section("Format and model", [
      shape,
      formatHint,
      ...(state.profiles.length ? [model, modelHint, manageModels] : []),
      ...encoderChoices(),
      field("Language people speak", state.project.language || "", (v) => {
        if (v.trim()) state.project.language = v.trim();
        else delete state.project.language;
      }, { placeholder: "English" }),
    ]),
    loraSection(),
    section("Overview", [facts]),
  );
}

function renderBinPanel(panel, bin) {
  const inside = state.project.clips.filter((c) => c.bin === bin.id);
  const head = document.createElement("div");
  head.className = "inspector-head";
  const name = document.createElement("input");
  name.className = "name-input";
  name.value = bin.name;
  name.setAttribute("aria-label", "Bin name");
  name.dataset.key = "name";
  name.addEventListener("focus", () => checkpoint(), { once: true });
  name.addEventListener("input", () => {
    if (!name.value.trim()) return;
    bin.name = name.value;
    save();
    renderBin();
  });
  name.addEventListener("change", () => {
    if (!name.value.trim()) name.value = bin.name;
  });
  const row = document.createElement("div");
  row.className = "head-row";
  row.append(name, trashButton("Delete bin", () => deleteBin(bin)));
  const line = document.createElement("p");
  line.className = "status";
  line.textContent = inside.length ? inside.length + (inside.length === 1 ? " clip" : " clips") : "Empty";
  head.append(row, line);
  const about = document.createElement("p");
  about.className = "hint";
  about.textContent = "Drag clips onto this bin in the Clips panel to file them here, or onto the list below the bins to take them out. While the bin is selected, new clips and imports go in it.";
  const keep = document.createElement("p");
  keep.className = "hint";
  keep.textContent = "Deleting the bin keeps its clips.";
  panel.replaceChildren(head, section("Bin", [about, keep]));
}

// What the inspector last showed, so a new selection starts at the top and on its own tab.
let inspected = "";

// The inspector's tabs: the parts of whatever is selected, then the project, which is always there.
function renderTabs(tabs) {
  const bar = $("inspTabs");
  const focused = bar.contains(document.activeElement);
  bar.replaceChildren(...tabs.map(([id, label, count]) => {
    const tab = document.createElement("button");
    tab.type = "button";
    tab.className = "insp-tab";
    tab.dataset.tab = id;
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-selected", String(state.tab === id));
    tab.tabIndex = state.tab === id ? 0 : -1;
    tab.textContent = label;
    if (count) {
      const number = document.createElement("span");
      number.className = "insp-count";
      number.textContent = count;
      tab.append(number);
    }
    tab.addEventListener("click", () => showTab(id));
    return tab;
  }));
  if (focused) bar.querySelector('[aria-selected="true"]')?.focus();
}

function showTab(id) {
  state.tab = id;
  renderInspector();
  $("inspector").scrollTop = 0;
}

function renderInspector() {
  const panel = $("inspBody");
  const person = (state.project.cast || []).find((c) => c.id === state.person);
  const clip = person ? null : clipOf(state.clip);
  const bin = !clip && !person ? selectedBin() : null;
  const placed = clip ? state.project.timeline.find((i) => i.id === state.item) : null;
  const tabs = [];
  if (person) tabs.push(["shot", "Cast member"]);
  else if (bin) tabs.push(["shot", "Bin"]);
  else if (clip && clip.kind === "footage") tabs.push(["shot", "Footage"]);
  else if (clip) tabs.push(["shot", "Shot"], ["takes", "Takes", clip.takes.length]);
  if (placed) tabs.push(["timeline", "Timeline"]);
  tabs.push(["project", "Project"]);
  const key = person ? "p" + person.id : clip ? "c" + clip.id : bin ? "b" + bin.id : "";
  if (key !== inspected) {
    if (key && state.tab === "project") state.tab = "shot";
    inspected = key;
    $("inspector").scrollTop = 0;
  }
  if (!tabs.some(([id]) => id === state.tab)) state.tab = tabs[0][0];
  renderTabs(tabs);
  if (state.tab === "project") {
    renderScene(panel, Boolean(key));
    return;
  }
  if (person) {
    renderPerson(panel, person);
    return;
  }
  if (bin) {
    renderBinPanel(panel, bin);
    return;
  }
  const job = jobFor(clip.id);
  const status = clipStatus(clip);

  const head = document.createElement("div");
  head.className = "inspector-head";
  const name = document.createElement("input");
  name.className = "name-input";
  name.value = clip.name;
  name.setAttribute("aria-label", "Clip name");
  name.dataset.key = "name";
  name.addEventListener("focus", () => checkpoint(), { once: true });
  name.addEventListener("input", () => {
    clip.name = name.value;
    save();
    renderBin();
    renderTimeline();
  });
  const line = document.createElement("p");
  line.className = "status " + status.kind;
  line.dataset.status = clip.id;
  line.textContent = status.text + ", " + clipSeconds(clip);
  const row = document.createElement("div");
  row.className = "head-row";
  row.append(name, trashButton("Delete clip", () => deleteClip(clip)));
  head.append(row, line);

  const parts = [head];
  if (clip.error && !job) {
    const notice = document.createElement("p");
    notice.className = "notice";
    notice.textContent = "The last render failed. " + clip.error;
    parts.push(notice);
  }
  if (isStale(clip) && !job) {
    const notice = document.createElement("p");
    notice.className = "notice";
    notice.textContent = "The clip this one joins has changed, so the join may show. Render it again to fix that.";
    parts.push(notice);
  }

  if (clip.kind === "footage") {
    const take = activeTake(clip);
    const about = document.createElement("p");
    about.className = "hint";
    about.textContent = "Imported from " + (take?.source || "a file") + ". Footage plays as it is. Select it on the timeline and use Extend or Bridge to generate new shots from it.";
    if (state.tab === "timeline") parts.push(section("On the timeline", timelineControls(placed)));
    else parts.push(section("Footage", [about]));
    panel.replaceChildren(...parts);
    return;
  }
  if (state.tab === "timeline") parts.push(section("On the timeline", timelineControls(placed)));

  const timing = document.createElement("div");
  timing.className = "two";
  timing.append(
    field("Camera", clip.camera, (v) => (clip.camera = v), { placeholder: "Slow push in" }),
    field("Seconds", clip.seconds, (v) => {
      if (!(Number(v) >= 0.5 && Number(v) <= 60)) return;
      clip.seconds = Number(v);
      resizePlanned(clip);
    }, { type: "number" }),
  );
  if (state.tab === "shot") parts.push(section("Shot", [
    field("What happens", clip.action, (v) => (clip.action = v), { big: true, placeholder: "Describe what we see and hear, in the order it happens." }),
    timing,
    ...framesFor(clip),
    ...(state.project.cast?.some((p) => p.file || p.refmod) ? castPicker(clip) : []),
  ]));

  const voiced = ["speaker", "dialogue", "sound", "music"].filter((k) => String(clip[k] || "").trim()).length;
  if (state.tab === "shot") parts.push(disclosure("Dialogue and sound", voiced > 0, voiced ? voiced + " set" : "", [
    field("Who speaks", clip.speaker, (v) => (clip.speaker = v), {
      placeholder: (state.project.cast || []).some((p) => p.voice) ? "@" + state.project.cast.find((p) => p.voice).name + " uses their voice" : "The old keeper, low and gravelly",
    }),
    field("What they say", clip.dialogue, (v) => (clip.dialogue = v), { area: true }),
    field("Language of the lines", clip.language || "", (v) => {
      if (v.trim()) clip.language = v.trim();
      else delete clip.language;
    }, { placeholder: state.project.language || "English" }),
    field("Sound", clip.sound, (v) => (clip.sound = v), { area: true, placeholder: "Rain on glass, distant thunder" }),
    field("Music", clip.music, (v) => (clip.music = v), { placeholder: "None" }),
  ]));

  if (state.tab === "shot") {
    parts.push(disclosure("Picture to a track", Boolean(clip.track), clip.track ? "Set" : "", trackSection(clip)));
    parts.push(disclosure("Model", Boolean(profileOf(clip.profile)), profileOf(clip.profile)?.title || "", clipModel(clip)));
  }

  const seed = field("Seed. Leave empty for a new one each render", clip.seed ?? "", (v) => {
    clip.seed = v.trim() === "" ? null : Number(v);
  }, { type: "number" });
  if (state.tab !== "takes") {
    // Only the Takes tab lists them.
  } else if (clip.takes.length) {
    parts.push(section("Takes", [...versions(clip), seed]));
  } else {
    const none = document.createElement("p");
    none.className = "hint";
    none.textContent = "No takes yet. Render a draft to make the first one.";
    parts.push(section("Takes", [none, seed]));
  }

  const foot = document.createElement("div");
  foot.className = "inspector-foot";
  const actions = document.createElement("div");
  actions.className = "actions";
  if (job) {
    const progress = document.createElement("p");
    progress.className = "foot-status";
    progress.dataset.status = clip.id;
    progress.textContent = status.text;
    const cancel = document.createElement("button");
    cancel.className = "quiet";
    cancel.textContent = "Cancel render";
    cancel.addEventListener("click", () => cancelJob(job));
    actions.append(cancel);
    foot.append(progress);
  } else {
    for (const [quality, label, style] of [["draft", "Render draft", "quiet"], ["final", "Render final", "primary"]]) {
      const button = document.createElement("button");
      button.className = style;
      button.textContent = label;
      button.addEventListener("click", () => renderClip(clip, quality));
      actions.append(button);
    }
  }
  foot.append(actions);
  parts.push(foot);
  panel.replaceChildren(...parts);
}

const SAVED_FIELDS = ["action", "camera", "speaker", "dialogue", "sound", "music", "seconds", "cast", "track"];

function chooseTake(clip, take) {
  checkpoint();
  clip.active = take.id;
  const [, last] = linkedRange(clip.link, take.frames);
  for (const item of state.project.timeline) {
    if (item.clip === clip.id) item.out = Math.min(item.out, last);
  }
  save();
  renderAll();
}

function cycleTake(step) {
  const clip = clipOf(state.clip);
  if (!clip || clip.takes.length < 2) return;
  const index = clip.takes.indexOf(activeTake(clip));
  const take = clip.takes[(index + step + clip.takes.length) % clip.takes.length];
  chooseTake(clip, take);
  toast("Take " + (clip.takes.indexOf(take) + 1) + " of " + clip.takes.length);
}

function versions(clip) {
  const takes = document.createElement("div");
  takes.className = "takes";
  const current = activeTake(clip);
  clip.takes.forEach((take, index) => {
    const button = document.createElement("button");
    button.className = "take" + (take === current ? " on" : "");
    const img = document.createElement("img");
    img.alt = "";
    img.src = viewUrl(take.poster, take.subfolder);
    const label = document.createElement("span");
    label.textContent = "Take " + (index + 1) + ", " + take.quality;
    const madeWith = take.models?.profile_title || profileOf(take.models?.profile)?.title;
    if (madeWith) label.title = "Made with " + madeWith + (take.models?.steps ? ", " + take.models.steps + " steps, " + take.models.sampler : "");
    const seed = document.createElement("small");
    seed.textContent = "Seed " + take.seed;
    button.append(img, label, seed);
    button.addEventListener("click", () => chooseTake(clip, take));
    takes.append(button);
  });
  const tools = document.createElement("div");
  tools.className = "take-tools";
  const button = (text, run, disabled = false) => {
    const element = document.createElement("button");
    element.className = "quiet";
    element.textContent = text;
    element.disabled = disabled;
    element.addEventListener("click", run);
    tools.append(element);
  };
  button("Keep this seed", () => {
    checkpoint();
    clip.seed = current.seed;
    save();
    renderInspector();
    toast("Renders of this clip now use seed " + current.seed + ".");
  });
  button("Restore its prompt", () => {
    checkpoint();
    for (const name of SAVED_FIELDS) {
      if (current.saved && name in current.saved) clip[name] = current.saved[name];
    }
    save();
    renderInspector();
  }, !current.saved);
  button("Delete take", () => {
    checkpoint();
    const number = clip.takes.indexOf(current) + 1;
    clip.takes = clip.takes.filter((t) => t !== current);
    clip.active = clip.takes.at(-1)?.id ?? null;
    save();
    renderAll();
    toast("Deleted take " + number + " of " + clip.name + ".", false, undoAction);
  }, clip.takes.length < 2);
  return [takes, tools];
}

function setSpeed(item, speed) {
  const value = Math.min(4, Math.max(0.25, speed));
  const oldEnd = itemEnd(item);
  item.speed = value;
  const shift = itemEnd(item) - oldEnd;
  for (const other of state.project.timeline) {
    if (other !== item && other.start >= oldEnd - (other.transition?.frames || 0)) other.start += shift;
  }
  save();
  renderTimeline();
  showFrame();
}

function previousItem(item) {
  return sorted().filter((other) => other !== item && other.start < item.start).at(-1) || null;
}

function setTransition(item, type, frames) {
  const before = previousItem(item);
  const old = item.transition?.frames || 0;
  if (type !== "none" && (!before || itemEnd(before) < item.start + old)) {
    toast("Put this clip right after another one to add a transition.", true);
    renderInspector();
    return;
  }
  const limit = before ? Math.min(itemLength(before), itemLength(item)) - 1 : 0;
  const next = type === "none" ? 0 : Math.max(2, Math.min(frames, limit));
  const shift = next - old;
  for (const other of state.project.timeline) {
    if (other === item || other.start > item.start) other.start -= shift;
  }
  if (type === "none") delete item.transition;
  else item.transition = { type, frames: next };
  save();
  renderTimeline();
  renderInspector();
  showFrame();
}

function timelineControls(item) {
  const box = document.createElement("div");
  box.className = "on-timeline";

  const speeds = document.createElement("div");
  speeds.className = "speeds";
  const percent = document.createElement("input");
  percent.type = "number";
  percent.min = "25";
  percent.max = "400";
  percent.step = "5";
  percent.value = String(Math.round(speedOf(item) * 100));
  percent.setAttribute("aria-label", "Speed in percent");
  percent.addEventListener("change", () => {
    checkpoint();
    setSpeed(item, Number(percent.value) / 100 || 1);
    percent.value = String(Math.round(speedOf(item) * 100));
  });
  const label = document.createElement("span");
  label.textContent = "Speed %";
  speeds.append(label, percent);
  for (const value of [50, 100, 200]) {
    const preset = document.createElement("button");
    preset.className = "quiet";
    preset.textContent = value + "%";
    preset.addEventListener("click", () => {
      checkpoint();
      setSpeed(item, value / 100);
      percent.value = String(value);
    });
    speeds.append(preset);
  }

  const row = document.createElement("div");
  row.className = "speeds";
  const name = document.createElement("span");
  name.textContent = "Transition in";
  const kind = document.createElement("select");
  kind.setAttribute("aria-label", "Transition in");
  for (const [value, text] of [["none", "None"], ["dissolve", "Dissolve"], ["black", "Dip to black"], ["wipe", "Wipe"]]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = text;
    option.selected = (item.transition?.type || "none") === value;
    kind.append(option);
  }
  const length = document.createElement("input");
  length.type = "number";
  length.min = "2";
  length.step = "2";
  length.value = String(item.transition?.frames || 12);
  length.setAttribute("aria-label", "Transition length in frames");
  const apply = () => {
    checkpoint();
    setTransition(item, kind.value, Number(length.value) || 12);
  };
  kind.addEventListener("change", apply);
  length.addEventListener("change", apply);
  const unit = document.createElement("span");
  unit.textContent = "frames";
  row.append(name, kind, length, unit);
  box.append(speeds, row);
  return [box];
}

function framesFor(clip) {
  const slots = [["first", "First frame"], ["last", "Last frame"]].filter(([side]) => frameUsable(clip, side));
  if (!slots.length) return [];
  const row = document.createElement("div");
  row.className = "frames";
  row.append(...slots.map(([side, label]) => frameSlot(clip, side, label)));
  return [row];
}

/* Frame strips and waveforms, made once per render on the server */

const peakCache = new Map();

async function ensureStrips() {
  const project = state.project;
  for (const clip of project.clips) {
    for (const take of clip.takes) {
      if (take.strip || state.project !== project) continue;
      try {
        take.strip = await postJson("/strip", { filename: take.file, subfolder: take.subfolder, frames: take.frames });
      } catch (_) {
        continue;
      }
      save();
      renderBin();
      renderTimeline();
    }
  }
}

async function loadPeaks(take) {
  const key = take.file;
  if (!peakCache.has(key)) {
    peakCache.set(key, fetch(viewUrl(take.strip.peaks, take.subfolder)).then((r) => r.json()).catch(() => []));
  }
  return peakCache.get(key);
}

// Real frames from across the clip, matched to the trim and speed rather than stretched.
function fillFilm(film, item, take, height) {
  if (!take.strip) {
    film.style.backgroundImage = "url(" + viewUrl(take.poster, take.subfolder) + ")";
    return;
  }
  const { count, step, w, h } = take.strip;
  const tile = height * (w / h);
  const width = px(itemLength(item));
  const sprite = "url(" + viewUrl(take.strip.sprite, take.subfolder) + ")";
  const tiles = [];
  for (let x = 0; x < width && tiles.length < 400; x += tile) {
    const frame = item.in + ((x + tile / 2) / state.zoom) * speedOf(item);
    const index = Math.min(count - 1, Math.max(0, Math.floor(frame / step)));
    const cell = document.createElement("div");
    cell.className = "tile";
    cell.style.left = x + "px";
    cell.style.width = tile + "px";
    cell.style.backgroundImage = sprite;
    cell.style.backgroundSize = count * tile + "px " + height + "px";
    cell.style.backgroundPosition = -index * tile + "px 0";
    tiles.push(cell);
  }
  film.replaceChildren(...tiles);
}

function drawWave(canvas, item, take, height, selected) {
  const width = Math.max(1, Math.round(px(itemLength(item))));
  const ratio = Math.min(2, window.devicePixelRatio || 1);
  const scale = Math.min(1, 16000 / (width * ratio));
  canvas.width = Math.max(1, Math.round(width * ratio * scale));
  canvas.height = Math.max(1, Math.round(height * ratio));
  canvas.style.width = width + "px";
  canvas.style.height = height + "px";
  if (!take.strip) return;
  loadPeaks(take).then((peaks) => {
    if (!peaks.length || !canvas.isConnected) return;
    // Each clip is drawn against its own loudest moment, so quiet takes still read.
    const loudest = Math.max(35, ...peaks);
    const context = canvas.getContext("2d");
    context.setTransform(ratio * scale, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    context.fillStyle = selected ? "#8fd6c8" : "#5fae9f";
    const middle = height / 2;
    for (let x = 0; x < width; x += 2) {
      const frame = item.in + (x / state.zoom) * speedOf(item);
      const value = peaks[Math.min(peaks.length - 1, Math.max(0, Math.floor(frame)))] || 0;
      const bar = Math.max(1, (value / loudest) * (height - 6));
      context.fillRect(x, middle - bar / 2, 1.4, bar);
    }
  });
}

/* Timeline height */

function setTimelineHeight(value, keep = false) {
  const height = Math.round(Math.min(window.innerHeight * 0.72, Math.max(190, value)));
  document.documentElement.style.setProperty("--tl-h", height + "px");
  if (keep) {
    try { localStorage.setItem("h3take.timeline", String(height)); } catch (_) {}
  }
  return height;
}

// The clips panel and the inspector are as wide as the user drags them, on this browser.
const SIDES = {
  splitBin: { name: "--bin-w", key: "h3take.bin", min: 200, max: 520, panel: () => document.querySelector(".bin"), sign: 1 },
  splitInspector: { name: "--insp-w", key: "h3take.inspector", min: 300, max: 680, panel: () => $("inspector"), sign: -1 },
};

function setSideWidth(side, width, keep = false) {
  // The viewer always keeps room for itself, whatever the panels ask for.
  const other = Object.values(SIDES).find((s) => s !== side).panel().getBoundingClientRect().width;
  const room = window.innerWidth - other - 12 - 360;
  const wide = Math.round(Math.max(side.min, Math.min(side.max, room, width)));
  document.documentElement.style.setProperty(side.name, wide + "px");
  if (keep) {
    try { localStorage.setItem(side.key, String(wide)); } catch (_) {}
  }
}

function bindSideSplitters() {
  for (const [id, side] of Object.entries(SIDES)) {
    const handle = $(id);
    try {
      const saved = Number(localStorage.getItem(side.key));
      if (saved) setSideWidth(side, saved);
    } catch (_) {}
    const settle = () => state.project && renderTimeline();
    handle.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      handle.classList.add("dragging");
      const from = side.panel().getBoundingClientRect().width;
      const move = (ev) => setSideWidth(side, from + side.sign * (ev.clientX - event.clientX));
      const up = (ev) => {
        handle.classList.remove("dragging");
        setSideWidth(side, from + side.sign * (ev.clientX - event.clientX), true);
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        settle();
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    });
    handle.addEventListener("dblclick", () => {
      document.documentElement.style.removeProperty(side.name);
      try { localStorage.removeItem(side.key); } catch (_) {}
      settle();
    });
    handle.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const now = side.panel().getBoundingClientRect().width;
      setSideWidth(side, now + side.sign * (event.key === "ArrowRight" ? 24 : -24), true);
      settle();
    });
  }
}

function bindSplitter() {
  const handle = $("splitter");
  try {
    const saved = Number(localStorage.getItem("h3take.timeline"));
    if (saved) setTimelineHeight(saved);
  } catch (_) {}
  let frame = 0;
  const redraw = () => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => state.project && renderTimeline());
  };
  handle.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    handle.classList.add("dragging");
    const move = (ev) => {
      setTimelineHeight(window.innerHeight - ev.clientY - 3);
      redraw();
    };
    const up = (ev) => {
      handle.classList.remove("dragging");
      setTimelineHeight(window.innerHeight - ev.clientY - 3, true);
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      redraw();
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  });
  handle.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
    event.preventDefault();
    const now = $("timeline").getBoundingClientRect().height;
    setTimelineHeight(now + (event.key === "ArrowUp" ? 24 : -24), true);
    redraw();
  });
}

function renderRuler(width) {
  const ruler = $("ruler");
  const seconds = Math.ceil(width / px(FPS));
  const step = [1, 2, 5, 10, 30, 60].find((s) => px(FPS * s) >= 70) || 120;
  const ticks = [];
  for (let s = 0; s <= seconds; s += step) {
    const tick = document.createElement("div");
    tick.className = "tick";
    tick.style.left = px(s * FPS) + "px";
    tick.textContent = Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
    ticks.push(tick);
    const half = document.createElement("div");
    half.className = "tick minor";
    half.style.left = px((s + step / 2) * FPS) + "px";
    ticks.push(half);
  }
  ruler.replaceChildren(...ticks);
}

function renderTimeline() {
  const width = Math.max($("tlScroll").clientWidth, px(sceneEnd() + FPS * 20));
  $("tlContent").style.width = width + "px";
  renderRuler(width);
  const blocks = [];
  const waves = [];
  const filmHeight = Math.max(30, $("laneV").clientHeight - 20);
  const waveHeight = Math.max(16, $("laneA").clientHeight - 12);
  for (const item of state.project.timeline) {
    const clip = clipOf(item.clip);
    if (!clip) continue;
    const take = activeTake(clip);
    const job = jobFor(clip.id);
    const block = document.createElement("div");
    const status = job ? (state.running === job ? "rendering" : "queued") : !take ? "todo" : isStale(clip) ? "stale" : "";
    block.className = ("block " + status + (item.id === state.item ? " on" : "")).trim();
    block.style.left = px(item.start) + "px";
    block.style.width = Math.max(14, px(itemLength(item))) + "px";
    block.dataset.item = item.id;
    block.tabIndex = 0;
    block.setAttribute("role", "button");
    block.setAttribute("aria-pressed", String(item.id === state.item));
    block.setAttribute("aria-label", clip.name + ", " + clipStatus(clip).text + ", at " + timecode(item.start) + " on the timeline");
    block.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      event.stopPropagation();
      selectClip(clip.id, item.id);
    });
    const film = document.createElement("div");
    film.className = "film";
    if (job && state.running === job && job.motion) {
      const live = document.createElement("video");
      live.className = "livestrip";
      live.muted = true;
      live.loop = true;
      live.autoplay = true;
      live.playsInline = true;
      live.dataset.url = job.motion;
      live.src = job.motion;
      film.append(live);
    } else if (job?.preview) {
      film.classList.add("still");
      film.style.backgroundImage = "url(" + job.preview + ")";
    } else if (take) {
      fillFilm(film, item, take, filmHeight);
    }
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = status === "stale" ? clip.name + ", source changed" : clip.name;
    if (speedOf(item) !== 1) {
      const rate = document.createElement("span");
      rate.className = "rate";
      rate.textContent = " " + Math.round(speedOf(item) * 100) + "%";
      name.append(rate);
    }
    block.append(film, name);
    if (take && clip.kind !== "footage" && px(itemLength(item)) >= 64) {
      // Whether the take on the timeline is a quick draft or a finished final.
      const grade = document.createElement("div");
      const final = take.quality === "final";
      grade.className = "grade " + (final ? "final" : "draft");
      grade.textContent = final ? "Final" : "Draft";
      grade.title = final ? "A final render" : "A draft. Render a final when the shot is right.";
      block.append(grade);
    }
    if (item.transition) {
      const mark = document.createElement("div");
      mark.className = "tr";
      mark.style.width = px(item.transition.frames) + "px";
      mark.title = item.transition.type + ", " + item.transition.frames + " frames";
      block.append(mark);
    }
    if (job && state.running === job) {
      const bar = document.createElement("div");
      bar.className = "bar";
      bar.style.width = Math.round((job.progress || 0) * 100) + "%";
      block.append(bar);
    }
    for (const side of ["l", "r"]) {
      const edge = document.createElement("div");
      edge.className = "edge " + side;
      block.append(edge);
    }
    bindBlock(block, item, clip);
    blocks.push(block);

    const wave = document.createElement("div");
    wave.className = "wave" + (take ? "" : " todo") + (item.id === state.item ? " on" : "");
    wave.dataset.item = item.id;
    wave.style.left = block.style.left;
    wave.style.width = block.style.width;
    if (take) {
      const canvas = document.createElement("canvas");
      wave.append(canvas);
      drawWave(canvas, item, take, waveHeight, item.id === state.item);
    }
    waves.push(wave);
  }
  const order = sorted();
  for (let i = 0; i + 1 < order.length; i++) {
    const before = order[i];
    const after = order[i + 1];
    const space = px(after.start - itemEnd(before));
    if (space < 40 || !bridgeable(before, after)) continue;
    const hint = document.createElement("div");
    hint.className = "gap-hint";
    hint.style.left = px(itemEnd(before)) + "px";
    hint.style.width = space + "px";
    const go = document.createElement("button");
    go.className = "gap-bridge";
    const what = "Generate a bridge: a clip that connects " + clipOf(before.clip).name + " to " + clipOf(after.clip).name;
    go.setAttribute("aria-label", what);
    go.title = what;
    go.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 6v12M6 12h12"/></svg>';
    if (space >= 170) {
      const label = document.createElement("span");
      label.textContent = "Generate a bridge";
      go.append(label);
    }
    go.addEventListener("pointerdown", (event) => event.stopPropagation());
    go.addEventListener("click", () => {
      state.clip = before.clip;
      state.item = before.id;
      state.gap = null;
      addLinked("bridge");
    });
    hint.append(go);
    blocks.push(hint);
  }
  if (state.gap) {
    const mark = document.createElement("div");
    mark.className = "gapmark";
    mark.style.left = px(state.gap.start) + "px";
    mark.style.width = px(state.gap.end - state.gap.start) + "px";
    blocks.push(mark);
  }
  // Drawing the lane again must not take the focus away from the clip that had it.
  const focused = $("laneV").contains(document.activeElement) ? document.activeElement.dataset.item : null;
  $("laneV").replaceChildren(...blocks);
  $("laneA").replaceChildren(...waves);
  if (focused) $("laneV").querySelector('[data-item="' + CSS.escape(focused) + '"]')?.focus();
  placePlayhead();
  updateTools();
}

function placePlayhead() {
  $("playhead").style.left = px(state.playhead) + "px";
  updateClock();
}

function updateClock() {
  const now = timecode(here());
  const frames = document.createElement("span");
  frames.className = "frames-part";
  frames.textContent = now.slice(5);
  $("time").replaceChildren(now.slice(0, 5), frames);
  $("total").textContent = "of " + timecode(viewEnd());
}

function snapPoints(skip) {
  const points = [0, state.playhead];
  for (const item of state.project.timeline) {
    if (item !== skip) points.push(item.start, itemEnd(item));
  }
  return points;
}

function snap(frames, points) {
  const reach = 8 / state.zoom;
  let best = frames;
  let distance = reach;
  for (const point of points) {
    if (Math.abs(point - frames) < distance) {
      distance = Math.abs(point - frames);
      best = point;
    }
  }
  return best;
}

function bindBlock(block, item, clip) {
  block.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    selectClip(clip.id, item.id);
    const live = document.querySelector('.block[data-item="' + item.id + '"]');
    const sound = document.querySelector('.wave[data-item="' + item.id + '"]');
    const edge = event.target.classList.contains("l") ? "l" : event.target.classList.contains("r") ? "r" : null;
    const origin = { x: event.clientX, start: item.start, in: item.in, out: item.out };
    const limit = activeTake(clip)?.frames ?? Infinity;
    const points = snapPoints(item);
    let changed = false;
    const move = (ev) => {
      const delta = Math.round((ev.clientX - origin.x) / state.zoom);
      if (delta !== 0 && !changed) {
        checkpoint();
        changed = true;
      }
      const rate = speedOf(item);
      if (edge === "l") {
        const shift = Math.max(-origin.in, Math.min(Math.round(delta * rate), origin.out - origin.in - 6));
        item.in = origin.in + shift;
        item.start = origin.start + Math.round(shift / rate);
      } else if (edge === "r") {
        item.out = Math.min(limit, Math.max(origin.in + 6, origin.out + Math.round(delta * rate)));
      } else {
        const wanted = Math.max(0, origin.start + delta);
        const startSnap = snap(wanted, points);
        const endSnap = snap(wanted + itemLength(item), points) - itemLength(item);
        item.start = Math.max(0, startSnap !== wanted ? startSnap : endSnap);
      }
      live.style.left = px(item.start) + "px";
      live.style.width = Math.max(14, px(itemLength(item))) + "px";
      if (sound) {
        sound.style.left = live.style.left;
        sound.style.width = live.style.width;
      }
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      if (changed) {
        settle(item);
        save();
      }
      renderTimeline();
      showFrame();
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  });
}

function splitAtPlayhead() {
  const item = itemAt(state.playhead);
  if (!item || state.playhead <= item.start || state.playhead >= itemEnd(item)) {
    toast("Move the playhead over a clip to split it.");
    return;
  }
  checkpoint();
  const cut = Math.round(item.in + (state.playhead - item.start) * speedOf(item));
  state.project.timeline.push({ id: uid("i"), clip: item.clip, start: state.playhead, in: cut, out: item.out, speed: item.speed });
  item.out = cut;
  save();
  renderTimeline();
}

function removeSelected() {
  if (state.gap) {
    closeGap();
    return;
  }
  if (!state.item) {
    toast("Select a clip or a gap on the timeline first.");
    return;
  }
  checkpoint();
  state.project.timeline = state.project.timeline.filter((i) => i.id !== state.item);
  state.item = null;
  save();
  renderTimeline();
  showFrame();
}

/* Player */

let front = $("videoA");
let back = $("videoB");
let lastTick = 0;

function itemAt(frame) {
  return state.project.timeline.find((i) => frame >= i.start && frame < itemEnd(i)) || null;
}

function nextItem(frame) {
  return sorted().find((i) => i.start > frame) || null;
}

function sourceOf(item) {
  const take = activeTake(clipOf(item?.clip));
  return take ? viewUrl(take.file, take.subfolder) : null;
}

function load(video, url) {
  if (video.dataset.url !== url) {
    video.dataset.url = url;
    video.src = url;
    video.addEventListener("loadeddata", () => {
      if (!state.playing) showFrame();
    }, { once: true });
  }
}

function showLive(job) {
  const live = $("live");
  const image = $("preview");
  const showing = !!job && !state.playing;
  const motion = $("motion");
  live.hidden = !showing;
  image.hidden = !(showing && job.preview && !job.motion);
  motion.hidden = !(showing && job.motion);
  if (!showing) {
    motion.pause();
    return;
  }
  if (job.preview && !job.motion) image.src = job.preview;
  if (job.motion && motion.dataset.url !== job.motion) {
    const resume = motion.currentTime;
    motion.dataset.url = job.motion;
    motion.src = job.motion;
    motion.onloadedmetadata = () => {
      motion.currentTime = resume % (motion.duration || 1);
      motion.play().catch(() => {
        motion.muted = true;
        motion.play().catch(() => {});
      });
    };
  }
  live.textContent = (clipOf(job.clip)?.name || "Clip") + ": " + (job.stage || "Waiting in the queue") + (job.max ? ", step " + job.step + " of " + job.max : "");
}

function speedOf(item) {
  return item?.speed || 1;
}

function sourceSeconds(item, frame) {
  return (item.in + (frame - item.start) * speedOf(item)) / FPS;
}

// The clip coming in over this frame, when a transition is running.
function incoming(frame) {
  return state.project.timeline.find((i) => i.transition && frame >= i.start && frame < i.start + i.transition.frames) || null;
}

// The clip the player follows. During a transition that is the one going out.
function leading(frame) {
  const coming = incoming(frame);
  const others = state.project.timeline.filter((i) => i !== coming && frame >= i.start && frame < itemEnd(i));
  return others[0] || coming;
}

// Seconds the next clip runs, silent, before its cut.
const PREROLL = 0.4;

function clearMix() {
  for (const video of [front, back]) {
    video.style.opacity = "";
    video.style.clipPath = "";
    video.volume = 1;
    video.muted = false;
  }
  front.style.zIndex = "1";
  back.style.zIndex = "2";
}

function applyMix(item, frame) {
  const progress = Math.min(1, Math.max(0, (frame - item.start) / item.transition.frames));
  const kind = item.transition.type;
  front.style.zIndex = "1";
  back.style.zIndex = "2";
  back.classList.add("show");
  back.muted = false;
  front.volume = 1 - progress;
  back.volume = progress;
  if (kind === "wipe") {
    back.style.opacity = "";
    back.style.clipPath = "inset(0 " + (100 - progress * 100) + "% 0 0)";
  } else if (kind === "black") {
    front.style.opacity = String(Math.max(0, 1 - progress * 2));
    back.style.opacity = String(Math.max(0, progress * 2 - 1));
    back.style.clipPath = "";
  } else {
    back.style.opacity = String(progress);
    back.style.clipPath = "";
  }
}

function place(video, item, frame, tolerance = 0.01) {
  load(video, sourceOf(item));
  if (video.playbackRate !== speedOf(item)) video.playbackRate = speedOf(item);
  const seconds = sourceSeconds(item, frame);
  if (Math.abs(video.currentTime - seconds) > tolerance) video.currentTime = seconds;
}

// Switches the viewer between the cut (null) and one clip on its own.
function setSource(clipId) {
  if (state.source === clipId) return;
  if (state.playing) setPlaying(false);
  state.source = clipId;
  state.sourceFrame = 0;
}

function showSource(clip) {
  const take = activeTake(clip);
  const job = jobFor(clip.id);
  const slate = $("slate");
  clearMix();
  back.pause();
  back.classList.remove("show");
  if (take) {
    state.sourceFrame = Math.min(state.sourceFrame, take.frames - 1);
    load(front, viewUrl(take.file, take.subfolder));
    front.playbackRate = 1;
    if (!state.playing && Math.abs(front.currentTime - state.sourceFrame / FPS) > 0.01) front.currentTime = state.sourceFrame / FPS;
    front.classList.add("show");
    slate.textContent = "";
  } else {
    front.classList.remove("show");
    slate.textContent = job ? "" : "This clip has not been rendered yet. Press Render draft.";
  }
  showLive(job);
  updateClock();
}

function showFrame() {
  const clip = viewing();
  $("sourceTag").hidden = !clip;
  $("timeline").classList.toggle("source", Boolean(clip));
  if (clip) {
    $("sourceName").textContent = clip.name;
    showSource(clip);
    return;
  }
  // Stopped on the very end of the cut, the viewer keeps the last frame instead of going empty.
  const end = sceneEnd();
  const frame = end && state.playhead >= end ? end - 1 : state.playhead;
  const item = leading(frame);
  const url = sourceOf(item);
  const job = state.running || jobFor(item?.clip || state.clip);
  const slate = $("slate");
  clearMix();
  if (url) {
    place(front, item, frame);
    front.classList.add("show");
    back.classList.remove("show");
    slate.textContent = "";
    const coming = incoming(frame);
    if (coming && coming !== item && sourceOf(coming)) {
      place(back, coming, frame);
      applyMix(coming, frame);
    }
  } else {
    front.classList.remove("show");
    back.classList.remove("show");
    if (item) slate.textContent = jobFor(item.clip) ? "" : "This clip has not been rendered yet. Select it and press Render draft.";
    else slate.textContent = state.project.clips.length ? "" : "Press New clip, describe the shot, and render a draft.";
  }
  showLive(job);
  placePlayhead();
}

function swap() {
  [front, back] = [back, front];
  front.classList.add("show");
  back.classList.remove("show");
  clearMix();
}

function setPlaying(on) {
  state.playing = on;
  $("play").classList.toggle("playing", on);
  $("play").setAttribute("aria-label", on ? "Pause" : "Play");
  if (!on) {
    front.pause();
    back.pause();
    showFrame();
    return;
  }
  if (viewing()) {
    if (!sourceEnd()) {
      setPlaying(false);
      return;
    }
    if (state.sourceFrame >= sourceEnd() - 1) state.sourceFrame = 0;
    showFrame();
    front.play().catch(() => {});
    requestAnimationFrame(tickSource);
    return;
  }
  if (state.playhead >= sceneEnd()) state.playhead = 0;
  showFrame();
  lastTick = performance.now();
  const item = leading(state.playhead);
  if (item && sourceOf(item)) front.play().catch(() => {});
  requestAnimationFrame(tick);
}

function tick(now) {
  if (!state.playing) return;
  const item = leading(state.playhead);
  const url = sourceOf(item);
  if (url && !front.paused && front.dataset.url === url) {
    // Never before the clip's own start. A video's time can read a hair under its in-point, and a
    // playhead a hair before the cut would hand the picture back to the clip that just ended.
    state.playhead = Math.max(item.start, item.start + (front.currentTime * FPS - item.in) / speedOf(item));
  } else {
    state.playhead += ((now - lastTick) / 1000) * FPS;
  }
  lastTick = now;

  const coming = incoming(state.playhead);
  if (coming && coming !== item && sourceOf(coming)) {
    if (back.paused || back.dataset.url !== sourceOf(coming)) {
      place(back, coming, state.playhead, 0.08);
      back.play().catch(() => {});
    }
    applyMix(coming, state.playhead);
  } else {
    const following = item ? sorted().find((i) => i.start >= itemEnd(item) - (i.transition?.frames || 0) && i !== item) : nextItem(state.playhead);
    if (following && sourceOf(following) && back.paused) {
      load(back, sourceOf(following));
      // A video takes a moment to start, which would be a hole in the sound at every cut. So the
      // next clip waits a little before its in-point and starts early and silent, to be running
      // when the cut comes. A clip cut in at its very first frame has nothing before it to run.
      const rate = speedOf(following);
      const parked = Math.max(0, following.in / FPS - PREROLL * rate);
      const lead = (following.start - state.playhead) / FPS;
      if (lead > 0 && lead * rate <= following.in / FPS - parked && back.readyState >= 3 && !back.seeking) {
        back.volume = 0;
        back.playbackRate = rate;
        back.play().catch(() => {});
      } else if (!back.seeking && Math.abs(back.currentTime - parked) > 0.01) {
        // Once is enough. Seeking again on every frame keeps the next clip from being ready.
        back.currentTime = parked;
      }
    } else if (following && back.volume === 0 && !back.paused && back.dataset.url === sourceOf(following)) {
      // While it runs silent, nudge its speed so it is on its in-point exactly when the cut comes:
      // early would skip a frame, late would show one from before the cut.
      // It settles at its own speed a little before the cut: a change of speed at the cut itself
      // makes the sound stumble.
      const rate = speedOf(following);
      const lead = (following.start - state.playhead) / FPS;
      const off = back.currentTime - (following.in / FPS - lead * rate);
      const steer = lead < 0.12 ? 1 : off < -0.015 ? 1.3 : off > 0.015 ? 0.75 : 1;
      if (back.playbackRate !== rate * steer) back.playbackRate = rate * steer;
    }
  }

  const current = leading(state.playhead);
  if (current !== item) {
    if (current && sourceOf(current)) {
      if (back.dataset.url === sourceOf(current)) swap();
      place(front, current, state.playhead, 0.08);
      front.classList.add("show");
      front.volume = 1;
      front.play().catch(() => {});
      if (!incoming(state.playhead)) back.pause();
    } else {
      front.pause();
      front.classList.remove("show");
      clearMix();
    }
  }
  if (state.playhead >= sceneEnd()) {
    state.playhead = sceneEnd();
    setPlaying(false);
    return;
  }
  placePlayhead();
  const scroller = $("tlScroll");
  const x = px(state.playhead);
  if (x > scroller.scrollLeft + scroller.clientWidth - 40) scroller.scrollLeft = x - 80;
  requestAnimationFrame(tick);
}

// A clip on its own plays straight through its active take, at its own speed.
function tickSource() {
  if (!state.playing) return;
  if (!viewing()) {
    setPlaying(false);
    return;
  }
  const end = sourceEnd();
  if (front.ended || front.currentTime * FPS >= end - 0.5) {
    state.sourceFrame = end - 1;
    setPlaying(false);
    return;
  }
  state.sourceFrame = front.currentTime * FPS;
  updateClock();
  requestAnimationFrame(tickSource);
}

function seek(frame) {
  if (viewing()) state.sourceFrame = Math.max(0, Math.min(sourceEnd() - 1, Math.round(frame)));
  else state.playhead = Math.max(0, Math.round(frame));
  if (state.playing) setPlaying(false);
  showFrame();
}

/* Rendering */

function updateQueue() {
  const box = $("queue");
  const count = state.jobs.size;
  if (!count) {
    box.replaceChildren();
    return;
  }
  const front = () => state.running || state.jobs.values().next().value;
  const clip = state.running && clipOf(state.running.clip);
  const waiting = count - (state.running ? 1 : 0);
  const parts = [];
  if (clip) parts.push(clip.name + ", " + Math.round((state.running.progress || 0) * 100) + "%");
  else if (state.running) parts.push("Rendering");
  if (waiting) parts.push(waiting + " waiting");
  let chip = box.firstElementChild;
  if (!chip) {
    chip = document.createElement("button");
    chip.className = "queue-chip";
    chip.title = "Show the clip that is rendering";
    chip.addEventListener("click", () => {
      const job = front();
      if (!job || !clipOf(job.clip)) return;
      selectClip(job.clip, state.project.timeline.find((i) => i.clip === job.clip)?.id ?? null);
    });
    box.append(chip);
  }
  chip.textContent = parts.join(", ");
}

// Renders are queued by the server, which also attaches the finished take to the project.
// The editor only shows progress, so closing it never loses a render.
async function renderClip(clip, quality) {
  if (jobFor(clip.id)) return;
  if (!clip.action.trim()) {
    selectClip(clip.id);
    toast("Describe what happens in " + clip.name + " before rendering it.", true);
    return;
  }
  try {
    await flush();
    const result = await postJson("/project/" + state.folder + "/render", { clip: clip.id, quality, client_id: clientId });
    state.jobs.set(result.job.id, { ...result.job, step: 0, max: 0, progress: 0 });
    if (!unsaved()) adopt(result.project);
    updateQueue();
    renderAll();
  } catch (error) {
    toast(error.message, true);
  }
}

async function cancelJob(job) {
  try {
    await flush();
    const result = await postJson("/project/" + state.folder + "/cancel", { job: job.id, client_id: clientId });
    job.reported = true;
    state.jobs.delete(job.id);
    if (!unsaved()) adopt(result.project);
    updateQueue();
    renderAll();
    toast("Render cancelled.");
  } catch (error) {
    toast(error.message, true);
  }
}

// The jobs shown are the project's own record of what is rendering, wherever it was started.
function syncJobs() {
  const next = new Map();
  for (const job of state.project.pending || []) next.set(job.id, state.jobs.get(job.id) || { ...job, step: 0, max: 0, progress: 0 });
  for (const [id, job] of state.jobs) {
    if (next.has(id)) continue;
    if (state.running === job) state.running = null;
    if (job.preview) URL.revokeObjectURL(job.preview);
  }
  state.jobs = next;
  updateQueue();
}

function connect() {
  const socket = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws?clientId=" + clientId);
  socket.binaryType = "arraybuffer";
  socket.onclose = () => setTimeout(connect, 2000);
  // Anything missed while the connection was down is in the project.
  socket.onopen = () => refreshProject(false).catch(() => {});
  socket.onmessage = (event) => {
    if (typeof event.data === "string") onMessage(JSON.parse(event.data));
    else onBinary(event.data);
  };
}

function onBinary(buffer) {
  const view = new DataView(buffer);
  const kind = view.getUint32(0);
  let start;
  let mime = "image/jpeg";
  if (kind === 1) {
    mime = view.getUint32(4) === 2 ? "image/png" : "image/jpeg";
    start = 8;
  } else if (kind === 4) {
    const length = view.getUint32(4);
    const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, length)));
    mime = meta.image_type || mime;
    start = 8 + length;
  } else {
    return;
  }
  const job = state.running;
  if (!job) return;
  if (job.preview) URL.revokeObjectURL(job.preview);
  job.preview = URL.createObjectURL(new Blob([new Uint8Array(buffer, start)], { type: mime }));
  const still = document.querySelector(".block.rendering .film.still");
  if (still) still.style.backgroundImage = "url(" + job.preview + ")";
  showLive(job);
}

function onMessage({ type, data }) {
  if (type === "h3take_project") {
    if (data.folder !== state.folder || data.client === clientId) return;
    if ((data.revision || 0) > (state.project?.revision || 0)) refreshProject(data.source !== "render").catch(() => {});
    return;
  }
  const job = data?.prompt_id ? state.jobs.get(data.prompt_id) : null;
  if (!job) return;
  if (type === "h3take_preview") {
    job.motion = viewUrl(data.filename, data.subfolder);
    showLive(job);
    const live = document.querySelector(".block.rendering video.livestrip");
    if (!live) renderTimeline();
    else if (live.dataset.url !== job.motion) {
      live.dataset.url = job.motion;
      live.src = job.motion;
      live.play().catch(() => {});
    }
    return;
  }
  if (state.running !== job) {
    state.running = job;
    updateQueue();
    renderTimeline();
    renderInspector();
  }
  if (type === "executing" && data.node) {
    job.stage = STAGES[data.node] || job.stage;
    if (data.node !== "15") job.max = 0;
    showFrame();
  } else if (type === "progress" && String(data.node) === "15") {
    job.step = data.value;
    job.max = data.max;
    job.stage = STAGES[15];
    job.progress = data.value / data.max;
    updateQueue();
    const bar = document.querySelector(".block.rendering .bar");
    if (bar) bar.style.width = Math.round(job.progress * 100) + "%";
    const clip = clipOf(job.clip);
    if (clip) {
      const text = clipStatus(clip).text;
      document.querySelectorAll('[data-status="' + clip.id + '"]').forEach((el) => {
        el.textContent = el.classList.contains("status") ? text + ", " + clipSeconds(clip) : text;
      });
    }
    showLive(job);
  } else if (type === "executed" && String(data.node) === "19") {
    job.stage = "Adding the take";
    showFrame();
  } else if (type === "execution_error") {
    job.reported = true;
    toast("The render of " + (clipOf(job.clip)?.name || "a clip") + " failed: " + (data.exception_message || "see the ComfyUI log"), true);
  }
}

/* Project */

function renderAll() {
  const focus = rememberFocus();
  renderCast();
  renderBin();
  renderInspector();
  renderTimeline();
  showFrame();
  restoreFocus(focus);
}

// A re-render must not take the caret away from someone typing in the inspector.
function rememberFocus() {
  const el = document.activeElement;
  if (!el || !$("inspector").contains(el) || !el.dataset.key) return null;
  return { key: el.dataset.key, start: el.selectionStart, end: el.selectionEnd };
}

function restoreFocus(focus) {
  if (!focus) return;
  const el = $("inspector").querySelector('[data-key="' + CSS.escape(focus.key) + '"]');
  if (!el) return;
  state.quiet = true;
  el.focus();
  state.quiet = false;
  try { el.setSelectionRange(focus.start, focus.end); } catch (_) {}
}

async function openProject(folder) {
  await flush();
  state.folder = folder;
  state.project = await api("/project/" + folder);
  state.edits = state.savedEdits = 0;
  state.clip = null;
  state.item = null;
  state.bin = null;
  setSource(null);
  loadFolded();
  state.playhead = 0;
  undo = [];
  redo = [];
  try { localStorage.setItem("h3take.project", folder); } catch (_) {}
  state.running = null;
  syncJobs();
  renderAll();
  fitTimeline();
  ensureStrips();
}

async function refreshProjects(selected) {
  const { projects, profiles, finals, loras, realism, checkpoints, samplers, schedulers, presets, refmods, encoders } = await api("/projects");
  state.encoders = encoders || { text_encoders: [], vaes: [], auto: {}, missing: [] };
  state.presets = presets || [];
  state.refmods = refmods || { installed: false, files: [] };
  // Adding a cast member from a RefMod only makes sense with the pack and a file to pick.
  $("addRefmod").hidden = !(state.refmods.installed && state.refmods.files.length);
  state.checkpoints = checkpoints || [];
  state.samplers = samplers || [];
  state.schedulers = schedulers || [];
  state.profiles = profiles || [];
  state.finals = finals || [];
  state.loras = loras || [];
  state.realism = realism || null;
  $("project").replaceChildren(...projects.map((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name.replaceAll("_", " ");
    option.selected = name === selected;
    return option;
  }));
  return projects;
}

async function createProject(name) {
  const { folder } = await postJson("/projects", { name });
  await refreshProjects(folder);
  await openProject(folder);
}

async function start() {
  let remembered = null;
  try { remembered = localStorage.getItem("h3take.project"); } catch (_) {}
  const projects = await refreshProjects(remembered);
  if (!projects.length) await createProject("First scene");
  else await openProject(projects.includes(remembered) ? remembered : projects[0]);
  connect();
}

$("project").addEventListener("change", (event) => openProject(event.target.value));
// A new project is named in a dialog of the editor's own. The browser's prompt box does not
// open in every place the editor runs, such as an embedded browser.
$("newProject").addEventListener("click", () => {
  const hint = $("creatorHint");
  $("creatorName").value = "";
  $("creatorFormat").value = "16:9";
  hint.textContent = "Its name becomes its folder in ComfyUI's output folder. The format can be changed later.";
  hint.classList.remove("bad");
  $("creatorGo").disabled = false;
  $("creator").showModal();
  $("creatorName").focus();
});
$("creatorClose").addEventListener("click", () => $("creator").close());
$("creatorForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const hint = $("creatorHint");
  const name = $("creatorName").value.trim();
  const bad = (text) => {
    hint.textContent = text;
    hint.classList.add("bad");
    $("creatorName").focus();
  };
  if (!/[A-Za-z0-9]/.test(name)) return bad("Give it a name with letters or numbers.");
  const folder = name.replace(/[^A-Za-z0-9 _-]/g, "").trim().replaceAll(" ", "_").slice(0, 64);
  if ([...$("project").options].some((o) => o.value.toLowerCase() === folder.toLowerCase())) return bad("There is already a project called " + folder.replaceAll("_", " ") + ". Pick another name.");
  $("creatorGo").disabled = true;
  try {
    await createProject(name);
    const format = $("creatorFormat").value;
    if (format !== (state.project.format || "16:9")) {
      state.project.format = format;
      save();
      renderAll();
    }
    $("creator").close();
  } catch (error) {
    bad(error.message);
  }
  $("creatorGo").disabled = false;
});
$("newClip").addEventListener("click", addClip);
$("addCast").addEventListener("click", addCastMember);
$("addVoice").addEventListener("click", addVoiceMember);
$("addRefmod").addEventListener("click", addRefmodMember);
$("importClip").addEventListener("click", importFootage);
$("newBin").addEventListener("click", addBin);

// Clips drag onto a bin to file them there, or onto the loose list to take them out of their bin.
const clipList = $("clips");
clipList.addEventListener("dragover", (event) => {
  const clip = clipOf(draggedClip);
  if (!clip) return;
  const target = dropBin(event);
  if ((binOf(clip.bin)?.id || null) === target) {
    showDrop(undefined);
    return;
  }
  event.preventDefault();
  event.dataTransfer.dropEffect = "move";
  showDrop(target);
});
clipList.addEventListener("dragleave", (event) => {
  if (!clipList.contains(event.relatedTarget)) showDrop(undefined);
});
clipList.addEventListener("drop", (event) => {
  const clip = clipOf(draggedClip);
  showDrop(undefined);
  if (!clip) return;
  event.preventDefault();
  fileClip(clip, dropBin(event));
});
$("close").addEventListener("click", () => window.parent.postMessage("h3-editor-close", location.origin));
$("renderAll").addEventListener("click", () => {
  const waiting = sorted().map((i) => clipOf(i.clip)).filter((c, i, all) => c && !c.takes.length && all.indexOf(c) === i);
  if (!waiting.length) return toast("Every clip on the timeline already has a render.");
  const ready = waiting.filter((c) => c.action.trim() && !jobFor(c.id));
  const blank = waiting.filter((c) => !c.action.trim());
  (async () => {
    if (ready.length) {
      try {
        await flush();
        // One batch, so the server can run the clips that share a model one after another.
        const ops = ready.map((c) => ({ op: "render", clip: c.id, quality: "draft", client_id: clientId }));
        await postJson("/project/" + state.folder + "/ops", { ops });
        adopt(await api("/project/" + state.folder));
        updateQueue();
        renderAll();
      } catch (error) {
        // One clip that cannot render stops a batch. Asked for one at a time, the others still go.
        try { adopt(await api("/project/" + state.folder)); } catch (_) {}
        for (const clip of ready) await renderClip(clip, "draft");
      }
    }
    if (blank.length) {
      toast((blank.length === 1 ? blank[0].name + " has" : blank.length + " clips have") + " nothing in What happens yet, so " + (blank.length === 1 ? "it was" : "they were") + " left out.", true);
    }
  })();
});
// Export joins the timeline into one video file. The dialog says what will be in it first.
const megabytes = (bytes) => (bytes >= 1e9 ? (bytes / 1e9).toFixed(2) + " GB" : Math.max(0.1, bytes / 1e6).toFixed(1) + " MB");

async function openExporter() {
  const dialog = $("exporter");
  const result = $("exporterResult");
  const go = $("exporterGo");
  clearTimeout(saveTimer);
  await flush();
  let plan;
  try {
    plan = (await postJson("/project/" + state.folder + "/export", { plan: true })).plan;
  } catch (error) {
    toast(error.message, true);
    return;
  }
  if (!plan.clips) {
    toast("There are no rendered clips on the timeline to export.", true);
    return;
  }
  const full = [plan.width, plan.height];
  const even = (side, scale) => Math.max(2, 2 * Math.round((side * scale) / 2));
  $("exporterScale").replaceChildren(...[[1, "As rendered"], [0.75, "Three quarters"], [0.5, "Half"]].map(([scale, label]) => {
    const option = document.createElement("option");
    option.value = String(scale);
    option.textContent = label + ", " + even(full[0], scale) + " x " + even(full[1], scale);
    return option;
  }));
  $("exporterSummary").textContent = timecode(plan.frames) + " (" + plan.seconds + " seconds), " + plan.clips + (plan.clips === 1 ? " clip" : " clips") + ", with sound, as an MP4.";
  const notes = [];
  if (plan.drafts) notes.push(plan.drafts + (plan.drafts === 1 ? " clip is still a draft" : " clips are still drafts") + ". Drafts are small, so they will look soft next to finals.");
  if (plan.missing) notes.push(plan.missing + (plan.missing === 1 ? " clip has no render and is" : " clips have no render and are") + " left out.");
  if (plan.gaps) notes.push(plan.gaps + (plan.gaps === 1 ? " gap" : " gaps") + " on the timeline will be black and silent.");
  $("exporterNotes").replaceChildren(...notes.map((text) => {
    const item = document.createElement("li");
    item.textContent = text;
    return item;
  }));
  $("exporterName").value = state.folder;
  try { $("exporterDownload").checked = localStorage.getItem("h3take.export.download") !== "no"; } catch (_) {}
  result.textContent = "It is saved in ComfyUI's output folder, under " + "h3take_projects/" + state.folder + "/exports.";
  result.classList.remove("bad");
  $("exporterLink").hidden = true;
  go.disabled = false;
  go.textContent = "Export";
  dialog.showModal();
}

async function runExport() {
  const go = $("exporterGo");
  const result = $("exporterResult");
  const link = $("exporterLink");
  const download = $("exporterDownload").checked;
  try { localStorage.setItem("h3take.export.download", download ? "yes" : "no"); } catch (_) {}
  go.disabled = true;
  go.textContent = "Exporting";
  link.hidden = true;
  result.classList.remove("bad");
  result.textContent = "Joining the clips. A long cut can take a minute or two.";
  try {
    const made = await postJson("/project/" + state.folder + "/export", {
      scale: Number($("exporterScale").value), quality: $("exporterQuality").value, name: $("exporterName").value.trim(),
    });
    link.href = viewUrl(made.filename, made.subfolder);
    link.setAttribute("download", made.filename);
    link.hidden = false;
    link.textContent = download ? "Download again" : "Download";
    result.textContent = "Saved " + made.filename + ", " + megabytes(made.bytes) + ", in " + made.subfolder + " of ComfyUI's output folder.";
    if (download) link.click();
  } catch (error) {
    result.classList.add("bad");
    result.textContent = error.message;
  }
  go.disabled = false;
  go.textContent = "Export again";
}

$("export").addEventListener("click", openExporter);
$("exporterGo").addEventListener("click", runExport);
$("exporterClose").addEventListener("click", () => $("exporter").close());
$("modelsClose").addEventListener("click", () => $("models").close());
$("play").addEventListener("click", () => setPlaying(!state.playing));
$("shortcuts").addEventListener("click", () => $("keys").showModal());
$("keysClose").addEventListener("click", () => $("keys").close());
bindSplitter();
bindSideSplitters();
$("toStart").addEventListener("click", () => seek(0));
$("back").addEventListener("click", () => seek(here() - 1));
$("forward").addEventListener("click", () => seek(here() + 1));
$("toTimeline").addEventListener("click", () => {
  setSource(null);
  showFrame();
});
$("split").addEventListener("click", splitAtPlayhead);
$("extendBack").addEventListener("click", () => addLinked("extend_before"));
$("extendForward").addEventListener("click", () => addLinked("extend_after"));
$("bridge").addEventListener("click", () => addLinked("bridge"));
$("merge").addEventListener("click", mergeWithNext);
$("ripple").addEventListener("click", rippleDelete);
$("closeGap").addEventListener("click", closeGap);
$("fit").addEventListener("click", fitTimeline);
// A menu opens under its button, or over it when the window is short, and closes once something is chosen.
for (const [button, id] of [["generate", "generateMenu"], ["castAdd", "castMenu"]]) {
  const menu = $(id);
  menu.addEventListener("toggle", (event) => {
    if (event.newState !== "open") return;
    const at = $(button).getBoundingClientRect();
    const below = at.bottom + 6 + menu.offsetHeight <= window.innerHeight;
    menu.style.left = Math.max(8, Math.min(at.left, window.innerWidth - menu.offsetWidth - 8)) + "px";
    menu.style.top = (below ? at.bottom + 6 : at.top - 6 - menu.offsetHeight) + "px";
  });
  menu.addEventListener("click", (event) => {
    if (event.target.closest("button")) menu.hidePopover();
  });
}
$("remove").addEventListener("click", removeSelected);
$("zoom").addEventListener("input", (event) => setZoom(Number(event.target.value)));
$("tlScroll").addEventListener("wheel", (event) => {
  if (!event.ctrlKey && !event.metaKey) return;
  event.preventDefault();
  const anchor = event.clientX - $("tlScroll").getBoundingClientRect().left;
  setZoom(state.zoom * (event.deltaY < 0 ? 1.15 : 1 / 1.15), anchor);
}, { passive: false });

$("ruler").addEventListener("pointerdown", (event) => {
  // A drag along the ruler moves the playhead. It must not start a text selection across the page.
  event.preventDefault();
  const frameAt = (ev) => (ev.clientX - $("ruler").getBoundingClientRect().left) / state.zoom;
  setSource(null);
  seek(frameAt(event));
  const move = (ev) => seek(frameAt(ev));
  const up = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", up);
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
});

const lane = $("laneV");
lane.addEventListener("dragover", (event) => {
  event.preventDefault();
  lane.classList.add("drop");
});
lane.addEventListener("dragleave", () => lane.classList.remove("drop"));
lane.addEventListener("drop", (event) => {
  event.preventDefault();
  lane.classList.remove("drop");
  const clip = clipOf(event.dataTransfer.getData("text/h3clip"));
  if (!clip) return;
  checkpoint();
  const frame = Math.max(0, Math.round((event.clientX - lane.getBoundingClientRect().left) / state.zoom));
  placeOnTimeline(clip, snap(frame, snapPoints(null)));
  save();
  renderTimeline();
});
$("laneV").addEventListener("pointerdown", (event) => {
  if (event.target !== lane && !event.target.classList.contains("gapmark")) return;
  const frame = (event.clientX - lane.getBoundingClientRect().left) / state.zoom;
  selectClip(null);
  state.gap = gapAt(frame);
  renderTimeline();
});

window.addEventListener("keydown", (event) => {
  if (/INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) return;
  if (document.querySelector("dialog[open], [popover]:popover-open")) return;
  const tab = event.target.closest?.("#inspTabs .insp-tab");
  if (tab && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
    // In the inspector's tabs the arrows move between tabs, as tabs do everywhere.
    event.preventDefault();
    const next = event.key === "ArrowLeft" ? tab.previousElementSibling : tab.nextElementSibling;
    if (next) showTab(next.dataset.tab);
    return;
  }
  if (event.key === "Escape") {
    if (state.clip || state.item || state.person || state.bin || state.gap) selectClip(null);
  } else if (event.code === "Space") {
    event.preventDefault();
    setPlaying(!state.playing);
  } else if (event.key === "?") $("keys").showModal();
  else if (event.key.toLowerCase() === "k" && !event.ctrlKey && !event.metaKey) setPlaying(false);
  else if (event.key.toLowerCase() === "l" && !event.ctrlKey && !event.metaKey) {
    if (!state.playing) setPlaying(true);
  } else if (event.key.toLowerCase() === "j" && !event.ctrlKey && !event.metaKey) seek(here() - FPS);
  else if (event.key === "ArrowLeft") seek(here() - (event.shiftKey ? FPS : 1));
  else if (event.key === "ArrowRight") seek(here() + (event.shiftKey ? FPS : 1));
  else if (event.key === "ArrowUp") {
    event.preventDefault();
    jumpEdit(-1);
  } else if (event.key === "ArrowDown") {
    event.preventDefault();
    jumpEdit(1);
  } else if (event.key === "Home") seek(0);
  else if (event.key === "End") seek(viewEnd());
  else if (event.key === "[") cycleTake(-1);
  else if (event.key === "]") cycleTake(1);
  else if (event.key === "\\") fitTimeline();
  else if (event.key === "Delete" || event.key === "Backspace") {
    // In the Clips panel the key deletes the clip or bin that has focus. On the timeline it only removes the clip from there.
    const row = event.target.closest?.("#clips .clip");
    const bin = event.target.closest?.("#clips .bin-head");
    if (row) deleteClip(clipOf(row.dataset.clip));
    else if (bin) deleteBin(binOf(bin.closest(".bin-group").dataset.bin));
    else if (event.shiftKey) rippleDelete();
    else removeSelected();
  }
  else if (event.key.toLowerCase() === "s" && !event.ctrlKey && !event.metaKey) splitAtPlayhead();
  else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    if (event.shiftKey) restore(redo, undo);
    else restore(undo, redo);
  }
});
window.addEventListener("resize", () => {
  for (const side of Object.values(SIDES)) {
    let saved = 0;
    try { saved = Number(localStorage.getItem(side.key)); } catch (_) {}
    if (saved) setSideWidth(side, saved);
  }
  if (state.project) renderTimeline();
});

start().catch((error) => toast(error.message, true));
