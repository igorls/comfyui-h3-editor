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

import { VfxRuntime } from "./vfx-runtime.js";
import * as writer from "./writer.js";

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
  sizes: null,
  upscalers: [],
  checkpoints: [],
  presets: [],
  refmods: { installed: false, files: [] },
  encoders: { text_encoders: [], vaes: [], auto: {}, missing: [] },
  samplers: [],
  schedulers: [],
  bin: null,
  sound: null,
  effect: null,
  range: null,
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
  if (!effectOf(state.effect)) state.effect = null;
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

// A clip on the timeline shows a window of its take. Slipping slides the take under that window:
// the clip keeps its place and its length, and plays an earlier or a later part of the take.
function takeRoom(item) {
  const clip = clipOf(item.clip);
  const take = clip && activeTake(clip);
  if (!take) return null;
  const [first, last] = linkedRange(clip.link, take.frames);
  return { first, last, before: Math.max(0, item.in - first), after: Math.max(0, last - item.out) };
}

function slipBy(item, frames) {
  const room = takeRoom(item);
  if (!room) return 0;
  const shift = Math.max(-room.before, Math.min(room.after, Math.round(frames)));
  item.in += shift;
  item.out += shift;
  return shift;
}

function slipSelected(frames) {
  const item = state.project.timeline.find((i) => i.id === state.item);
  const room = item && takeRoom(item);
  if (!room) return;
  if (!room.before && !room.after) {
    toast("This clip already shows its whole take, so there is nothing to slip.");
    return;
  }
  checkpoint();
  if (slipBy(item, frames)) {
    save();
    renderAll();
  }
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
  $("remove").disabled = !(state.item || state.gap || state.sound || state.effect);
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

// A place a picture file can be dropped on: it lights up while one is dragged over it, and hands the dropped
// picture to take(file). Anything that is not a picture is turned away with a message.
function dropTarget(element, take, what = "a picture") {
  const pictures = (event) => [...(event.dataTransfer?.items || [])].some((row) => row.kind === "file");
  let depth = 0;
  element.addEventListener("dragenter", (event) => {
    if (!pictures(event)) return;
    event.preventDefault();
    depth += 1;
    element.classList.add("drop-hot");
  });
  element.addEventListener("dragover", (event) => {
    if (!pictures(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  });
  element.addEventListener("dragleave", () => {
    depth = Math.max(0, depth - 1);
    if (!depth) element.classList.remove("drop-hot");
  });
  element.addEventListener("drop", (event) => {
    if (!pictures(event)) return;
    event.preventDefault();
    event.stopPropagation();
    depth = 0;
    element.classList.remove("drop-hot");
    const file = [...event.dataTransfer.files].find((f) => /^image\/(png|jpeg|webp)$/.test(f.type));
    if (!file) return toast("Drop " + what + ": a PNG, JPEG or WebP file.", true);
    take(file);
  });
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
  // The box takes the project's shape, so a frame shows whole.
  const [wide, tall] = String(state.project.format || "16:9").split(":").map(Number);
  if (wide > 0 && tall > 0) picture.style.setProperty("--frame", String(wide / tall));
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
    picture.title = "Drop a picture here to replace it";
    button("Remove", () => setClipFrame(clip, side, null));
  } else {
    picture.textContent = "Drop a picture here";
    button("Use playhead frame", () => grabFrame(clip, side));
    button("Upload", () => uploadFrame(clip, side));
  }
  dropTarget(picture, async (file) => {
    try {
      setClipFrame(clip, side, await uploadPicture(file));
      toast(label + " of " + clip.name + " is " + file.name + ".");
    } catch (error) {
      toast(error.message, true);
    }
  }, "a picture for the " + label.toLowerCase());
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
    button.dataset.person = person.id;
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
    button.draggable = true;
    button.addEventListener("dragstart", (event) => {
      // Dropped on a text field, the browser types this where it lands.
      event.dataTransfer.setData("text/plain", "@" + person.name + " ");
      event.dataTransfer.setData("application/x-h3-cast", person.id);
      event.dataTransfer.effectAllowed = "copy";
    });
    button.addEventListener("click", () => {
      setSource(null);
      state.person = person.id;
      state.clip = null;
      state.item = null;
      state.bin = null;
      state.effect = null;
      state.sound = null;
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
  if (state.refmods.installed && person.file) {
    const making = current && !known;
    const make = document.createElement("button");
    make.className = "quiet wide";
    make.dataset.key = "Make a RefMod";
    make.textContent = making ? "Making the RefMod" : person.refmod ? "Make it again from these pictures" : "Make a RefMod from these pictures";
    make.disabled = making;
    make.addEventListener("click", () => makeRefmod(person));
    parts.push(make);
  }
  return parts;
}

// A cast member dropped into a clip's text is in the shot, so the clip takes its pictures too.
$("inspector").addEventListener("drop", (event) => {
  const id = event.dataTransfer?.getData("application/x-h3-cast");
  const clip = clipOf(state.clip);
  if (!id || !clip || !event.target.matches?.("textarea, input[type=text], input:not([type])")) return;
  const person = (state.project.cast || []).find((c) => c.id === id);
  if (!person || !(person.file || person.refmod) || (clip.cast || []).includes(id)) return;
  // After the browser has typed the name and the field has saved it.
  setTimeout(() => {
    const live = clipOf(clip.id);
    if (!live || (live.cast || []).includes(id)) return;
    live.cast = [...(live.cast || []), id];
    save();
    renderInspector();
    toast(person.name + " is now in the cast of " + live.name + ".");
  }, 0);
});

// Makes a RefMod from a cast member's pictures with the RefMod pack, then waits for its file to appear.
async function makeRefmod(person) {
  try {
    await flush();
    const reply = await postJson("/project/" + state.folder + "/ops", { ops: [{ op: "make_refmod", cast: person.id, client_id: clientId }] });
    const name = reply.results[0].refmod;
    adopt(await api("/project/" + state.folder));
    renderAll();
    toast("Making a RefMod from the pictures of " + person.name + ". It takes a minute or so.");
    for (let tries = 0; tries < 90; tries++) {
      await new Promise((done) => setTimeout(done, 4000));
      const { refmods } = await api("/projects");
      if (!refmods?.files?.includes(name)) continue;
      state.refmods = refmods;
      renderInspector();
      toast("The RefMod of " + person.name + " is ready. New renders with " + person.name + " use it.");
      return;
    }
    toast("The RefMod of " + person.name + " has not appeared yet. Look at the ComfyUI log if it does not.", true);
  } catch (error) {
    toast(error.message, true);
  }
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

function removePerson(person) {
  checkpoint();
  state.project.cast = state.project.cast.filter((c) => c !== person);
  for (const clip of state.project.clips) clip.cast = (clip.cast || []).filter((id) => id !== person.id);
  if (state.person === person.id) state.person = null;
  save();
  renderAll();
  toast("Removed " + person.name + " from the cast.", false, undoAction);
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
  row.append(title, trashButton("Remove " + person.name + " from the cast", () => removePerson(person)));
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
  state.sound = null;
  state.effect = null;
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
  const quality = take.upscaled ? "Upscaled" : take.quality === "final" ? "Final" : "Draft";
  return { kind: "", text: clip.takes.length > 1 ? quality + ", take " + (clip.takes.indexOf(take) + 1) + " of " + clip.takes.length : quality };
}

function clipSeconds(clip) {
  const take = activeTake(clip);
  return (take ? take.frames / FPS : clip.seconds).toFixed(1) + " s";
}

// Run the pointer across a thumbnail to scrub through the clip, frame strip permitting.
// fit is "cover" for the small thumbnails of the Clips panel, which crop to fill, or "contain" for a picture
// that must show whole, such as a take in the Takes tab.
function skim(thumb, take, fit = "cover") {
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
    // Each frame sits in the thumbnail as the poster does, cropped to cover it or shown whole, never squeezed.
    const scale = w && h ? (fit === "contain" ? Math.min : Math.max)(rect.width / w, rect.height / h) : 0;
    const tileW = scale ? w * scale : rect.width;
    const tileH = scale ? h * scale : rect.height;
    thumb.style.backgroundImage = sprite;
    thumb.style.backgroundSize = count * tileW + "px " + tileH + "px";
    if (fit === "contain") {
      // A frame smaller than the box would show its neighbours in the strip beside it: the picture is clipped
      // to a box of the frame's own size, in the middle.
      const sideX = Math.max(0, (rect.width - tileW) / 2);
      const sideY = Math.max(0, (rect.height - tileH) / 2);
      thumb.style.padding = sideY + "px " + sideX + "px";
      thumb.style.backgroundClip = "content-box";
      thumb.style.backgroundOrigin = "content-box";
      thumb.style.backgroundPosition = -index * tileW + "px 0px";
    } else {
      thumb.style.backgroundPosition = (rect.width - tileW) / 2 - index * tileW + "px " + (rect.height - tileH) / 2 + "px";
    }
    line.hidden = false;
    line.style.left = share * 100 + "%";
  });
  thumb.addEventListener("pointerleave", () => {
    thumb.style.backgroundImage = poster;
    thumb.style.backgroundSize = "";
    thumb.style.backgroundPosition = "";
    thumb.style.padding = "";
    thumb.style.backgroundClip = "";
    thumb.style.backgroundOrigin = "";
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
  if (state.project.notes) state.project.notes = state.project.notes.filter((n) => n.clip !== clip.id);
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
  const open = notesOf().filter((n) => n.clip === clip.id && n.status !== "resolved").length;
  if (open) {
    const count = document.createElement("span");
    count.className = "notes";
    count.textContent = open === 1 ? "1 note" : open + " notes";
    meta.append(count);
  }
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
// Files the user starred in a picker, kept on this browser under the picker's own key.
function starred(key) {
  try { return new Set(JSON.parse(localStorage.getItem(key) || "[]")); } catch (_) { return new Set(); }
}

function pickFile2(title, files, hint, favoritesKey = null) {
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
      const stars = favoritesKey ? starred(favoritesKey) : null;
      // Favourites come first, each group in the order the folder lists them.
      if (stars) shown = [...shown.filter((f) => stars.has(f)), ...shown.filter((f) => !stars.has(f))];
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
        if (!stars) return item;
        const row = document.createElement("div");
        row.className = "picker-row";
        const star = document.createElement("button");
        star.className = "picker-star";
        star.setAttribute("aria-pressed", String(stars.has(file)));
        star.setAttribute("aria-label", (stars.has(file) ? "Remove " : "Add ") + loraName(file) + (stars.has(file) ? " from" : " to") + " favorites");
        star.title = stars.has(file) ? "A favorite. Click to remove it" : "Make it a favorite, so it lists first";
        star.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4.5l2.3 4.7 5.2.8-3.8 3.6.9 5.2-4.6-2.5-4.6 2.5.9-5.2-3.8-3.6 5.2-.8z"/></svg>';
        star.addEventListener("click", () => {
          if (stars.has(file)) stars.delete(file);
          else stars.add(file);
          try { localStorage.setItem(favoritesKey, JSON.stringify([...stars])); } catch (_) {}
          draw();
        });
        row.append(item, star);
        return row;
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

// Rows whose timing control is open, so it stays open while the panel is drawn again.
const loraTiming = new Set();

// The project's LoRAs, which every render samples with, or one clip's own, each at its own strength.
function loraSection(clip = null) {
  // A save swaps the project for the server's copy, so the list is found again each time it is changed.
  const owner = () => (clip ? clipOf(clip.id) : state.project) || {};
  const rows = (owner().loras || []).filter((row) => row && typeof row.file === "string");
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
  if (!clip && state.realism?.installed) {
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
    const live = () => (owner().loras || []).find((r) => r && r.id === entry.id) || {};
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
    // When it joins: on every step, or only after the first ones have set the composition.
    const start = typeof entry.start_sigma === "number" ? entry.start_sigma : 1;
    const timed = start < 1 || loraTiming.has(entry.id);
    const when = document.createElement("button");
    when.className = "quiet small";
    when.dataset.key = "LoRA " + entry.id + " timing";
    when.textContent = start < 1 ? "From " + start.toFixed(2) : "All steps";
    when.title = "When the LoRA joins the render. Click to " + (timed ? "hide" : "set") + " it";
    when.setAttribute("aria-expanded", String(timed));
    when.disabled = start < 1;
    when.addEventListener("click", () => {
      if (loraTiming.has(entry.id)) loraTiming.delete(entry.id);
      else loraTiming.add(entry.id);
      renderInspector();
    });
    head.append(when);
    head.append(trashButton("Remove " + loraName(entry.file), () => change(() => {
      owner().loras = owner().loras.filter((r) => r !== live());
      if (clip && !owner().loras.length) delete owner().loras;
    })));
    const strength = typeof entry.strength === "number" ? entry.strength : 1;
    row.append(head, strengthSlider("LoRA " + entry.id, loraName(entry.file) + " strength", strength, limits.lora_range, 1, (v) => (live().strength = v)));
    if (timed) {
      const timing = document.createElement("div");
      timing.className = "lora-start";
      const label = document.createElement("span");
      label.textContent = "Start";
      label.title = "1 runs it on every step. Lower leaves the first steps alone, so the framing and pose stay as they would be without it. Try 0.75.";
      timing.append(label, strengthSlider("LoRA " + entry.id + " start", loraName(entry.file) + " start, from 1 for every step down to 0", start, [0, 1], 1, (v) => {
        if (v < 1) live().start_sigma = v;
        else delete live().start_sigma;
        when.textContent = v < 1 ? "From " + v.toFixed(2) : "All steps";
        when.disabled = v < 1;
      }));
      row.append(timing);
    }
    list.append(row);
  }
  if (list.children.length) children.push(list);
  const free = state.loras.filter((f) => !rows.some((r) => sameFile(r.file, f)) && !(state.realism && sameFile(state.realism.file, f)));
  if (rows.length >= limits.limit) {
    const full = document.createElement("p");
    full.className = "hint";
    full.textContent = (clip ? "A clip" : "A project") + " takes up to " + limits.limit + " LoRAs. Remove one to add another.";
    children.push(full);
  } else if (free.length) {
    const add = document.createElement("button");
    add.className = "quiet wide";
    add.textContent = "Add a LoRA";
    add.dataset.key = clip ? "Add a LoRA to this clip" : "Add a LoRA";
    add.addEventListener("click", async () => {
      const file = await pickFile2(clip ? "Add a LoRA to " + clip.name : "Add a LoRA", free,
        "From models/loras. Only LoRAs trained on MiniMax H3 have an effect. Star the ones you use, and they list first.", "h3take.loraFavorites");
      if (!file) return;
      change(() => {
        owner().loras = [...(owner().loras || []), { id: "l" + crypto.randomUUID().replaceAll("-", "").slice(0, 8), file, strength: 1, on: true }];
      });
    });
    children.push(add);
  }
  const hint = document.createElement("p");
  hint.className = "hint";
  if (clip) hint.textContent = "Only this clip samples with these, after the project's. Add one the project also has to give it another strength here, or untick it to leave it out of this clip.";
  else hint.textContent = state.loras.length
    ? "Every new render samples with these, drafts and finals alike. They only work if they were trained on MiniMax H3."
    : "Put LoRAs trained on MiniMax H3 in models/loras, then reload this page to add them here.";
  children.push(hint);
  return clip ? children : section("LoRAs", children);
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
    // The sizes on offer follow the format.
    renderInspector();
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
  manageModels.type = "button";
  manageModels.className = "label-action";
  manageModels.textContent = "Models and their settings";
  manageModels.dataset.key = "Models";
  manageModels.addEventListener("click", (event) => {
    event.preventDefault();
    openModels(null);
  });
  modelTitle.classList.add("label-row");
  modelTitle.append(manageModels);
  panel.replaceChildren(
    head,
    section("Look", [field("Style for every clip", state.project.style, (v) => (state.project.style = v), { area: true })]),
    section("Format and model", [
      shape,
      formatHint,
      ...sizeChoices(),
      ...(state.profiles.length ? [model, modelHint] : []),
      ...encoderChoices(),
      field("Language people speak", state.project.language || "", (v) => {
        if (v.trim()) state.project.language = v.trim();
        else delete state.project.language;
      }, { placeholder: "English" }),
    ]),
    ...[loraSection(), writingSection(), notesSection(), section("Overview", [facts])].filter(Boolean),
  );
}

// The sizes drafts and finals render at. The first of each list is the usual one, and a project
// names a size only when it picks another.
function sizeChoices() {
  const table = state.sizes?.[state.project.format || "16:9"];
  if (!table) return [];
  const names = { small: "Small", medium: "Medium", large: "Large", full: "Full", sharp: "Sharp" };
  // A sharp final goes through RelayVSR, so it is on offer only where that is installed.
  const relay = (state.upscalers || []).find((row) => row.id === "relayvsr");
  const choice = (quality, title) => {
    const label = document.createElement("label");
    label.className = "choice";
    const name = document.createElement("span");
    name.textContent = title;
    const select = document.createElement("select");
    select.dataset.key = title;
    const usual = table[quality][0].id;
    const current = table[quality].some((row) => row.id === state.project.sizes?.[quality]) ? state.project.sizes[quality] : usual;
    for (const row of table[quality]) {
      const option = document.createElement("option");
      option.value = row.id;
      option.textContent = names[row.id] + ", " + row.out[0] + " × " + row.out[1];
      option.selected = row.id === current;
      if (row.id === "sharp" && relay && !relay.installed) {
        option.disabled = row.id !== current;
        option.textContent += " (needs RelayVSR)";
      }
      select.append(option);
    }
    select.addEventListener("change", () => {
      checkpoint();
      const sizes = { ...(state.project.sizes || {}) };
      if (select.value === usual) delete sizes[quality];
      else sizes[quality] = select.value;
      if (Object.keys(sizes).length) state.project.sizes = sizes;
      else delete state.project.sizes;
      save();
    });
    label.append(name, select);
    return label;
  };
  const hint = document.createElement("p");
  hint.className = "hint";
  const full = table.final.find((row) => row.id === "full");
  const sharp = table.final.find((row) => row.id === "sharp");
  hint.textContent = "New renders use these sizes. A smaller one needs less GPU memory and renders faster."
    + (full ? " Full samples at " + full.width + " × " + full.height + ", then redraws its last steps at " + full.out[0] + " × " + full.out[1] + "." : "")
    + (sharp ? " Sharp samples at the same size once, enlarges the picture two times with RelayVSR and scales it down to "
      + sharp.out[0] + " × " + sharp.out[1] + ": quicker than Full, with crisper edges and the smoother skin of the sampled size."
      + (relay && !relay.installed ? " It needs " + relay.missing.join(" and ") + " from " + relay.source + " on Hugging Face." : "") : "")
    + " Steps are set per model, in Models and their settings.";
  const pair = document.createElement("div");
  pair.className = "pair";
  pair.append(choice("draft", "Draft size"), choice("final", "Final size"));
  return [pair, hint];
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

// Explanations longer than this are help for people new to the editor: hidden unless Help is on, and kept as the
// tip of the control they explain. Shorter lines say what is the case, and always show.
const HELP_LONGER = 90;
let showHelp = false;
try { showHelp = localStorage.getItem("h3take.help") === "on"; } catch (_) {}

function tidyHints(panel) {
  $("inspector").classList.toggle("show-help", showHelp);
  for (const hint of panel.querySelectorAll(".hint")) {
    const text = hint.textContent.trim();
    if (text.length <= HELP_LONGER) continue;
    hint.classList.add("help");
    // The tip goes on what the explanation follows: a field, a choice, a button or a row of them.
    const before = hint.previousElementSibling;
    const target = before && (before.matches(".field, .choice, .pair, button, .ways, .row, .speeds, .voice, .picture-row, h3") ? before : null);
    if (target && !target.title) target.title = text;
  }
}

function helpToggle() {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "insp-help";
  button.setAttribute("aria-pressed", String(showHelp));
  button.setAttribute("aria-label", "Help");
  button.title = showHelp ? "Hide the explanations" : "Show the explanations under each setting";
  button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.5"/><path d="M9.6 9.4a2.5 2.5 0 0 1 4.8.9c0 1.7-2.4 2.2-2.4 3.7M12 16.8h.01"/></svg>';
  button.addEventListener("click", () => {
    showHelp = !showHelp;
    try { localStorage.setItem("h3take.help", showHelp ? "on" : "off"); } catch (_) {}
    renderInspector();
  });
  return button;
}

function renderInspector() {
  drawInspector();
  tidyHints($("inspBody"));
  $("inspTabs").querySelector(".insp-help")?.remove();
  $("inspTabs").append(helpToggle());
}

function drawInspector() {
  const panel = $("inspBody");
  const effect = effectOf(state.effect);
  if (effect) {
    const own = "x" + effect.id;
    if (own !== inspected) {
      state.tab = "shot";
      inspected = own;
      $("inspector").scrollTop = 0;
    }
    if (state.tab !== "project") state.tab = "shot";
    renderTabs([["shot", "VFX"], ["project", "Project"]]);
    if (state.tab === "project") renderScene(panel, true);
    else renderEffect(panel, effect);
    return;
  }
  const person = (state.project.cast || []).find((c) => c.id === state.person);
  const clip = person ? null : clipOf(state.clip);
  const bin = !clip && !person ? selectedBin() : null;
  const sound = !clip && !person && !bin ? soundOf(state.sound) : null;
  const placed = clip ? state.project.timeline.find((i) => i.id === state.item) : null;
  const tabs = [];
  if (person) tabs.push(["shot", "Cast member"]);
  else if (bin) tabs.push(["shot", "Bin"]);
  else if (sound) tabs.push(["shot", "Sound"]);
  else if (clip && clip.kind === "footage") tabs.push(["shot", "Footage"]);
  else if (clip) tabs.push(["shot", "Shot"], ["takes", "Takes", clip.takes.length]);
  if (placed) tabs.push(["timeline", "Timeline"]);
  tabs.push(["project", "Project"]);
  const key = person ? "p" + person.id : clip ? "c" + clip.id : bin ? "b" + bin.id : sound ? "s" + sound.id : "";
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
  if (sound) {
    renderSoundPanel(panel, sound);
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
    rewriteButton(clip),
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
    const own = (clip.loras || []).length;
    if (state.loras.length || own) parts.push(disclosure("LoRAs for this clip", own > 0, own ? own + " set" : "", loraSection(clip)));
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
  let upscale = null;
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
    const shown = activeTake(clip);
    const upscalers = (state.upscalers || []).filter((row) => row.installed);
    if (shown && shown.quality === "final" && !shown.upscaled && upscalers.length) {
      // A finished final can be enlarged two times: a new take of the same picture and sound, at twice the size.
      // Which upscaler is the user's choice, among the installed ones: they do very different things to a picture.
      const shape = takeShape(shown);
      const remembered = (key, fallback) => {
        try {
          return localStorage.getItem(key) || fallback;
        } catch (_) {
          return fallback;      // no storage: the choice lasts for this visit
        }
      };
      const remember = (key, value) => {
        try {
          localStorage.setItem(key, value);
        } catch (_) {
          // Kept for this visit only.
        }
      };
      const which = document.createElement("select");
      which.setAttribute("aria-label", "Upscaler");
      const chosen = remembered("h3.upscaler", upscalers[0].id);
      for (const row of upscalers) {
        const option = document.createElement("option");
        option.value = row.id;
        option.textContent = row.title;
        option.selected = row.id === chosen;
        which.append(option);
      }
      // RelayVSR only: how much of the detail it adds is kept. All of it sharpens most, and can age a face.
      const detail = document.createElement("select");
      detail.setAttribute("aria-label", "How much detail the upscale adds");
      detail.title = "How much of the detail the upscaler adds is kept. More is sharper; on a face, all of it brings out pores and shine and can make it look older.";
      const kept = remembered("h3.upscaleDetail", "0.5");
      for (const [value, label] of [["0.25", "A little detail"], ["0.5", "Some detail"], ["0.75", "More detail"], ["1", "All detail"]]) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        option.selected = value === kept;
        detail.append(option);
      }
      detail.addEventListener("change", () => remember("h3.upscaleDetail", detail.value));
      const button = document.createElement("button");
      button.className = "quiet";
      button.textContent = "Upscale 2x";
      const settle = () => {
        const row = upscalers.find((r) => r.id === which.value) || upscalers[0];
        detail.hidden = !row.detail;
        which.title = row.about;
        button.title = "Enlarge this final to " + 2 * shape[0] + " x " + 2 * shape[1] + " with " + row.title + ". " + row.about
          + " It becomes a new take; do it once the cut is settled.";
      };
      which.addEventListener("change", () => {
        remember("h3.upscaler", which.value);
        settle();
      });
      settle();
      button.addEventListener("click", () => renderClip(clip, "upscale", detail.hidden ? undefined : Number(detail.value), which.value));
      // Its own line under the render buttons: a choice made after a final, not a third way to render one.
      upscale = document.createElement("div");
      upscale.className = "actions upscale-row";
      upscale.setAttribute("role", "group");
      upscale.setAttribute("aria-label", "Upscale this final");
      upscale.append(which, detail, button);
    }
  }
  foot.append(actions);
  if (upscale) foot.append(upscale);
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
    // Its picture scrubs through the take as the pointer crosses it, like a clip in the Clips panel.
    const img = document.createElement("div");
    img.className = "take-thumb";
    // The box takes the project's shape, the same for every take so they line up, and the picture shows whole in it.
    const [wide, tall] = String(state.project.format || "16:9").split(":").map(Number);
    const shape = wide > 0 && tall > 0 ? wide / tall : 16 / 9;
    img.style.setProperty("--frame", String(shape));
    skim(img, take, "contain");
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
  row.className = "speeds transition";
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
  const use = takeUse(item);
  if (use) box.append(use);
  return [box];
}

// How much of its take a clip shows, and a window to drag: the clip stays where it is and shows another part.
function takeUse(item) {
  const room = takeRoom(item);
  if (!room) return null;
  const total = room.last - room.first;
  const box = document.createElement("div");
  box.className = "take-use";
  const title = document.createElement("span");
  title.textContent = "Part of the take";
  const bar = document.createElement("div");
  bar.className = "take-bar";
  const win = document.createElement("div");
  win.className = "take-window";
  const place = () => {
    win.style.left = (100 * (item.in - room.first)) / total + "%";
    win.style.width = (100 * (item.out - item.in)) / total + "%";
  };
  place();
  bar.append(win);
  const note = document.createElement("p");
  note.className = "hint";
  const say = () => {
    const before = item.in - room.first;
    const after = room.last - item.out;
    note.textContent = !before && !after
      ? "The clip shows its whole take."
      : "Shows " + (item.out - item.in) + " of " + total + " frames: " + before + " unused before, " + after + " after. "
        + "Drag the window, Alt-drag the clip, or press , and . to show another part. The clip keeps its place and length.";
  };
  say();
  const free = room.before || room.after;
  win.tabIndex = free ? 0 : -1;
  win.setAttribute("role", "slider");
  win.setAttribute("aria-label", "Part of the take the clip shows");
  win.setAttribute("aria-valuemin", "0");
  win.setAttribute("aria-valuemax", String(total - (item.out - item.in)));
  win.setAttribute("aria-valuenow", String(item.in - room.first));
  if (free) {
    win.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      const origin = { x: event.clientX, in: item.in, out: item.out };
      const scale = total / bar.getBoundingClientRect().width;
      let changed = false;
      const move = (ev) => {
        const delta = Math.round((ev.clientX - origin.x) * scale);
        if (delta !== 0 && !changed) {
          checkpoint();
          changed = true;
        }
        item.in = origin.in;
        item.out = origin.out;
        slipBy(item, delta);
        place();
        say();
        showFrame();
      };
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        if (changed) {
          save();
          renderAll();
        }
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    });
    win.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      event.stopPropagation();
      slipSelected((event.key === "ArrowRight" ? 1 : -1) * (event.shiftKey ? 12 : 1));
    });
  }
  box.append(title, bar, note);
  return box;
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

/* Extra sound: music, narration or effects on lanes of their own, under the clips' own sound. */

const SOUND_LANES = 4;
const soundOf = (id) => (state.project?.sounds || []).find((s) => s.id === id) || null;
const soundPlayers = new Map();

// The lanes follow the sounds: as many as the highest lane in use, and none in a project without extra sound.
function layoutSoundLanes(count) {
  const heads = document.querySelector(".tl-heads");
  const content = $("tlContent");
  if (Number(content.dataset.lanes || 0) !== count) {
    content.dataset.lanes = count;
    for (const old of document.querySelectorAll(".tl-head.extra, .lane.extra")) old.remove();
    for (let n = 0; n < count; n++) {
      const head = document.createElement("div");
      head.className = "tl-head extra";
      head.textContent = "Sound " + (n + 2);
      heads.append(head);
      const lane = document.createElement("div");
      lane.className = "lane a extra";
      lane.dataset.lane = n;
      content.insertBefore(lane, $("playhead"));
    }
    const rows = count ? "28px 34px minmax(56px, 2fr) minmax(36px, 1fr)" + " 42px".repeat(count) : "";
    heads.style.gridTemplateRows = rows;
    content.style.gridTemplateRows = rows;
    // The timeline grows by its extra lanes, so the picture lane keeps its height.
    document.documentElement.style.setProperty("--tl-extra", count * 42 + "px");
  }
  // The lanes scroll sideways and their names do not, so the names leave room for the scrollbar under the lanes.
  heads.style.paddingBottom = $("tlScroll").offsetHeight - $("tlScroll").clientHeight + "px";
  return [...content.querySelectorAll(".lane.extra")];
}

function renderSounds() {
  const sounds = (state.project.sounds || []).filter((s) => s && typeof s.file === "string");
  const lanes = layoutSoundLanes(sounds.length ? Math.min(SOUND_LANES, Math.max(...sounds.map((s) => s.lane || 0)) + 1) : 0);
  const focused = document.activeElement?.dataset?.sound || null;
  const made = lanes.map(() => []);
  for (const sound of sounds) {
    const block = document.createElement("div");
    const selected = sound.id === state.sound;
    block.className = "wave sound" + (selected ? " on" : "");
    block.dataset.sound = sound.id;
    block.style.left = px(sound.start) + "px";
    block.style.width = Math.max(10, px(sound.frames)) + "px";
    block.tabIndex = 0;
    block.setAttribute("role", "button");
    block.setAttribute("aria-pressed", String(selected));
    block.setAttribute("aria-label", sound.name + ", extra sound on lane " + ((sound.lane || 0) + 2) + ", at " + timecode(sound.start) + " on the timeline");
    const canvas = document.createElement("canvas");
    // The waveform reads the same loudness file a clip's does, made when the sound was added.
    const from = (sound.offset || 0) * FPS;
    drawWave(canvas, { in: from, out: from + sound.frames }, { file: sound.file, subfolder: sound.subfolder,
      strip: { peaks: sound.file.replace(/\.[^.]+$/, "") + ".peaks.json" } }, 30, selected);
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = sound.name;
    block.append(canvas, name);
    for (const side of ["l", "r"]) {
      const edge = document.createElement("div");
      edge.className = "edge " + side;
      block.append(edge);
    }
    bindSound(block, sound.id);
    (made[Math.min(lanes.length - 1, sound.lane || 0)] || made[0]).push(block);
  }
  lanes.forEach((lane, n) => lane.replaceChildren(...made[n]));
  if (focused) document.querySelector('.wave.sound[data-sound="' + CSS.escape(focused) + '"]')?.focus();
}

function selectSound(id) {
  setSource(null);
  state.effect = null;
  state.sound = id;
  state.clip = null;
  state.item = null;
  state.gap = null;
  state.person = null;
  state.bin = null;
  renderAll();
}

function removeSound(id) {
  const sound = soundOf(id);
  if (!sound) return;
  checkpoint();
  state.project.sounds = state.project.sounds.filter((s) => s.id !== id);
  state.sound = null;
  save();
  renderAll();
  toast("Removed " + sound.name + " from the timeline.", false, undoAction);
}

function bindSound(block, id) {
  block.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    event.stopPropagation();
    selectSound(id);
  });
  block.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    const edge = event.target.classList.contains("l") ? "l" : event.target.classList.contains("r") ? "r" : null;
    selectSound(id);
    const first = soundOf(id);
    if (!first) return;
    const origin = { x: event.clientX, start: first.start, offset: first.offset || 0, frames: first.frames };
    const points = snapPoints(null);
    let changed = false;
    const move = (ev) => {
      // A save can swap the project for the server's copy, so the sound is found again on every move.
      const sound = soundOf(id);
      const live = document.querySelector('.wave.sound[data-sound="' + CSS.escape(id) + '"]');
      if (!sound || !live) return;
      const delta = Math.round((ev.clientX - origin.x) / state.zoom);
      if (delta !== 0 && !changed) {
        checkpoint();
        changed = true;
      }
      if (edge === "l") {
        // The left end trims the start of the sound: it cannot go before the file begins, or before frame 0.
        const shift = Math.max(-Math.round(origin.offset * FPS), -origin.start, Math.min(delta, origin.frames - 6));
        sound.start = origin.start + shift;
        sound.offset = Math.max(0, origin.offset + shift / FPS);
        sound.frames = origin.frames - shift;
      } else if (edge === "r") {
        const room = Math.max(6, Math.round((sound.seconds - sound.offset) * FPS));
        sound.frames = Math.min(room, Math.max(6, origin.frames + delta));
      } else {
        const wanted = Math.max(0, origin.start + delta);
        const startSnap = snap(wanted, points);
        const endSnap = snap(wanted + sound.frames, points) - sound.frames;
        sound.start = Math.max(0, startSnap !== wanted ? startSnap : endSnap);
        // Dragged up or down onto another lane, it moves there.
        const over = document.elementFromPoint(ev.clientX, ev.clientY)?.closest?.(".lane.extra");
        if (over && Number(over.dataset.lane) !== (sound.lane || 0)) {
          sound.lane = Number(over.dataset.lane);
          over.append(live);
        }
      }
      live.style.left = px(sound.start) + "px";
      live.style.width = Math.max(10, px(sound.frames)) + "px";
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      if (changed) save();
      renderTimeline();
      renderInspector();
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  });
}

// Lays a sound file under the cut at the playhead, on the first lane where nothing is in its way.
async function addSound() {
  const file = await pickFile($("soundFile"));
  if (!file) return;
  toast("Adding " + file.name + ".");
  try {
    const form = new FormData();
    form.append("file", file);
    const asset = await api("/project/" + state.folder + "/asset?sound=1", { method: "POST", body: form });
    const start = Math.max(0, Math.round(state.playhead));
    const frames = Math.max(1, Math.round(asset.seconds * FPS));
    const sounds = state.project.sounds || [];
    const clear = (lane) => !sounds.some((s) => (s.lane || 0) === lane && s.start < start + frames && start < s.start + s.frames);
    const lane = [...Array(SOUND_LANES).keys()].find(clear) ?? 0;
    checkpoint();
    const sound = { id: uid("s"), name: file.name.replace(/\.[^.]+$/, "").slice(0, 80) || "Sound", file: asset.file, subfolder: asset.subfolder,
      seconds: asset.seconds, start, offset: 0, frames, gain: 1, lane };
    state.project.sounds = [...sounds, sound];
    save();
    selectSound(sound.id);
    toast(sound.name + " is on the timeline, on Sound " + (lane + 2) + ". Drag it to move it, and its ends to trim it.");
  } catch (error) {
    toast(error.message, true);
  }
}

function renderSoundPanel(panel, sound) {
  const live = () => soundOf(sound.id) || {};
  const head = document.createElement("div");
  head.className = "inspector-head";
  const name = document.createElement("input");
  name.className = "name-input";
  name.value = sound.name;
  name.setAttribute("aria-label", "Sound name");
  name.dataset.key = "name";
  name.addEventListener("focus", () => checkpoint(), { once: true });
  name.addEventListener("input", () => {
    live().name = name.value;
    save();
    renderTimeline();
  });
  const row = document.createElement("div");
  row.className = "head-row";
  row.append(name, trashButton("Remove this sound from the timeline", () => removeSound(sound.id)));
  const line = document.createElement("p");
  line.className = "status";
  line.textContent = "Extra sound, " + (sound.frames / FPS).toFixed(1) + " s of " + sound.seconds.toFixed(1) + " s, from " + timecode(sound.start);
  head.append(row, line);
  const volume = document.createElement("label");
  volume.className = "field";
  const volumeTitle = document.createElement("span");
  volumeTitle.textContent = "Volume";
  volume.append(volumeTitle);
  const lanes = [...Array(SOUND_LANES).keys()].map((n) => [String(n), "Sound " + (n + 2)]);
  const lane = choice("Lane", lanes, String(sound.lane || 0), (value) => {
    live().lane = Number(value);
    renderTimeline();
  });
  const hint = document.createElement("p");
  hint.className = "hint";
  hint.textContent = "Drag it on the timeline to move it, up or down to change its lane, and drag its ends to trim it. The export mixes it under the clips' own sound and stops it where the cut ends. Above 1 it is louder in the export; the preview tops out at 1.";
  panel.replaceChildren(head, section("Sound", [
    volume,
    strengthSlider("Sound volume", sound.name + " volume", typeof sound.gain === "number" ? sound.gain : 1, [0, 2], 1, (v) => (live().gain = v)),
    lane,
    hint,
  ]));
}

// Keeps each extra sound in step with the playhead while the cut plays, and silent otherwise.
function syncSounds() {
  const sounds = state.project?.sounds || [];
  const known = new Set(sounds.map((s) => s.id));
  for (const [id, player] of soundPlayers) {
    if (known.has(id)) continue;
    player.pause();
    soundPlayers.delete(id);
  }
  const running = state.playing && !viewing();
  for (const sound of sounds) {
    let player = soundPlayers.get(sound.id);
    if (!player) {
      player = new Audio(viewUrl(sound.file, sound.subfolder));
      player.preload = "auto";
      soundPlayers.set(sound.id, player);
    }
    const at = (state.playhead - sound.start) / FPS;
    if (!running || at < 0 || at >= sound.frames / FPS || !(sound.gain > 0)) {
      if (!player.paused) player.pause();
      continue;
    }
    player.volume = Math.min(1, sound.gain);
    const wanted = (sound.offset || 0) + at;
    if (player.paused) {
      player.currentTime = wanted;
      player.play().catch(() => {});
    } else if (Math.abs(player.currentTime - wanted) > 0.2) {
      player.currentTime = wanted;
    }
  }
}

/* Timeline height */

function setTimelineHeight(value, keep = false) {
  // The viewer and the panels above keep enough room to work in, however short the window.
  const height = Math.round(Math.max(190, Math.min(window.innerHeight * 0.72, window.innerHeight - 394, value)));
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
  ruler.replaceChildren(...ticks, ...noteMarks());
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
    const room = take && takeRoom(item);
    if (room?.before) block.classList.add("more-before");
    if (room?.after) block.classList.add("more-after");
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
  for (const cut of touchingCuts()) {
    const found = cutScores.get(cut.key);
    if (!found || found.score === null || found.score >= CUT_FINE) continue;
    const mark = document.createElement("button");
    mark.className = "cut-score" + (found.score < CUT_BAD ? " bad" : "");
    mark.style.left = px(cut.after.start) + "px";
    mark.textContent = found.score;
    const names = clipOf(cut.before.clip).name + " to " + clipOf(cut.after.clip).name;
    const why = found.kind === "jump"
      ? "Jump cut: " + (found.notes.join(", ") || "the same shot, not continuous") + ". Extend the first clip so the second continues from its last frame, or change the framing."
      : found.notes.join(", ") || found.kind;
    mark.title = "Cut score " + found.score + " of 100, " + names + ". " + why;
    mark.setAttribute("aria-label", mark.title);
    mark.addEventListener("pointerdown", (event) => event.stopPropagation());
    mark.addEventListener("click", () => {
      state.clip = cut.before.clip;
      state.item = cut.before.id;
      state.gap = null;
      seek(cut.after.start - 1);
      renderAll();
    });
    blocks.push(mark);
  }
  for (const row of shownTakes()) {
    // An extension whose picture has moved away from where its shot began.
    const found = fineScores.get(row.key);
    if (!found || !found.drift || !found.drift.drifted) continue;
    const mark = document.createElement("button");
    mark.className = "soft-mark drift-mark";
    mark.style.left = px(row.item.start + (itemEnd(row.item) - row.item.start) / 2) + "px";
    mark.textContent = "Drift";
    const amount = Math.round(Math.abs(found.drift.change) * 100);
    const first = state.project.timeline.find((item) => item.id === found.drift.root);
    mark.title = clipOf(row.item.clip).name + " has drifted from " + (first ? clipOf(first.clip).name : "the take its shot began with")
      + ": its picture is " + amount + "% " + (found.drift.change > 0 ? "harsher, which shows as shine and deeper lines on a face" : "softer")
      + ". Each extension adds to it on this model. Render these lines as a clip of their own from a first frame, or render the shot on a model that holds it.";
    mark.setAttribute("aria-label", mark.title);
    mark.addEventListener("pointerdown", (event) => event.stopPropagation());
    mark.addEventListener("click", () => {
      state.clip = row.item.clip;
      state.item = row.item.id;
      state.gap = null;
      renderAll();
    });
    blocks.push(mark);
  }
  for (const row of shownTakes()) {
    const found = fineScores.get(row.key);
    if (!found || !found.soft) continue;
    const mark = document.createElement("button");
    mark.className = "soft-mark";
    mark.style.left = px(row.item.start + (itemEnd(row.item) - row.item.start) / 2) + "px";
    mark.textContent = "Soft";
    const draft = row.take.quality === "draft";
    mark.title = clipOf(row.item.clip).name + " is soft next to the rest of the cut: " + Math.round(found.part * 100) + "% of its fine detail. "
      + (draft ? "It is a draft among finals: render a final." : "Its picture is smaller than the cut's, or blurred.");
    mark.setAttribute("aria-label", mark.title);
    mark.addEventListener("pointerdown", (event) => event.stopPropagation());
    mark.addEventListener("click", () => {
      state.clip = row.item.clip;
      state.item = row.item.id;
      state.gap = null;
      seek(row.item.start);
      renderAll();
    });
    blocks.push(mark);
  }
  for (const row of cutExtras.effects) {
    const effect = effectOf(row.effect);
    if (!effect || activeTake(effect)?.id !== row.take || row.score === null || row.score >= CUT_FINE) continue;
    const at = row.edge === "start" ? effect.start : effect.start + effect.frames;
    const why = (row.edge === "start" ? "Cut into " : "Cut out of ") + effect.name + ". "
      + (row.kind === "jump" ? "Jump cut: " + (row.notes.join(", ") || "the same shot, not continuous") + ". Make the VFX clip start and end on the picture under it, or move it to a cut."
        : row.notes.join(", ") || row.kind);
    blocks.push(cutMark(at, row.score, "Cut score " + row.score + " of 100. " + why, () => selectEffect(effect.id)));
  }
  for (const row of cutExtras.inside) {
    const item = state.project.timeline.find((i) => i.id === row.item);
    if (!item || activeTake(clipOf(item.clip))?.id !== row.take || row.frame < item.in || row.frame >= item.out) continue;
    const at = item.start + Math.round((row.frame - item.in) / speedOf(item));
    if (row.frames) {
      blocks.push(...stillMarks(item, row, at));
      continue;
    }
    const why = clipOf(item.clip).name + " changes picture inside the take: " + row.notes.join(", ")
      + ". Trim the clip to end before it, or render it again with another seed.";
    blocks.push(cutMark(at, row.score, why, () => {
      state.clip = item.clip;
      state.item = item.id;
      state.gap = null;
      seek(Math.max(item.start, at - 1));
      renderAll();
    }));
  }
  refreshCutScores();
  refreshFineScores();
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
  renderEffects();
  showRange();
  renderSounds();
  if (focused) $("laneV").querySelector('[data-item="' + CSS.escape(focused) + '"]')?.focus();
  placePlayhead();
  updateTools();
  renderChecks();
}

// Cut scores: the server measures each cut between two rendered clips that touch, from their pictures alone.
// Only cuts that need a look are marked: under CUT_FINE in amber, under CUT_BAD in red.
const CUT_FINE = 70;
const CUT_BAD = 40;
const cutScores = new Map();
let cutsAsked = "";
let cutsTimer = null;

// What else the server's cut scores hold: the cuts into and out of VFX clips, and takes that change picture inside.
const cutExtras = { effects: [], inside: [] };

function cutMark(frame, score, title, act) {
  const mark = document.createElement("button");
  mark.className = "cut-score" + (score < CUT_BAD ? " bad" : "");
  mark.style.left = px(frame) + "px";
  mark.textContent = score;
  mark.title = title;
  mark.setAttribute("aria-label", title);
  mark.addEventListener("pointerdown", (event) => event.stopPropagation());
  mark.addEventListener("click", act);
  return mark;
}

// Checks: every finding the timeline marks, in one list for the header. Each has a severity (bad: fix it; warn: look
// at it), the badge its mark wears on the timeline, what it is, where, and what a click on it does.
const RATING_LOOK = 70;

function findings() {
  const found = [];
  const jumpTo = (item, frame) => () => {
    state.clip = item.clip;
    state.item = item.id;
    state.gap = null;
    seek(frame);
    renderAll();
  };
  for (const cut of touchingCuts()) {
    const row = cutScores.get(cut.key);
    if (!row || row.score === null || row.score >= CUT_FINE) continue;
    found.push({ sev: row.score < CUT_BAD ? "bad" : "warn", badge: String(row.score), frame: cut.after.start,
      title: (row.kind === "jump" ? "Jump cut, " : "Cut, ") + clipOf(cut.before.clip).name + " to " + clipOf(cut.after.clip).name,
      detail: row.notes.join(", ") || row.kind, act: () => { jumpTo(cut.before, cut.after.start - 1)(); } });
  }
  for (const row of cutExtras.effects) {
    const effect = effectOf(row.effect);
    if (!effect || activeTake(effect)?.id !== row.take || row.score === null || row.score >= CUT_FINE) continue;
    const at = row.edge === "start" ? effect.start : effect.start + effect.frames;
    found.push({ sev: row.score < CUT_BAD ? "bad" : "warn", badge: String(row.score), frame: at,
      title: (row.edge === "start" ? "Cut into " : "Cut out of ") + effect.name, detail: row.notes.join(", ") || row.kind,
      act: () => selectEffect(effect.id) });
  }
  for (const row of cutExtras.inside) {
    const item = state.project.timeline.find((i) => i.id === row.item);
    if (!item || activeTake(clipOf(item.clip))?.id !== row.take || row.frame < item.in || row.frame >= item.out) continue;
    const at = item.start + Math.round((row.frame - item.in) / speedOf(item));
    const name = clipOf(item.clip).name;
    const still = { frozen: "Frozen", still: "Still" }[row.kind];
    const long = row.length ? (row.length / FPS).toFixed(row.length < FPS ? 2 : 1) + " s" : "";
    found.push({ sev: row.kind === "still" ? "warn" : "bad", badge: still || String(row.score), word: Boolean(still), frame: at,
      title: still ? name + (row.kind === "frozen" ? " stops on one frame" : " barely moves")
        : name + (row.kind === "fade" ? " fades inside its take" : " cuts inside its take"),
      detail: row.kind === "frozen" ? "Held for " + long + ", " + row.length + " frames. Trim before it or render another seed"
        : row.kind === "still" ? "For " + long + ". Fine if the shot should be still"
        : row.notes.join(", "), act: jumpTo(item, at) });
  }
  for (const row of shownTakes()) {
    const fine = fineScores.get(row.key);
    if (!fine) continue;
    const name = clipOf(row.item.clip).name;
    if (fine.drift && fine.drift.drifted) {
      found.push({ sev: "bad", badge: "Drift", word: true, frame: row.item.start, title: name + " has drifted",
        detail: Math.round(Math.abs(fine.drift.change) * 100) + "% " + (fine.drift.change > 0 ? "harsher" : "softer") + " than where its shot began",
        act: jumpTo(row.item, row.item.start) });
    }
    if (fine.soft) {
      found.push({ sev: "warn", badge: "Soft", word: true, frame: row.item.start, title: name + " is soft",
        detail: Math.round(fine.part * 100) + "% of the cut's detail" + (row.take.quality === "draft" ? ", a draft among finals" : ""),
        act: jumpTo(row.item, row.item.start) });
    }
  }
  for (const rating of state.project.ratings || []) {
    const mean = ratingMean(rating);
    if (mean === null || mean >= RATING_LOOK) continue;
    const low = (rating.rows || []).filter((r) => !r.error).sort((a, b) => a.part - b.part)[0];
    found.push({ sev: "warn", badge: mean + "%", frame: rating.from, title: "Rated " + timecode(rating.from) + " to " + timecode(rating.to),
      detail: (low ? "Lowest: " + low.title + ", " + Math.round(low.part * 100) + "%. " : "") + "Rated " + String(rating.at || "").replace("T", " ").slice(0, 16),
      act: () => { seek(rating.from); renderAll(); } });
  }
  for (const f of found) f.detail = f.detail.charAt(0).toUpperCase() + f.detail.slice(1);
  return found.sort((a, b) => (a.sev === b.sev ? a.frame - b.frame : a.sev === "bad" ? -1 : 1));
}

// Whether the cut's scores have answered for this project yet: until then the header says it is checking, not clear.
let checksFor = null;
const checked = () => checksFor === state.folder;

function renderChecks() {
  const chip = $("checks");
  if (!state.project || !state.project.timeline.some((item) => activeTake(clipOf(item.clip)))) {
    chip.hidden = true;
    return;
  }
  const found = findings();
  const bad = found.filter((f) => f.sev === "bad").length;
  const warn = found.length - bad;
  chip.hidden = false;
  chip.className = "checks-chip" + (bad ? " bad" : warn ? " warn" : found.length || !checked() ? "" : " clear");
  const parts = [];
  if (bad) parts.push(bad + " to fix");
  if (warn) parts.push(warn + " to check");
  chip.textContent = parts.length ? parts.join(", ") + (checked() ? "" : ", checking") : checked() ? "Checks clear" : "Checking";
  chip.title = found.length ? "What the checks found in this cut" : "Cuts, soft clips, drift and takes that stop moving are checked as clips render";
  $("checksSummary").textContent = found.length ? found.length + " found" : "";
  if (!$("checksPanel").matches(":popover-open")) return;
  $("checksList").replaceChildren(...(found.length ? found.map(checkRow) : [Object.assign(document.createElement("p"), {
    className: "checks-empty", textContent: checked() ? "Nothing to look at. Every rendered cut scores well and no take stops, drifts or goes soft." : "The cut is still being measured." })]));
}

function checkRow(f) {
  const row = document.createElement("button");
  row.className = "check-row";
  const badge = document.createElement("span");
  badge.className = "check-badge " + f.sev + (f.word ? " word" : "");
  badge.textContent = f.badge;
  const text = document.createElement("span");
  text.className = "check-text";
  const title = document.createElement("b");
  title.textContent = f.title;
  const detail = document.createElement("span");
  detail.textContent = f.detail;
  text.append(title, detail);
  const when = document.createElement("span");
  when.className = "check-time";
  when.textContent = timecode(f.frame);
  row.append(badge, text, when);
  row.addEventListener("click", () => {
    $("checksPanel").hidePopover();
    f.act();
  });
  return row;
}

// A stretch of a take where the picture stops: frozen (the same frame held) or still (almost nothing moves).
// A bar under the stretch shows how long it lasts, and a mark in its middle says what it is.
function stillMarks(item, row, at) {
  const shown = Math.max(1, Math.min(Math.round((row.frames || 1) / speedOf(item)), itemEnd(item) - at));
  const frozen = row.kind === "frozen";
  const seconds = (shown / FPS).toFixed(shown < FPS ? 2 : 1);
  const bar = document.createElement("div");
  bar.className = "still-span" + (frozen ? " frozen" : "");
  bar.style.left = px(at) + "px";
  bar.style.width = Math.max(2, px(shown)) + "px";
  const mark = document.createElement("button");
  mark.className = "soft-mark still-mark" + (frozen ? " frozen" : "");
  mark.style.left = px(at + shown / 2) + "px";
  mark.textContent = frozen ? "Frozen" : "Still";
  mark.title = clipOf(item.clip).name + (frozen
    ? ": the picture stops for " + seconds + " s, the same frame held. Trim the clip to end before it, or render it again with another seed."
    : ": almost nothing moves for " + seconds + " s. If it should move, render it again with another seed, or trim it shorter.");
  mark.setAttribute("aria-label", mark.title);
  mark.addEventListener("pointerdown", (event) => event.stopPropagation());
  mark.addEventListener("click", () => {
    state.clip = item.clip;
    state.item = item.id;
    state.gap = null;
    seek(at);
    renderAll();
  });
  return [bar, mark];
}

function touchingCuts() {
  const order = sorted();
  const found = [];
  for (let i = 0; i + 1 < order.length; i++) {
    const before = order[i];
    const after = order[i + 1];
    const takes = [activeTake(clipOf(before.clip)), activeTake(clipOf(after.clip))];
    if (itemEnd(before) !== after.start || !takes[0] || !takes[1]) continue;
    found.push({ before, after, key: [before.id, takes[0].id, before.out, after.id, takes[1].id, after.in].join("|") });
  }
  return found;
}

function refreshCutScores() {
  const wanted = touchingCuts();
  // Asked again whenever a cut, a shown take or a VFX clip changes: the answer covers all three.
  const signature = state.folder + "/" + wanted.map((c) => c.key).join(",") + "/" + shownTakes().map((c) => c.key).join(",") + "/"
    + (state.project.effects || []).map((e) => [e.id, activeTake(e)?.id, e.start, e.frames].join("|")).join(",");
  if (signature === cutsAsked) return;
  clearTimeout(cutsTimer);
  // Wait until the project is saved and still: the server scores what it has on disk.
  cutsTimer = setTimeout(async () => {
    if (unsaved()) return refreshCutScores();
    cutsAsked = signature;
    const folder = state.folder;
    try {
      const reply = await api("/project/" + encodeURIComponent(folder) + "/cuts");
      if (folder !== state.folder) return;
      const now = new Map(touchingCuts().map((c) => [c.before.id + ">" + c.after.id, c.key]));
      for (const cut of reply.cuts) {
        const key = now.get(cut.before + ">" + cut.after);
        if (key) cutScores.set(key, cut);
      }
      cutExtras.effects = reply.effects || [];
      cutExtras.inside = reply.inside || [];
      checksFor = folder;
      renderTimeline();
    } catch (_) {
      // An older server has no cut scores; the timeline works without them.
    }
  }, 900);
}

// Detail: the server measures how sharp each rendered clip is at the size of the cut. A clip far under the level of
// the rest gets a mark: a draft left among finals, or footage smaller than the cut. Whether a clip is soft depends on
// the whole cut, so every answer replaces what was known.
const fineScores = new Map();
let fineAsked = "";
let fineTimer = null;

function shownTakes() {
  const shape = (cutShape() || []).join("x");
  const found = [];
  for (const item of sorted()) {
    const take = activeTake(clipOf(item.clip));
    if (take) found.push({ item, take, key: [item.id, take.id, item.in, item.out, shape].join("|") });
  }
  return found;
}

function refreshFineScores() {
  const wanted = shownTakes();
  const signature = state.folder + "/" + wanted.map((c) => c.key).join(",");
  if (signature === fineAsked || wanted.length < 2) return;
  clearTimeout(fineTimer);
  fineTimer = setTimeout(async () => {
    if (unsaved()) return refreshFineScores();
    fineAsked = signature;
    const folder = state.folder;
    try {
      const reply = await api("/project/" + encodeURIComponent(folder) + "/detail");
      if (folder !== state.folder) return;
      const now = new Map(shownTakes().map((c) => [c.item.id, c.key]));
      fineScores.clear();
      for (const row of reply.clips) {
        const key = now.get(row.item);
        if (key) fineScores.set(key, row);
      }
      renderTimeline();
    } catch (_) {
      // An older server has no detail scores; the timeline works without them.
    }
  }, 1200);
}

/* VFX: clips drawn by code, on a lane above the picture */

const effectOf = (id) => (state.project?.effects || []).find((e) => e.id === id) || null;
const effectAt = (frame) => (state.project?.effects || []).find((e) => frame >= e.start && frame < e.start + e.frames) || null;
const fx = { runtime: null, key: "", source: null, params: "", meta: null, busy: false, wanted: null, error: null, rendering: null };

// What a take of a VFX clip was made from: the same list the server stamps on the take, so the two can be compared.
function effectStamp(effect) {
  const first = effect.start, last = effect.start + effect.frames;
  const below = sorted().filter((item) => item.start < last && first < itemEnd(item)).map((item) => {
    const take = activeTake(clipOf(item.clip));
    return [item.clip, take ? take.id : null, item.start, item.in, item.out, speedOf(item), item.transition?.type ?? null, item.transition?.frames ?? null];
  });
  return { rev: effect.rev || 0, start: effect.start, below };
}

function effectState(effect) {
  if (fx.rendering?.id === effect.id) return "busy";
  if (fx.error?.id === effect.id || effect.error) return "bad";
  const take = activeTake(effect);
  if (!take) return "todo";
  return JSON.stringify(take.stamp) === JSON.stringify(effectStamp(effect)) ? "" : "stale";
}

function touchEffect(effect) {
  effect.rev = (effect.rev || 0) + 1;      // what it draws changed: its take is out of date
  delete effect.error;
}

function selectEffect(id) {
  setSource(null);
  state.effect = id;
  state.clip = null;
  state.item = null;
  state.gap = null;
  state.person = null;
  state.bin = null;
  state.sound = null;
  renderAll();
}

function effectRoom(effect, start, frames) {
  // The lane holds one clip at a time: a move or a stretch stops at its neighbours.
  let low = 0, high = Infinity;
  for (const other of state.project.effects || []) {
    if (other.id === effect.id) continue;
    if (other.start + other.frames <= effect.start) low = Math.max(low, other.start + other.frames);
    else if (other.start >= effect.start + effect.frames) high = Math.min(high, other.start);
  }
  start = Math.max(low, start);
  frames = Math.max(6, Math.min(frames, high - start));
  if (start + frames > high) start = Math.max(low, high - frames);
  return [start, frames];
}

function addEffect() {
  checkpoint();
  const effects = state.project.effects || (state.project.effects = []);
  let start = Math.round(state.playhead);
  const frames = 3 * FPS;
  if (effects.some((e) => start < e.start + e.frames && e.start < start + frames)) start = Math.max(...effects.map((e) => e.start + e.frames));
  const effect = { id: "x" + Math.random().toString(16).slice(2, 10), name: "VFX " + (effects.length + 1), start, frames, source: EFFECT_STARTER,
    params: {}, rev: 1, takes: [], active: null };
  effects.push(effect);
  save();
  selectEffect(effect.id);
  toast("Added a VFX clip. It draws a title over the picture below: edit its code in the panel.");
}

function removeEffect(id) {
  const effect = effectOf(id);
  if (!effect) return;
  checkpoint();
  state.project.effects = state.project.effects.filter((e) => e.id !== id);
  state.effect = null;
  save();
  renderAll();
  toast("Removed " + effect.name + " from the VFX lane.", false, undoAction);
}

function renderEffects() {
  const blocks = [];
  for (const effect of state.project.effects || []) {
    const block = document.createElement("div");
    const status = effectState(effect);
    block.className = ("fx " + status + (effect.id === state.effect ? " on" : "")).trim();
    block.dataset.effect = effect.id;
    block.style.left = px(effect.start) + "px";
    block.style.width = Math.max(12, px(effect.frames)) + "px";
    block.tabIndex = 0;
    block.setAttribute("role", "button");
    const what = { todo: "not rendered", stale: "out of date", bad: "has an error", busy: "rendering", "": "rendered" }[status];
    block.setAttribute("aria-label", effect.name + ", VFX clip, " + what + ", at " + timecode(effect.start));
    block.title = effect.name + ": " + what;
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = effect.name;
    block.append(name);
    if (status === "busy") {
      // The render's progress, on the clip itself: a bar that fills, and how far along it is.
      const bar = document.createElement("div");
      bar.className = "fx-bar";
      bar.setAttribute("role", "progressbar");
      bar.setAttribute("aria-valuemin", "0");
      bar.setAttribute("aria-valuemax", "100");
      const note = document.createElement("div");
      note.className = "fx-note";
      block.append(bar, note);
    }
    for (const side of ["l", "r"]) {
      const edge = document.createElement("div");
      edge.className = "edge " + side;
      block.append(edge);
    }
    block.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      event.stopPropagation();
      selectEffect(effect.id);
    });
    block.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const edge = event.target.classList.contains("l") ? "l" : event.target.classList.contains("r") ? "r" : null;
      if (state.effect !== effect.id) selectEffect(effect.id);
      const live = effectOf(effect.id);
      const origin = { x: event.clientX, start: live.start, frames: live.frames };
      let changed = false;
      const move = (ev) => {
        const delta = Math.round((ev.clientX - origin.x) / state.zoom);
        if (delta !== 0 && !changed) {
          checkpoint();
          changed = true;
        }
        if (edge === "l") {
          const shift = Math.min(delta, origin.frames - 6);
          [live.start, live.frames] = effectRoom(live, Math.max(0, origin.start + shift), origin.frames - shift);
        } else if (edge === "r") {
          [live.start, live.frames] = effectRoom(live, origin.start, Math.max(6, origin.frames + delta));
        } else {
          [live.start, live.frames] = effectRoom(live, Math.max(0, origin.start + delta), origin.frames);
        }
        const el = document.querySelector('.fx[data-effect="' + live.id + '"]');
        if (el) {
          el.style.left = px(live.start) + "px";
          el.style.width = Math.max(12, px(live.frames)) + "px";
        }
      };
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        if (changed) {
          if (live.frames !== origin.frames) touchEffect(live);
          save();
          renderAll();
        }
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    });
    blocks.push(block);
  }
  $("laneX").replaceChildren(...blocks);
  showEffectProgress();
}

// How far a VFX render has got, wherever it shows: on the clip, on the Render button and in the page's title, so it
// can be read from another tab.
function showEffectProgress() {
  const job = fx.rendering;
  if (!job) return;
  const part = job.total ? job.done / job.total : 0;
  const percent = Math.floor(part * 100);
  const spent = (performance.now() - job.began) / 1000;
  const left = job.done > 24 ? (spent / job.done) * (job.total - job.done) : null;
  const time = left === null ? "" : left >= 90 ? ", about " + Math.round(left / 60) + " min left" : ", about " + Math.max(1, Math.round(left)) + " s left";
  const block = document.querySelector('.fx[data-effect="' + job.id + '"]');
  if (block) {
    const bar = block.querySelector(".fx-bar");
    if (bar) {
      bar.style.width = percent + "%";
      bar.setAttribute("aria-valuenow", String(percent));
    }
    const note = block.querySelector(".fx-note");
    if (note) note.textContent = percent + "%" + time;
    block.title = job.name + ": rendering, frame " + job.done + " of " + job.total + time;
  }
  const label = document.querySelector("#inspBody .row button.primary");
  if (label && state.effect === job.id) label.textContent = "Rendering " + job.done + " of " + job.total + time;
  document.title = percent + "% " + job.name + " · " + job.title;
}

// A VFX render runs in this page. Closing or reloading it would throw the render away, so the browser asks first.
window.addEventListener("beforeunload", (event) => {
  if (!fx.rendering) return;
  event.preventDefault();
  event.returnValue = "";
});

// A VFX clip's named pictures, read once each and handed to its code as fresh bitmaps (a bitmap given away is gone).
const inputBlobs = new Map();
async function effectInputs(effect) {
  const out = {};
  for (const [name, asset] of Object.entries(effect.inputs || {})) {
    const url = viewUrl(asset.file, asset.subfolder);
    if (!inputBlobs.has(url)) {
      inputBlobs.set(url, fetch(url).then((reply) => {
        if (!reply.ok) throw new Error("The input " + name + " (" + asset.file + ") could not be read.");
        return reply.blob();
      }));
    }
    try {
      out[name] = await createImageBitmap(await inputBlobs.get(url));
    } catch (error) {
      inputBlobs.delete(url);
      throw error;
    }
  }
  return out;
}

// A clip's data can be large, so the preview tells one version from the next by a mark, not by comparing it whole.
const dataMarks = new WeakMap();
let dataMarkNext = 1;
function dataMark(effect) {
  if (!effect.data || typeof effect.data !== "object") return String(effect.data ?? "");
  if (!dataMarks.has(effect.data)) dataMarks.set(effect.data, dataMarkNext++);
  return "d" + dataMarks.get(effect.data);
}

function previewSize() {
  const shape = cutShape() || state.sizes?.[state.project.format || "16:9"]?.final?.[0]?.out || [1920, 1080];
  const w = Math.min(960, shape[0]);
  return [2 * Math.round(w / 2), 2 * Math.round((w * shape[1]) / shape[0] / 2)];
}

function renderSize() {
  const shape = cutShape() || state.sizes?.[state.project.format || "16:9"]?.final?.[0]?.out || [1920, 1080];
  return [2 * Math.round(shape[0] / 2), 2 * Math.round(shape[1] / 2)];
}

// The picture under the playhead, as the viewer is showing it.
async function belowNow(w, h) {
  if (!front.classList.contains("show")) return null;
  if (front.seeking || front.readyState < 2) {
    // A long clip takes a moment to land on a frame it was just sent to: wait for it, but not for ever.
    await new Promise((resolve) => {
      const done = () => {
        front.removeEventListener("seeked", done);
        front.removeEventListener("canplay", done);
        resolve();
      };
      front.addEventListener("seeked", done);
      front.addEventListener("canplay", done);
      setTimeout(done, 2000);
    });
  }
  if (front.readyState < 2 || !front.videoWidth) return null;
  try {
    return await createImageBitmap(front, { resizeWidth: w, resizeHeight: h, resizeQuality: "medium" });
  } catch (_) {
    return null;
  }
}

function showEffectError() {
  const box = document.getElementById("fxError");
  if (!box) return;
  const error = fx.error && fx.error.id === state.effect ? fx.error : null;
  box.hidden = !error;
  box.textContent = error ? (error.line ? "Line " + error.line + ": " : "") + error.message : "";
}

// Draws the VFX clip under the playhead into the viewer, live, from its code. Calls made while one frame is being
// drawn collapse into the latest, so scrubbing and playback never queue up.
async function previewEffect() {
  const canvas = $("vfx");
  const effect = viewing() ? null : effectAt(Math.floor(state.playhead));
  if (!effect) {
    canvas.hidden = true;
    fx.wanted = null;
    return;
  }
  fx.wanted = { id: effect.id, frame: Math.floor(state.playhead) };
  if (fx.busy) return;
  fx.busy = true;
  try {
    while (fx.wanted) {
      const want = fx.wanted;
      fx.wanted = null;
      const e = effectOf(want.id);
      if (!e) break;
      const [w, h] = previewSize();
      const key = [e.id, w, h, e.frames, JSON.stringify(e.inputs || {}), dataMark(e)].join("|");
      const params = JSON.stringify(e.params || {});
      try {
        if (fx.key !== key || fx.source !== e.source) {
          fx.runtime?.dispose();
          fx.runtime = new VfxRuntime();
          fx.key = "";
          fx.meta = await fx.runtime.load({ source: e.source, width: w, height: h, fps: FPS, frames: e.frames, params: e.params || {},
            inputs: await effectInputs(e), data: e.data ?? null });
          fx.key = key;
          fx.source = e.source;
          fx.params = params;
          if (fx.error?.id === e.id) fx.error = null;
          renderEffectControls();
        } else if (fx.params !== params) {
          await fx.runtime.setParams(e.params || {});
          fx.params = params;
        }
        const bitmap = await fx.runtime.frame(want.frame - e.start, await belowNow(w, h));
        if (canvas.width !== w || canvas.height !== h) {
          canvas.width = w;
          canvas.height = h;
        }
        canvas.getContext("2d").drawImage(bitmap, 0, 0);
        bitmap.close();
        canvas.hidden = false;
        if (fx.error?.id === e.id) fx.error = null;
      } catch (error) {
        fx.error = { id: e.id, message: error.message || String(error), line: error.line ?? null };
        fx.source = null;          // try again when the code changes
        fx.key = "";
        canvas.hidden = true;
      }
      showEffectError();
    }
  } finally {
    fx.busy = false;
  }
}

function renderEffectControls() {
  const box = document.getElementById("fxControls");
  const effect = effectOf(state.effect);
  if (!box || !effect) return;
  const controls = fx.meta?.controls && fx.source === effect.source ? fx.meta.controls : {};
  const rows = [];
  for (const [key, c] of Object.entries(controls)) {
    const row = document.createElement("label");
    row.className = "fx-control";
    const label = document.createElement("span");
    label.textContent = c.label || key;
    const current = key in (effect.params || {}) ? effect.params[key] : c.default;
    const set = (value) => {
      if (!fx.editing) {
        checkpoint();
        fx.editing = true;
        setTimeout(() => { fx.editing = false; }, 800);
      }
      effect.params = { ...(effect.params || {}), [key]: value };
      touchEffect(effect);
      save();
      previewEffect();
    };
    row.append(label);
    if (c.type === "number" || c.type === "int") {
      const input = document.createElement("input");
      input.type = "range";
      input.min = c.min ?? 0;
      input.max = c.max ?? 1;
      input.step = c.step ?? (c.type === "int" ? 1 : 0.01);
      input.value = current;
      const out = document.createElement("output");
      out.textContent = String(current);
      input.addEventListener("input", () => { out.textContent = input.value; set(Number(input.value)); });
      row.append(input, out);
    } else if (c.type === "bool") {
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = Boolean(current);
      input.addEventListener("change", () => set(input.checked));
      row.append(input);
    } else if (c.type === "enum") {
      const input = document.createElement("select");
      for (const option of c.options || []) {
        const o = document.createElement("option");
        o.value = o.textContent = option;
        o.selected = option === current;
        input.append(o);
      }
      input.addEventListener("change", () => set(input.value));
      row.append(input);
    } else if (c.type === "color") {
      const input = document.createElement("input");
      input.type = "color";
      input.value = current || "#ffffff";
      input.addEventListener("input", () => set(input.value));
      row.append(input);
    } else {
      const input = document.createElement("input");
      input.type = "text";
      input.value = current ?? "";
      input.addEventListener("input", () => set(input.value));
      row.append(input);
    }
    rows.push(row);
  }
  if (!rows.length) {
    const none = document.createElement("p");
    none.className = "hint";
    none.textContent = "The code declares no controls. Export a meta object with controls to get sliders here.";
    rows.push(none);
  }
  box.replaceChildren(...rows);
}

function renderEffect(panel, effect) {
  const status = effectState(effect);
  const state_ = document.createElement("p");
  state_.className = "fx-state";
  state_.id = "fxState";
  const seconds = (effect.frames / FPS).toFixed(2).replace(/\.?0+$/, "");
  state_.textContent = "On the VFX lane at " + timecode(effect.start) + ", " + seconds + " s. "
    + ({ todo: "Not rendered: the export leaves it out until it is.", stale: "Out of date: it or the picture below changed since its render.",
      bad: "Its code has an error.", busy: "Rendering.", "": "Rendered." }[status]);

  const code = document.createElement("textarea");
  code.className = "code";
  code.spellcheck = false;
  code.value = effect.source;
  code.setAttribute("aria-label", "The code that draws this clip");
  code.addEventListener("focus", () => checkpoint(), { once: true });
  let wait = null;
  code.addEventListener("input", () => {
    effect.source = code.value;
    touchEffect(effect);
    save();
    clearTimeout(wait);
    wait = setTimeout(previewEffect, 350);
  });
  code.addEventListener("keydown", (event) => {
    if (event.key !== "Tab") return;
    event.preventDefault();
    code.setRangeText("  ", code.selectionStart, code.selectionEnd, "end");
    code.dispatchEvent(new Event("input"));
  });
  const error = document.createElement("p");
  error.className = "fx-error";
  error.id = "fxError";
  error.hidden = true;

  const controls = document.createElement("div");
  controls.className = "fx-controls";
  controls.id = "fxControls";

  const actions = document.createElement("div");
  actions.className = "row";
  const go = document.createElement("button");
  go.className = "primary";
  go.textContent = fx.rendering?.id === effect.id ? "Rendering " + fx.rendering.done + " of " + fx.rendering.total : "Render";
  go.disabled = Boolean(fx.rendering);
  go.title = "Draw every frame over the picture below and keep the result as this clip's take";
  go.addEventListener("click", () => renderEffectTake(effect.id));
  const drop = document.createElement("button");
  drop.className = "quiet";
  drop.textContent = "Remove";
  drop.addEventListener("click", () => removeEffect(effect.id));
  actions.append(go, drop);

  panel.replaceChildren(
    section("VFX clip", [field("Name", effect.name, (v) => { effect.name = v || "VFX"; renderEffects(); }), state_]),
    section("Controls", [controls]),
    ...(Object.keys(effect.inputs || {}).length ? [section("Pictures it can draw", [Object.assign(document.createElement("p"), {
      className: "hint", textContent: Object.entries(effect.inputs).map(([name, asset]) => "env.inputs." + name + ": " + asset.file).join("\n"),
      style: "white-space: pre-wrap" })])] : []),
    section("Code", [code, error]),
    actions,
  );
  renderEffectControls();
  showEffectError();
}

// One exact frame of the picture lane, for a render: the clip under that frame of the timeline, at that frame.
const fxSource = document.createElement("video");
fxSource.muted = true;
fxSource.preload = "auto";
fxSource.playsInline = true;

async function exactBelow(frame, w, h, work) {
  const item = leading(frame);
  const url = item && frame < itemEnd(item) ? sourceOf(item) : null;
  if (!url) return null;
  if (fxSource.dataset.url !== url) {
    fxSource.dataset.url = url;
    fxSource.src = url;
    await new Promise((resolve, reject) => {
      fxSource.addEventListener("loadeddata", resolve, { once: true });
      fxSource.addEventListener("error", () => reject(new Error("could not read the clip below")), { once: true });
    });
  }
  const seconds = sourceSeconds(item, frame) + 0.5 / FPS;      // the middle of the frame, clear of rounding
  if (Math.abs(fxSource.currentTime - seconds) > 1e-4) {
    await new Promise((resolve) => {
      fxSource.addEventListener("seeked", resolve, { once: true });
      fxSource.currentTime = seconds;
    });
  }
  // Cover the frame, as the viewer and the export do.
  const scale = Math.max(w / fxSource.videoWidth, h / fxSource.videoHeight);
  const dw = fxSource.videoWidth * scale, dh = fxSource.videoHeight * scale;
  work.drawImage(fxSource, (w - dw) / 2, (h - dh) / 2, dw, dh);
  return createImageBitmap(work.canvas);
}

// Renders a VFX clip: every frame drawn by its code over the exact picture below, sent to the server, made into a take.
async function renderEffectTake(id, claimed = null) {
  const effect = effectOf(id);
  if (!effect || fx.rendering) return;
  const folder = state.folder;
  fx.rendering = { id, done: 0, total: effect.frames, name: effect.name, began: performance.now(), title: document.title };
  renderAll();
  const runtime = new VfxRuntime();
  const path = "/project/" + encodeURIComponent(folder) + "/effect/" + encodeURIComponent(id);
  try {
    for (let tries = 0; unsaved() && tries < 60; tries++) await new Promise((r) => setTimeout(r, 150));
    if (unsaved()) throw new Error("The project is still saving. Try again in a moment.");
    const [w, h] = renderSize();
    await runtime.load({ source: effect.source, width: w, height: h, fps: FPS, frames: effect.frames, params: effect.params || {},
      inputs: await effectInputs(effect), data: effect.data ?? null });
    const work = Object.assign(document.createElement("canvas"), { width: w, height: h }).getContext("2d");
    const out = Object.assign(document.createElement("canvas"), { width: w, height: h }).getContext("2d");
    const job = claimed || "j" + Math.random().toString(16).slice(2, 12);
    let batch = [];
    const send = async () => {
      if (!batch.length) return;
      const response = await fetch("/h3take/api" + path + "/frames?job=" + job, { method: "POST", body: new Blob(batch) });
      if (!response.ok) throw new Error((await response.text()) || "The server refused the frames.");
      batch = [];
    };
    for (let k = 0; k < effect.frames; k++) {
      const bitmap = await runtime.frame(k, await exactBelow(effect.start + k, w, h, work));
      out.drawImage(bitmap, 0, 0, w, h);
      bitmap.close();
      const blob = await new Promise((resolve) => out.canvas.toBlob(resolve, "image/jpeg", 0.93));
      const size = new DataView(new ArrayBuffer(4));
      size.setUint32(0, blob.size);
      batch.push(size.buffer, blob);
      if (batch.length >= 24) await send();
      fx.rendering.done = k + 1;
      showEffectProgress();
    }
    await send();
    const result = await postJson(path + "/take", { job });
    if (folder === state.folder && !unsaved()) adopt(result.project);
    toast(effect.name + " is rendered.");
  } catch (error) {
    fx.error = { id, message: error.message || String(error), line: error.line ?? null };
    toast("The VFX render stopped: " + (error.message || error), true);
    if (claimed) {
      // An agent asked for this render: tell it why it failed, on the clip.
      const text = (error.line ? "Line " + error.line + ": " : "") + (error.message || String(error));
      postJson(path + "/take", { error: text }).then((result) => {
        if (folder === state.folder && !unsaved()) adopt(result.project);
      }).catch(() => {});
    }
  } finally {
    runtime.dispose();
    document.title = fx.rendering.title;
    fx.rendering = null;
    renderAll();
  }
}

// An agent asks for a VFX render with the render_effect operation. The code runs in a browser, so this editor looks
// for such requests, claims one so no other open editor does it too, and renders it.
async function serveEffectRequests() {
  if (!state.project || fx.rendering || unsaved()) return;
  const waiting = (state.project.effects || []).find((e) => e.queued === true);
  if (!waiting) return;
  const job = "j" + Math.random().toString(16).slice(2, 12);
  const folder = state.folder;
  try {
    const reply = await postJson("/project/" + encodeURIComponent(folder) + "/effect/" + encodeURIComponent(waiting.id) + "/take", { claim: job });
    if (!reply.claimed || folder !== state.folder) {
      waiting.queued = { job: "elsewhere" };      // another editor has it; the next project update says how it went
      return;
    }
    waiting.queued = { job };
    await renderEffectTake(waiting.id, job);
  } catch (_) {
    // The server was not reachable just now; the next look tries again.
  }
}
setInterval(serveEffectRequests, 2000);

const EFFECT_STARTER = `// A VFX clip is a module that draws each frame. frame(env, k) must depend only on k.
// env: canvas, width, height, fps, frames, params, THREE, below (the picture under this frame, or null)
export const meta = {
  controls: {
    text: { type: 'text', default: 'Title' },
    size: { type: 'number', min: 0.03, max: 0.3, step: 0.005, default: 0.09, label: 'Size' },
    color: { type: 'color', default: '#ece8df' },
  },
};

export function frame(env, k) {
  const c = env.canvas.getContext('2d');
  const { width: w, height: h } = env;
  c.clearRect(0, 0, w, h);
  if (env.below) c.drawImage(env.below, 0, 0, w, h);
  else { c.fillStyle = '#1b1e23'; c.fillRect(0, 0, w, h); }
  const t = Math.min(1, k / 12);                       // half a second in
  const ease = 1 - Math.pow(1 - t, 3);
  c.font = '700 ' + Math.round(h * env.params.size) + 'px "Segoe UI", system-ui, sans-serif';
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  c.globalAlpha = ease;
  c.shadowColor = 'rgba(0,0,0,0.6)';
  c.shadowBlur = h * 0.02;
  c.fillStyle = env.params.color;
  c.fillText(env.params.text, w / 2, h * 0.5 + (1 - ease) * h * 0.04);
  c.globalAlpha = 1;
  c.shadowBlur = 0;
}
`;

/* Rating a range of the cut with Gemini Decision */

// The range marked on the timeline, in frames: I and O at the playhead, or a Shift-drag on the ruler.
function setRange(a, b) {
  const end = sceneEnd();
  if (a === null || b === null) state.range = null;
  else {
    const first = Math.max(0, Math.min(Math.round(a), Math.round(b))), last = Math.min(end || Infinity, Math.max(Math.round(a), Math.round(b)));
    state.range = last > first ? { a: first, b: last } : null;
  }
  showRange();
}

function showRange() {
  const band = $("range");
  band.hidden = !state.range;
  if (!state.range) return;
  band.style.left = px(state.range.a) + "px";
  band.style.width = Math.max(2, px(state.range.b - state.range.a)) + "px";
}

// What Rate will look at: the marked range, else the selected clip on the timeline, else the cut from its start.
const RATE_MOST = 180 * FPS;
function rateSpan() {
  if (state.range) return { a: state.range.a, b: state.range.b, what: "the marked range" };
  const item = state.project.timeline.find((i) => i.id === state.item);
  if (item) return { a: item.start, b: itemEnd(item), what: clipOf(item.clip).name };
  return { a: 0, b: Math.min(sceneEnd(), RATE_MOST), what: sceneEnd() > RATE_MOST ? "the first three minutes of the cut" : "the whole cut" };
}

const rater = { busy: false, key: null };

function raterKey() {
  const box = $("raterKey");
  const set = rater.key?.set;
  const note = document.createElement("span");
  note.textContent = set ? "A Gemini API key is set" + (rater.key.source === "environment" ? " in ComfyUI's environment." : ", saved with this ComfyUI.")
    : "Rating needs your Gemini API key. It is saved with this ComfyUI, never in a project.";
  const input = document.createElement("input");
  input.type = "password";
  input.autocomplete = "off";
  input.placeholder = set ? "Paste another key to replace it" : "Paste your Gemini API key";
  input.setAttribute("aria-label", "Gemini API key");
  const keep = document.createElement("button");
  keep.className = "quiet";
  keep.textContent = set ? "Replace" : "Save key";
  keep.addEventListener("click", async () => {
    if (!input.value.trim()) return;
    try {
      rater.key = (await postJson("/settings", { gemini_key: input.value.trim() }, "PUT")).gemini;
      $("raterResult").textContent = "The key is saved.";
    } catch (error) {
      $("raterResult").textContent = error.message;
    }
    raterKey();
  });
  box.replaceChildren(note, input, keep);
  $("raterGo").disabled = rater.busy || !rater.key?.set;
}

function raterRows(rating) {
  const rows = (rating?.rows || []).map((row) => {
    const box = document.createElement("div");
    box.className = "rater-row";
    const what = document.createElement("div");
    what.className = "what";
    what.textContent = row.title;
    const meter = document.createElement("div");
    meter.className = "meter";
    const num = document.createElement("div");
    num.className = "num";
    const level = document.createElement("div");
    level.className = "level";
    if (row.error) {
      num.textContent = "–";
      level.textContent = row.error;
    } else {
      const fill = document.createElement("i");
      fill.style.width = Math.round(row.part * 100) + "%";
      fill.className = row.part < 0.45 ? "low" : row.part < 0.7 ? "mid" : "";
      meter.append(fill);
      meter.setAttribute("role", "img");
      meter.setAttribute("aria-label", row.title + ": " + (row.of === 1 ? Math.round(row.part * 100) + "% likely" : row.score.toFixed(1) + " of " + row.of));
      // A check has two ends, so it reads as how likely the good one is. A scale reads as a level.
      num.textContent = row.of === 1 ? Math.round(row.part * 100) + "%" : row.score.toFixed(1) + " / " + row.of;
      level.textContent = row.level + (row.of === 1 || row.confidence === null || row.confidence === undefined ? "" : " · " + Math.round(row.confidence * 100) + "% sure");
    }
    box.append(what, meter, num, level);
    return box;
  });
  $("raterRows").replaceChildren(...rows);
}

const ratingMean = (rating) => {
  const parts = (rating.rows || []).filter((r) => !r.error).map((r) => r.part);
  return parts.length ? Math.round((100 * parts.reduce((a, b) => a + b, 0)) / parts.length) : null;
};

function raterPast(span) {
  // Earlier ratings that looked at some of the same frames: how this stretch scored before.
  const past = (state.project.ratings || []).filter((r) => r.from < span.b && span.a < r.to).slice(0, 6);
  $("raterPast").textContent = past.length
    ? "Earlier ratings here:\n" + past.map((r) => r.at.replace("T", " ") + "  " + timecode(r.from) + " to " + timecode(r.to) + "  " + (ratingMean(r) ?? "–") + "%").join("\n")
    : "";
  return past;
}

async function openRater() {
  await flush();
  const span = rateSpan();
  const seconds = (span.b - span.a) / FPS;
  $("raterSummary").textContent = "Rate " + span.what + ": " + timecode(span.a) + " to " + timecode(span.b) + ", " + seconds.toFixed(1) + " s."
    + (span.b - span.a > RATE_MOST ? " That is more than three minutes: mark a shorter range." : "");
  $("raterResult").textContent = "";
  const past = raterPast(span);
  const same = past.find((r) => r.from === span.a && r.to === span.b);
  raterRows(same || null);
  $("rater").showModal();
  try {
    rater.key = (await api("/settings")).gemini;
  } catch (_) {
    rater.key = { set: false };
    $("raterResult").textContent = "This ComfyUI is running an older H3 Editor: restart it to use ratings.";
  }
  raterKey();
  $("raterGo").onclick = async () => {
    rater.busy = true;
    $("raterGo").disabled = true;
    $("raterGo").textContent = "Rating";
    $("raterResult").textContent = "Exporting the range and sending it. A minute of video takes about half a minute.";
    const folder = state.folder;
    try {
      const reply = await postJson("/project/" + encodeURIComponent(folder) + "/rate", { from: span.a, to: span.b });
      if (folder === state.folder && !unsaved()) adopt(reply.project);
      raterRows(reply.rating);
      const mean = ratingMean(reply.rating);
      $("raterResult").textContent = (mean === null ? "" : "Overall " + mean + "%. ") + (reply.rating.tokens ? reply.rating.tokens.toLocaleString() + " tokens." : "");
      raterPast(span);
    } catch (error) {
      $("raterResult").textContent = error.message;
    }
    rater.busy = false;
    $("raterGo").textContent = "Rate";
    $("raterGo").disabled = !rater.key?.set;
  };
}

$("rate").addEventListener("click", openRater);

/* Writing with a language model. The calls are made by writer.js, from this browser. */
const writing = { busy: false, made: null, added: [], draw: null };

// Pictures for the writing model. The cast's own pictures always go, so it sees who it writes about; the user can
// add more for one dialog (a place, a product, a frame of the cut). Each is shrunk to a JPEG of at most 768 pixels.
const WRITER_CAST_MOST = 6;
const WRITER_ADDED_MOST = 4;
const shrunk = new Map();

async function shrinkPicture(source) {
  const key = typeof source === "string" ? source : null;
  if (key && shrunk.has(key)) return shrunk.get(key);
  const blob = typeof source === "string" ? await (await fetch(source)).blob() : source;
  const bitmap = await createImageBitmap(blob);
  const scale = Math.min(1, 768 / Math.max(bitmap.width, bitmap.height));
  const canvas = new OffscreenCanvas(Math.max(1, Math.round(bitmap.width * scale)), Math.max(1, Math.round(bitmap.height * scale)));
  canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  const out = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.85 });
  const url = await new Promise((done, fail) => {
    const reader = new FileReader();
    reader.onload = () => done(reader.result);
    reader.onerror = () => fail(new Error("Could not read a picture for the writing model."));
    reader.readAsDataURL(out);
  });
  const made = { blob: out, url };
  if (key) shrunk.set(key, made);
  return made;
}

const writerCast = () => (state.project.cast || []).filter((person) => person.file && person.name).slice(0, WRITER_CAST_MOST);

async function writerPictures() {
  const rows = [];
  for (const person of writerCast()) rows.push({ label: "@" + person.name + ", a cast member: how they look", ...(await shrinkPicture(assetUrl(person))) });
  for (const added of writing.added) rows.push({ label: "A picture the user added" + (added.name ? ": " + added.name : ""), ...(await shrinkPicture(added.blob)) });
  return rows;
}

function addWriterPicture(blob, name) {
  if (writing.added.length >= WRITER_ADDED_MOST) return toast("Up to " + WRITER_ADDED_MOST + " added pictures. Remove one first.", true);
  writing.added.push({ blob, name: name || "", thumb: URL.createObjectURL(blob) });
  writing.draw?.();
}

// The frame the viewer shows right now, as a picture.
async function viewerPicture() {
  const video = [...document.querySelectorAll("#screen video.show")].pop();
  if (!video || video.readyState < 2 || !video.videoWidth) throw new Error("The viewer shows no frame. Move the playhead over a rendered clip.");
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  return new Promise((done) => canvas.toBlob(done, "image/jpeg", 0.92));
}

function picturesStrip() {
  const box = document.createElement("div");
  box.className = "writer-pictures";
  const head = document.createElement("span");
  head.className = "writer-pictures-head";
  head.textContent = "Pictures it sees";
  const thumbs = document.createElement("div");
  thumbs.className = "writer-thumbs";
  const hint = document.createElement("p");
  hint.className = "hint";
  const tile = (src, caption, title, remove = null) => {
    const figure = document.createElement("figure");
    figure.className = "writer-thumb";
    figure.title = title;
    const image = document.createElement("img");
    image.alt = "";
    image.src = src;
    const label = document.createElement("figcaption");
    label.textContent = caption;
    figure.append(image, label);
    if (remove) {
      const drop = document.createElement("button");
      drop.type = "button";
      drop.className = "writer-thumb-remove";
      drop.setAttribute("aria-label", "Remove this picture");
      drop.textContent = "\u00d7";
      drop.addEventListener("click", remove);
      figure.append(drop);
    }
    return figure;
  };
  const adder = (label, title, run) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "quiet writer-thumb-add";
    button.textContent = label;
    button.title = title;
    button.addEventListener("click", run);
    return button;
  };
  writing.draw = () => {
    const cast = writerCast();
    thumbs.replaceChildren(
      ...cast.map((person) => tile(assetUrl(person), "@" + person.name, "Always sent: how @" + person.name + " looks")),
      ...writing.added.map((added, index) => tile(added.thumb, added.name || "Added", "Sent with this request", () => {
        URL.revokeObjectURL(added.thumb);
        writing.added.splice(index, 1);
        writing.draw();
      })),
      ...(writing.added.length < WRITER_ADDED_MOST ? [
        adder("Add picture", "Upload a picture for the model to look at", async () => {
          const file = await pickFile($("frameFile"));
          if (file) addWriterPicture(file, file.name);
        }),
        adder("Playhead frame", "Add the frame the viewer shows", async () => {
          try {
            addWriterPicture(await viewerPicture(), "Frame at " + timecode(state.playhead));
          } catch (error) {
            toast(error.message, true);
          }
        }),
      ] : []),
    );
    hint.textContent = (cast.length ? "The cast's pictures always go with it. " : "")
      + "Drop, paste or add a picture of a place, a product or a frame, and the model writes from what it shows.";
  };
  writing.draw();
  dropTarget(box, (file) => addWriterPicture(file, file.name), "a picture for the model");
  box.append(head, thumbs, hint);
  return box;
}

$("writer").addEventListener("paste", (event) => {
  const file = [...(event.clipboardData?.files || [])].find((f) => f.type.startsWith("image/"));
  if (!file || !writing.draw) return;
  event.preventDefault();
  addWriterPicture(file, "Pasted picture");
});

// A labelled input for a dialog. Unlike field(), it changes nothing in the project.
function plainField(label, value, { area = false, type = "text", placeholder = "" } = {}) {
  const wrap = document.createElement("label");
  wrap.className = "field";
  const title = document.createElement("span");
  title.textContent = label;
  const input = document.createElement(area ? "textarea" : "input");
  if (!area) input.type = type;
  if (placeholder) input.placeholder = placeholder;
  input.autocomplete = "off";
  input.spellcheck = area;
  input.value = value ?? "";
  wrap.append(title, input);
  return { wrap, input };
}

// The models the endpoint listed last, for the model picker. A new provider starts with none.
let writerModels = [];

// A text field that also offers a list to pick from: typing still takes any value, the chevron (or Down) opens the
// list, which narrows to what is typed and marks the value in use. The list is a popover inside the field, so it
// works inside a modal dialog. names() gives the choices; pick(name) is told what was picked.
function comboBox(field, names, pick) {
  const box = document.createElement("div");
  box.className = "combo";
  field.input.replaceWith(box);
  const input = field.input;
  input.setAttribute("role", "combobox");
  input.setAttribute("aria-autocomplete", "list");
  input.setAttribute("aria-expanded", "false");
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "combo-toggle";
  toggle.setAttribute("aria-label", "Show the listed models");
  toggle.tabIndex = -1;
  toggle.innerHTML = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4.5 6.5L8 10l3.5-3.5"/></svg>';
  const list = document.createElement("div");
  list.className = "menu combo-list";
  list.id = "combo" + Math.random().toString(16).slice(2, 8);
  list.setAttribute("popover", "");
  list.setAttribute("role", "listbox");
  input.setAttribute("aria-controls", list.id);
  box.append(input, toggle, list);

  const options = () => [...list.querySelectorAll(".combo-option")];
  const choose = (name) => {
    input.value = name;
    pick(name);
    list.hidePopover();
    input.focus();
  };
  const fill = (all) => {
    const typed = input.value.trim().toLowerCase();
    const shown = all || !typed ? names() : names().filter((name) => name.toLowerCase().includes(typed));
    const rows = shown.map((name) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = "combo-option";
      row.setAttribute("role", "option");
      const current = name === input.value.trim();
      row.setAttribute("aria-selected", String(current));
      row.innerHTML = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 8.5l3 3 6-7"/></svg>';
      const label = document.createElement("span");
      label.textContent = name;
      row.append(label);
      row.addEventListener("click", () => choose(name));
      return row;
    });
    list.replaceChildren(...(rows.length ? rows : [Object.assign(document.createElement("p"), { className: "combo-empty",
      textContent: names().length ? "No listed model has that in its name. Keep typing to use your own." : "Nothing listed yet. Press List models, or type a name." })]));
  };
  const place = () => {
    const at = input.getBoundingClientRect();
    list.style.width = at.width + "px";
    list.style.left = at.left + "px";
    const below = at.bottom + 4 + list.offsetHeight <= window.innerHeight - 8;
    list.style.top = (below ? at.bottom + 4 : at.top - 4 - list.offsetHeight) + "px";
  };
  const open = (all = false) => {
    fill(all);
    if (!list.matches(":popover-open")) list.showPopover();
    place();
    (options().find((row) => row.getAttribute("aria-selected") === "true") || options()[0])?.scrollIntoView({ block: "nearest" });
  };
  list.addEventListener("toggle", (event) => input.setAttribute("aria-expanded", String(event.newState === "open")));
  toggle.addEventListener("pointerdown", (event) => event.preventDefault());       // keep the caret in the field
  toggle.addEventListener("click", () => (list.matches(":popover-open") ? list.hidePopover() : (input.focus(), open(true))));
  input.addEventListener("input", () => {
    if (names().length) open();
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown" && names().length) {
      event.preventDefault();
      if (!list.matches(":popover-open")) open(true);
      options()[0]?.focus();
    } else if (event.key === "Escape" && list.matches(":popover-open")) {
      event.preventDefault();
      event.stopPropagation();                                                      // close the list, not the dialog
      list.hidePopover();
    }
  });
  list.addEventListener("keydown", (event) => {
    const rows = options();
    const at = rows.indexOf(document.activeElement);
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const next = at + (event.key === "ArrowDown" ? 1 : -1);
      if (next < 0) input.focus();
      else rows[Math.min(next, rows.length - 1)]?.focus();
    } else if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      list.hidePopover();
      input.focus();
    } else if (event.key.length === 1 || event.key === "Backspace") {
      input.focus();                                                                // typing goes on in the field
    }
  });
  return { open };
}

function writerNote(text, bad = false, target = "writerResult") {
  $(target).textContent = text;
  $(target).classList.toggle("bad", bad);
}

function openWriterSetup(then = null) {
  const draft = writer.settings();
  const note = (text, bad = false) => writerNote(text, bad, "writerSetupResult");
  const draw = () => {
    const kind = writer.provider(draft.type);
    const type = document.createElement("label");
    type.className = "choice";
    const typeTitle = document.createElement("span");
    typeTitle.textContent = "Provider";
    const types = document.createElement("select");
    for (const row of writer.PROVIDERS) {
      const option = document.createElement("option");
      option.value = row.id;
      option.textContent = row.title;
      option.selected = row.id === draft.type;
      types.append(option);
    }
    types.addEventListener("change", () => {
      // A model name belongs to its provider: a ComfyUI file means nothing to an endpoint, and the other way round.
      if (writer.provider(draft.type).local !== writer.provider(types.value).local) draft.model = "";
      draft.type = types.value;
      draft.url = "";
      writerModels = [];
      note("");
      if (writer.provider(draft.type).local && !draft.model) draft.model = writer.COMFY_MODEL;
      draw();
    });
    type.append(typeTitle, types);
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent = kind.hint;
    if (kind.local && !draft.model) draft.model = writer.COMFY_MODEL;
    const address = plainField("Address", draft.url, { placeholder: kind.address || "Google's own address" });
    address.input.addEventListener("input", () => (draft.url = address.input.value));
    const model = plainField("Model", draft.model, { placeholder: kind.local ? "A file in models/text_encoders" : "The model's name, as the endpoint lists it" });
    const picker = comboBox(model, () => writerModels, (name) => (draft.model = name));
    model.input.addEventListener("input", () => (draft.model = model.input.value));
    const list = document.createElement("button");
    list.className = "quiet";
    list.textContent = "List models";
    list.addEventListener("click", async () => {
      note(kind.local ? "Asking ComfyUI for its text models." : "Asking the endpoint for its models.");
      list.disabled = true;
      try {
        writerModels = await writer.listModels(draft);
        note(writerModels.length ? writerModels.length + (writerModels.length === 1 ? " model" : " models") + ". Pick one, or type another name."
          : "The endpoint lists no models. Type the model's name.");
        if (writerModels.length) picker.open(true);
      } catch (error) {
        note(error.message, true);
      }
      list.disabled = false;
    });
    const modelRow = document.createElement("div");
    modelRow.className = "writer-model";
    modelRow.append(model.wrap, list);
    const key = plainField(kind.key ? "API key" : "API key, if the endpoint asks for one", draft.key, { type: "password", placeholder: "Paste your key" });
    key.input.addEventListener("input", () => (draft.key = key.input.value));
    // A model run by this ComfyUI has no address and needs no key.
    $("writerSetupFields").replaceChildren(...(kind.local ? [type, hint, modelRow] : [type, hint, address.wrap, modelRow, key.wrap]));
  };
  draw();
  note("");
  $("writerSetupTest").onclick = async () => {
    if (!writer.ready(draft)) return note("Name a model first" + (writer.provider(draft.type).key ? ", and paste a key." : "."), true);
    $("writerSetupTest").disabled = true;
    note("Asking the model.");
    try {
      note("It answered: " + (await writer.test(draft)));
    } catch (error) {
      note(error.message, true);
    }
    $("writerSetupTest").disabled = false;
  };
  $("writerSetupSave").onclick = () => {
    if (!writer.saveSettings(draft)) return note("This browser would not keep the settings.", true);
    $("writerSetup").close();
    renderInspector();
    if (then && writer.ready()) then();
  };
  $("writerSetup").showModal();
}

// One written shot, as a card: its name and length, then each field that holds something.
function shotCard(shot, title = null) {
  const card = document.createElement("div");
  card.className = "writer-shot";
  if (title) {
    const head = document.createElement("h3");
    head.textContent = title;
    if (shot.seconds) {
      const length = document.createElement("span");
      length.textContent = shot.seconds + " s";
      head.append(length);
    }
    card.append(head);
  }
  const list = document.createElement("dl");
  for (const key of writer.FIELDS) {
    if (!shot[key]) continue;
    const dt = document.createElement("dt");
    dt.textContent = writer.LABELS[key];
    const dd = document.createElement("dd");
    dd.textContent = shot[key];
    list.append(dt, dd);
  }
  card.append(list);
  return card;
}

// The writing dialog, for one clip or for a list of new ones. run asks the model and returns
// what it wrote, show draws that, and apply puts it into the project.
function openWriter(job) {
  if (!writer.ready()) return openWriterSetup(() => openWriter(job));
  writing.made = null;
  $("writerTitle").textContent = job.title;
  // Which model writes, with a way to choose another without leaving for the Project tab.
  const given = writer.settings();
  const using = document.createElement("div");
  using.className = "writer-using";
  const named = document.createElement("span");
  named.textContent = "Uses " + given.model + (writer.provider(given.type).local ? ", run by this ComfyUI." : " on " + writer.provider(given.type).title + ".");
  const change = document.createElement("button");
  change.className = "quiet small";
  change.textContent = "Change";
  change.title = "Choose another model or provider";
  change.addEventListener("click", () => {
    if (writing.busy) return;
    $("writer").close();
    openWriterSetup(() => openWriter(job));
  });
  using.append(named, change);
  for (const added of writing.added) URL.revokeObjectURL(added.thumb);
  writing.added = [];
  $("writerAsk").replaceChildren(...job.fields, picturesStrip(), using);
  $("writerPreview").replaceChildren();
  $("writerGo").textContent = job.go;
  $("writerUse").textContent = job.use(null);
  $("writerUse").disabled = true;
  writerNote("");
  $("writerGo").onclick = async () => {
    if (writing.busy) return;
    writing.busy = true;
    $("writerGo").disabled = true;
    $("writerUse").disabled = true;
    writerNote("Writing. A local model can take a minute.");
    const folder = state.folder;
    try {
      const pictures = await writerPictures();
      if (pictures.length) writerNote("Writing, with " + pictures.length + (pictures.length === 1 ? " picture" : " pictures") + ". A local model can take a minute.");
      const made = await job.run(pictures);
      if (folder !== state.folder || !$("writer").open) return;
      writing.made = made;
      $("writerPreview").replaceChildren(...job.show(made));
      $("writerUse").textContent = job.use(made);
      $("writerUse").disabled = false;
      $("writerGo").textContent = "Write again";
      const seen = writer.lastSeen();
      writerNote("Read it before you use it. Nothing has changed yet."
        + (seen.blind ? " This model takes no pictures, so it wrote from the text alone." : seen.pictures ? " It saw " + seen.pictures + (seen.pictures === 1 ? " picture." : " pictures.") : ""));
    } catch (error) {
      writerNote(error.message, true);
    } finally {
      writing.busy = false;
      $("writerGo").disabled = false;
    }
  };
  $("writerUse").onclick = () => {
    if (!writing.made) return;
    job.apply(writing.made);
    writing.made = null;
    $("writer").close();
  };
  $("writer").showModal();
  $("writerAsk").querySelector("textarea")?.focus();
}

// The clips on either side of this one in the cut, so the model keeps the shot in step with them.
function neighbours(clip) {
  const placed = [...state.project.timeline].sort((a, b) => a.start - b.start);
  const at = placed.findIndex((item) => item.clip === clip.id);
  const other = (item) => (item ? clipOf(item.clip) || null : null);
  return at < 0 ? { before: null, after: null } : { before: other(placed[at - 1]), after: other(placed[at + 1]) };
}

// Cast members with a picture or a RefMod join a clip's cast when its text places them.
function castIn(text) {
  return (state.project.cast || [])
    .filter((person) => (person.file || person.refmod) && person.name && String(text).split("@" + person.name).slice(1).some((rest) => !/^[\w-]/.test(rest)))
    .map((person) => person.id);
}

function openRewrite(clip) {
  const empty = !String(clip.action || "").trim();
  const directions = plainField(empty ? "What is this shot" : "What should change", "", {
    area: true,
    placeholder: empty ? "A keeper climbs the lighthouse stairs at dusk, out of breath." : "Leave empty to tighten it as it is. Or: make it night, slow the camera, add a line.",
  });
  openWriter({
    title: (empty ? "Write " : "Rewrite ") + clip.name,
    fields: [directions.wrap],
    go: empty ? "Write" : "Rewrite",
    use: () => "Use it",
    run: (pictures) => writer.rewrite({ project: state.project, clip, ...neighbours(clip), directions: directions.input.value, pictures }),
    show: (made) => [shotCard(made)],
    apply: (made) => {
      const live = clipOf(clip.id);
      if (!live) return;
      checkpoint();
      for (const key of writer.FIELDS) live[key] = made[key];
      const placed = castIn(made.action).filter((id) => !(live.cast || []).includes(id));
      if (placed.length) live.cast = [...(live.cast || []), ...placed];
      save();
      renderAll();
      toast((empty ? "Wrote " : "Rewrote ") + live.name + ". Ctrl+Z brings the old text back.");
    },
  });
}

function openWriteShots() {
  const brief = plainField("Describe the video", "", {
    area: true,
    placeholder: "Who is in it, where it happens, what happens, and what is said. For example: a 20 second teaser for a seaside cafe opening at dawn.",
  });
  brief.input.classList.add("big");
  const count = plainField("Shots", 4, { type: "number" });
  const each = plainField("Seconds each", 5, { type: "number" });
  for (const [input, [low, high]] of [[count.input, writer.LIMITS.shots], [each.input, writer.LIMITS.seconds]]) {
    input.min = low;
    input.max = high;
  }
  const numbers = document.createElement("div");
  numbers.className = "writer-numbers";
  numbers.append(count.wrap, each.wrap);
  openWriter({
    title: "Write shots",
    fields: [brief.wrap, numbers],
    go: "Write shots",
    use: (made) => (made ? "Add " + made.length + (made.length === 1 ? " clip" : " clips") : "Add clips"),
    run: async (pictures) => {
      if (!brief.input.value.trim()) throw new Error("Describe the video first.");
      return writer.writeShots({ project: state.project, brief: brief.input.value, count: count.input.value, seconds: each.input.value, pictures });
    },
    show: (made) => made.map((shot) => shotCard(shot, shot.name)),
    apply: (made) => {
      checkpoint();
      const bin = selectedBin();
      let first = null;
      for (const shot of made) {
        const clip = { id: uid("c"), name: shot.name, action: shot.action, camera: shot.camera, speaker: shot.speaker, dialogue: shot.dialogue,
          sound: shot.sound, music: shot.music, seconds: shot.seconds, takes: [], active: null };
        const cast = castIn(shot.action);
        if (cast.length) clip.cast = cast;
        if (bin) clip.bin = bin.id;
        state.project.clips.push(clip);
        placeOnTimeline(clip, sceneEnd());
        first = first || clip;
      }
      state.clip = first.id;
      state.person = null;
      state.item = state.project.timeline.find((item) => item.clip === first.id)?.id || null;
      state.tab = "shot";
      save();
      renderAll();
      toast("Added " + made.length + (made.length === 1 ? " clip" : " clips") + " at the end of the cut. Render all drafts to see them.");
    },
  });
}

function rewriteButton(clip) {
  const button = document.createElement("button");
  button.className = "quiet wide";
  button.dataset.key = "Rewrite";
  button.textContent = String(clip.action || "").trim() ? "Rewrite with a language model" : "Write with a language model";
  button.title = "A language model writes this shot's text, for you to read before it replaces anything";
  button.addEventListener("click", () => openRewrite(clip));
  return button;
}

function writingSection() {
  const given = writer.settings();
  const set = writer.ready(given);
  const shots = document.createElement("button");
  shots.className = "quiet wide";
  shots.dataset.key = "Write shots";
  shots.textContent = "Write shots";
  shots.title = "Describe the video, and a language model writes its shots as new clips";
  shots.addEventListener("click", openWriteShots);
  const setup = document.createElement("button");
  setup.className = "quiet wide";
  setup.dataset.key = "Writing setup";
  setup.textContent = set ? "Change the language model" : "Set up a language model";
  setup.addEventListener("click", () => openWriterSetup());
  const hint = document.createElement("p");
  hint.className = "hint";
  hint.textContent = set ? "Writing uses " + given.model + (writer.provider(given.type).local ? ", run by this ComfyUI." : " on " + writer.provider(given.type).title + ", from this browser.")
    : "A language model can write a video's shots, or rewrite one clip: a text model run by this ComfyUI, or any OpenAI-compatible endpoint, Anthropic, Gemini or Ollama.";
  const pair = document.createElement("div");
  pair.className = "pair buttons";
  pair.append(shots, setup);
  return section("Writing", [pair, hint]);
}

$("writerClose").addEventListener("click", () => $("writer").close());
$("writerSetupClose").addEventListener("click", () => $("writerSetup").close());
$("raterClose").addEventListener("click", () => $("rater").close());

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
  drawNotes();
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
    const slipping = event.altKey && !edge;
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
      if (slipping) {
        // The picture follows the pointer: dragging right brings in what came before.
        item.in = origin.in;
        item.out = origin.out;
        slipBy(item, -delta * rate);
        seek(Math.max(item.start, Math.min(itemEnd(item) - 1, state.playhead)));
        return;
      }
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
        if (!slipping) settle(item);
        save();
      }
      if (slipping) renderAll();
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
  if (state.sound && !state.item) {
    removeSound(state.sound);
    return;
  }
  if (state.effect && !state.item) {
    removeEffect(state.effect);
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

/* Context menus: a right click on a clip, a block of the timeline, a VFX clip, a sound, a cast member or the viewer
   selects it and offers what can be done to it, the same actions as the buttons and keys. An action that cannot be
   done right now stays in the list, greyed, with the reason in its tip. */

// One row: a label, an optional key, and what it does. "-" is a separator.
const act = (label, run, { key = "", off = "", danger = false } = {}) => ({ label, run, key, off, danger });

function itemActions(item) {
  const clip = clipOf(item.clip);
  const rendered = Boolean(activeTake(clip));
  const busy = Boolean(jobFor(clip.id));
  const next = followingItem(item);
  const holdable = rendered && itemLength(item) >= HOLD;
  const inside = state.playhead > item.start && state.playhead < itemEnd(item);
  return [
    act("Render draft", () => renderClip(clip, "draft"), { off: busy ? clip.name + " is rendering" : "" }),
    act("Render final", () => renderClip(clip, "final"), { off: busy ? clip.name + " is rendering" : "" }),
    "-",
    act(String(clip.action || "").trim() ? "Rewrite with a language model" : "Write with a language model", () => openRewrite(clip)),
    "-",
    act("Extend forward", () => addLinked("extend_after"), { off: holdable ? "" : "Render it first; it needs " + HOLD + " frames to carry on from" }),
    act("Extend back", () => addLinked("extend_before"), { off: holdable ? "" : "Render it first; it needs " + HOLD + " frames to carry on from" }),
    act("Bridge to next", () => addLinked("bridge"), { off: bridgeable(item, next) ? "" : "Needs a rendered clip after it" }),
    act("Merge with next", mergeWithNext, { off: next && activeTake(clipOf(next.clip)) && next.start === itemEnd(item) ? "" : "Needs a rendered clip right after it" }),
    "-",
    act("Split at the playhead", splitAtPlayhead, { key: "S", off: inside ? "" : "Move the playhead over this clip" }),
    act("Show its takes", () => { state.tab = "takes"; renderAll(); }, { off: clip.takes.length ? "" : "Not rendered yet" }),
    "-",
    act("Remove from the timeline", removeSelected, { key: "Delete" }),
    act("Ripple delete", rippleDelete, { key: "Shift+Delete", danger: true }),
  ];
}

function clipActions(clip) {
  const busy = Boolean(jobFor(clip.id));
  return [
    act("Render draft", () => renderClip(clip, "draft"), { off: busy ? clip.name + " is rendering" : "" }),
    act("Render final", () => renderClip(clip, "final"), { off: busy ? clip.name + " is rendering" : "" }),
    "-",
    act(String(clip.action || "").trim() ? "Rewrite with a language model" : "Write with a language model", () => openRewrite(clip)),
    act("Add to the end of the cut", () => {
      checkpoint();
      placeOnTimeline(clip, sceneEnd());
      state.item = state.project.timeline.at(-1).id;
      save();
      renderAll();
    }),
    "-",
    act("Delete clip", () => deleteClip(clip), { danger: true }),
  ];
}

function viewerActions() {
  const clip = clipOf(state.clip);
  const shown = viewing() ? activeTake(viewing()) : activeTake(clipOf(itemAt(state.playhead)?.clip));
  const off = !clip ? "Select the clip to give it this frame" : !shown ? "The viewer shows no rendered frame" : "";
  const rows = [];
  for (const [side, label] of [["first", "First frame"], ["last", "Last frame"]]) {
    rows.push(act("Use as the " + label.toLowerCase() + (clip ? " of " + clip.name : ""), () => grabFrame(clip, side),
      { off: off || (frameUsable(clip, side) ? "" : clip.name + " carries on from another clip there") }));
  }
  rows.push("-", act("Note on this frame", startNote, { key: "N", off: shown ? "" : "The viewer shows no rendered frame" }));
  return rows;
}

// What a right click landed on: the actions for it, after selecting it as a left click would.
function actionsAt(target) {
  const block = target.closest("#laneV .block[data-item]");
  if (block) {
    const item = state.project.timeline.find((i) => i.id === block.dataset.item);
    if (!item) return null;
    selectClip(item.clip, item.id);
    return { title: clipOf(item.clip).name, rows: itemActions(item) };
  }
  const row = target.closest("#clips .clip[data-clip]");
  if (row) {
    const clip = clipOf(row.dataset.clip);
    if (!clip) return null;
    selectClip(clip.id);
    return { title: clip.name, rows: clipActions(clip) };
  }
  const effectBlock = target.closest("#laneX [data-effect]");
  if (effectBlock) {
    const effect = effectOf(effectBlock.dataset.effect);
    if (!effect) return null;
    selectEffect(effect.id);
    return { title: effect.name, rows: [act("Render", () => renderEffectTake(effect.id), { off: fx.rendering ? "A VFX clip is rendering" : "" }), "-",
      act("Remove from the VFX lane", () => removeEffect(effect.id), { key: "Delete" })] };
  }
  const sound = target.closest("[data-sound]");
  if (sound) {
    const found = (state.project.sounds || []).find((s) => s.id === sound.dataset.sound);
    if (!found) return null;
    selectSound(found.id);
    return { title: found.name || "Sound", rows: [act("Remove from the sound lane", () => removeSound(found.id), { key: "Delete" })] };
  }
  const member = target.closest("#cast [data-person]");
  if (member) {
    const person = (state.project.cast || []).find((p) => p.id === member.dataset.person);
    if (!person) return null;
    member.click();
    return { title: person.name, rows: [act("Use the voice of the clip under the playhead", () => grabVoice(person)), "-",
      act("Remove from the cast", () => removePerson(person), { danger: true })] };
  }
  if (target.closest("#screen")) return { title: "Viewer", rows: viewerActions() };
  return null;
}

function openContextMenu(found, x, y, keyboard = false) {
  const menu = $("contextMenu");
  const head = document.createElement("p");
  head.className = "context-title";
  head.textContent = found.title;
  menu.replaceChildren(head, ...found.rows.map((row) => {
    if (row === "-") return Object.assign(document.createElement("hr"), { className: "context-rule" });
    const button = document.createElement("button");
    button.setAttribute("role", "menuitem");
    button.className = "context-row" + (row.danger ? " danger" : "");
    const label = document.createElement("span");
    label.textContent = row.label;
    button.append(label);
    if (row.key) {
      const key = document.createElement("kbd");
      key.textContent = row.key;
      button.append(key);
    }
    if (row.off) {
      button.setAttribute("aria-disabled", "true");
      button.title = row.off;
    }
    button.addEventListener("click", () => {
      if (row.off) return toast(row.off + ".");
      menu.hidePopover();
      row.run();
    });
    return button;
  }));
  menu.style.left = "0px";
  menu.style.top = "0px";
  menu.showPopover();
  menu.style.left = Math.max(8, Math.min(x, window.innerWidth - menu.offsetWidth - 8)) + "px";
  menu.style.top = Math.max(8, Math.min(y, window.innerHeight - menu.offsetHeight - 8)) + "px";
  // From the keyboard the first action takes the focus; from the mouse the menu does, so arrows still work.
  if (keyboard) menu.querySelector(".context-row:not([aria-disabled])")?.focus();
  else menu.focus();
}

document.addEventListener("contextmenu", (event) => {
  if (!state.project || event.target.closest("dialog, input, textarea, select, [contenteditable], #contextMenu")) return;
  const found = actionsAt(event.target);
  if (!found) return;
  event.preventDefault();
  openContextMenu(found, event.clientX, event.clientY);
});

// The keyboard's menu key, or Shift+F10, opens the same menu for what has the focus.
document.addEventListener("keydown", (event) => {
  if (!(event.key === "ContextMenu" || (event.key === "F10" && event.shiftKey))) return;
  const focused = document.activeElement;
  if (!focused || !state.project || focused.closest("dialog, input, textarea, select")) return;
  const found = actionsAt(focused);
  if (!found) return;
  event.preventDefault();
  const at = focused.getBoundingClientRect();
  openContextMenu(found, at.left + 12, at.bottom - 4, true);
});

$("contextMenu").addEventListener("keydown", (event) => {
  if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
  event.preventDefault();
  const rows = [...$("contextMenu").querySelectorAll(".context-row")];
  const at = rows.indexOf(document.activeElement);
  const step = event.key === "ArrowDown" ? 1 : -1;
  rows[at < 0 ? (step > 0 ? 0 : rows.length - 1) : (at + step + rows.length) % rows.length]?.focus();
});

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
    video.classList.remove("under");
    video.src = url;
    video.addEventListener("loadeddata", () => {
      if (!state.playing) showFrame();
    }, { once: true });
  }
}

// The live preview loops a clip that is still being sampled, so its sound is off until asked for.
let liveSound = false;
try { liveSound = localStorage.getItem("h3take.liveSound") === "on"; } catch (_) {}

function setLiveSound(on) {
  liveSound = on;
  try { localStorage.setItem("h3take.liveSound", on ? "on" : "off"); } catch (_) {}
  $("motion").muted = !on;
  const button = $("liveSound");
  button.setAttribute("aria-pressed", String(on));
  button.dataset.tip = on ? "Mute the preview" : "Hear the preview";
}

function showLive(job) {
  const live = $("live");
  const image = $("preview");
  const showing = !!job && !state.playing;
  const motion = $("motion");
  live.hidden = !showing;
  image.hidden = !(showing && job.preview && !job.motion);
  motion.hidden = !(showing && job.motion);
  $("liveSound").hidden = motion.hidden;
  if (!showing) {
    motion.pause();
    return;
  }
  if (job.preview && !job.motion) image.src = job.preview;
  if (job.motion && motion.dataset.url !== job.motion) {
    const resume = motion.currentTime;
    motion.dataset.url = job.motion;
    motion.muted = !liveSound;
    motion.src = job.motion;
    motion.onloadedmetadata = () => {
      motion.currentTime = resume % (motion.duration || 1);
      motion.play().catch(() => {
        // The browser would not start it with sound: play it muted, and say so on the button.
        setLiveSound(false);
        motion.play().catch(() => {});
      });
    };
  }
  live.textContent = (clipOf(job.clip)?.name || "Clip") + ": " + (job.stage || "Waiting in the queue") + (job.max ? ", step " + job.step + " of " + job.max : "");
}

setLiveSound(liveSound);
$("liveSound").addEventListener("click", () => setLiveSound(!liveSound));

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

// The cut has one shape, as the export does: that of its largest picture. Every clip fills it, so a draft and a
// final of slightly different proportions do not change the picture's height from one clip to the next.
function takeShape(take) {
  if (Array.isArray(take.size) && take.size[0] > 0 && take.size[1] > 0) return take.size;
  const rows = state.sizes?.[state.project.format || "16:9"]?.[take.quality === "draft" ? "draft" : "final"];
  return rows?.[0]?.out || [16, 9];
}

function cutShape() {
  let best = null;
  for (const item of state.project.timeline) {
    const take = activeTake(clipOf(item.clip));
    if (!take) continue;
    const shape = takeShape(take);
    if (!best || shape[0] > best[0]) best = shape;
  }
  return best;
}

function showFrame() {
  showPicture();
  previewEffect();
}

function showPicture() {
  const clip = viewing();
  const shape = clip ? null : cutShape();
  $("screen").classList.toggle("cut", Boolean(shape));
  if (shape) $("screen").style.setProperty("--frame", String(shape[0] / shape[1]));
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
  front.classList.remove("under");
  back.classList.remove("under");
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

// At a cut the next clip's video may still be seeking to its first frame, with nothing drawn yet, and the
// screen would show black for a moment. The clip that just ended stays underneath on its last frame until
// the next one has drawn a frame of its own.
function cover(outgoing, coming) {
  const lift = () => outgoing.classList.remove("under");
  if (coming.readyState >= 2 && !coming.seeking) return;
  outgoing.classList.add("under");
  if (coming.requestVideoFrameCallback) coming.requestVideoFrameCallback(lift);
  else coming.addEventListener("seeked", lift, { once: true });
  setTimeout(lift, 500);                                   // never left behind if the next clip fails to load
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
    syncSounds();
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
  syncSounds();
  previewEffect();

  const coming = incoming(state.playhead);
  if (coming && coming !== item && sourceOf(coming)) {
    if (back.paused || back.dataset.url !== sourceOf(coming)) {
      place(back, coming, state.playhead, 0.08);
      back.play().catch(() => {});
    }
    applyMix(coming, state.playhead);
  } else {
    const following = item ? sorted().find((i) => i.start >= itemEnd(item) - (i.transition?.frames || 0) && i !== item) : nextItem(state.playhead);
    // The clip that just ended may still be covering the cut: it gets the next source once it is free.
    if (following && sourceOf(following) && back.paused && !back.classList.contains("under")) {
      load(back, sourceOf(following));
      // A video takes a moment to start, which would be a hole in the sound at every cut. So the
      // next clip waits a little before its in-point and starts early and silent, to be running
      // when the cut comes. A clip cut in at its very first frame has nothing before it to run.
      const rate = speedOf(following);
      const parked = Math.max(0, following.in / FPS - PREROLL * rate);
      const lead = (following.start - state.playhead) / FPS;
      if (lead > 0 && lead * rate <= following.in / FPS - parked && back.readyState >= 2 && !back.seeking) {
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
      const before = front;
      if (back.dataset.url === sourceOf(current)) swap();
      place(front, current, state.playhead, 0.08);
      front.classList.add("show");
      if (before !== front) cover(before, front);
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
async function renderClip(clip, quality, detail, upscaler) {
  if (jobFor(clip.id)) return;
  if (!clip.action.trim()) {
    selectClip(clip.id);
    toast("Describe what happens in " + clip.name + " before rendering it.", true);
    return;
  }
  try {
    await flush();
    const body = { clip: clip.id, quality, client_id: clientId };
    if (detail !== undefined) body.detail = detail;
    if (upscaler) body.upscaler = upscaler;
    const result = await postJson("/project/" + state.folder + "/render", body);
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
  const { projects, titles, profiles, finals, loras, realism, checkpoints, samplers, schedulers, presets, refmods, encoders, sizes, upscalers } = await api("/projects");
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
  state.sizes = sizes || null;
  state.upscalers = upscalers || [];
  $("project").replaceChildren(...projects.map((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = titles?.[name] || name.replaceAll("_", " ");
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
// A small dialog that asks for a name, or only for a yes. Resolves to the name, to true, or to null when it is closed.
function ask({ title, value = null, note = "", confirm, danger = false }) {
  const dialog = $("asker");
  const input = $("askerInput");
  $("askerTitle").textContent = title;
  $("askerField").hidden = value === null;
  input.value = value ?? "";
  $("askerNote").textContent = note;
  $("askerNote").classList.remove("bad");
  $("askerGo").textContent = confirm;
  $("askerGo").classList.toggle("danger", danger);
  return new Promise((resolve) => {
    const finish = (answer) => {
      $("askerForm").removeEventListener("submit", submit);
      $("askerClose").removeEventListener("click", cancel);
      dialog.removeEventListener("close", cancel);
      if (dialog.open) dialog.close();
      resolve(answer);
    };
    const cancel = () => finish(null);
    function submit(event) {
      event.preventDefault();
      if (value === null) return finish(true);
      const name = input.value.trim();
      if (/[A-Za-z0-9]/.test(name)) return finish(name);
      $("askerNote").textContent = "Give it a name with letters or numbers.";
      $("askerNote").classList.add("bad");
      input.focus();
    }
    $("askerForm").addEventListener("submit", submit);
    $("askerClose").addEventListener("click", cancel);
    dialog.addEventListener("close", cancel);
    dialog.showModal();
    if (value === null) $("askerGo").focus();
    else {
      input.focus();
      input.select();
    }
  });
}

const projectTitle = () => state.project?.name || (state.folder || "").replaceAll("_", " ");

$("projectRename").addEventListener("click", async () => {
  const name = await ask({ title: "Rename the project", value: projectTitle(), confirm: "Rename",
    note: "Only its name changes. Its folder in ComfyUI's output keeps the name it was made with." });
  if (!name || name === projectTitle()) return;
  try {
    await flush();
    const reply = await postJson("/project/" + state.folder + "/rename", { name, client_id: clientId });
    adopt(reply.project);
    await refreshProjects(state.folder);
    toast("The project is now called " + name + ".");
  } catch (error) {
    toast(error.message, true);
  }
});

$("projectDuplicate").addEventListener("click", async () => {
  const name = await ask({ title: "Duplicate the project", value: projectTitle() + " copy", confirm: "Duplicate",
    note: "The copy gets its own folder, with the clips, takes and pictures. A project with many finals takes a while to copy." });
  if (!name) return;
  try {
    await flush();
    toast("Copying " + projectTitle() + ".");
    const { folder } = await postJson("/project/" + state.folder + "/duplicate", { name, client_id: clientId });
    await refreshProjects(folder);
    await openProject(folder);
    toast("This is " + name + ", a copy. The original is untouched.");
  } catch (error) {
    toast(error.message, true);
  }
});

$("projectDelete").addEventListener("click", async () => {
  const title = projectTitle();
  const sure = await ask({ title: "Delete " + title + "?", confirm: "Delete project", danger: true,
    note: "It moves to the .trash folder inside h3take_projects in ComfyUI's output, with all its renders. Nothing is erased: move the folder back by hand to have it again." });
  if (!sure) return;
  try {
    await flush();
    const { trash } = await api("/project/" + state.folder, { method: "DELETE" });
    const left = await refreshProjects(null);
    if (left.length) await openProject(left[0]);
    else await createProject("First scene");
    toast(title + " is in " + trash + ".");
  } catch (error) {
    toast(error.message, true);
  }
});

$("newClip").addEventListener("click", addClip);
$("addEffect").addEventListener("click", addEffect);
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
$("addSound").addEventListener("click", addSound);
// A click on an empty part of a sound lane selects nothing.
$("tlContent").addEventListener("pointerdown", (event) => {
  if (event.target.classList.contains("extra") && event.target.classList.contains("lane")) selectClip(null);
});
// A menu opens under its button, or over it when the window is short, and closes once something is chosen.
for (const [button, id] of [["generate", "generateMenu"], ["castAdd", "castMenu"], ["projectMore", "projectMenu"]]) {
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
// The checks open under their chip, lined up with its right edge, and list what is found as it is.
$("checksPanel").addEventListener("toggle", (event) => {
  if (event.newState !== "open") return;
  const panel = $("checksPanel");
  panel.style.left = "0px";
  panel.style.top = "0px";
  renderChecks();
  const at = $("checks").getBoundingClientRect();
  panel.style.left = Math.max(8, Math.min(at.right - panel.offsetWidth, window.innerWidth - panel.offsetWidth - 8)) + "px";
  panel.style.top = at.bottom + 6 + "px";
});
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
  // With Shift held, the drag also marks a range, from where it started to where it is.
  const from = event.shiftKey ? frameAt(event) : null;
  const move = (ev) => {
    seek(frameAt(ev));
    if (from !== null) setRange(from, frameAt(ev));
  };
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
  if (event.key === "Escape" && noting.active) {
    closeNote();
    drawNotes();
    return;
  }
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
    if (state.clip || state.item || state.person || state.bin || state.gap || state.sound || state.effect) selectClip(null);
    else if (state.range) setRange(null, null);
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
  else if (event.key === ",") slipSelected(event.shiftKey ? -12 : -1);
  else if (event.key === "." ) slipSelected(event.shiftKey ? 12 : 1);
  else if (event.key === "<") slipSelected(-12);
  else if (event.key === ">") slipSelected(12);
  else if (event.key === "i" || event.key === "I") setRange(here(), state.range && state.range.b > here() ? state.range.b : sceneEnd());
  else if (event.key === "o" || event.key === "O") setRange(state.range && state.range.a < here() ? state.range.a : 0, here());
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
  else if (event.key.toLowerCase() === "n" && !event.ctrlKey && !event.metaKey && !event.altKey) {
    // Kept from the note's text box, which takes focus as the key goes down.
    event.preventDefault();
    startNote();
  }
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
  // The timeline gives way too, and comes back to the height it was dragged to when there is room again.
  let tall = 0;
  try { tall = Number(localStorage.getItem("h3take.timeline")); } catch (_) {}
  setTimelineHeight(tall || $("timeline").getBoundingClientRect().height - (parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--tl-extra")) || 0));
  if (state.project) renderTimeline();
});

/* Notes pinned to frames. A reviewer stops on a moment, points at it with a pin, a box or a
   drawing, and writes what should change. A note keeps the clip, the take and the frame inside
   that take, so it stays on its picture when the clip moves on the timeline. Agents read the
   notes, with a still of each frame and the drawing on it, through the summary and h3.py notes. */
const noting = { active: null, selected: null };
const NOTE_INK = "#ffb547";

const notesOf = () => (state.project?.notes || []).filter((n) => n && n.id && clipOf(n.clip));

// The clip, the take and the frame inside it that the viewer shows now, or null where there is no picture.
function noteTarget() {
  const alone = viewing();
  if (alone) {
    const take = activeTake(alone);
    return take ? { clip: alone, take, frame: Math.max(0, Math.min(take.frames - 1, Math.round(state.sourceFrame))), item: null } : null;
  }
  const end = sceneEnd();
  const at = end && state.playhead >= end ? end - 1 : state.playhead;
  const item = leading(at);
  const clip = item && clipOf(item.clip);
  const take = clip && activeTake(clip);
  if (!take) return null;
  const frame = Math.max(0, Math.min(take.frames - 1, Math.round(item.in + (at - item.start) * speedOf(item))));
  return { clip, take, frame, item };
}

// Where a note's frame is on the timeline now, or null when the cut does not show that frame.
function notePlace(note) {
  for (const item of sorted()) {
    if (item.clip === note.clip && note.frame >= item.in && note.frame < item.out) return Math.round(item.start + (note.frame - item.in) / speedOf(item));
  }
  return null;
}

function notesHere(target = noteTarget()) {
  if (!target) return [];
  return notesOf().filter((n) => n.clip === target.clip.id
    && (n.range ? target.frame >= n.range.from && target.frame <= n.range.to : target.frame === n.frame));
}

// The picture's own rectangle inside the viewer, as the video element draws it.
function frameRect() {
  const video = front;
  if (!video.classList.contains("show") || !video.videoWidth) return null;
  const screen = $("screen").getBoundingClientRect();
  const box = video.getBoundingClientRect();
  const scale = Math.min(box.width / video.videoWidth, box.height / video.videoHeight);
  const width = video.videoWidth * scale, height = video.videoHeight * scale;
  return { left: box.left - screen.left + (box.width - width) / 2, top: box.top - screen.top + (box.height - height) / 2, width, height };
}

function drawShape(ctx, shape, rect, scale, faint = false) {
  if (!shape) return;
  const x = (v) => rect.left + v * rect.width, y = (v) => rect.top + v * rect.height;
  ctx.save();
  ctx.lineJoin = ctx.lineCap = "round";
  for (const [color, width] of [["rgba(0, 0, 0, 0.55)", 5 * scale], [NOTE_INK, 2.5 * scale]]) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.globalAlpha = faint ? 0.55 : 1;
    ctx.beginPath();
    if (shape.type === "pin") {
      ctx.arc(x(shape.x), y(shape.y), 11 * scale, 0, Math.PI * 2);
      ctx.moveTo(x(shape.x) + 2 * scale, y(shape.y));
      ctx.arc(x(shape.x), y(shape.y), 2 * scale, 0, Math.PI * 2);
    } else if (shape.type === "rect") {
      ctx.rect(x(shape.x), y(shape.y), shape.w * rect.width, shape.h * rect.height);
    } else {
      for (const path of shape.paths || []) {
        path.forEach(([px_, py], index) => (index ? ctx.lineTo(x(px_), y(py)) : ctx.moveTo(x(px_), y(py))));
      }
    }
    ctx.stroke();
  }
  ctx.restore();
}

// Called with every new frame in the viewer: the notes on this frame, and the one being written.
function drawNotes() {
  const layer = $("noteLayer");
  if (!layer || !state.project) return;
  const screen = $("screen");
  const ratio = window.devicePixelRatio || 1;
  const width = screen.clientWidth, height = screen.clientHeight;
  if (layer.width !== Math.round(width * ratio) || layer.height !== Math.round(height * ratio)) {
    layer.width = Math.round(width * ratio);
    layer.height = Math.round(height * ratio);
  }
  const ctx = layer.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const target = noteTarget();
  const writing = noting.active;
  if (writing && (!target || target.clip.id !== writing.clip || target.frame !== writing.frame)) closeNote();
  const rect = frameRect();
  const here = writing ? [] : notesHere(target);
  if (rect) {
    for (const note of here) drawShape(ctx, note.shape, rect, 1, note.status === "resolved");
    if (noting.active?.shape) drawShape(ctx, noting.active.shape, rect, 1);
  }
  // The box steps aside when the drawing reaches under it.
  const box = $("noteBox");
  const shape = noting.active?.shape;
  if (shape && rect && !box.hidden && !box.classList.contains("moved")) {
    const points = shape.type === "stroke" ? shape.paths.flat() : shape.type === "rect" ? [[shape.x, shape.y], [shape.x + shape.w, shape.y + shape.h]] : [[shape.x, shape.y]];
    const right = rect.left + Math.max(...points.map((p) => p[0])) * rect.width;
    const top = rect.top + Math.min(...points.map((p) => p[1])) * rect.height;
    if (right > screen.clientWidth - box.offsetWidth - 28 && top < box.offsetHeight + 28) box.classList.add("aside");
  } else if (!noting.active) box.classList.remove("aside");
  const tag = $("noteTag");
  tag.hidden = !here.length;
  tag.replaceChildren(...here.slice(0, 3).map((note) => {
    const line = document.createElement("button");
    line.className = "note-line" + (note.status === "resolved" ? " done" : "");
    line.textContent = note.text;
    line.title = (note.status === "resolved" ? "Resolved. " : "") + "Show this note in the Project tab";
    line.addEventListener("click", () => showNoteInList(note.id));
    return line;
  }));
}

function startNote() {
  if (state.playing) setPlaying(false);
  const target = noteTarget();
  if (!target) {
    toast("Stop on a rendered frame to write a note there.");
    return;
  }
  if (!frameRect()) {
    toast("Wait for the picture to load, then press N again.");
    return;
  }
  let range = null;
  const span = state.range;
  if (!viewing() && span && target.item && state.playhead >= span.a && state.playhead < span.b) {
    // A marked range on the timeline becomes a range of the same take, kept inside the clip under the playhead.
    const item = target.item, speed = speedOf(item);
    const take = (frame) => Math.max(item.in, Math.min(item.out - 1, Math.round(item.in + (frame - item.start) * speed)));
    range = { from: take(Math.max(span.a, item.start)), to: take(Math.min(span.b, itemEnd(item)) - 1) };
    if (range.to <= range.from) range = null;
  }
  noting.active = { clip: target.clip.id, take: target.take.id, frame: range ? range.from : target.frame, range, tool: "rect", shape: null, drawing: null };
  if (range) {
    // The note opens on its first frame, so the still shows where the range starts.
    seek(viewing() ? range.from : notePlace({ clip: target.clip.id, frame: range.from }) ?? state.playhead);
    noting.active.frame = noteTarget()?.frame ?? range.from;
  }
  const box = $("noteBox");
  $("noteWhere").textContent = target.clip.name + ", " + (range ? "frames " + range.from + " to " + range.to : "frame " + noting.active.frame) + " of take " + (target.clip.takes.indexOf(target.take) + 1);
  $("noteText").value = "";
  setNoteTool("rect");
  box.classList.remove("aside", "moved");
  box.style.left = box.style.top = "";
  box.hidden = false;
  $("noteLayer").classList.add("drawing");
  $("noteText").focus();
  drawNotes();
}

function setNoteTool(tool) {
  if (!noting.active) return;
  noting.active.tool = tool;
  for (const button of document.querySelectorAll("#noteTools button")) button.setAttribute("aria-pressed", String(button.dataset.tool === tool));
  $("noteHint").textContent = tool === "stroke" ? "Draw on the picture. Each stroke adds to the drawing."
    : tool === "pin" ? "Click the picture to pin the spot." : "Drag a box on the picture, or click to pin a spot.";
}

function closeNote() {
  noting.active = null;
  $("noteBox").hidden = true;
  $("noteLayer").classList.remove("drawing");
}

// The frame as the viewer shows it, with the note's drawing on it, as a PNG.
function noteStill(shape) {
  const video = front;
  if (!video.videoWidth) return Promise.resolve(null);
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  try {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    drawShape(ctx, shape, { left: 0, top: 0, width: canvas.width, height: canvas.height }, Math.max(1, canvas.width / 900));
    return new Promise((resolve) => canvas.toBlob((blob) => resolve(blob), "image/png"));
  } catch (_) {
    return Promise.resolve(null);
  }
}

async function saveNote() {
  const writing = noting.active;
  if (!writing) return;
  const text = $("noteText").value.trim();
  if (!text) {
    $("noteHint").textContent = "Write what should change here, then press Enter.";
    $("noteText").focus();
    return;
  }
  const still = noteStill(writing.shape);
  const note = { id: uid("n"), clip: writing.clip, take: writing.take, frame: writing.frame, range: writing.range,
    shape: writing.shape, text, status: "open", created: new Date().toISOString() };
  checkpoint();
  state.project.notes = [...(state.project.notes || []), note];
  closeNote();
  save();
  renderTimeline();
  renderInspector();
  renderBin();
  drawNotes();
  toast("Note saved. Agents see it in the summary and with h3.py notes.");
  const folder = state.folder;
  try {
    const blob = await still;
    if (!blob) return;
    const asset = await uploadPicture(new File([blob], note.id + ".png", { type: "image/png" }));
    const live = folder === state.folder && (state.project.notes || []).find((n) => n.id === note.id);
    if (live) {
      live.still = { file: asset.file, subfolder: asset.subfolder };
      save();
    }
  } catch (_) {
    // The note stands without its picture. An agent then takes the frame without the drawing.
  }
}

function goToNote(note) {
  noting.selected = note.id;
  const clip = clipOf(note.clip);
  const place = notePlace(note);
  if (place === null) {
    // Its frame is not in the cut: show it in the clip on its own.
    setSource(note.clip);
    selectClip(note.clip);
    seek(note.frame);
  } else {
    if (viewing()) setSource(null);
    const item = sorted().find((i) => i.clip === note.clip && note.frame >= i.in && note.frame < i.out);
    selectClip(clip.id, item?.id || null);
    seek(place);
  }
}

function setNoteStatus(note, open) {
  checkpoint();
  note.status = open ? "open" : "resolved";
  if (open) delete note.resolved;
  else note.resolved = new Date().toISOString();
  save();
  renderTimeline();
  renderInspector();
  renderBin();
  drawNotes();
}

function removeNote(note) {
  checkpoint();
  state.project.notes = (state.project.notes || []).filter((n) => n.id !== note.id);
  save();
  renderTimeline();
  renderInspector();
  renderBin();
  drawNotes();
  toast("Deleted the note.", false, undoAction);
}

function showNoteInList(id) {
  noting.selected = id;
  showTab("project");
  requestAnimationFrame(() => document.querySelector('.note-row[data-note="' + CSS.escape(id) + '"]')?.scrollIntoView({ block: "nearest" }));
}

function notesSection() {
  const notes = notesOf();
  if (!notes.length) return null;
  const open = notes.filter((n) => n.status !== "resolved");
  const order = (n) => [n.status === "resolved" ? 1 : 0, notePlace(n) ?? 1e9];
  const rows = [...notes].sort((a, b) => order(a)[0] - order(b)[0] || order(a)[1] - order(b)[1]).map((note) => {
    const clip = clipOf(note.clip);
    const place = notePlace(note);
    const row = document.createElement("div");
    row.className = "note-row" + (note.status === "resolved" ? " done" : "") + (noting.selected === note.id ? " on" : "");
    row.dataset.note = note.id;
    const jump = document.createElement("button");
    jump.className = "note-jump";
    jump.title = "Go to this frame";
    const where = document.createElement("span");
    where.className = "note-where";
    const older = activeTake(clip)?.id !== note.take;
    where.textContent = clip.name + ", " + (place === null ? "not in the cut" : timecode(place)) + (note.range ? ", a range" : "") + (older ? ", on an earlier take" : "");
    const text = document.createElement("span");
    text.className = "note-text";
    text.textContent = note.text;
    jump.append(where, text);
    if (note.resolution) {
      const answer = document.createElement("span");
      answer.className = "note-answer";
      answer.textContent = note.resolution;
      jump.append(answer);
    }
    jump.addEventListener("click", () => goToNote(note));
    const done = document.createElement("button");
    done.className = "quiet";
    done.textContent = note.status === "resolved" ? "Reopen" : "Resolve";
    done.addEventListener("click", () => setNoteStatus(note, note.status === "resolved"));
    row.append(jump, done, trashButton("Delete this note", () => removeNote(note)));
    return row;
  });
  const hint = document.createElement("p");
  hint.className = "hint";
  hint.textContent = "Stop on a frame and press N to write a note. Agents read them with h3.py notes, each with a picture of its frame.";
  return section("Notes" + (open.length ? ", " + open.length + " open" : ""), [...rows, hint]);
}

// Marks on the ruler where each note's frame is in the cut.
function noteMarks() {
  return notesOf().map((note) => {
    const place = notePlace(note);
    if (place === null) return null;
    const mark = document.createElement("button");
    mark.className = "note-mark" + (note.status === "resolved" ? " done" : "");
    mark.style.left = px(place) + "px";
    if (note.range) {
      const end = notePlace({ clip: note.clip, frame: note.range.to });
      if (end !== null && end > place) mark.style.width = Math.max(6, px(end - place)) + "px";
    }
    mark.title = (note.status === "resolved" ? "Resolved: " : "Note: ") + note.text;
    mark.setAttribute("aria-label", mark.title);
    mark.addEventListener("pointerdown", (event) => event.stopPropagation());
    mark.addEventListener("click", (event) => {
      event.stopPropagation();
      goToNote(note);
    });
    return mark;
  }).filter(Boolean);
}

function bindNoteLayer() {
  const layer = $("noteLayer");
  const point = (event) => {
    const rect = frameRect();
    if (!rect) return null;
    const screen = $("screen").getBoundingClientRect();
    const clamp = (v) => Math.round(Math.max(0, Math.min(1, v)) * 10000) / 10000;
    return [clamp((event.clientX - screen.left - rect.left) / rect.width), clamp((event.clientY - screen.top - rect.top) / rect.height)];
  };
  layer.addEventListener("pointerdown", (event) => {
    const writing = noting.active;
    const at = writing && point(event);
    if (!at) return;
    event.preventDefault();
    layer.setPointerCapture(event.pointerId);
    writing.drawing = { start: at, path: [at] };
    if (writing.tool === "stroke") {
      const paths = writing.shape?.type === "stroke" ? writing.shape.paths : [];
      writing.shape = { type: "stroke", paths: [...paths, writing.drawing.path] };
    } else writing.shape = { type: "pin", x: at[0], y: at[1] };
    drawNotes();
  });
  layer.addEventListener("pointermove", (event) => {
    const writing = noting.active;
    if (!writing?.drawing) return;
    const at = point(event);
    if (!at) return;
    const [x0, y0] = writing.drawing.start;
    if (writing.tool === "stroke") {
      const last = writing.drawing.path.at(-1);
      if (Math.hypot(at[0] - last[0], at[1] - last[1]) > 0.002 && writing.drawing.path.length < 1900) writing.drawing.path.push(at);
    } else if (writing.tool === "rect" && Math.hypot(at[0] - x0, at[1] - y0) > 0.01) {
      writing.shape = { type: "rect", x: Math.min(x0, at[0]), y: Math.min(y0, at[1]), w: Math.abs(at[0] - x0), h: Math.abs(at[1] - y0) };
    }
    drawNotes();
  });
  const finish = () => {
    if (!noting.active?.drawing) return;
    noting.active.drawing = null;
    if (noting.active.shape?.type === "stroke") noting.active.shape.paths = noting.active.shape.paths.filter((p) => p.length);
    $("noteText").focus();
    drawNotes();
  };
  layer.addEventListener("pointerup", finish);
  layer.addEventListener("pointercancel", finish);
  for (const button of document.querySelectorAll("#noteTools button")) button.addEventListener("click", () => setNoteTool(button.dataset.tool));
  $("noteClear").addEventListener("click", () => {
    if (!noting.active) return;
    noting.active.shape = null;
    drawNotes();
    $("noteText").focus();
  });
  $("noteSave").addEventListener("click", saveNote);
  $("noteCancel").addEventListener("click", () => {
    closeNote();
    drawNotes();
  });
  $("noteText").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      saveNote();
    } else if (event.key === "Escape") {
      event.preventDefault();
      closeNote();
      drawNotes();
    }
  });
  $("addNote").addEventListener("click", startNote);
  // The box moves by its header, off whatever part of the picture it covers.
  $("noteWhere").addEventListener("pointerdown", (event) => {
    const box = $("noteBox");
    const screen = $("screen");
    const from = { x: event.clientX, y: event.clientY, left: box.offsetLeft, top: box.offsetTop };
    event.preventDefault();
    $("noteWhere").setPointerCapture(event.pointerId);
    const move = (moved) => {
      const left = Math.max(0, Math.min(screen.clientWidth - box.offsetWidth, from.left + moved.clientX - from.x));
      const top = Math.max(0, Math.min(screen.clientHeight - box.offsetHeight, from.top + moved.clientY - from.y));
      box.classList.add("moved");
      box.style.left = left + "px";
      box.style.top = top + "px";
    };
    const stop = () => {
      $("noteWhere").removeEventListener("pointermove", move);
      $("noteWhere").removeEventListener("pointerup", stop);
      $("noteText").focus();
    };
    $("noteWhere").addEventListener("pointermove", move);
    $("noteWhere").addEventListener("pointerup", stop);
  });
  new ResizeObserver(() => drawNotes()).observe($("screen"));
  for (const video of [$("videoA"), $("videoB")]) video.addEventListener("seeked", () => drawNotes());
}
bindNoteLayer();

start().catch((error) => toast(error.message, true));
