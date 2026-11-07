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
};
let undo = [];
let redo = [];
let saveTimer = 0;

const px = (frames) => frames * state.zoom;
const uid = (prefix) => prefix + Math.random().toString(36).slice(2, 8);
const clipOf = (id) => state.project.clips.find((c) => c.id === id);
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

function toast(message, bad = false) {
  const el = $("toast");
  el.textContent = message;
  el.classList.toggle("bad", bad);
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (el.hidden = true), bad ? 7000 : 3500);
}

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
  state.project = project;
  syncJobs();
  if (same) return;
  if (!clipOf(state.clip)) state.clip = null;
  if (!state.project.timeline.some((i) => i.id === state.item)) state.item = null;
  if (state.person && !(state.project.cast || []).some((p) => p.id === state.person)) state.person = null;
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
  state.project.clips.push(clip);
  placeOnTimeline(clip, sceneEnd());
  state.clip = clip.id;
  state.person = null;
  state.item = state.project.timeline.at(-1).id;
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

function updateTools() {
  const item = selectedRendered();
  const next = item && followingItem(item);
  const holdable = item && itemLength(item) >= HOLD;
  $("extendBack").disabled = !holdable;
  $("extendForward").disabled = !holdable;
  $("bridge").disabled = !(holdable && next && activeTake(clipOf(next.clip)) && itemLength(next) >= HOLD);
  $("merge").disabled = !(item && next && activeTake(clipOf(next.clip)) && next.start === itemEnd(item));
  $("ripple").disabled = !state.item;
  $("closeGap").disabled = !state.gap;
}

function shiftFrom(frame, by, skip = null) {
  for (const other of state.project.timeline) {
    if (other !== skip && other.start >= frame) other.start += by;
  }
}

function addLinked(mode) {
  const item = selectedRendered();
  if (!item) return;
  const source = clipOf(item.clip);
  const next = followingItem(item);
  const seconds = 4.25;
  const link = { mode, hold: HOLD, a: sideOf(item) };
  if (mode === "bridge") link.b = sideOf(next);
  checkpoint();
  const titles = { extend_after: "After ", extend_before: "Before ", bridge: "Bridge from " };
  const clip = {
    id: uid("c"), name: titles[mode] + source.name, action: "", camera: "", speaker: "", dialogue: "",
    sound: "", music: "", seconds, takes: [], active: null, link,
  };
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
  const item = itemAt(state.playhead);
  const take = activeTake(clipOf(item?.clip));
  if (!take) {
    toast("Move the playhead over a rendered clip, then press Use playhead frame.");
    return;
  }
  try {
    const frame = Math.min(take.frames - 1, Math.round(item.in + (state.playhead - item.start) * speedOf(item)));
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
  const pictured = state.project.cast.filter((p) => p.file);
  for (const person of pictured) {
    const on = (clip.cast || []).includes(person.id);
    const pick = document.createElement("button");
    pick.className = "pick" + (on ? " on" : "");
    pick.setAttribute("aria-pressed", String(on));
    const image = document.createElement("img");
    image.alt = "";
    image.src = assetUrl(person);
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
    } else {
      image = document.createElement("span");
      image.className = "voice-mark";
    }
    const name = document.createElement("span");
    name.textContent = person.name;
    button.append(image, name);
    button.addEventListener("click", () => {
      state.person = person.id;
      state.clip = null;
      state.item = null;
      renderAll();
    });
    return button;
  }));
}

function renderPerson(panel, person) {
  const head = document.createElement("div");
  head.className = "panel-head";
  const title = document.createElement("h2");
  title.textContent = person.name;
  const remove = document.createElement("button");
  remove.className = "quiet";
  remove.textContent = "Remove";
  remove.addEventListener("click", () => {
    checkpoint();
    state.project.cast = state.project.cast.filter((c) => c !== person);
    for (const clip of state.project.clips) clip.cast = (clip.cast || []).filter((id) => id !== person.id);
    state.person = null;
    save();
    renderAll();
  });
  head.append(title, remove);
  const parts = [head];
  if (person.file) {
    const image = document.createElement("img");
    image.className = "portrait";
    image.alt = person.name;
    image.src = assetUrl(person);
    parts.push(image);
  }
  parts.push(
    field("Name", person.name, (v) => {
      person.name = v.trim() || person.name;
      title.textContent = person.name;
      renderCast();
    }),
    person.file
      ? field("What they look like", person.description, (v) => (person.description = v), { big: true })
      : field("How they sound", person.description, (v) => (person.description = v), { area: true, placeholder: "A warm, calm female narrator" }),
    section("Voice", [voiceSection(person)]),
  );
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
  state.clip = clipId;
  state.item = itemId;
  state.gap = null;
  state.person = null;
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
    const index = Math.floor(share * take.strip.count);
    thumb.style.backgroundImage = sprite;
    thumb.style.backgroundSize = take.strip.count * 100 + "% 100%";
    thumb.style.backgroundPosition = (take.strip.count > 1 ? (index / (take.strip.count - 1)) * 100 : 0) + "% 0";
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

function renderBin() {
  const list = $("clips");
  $("clipCount").textContent = state.project.clips.length ? String(state.project.clips.length) : "";
  if (!state.project.clips.length) {
    list.innerHTML = '<p class="empty">Add a clip, describe the shot, and render a draft.</p>';
    return;
  }
  list.replaceChildren(...state.project.clips.map((clip) => {
    const take = activeTake(clip);
    const status = clipStatus(clip);
    const button = document.createElement("button");
    button.className = "clip" + (clip.id === state.clip ? " on" : "");
    button.draggable = true;
    button.addEventListener("click", () => selectClip(clip.id));
    button.addEventListener("dragstart", (event) => event.dataTransfer.setData("text/h3clip", clip.id));
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
  }));
  list.querySelector(".clip.on")?.scrollIntoView({ block: "nearest" });
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

function renderScene(panel) {
  const head = document.createElement("div");
  head.className = "inspector-head";
  const title = document.createElement("h2");
  title.textContent = "Scene";
  const line = document.createElement("p");
  line.className = "status";
  line.textContent = "Select a clip to edit it. Press ? for shortcuts.";
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
  panel.replaceChildren(
    head,
    section("Look", [field("Style for every clip", state.project.style, (v) => (state.project.style = v), { area: true })]),
    section("Project", [
      shape,
      formatHint,
      field("Language people speak", state.project.language || "", (v) => {
        if (v.trim()) state.project.language = v.trim();
        else delete state.project.language;
      }, { placeholder: "English" }),
    ]),
    section("Overview", [facts]),
  );
}

function renderInspector() {
  const panel = $("inspector");
  const clip = clipOf(state.clip);
  const person = (state.project.cast || []).find((c) => c.id === state.person);
  if (person) {
    renderPerson(panel, person);
    return;
  }
  if (!clip) {
    renderScene(panel);
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
  head.append(name, line);

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
    parts.push(section("Footage", [about]));
    const placed = state.project.timeline.find((i) => i.id === state.item);
    if (placed) parts.push(section("On the timeline", timelineControls(placed)));
    panel.replaceChildren(...parts);
    return;
  }

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
  parts.push(section("Shot", [
    field("What happens", clip.action, (v) => (clip.action = v), { big: true, placeholder: "Describe what we see and hear, in the order it happens." }),
    timing,
    ...framesFor(clip),
    ...(state.project.cast?.some((p) => p.file) ? castPicker(clip) : []),
  ]));

  const voiced = ["speaker", "dialogue", "sound", "music"].filter((k) => String(clip[k] || "").trim()).length;
  parts.push(disclosure("Dialogue and sound", voiced > 0, voiced ? voiced + " set" : "", [
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

  const seed = field("Seed. Leave empty for a new one each render", clip.seed ?? "", (v) => {
    clip.seed = v.trim() === "" ? null : Number(v);
  }, { type: "number" });
  if (clip.takes.length) {
    parts.push(section("Takes", [...versions(clip), seed]));
  } else {
    const none = document.createElement("p");
    none.className = "hint";
    none.textContent = "No takes yet. Render a draft to make the first one.";
    parts.push(section("Takes", [none, seed]));
  }

  const item = state.project.timeline.find((i) => i.id === state.item);
  if (item) parts.push(section("On the timeline", timelineControls(item)));

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

const SAVED_FIELDS = ["action", "camera", "speaker", "dialogue", "sound", "music", "seconds", "cast"];

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
    clip.takes = clip.takes.filter((t) => t !== current);
    clip.active = clip.takes.at(-1)?.id ?? null;
    save();
    renderAll();
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
    wave.style.left = block.style.left;
    wave.style.width = block.style.width;
    if (take) {
      const canvas = document.createElement("canvas");
      wave.append(canvas);
      drawWave(canvas, item, take, waveHeight, item.id === state.item);
    }
    waves.push(wave);
  }
  if (state.gap) {
    const mark = document.createElement("div");
    mark.className = "gapmark";
    mark.style.left = px(state.gap.start) + "px";
    mark.style.width = px(state.gap.end - state.gap.start) + "px";
    blocks.push(mark);
  }
  $("laneV").replaceChildren(...blocks);
  $("laneA").replaceChildren(...waves);
  $("total").textContent = "of " + timecode(sceneEnd());
  placePlayhead();
  updateTools();
}

function placePlayhead() {
  $("playhead").style.left = px(state.playhead) + "px";
  $("time").textContent = timecode(state.playhead);
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

function clearMix() {
  for (const video of [front, back]) {
    video.style.opacity = "";
    video.style.clipPath = "";
    video.volume = 1;
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
  video.playbackRate = speedOf(item);
  const seconds = sourceSeconds(item, frame);
  if (Math.abs(video.currentTime - seconds) > tolerance) video.currentTime = seconds;
}

function showFrame() {
  const item = leading(state.playhead);
  const url = sourceOf(item);
  const job = state.running || jobFor(item?.clip || state.clip);
  const slate = $("slate");
  clearMix();
  if (url) {
    place(front, item, state.playhead);
    front.classList.add("show");
    back.classList.remove("show");
    slate.textContent = "";
    const coming = incoming(state.playhead);
    if (coming && coming !== item && sourceOf(coming)) {
      place(back, coming, state.playhead);
      applyMix(coming, state.playhead);
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
    state.playhead = item.start + (front.currentTime * FPS - item.in) / speedOf(item);
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
      back.currentTime = following.in / FPS;
    }
  }

  const current = leading(state.playhead);
  if (current !== item) {
    if (current && sourceOf(current)) {
      if (back.dataset.url === sourceOf(current)) swap();
      place(front, current, state.playhead, 0.08);
      front.classList.add("show");
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

function seek(frame) {
  state.playhead = Math.max(0, Math.round(frame));
  if (state.playing) setPlaying(false);
  showFrame();
}

/* Rendering */

function updateQueue() {
  const count = state.jobs.size;
  $("queue").textContent = count ? (count === 1 ? "Rendering 1 clip" : "Rendering " + count + " clips") : "";
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
  state.playhead = 0;
  undo = [];
  redo = [];
  try { localStorage.setItem("h3take.project", folder); } catch (_) {}
  state.running = null;
  syncJobs();
  renderAll();
  ensureStrips();
}

async function refreshProjects(selected) {
  const { projects } = await api("/projects");
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
$("newProject").addEventListener("click", () => {
  const name = prompt("Project name");
  if (name) createProject(name).catch((e) => toast(e.message, true));
});
$("newClip").addEventListener("click", addClip);
$("addCast").addEventListener("click", addCastMember);
$("addVoice").addEventListener("click", addVoiceMember);
$("importClip").addEventListener("click", importFootage);
$("close").addEventListener("click", () => window.parent.postMessage("h3-editor-close", location.origin));
$("renderAll").addEventListener("click", () => {
  const waiting = sorted().map((i) => clipOf(i.clip)).filter((c, i, all) => c && !c.takes.length && all.indexOf(c) === i);
  if (!waiting.length) return toast("Every clip on the timeline already has a render.");
  (async () => {
    for (const clip of waiting) await renderClip(clip, "draft");
  })();
});
$("export").addEventListener("click", async () => {
  clearTimeout(saveTimer);
  $("export").disabled = true;
  try {
    await flush();
    const result = await postJson("/project/" + state.folder + "/export", {});
    toast("Exported " + result.filename + " to " + result.subfolder);
  } catch (error) {
    toast(error.message, true);
  }
  $("export").disabled = false;
});
$("play").addEventListener("click", () => setPlaying(!state.playing));
$("shortcuts").addEventListener("click", () => $("keys").showModal());
$("keysClose").addEventListener("click", () => $("keys").close());
bindSplitter();
$("toStart").addEventListener("click", () => seek(0));
$("back").addEventListener("click", () => seek(state.playhead - 1));
$("forward").addEventListener("click", () => seek(state.playhead + 1));
$("split").addEventListener("click", splitAtPlayhead);
$("extendBack").addEventListener("click", () => addLinked("extend_before"));
$("extendForward").addEventListener("click", () => addLinked("extend_after"));
$("bridge").addEventListener("click", () => addLinked("bridge"));
$("merge").addEventListener("click", mergeWithNext);
$("ripple").addEventListener("click", rippleDelete);
$("closeGap").addEventListener("click", closeGap);
$("fit").addEventListener("click", fitTimeline);
$("remove").addEventListener("click", removeSelected);
$("zoom").addEventListener("input", (event) => setZoom(Number(event.target.value)));
$("tlScroll").addEventListener("wheel", (event) => {
  if (!event.ctrlKey && !event.metaKey) return;
  event.preventDefault();
  const anchor = event.clientX - $("tlScroll").getBoundingClientRect().left;
  setZoom(state.zoom * (event.deltaY < 0 ? 1.15 : 1 / 1.15), anchor);
}, { passive: false });

$("ruler").addEventListener("pointerdown", (event) => {
  const frameAt = (ev) => (ev.clientX - $("ruler").getBoundingClientRect().left) / state.zoom;
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
  if ($("keys").open) return;
  if (event.code === "Space") {
    event.preventDefault();
    setPlaying(!state.playing);
  } else if (event.key === "?") $("keys").showModal();
  else if (event.key.toLowerCase() === "k" && !event.ctrlKey && !event.metaKey) setPlaying(false);
  else if (event.key.toLowerCase() === "l" && !event.ctrlKey && !event.metaKey) {
    if (!state.playing) setPlaying(true);
  } else if (event.key.toLowerCase() === "j" && !event.ctrlKey && !event.metaKey) seek(state.playhead - FPS); else if (event.key === "ArrowLeft") seek(state.playhead - (event.shiftKey ? FPS : 1));
  else if (event.key === "ArrowRight") seek(state.playhead + (event.shiftKey ? FPS : 1));
  else if (event.key === "ArrowUp") {
    event.preventDefault();
    jumpEdit(-1);
  } else if (event.key === "ArrowDown") {
    event.preventDefault();
    jumpEdit(1);
  } else if (event.key === "Home") seek(0);
  else if (event.key === "End") seek(sceneEnd());
  else if (event.key === "[") cycleTake(-1);
  else if (event.key === "]") cycleTake(1);
  else if (event.key === "\\") fitTimeline();
  else if (event.key === "Delete" || event.key === "Backspace") {
    if (event.shiftKey) rippleDelete();
    else removeSelected();
  }
  else if (event.key.toLowerCase() === "s" && !event.ctrlKey && !event.metaKey) splitAtPlayhead();
  else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    if (event.shiftKey) restore(redo, undo);
    else restore(undo, redo);
  }
});
window.addEventListener("resize", () => state.project && renderTimeline());

start().catch((error) => toast(error.message, true));
