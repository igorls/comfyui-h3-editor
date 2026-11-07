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
  playhead: 0,
  playing: false,
  zoom: 3,
  jobs: new Map(),
  running: null,
};
let undo = [];
let redo = [];
let saveTimer = 0;

const px = (frames) => frames * state.zoom;
const uid = (prefix) => prefix + Math.random().toString(36).slice(2, 8);
const clipOf = (id) => state.project.clips.find((c) => c.id === id);
const itemLength = (item) => item.out - item.in;
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
  if (!response.ok) throw new Error((await response.text()) || "The request failed.");
  return response.json();
}

const postJson = (path, body, method = "POST") =>
  api(path, { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function save() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    postJson("/project/" + state.folder, state.project, "PUT").catch((e) => toast(e.message, true));
  }, 300);
}

// Call before a change that undo should be able to reverse.
function checkpoint() {
  undo.push(JSON.stringify(state.project));
  if (undo.length > 100) undo.shift();
  redo = [];
}

function restore(from, to) {
  if (!from.length) return;
  to.push(JSON.stringify(state.project));
  state.project = JSON.parse(from.pop());
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
    if (item.start < end) item.start = end;
    end = itemEnd(item);
  }
}

function addClip() {
  checkpoint();
  const number = state.project.clips.length + 1;
  const clip = { id: uid("c"), name: "Clip " + number, action: "", camera: "", speaker: "", dialogue: "", sound: "", music: "", seconds: 5, takes: [], active: null };
  state.project.clips.push(clip);
  placeOnTimeline(clip, sceneEnd());
  state.clip = clip.id;
  state.item = state.project.timeline.at(-1).id;
  save();
  renderAll();
  document.querySelector("#inspector textarea.big")?.focus();
}

function placeOnTimeline(clip, start) {
  const take = activeTake(clip);
  const frames = take ? take.frames : clip.link ? linkedFrames(clip.link, clip.seconds) : Math.round(clip.seconds * FPS);
  const [first, last] = linkedRange(clip.link, frames);
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

function isStale(clip) {
  return clip.takes.length > 0 && linkSides(clip.link).some(sideChanged);
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
  state.item = placed.id;
  save();
  renderAll();
  document.querySelector("#inspector textarea.big")?.focus();
}

function selectClip(clipId, itemId = null) {
  state.clip = clipId;
  state.item = itemId;
  renderBin();
  renderInspector();
  renderTimeline();
  showFrame();
}

function renderBin() {
  const list = $("clips");
  if (!state.project.clips.length) {
    list.innerHTML = '<p class="empty">Add a clip to start your scene.</p>';
    return;
  }
  list.replaceChildren(...state.project.clips.map((clip) => {
    const take = activeTake(clip);
    const button = document.createElement("button");
    button.className = "clip" + (clip.id === state.clip ? " on" : "");
    button.draggable = true;
    button.addEventListener("click", () => selectClip(clip.id));
    button.addEventListener("dragstart", (event) => event.dataTransfer.setData("text/h3clip", clip.id));
    if (take) {
      const img = document.createElement("img");
      img.className = "thumb";
      img.alt = "";
      img.src = viewUrl(take.poster, take.subfolder);
      button.append(img);
    } else {
      const none = document.createElement("span");
      none.className = "thumb none";
      none.textContent = jobFor(clip.id) ? "Rendering" : "Not rendered";
      button.append(none);
    }
    const meta = document.createElement("span");
    meta.className = "meta";
    const name = document.createElement("span");
    name.textContent = clip.name;
    const length = document.createElement("span");
    length.textContent = (take ? take.frames / FPS : clip.seconds).toFixed(1) + "s";
    meta.append(name, length);
    button.append(meta);
    return button;
  }));
}

function field(label, value, onInput, { big = false, area = false, type = "text" } = {}) {
  const wrap = document.createElement("label");
  wrap.className = "field";
  const title = document.createElement("span");
  title.textContent = label;
  const input = document.createElement(area || big ? "textarea" : "input");
  if (big) input.className = "big";
  if (!area && !big) input.type = type;
  input.value = value ?? "";
  input.addEventListener("focus", () => checkpoint(), { once: true });
  input.addEventListener("input", () => {
    onInput(input.value);
    save();
  });
  wrap.append(title, input);
  return wrap;
}

function renderInspector() {
  const panel = $("inspector");
  const clip = clipOf(state.clip);
  if (!clip) {
    const head = document.createElement("div");
    head.className = "panel-head";
    head.innerHTML = "<h2>Scene</h2>";
    panel.replaceChildren(
      head,
      field("Style for every clip", state.project.style, (v) => (state.project.style = v), { area: true }),
    );
    return;
  }
  const head = document.createElement("div");
  head.className = "panel-head";
  head.innerHTML = "<h2></h2>";
  head.firstChild.textContent = clip.name;

  const two = document.createElement("div");
  two.className = "two";
  two.append(
    field("Name", clip.name, (v) => {
      clip.name = v;
      renderBin();
      renderTimeline();
    }),
    field("Seconds", clip.seconds, (v) => {
      if (Number(v) > 0) clip.seconds = Number(v);
    }, { type: "number" }),
  );

  const actions = document.createElement("div");
  actions.className = "actions";
  const job = jobFor(clip.id);
  if (job) {
    const cancel = document.createElement("button");
    cancel.className = "quiet";
    cancel.textContent = "Cancel render";
    cancel.addEventListener("click", () => cancelJob(job));
    actions.append(cancel);
  } else {
    for (const [quality, label, style] of [["draft", "Render draft", "quiet"], ["final", "Render final", "primary"]]) {
      const button = document.createElement("button");
      button.className = style;
      button.textContent = label;
      button.addEventListener("click", () => renderClip(clip, quality));
      actions.append(button);
    }
  }

  const stale = isStale(clip) && !job;
  const notice = document.createElement("p");
  notice.className = "notice";
  notice.textContent = "The clip this one joins has changed, so the join may show. Render it again to fix that.";
  const children = [
    head,
    ...(stale ? [notice] : []),
    two,
    field("What happens", clip.action, (v) => (clip.action = v), { big: true }),
    field("Camera", clip.camera, (v) => (clip.camera = v)),
    field("Who speaks", clip.speaker, (v) => (clip.speaker = v)),
    field("What they say", clip.dialogue, (v) => (clip.dialogue = v), { area: true }),
    field("Sound", clip.sound, (v) => (clip.sound = v), { area: true }),
    field("Music", clip.music, (v) => (clip.music = v)),
    actions,
  ];
  if (clip.takes.length) {
    const title = document.createElement("h3");
    title.textContent = "Takes";
    const takes = document.createElement("div");
    takes.className = "takes";
    clip.takes.forEach((take, index) => {
      const button = document.createElement("button");
      button.className = "take" + (take === activeTake(clip) ? " on" : "");
      const img = document.createElement("img");
      img.alt = "";
      img.src = viewUrl(take.poster, take.subfolder);
      const label = document.createElement("span");
      label.textContent = "Take " + (index + 1) + ", " + take.quality;
      button.append(img, label);
      button.addEventListener("click", () => {
        checkpoint();
        clip.active = take.id;
        for (const item of state.project.timeline) {
          if (item.clip === clip.id) item.out = Math.min(item.out, take.frames);
        }
        save();
        renderAll();
      });
      takes.append(button);
    });
    children.push(title, takes);
  }
  panel.replaceChildren(...children);
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
  }
  ruler.replaceChildren(...ticks);
}

function renderTimeline() {
  const width = Math.max($("tlScroll").clientWidth, px(sceneEnd() + FPS * 20));
  $("tlContent").style.width = width + "px";
  renderRuler(width);
  const blocks = [];
  const waves = [];
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
    const strip = document.createElement("div");
    strip.className = "strip";
    if (job?.preview) strip.style.backgroundImage = "url(" + job.preview + ")";
    else if (take) strip.style.backgroundImage = "url(" + viewUrl(take.poster, take.subfolder) + ")";
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = status === "stale" ? clip.name + ", source changed" : clip.name;
    block.append(strip, name);
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
    waves.push(wave);
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
      if (edge === "l") {
        const shift = Math.max(-origin.in, Math.min(delta, origin.out - origin.in - 6));
        item.in = origin.in + shift;
        item.start = origin.start + shift;
      } else if (edge === "r") {
        item.out = Math.min(limit, Math.max(origin.in + 6, origin.out + delta));
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
  const cut = item.in + (state.playhead - item.start);
  state.project.timeline.push({ id: uid("i"), clip: item.clip, start: state.playhead, in: cut, out: item.out });
  item.out = cut;
  save();
  renderTimeline();
}

function removeSelected() {
  if (!state.item) {
    toast("Select a clip on the timeline first.");
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
  }
}

function showLive(job) {
  const live = $("live");
  const image = $("preview");
  const showing = !!job && !state.playing;
  live.hidden = !showing;
  image.hidden = !(showing && job.preview);
  if (!showing) return;
  if (job.preview) image.src = job.preview;
  live.textContent = (clipOf(job.clip)?.name || "Clip") + ": " + (job.stage || "Waiting in the queue") + (job.max ? ", step " + job.step + " of " + job.max : "");
}

function showFrame() {
  const item = itemAt(state.playhead);
  const url = sourceOf(item);
  const job = state.running || jobFor(item?.clip || state.clip);
  const slate = $("slate");
  if (url) {
    load(front, url);
    const seek = (item.in + state.playhead - item.start) / FPS;
    if (Math.abs(front.currentTime - seek) > 0.01) front.currentTime = seek;
    front.classList.add("show");
    back.classList.remove("show");
    slate.textContent = "";
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
}

function setPlaying(on) {
  state.playing = on;
  $("play").textContent = on ? "❚❚" : "▶";
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
  const item = itemAt(state.playhead);
  if (item && sourceOf(item)) front.play().catch(() => {});
  requestAnimationFrame(tick);
}

function tick(now) {
  if (!state.playing) return;
  const item = itemAt(state.playhead);
  const url = sourceOf(item);
  if (url && !front.paused && front.dataset.url === url) {
    state.playhead = item.start + front.currentTime * FPS - item.in;
  } else {
    state.playhead += ((now - lastTick) / 1000) * FPS;
  }
  lastTick = now;

  const following = item ? nextItem(item.start) : nextItem(state.playhead);
  if (following && sourceOf(following)) {
    load(back, sourceOf(following));
    if (back.paused) back.currentTime = following.in / FPS;
  }
  const current = itemAt(state.playhead);
  if (current !== item) {
    if (current && sourceOf(current)) {
      if (back.dataset.url === sourceOf(current)) swap();
      else load(front, sourceOf(current));
      front.currentTime = (current.in + state.playhead - current.start) / FPS;
      front.classList.add("show");
      front.play().catch(() => {});
      back.pause();
    } else {
      front.pause();
      front.classList.remove("show");
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
  $("queue").textContent = count ? "Rendering " + count + (count === 1 ? " clip" : " clips") : "";
}

async function renderClip(clip, quality) {
  if (jobFor(clip.id)) return;
  if (!clip.action.trim()) {
    selectClip(clip.id);
    toast("Describe what happens in " + clip.name + " before rendering it.", true);
    return;
  }
  refreshLink(clip);
  const take = uid("t");
  const seed = Math.floor(Math.random() * 2 ** 31);
  try {
    const { graph } = await postJson("/project/" + state.folder + "/graph", { project: state.project, clip, take, quality, seed });
    const response = await fetch("/prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: graph, client_id: clientId }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error?.message || "ComfyUI refused the render.");
    const job = { id: body.prompt_id, clip: clip.id, take, quality, seed };
    state.project.pending = [...(state.project.pending || []), job];
    state.jobs.set(job.id, { ...job, step: 0, max: 0, progress: 0 });
    save();
    updateQueue();
    renderAll();
  } catch (error) {
    toast(error.message, true);
  }
}

async function cancelJob(job) {
  if (state.running === job) await fetch("/interrupt", { method: "POST" });
  else await fetch("/queue", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ delete: [job.id] }) });
  finishJob(job);
  toast("Render cancelled.");
}

function finishJob(job) {
  state.jobs.delete(job.id);
  state.project.pending = (state.project.pending || []).filter((p) => p.id !== job.id);
  save();
  if (state.running === job) state.running = null;
  if (job.preview) URL.revokeObjectURL(job.preview);
  updateQueue();
  renderAll();
}

async function acceptRender(job, file) {
  if (job.accepting) return;
  job.accepting = true;
  const clip = clipOf(job.clip);
  if (!clip) return finishJob(job);
  const info = await postJson("/probe", file);
  const take = { id: job.take, file: file.filename, subfolder: file.subfolder, poster: info.poster, frames: info.frames, quality: job.quality, seed: job.seed };
  const fresh = !clip.takes.length;
  clip.takes.push(take);
  clip.active = take.id;
  const [first, last] = linkedRange(clip.link, info.frames);
  for (const item of state.project.timeline) {
    if (item.clip !== clip.id) continue;
    item.in = fresh ? first : item.in;
    item.out = fresh ? last : Math.min(item.out, last);
  }
  settle(null);
  save();
  finishJob(job);
  toast(clip.name + " is ready.");
}

function savedFile(history) {
  const groups = Object.values(history.outputs?.["19"] || {}).filter(Array.isArray).flat();
  return groups.find((g) => g && g.filename);
}

// Renders belong to the project, not this window. Picks them up after a refresh or a missed message.
async function checkJob(job) {
  const history = (await (await fetch("/history/" + job.id)).json())[job.id];
  if (!history) return false;
  const file = savedFile(history);
  if (file) await acceptRender(job, { filename: file.filename, subfolder: file.subfolder });
  else finishJob(job);
  return true;
}

async function resumeJobs() {
  const queue = await (await fetch("/queue")).json();
  const active = new Set([...queue.queue_running, ...queue.queue_pending].map((entry) => entry[1]));
  for (const pending of state.project.pending || []) {
    const job = { ...pending, step: 0, max: 0, progress: 0 };
    state.jobs.set(job.id, job);
    if (!active.has(job.id) && !(await checkJob(job))) finishJob(job);
  }
  updateQueue();
  renderAll();
}

setInterval(() => {
  for (const job of state.jobs.values()) checkJob(job).catch(() => {});
}, 4000);

function connect() {
  const socket = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws?clientId=" + clientId);
  socket.binaryType = "arraybuffer";
  socket.onclose = () => setTimeout(connect, 2000);
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
  const strip = document.querySelector('.block.rendering .strip');
  if (strip) strip.style.backgroundImage = "url(" + job.preview + ")";
  showLive(job);
}

function onMessage({ type, data }) {
  const job = data?.prompt_id ? state.jobs.get(data.prompt_id) : null;
  if (!job) return;
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
    showLive(job);
  } else if (type === "executed" && String(data.node) === "19") {
    const groups = Object.values(data.output || {}).filter(Array.isArray).flat();
    const file = groups.find((g) => g && g.filename);
    if (file) acceptRender(job, { filename: file.filename, subfolder: file.subfolder }).catch((e) => toast(e.message, true));
  } else if (type === "execution_error") {
    toast("The render failed: " + (data.exception_message || "see the ComfyUI log"), true);
    finishJob(job);
  } else if (type === "execution_interrupted") {
    finishJob(job);
  }
}

/* Project */

function renderAll() {
  renderBin();
  renderInspector();
  renderTimeline();
  showFrame();
}

async function openProject(folder) {
  state.folder = folder;
  state.project = await api("/project/" + folder);
  state.clip = null;
  state.item = null;
  state.playhead = 0;
  undo = [];
  redo = [];
  try { localStorage.setItem("h3take.project", folder); } catch (_) {}
  state.jobs.clear();
  state.running = null;
  renderAll();
  await resumeJobs();
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
$("close").addEventListener("click", () => window.parent.postMessage("h3-editor-close", location.origin));
$("renderAll").addEventListener("click", () => {
  const waiting = sorted().map((i) => clipOf(i.clip)).filter((c, i, all) => c && !c.takes.length && all.indexOf(c) === i);
  if (!waiting.length) return toast("Every clip on the timeline already has a render.");
  waiting.forEach((clip) => renderClip(clip, "draft"));
});
$("export").addEventListener("click", async () => {
  clearTimeout(saveTimer);
  $("export").disabled = true;
  try {
    await postJson("/project/" + state.folder, state.project, "PUT");
    const result = await postJson("/project/" + state.folder + "/export", {});
    toast("Exported " + result.filename + " to " + result.subfolder);
  } catch (error) {
    toast(error.message, true);
  }
  $("export").disabled = false;
});
$("play").addEventListener("click", () => setPlaying(!state.playing));
$("toStart").addEventListener("click", () => seek(0));
$("back").addEventListener("click", () => seek(state.playhead - 1));
$("forward").addEventListener("click", () => seek(state.playhead + 1));
$("split").addEventListener("click", splitAtPlayhead);
$("extendBack").addEventListener("click", () => addLinked("extend_before"));
$("extendForward").addEventListener("click", () => addLinked("extend_after"));
$("bridge").addEventListener("click", () => addLinked("bridge"));
$("remove").addEventListener("click", removeSelected);
$("zoom").addEventListener("input", (event) => {
  state.zoom = Number(event.target.value);
  renderTimeline();
});

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
  if (event.target === lane) selectClip(null);
});

window.addEventListener("keydown", (event) => {
  if (/INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) return;
  if (event.code === "Space") {
    event.preventDefault();
    setPlaying(!state.playing);
  } else if (event.key === "ArrowLeft") seek(state.playhead - (event.shiftKey ? FPS : 1));
  else if (event.key === "ArrowRight") seek(state.playhead + (event.shiftKey ? FPS : 1));
  else if (event.key === "Delete" || event.key === "Backspace") removeSelected();
  else if (event.key.toLowerCase() === "s" && !event.ctrlKey && !event.metaKey) splitAtPlayhead();
  else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    if (event.shiftKey) restore(redo, undo);
    else restore(undo, redo);
  }
});
window.addEventListener("resize", () => state.project && renderTimeline());

start().catch((error) => toast(error.message, true));
