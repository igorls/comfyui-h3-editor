// Exercise the real editor functions with deterministic media clocks and a fake server.
// The editor has no bundler; extracting top-level functions keeps these tests dependency-free.
import { readFileSync } from "node:fs";
import vm from "node:vm";
import test from "node:test";
import assert from "node:assert/strict";
import { curveLevel, soundEnvelope, setSoundLevel, resumeSoundMixer, disconnectSound } from "../editor/sound-envelope.js";

const source = readFileSync(new URL("../editor/editor.js", import.meta.url), "utf8");
function functions(context, names) {
  for (const name of names) {
    const start = source.search(new RegExp(`^(?:async )?function ${name}\\(`, "m"));
    assert.notEqual(start, -1, name);
    const end = source.indexOf("\n}", start) + 2;
    vm.runInContext(source.slice(start, end), context);
  }
}
function element() {
  return { style: {}, dataset: {}, classList: { toggle() {}, add() {}, remove() {}, contains: () => false },
    setAttribute(name, value) { this[name] = value; }, after() {}, remove() {}, clientWidth: 800, scrollLeft: 0 };
}

test("placing an older take keeps its render, prompt and style markers independent", () => {
  const markers = [{ frame: 0, style: "Graphite" }, { frame: 87, style: "Paper" }, { frame: 174, style: "Clay" }];
  const guides = [{ frame: 34, file: "pose.png", subfolder: "assets" }];
  const draft = { id: "draft", file: "draft.mp4", frames: 175, seed: 42,
    saved: { action: "A rowboat", seconds: 7, style_markers: markers, mid_guides: guides } };
  const original = { id: "shot", name: "Shot", active: "final", action: "New prompt", seconds: 8,
    style_markers: [], error: "An unrelated failure", takes: [draft, { id: "final", file: "final.mp4", frames: 192 }] };
  const project = { clips: [original], timeline: [{ clip: "shot", start: 0, in: 10, out: 192 }] };
  let next = 0, checkpoints = 0, saves = 0;
  const end = (item) => item.start + item.out - item.in;
  const c = vm.createContext({ state: { project }, structuredClone, itemEnd: end, FPS: 24,
    sceneEnd: () => Math.max(0, ...project.timeline.map(end)),
    sorted: () => [...project.timeline].sort((a, b) => a.start - b.start),
    uid: (prefix) => prefix + ++next, checkpoint: () => checkpoints++, save: () => saves++,
    selectClip() {}, seek() {}, toast() {}, undoAction: {},
  });
  vm.runInContext(source.match(/^const SAVED_FIELDS.*$/m)[0], c);
  functions(c, ["activeTake", "plannedFrames", "placeOnTimeline", "settle", "linkedRange", "placeTakeAsClip", "restoreTakeFields"]);
  assert.equal(c.placeTakeAsClip(original, "missing"), undefined);
  const item = c.placeTakeAsClip(original, "draft", 0);
  const placed = project.clips[1];
  assert.equal(item.clip, placed.id);
  assert.equal(item.out - item.in, 175);
  assert.equal(project.timeline[0].start, 175);
  assert.equal(placed.active, "draft");
  assert.equal(placed.seed, 42);
  assert.equal(placed.takes[0].file, "draft.mp4");
  assert.equal(placed.action, "A rowboat");
  assert.equal(placed.error, undefined);
  assert.deepEqual(placed.style_markers, markers);
  assert.deepEqual(placed.mid_guides, guides);
  placed.mid_guides[0].frame++;
  assert.equal(placed.takes[0].saved.mid_guides[0].frame, 34);
  c.restoreTakeFields(placed, { action: "Old take" });
  assert.equal(placed.mid_guides, undefined);
  placed.style_markers[1].style = "Ink";
  assert.equal(placed.takes[0].saved.style_markers[1].style, "Paper");
  assert.equal(original.active, "final");
  assert.equal(original.takes.length, 2);
  assert.equal(checkpoints, 1);
  assert.equal(saves, 1);
});
test("guide positions follow the selected placement, trim and fractional speed", () => {
  const clip = { id: "a", seconds: 5 }, items = [
    { id: "a1", clip: "a", start: 20, in: 10, out: 50, speed: 2 },
    { id: "a2", clip: "a", start: 24, in: 30, out: 42, speed: 0.5 },
  ];
  const state = { project: { timeline: items }, playhead: 27, item: "a2" };
  let viewer = null;
  const c = vm.createContext({ state, FPS: 24, JOINT: 51, viewing: () => viewer,
    itemEnd: i => i.start + Math.round((i.out - i.in) / i.speed), speedOf: i => i.speed,
    endHold: link => link.end_hold || link.hold, arrivesOnPin: link => link.arrive === "pin" });
  functions(c, ["nearestEven", "snapFrames", "linkedFrames", "midGuideLimits", "guideFrameAtPlayhead"]);
  assert.equal(c.guideFrameAtPlayhead(clip), 32);
  state.playhead = 47;
  assert.equal(c.guideFrameAtPlayhead(clip), 41); // Last displayed half-speed frame, trim out is exclusive.
  state.playhead = 48;
  assert.equal(c.guideFrameAtPlayhead(clip), null);
  viewer = clip; state.sourceFrame = 88;
  assert.equal(c.guideFrameAtPlayhead(clip), 88);
  assert.equal(c.guideFrameAtPlayhead({ id: "else" }), null);
  assert.equal(c.midGuideLimits(clip, (5 + 17 * 2.5) / 24).last, 37);
  assert.equal(c.midGuideLimits({ seconds: 5, link: { mode: "bridge", hold: 39 } }).limit, 7);
});
function video() {
  return { ...element(), currentTime: 0, paused: true, ended: false, readyState: 4, seeking: false,
    play() { this.paused = false; this.ended = false; return Promise.resolve(); }, pause() { this.paused = true; } };
}
function playback() {
  const elements = new Map();
  const frames = new Map();
  let next = 0;
  const c = vm.createContext({ FPS: 24, Math, performance: { now: () => 0 },
    state: { playing: false, loop: false, playhead: 0, source: null, range: null, sourceRange: null,
      project: { timeline: [{ start: 0, in: 0, out: 48, clip: "a" }, { start: 48, in: 12, out: 60, clip: "b" }],
        sounds: [{ id: "music", start: 0, frames: 96, gain: 1, offset: 0, file: "music.wav" }] } },
    front: video(), back: video(), lastTick: 0, tickGap: 16, playbackRequest: 0, PREROLL: 0.15,
    $: (id) => { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); },
    localStorage: { setItem() {} }, soundPlayers: new Map(), Audio: video,
    setSoundLevel, resumeSoundMixer, disconnectSound,
    viewUrl: (file) => file, heard: () => 1, trackAudible: () => true, monitoredTrack: () => 1,
    relevel() {}, setMix() {}, clearMix() {}, previewEffect() {},
    updateClock() {}, placePlayhead() {}, cutDue: () => null, incoming: () => null, cover() {}, applyMix() {},
    requestAnimationFrame: (callback) => { frames.set(++next, callback); return next; },
    cancelAnimationFrame: (id) => frames.delete(id),
    px: (frame) => frame, timecode: String,
  });
  c.viewing = () => c.state.source;
  c.sourceEnd = () => c.state.source ? 72 : 0;
  c.itemEnd = (item) => item.start + item.out - item.in;
  c.sceneEnd = () => c.state.project.timeline.reduce((end, item) => Math.max(end, c.itemEnd(item)), 0);
  c.viewEnd = () => c.viewing() ? c.sourceEnd() : c.sceneEnd();
  c.here = () => c.viewing() ? c.state.sourceFrame : c.state.playhead;
  c.leading = (frame) => c.state.project.timeline.find((item) => frame >= item.start && frame < c.itemEnd(item));
  c.sorted = () => c.state.project.timeline;
  c.nextItem = (frame) => c.sorted().find((item) => item.start > frame);
  c.sourceOf = (item) => item ? item.clip + ".mp4" : null;
  c.speedOf = () => 1;
  c.place = (target, item, frame) => { target.dataset.url = c.sourceOf(item); target.currentTime = (item.in + frame - item.start) / 24; };
  c.load = (target, url) => { target.dataset.url = url; };
  c.swap = () => { [c.front, c.back] = [c.back, c.front]; };
  c.showFrame = () => {
    if (!c.viewing()) {
      const item = c.leading(Math.min(c.state.playhead, c.sceneEnd() - 1));
      if (item) c.place(c.front, item, c.state.playhead);
    }
    c.showRange();
  };
  functions(c, ["boundedRange", "currentRange", "setRange", "showRange", "markIn", "markOut", "setLoop", "setSource",
    "setPlaying", "restartPlayback", "tick", "tickSource", "syncSounds"]);
  c.step = (mediaFrame, now = 16) => {
    assert.equal(frames.size, 1, "one playback callback at a time");
    const [id, callback] = frames.entries().next().value;
    frames.delete(id);
    c.front.currentTime = mediaFrame / 24;
    callback(now);
  };
  return { c, frames, elements };
}

test("loop button changes mode; In/Out include the displayed Out frame", () => {
  const { c, elements } = playback();
  c.state.playhead = 12; c.markIn();
  c.state.playhead = 23; c.markOut();
  assert.equal(JSON.stringify(c.state.range), '{"a":12,"b":24}');
  c.setLoop(true);
  assert.equal(elements.get("loop")["aria-pressed"], "true");
  c.setPlaying(true);
  c.step(24);
  assert.equal(c.state.playhead, 12);
  assert.equal(c.state.playing, true);
});

test("whole-cut looping seeks back and resets extra sound to the same time", () => {
  const { c } = playback();
  c.state.loop = true; c.state.playhead = 90;
  c.setPlaying(true); c.step(60);
  assert.equal(c.state.playhead, 0);
  c.step(1, 32);
  assert.equal(c.soundPlayers.get("music").currentTime, 1 / 24);
  assert.equal(c.soundPlayers.get("music").paused, false);
});

test("picture audio starts the mixer even without any added sounds", () => {
  const { c } = playback();
  c.state.project.sounds = [];
  let started = false;
  c.resumeSoundMixer = () => { started = true; };
  c.setPlaying(true);
  assert.equal(started, true);
});

test("sound mix follows source gain points and both placement fades when seeking", () => {
  const { c } = playback();
  const sound = c.state.project.sounds[0];
  Object.assign(sound, { seconds: 4, frames: 48, offset: 0.5, fade_in: 12, fade_out: 24,
    gain_points: [{ frame: 12, gain: 0.5 }, { frame: 24, gain: 1 }, { frame: 36, gain: 0.25 }] });
  assert.equal(curveLevel(sound, 18), 0.75);
  assert.equal(soundEnvelope(sound, 6), 0.375);
  assert.equal(soundEnvelope(sound, 48), 0);
  c.state.playing = true; c.state.playhead = 6; c.syncSounds();
  assert.equal(c.soundPlayers.get("music").volume, 0.375);
  c.state.playhead = 30; c.syncSounds();
  assert.ok(Math.abs(c.soundPlayers.get("music").volume - 0.325 * 0.75) < 1e-9);
});

test("once mode stops, while clip previews loop their own independent range", () => {
  const { c } = playback();
  c.state.playhead = 90; c.setPlaying(true); c.step(60);
  assert.equal(c.state.playing, false);
  c.state.source = "a"; c.state.sourceFrame = 0;
  c.state.range = { a: 40, b: 80 };
  c.state.sourceRange = { a: 6, b: 18 }; c.setLoop(true);
  c.setPlaying(true); c.step(18);
  assert.equal(c.state.sourceFrame, 6);
  assert.equal(c.front.currentTime, 6 / 24);
  assert.equal(c.state.range.a, 40);
  c.setRange(null, null); c.state.sourceFrame = 70;
  c.setPlaying(true); c.step(72);
  assert.equal(c.state.sourceFrame, 0);
});

test("rapid playback toggles leave one callback; empty timelines cannot spin", () => {
  const { c, frames } = playback();
  c.setPlaying(true); c.setPlaying(false); c.setPlaying(true); c.setPlaying(true);
  assert.equal(frames.size, 1);
  c.setPlaying(false); c.state.project.timeline = []; c.setPlaying(true);
  assert.equal(frames.size, 0);
  assert.equal(c.state.playing, false);
});

test("ranges are clamped after a trim and reset when a source changes", () => {
  const { c } = playback();
  c.setRange(40, 200);
  c.state.project.timeline.pop(); c.showRange();
  assert.equal(c.state.range.b, 48);
  c.state.sourceRange = { a: 1, b: 8 }; c.setSource("a");
  assert.equal(c.state.sourceRange, null);
});

function batch() {
  const clips = [{ id: "a", action: "Shot", takes: [{ quality: "draft" }] },
    { id: "b", action: "Shot", takes: [{ quality: "final" }] },
    { id: "c", action: "", takes: [] }, { id: "d", kind: "footage", action: "Footage", takes: [] }];
  const calls = [];
  const c = vm.createContext({ state: { folder: "first", project: { clips }, jobs: new Map() }, clientId: "test", batchRendering: false, batchCancelling: false,
    sorted: () => [{ clip: "a" }, { clip: "a" }, { clip: "b" }, { clip: "c" }, { clip: "d" }],
    clipOf: (id) => clips.find((clip) => clip.id === id), activeTake: (clip) => clip.takes.at(-1),
    jobFor: (id) => [...c.state.jobs.values()].find((job) => job.clip === id),
    flush: async () => {}, unsaved: () => false, toast() {},
    postJson: async (path, body) => { calls.push({ path, body }); return { cancelled: ["final1"] }; },
    api: async () => ({ clips }), adopt() {}, renderClip: async (clip, quality) => calls.push({ clip: clip.id, quality }),
  });
  functions(c, ["renderTimelineClips", "cancelRenders"]);
  return { c, calls };
}

test("batch finals skip duplicates, imported footage, blank shots, finals and busy clips", async () => {
  const { c, calls } = batch();
  await c.renderTimelineClips("final");
  assert.equal(JSON.stringify(calls[0].body.ops.map(({ clip, quality }) => [clip, quality])), '[["a","final"]]');
  calls.length = 0; c.state.jobs.set("a", { clip: "a" });
  await c.renderTimelineClips("final");
  assert.equal(calls.length, 0);
});

test("cancel finals is one project-scoped request with a quality filter", async () => {
  const { c, calls } = batch();
  await c.cancelRenders("final");
  assert.equal(calls[0].path, "/project/first/cancel");
  assert.equal(calls[0].body.all, true);
  assert.equal(calls[0].body.quality, "final");
});

test("a partially queued batch refreshes jobs before retrying other clips", async () => {
  const { c, calls } = batch();
  c.postJson = async () => { throw new Error("partial failure"); };
  c.adopt = () => c.state.jobs.set("a", { clip: "a" });
  await c.renderTimelineClips("final");
  assert.equal(calls.length, 0, "the already queued clip is not retried");
});

test("a project switch while flushing cannot send a clip render to the new project", async () => {
  const { c, calls } = batch();
  c.flush = async () => { c.state.folder = "second"; };
  functions(c, ["renderClip"]);
  await c.renderClip(c.state.project.clips[0], "final");
  assert.equal(calls.length, 0);
});

test("spectrograms crop source time at the clip's trim, including changed speed", async () => {
  const drawing = [];
  const canvas = { ...element(), isConnected: true, getContext: () => ({ drawImage: (...args) => drawing.push(args) }) };
  const c = vm.createContext({ state: { soundDisplay: "spectrogram", zoom: 3 }, FPS: 24, Math,
    window: { devicePixelRatio: 1 }, px: (frame) => frame * 3, itemLength: (item) => (item.out - item.in) / (item.speed || 1),
    spectrumCache: new Map(), viewUrl: (file, folder) => folder + "/" + file,
    postJson: async () => ({ spectrogram: { filename: "s.png", seconds: 12 } }),
    Image: class { width = 2048; height = 256; decode() { return Promise.resolve(); } },
    document: { createElement: element }, queueMicrotask,
  });
  functions(c, ["drawWave", "drawSpectrogram"]);
  c.drawWave(canvas, { in: 48, out: 96, speed: 2 }, { file: "take.mp4", subfolder: "p" }, 30, false);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(drawing.length, 1);
  assert.equal(drawing[0][1], 2048 * 48 / 288);
  assert.equal(drawing[0][3], 2048 * 48 / 288);
  assert.equal(canvas.style.width, "72px");
});
