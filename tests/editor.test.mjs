// Exercise the real editor functions with deterministic media clocks and a fake server.
// The editor has no bundler; extracting top-level functions keeps these tests dependency-free.
import { readFileSync } from "node:fs";
import vm from "node:vm";
import {
  buildAtlasPayload,
  buildPropPayload,
  buildSpriteConfig,
  CAST_VIEWS,
  compassStatusFor,
  defaultCanvasForCamera,
  draftSettingsSummary,
  fitZoomFor,
  parseTileSize,
  pickedSprite,
  referenceForDirection,
  screenToTilePoint,
  shouldUseHd,
  spriteSlots,
  zoomAtPoint,
} from "../editor/sprites.js";
import test from "node:test";
import assert from "node:assert/strict";
import { curveLevel, soundEnvelope, setSoundLevel, resumeSoundMixer, disconnectSound } from "../editor/sound-envelope.js";
import { formatJobMeta, mergeQueueState, relativeTime, summarizeQueue } from "../editor/queue.js";
import * as writer from "../editor/writer.js";

const source = readFileSync(new URL("../editor/editor.js", import.meta.url), "utf8");

test("sprite slots preserve the saved character, camera and direction when a clip changes", () => {
  const old = { direction: "NW", camera: "rpg" };
  const take = { id: "original", sprite: { poses: 7 }, saved: { sprite: old, cast: ["herald"] } };
  const clip = { id: "run", active: take.id, sprite: { direction: "E", camera: "level" }, cast: ["prowler"], takes: [take] };
  const project = { clips: [clip] };
  const slots = spriteSlots(project, "herald", "rpg", "run");
  const prefs = { picks: { "rpg/run/NE": { clip: clip.id, take: take.id, mirror: true } } };
  assert.equal(slots.NW.clips.length, 0);
  assert.equal(pickedSprite(slots, prefs, "rpg", "run", "NE").take, take);
  assert.equal(pickedSprite(slots, prefs, "rpg", "run", "NE").mirror, true);
  assert.equal(pickedSprite(slots, {}, "rpg", "run", "NE"), null);
  assert.deepEqual(spriteSlots(project, "prowler", "level", "run").E, { clips: [clip], takes: [] });
});
test("named motions keep separate direction slots and use each take's recorded animation", () => {
  const cast = ["herald"];
  const original = { camera: "rpg", direction: "NE", motion: "prompt", animation: "Wing stretch" };
  const take = { id: "stretch", sprite: { poses: 7 }, saved: { sprite: original, cast } };
  const clip = { id: "hover", cast, sprite: { ...original, animation: "Hover" }, takes: [take] };
  const legacy = { id: "guided", cast, sprite: { camera: "rpg", direction: "NE", guide: { file: "pose.mp4" } } };
  const project = { clips: [clip, legacy] };
  assert.deepEqual(spriteSlots(project, "herald", "rpg", "Wing stretch").NE, { clips: [], takes: [{ clip, take }] });
  assert.deepEqual(spriteSlots(project, "herald", "rpg", "Hover").NE, { clips: [clip], takes: [] });
  assert.deepEqual(spriteSlots(project, "herald", "rpg", "custom").NE, { clips: [legacy], takes: [] });
});

test("sprite v2 config, custom export tile sizes, webp toggle and 4 MP prop payloads", () => {
  assert.equal(defaultCanvasForCamera("rpg"), "square");
  assert.equal(defaultCanvasForCamera("level"), "portrait");
  assert.equal(defaultCanvasForCamera("overhead"), "portrait");
  assert.deepEqual(CAST_VIEWS.map(([v]) => v), ["", "front", "side", "back", "left"]);

  assert.deepEqual(
    buildSpriteConfig({ camera: "rpg", motion: "run", direction: "SE", canvas: "square", frames: 56 }),
    { camera: "rpg", direction: "SE", poses: 7, canvas: "square", frames: 56 },
  );
  assert.deepEqual(
    buildSpriteConfig({ camera: "rpg", motion: "idle", direction: "S", frames: 39 }),
    { camera: "rpg", direction: "S", poses: 7, motion: "idle", canvas: "square" },
  );
  assert.deepEqual(
    buildSpriteConfig({ camera: "level", motion: "cast", direction: "E", motionPrompt: "Raises staff.", canvas: "landscape", frames: 73 }),
    { camera: "level", direction: "E", poses: 7, motion: "prompt", animation: "cast", canvas: "landscape", frames: 73 },
  );

  assert.equal(parseTileSize("160"), 160);
  assert.equal(parseTileSize("custom", "160"), 160);
  assert.deepEqual(parseTileSize("custom", "144x176"), [144, 176]);
  assert.equal(parseTileSize("custom", "128×128"), 128);
  assert.throws(() => parseTileSize("custom", "161"), /even/);
  assert.throws(() => parseTileSize("custom", "16x64"), /32–512/);

  const picks = [{ motion: "run", direction: "S", clip: "c1", take: "t1", mirror: false }];
  assert.deepEqual(
    buildAtlasPayload({ tile: "160", fps: "24", style: "illustrated", webp: true, picks }),
    { tile: 160, fps: 24, style: "illustrated", webp: true, picks },
  );
  assert.deepEqual(
    buildAtlasPayload({ tile: "custom", customTile: "144x176", fps: "", style: "indexed", webp: false, picks }),
    { tile: 144, tile_size: [144, 176], fps: null, style: "indexed", webp: false, picks },
  );

  const ref = { file: "hero_S.png", subfolder: "refs" };
  const base = { file: "lantern_unlit.png", subfolder: "props" };
  assert.deepEqual(
    buildPropPayload({
      kind: "prop", name: "tree_oak", description: "Mossy oak tree", size: "square", camera: "rpg",
      style_ref: ref, top_canopy: true, output_size: "256x288",
    }),
    {
      kind: "prop", name: "tree_oak", description: "Mossy oak tree", size: "square", camera: "rpg",
      style_ref: ref, top_canopy: true, output_size: [256, 288],
    },
  );
  assert.deepEqual(
    buildPropPayload({
      kind: "state_variant", name: "lantern_lit", description: "Warm flame glow", size: "square", camera: "rpg",
      style_ref: ref, base_prop: base,
    }),
    {
      kind: "state_variant", name: "lantern_lit", description: "Warm flame glow", size: "square", camera: "rpg",
      style_ref: ref, base_prop: base,
    },
  );
  assert.throws(() => buildPropPayload({ kind: "state_variant", name: "x", description: "glow" }), /base prop/);
  assert.deepEqual(
    buildPropPayload({
      kind: "turnaround", name: "hero_back", size: "portrait", camera: "rpg",
      style_ref: ref, base_prop: { file: "hero_E.png", subfolder: "refs" },
      view: "back", proportions: "chibi", holding: "a crystal staff",
    }),
    {
      kind: "turnaround", name: "hero_back", description: "", size: "portrait", camera: "rpg",
      style_ref: ref, base_prop: { file: "hero_E.png", subfolder: "refs" },
      view: "back", proportions: "chibi", holding: "a crystal staff",
    },
  );
  assert.throws(() => buildPropPayload({ kind: "turnaround", name: "hero_back" }), /front style reference/);
});

test("sprite clips with 56 frames report 56 planned frames and guide limits up to 54", () => {
  const clip = { id: "s56", seconds: 5, takes: [], sprite: { direction: "E", camera: "rpg", canvas: "square", frames: 56 } };
  const c = vm.createContext({
    FPS: 24, JOINT: 51,
    activeTake: () => null,
    endHold: (l) => l.end_hold || l.hold,
    arrivesOnPin: (l) => l.arrive === "pin",
  });
  functions(c, ["nearestEven", "snapFrames", "linkedFrames", "plannedFrames", "midGuideLimits"]);
  assert.equal(c.plannedFrames(clip), 56);
  assert.equal(c.midGuideLimits(clip).last, 54);
  assert.equal(c.plannedFrames({ id: "s39", seconds: 5, takes: [], sprite: { direction: "S", camera: "rpg" } }), 39);
});

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

test("the comparison video follows a playing take and seeks to the same paused frame", () => {
  const { c } = playback();
  const follower = video();
  c.$ = () => follower;
  functions(c, ["syncComparison"]);
  c.state.playing = true;
  c.front.paused = false;
  c.syncComparison(1);
  assert.equal(follower.currentTime, 1);
  assert.equal(follower.paused, false);
  c.syncComparison(1 + 1 / 48);
  assert.equal(follower.currentTime, 1);
  c.state.playing = false;
  c.syncComparison(1 + 1 / 48);
  assert.equal(follower.currentTime, 1 + 1 / 48);
  assert.equal(follower.paused, true);
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

test("partial draft finalization detects split parts, blocks finished halves, and posts part range", async () => {
  const calls = [], toasts = [];
  const draftEdit = { id: "t1", frames: 240, quality: "draft", edit: { take: "t0", in: 0, out: 240, prompt: "Red jacket" } };
  const partialFinal = { id: "t2", frames: 240, quality: "final", finished: "t1", final_ranges: [[100, 240]], edit: draftEdit.edit };
  const clip = { id: "c1", kind: "footage", name: "Footage", action: "", takes: [{ id: "t0", frames: 240, quality: "footage" }, draftEdit], active: "t1" };
  const left = { id: "i1", clip: "c1", start: 0, in: 0, out: 100 };
  const right = { id: "i2", clip: "c1", start: 100, in: 100, out: 240 };
  const c = vm.createContext({
    state: { folder: "cut", item: "i2", range: null, sourceRange: null, jobs: new Map(),
      project: { clips: [clip], timeline: [left, right], sizes: {} } },
    FPS: 24, clientId: "tab", Math,
    viewing: () => null,
    sorted: () => [left, right],
    speedOf: () => 1,
    itemLength: (item) => item.out - item.in,
    itemEnd: (item) => item.start + item.out - item.in,
    flush: async () => {},
    postJson: async (path, body) => { calls.push({ path, body }); return { job: { id: "j1", clip: body.clip, part: body.part }, project: c.state.project }; },
    unsaved: () => false, adopt() {}, updateQueue() {}, renderAll() {}, selectClip() {},
    toast: (msg, bad) => toasts.push({ msg, bad: Boolean(bad) }),
  });
  functions(c, ["activeTake", "jobFor", "timecode", "boundedRange", "takeRangeStatus", "finishRange", "finishOff", "finishDraft", "renderClip"]);
  assert.equal(c.finishOff(clip, right), "");
  assert.equal(JSON.stringify(c.finishRange(clip, draftEdit, right)), '{"a":100,"b":240,"partial":true}');
  assert.equal(JSON.stringify(c.finishRange(clip, draftEdit, null)), '{"a":0,"b":240,"partial":false}');
  assert.equal(await c.finishDraft(clip, right), true);
  assert.equal(calls.length, 1);
  assert.equal(JSON.stringify(calls[0].body.part), '{"in":100,"out":240}');
  assert.match(toasts[0].msg, /draft latents/);

  // Fallback notice is surfaced verbatim when the server could not reuse draft latents.
  c.state.jobs.clear();
  c.postJson = async (path, body) => {
    calls.push({ path, body });
    return { job: { id: "j2", clip: body.clip, part: body.part, finish_fallback: "This draft has no saved latents." }, project: c.state.project };
  };
  assert.equal(await c.finishDraft(clip, left), true);
  assert.equal(toasts[1].msg, "This draft has no saved latents.");

  // After partial finish activates, right is final while left is still finishable as draft.
  c.state.jobs.clear();
  clip.takes.push(partialFinal);
  clip.active = "t2";
  assert.equal(c.takeRangeStatus(partialFinal, 100, 240), "final");
  assert.equal(c.takeRangeStatus(partialFinal, 0, 100), "draft");
  assert.equal(c.takeRangeStatus(partialFinal, 50, 150), "partial");
  assert.match(c.finishOff(clip, right), /already a final/);
  assert.equal(c.finishOff(clip, left), "");
});

test("managed render queue merges live progress, external ComfyUI jobs, props and summaries", () => {
  const snapshot = {
    running: [{ id: "p1", kind: "sprite", folder: "mascots", project: "Mascots", clip: "c1", name: "Mantis · run · E", quality: "draft", size: [512, 512], frames: 39, sprite: { direction: "E" } }],
    pending: [
      { id: "p2", kind: "prop", folder: "mascots", project: "Mascots", clip: null, name: "stone_lantern", quality: "prop", size: [2048, 2048], frames: 1, position: 1 },
      { id: "p3", kind: "comfyui", folder: null, project: null, clip: null, name: "ComfyUI workflow", quality: null, position: 2 },
    ],
    history: [
      { id: "p0", kind: "clip", folder: "mascots", clip: "c0", name: "Intro", quality: "draft", state: "failed", error: "OOM", at: 950 },
    ],
  };
  const localJobs = new Map([
    ["p1", { id: "p1", clip: "c1", stage: "Sampling", step: 4, max: 8, progress: 0.5 }],
    ["p4", { id: "p4", clip: "c2", quality: "final", frames: 56 }],
  ]);
  const merged = mergeQueueState(snapshot, localJobs, localJobs.get("p1"), "mascots", (id) => (id === "c2" ? { id: "c2", name: "Hero close-up" } : null));
  assert.equal(merged.running.length, 1);
  assert.equal(merged.running[0].progress, 0.5);
  assert.equal(merged.running[0].stage, "Sampling");
  assert.equal(merged.pending.length, 3);
  assert.equal(merged.pending[2].name, "Hero close-up");

  const sum = summarizeQueue(merged);
  assert.equal(sum.busy, true);
  assert.equal(sum.runningCount, 1);
  assert.equal(sum.pendingCount, 3);
  assert.equal(sum.failedCount, 1);
  assert.equal(sum.label, "Mantis · run · E · 50% · +3 queued");
  assert.equal(sum.summary, "1 running · 3 waiting");

  assert.deepEqual(formatJobMeta(merged.running[0], "mascots"), ["Sprite draft", "E", "512×512", "39f"]);
  assert.deepEqual(formatJobMeta(merged.pending[0], "mascots"), ["4 MP prop", "2048×2048"]);
  assert.deepEqual(formatJobMeta(merged.pending[1], "mascots"), ["ComfyUI"]);
  assert.equal(relativeTime(950, 1000), "50s ago");
  assert.equal(relativeTime(995, 1000), "just now");

  const idleSum = summarizeQueue({ running: [], pending: [], history: snapshot.history });
  assert.equal(idleSum.busy, false);
  assert.equal(idleSum.label, "Queue · 1 failed");
  assert.equal(idleSum.summary, "Idle");
});

test("ollama requests keep_alive: 0 when sharing the GPU with ComfyUI", async () => {
  assert.deepEqual(writer.chatOptions({ type: "ollama", freeVram: true }), { parameters: { keep_alive: 0 } });
  assert.equal(writer.chatOptions({ type: "ollama", freeVram: false }), undefined);
  assert.equal(writer.chatOptions({ type: "openai", freeVram: true }), undefined);
  assert.equal(writer.chatOptions({ type: "comfyui", freeVram: true }), undefined);

  const realFetch = globalThis.fetch;
  const bodies = [];
  let freed = false;
  let vramChecks = 0;
  globalThis.fetch = async (url, init = {}) => {
    const target = String(url);
    if (target === "/queue") return new Response(JSON.stringify({ queue_running: [], queue_pending: [] }), { status: 200 });
    if (target === "/system_stats") {
      vramChecks += 1;
      const free = vramChecks === 1 ? 1024 * 1024 * 1024 : 8 * 1024 * 1024 * 1024;
      return new Response(JSON.stringify({ devices: [{ type: "cuda", vram_free: free }] }), { status: 200 });
    }
    if (target === "/free") {
      freed = true;
      return new Response("{}", { status: 200 });
    }
    if (target === "http://localhost:11434/api/chat") {
      bodies.push(JSON.parse(init.body));
      return new Response(JSON.stringify({ message: { role: "assistant", content: "Ready." }, done_reason: "stop" }), {
        status: 200, headers: { "Content-Type": "application/json" },
      });
    }
    throw new Error("Unexpected fetch: " + target);
  };
  try {
    const base = { type: "ollama", url: "http://localhost:11434", model: "gemma4-uncensored", key: "" };
    assert.equal(await writer.test({ ...base, freeVram: true }), "Ready.");
    assert.equal(freed, true);
    assert.equal(bodies.length, 1);
    assert.equal(bodies[0].keep_alive, 0);
    assert.equal("keep_alive" in (bodies[0].options || {}), false);

    await writer.test({ ...base, freeVram: false });
    assert.equal(bodies.length, 2);
    assert.equal("keep_alive" in bodies[1], false);
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("motion transfer helpers cover SAVED_FIELDS, plannedFrames, window counts, required packs, and relay stages", () => {
  const c = vm.createContext({ FPS: 24, MOTION_MAX_SECONDS: 30, structuredClone,
    activeTake: (clip) => (clip.takes || []).find((t) => t.id === clip.active) || (clip.takes || []).at(-1) || null,
    linkedFrames: () => 141, snapFrames: () => 124, jobFor: () => null,
    promptEditRange: (clip, take) => ({ a: 0, b: take.frames }),
  });
  vm.runInContext(source.match(/^const STAGES = [\s\S]*?^const STEPPED = .*$/m)[0], c);
  vm.runInContext(source.match(/^const SAVED_FIELDS.*$/m)[0], c);
  functions(c, [
    "stageForNode", "isSteppedNode", "plannedFrames", "clipSeconds",
    "restoreTakeFields", "motionUnavailable", "motionWindowSummary",
    "motionRequiredGroups", "motionRangeBounds",
  ]);

  // SAVED_FIELDS includes motion and restoreTakeFields restores/clears it
  assert.ok(vm.runInContext("SAVED_FIELDS", c).includes("motion"));
  const motionRecipe = { mode: "replace", style: "preserve", source: { clip: "c0", in: 0, out: 230, frames: 230 }, character: { file: "rin.png" } };
  const clip = { id: "c1", seconds: 5, action: "", takes: [], motion: motionRecipe };
  assert.equal(c.plannedFrames(clip), 230);
  assert.equal(c.clipSeconds(clip), (230 / 24).toFixed(1) + " s");
  c.restoreTakeFields(clip, { action: "plain" });
  assert.equal(clip.motion, undefined);
  c.restoreTakeFields(clip, { action: "", motion: motionRecipe });
  assert.deepEqual(clip.motion, motionRecipe);

  // Window relay summary matches 124f / 22f overlap / 141f single-pass threshold
  assert.equal(JSON.stringify(c.motionWindowSummary(120)), JSON.stringify({ windows: 1, label: "Single pass" }));
  assert.equal(JSON.stringify(c.motionWindowSummary(141)), JSON.stringify({ windows: 1, label: "Single pass" }));
  assert.equal(JSON.stringify(c.motionWindowSummary(230)), JSON.stringify({ windows: 2, label: "2-window relay" }));
  assert.equal(JSON.stringify(c.motionWindowSummary(720)), JSON.stringify({ windows: 7, label: "7-window relay" }));

  // Required model packs for each combination
  assert.equal(JSON.stringify(c.motionRequiredGroups("replace", 0, "draft")), JSON.stringify(["replace"]));
  assert.equal(JSON.stringify(c.motionRequiredGroups("replace", 0.75, "sharp")), JSON.stringify(["replace", "pose", "sharp"]));
  assert.equal(JSON.stringify(c.motionRequiredGroups("replace", 0, "fast_sharp")), JSON.stringify(["replace", "sharp"]));
  assert.equal(JSON.stringify(c.motionRequiredGroups("rescene", 0, "realism_fast")), JSON.stringify([]));
  assert.equal(JSON.stringify(c.motionRequiredGroups("rescene", 0.75, "realism_fast")), JSON.stringify(["pose"]));
  assert.equal(JSON.stringify(c.motionRequiredGroups("rescene", 0, "fast_sharp")), JSON.stringify(["sharp"]));
  assert.equal(JSON.stringify(c.motionRequiredGroups("choreography", 1.0, "final")), JSON.stringify(["pose"]));
  assert.equal(JSON.stringify(c.motionRequiredGroups("choreography", 1.0, "sharp")), JSON.stringify(["pose", "sharp"]));

  // Double-latent-carry 1088p realism windows (107f / 85f stride)
  assert.equal(JSON.stringify(c.motionWindowSummary(107, "realism_fast")), JSON.stringify({ windows: 1, label: "Single pass" }));
  assert.equal(JSON.stringify(c.motionWindowSummary(342, "realism_fast")), JSON.stringify({ windows: 4, label: "4-window 1088p relay" }));
  assert.equal(c.stageForNode("365"), "Refining 1088p window 2");
  assert.equal(c.isSteppedNode("365"), true);

  // Range bounds clamp >30s source clips to 30s default and preserve existing motion source bounds
  const longClip = { id: "f1", takes: [{ id: "t1", frames: 960 }], active: "t1" };
  assert.equal(JSON.stringify(c.motionRangeBounds(longClip, longClip.takes[0], null)), JSON.stringify({ a: 0, b: 720, total: 960, clamped: true }));
  assert.equal(JSON.stringify(c.motionRangeBounds(clip, null, null)), JSON.stringify({ a: 0, b: 230, total: 230, clamped: false }));

  // motionUnavailable allows opening the dialog for a clip with a take or an existing motion clip
  assert.equal(c.motionUnavailable(longClip), "");
  assert.equal(c.motionUnavailable(clip), "");
  assert.notEqual(c.motionUnavailable({ id: "blank", takes: [] }), "");

  // STAGES and STEPPED cover SAM 3.1, plate, RelayVSR 2x, and dynamic multi-window relay samplers
  assert.equal(c.stageForNode("240"), "Tracking subject mask (SAM 3.1)");
  assert.equal(c.stageForNode("250"), "Compositing source plate");
  assert.equal(c.stageForNode("34"), "Upscaling 2x (RelayVSR)");
  assert.equal(c.stageForNode("324"), "Locking relay window 2");
  assert.equal(c.stageForNode("326"), "Sampling window 2");
  assert.equal(c.stageForNode("328"), "Joining window 2");
  assert.equal(c.stageForNode("346"), "Sampling window 3");
  assert.equal(c.isSteppedNode("15"), true);
  assert.equal(c.isSteppedNode("34"), true);
  assert.equal(c.isSteppedNode("326"), true);
  assert.equal(c.isSteppedNode("346"), true);
  assert.equal(c.isSteppedNode("328"), false);
});

test("motion clips with empty action pass both single renderClip and batch renderTimelineClips guards", async () => {
  const clips = [
    { id: "m1", name: "Rin · eira3", action: "", takes: [], motion: { mode: "replace", source: { in: 0, out: 120 } } },
    { id: "blank", name: "Blank", action: "", takes: [] },
  ];
  const calls = [];
  const c = vm.createContext({
    state: { folder: "proj", project: { clips }, jobs: new Map() }, clientId: "test",
    batchRendering: false, batchCancelling: false,
    sorted: () => [{ clip: "m1" }, { clip: "blank" }],
    clipOf: (id) => clips.find((clip) => clip.id === id),
    activeTake: (clip) => clip.takes.at(-1) || null,
    jobFor: () => null, flush: async () => {}, unsaved: () => false,
    selectClip() {}, toast() {}, adopt() {}, updateQueue() {}, renderAll() {},
    api: async () => ({ clips }),
    postJson: async (path, body) => {
      calls.push({ path, body });
      return { job: { id: "j1", clip: body.clip || "m1", quality: body.quality || "draft" }, project: { clips } };
    },
  });
  functions(c, ["renderClip", "renderTimelineClips"]);

  assert.equal(await c.renderClip(clips[0], "draft"), true);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].path, "/project/proj/render");

  calls.length = 0;
  await c.renderTimelineClips("draft");
  assert.equal(calls.length, 1);
  assert.equal(JSON.stringify(calls[0].body.ops.map(({ clip, quality }) => [clip, quality])), '[["m1","draft"]]');
});

test("drag-and-drop file classifiers and motionCastMembers project+catalog picker helpers", () => {
  const c = vm.createContext({});
  for (const name of ["FOOTAGE_EXT", "PICTURE_EXT", "AUDIO_EXT"]) {
    const m = source.match(new RegExp(`const ${name} = [^;]+;`));
    assert.ok(m, `missing ${name}`);
    vm.runInContext(m[0], c);
  }
  functions(c, ["isPictureFile", "isFootageFile", "isAudioFile", "motionMemberPicture", "motionCastMembers"]);

  assert.equal(c.isFootageFile({ name: "dance.mp4", type: "video/mp4" }), true);
  assert.equal(c.isFootageFile({ name: "clip.MOV", type: "" }), true);
  assert.equal(c.isFootageFile({ name: "still.webp", type: "image/webp" }), true);
  assert.equal(c.isFootageFile({ name: "song.mp3", type: "audio/mpeg" }), false);
  assert.equal(c.isAudioFile({ name: "song.mp3", type: "audio/mpeg" }), true);
  assert.equal(c.isAudioFile({ name: "voice.WAV", type: "" }), true);
  assert.equal(c.isPictureFile({ name: "ruby.png", type: "image/png" }), true);
  assert.equal(c.isPictureFile({ name: "dance.mp4", type: "video/mp4" }), false);

  const project = {
    cast: [
      { id: "p1", name: "Maya", file: "maya.png", subfolder: "proj/assets", catalog: { id: "cat1" } },
      { id: "p2", name: "VoiceOnly", voice: { file: "v.wav" } },
      { id: "p3", name: "AltOnly", pictures: [{ file: "alt.png", subfolder: "proj/assets" }] },
    ],
  };
  const catalog = [
    { id: "cat1", name: "Maya", file: "maya.png", subfolder: "cat" },
    { id: "cat2", name: "Ruby", file: "ruby.png", subfolder: "cat" },
    { id: "cat3", name: "Narrator", voice: { file: "n.wav" } },
  ];
  const rows = c.motionCastMembers(project, catalog);
  assert.equal(JSON.stringify(rows.map((r) => [r.kind, r.id, r.name, r.file])), JSON.stringify([
    ["cast", "p1", "Maya", "maya.png"],
    ["cast", "p3", "AltOnly", "alt.png"],
    ["catalog", "cat2", "Ruby", "ruby.png"],
  ]));
});

test("motion tab helpers map SwapGuard (248) and relay windows to live pipeline stages", () => {
  const c = vm.createContext({});
  for (const name of ["STAGES", "STEPPED"]) {
    const m = source.match(new RegExp(`const ${name} = [^;]+;`));
    assert.ok(m, `missing ${name}`);
    vm.runInContext(m[0], c);
  }
  functions(c, ["stageForNode", "isSteppedNode", "motionStageForNode", "formatMotionBytes"]);

  assert.equal(c.stageForNode("248"), "Checking character swap");
  const stages = [
    { id: "source", nodes: ["213"] },
    { id: "cutout", nodes: ["214", "239"] },
    { id: "segmentation", nodes: ["238", "240", "245"] },
    { id: "pose", nodes: ["215", "216", "217", "218", "219"] },
    { id: "sampling", nodes: ["10", "15", "248", "16"] },
    { id: "plate", nodes: ["250", "34", "19"] },
  ];
  assert.equal(c.motionStageForNode(stages, "240"), "segmentation");
  assert.equal(c.motionStageForNode(stages, "248"), "sampling");
  assert.equal(c.motionStageForNode(stages, "260"), "sampling");
  assert.equal(c.motionStageForNode(stages, "326"), "sampling");
  assert.equal(c.motionStageForNode(stages, "250"), "plate");
  assert.equal(c.motionStageForNode(stages, "99"), null);
  assert.equal(c.formatMotionBytes(2097152), "2.00 MB");
  assert.equal(c.formatMotionBytes(4096), "4 KB");
});

test("RefMod helpers format labels, sidecar preview URLs and addRefmodMember creates stem-named cast members", () => {
  const project = { cast: [{ id: "p0", name: "Fenka", file: "fenka.png" }] };
  const state = {
    project,
    person: null,
    clip: "c1",
    item: "i1",
    refmods: {
      installed: true,
      files: ["Fenka.safetensors", "sub/Sarah.safetensors", "Plain.safetensors"],
      previews: { "Fenka.safetensors": 1700000100, "sub/Sarah.safetensors": 1700000200 },
    },
  };
  let next = 0, checkpoints = 0, saves = 0, rendered = 0, shownPanel = null, lastToast = null;
  const c = vm.createContext({
    state,
    encodeURIComponent,
    uid: (prefix) => prefix + ++next,
    checkpoint: () => checkpoints++,
    save: () => saves++,
    renderAll: () => rendered++,
    setCollectionOpen() {},
    showWorkspacePanel: (p) => { shownPanel = p; },
    toast: (msg) => { lastToast = msg; },
    undoAction: { label: "Undo" },
  });
  functions(c, ["refmodLabel", "refmodFolder", "refmodPreviewUrl", "uniqueCastName", "addRefmodMember"]);

  assert.equal(c.refmodLabel("Fenka.safetensors"), "Fenka");
  assert.equal(c.refmodLabel("sub\\Sarah.safetensors"), "Sarah");
  assert.equal(c.refmodFolder("sub\\Sarah.safetensors"), "sub");
  assert.equal(c.refmodFolder("Fenka.safetensors"), "");
  assert.equal(
    c.refmodPreviewUrl("sub/Sarah.safetensors"),
    "/h3take/api/refmods/preview?file=sub%2FSarah.safetensors&v=1700000200",
  );
  assert.equal(c.refmodPreviewUrl("Plain.safetensors"), null);

  // Adding a RefMod whose stem collides with an existing cast member deduplicates cleanly
  const first = c.addRefmodMember("Fenka.safetensors");
  assert.equal(first.name, "Fenka 2");
  assert.equal(JSON.stringify(first.refmod), JSON.stringify({ file: "Fenka.safetensors", strength: 1 }));
  assert.equal(state.person, first.id);
  assert.equal(state.clip, null);
  assert.equal(shownPanel, "inspector");
  assert.equal(lastToast, "Fenka 2 added to Cast.");

  // Adding a subfolder RefMod uses its file stem
  const second = c.addRefmodMember("sub/Sarah.safetensors");
  assert.equal(second.name, "Sarah");
  assert.equal(second.refmod.file, "sub/Sarah.safetensors");
  assert.equal(checkpoints, 2);
  assert.equal(saves, 2);
  assert.equal(rendered, 2);
});

test("cast picture helpers reorder, promote, replace preserving view, remove safely, and cap appends at 8", () => {
  const c = vm.createContext({ CAST_PICTURES: 8, CAST_VIEWS });
  functions(c, [
    "castViewLabel",
    "castPicturesOf",
    "applyCastPictures",
    "moveCastPicture",
    "promoteCastPicture",
    "replaceCastPicture",
    "canRemoveCastPicture",
    "removeCastPicture",
    "appendCastPictures",
    "setCastPictureView",
  ]);

  assert.equal(c.castViewLabel("front"), "Front (S)");
  assert.equal(c.castViewLabel(""), "Any view");

  const person = {
    id: "p1",
    name: "Solaris",
    file: "main.png",
    subfolder: "cast",
    view: "front",
    pictures: [
      { file: "side.png", subfolder: "cast", view: "side" },
      { file: "full.png", subfolder: "cast" },
    ],
  };

  const plain = (v) => JSON.parse(JSON.stringify(v));

  assert.deepEqual(plain(c.castPicturesOf(person)), [
    { file: "main.png", subfolder: "cast", view: "front" },
    { file: "side.png", subfolder: "cast", view: "side" },
    { file: "full.png", subfolder: "cast" },
  ]);

  // Promote index 2 to Main (#1): shifts old Main to index 1 and keeps all views intact
  assert.equal(c.promoteCastPicture(person, 2), true);
  assert.equal(person.file, "full.png");
  assert.equal(person.view, undefined);
  assert.deepEqual(plain(person.pictures), [
    { file: "main.png", subfolder: "cast", view: "front" },
    { file: "side.png", subfolder: "cast", view: "side" },
  ]);

  // Move index 1 back to index 0
  assert.equal(c.moveCastPicture(person, 1, 0), true);
  assert.equal(person.file, "main.png");
  assert.equal(person.view, "front");

  // Replace Main picture preserves view tag by default
  assert.equal(c.replaceCastPicture(person, 0, { file: "main_v2.png", subfolder: "cast2" }), true);
  assert.equal(person.file, "main_v2.png");
  assert.equal(person.subfolder, "cast2");
  assert.equal(person.view, "front");

  // Replace secondary slot preserves its view tag
  assert.equal(c.replaceCastPicture(person, 2, { file: "side_v2.png", subfolder: "cast2" }), true);
  assert.equal(person.pictures[1].file, "side_v2.png");
  assert.equal(person.pictures[1].view, "side");

  // Set and clear view tag
  assert.equal(c.setCastPictureView(person, 1, "back"), true);
  assert.equal(person.pictures[0].view, "back");
  assert.equal(c.setCastPictureView(person, 0, ""), true);
  assert.equal(person.view, undefined);

  // Removing Main promotes index 1 to Main
  assert.equal(c.removeCastPicture(person, 0), true);
  assert.equal(person.file, "full.png");
  assert.equal(person.view, "back");
  assert.equal(person.pictures.length, 1);

  // Removing down to 1 picture succeeds, then removing the last picture on picture-only member is blocked
  assert.equal(c.removeCastPicture(person, 1), true);
  assert.equal(person.pictures, undefined);
  assert.equal(c.canRemoveCastPicture(person), false);
  assert.equal(c.removeCastPicture(person, 0), false);
  assert.equal(person.file, "full.png");

  // Voice-backed member CAN remove its last picture and later append fresh ones
  const voiced = { id: "p2", name: "Narrator", voice: { file: "v.wav", subfolder: "" }, file: "only.png", subfolder: "" };
  assert.equal(c.canRemoveCastPicture(voiced), true);
  assert.equal(c.removeCastPicture(voiced, 0), true);
  assert.equal(voiced.file, undefined);
  const firstAdd = c.appendCastPictures(voiced, [{ file: "a.png", subfolder: "" }, { file: "b.png", subfolder: "" }]);
  assert.deepEqual(plain(firstAdd), { added: 2, skipped: 0, total: 2 });
  assert.equal(voiced.file, "a.png");
  assert.equal(voiced.pictures.length, 1);

  // Appending past CAST_PICTURES (8) caps cleanly and reports skipped count
  const batch = Array.from({ length: 10 }, (_, i) => ({ file: `p${i}.png`, subfolder: "" }));
  const capped = c.appendCastPictures(voiced, batch);
  assert.deepEqual(plain(capped), { added: 6, skipped: 4, total: 8 });
  assert.equal(c.castPicturesOf(voiced).length, 8);
});

test("writer.cleanCastDescription sanitizes model prose and describeCast requires vision pictures", async () => {
  assert.equal(
    writer.cleanCastDescription('<think>internal</think>\n"Description: a weathered sailor with a salt-streaked beard\nand navy wool peacoat"'),
    "a weathered sailor with a salt-streaked beard and navy wool peacoat",
  );
  assert.equal(
    writer.cleanCastDescription("```text\na silver espresso machine with brass gauges\n```"),
    "a silver espresso machine with brass gauges",
  );

  await assert.rejects(
    () => writer.describeCast({ person: { name: "Solaris" }, pictures: [] }, { type: "comfyui", url: "", model: "qwen.safetensors", key: "", freeVram: false }),
    /at least one reference picture/i,
  );

  // Mock ComfyUI vision call: uploads image, submits graph with LoadImage wired into TextGenerateV2, returns cleaned description
  const originalFetch = globalThis.fetch;
  let uploadedCount = 0;
  let promptGraph = null;
  globalThis.fetch = async (url, options = {}) => {
    const u = String(url);
    if (u.endsWith("/upload/image")) {
      uploadedCount += 1;
      return { ok: true, json: async () => ({ name: `ref_${uploadedCount}.jpg`, subfolder: "h3take_writer" }) };
    }
    if (u.endsWith("/prompt")) {
      promptGraph = JSON.parse(options.body).prompt;
      return { ok: true, json: async () => ({ prompt_id: "pid_cast_1" }) };
    }
    if (u.endsWith("/history/pid_cast_1")) {
      return {
        ok: true,
        json: async () => ({
          pid_cast_1: {
            status: { status_str: "success" },
            outputs: { "4": { text: ['"a sun-kissed woman with amber-hazel eyes and a gold satin halter top"'] } },
          },
        }),
      };
    }
    throw new Error("unexpected fetch " + u);
  };
  try {
    const fakePic = { label: "Picture 1 (Main), Front (S)", blob: new Blob(["img"], { type: "image/jpeg" }), url: "data:image/jpeg;base64,AA==" };
    const result = await writer.describeCast(
      { person: { name: "Solaris" }, project: { style: "cinematic" }, pictures: [fakePic] },
      { type: "comfyui", url: "", model: "qwen3.5_9b_w4a8_gptq.safetensors", key: "", freeVram: false },
    );
    assert.equal(result, "a sun-kissed woman with amber-hazel eyes and a gold satin halter top");
    assert.equal(uploadedCount, 1);
    assert.deepEqual(promptGraph["3"].inputs["images.image1"], ["10", 0]);
    assert.equal(writer.lastSeen().pictures, 1);
    assert.equal(writer.lastSeen().blind, false);
  } finally {
    globalThis.fetch = originalFetch;
  }

  // When a vision model rejects images, describeCast MUST fail instead of silently falling back to blind text
  let promptCalls = 0;
  globalThis.fetch = async (url) => {
    const u = String(url);
    if (u.endsWith("/upload/image")) return { ok: true, json: async () => ({ name: "ref.jpg", subfolder: "h3take_writer" }) };
    if (u.endsWith("/prompt")) {
      promptCalls += 1;
      return { ok: false, json: async () => ({ error: "This model does not support image inputs" }) };
    }
    throw new Error("unexpected fetch " + u);
  };
  try {
    const fakePic = { label: "Picture 1 (Main)", blob: new Blob(["img"], { type: "image/jpeg" }), url: "data:image/jpeg;base64,AA==" };
    await assert.rejects(
      () => writer.describeCast(
        { person: { name: "Solaris" }, pictures: [fakePic] },
        { type: "comfyui", url: "", model: "text_only.safetensors", key: "", freeVram: false },
      ),
      /vision-capable model/i,
    );
    assert.equal(promptCalls, 1);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("cast @ mention autocomplete detects trigger boundaries, ranks matches, and applies mentions cleanly", () => {
  const c = vm.createContext({});
  vm.runInContext(source.match(/const MENTION_BOUNDARY = [^\n]+/)[0], c);
  functions(c, ["castMentionContext", "matchCastMentions", "applyCastMention"]);

  const plain = (v) => JSON.parse(JSON.stringify(v));

  // Trigger boundaries: start of text, after space/punctuation; ignores emails & newlines
  assert.deepEqual(plain(c.castMentionContext("@", 1)), { start: 0, end: 1, query: "" });
  assert.deepEqual(plain(c.castMentionContext("Close-up on @sol", 16)), { start: 12, end: 16, query: "sol" });
  assert.deepEqual(plain(c.castMentionContext("(@Sor", 5)), { start: 1, end: 5, query: "Sor" });
  assert.equal(c.castMentionContext("mail@solaris.com", 12), null);
  assert.equal(c.castMentionContext("@sol\nnext line", 14), null);

  const cast = [
    { id: "p1", name: "Narrator", voice: { file: "v.wav" } },
    { id: "p2", name: "Solaris", file: "s.png", voice: { file: "sv.wav" } },
    { id: "p3", name: "Sorceress", file: "m.png" },
    { id: "p4", name: "Old Keeper", refmod: { file: "keeper.safetensors" } },
  ];

  // Visual mode excludes voice-only narrator
  assert.deepEqual(
    plain(c.matchCastMentions(cast, "", { mode: "visual" }).map((p) => p.name)),
    ["Solaris", "Sorceress", "Old Keeper"],
  );
  // Prefix ranks ahead of substring
  assert.deepEqual(
    plain(c.matchCastMentions(cast, "so", { mode: "visual" }).map((p) => p.name)),
    ["Solaris", "Sorceress"],
  );
  // Multi-word cast name matches cleanly, while trailing words beyond match close menu
  assert.deepEqual(
    plain(c.matchCastMentions(cast, "old k", { mode: "visual" }).map((p) => p.name)),
    ["Old Keeper"],
  );
  assert.deepEqual(plain(c.matchCastMentions(cast, "Solaris walks", { mode: "visual" })), []);

  // Voice mode prioritizes voiced cast first
  assert.deepEqual(
    plain(c.matchCastMentions(cast, "", { mode: "voice" }).map((p) => p.name)),
    ["Narrator", "Solaris", "Sorceress", "Old Keeper"],
  );

  // Applying a mention inserts trailing space unless followed by whitespace or punctuation
  assert.deepEqual(
    plain(c.applyCastMention("Close-up on @sol", 12, 16, "Solaris")),
    { text: "Close-up on @Solaris ", caret: 21 },
  );
  assert.deepEqual(
    plain(c.applyCastMention("Close-up on @sol, smiling", 12, 16, "Solaris")),
    { text: "Close-up on @Solaris, smiling", caret: 20 },
  );
});

test("unified text encoder choice switches cleanly between Automatic, Universal Conditioner, and local encoders", () => {
  const c = vm.createContext({});
  functions(c, ["textEncoderValue", "applyTextEncoderChoice"]);
  const plain = (v) => JSON.parse(JSON.stringify(v));

  const project = {};
  assert.equal(c.textEncoderValue(project.models), "");

  // Selecting a Universal Conditioner adapter clears local clip/projection and keeps VAE intact
  project.models = { clip: "qwen3vl_4b.safetensors", projection: "mmh3-4b.safetensors", vae: "vae_int8.safetensors" };
  c.applyTextEncoderChoice(project, "conditioner:unicond-h3-v4.4-6l-dual-seq.safetensors");
  assert.deepEqual(plain(project.models), {
    vae: "vae_int8.safetensors",
    conditioner: "unicond-h3-v4.4-6l-dual-seq.safetensors",
  });
  assert.equal(c.textEncoderValue(project.models), "conditioner:unicond-h3-v4.4-6l-dual-seq.safetensors");

  // Switching to a local text encoder clears conditioner while keeping VAE
  c.applyTextEncoderChoice(project, "clip:qwen3vl_4b_int8_convrot.safetensors");
  assert.deepEqual(plain(project.models), {
    vae: "vae_int8.safetensors",
    clip: "qwen3vl_4b_int8_convrot.safetensors",
  });
  assert.equal(c.textEncoderValue(project.models), "clip:qwen3vl_4b_int8_convrot.safetensors");

  // Switching back to Automatic deletes models entirely when nothing else is set
  delete project.models.vae;
  c.applyTextEncoderChoice(project, "");
  assert.equal(project.models, undefined);
  assert.equal(c.textEncoderValue(project.models), "");
});

test("sprite inspector helpers anchor cursor zoom, fit cleanly, auto-enable HD 256, and match directional references", () => {
  // Fit zoom frames a 64x64 sprite cleanly inside a 600x360 stage with padding and never overflows
  const fitZ = fitZoomFor(600, 360, 64, 64);
  assert.ok(fitZ >= 4.5 && fitZ <= 5, `expected ~4.75 fit zoom, got ${fitZ}`);
  assert.ok(64 * fitZ <= 360 - 56, "fitted sprite must leave vertical padding with zero head clipping");

  // Cursor-anchored zoom keeps the exact sprite pixel under (cursorX, cursorY) stationary
  const stageW = 800, stageH = 400;
  const startCam = { zoom: 4, panX: 20, panY: -10, fit: false };
  const cursorX = 540, cursorY = 130;
  const worldBeforeX = (cursorX - (stageW / 2 + startCam.panX)) / startCam.zoom;
  const worldBeforeY = (cursorY - (stageH / 2 + startCam.panY)) / startCam.zoom;
  const nextCam = zoomAtPoint(startCam, 8, cursorX, cursorY, stageW, stageH, { baseW: 64, baseH: 64 });
  const worldAfterX = (cursorX - (stageW / 2 + nextCam.panX)) / nextCam.zoom;
  const worldAfterY = (cursorY - (stageH / 2 + nextCam.panY)) / nextCam.zoom;
  assert.ok(Math.abs(worldAfterX - worldBeforeX) < 1e-6, "cursor X drifted during zoom");
  assert.ok(Math.abs(worldAfterY - worldBeforeY) < 1e-6, "cursor Y drifted during zoom");
  assert.equal(nextCam.fit, false);

  // Zoom respects min/max clamping
  assert.equal(zoomAtPoint(startCam, 99, 400, 200, stageW, stageH).zoom, 12);
  assert.equal(zoomAtPoint(startCam, 0.05, 400, 200, stageW, stageH).zoom, 0.5);

  // Auto HD 256 engages at >= 2x zoom while manual overrides take precedence
  assert.equal(shouldUseHd("auto", 1.5), false);
  assert.equal(shouldUseHd("auto", 2), true);
  assert.equal(shouldUseHd("auto", 5.5), true);
  assert.equal(shouldUseHd("on", 1), true);
  assert.equal(shouldUseHd("off", 8), false);

  // Draft settings compact summary
  assert.equal(
    draftSettingsSummary({ canvas: "square", keyColor: "white", frames: 39, artStylePreset: "" }),
    "512² · 39f · White key · Match refs",
  );
  assert.equal(
    draftSettingsSummary({ canvas: "portrait", keyColor: "green", frames: 56, artStylePreset: "pixel" }),
    "256×384 · 56f · Green key · Pixel art",
  );

  // Compass maturity badges
  assert.equal(compassStatusFor(null, true).kind, "rendering");
  assert.equal(compassStatusFor(null, false).kind, "empty");
  assert.equal(compassStatusFor({ mirror: true, take: { quality: "draft" } }, false).kind, "mirror");
  assert.equal(compassStatusFor({ take: { quality: "final" } }, false).kind, "final");
  assert.equal(compassStatusFor({ clip: { sprite: { self_pin: { file: "p.png" } } }, take: { quality: "draft" } }, false).kind, "loop");
  assert.equal(compassStatusFor({ clip: { sprite: {} }, take: { quality: "draft" } }, false).kind, "draft");

  // Heading-matched reference selection
  const cast = {
    id: "p1",
    name: "Nova",
    file: "front.png",
    view: "front",
    pictures: [
      { file: "side.png", view: "side" },
      { file: "back.png", view: "back" },
    ],
  };
  assert.equal(referenceForDirection(cast, "S").asset.file, "front.png");
  assert.equal(referenceForDirection(cast, "N").asset.file, "back.png");
  assert.equal(referenceForDirection(cast, "NE").asset.file, "back.png");
  assert.equal(referenceForDirection(cast, "E").asset.file, "side.png");
  assert.equal(referenceForDirection(cast, "W").asset.file, "side.png");
  assert.equal(referenceForDirection(cast, "SE").index, 0);
});

test("screenToTilePoint maps zoomed and panned canvas coordinates accurately into tile mask space", () => {
  // Exact origin, center, and bottom-right on a zoomed/panned 64x64 sprite drawn at 6x (384x384) into a 128x128 mask
  const origin = screenToTilePoint(120, 40, 120, 40, 384, 384, 128, 128);
  assert.deepEqual(origin, { x: 0, y: 0, u: 0, v: 0, inside: true });

  const center = screenToTilePoint(312, 232, 120, 40, 384, 384, 128, 128);
  assert.deepEqual(center, { x: 64, y: 64, u: 0.5, v: 0.5, inside: true });

  const corner = screenToTilePoint(504, 424, 120, 40, 384, 384, 128, 128);
  assert.deepEqual(corner, { x: 128, y: 128, u: 1, v: 1, inside: true });

  // Non-square portrait tile (96x128 mask) at 4x zoom with negative pan offset
  const quarter = screenToTilePoint(-10 + 64, 20 + 192, -10, 20, 256, 384, 96, 128);
  assert.equal(quarter.inside, true);
  assert.equal(quarter.x, 24);
  assert.equal(quarter.y, 64);

  // Out-of-bounds coordinates clamp cleanly to [0, maskW] x [0, maskH] and mark inside=false
  const outside = screenToTilePoint(80, 500, 120, 40, 384, 384, 128, 128);
  assert.equal(outside.inside, false);
  assert.equal(outside.x, 0);
  assert.equal(outside.y, 128);
  assert.ok(outside.u < 0 && outside.v > 1);
});
