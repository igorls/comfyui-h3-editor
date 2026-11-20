// Exercise the real editor functions with deterministic media clocks and a fake server.
// The editor has no bundler; extracting top-level functions keeps these tests dependency-free.
import { readFileSync } from "node:fs";
import vm from "node:vm";
import {
  buildAtlasPayload,
  buildPropPayload,
  buildSpriteConfig,
  CAST_VIEWS,
  defaultCanvasForCamera,
  parseTileSize,
  pickedSprite,
  spriteSlots,
} from "../editor/sprites.js";
import test from "node:test";
import assert from "node:assert/strict";
import { curveLevel, soundEnvelope, setSoundLevel, resumeSoundMixer, disconnectSound } from "../editor/sound-envelope.js";
import { formatJobMeta, mergeQueueState, relativeTime, summarizeQueue } from "../editor/queue.js";

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
