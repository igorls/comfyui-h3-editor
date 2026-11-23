const ids = ["NW", "N", "NE", "W", "", "E", "SW", "S", "SE"];
const presets = new Map([["idle", "Idle"], ["run", "Run"], ["custom", "Custom motion"]]);
export const CANVASES = ["square", "portrait", "landscape"];
export const SPRITE_LENGTHS = [39, 56, 73];
export const SPRITE_STYLE_PRESETS = {
  pixel: "16-bit SNES pixel art RPG sprite, crisp dark pixel outlines, clean limited-palette shading, no anti-aliasing blur",
  chibi: "Cozy 2.5-head chibi fantasy RPG illustration, crisp dark ink outlines, clean cel shading",
};
export const CAST_VIEWS = [
  ["", "Any view"],
  ["front", "Front (S)"],
  ["side", "Side (E)"],
  ["back", "Back (N)"],
  ["left", "Left (W)"],
];
const $ = (id) => document.getElementById(id);
const opposite = (direction) => direction.replace(/[EW]/, (c) => (c === "E" ? "W" : "E"));
const directions = (count) => ids.filter((d) => d && (count === 8 || d.length === 1));
const motionOf = (s) => s.animation || (s.motion === "idle" ? "idle" : s.guide || s.motion === "prompt" ? "custom" : "run");
const actorOf = (clip, cast) => cast?.[0] || clip.sprite_group || "clip-" + clip.id;
const slotKey = (camera, motion, direction) => [camera, motion, direction].join("/");

export function defaultCanvasForCamera(camera) {
  return camera === "rpg" ? "square" : "portrait";
}

export function resolveSpriteStyle(preset = "", custom = "") {
  if (!preset) return "";
  if (preset === "custom") return String(custom || "").trim();
  return SPRITE_STYLE_PRESETS[preset] || String(preset).trim();
}

export const SPRITE_BACKDROPS = ["white", "green", "blue"];

export function buildSpriteConfig({ camera = "rpg", motion = "run", direction = "S", motionPrompt = null, canvas, frames = 39, poses = 7, style = "", loop = false, backdrop = "white" } = {}) {
  const config = { camera, direction, poses };
  if (motion === "idle") config.motion = "idle";
  if (motionPrompt) Object.assign(config, { motion: "prompt", animation: motion });
  // A named animation is a gesture played once unless it is marked as a cycle (a walk), which is cut like the run.
  if (motionPrompt && loop) config.loop = true;
  if (SPRITE_BACKDROPS.includes(backdrop) && backdrop !== "white") config.backdrop = backdrop;
  const chosenCanvas = canvas || defaultCanvasForCamera(camera);
  if (CANVASES.includes(chosenCanvas)) config.canvas = chosenCanvas;
  const len = Number(frames);
  if (Number.isInteger(len) && len >= 22 && (len - 5) % 17 === 0 && len !== 39) config.frames = len;
  const cleanedStyle = String(style || "").trim();
  if (cleanedStyle) config.style = cleanedStyle;
  return config;
}

export function parseTileSize(selectValue, customText = "") {
  const validEven = (n) => Number.isInteger(n) && n >= 32 && n <= 512 && n % 2 === 0;
  if (selectValue !== "custom") {
    const n = Number(selectValue);
    if (!validEven(n)) throw new Error("Choose an even export tile size between 32 and 512 px.");
    return n;
  }
  const raw = String(customText || "").trim().toLowerCase();
  if (!raw) throw new Error("Enter an even export size (32–512), e.g. 160 or 144x176.");
  const parts = raw.split(/[x×,\s]+/).filter(Boolean);
  if (parts.length === 1) {
    const n = Number(parts[0]);
    if (!validEven(n)) throw new Error("Tile size must be an even integer in 32–512 px.");
    return n;
  }
  if (parts.length === 2) {
    const w = Number(parts[0]), h = Number(parts[1]);
    if (!validEven(w) || !validEven(h)) throw new Error("Tile width and height must be even integers in 32–512 px.");
    return w === h ? w : [w, h];
  }
  throw new Error("Use a single size like 160 or width x height like 144x176.");
}

export function buildAtlasPayload({ tile, customTile = "", fps = "", style = "illustrated", webp = true, picks = [] } = {}) {
  const parsed = parseTileSize(tile, customTile);
  const payload = {
    tile: Array.isArray(parsed) ? parsed[0] : parsed,
    fps: Number(fps) || null,
    style,
    webp: Boolean(webp),
    picks,
  };
  if (Array.isArray(parsed)) payload.tile_size = parsed;
  return payload;
}

export function buildPropPayload(fields = {}) {
  const kind = fields.kind || "prop";
  if (!["prop", "state_variant", "turnaround"].includes(kind)) {
    throw new Error("Choose a prop, state variant, or turnaround workflow.");
  }
  const name = String(fields.name || "").trim();
  if (!name) throw new Error("Enter a name for this asset.");
  const description = String(fields.description || "").trim();
  if ((kind === "prop" || kind === "state_variant") && !description) {
    throw new Error(kind === "state_variant" ? "Describe the state change." : "Describe the prop to generate.");
  }
  const size = fields.size || "square";
  if (!CANVASES.includes(size)) throw new Error("Choose a 4 MP square, portrait, or landscape size.");
  const payload = { kind, name, description, size, camera: fields.camera || "rpg" };
  const styleText = String(fields.style || "").trim();
  if (styleText) payload.style = styleText;
  if (fields.style_ref?.file) {
    payload.style_ref = { file: fields.style_ref.file, subfolder: fields.style_ref.subfolder || "" };
  }
  if (fields.base_prop?.file) {
    payload.base_prop = { file: fields.base_prop.file, subfolder: fields.base_prop.subfolder || "" };
  }
  if (kind === "state_variant" && !payload.base_prop) {
    throw new Error("Select or upload a base prop image for a paired state variant.");
  }
  if (kind === "turnaround") {
    if (!payload.style_ref) throw new Error("Select a front style reference plate for a turnaround.");
    payload.view = fields.view || "back";
    payload.proportions = fields.proportions || "adult";
    const holding = String(fields.holding || "").trim();
    if (holding) payload.holding = holding;
  }
  if (kind === "prop") {
    if (fields.top_canopy) payload.top_canopy = true;
    const outRaw = String(fields.output_size || "").trim();
    if (outRaw) {
      const parsed = parseTileSize("custom", outRaw);
      payload.output_size = Array.isArray(parsed) ? parsed : [parsed, parsed];
    }
  }
  if (fields.dry_run) payload.dry_run = true;
  return payload;
}

function cutoutLabel(sheet) {
  const depth = sheet.depth ? " + depth" : "";
  if (sheet.cutout === "foreground model") return "BiRefNet" + depth;
  if (sheet.cutout === "background key") return "Background key" + depth;
  return "Cutout unrecorded";
}
function button(label, action, pressed) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "quiet";
  b.textContent = label;
  b.onclick = action;
  if (pressed !== undefined) b.setAttribute("aria-pressed", String(pressed));
  return b;
}

export function spriteSlots(project, actor, camera, motion) {
  const slots = Object.fromEntries(directions(8).map((dir) => [dir, { clips: [], takes: [] }]));
  for (const clip of project.clips) {
    function slot(config, cast) {
      if (!config || actorOf(clip, cast) !== actor || config.camera !== camera || motionOf(config) !== motion)
        return null;
      return slots[config.direction];
    }
    slot(clip.sprite, clip.cast)?.clips.push(clip);
    for (const take of clip.takes || []) {
      if (!take.sprite) continue;
      const saved = take.saved || {};
      const target = slot(saved.sprite || clip.sprite, saved.cast ?? clip.cast);
      if (target) target.takes.push({ clip, take });
    }
  }
  return slots;
}

// Include saved takes even when their source shot has since changed character or motion.
function animationsFor(project, actor) {
  const animations = new Map([["idle", null], ["run", null]]);
  function include(clip, config, cast, action) {
    if (!config || actorOf(clip, cast) !== actor) return;
    const key = motionOf(config);
    if (!animations.has(key)) animations.set(key, config.animation ? action : null);
  }
  for (const clip of project.clips) {
    include(clip, clip.sprite, clip.cast, clip.action);
    for (const t of clip.takes || []) {
      if (t.sprite) include(clip, t.saved?.sprite || clip.sprite, t.saved?.cast ?? clip.cast, t.saved?.action ?? clip.action);
    }
  }
  for (const { id, prompt } of project.sprite_workspaces?.[actor]?.animations || []) animations.set(id, prompt);
  return animations;
}

export function pickedSprite(slots, prefs, camera, motion, dir) {
  const pick = prefs.picks?.[slotKey(camera, motion, dir)];
  if (pick) {
    const source = pick.mirror ? opposite(dir) : dir;
    const match = slots[source].takes.find((x) => x.clip.id === pick.clip && x.take.id === pick.take);
    if (match) return { ...match, mirror: Boolean(pick.mirror) };
  }
  const takes = slots[dir].takes;
  return takes.find((x) => x.clip.active === x.take.id) || takes[0] || null;
}

export function installSprites(editor) {
  const room = $("spritesRoom"),
    stage = $("spriteArena");
  const ctx = stage.getContext("2d");
  $("spriteReferencePanel").open = !matchMedia("(max-width: 720px)").matches;
  const keys = new Set(),
    pictures = new Map();
  let pointer = null;
  const inputs = {
    arrowup: "N",
    w: "N",
    arrowdown: "S",
    s: "S",
    arrowleft: "W",
    a: "W",
    arrowright: "E",
    d: "E",
  };
  let folder,
    actor,
    camera = "rpg",
    canvas = "square",
    keyColor = "white",
    clipFrames = 39,
    artStylePreset = "",
    artStyleCustom = "",
    motion = "run",
    dir = "S",
    count = 8,
    slots,
    prefs,
    animations,
    frames = [];
  let paused = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let clock = 0,
    last = 0,
    raf = 0,
    x = 0.5,
    y = 0.68,
    zoom = 3,
    speed = 1,
    backdrop = "matte",
    showPivot = false,
    hdMode = false,
    ground = null,
    busy = false,
    exporting = false;
  const hdSheets = new Map();
  let poseStripKey = "";
  let poseButtons = [];
  let audition = null;
  const ZOOM_LEVELS = [1, 2, 3, 4, 6];
  const open = () => !room.hidden;
  const project = () => editor.state.project;
  const chosen = (direction = dir) => pickedSprite(slots, prefs, camera, motion, direction);
  const current = () => audition || chosen();
  const style = () => $("spriteStyle").value;
  function sheetUrl(row, overrideSheet = null) {
    const sh = overrideSheet || row.take.sprite;
    return editor.viewUrl(sh[style()], sh.subfolder || row.take.sprite.subfolder || row.take.subfolder);
  }
  function hdKey(row) {
    return [folder, row.clip.id, row.take.id].join(":");
  }
  function activeDisplaySheet(row) {
    if (!row) return null;
    if (hdMode) {
      const cached = hdSheets.get(hdKey(row));
      if (cached && !cached.loading && !cached.error) return cached.sheet;
    }
    return row.take.sprite;
  }
  async function ensureHd(row) {
    if (!hdMode || !row || !editor.inspectSprite) return;
    const key = hdKey(row);
    const baseTile = row.take.sprite.tile_size?.[0] || row.take.sprite.tile || 64;
    if (baseTile >= 256 || hdSheets.has(key)) return;
    hdSheets.set(key, { loading: true });
    refresh();
    try {
      const sheet = await editor.inspectSprite(row.clip, row.take, 256);
      hdSheets.set(key, { loading: false, sheet });
    } catch (error) {
      hdSheets.set(key, { loading: false, error: error.message || "HD preview unavailable" });
      editor.toast("HD 256 preview unavailable: " + (error.message || "using base strip"), true);
    }
    refresh();
  }
  function remember(edit) {
    editor.checkpoint();
    const all = (project().sprite_workspaces ||= {});
    prefs = all[actor] ||= {};
    edit(prefs);
    editor.save();
  }
  function select(row, target = dir) {
    remember((p) => {
      const picks = (p.picks ||= {});
      picks[slotKey(camera, motion, target)] = { clip: row.clip.id, take: row.take.id, mirror: !!row.mirror };
    });
    audition = null;
    refresh();
  }
  function frame(row) {
    const node = document.createElement("i");
    node.className = "sprite-frame";
    node.style.backgroundImage = "url(" + sheetUrl(row) + ")";
    node.style.backgroundSize = 64 * row.take.sprite.poses + "px 64px";
    if (row.mirror) node.style.transform = "scaleX(-1)";
    frames.push([node, row]);
    return node;
  }
  function pickActor(id) {
    actor = id;
    prefs = project().sprite_workspaces?.[actor] || {};
    camera = prefs.camera || "rpg";
    canvas = CANVASES.includes(prefs.canvas) ? prefs.canvas : defaultCanvasForCamera(camera);
    keyColor = SPRITE_BACKDROPS.includes(prefs.backdrop) ? prefs.backdrop : "white";
    clipFrames = SPRITE_LENGTHS.includes(Number(prefs.frames)) ? Number(prefs.frames) : 39;
    artStylePreset = prefs.art_style_preset || "";
    artStyleCustom = prefs.art_style_custom || "";
    count = prefs.count || 8;
    motion = prefs.motion || "run";
    dir = "S";
    audition = null;
  }
  function focusClip(clip) {
    const t = clip?.takes?.find((t) => t.id === clip.active && t.sprite);
    const config = t?.saved?.sprite || clip?.sprite;
    if (!config) return;
    pickActor(actorOf(clip, t?.saved?.cast ?? clip.cast));
    camera = config.camera;
    canvas = CANVASES.includes(config.canvas) ? config.canvas : "portrait";
    keyColor = SPRITE_BACKDROPS.includes(config.backdrop) ? config.backdrop : "white";
    clipFrames = SPRITE_LENGTHS.includes(Number(config.frames)) ? Number(config.frames) : 39;
    motion = motionOf(config);
    dir = config.direction;
    if (dir.length === 2) count = 8;
    if (t) audition = { clip, take: t };
  }
  function flipMode(enabled, clip = null) {
    if (!project()) return;
    room.hidden = !enabled;
    $("app").classList.toggle("sprites-open", enabled);
    $("videoMode").setAttribute("aria-pressed", String(!enabled));
    $("spritesMode").setAttribute("aria-pressed", String(enabled));
    keys.clear();
    pointer = null;
    if (!enabled) {
      cancelAnimationFrame(raf);
      raf = 0;
      editor.restoreVideo();
      return;
    }
    editor.stopVideo();
    if (folder !== editor.state.folder) reset();
    if (clip) focusClip(clip);
    refresh();
    if (!raf) raf = requestAnimationFrame(tick);
  }
  function reset() {
    folder = editor.state.folder;
    actor = null;
    audition = null;
    ground = null;
    pictures.clear();
    hdSheets.clear();
    poseStripKey = "";
    poseButtons = [];
    keys.clear();
    pointer = null;
    x = 0.5;
    y = 0.68;
    clock = 0;
    last = 0;
  }
  function roster() {
    const people = new Map(
      (project().cast || []).filter((p) => p.file || p.pictures?.length || p.refmod).map((p) => [p.id, p]),
    );
    for (const clip of project().clips) {
      const add = (cfg, cast) => {
        if (!cfg) return;
        const id = actorOf(clip, cast);
        if (!people.has(id)) people.set(id, { id, name: clip.name });
      };
      add(clip.sprite, clip.cast);
      for (const t of clip.takes || [])
        if (t.sprite) add(t.saved?.sprite || clip.sprite, t.saved?.cast ?? clip.cast);
    }
    return people;
  }
  function refresh() {
    if (!open() || !project()) return;
    const focusKey = document.activeElement?.dataset.spriteFocus;
    if (folder !== editor.state.folder) reset();
    const people = roster();
    const background = project().sprite_ground;
    if (background?.file) {
      const url = editor.viewUrl(background.file, background.subfolder);
      if (ground?.getAttribute("src") !== url) {
        ground = new Image();
        ground.src = url;
      }
    } else ground = null;
    if (!people.has(actor)) pickActor(people.keys().next().value);
    prefs = project().sprite_workspaces?.[actor] || {};
    const person = people.get(actor);
    const cast = project().cast?.find((p) => p.id === actor);
    $("spritePeople").replaceChildren(
      ...Array.from(people.values(), (p) => {
        const b = button(
          p.name,
          () => {
            pickActor(p.id);
            refresh();
          },
          p.id === actor,
        );
        b.dataset.spriteFocus = "actor:" + p.id;
        return b;
      }),
    );
    $("spriteTitle").textContent = person?.name || "Add a character reference to start";
    const refs = $("spriteRefs");
    $("spriteReferencePanel").hidden = !cast;
    refs.replaceChildren();
    if (cast) {
      for (const asset of [cast, ...(cast.pictures || [])].filter((a) => a.file)) {
        const wrap = document.createElement("div");
        wrap.className = "sprite-ref-item";
        const img = document.createElement("img");
        img.alt = cast.name + (asset.view ? " (" + asset.view + ")" : "") + " reference";
        img.src = editor.viewUrl(asset.file, asset.subfolder);
        wrap.append(img);
        if (asset.view) {
          const tag = document.createElement("span");
          tag.className = "sprite-ref-tag";
          tag.textContent = asset.view;
          wrap.append(tag);
        }
        refs.append(wrap);
      }
      if (cast.refmod && editor.refmodVisual) {
        const wrap = document.createElement("div");
        wrap.className = "sprite-ref-item sprite-ref-mod";
        wrap.title = "RefMod: " + cast.refmod.file;
        wrap.append(editor.refmodVisual(cast.refmod.file, { alt: cast.name + " RefMod" }));
        refs.append(wrap);
      }
      const edit = button("Edit references", () => editor.editCast(cast.id));
      edit.dataset.spriteFocus = "references";
      refs.append(edit);
    }
    animations = animationsFor(project(), actor);
    if (!animations.has(motion)) motion = "run";
    slots = spriteSlots(project(), actor, camera, motion);
    frames = [];
    const motionSelect = $("spriteMotion");
    const options = Array.from(animations.keys());
    if (motionSelect.length !== options.length || options.some((m, i) => motionSelect.options[i].value !== m)) {
      motionSelect.replaceChildren(...options.map((m) => new Option(presets.get(m) || m, m)));
    }
    motionSelect.value = motion;
    $("spriteAddMotion").disabled = !actor;
    $("spriteEditMotion").hidden = presets.has(motion);
    $("spriteCamera").value = camera;
    $("spriteCanvas").value = canvas;
    $("spriteGround").value = keyColor;
    $("spriteFrames").value = String(clipFrames);
    $("spriteArtStyle").value = artStylePreset;
    $("spriteArtStyleCustom").value = artStyleCustom;
    $("spriteArtStyleCustomField").hidden = artStylePreset !== "custom";
    $("spriteCount").value = String(count);
    $("spriteTileCustomField").hidden = $("spriteTile").value !== "custom";
    $("spritePropBtn").disabled = !project();
    const compass = $("spriteCompass");
    compass.replaceChildren();
    const headings = directions(count);
    let filled = 0;
    for (const [pos, h] of ids.entries()) {
      if (!headings.includes(h)) continue;
      const b = button(
        "",
        () => {
          dir = h;
          audition = null;
          refresh();
        },
        h === dir,
      );
      b.style.gridArea = Math.floor(pos / 3) + 1 + " / " + ((pos % 3) + 1);
      const label = document.createElement("span");
      label.textContent = h;
      const row = chosen(h);
      b.dataset.spriteFocus = "direction:" + h;
      b.classList.toggle("empty", !row);
      if (row) {
        filled++;
        b.append(frame(row));
        b.title = h + " · " + cutoutLabel(row.take.sprite);
      } else {
        b.title = h + " · No selected take";
      }
      b.append(label);
      b.setAttribute("aria-label", h + (row?.mirror ? " mirrored" : row ? "" : " empty"));
      compass.append(b);
    }
    $("spriteCoverage").textContent = filled + " of " + count + " directions have takes";
    $("spritePlain").disabled = !background?.file;
    const show = current(),
      picked = chosen(),
      pending = slots[dir].clips.find((c) => editor.jobFor(c.id));
    const takes = $("spriteTakes");
    takes.replaceChildren();
    const rows = [...slots[dir].takes];
    if (show?.mirror) rows.push(show);
    for (const row of rows) {
      const label =
        (row.mirror ? "Mirrored " + opposite(dir) : row.take.quality === "draft" ? "Draft" : "Final") +
        " · " +
        row.take.seed;
      const selected = picked?.take.id === row.take.id && !!picked?.mirror === !!row.mirror;
      const b = button("", () => select(row), selected);
      b.dataset.spriteFocus = (row.mirror ? "mirror:" : "take:") + row.clip.id + ":" + row.take.id;
      const caption = document.createElement("span");
      caption.className = "sprite-take-label";
      caption.textContent = label;
      const film = document.createElement("span");
      film.className = "sprite-film";
      film.setAttribute("aria-hidden", "true");
      const sheet = row.take.sprite;
      for (let pose = 0; pose < sheet.poses; pose++) {
        const tile = document.createElement("i");
        tile.style.backgroundImage = "url(" + sheetUrl(row) + ")";
        tile.style.backgroundSize = sheet.poses * 32 + "px 32px";
        tile.style.backgroundPositionX = -pose * 32 + "px";
        if (row.mirror) tile.style.transform = "scaleX(-1)";
        film.append(tile);
      }
      const method = document.createElement("small");
      const savedCfg = row.take.saved?.sprite || row.clip.sprite;
      const meta = [cutoutLabel(row.take.sprite)];
      if (savedCfg?.canvas) meta.push(savedCfg.canvas);
      if (row.take.frames && row.take.frames !== 39) meta.push(row.take.frames + "f");
      if (savedCfg?.self_pin) meta.push("self-pinned");
      method.textContent = meta.join(" · ");
      b.append(frame(row), caption, method, film);
      takes.append(b);
    }
    if (!rows.length) takes.textContent = pending ? "Rendering…" : "No takes for this direction.";
    $("spriteTakesTitle").textContent = dir + " takes";
    $("spriteDraft").textContent = pending ? "Rendering " + dir + "…" : "New " + dir + " draft";
    $("spriteDraft").disabled =
      !actor || !!pending || busy || (motion === "custom" && !slots[dir].clips.length);
    $("spriteSelfPin").disabled =
      !show || !!show.mirror || motion === "idle" || !!editor.jobFor(show.clip.id) || busy;
    $("spriteEdit").disabled = !show && !slots[dir].clips.length;
    $("spriteFinish").disabled =
      !show || show.take.quality !== "draft" || !!editor.jobFor(show.clip.id) || busy;
    const partner = opposite(dir);
    $("spriteMirror").hidden = partner === dir;
    $("spriteMirror").textContent = audition?.mirror
      ? "Use mirrored " + partner
      : "Preview mirrored " + partner;
    $("spriteMirror").disabled = !slots[partner]?.takes.length;
    $("spritePlay").textContent = paused ? "Play" : "Pause";
    $("spriteZoom").value = String(zoom);
    $("spriteBackdrop").value = backdrop;
    $("spriteSpeed").value = String(speed);
    $("spritePivot").setAttribute("aria-pressed", String(showPivot));
    const hdEntry = show ? hdSheets.get(hdKey(show)) : null;
    $("spriteHd").setAttribute("aria-pressed", String(hdMode));
    $("spriteHd").disabled = !show;
    $("spriteHd").textContent = hdMode && hdEntry?.loading ? "HD…" : "HD 256";
    if (hdMode && show) void ensureHd(show);
    syncPoseStrip(show);
    const ready = exportPicks().length;
    $("spriteAtlas").disabled = !actor || exporting || !ready;
    $("spriteAtlas").textContent = ready
      ? "Export atlas · " + ready + (ready === 1 ? " loop" : " loops")
      : "Export atlas";
    $("spriteStatus").textContent = pending
      ? "Queued or rendering"
      : slots[dir].clips.find((c) => c.error)?.error ||
        (audition?.mirror ? "Previewing mirror" : audition ? "Previewing editor take" : "");
    if (focusKey) room.querySelector('[data-sprite-focus="' + CSS.escape(focusKey) + '"]')?.focus({ preventScroll: true });
    paint();
  }
  function exportPicks() {
    const out = [];
    for (const m of animations.keys()) {
      const s = spriteSlots(project(), actor, camera, m);
      for (const h of directions(count)) {
        const row = pickedSprite(s, prefs, camera, m, h);
        if (row)
          out.push({ motion: m, direction: h, clip: row.clip.id, take: row.take.id, mirror: !!row.mirror });
      }
    }
    return out;
  }
  function currentPoseIndex(sh) {
    if (!sh || !sh.poses) return 0;
    const raw = Math.floor(clock * sh.fps) % sh.poses;
    return ((raw % sh.poses) + sh.poses) % sh.poses;
  }
  function stepPose(delta) {
    const row = current();
    if (!row) return;
    paused = true;
    const sh = row.take.sprite;
    const next = (currentPoseIndex(sh) + delta + sh.poses) % sh.poses;
    clock = (next + 0.01) / (sh.fps || 12);
    refresh();
  }
  function jumpToPose(index) {
    const row = current();
    if (!row) return;
    paused = true;
    const sh = row.take.sprite;
    clock = (index + 0.01) / (sh.fps || 12);
    refresh();
  }
  function syncPoseStrip(row) {
    const strip = $("spritePoseStrip");
    if (!strip) return;
    if (!row) {
      poseStripKey = "";
      poseButtons = [];
      strip.replaceChildren();
      return;
    }
    const sh = activeDisplaySheet(row);
    const url = sheetUrl(row, sh);
    const key = [row.clip.id, row.take.id, row.mirror ? "m" : "", url, backdrop, sh.poses].join("|");
    if (key !== poseStripKey) {
      poseStripKey = key;
      poseButtons = [];
      strip.replaceChildren();
      const backdropClass = backdrop !== "matte" ? " backdrop-" + backdrop : "";
      for (let idx = 0; idx < sh.poses; idx++) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "sprite-pose-tile" + backdropClass;
        btn.title = "Inspect pose " + (idx + 1) + " of " + sh.poses;
        btn.setAttribute("aria-label", "Pose " + (idx + 1) + " of " + sh.poses);
        btn.onclick = () => jumpToPose(idx);
        const cell = document.createElement("i");
        cell.style.backgroundImage = "url(" + url + ")";
        cell.style.backgroundSize = sh.poses * 72 + "px 72px";
        cell.style.backgroundPositionX = -idx * 72 + "px";
        if (row.mirror) cell.style.transform = "scaleX(-1)";
        const num = document.createElement("span");
        num.textContent = String(idx + 1);
        btn.append(cell, num);
        poseButtons.push(btn);
        strip.append(btn);
      }
    }
    const active = currentPoseIndex(row.take.sprite);
    for (let i = 0; i < poseButtons.length; i++) {
      poseButtons[i].setAttribute("aria-pressed", String(i === active));
    }
  }
  function drawBackdrop(w, h) {
    if (ground?.naturalWidth) {
      ctx.fillStyle = "#555b5b";
      ctx.fillRect(0, 0, w, h);
      const scale = Math.max(w / ground.width, h / ground.height);
      ctx.drawImage(
        ground,
        (w - ground.width * scale) / 2,
        (h - ground.height * scale) / 2,
        ground.width * scale,
        ground.height * scale,
      );
      return;
    }
    if (backdrop === "dark") {
      ctx.fillStyle = "#141619";
      ctx.fillRect(0, 0, w, h);
    } else if (backdrop === "light") {
      ctx.fillStyle = "#ecebe6";
      ctx.fillRect(0, 0, w, h);
    } else if (backdrop === "checker") {
      const sq = 14;
      ctx.fillStyle = "#464c56";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "#2d3138";
      for (let yy = 0; yy < h; yy += sq) {
        for (let xx = ((yy / sq) % 2) * sq; xx < w; xx += sq * 2) {
          ctx.fillRect(xx, yy, sq, sq);
        }
      }
    } else {
      ctx.fillStyle = "#555b5b";
      ctx.fillRect(0, 0, w, h);
    }
  }
  function paint() {
    if (!slots || !open()) return;
    const reviewed = current()?.take.sprite;
    const phase = reviewed ? (((clock * reviewed.fps) % reviewed.poses) + reviewed.poses) % reviewed.poses / reviewed.poses : 0;
    for (const [node, row] of frames) {
      const sh = row.take.sprite;
      const pose = reviewed ? Math.floor(phase * sh.poses) : currentPoseIndex(sh);
      node.style.backgroundPositionX = -pose * 64 + "px";
    }
    const w = Math.max(1, stage.clientWidth),
      h = Math.max(1, stage.clientHeight);
    if (stage.width !== w || stage.height !== h) {
      stage.width = w;
      stage.height = h;
    }
    drawBackdrop(w, h);
    const row = current();
    const readout = $("spritePoseReadout");
    if (!row) {
      if (readout) readout.textContent = "WASD or arrow keys · Scroll to zoom";
      return;
    }
    const baseSh = row.take.sprite;
    const sh = activeDisplaySheet(row);
    const url = sheetUrl(row, sh);
    if (!pictures.has(url)) {
      const img = new Image();
      img.src = url;
      pictures.set(url, img);
    }
    const img = pictures.get(url);
    const pose = currentPoseIndex(baseSh);
    for (let i = 0; i < poseButtons.length; i++) {
      poseButtons[i].setAttribute("aria-pressed", String(i === pose));
    }
    if (readout) {
      const hdTag = sh !== baseSh ? " · HD " + (sh.tile_size?.[0] || sh.tile) + " px" : "";
      const speedTag = speed !== 1 ? ` (${speed}×)` : "";
      readout.textContent = `Pose ${pose + 1}/${baseSh.poses} · ${baseSh.fps} fps${speedTag} · ${zoom}×${hdTag}`;
    }
    if (!img.naturalWidth) return;
    const srcW = sh.tile_size?.[0] || sh.tile,
      srcH = sh.tile_size?.[1] || sh.tile,
      baseW = baseSh.tile_size?.[0] || baseSh.tile,
      baseH = baseSh.tile_size?.[1] || baseSh.tile,
      drawW = Math.round(baseW * zoom),
      drawH = Math.round(baseH * zoom),
      pivX = Math.round(baseSh.pivot[0] * zoom),
      pivY = Math.round(baseSh.pivot[1] * zoom),
      cx = Math.round(x * w),
      cy = Math.round(y * h);
    ctx.save();
    ctx.imageSmoothingEnabled = sh !== baseSh && srcW > drawW;
    ctx.translate(cx, cy);
    if (row.mirror) ctx.scale(-1, 1);
    ctx.drawImage(img, pose * srcW, 0, srcW, srcH, -pivX, -pivY, drawW, drawH);
    ctx.restore();
    if (showPivot) {
      ctx.save();
      ctx.strokeStyle = "rgba(240, 179, 84, 0.55)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(cx - pivX + 0.5, cy - pivY + 0.5, drawW - 1, drawH - 1);
      ctx.setLineDash([]);
      ctx.strokeStyle = "#f0b354";
      ctx.beginPath();
      ctx.moveTo(cx - 10, cy + 0.5);
      ctx.lineTo(cx + 10, cy + 0.5);
      ctx.moveTo(cx + 0.5, cy - 10);
      ctx.lineTo(cx + 0.5, cy + 10);
      ctx.stroke();
      ctx.restore();
    }
  }
  function tick(now) {
    if (!open()) return;
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
    last = now;
    let dx = Number(keys.has("E")) - Number(keys.has("W")),
      dy = Number(keys.has("S")) - Number(keys.has("N"));
    if (!dx && !dy && pointer) {
      const gapX = (pointer.x - x) * stage.width,
        gapY = (pointer.y - y) * stage.height;
      dx = Math.abs(gapX) > 4 ? Math.sign(gapX) : 0;
      dy = Math.abs(gapY) > 4 ? Math.sign(gapY) : 0;
    }
    if (count === 4 && dx && dy) dx = 0;
    if (dx || dy) {
      const h = (dy < 0 ? "N" : dy > 0 ? "S" : "") + (dx < 0 ? "W" : dx > 0 ? "E" : "");
      if (h !== dir) {
        dir = h;
        audition = null;
        refresh();
      }
      const moveSpeed = (90 * dt) / Math.hypot(dx, dy);
      x = Math.max(0.06, Math.min(0.94, x + (dx * moveSpeed) / stage.width));
      y = Math.max(0.33, Math.min(0.94, y + (dy * moveSpeed) / stage.height));
    }
    if (!paused) clock += dt * speed;
    paint();
    raf = requestAnimationFrame(tick);
  }
  function stepZoom(delta) {
    const idx = ZOOM_LEVELS.indexOf(zoom);
    const next = Math.max(0, Math.min(ZOOM_LEVELS.length - 1, (idx === -1 ? 2 : idx) + delta));
    if (ZOOM_LEVELS[next] !== zoom) {
      zoom = ZOOM_LEVELS[next];
      refresh();
    }
  }
  stage.addEventListener("wheel", (e) => {
    e.preventDefault();
    stepZoom(e.deltaY < 0 ? 1 : -1);
  }, { passive: false });
  stage.onkeydown = (e) => {
    if (e.key === " ") {
      e.preventDefault();
      paused = !paused;
      refresh();
      return;
    }
    if (e.key === "[" || e.key === ",") {
      e.preventDefault();
      stepPose(-1);
      return;
    }
    if (e.key === "]" || e.key === ".") {
      e.preventDefault();
      stepPose(1);
      return;
    }
    if (e.key === "+" || e.key === "=") {
      e.preventDefault();
      stepZoom(1);
      return;
    }
    if (e.key === "-") {
      e.preventDefault();
      stepZoom(-1);
      return;
    }
    const h = inputs[e.key.toLowerCase()];
    if (h) {
      e.preventDefault();
      keys.add(h);
    }
  };
  stage.onkeyup = (e) => {
    const h = inputs[e.key.toLowerCase()];
    if (h) {
      e.preventDefault();
      keys.delete(h);
    }
  };
  const point = (e) => {
    const bounds = stage.getBoundingClientRect();
    pointer = { x: (e.clientX - bounds.left) / bounds.width, y: (e.clientY - bounds.top) / bounds.height };
  };
  stage.onpointerdown = (e) => {
    stage.focus();
    stage.setPointerCapture(e.pointerId);
    point(e);
  };
  stage.onpointermove = (e) => {
    if (pointer) point(e);
  };
  stage.onpointerup = stage.onpointercancel = () => {
    pointer = null;
  };
  const stopMoving = () => {
    keys.clear();
    pointer = null;
  };
  stage.onblur = stopMoving;
  window.addEventListener("blur", stopMoving);
  $("spritesMode").onclick = () => flipMode(true);
  $("videoMode").onclick = () => flipMode(false);
  $("spriteAdd").onclick = async () => {
    await editor.addCast();
    if (editor.state.person) pickActor(editor.state.person);
    refresh();
  };
  $("spriteCamera").onchange = (e) => {
    camera = e.target.value;
    if (!prefs?.canvas) canvas = defaultCanvasForCamera(camera);
    audition = null;
    if (actor) remember((p) => (p.camera = camera));
    refresh();
  };
  $("spriteCanvas").onchange = (e) => {
    canvas = e.target.value;
    if (actor) remember((p) => (p.canvas = canvas));
    refresh();
  };
  $("spriteGround").onchange = (e) => {
    keyColor = e.target.value;
    if (actor) remember((p) => (keyColor === "white" ? delete p.backdrop : (p.backdrop = keyColor)));
    refresh();
  };
  $("spriteFrames").onchange = (e) => {
    clipFrames = Number(e.target.value);
    if (actor) remember((p) => (p.frames = clipFrames));
    refresh();
  };
  $("spriteArtStyle").onchange = (e) => {
    artStylePreset = e.target.value;
    if (actor) remember((p) => (p.art_style_preset = artStylePreset));
    refresh();
    if (artStylePreset === "custom") $("spriteArtStyleCustom").focus();
  };
  $("spriteArtStyleCustom").onchange = (e) => {
    artStyleCustom = e.target.value.trim();
    if (actor) remember((p) => (p.art_style_custom = artStyleCustom));
  };
  $("spriteMotion").onchange = (e) => {
    motion = e.target.value;
    audition = null;
    if (actor) remember((p) => (p.motion = motion));
    refresh();
  };
  const motionDialog = $("spriteMotionDialog");
  const nameInput = $("spriteMotionName");
  const promptInput = $("spriteMotionPrompt");
  const loopInput = $("spriteMotionLoop");
  const loops = (name) => !!project().sprite_workspaces?.[actor]?.animations?.find((def) => def.id === name)?.loop;
  function editMotion(editing) {
    const owner = actor, projectFolder = folder;
    nameInput.readOnly = editing;
    nameInput.value = editing ? motion : "";
    promptInput.value = editing ? animations.get(motion) || "" : "";
    loopInput.checked = editing ? loops(motion) : false;
    nameInput.setCustomValidity("");
    promptInput.setCustomValidity("");
    $("spriteMotionTitle").textContent = editing ? "Edit animation prompt" : "Add animation";
    $("spriteMotionSave").textContent = editing ? "Save prompt" : "Add animation";
    $("spriteMotionForm").onsubmit = (e) => {
      e.preventDefault();
      if (folder !== projectFolder || actor !== owner) return motionDialog.close();
      const name = nameInput.value.trim(), prompt = promptInput.value.trim(), loop = loopInput.checked;
      if (!editing) {
        const taken = [...presets.keys(), ...presets.values(), ...animations.keys()];
        const duplicate = taken.some((m) => m.toLowerCase() === name.toLowerCase());
        nameInput.setCustomValidity(duplicate ? "Choose a different animation name." : "");
        if (!name) nameInput.setCustomValidity("Enter an animation name.");
      }
      promptInput.setCustomValidity(prompt ? "" : "Describe the motion.");
      if (!e.target.reportValidity()) return;
      remember((p) => {
        const definitions = (p.animations ||= []);
        const existing = definitions.find((def) => def.id === name);
        const definition = existing || { id: name, prompt };
        if (!existing) definitions.push(definition);
        definition.prompt = prompt;
        if (loop) definition.loop = true;
        else delete definition.loop;
        if (editing) {
          for (const clip of project().clips) {
            if (actorOf(clip, clip.cast) !== actor || clip.sprite?.animation !== name) continue;
            clip.action = prompt;
            if (loop) clip.sprite.loop = true;
            else delete clip.sprite.loop;
          }
        }
        p.motion = name;
      });
      motion = name;
      audition = null;
      motionDialog.close();
      refresh();
    };
    motionDialog.showModal();
    (editing ? promptInput : nameInput).focus();
  }
  nameInput.oninput = () => nameInput.setCustomValidity("");
  promptInput.oninput = () => promptInput.setCustomValidity("");
  $("spriteMotionCancel").onclick = () => motionDialog.close();
  $("spriteAddMotion").onclick = () => editMotion(false);
  $("spriteEditMotion").onclick = () => editMotion(true);

  const propDialog = $("spritePropDialog");
  const propKind = $("spritePropKind");
  const propName = $("spritePropName");
  const propDesc = $("spritePropDescription");
  const propSize = $("spritePropSize");
  const propStyleRef = $("spritePropStyleRef");
  const propBaseRef = $("spritePropBaseRef");
  const propBaseField = $("spritePropBaseField");
  const propOptions = $("spritePropOptions");
  const propTurnaroundOptions = $("spritePropTurnaroundOptions");
  const propStatus = $("spritePropStatus");
  let uploadedBaseAsset = null;

  function castAssetOptions() {
    const out = [];
    const members = [...(project()?.cast || [])];
    members.sort((a, b) => (a.id === actor ? -1 : b.id === actor ? 1 : 0));
    for (const member of members) {
      for (const [idx, asset] of [member, ...(member.pictures || [])].filter((a) => a?.file).entries()) {
        const tag = asset.view ? " · " + asset.view : idx === 0 ? " · main" : " · #" + (idx + 1);
        out.push({
          label: member.name + tag,
          asset: { file: asset.file, subfolder: asset.subfolder || "" },
        });
      }
    }
    return out;
  }

  function populatePropRefs() {
    const assets = castAssetOptions();
    const fill = (select, noneLabel, extra) => {
      const prev = select.value;
      select.replaceChildren(new Option(noneLabel, ""));
      if (extra?.file) {
        select.append(new Option("Uploaded: " + extra.file, JSON.stringify(extra)));
      }
      for (const item of assets) {
        select.append(new Option(item.label, JSON.stringify(item.asset)));
      }
      if (prev && Array.from(select.options).some((o) => o.value === prev)) select.value = prev;
      else if (select === propStyleRef && assets.length) select.value = JSON.stringify(assets[0].asset);
    };
    fill(propStyleRef, "None (prompt only)");
    fill(propBaseRef, "None", uploadedBaseAsset);
  }

  function syncPropDialogKind() {
    const kind = propKind.value;
    propBaseField.hidden = kind === "prop";
    propOptions.hidden = kind !== "prop";
    propTurnaroundOptions.hidden = kind !== "turnaround";
    $("spritePropDescLabel").textContent =
      kind === "state_variant" ? "State change description" :
      kind === "turnaround" ? "Extra details (optional)" : "Prop description";
    propDesc.required = kind !== "turnaround";
    propDesc.placeholder =
      kind === "state_variant" ? "Glowing cyan-emerald runes and warm flickering flame inside the lantern." :
      kind === "turnaround" ? "Keep cloak clasp and leather satchel straps consistent." :
      "Mossy carved stone lantern with weathered roof and iron brackets.";
    $("spritePropBaseLabel").textContent =
      kind === "turnaround" ? "Side reference plate (optional, helps back view)" : "Base prop image (required)";
    propName.placeholder =
      kind === "state_variant" ? "lantern_lit" :
      kind === "turnaround" ? "hero_back" : "stone_lantern";
  }

  propKind.onchange = syncPropDialogKind;
  $("spritePropCancel").onclick = () => propDialog.close();
  $("spritePropUploadBase").onclick = () => $("spritePropBaseFile").click();
  $("spritePropBaseFile").onchange = async (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file) return;
    $("spritePropBaseStatus").textContent = "Uploading…";
    try {
      const asset = await editor.uploadPicture(file);
      uploadedBaseAsset = { file: asset.file, subfolder: asset.subfolder || "" };
      populatePropRefs();
      propBaseRef.value = JSON.stringify(uploadedBaseAsset);
      $("spritePropBaseStatus").textContent = "Ready: " + asset.file;
    } catch (error) {
      $("spritePropBaseStatus").textContent = "";
      editor.toast(error.message || "Could not upload base image.", true);
    }
  };

  async function refreshPropGallery() {
    const gallery = $("spritePropGallery");
    if (!gallery || !editor.listProps) return;
    gallery.textContent = "Loading…";
    try {
      const { props = [] } = await editor.listProps();
      gallery.replaceChildren();
      if (!props.length) {
        gallery.textContent = "No props or turnaround plates rendered yet in this project.";
        return;
      }
      for (const item of props) {
        const card = document.createElement("div");
        card.className = "sprite-prop-card";
        const img = document.createElement("img");
        img.src = editor.viewUrl(item.file, item.subfolder);
        img.alt = item.file;
        const nameEl = document.createElement("b");
        nameEl.textContent = item.file;
        nameEl.title = item.file;
        const actions = document.createElement("div");
        actions.className = "sprite-prop-card-actions";
        if (actor && editor.addPropToCast) {
          const viewPick = document.createElement("select");
          viewPick.setAttribute("aria-label", "Cast view tag for " + item.file);
          for (const [v, label] of CAST_VIEWS) viewPick.append(new Option(label, v));
          const lower = item.file.toLowerCase();
          if (lower.includes("front") || lower.includes("_s_")) viewPick.value = "front";
          else if (lower.includes("back") || lower.includes("_n_")) viewPick.value = "back";
          else if (lower.includes("left") || lower.includes("_w_")) viewPick.value = "left";
          else if (lower.includes("side") || lower.includes("right") || lower.includes("_e_")) viewPick.value = "side";
          const addBtn = document.createElement("button");
          addBtn.type = "button";
          addBtn.className = "quiet";
          addBtn.textContent = "+ Cast ref";
          addBtn.onclick = () => {
            editor.addPropToCast(actor, { file: item.file, subfolder: item.subfolder }, viewPick.value || null);
            populatePropRefs();
            refresh();
          };
          actions.append(viewPick, addBtn);
        }
        const baseBtn = document.createElement("button");
        baseBtn.type = "button";
        baseBtn.className = "quiet";
        baseBtn.textContent = "Use as base";
        baseBtn.onclick = () => {
          uploadedBaseAsset = { file: item.file, subfolder: item.subfolder };
          propKind.value = "state_variant";
          syncPropDialogKind();
          populatePropRefs();
          propBaseRef.value = JSON.stringify(uploadedBaseAsset);
          $("spritePropBaseStatus").textContent = "Selected: " + item.file;
        };
        const dl = document.createElement("a");
        dl.className = "quiet button";
        dl.href = editor.viewUrl(item.file, item.subfolder);
        dl.download = item.file;
        dl.textContent = "PNG";
        actions.append(baseBtn, dl);
        card.append(img, nameEl, actions);
        gallery.append(card);
      }
    } catch {
      gallery.textContent = "Could not load generated props.";
    }
  }

  $("spritePropRefresh").onclick = () => void refreshPropGallery();
  $("spritePropBtn").onclick = () => {
    if (!project()) return;
    uploadedBaseAsset = null;
    $("spritePropBaseStatus").textContent = "";
    propStatus.textContent = "";
    propSize.value = CANVASES.includes(canvas) ? canvas : "square";
    if (!$("spritePropStyle").value.trim()) {
      $("spritePropStyle").value = resolveSpriteStyle(artStylePreset, artStyleCustom);
    }
    populatePropRefs();
    syncPropDialogKind();
    void refreshPropGallery();
    propDialog.showModal();
    propName.focus();
  };

  $("spritePropForm").onsubmit = async (e) => {
    e.preventDefault();
    propStatus.textContent = "";
    const parseRef = (val) => (val ? JSON.parse(val) : null);
    let payload;
    try {
      payload = buildPropPayload({
        kind: propKind.value,
        name: propName.value,
        description: propDesc.value,
        style: $("spritePropStyle").value,
        size: propSize.value,
        camera,
        style_ref: parseRef(propStyleRef.value),
        base_prop: parseRef(propBaseRef.value),
        top_canopy: $("spritePropCanopy").checked,
        output_size: $("spritePropOutputSize").value,
        view: $("spritePropView").value,
        proportions: $("spritePropProportions").value,
        holding: $("spritePropHolding").value,
      });
    } catch (error) {
      propStatus.textContent = error.message;
      return;
    }
    $("spritePropQueue").disabled = true;
    propStatus.textContent = "Queueing 4 MP H3-Image render…";
    try {
      const result = await editor.queueProp(payload);
      const dims = result?.job?.native_size ? result.job.native_size.join("×") : "4 MP";
      editor.toast("Queued " + payload.kind.replace("_", " ") + " · " + (result?.job?.name || payload.name) + " (" + dims + ").");
      propDialog.close();
    } catch (error) {
      propStatus.textContent = error.message || "Could not queue prop render.";
    } finally {
      $("spritePropQueue").disabled = false;
    }
  };

  $("spriteCount").onchange = (e) => {
    count = Number(e.target.value);
    if (count === 4 && dir.length === 2) dir = "S";
    audition = null;
    if (actor) remember((p) => (p.count = count));
    refresh();
  };
  $("spriteTile").onchange = () => {
    $("spriteTileCustomField").hidden = $("spriteTile").value !== "custom";
    if ($("spriteTile").value === "custom") $("spriteTileCustom").focus();
  };
  $("spriteStyle").onchange = refresh;
  $("spriteZoom").onchange = (e) => {
    zoom = Number(e.target.value) || 3;
    refresh();
  };
  $("spriteBackdrop").onchange = (e) => {
    backdrop = e.target.value || "matte";
    refresh();
  };
  $("spritePivot").onclick = () => {
    showPivot = !showPivot;
    refresh();
  };
  $("spriteHd").onclick = () => {
    hdMode = !hdMode;
    refresh();
  };
  $("spriteSpeed").onchange = (e) => {
    speed = Number(e.target.value) || 1;
    refresh();
  };
  $("spriteCenter").onclick = () => {
    x = 0.5;
    y = 0.68;
    paint();
  };
  $("spritePlay").onclick = () => {
    paused = !paused;
    refresh();
  };
  $("spritePrev").onclick = () => stepPose(-1);
  $("spriteStep").onclick = () => stepPose(1);
  $("spriteGround").onclick = () => $("spriteGroundFile").click();
  $("spritePlain").onclick = () => {
    editor.checkpoint();
    delete project().sprite_ground;
    editor.save();
    refresh();
  };
  $("spriteGroundFile").onchange = async (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file) return;
    const before = folder;
    try {
      const asset = await editor.uploadPicture(file);
      if (folder === before) {
        editor.checkpoint();
        project().sprite_ground = { file: asset.file, subfolder: asset.subfolder };
        editor.save();
        refresh();
      }
    } catch {
      editor.toast("Could not load that ground image.", true);
    }
  };
  $("spriteMirror").onclick = () => {
    if (audition?.mirror) {
      select(audition);
      return;
    }
    const partner = opposite(dir);
    const rows = slots[partner].takes;
    const source = chosen(partner);
    const row = source && !source.mirror ? source : rows[0];
    if (row) {
      audition = { ...row, mirror: true };
      refresh();
    }
  };
  $("spriteSelfPin").onclick = async () => {
    const row = current();
    if (!row || row.mirror || motion === "idle" || busy) return;
    busy = true;
    refresh();
    $("spriteStatus").textContent = "Picking a clean mid-stride frame to close the loop…";
    try {
      const pin = await editor.selfPin(row.clip, row.take);
      editor.checkpoint();
      row.clip.sprite = {
        ...row.clip.sprite,
        canvas,
        self_pin: { file: pin.file, subfolder: pin.subfolder, stride: pin.stride },
      };
      if (clipFrames !== 39) row.clip.sprite.frames = clipFrames;
      else delete row.clip.sprite.frames;
      editor.save();
      editor.toast("Closing loop from frame " + pin.frame + " (pins " + pin.pins.join(", ") + ").");
      refresh();
      await editor.render(row.clip, "draft", Math.floor(Math.random() * 2147483648));
    } catch (error) {
      editor.toast(error.message || "Could not create a loop-closing self-pin.", true);
    } finally {
      busy = false;
      refresh();
    }
  };
  $("spriteEdit").onclick = () => {
    const row = current(),
      clip = row?.clip || slots[dir].clips[0];
    flipMode(false);
    editor.editClip(clip, row?.take);
  };
  $("spriteDraft").onclick = async () => {
    if (busy || !actor) return;
    busy = true;
    try {
      const old = chosen();
      if (old && !prefs.picks?.[slotKey(camera, motion, dir)]) select(old);
      let clip = slots[dir].clips[0];
      const spriteStyle = resolveSpriteStyle(artStylePreset, artStyleCustom);
      if (!clip) {
        clip = editor.makeClip(actor, roster().get(actor).name, camera, motion, dir, animations.get(motion), {
          canvas,
          frames: clipFrames,
          style: spriteStyle,
          backdrop: keyColor,
          loop: loops(motion),
        });
      } else if (
        clip.sprite &&
        ((clip.sprite.canvas || "portrait") !== canvas ||
          (clip.sprite.frames || 39) !== clipFrames ||
          (clip.sprite.style || "") !== spriteStyle ||
          (clip.sprite.backdrop || "white") !== keyColor)
      ) {
        editor.checkpoint();
        clip.sprite.canvas = canvas;
        if (keyColor !== "white") clip.sprite.backdrop = keyColor;
        else delete clip.sprite.backdrop;
        if (clipFrames !== 39) clip.sprite.frames = clipFrames;
        else delete clip.sprite.frames;
        if (spriteStyle) clip.sprite.style = spriteStyle;
        else delete clip.sprite.style;
        editor.save();
      }
      refresh();
      await editor.render(clip, "draft", Math.floor(Math.random() * 2147483648));
    } finally {
      busy = false;
      refresh();
    }
  };
  $("spriteFinish").onclick = async () => {
    const row = current();
    if (!row || busy) return;
    busy = true;
    try {
      editor.activate(row.clip, row.take);
      refresh();
      await editor.render(row.clip, "finish");
    } finally {
      busy = false;
      refresh();
    }
  };
  $("spriteAtlas").onclick = async () => {
    if (exporting) return;
    const picks = exportPicks(),
      name = roster().get(actor)?.name || "sprites";
    let payload;
    try {
      payload = buildAtlasPayload({
        tile: $("spriteTile").value,
        customTile: $("spriteTileCustom").value,
        fps: $("spriteFps").value,
        style: style(),
        webp: $("spriteWebp").checked,
        picks,
      });
    } catch (error) {
      editor.toast(error.message, true);
      return;
    }
    exporting = true;
    refresh();
    $("spriteStatus").textContent = "Packing " + picks.length + " animation slots…";
    try {
      await editor.download(name, payload);
    } catch (error) {
      editor.toast(error.message, true);
    } finally {
      exporting = false;
      refresh();
    }
  };
  return { refresh, openClip: (clip) => flipMode(true, clip), isOpen: open };
}
