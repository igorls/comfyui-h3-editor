const ids = ["NW", "N", "NE", "W", "", "E", "SW", "S", "SE"];
const presets = new Map([["idle", "Idle"], ["run", "Run"], ["custom", "Custom motion"]]);
export const CANVASES = ["square", "portrait", "landscape"];
export const SPRITE_LENGTHS = [39, 56, 73];
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

export function buildSpriteConfig({ camera = "rpg", motion = "run", direction = "S", motionPrompt = null, canvas, frames = 39, poses = 7 } = {}) {
  const config = { camera, direction, poses };
  if (motion === "idle") config.motion = "idle";
  if (motionPrompt) Object.assign(config, { motion: "prompt", animation: motion });
  const chosenCanvas = canvas || defaultCanvasForCamera(camera);
  if (CANVASES.includes(chosenCanvas)) config.canvas = chosenCanvas;
  const len = Number(frames);
  if (Number.isInteger(len) && len >= 22 && (len - 5) % 17 === 0 && len !== 39) config.frames = len;
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
  if (sheet.cutout === "foreground model") return "BiRefNet";
  if (sheet.cutout === "background key") return "Background key";
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
    clipFrames = 39,
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
    y = 0.62,
    ground = null,
    busy = false,
    exporting = false;
  let audition = null;
  const open = () => !room.hidden;
  const project = () => editor.state.project;
  const chosen = (direction = dir) => pickedSprite(slots, prefs, camera, motion, direction);
  const current = () => audition || chosen();
  const style = () => $("spriteStyle").value;
  function sheetUrl(row) {
    return editor.viewUrl(row.take.sprite[style()], row.take.sprite.subfolder || row.take.subfolder);
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
    clipFrames = SPRITE_LENGTHS.includes(Number(prefs.frames)) ? Number(prefs.frames) : 39;
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
    keys.clear();
    pointer = null;
    x = 0.5;
    y = 0.62;
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
    $("spriteFrames").value = String(clipFrames);
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
  function paint() {
    if (!slots || !open()) return;
    const reviewed = current()?.take.sprite;
    const phase = reviewed ? ((clock * reviewed.fps) % reviewed.poses) / reviewed.poses : 0;
    for (const [node, row] of frames) {
      const sh = row.take.sprite;
      const pose = reviewed ? Math.floor(phase * sh.poses) : Math.floor(clock * sh.fps) % sh.poses;
      node.style.backgroundPositionX = -pose * 64 + "px";
    }
    const w = Math.max(1, stage.clientWidth),
      h = Math.max(1, stage.clientHeight);
    if (stage.width !== w || stage.height !== h) {
      stage.width = w;
      stage.height = h;
    }
    ctx.fillStyle = "#555b5b";
    ctx.fillRect(0, 0, w, h);
    if (ground?.naturalWidth) {
      const scale = Math.max(w / ground.width, h / ground.height);
      ctx.drawImage(
        ground,
        (w - ground.width * scale) / 2,
        (h - ground.height * scale) / 2,
        ground.width * scale,
        ground.height * scale,
      );
    }
    const row = current();
    if (!row) return;
    const url = sheetUrl(row);
    if (!pictures.has(url)) {
      const img = new Image();
      img.src = url;
      pictures.set(url, img);
    }
    const img = pictures.get(url);
    if (!img.naturalWidth) return;
    const sh = row.take.sprite,
      tw = sh.tile_size?.[0] || sh.tile,
      th = sh.tile_size?.[1] || sh.tile,
      pose = Math.floor(clock * sh.fps) % sh.poses;
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.translate(Math.round(x * w), Math.round(y * h));
    if (row.mirror) ctx.scale(-1, 1);
    ctx.drawImage(img, pose * tw, 0, tw, th, -sh.pivot[0], -sh.pivot[1], tw, th);
    ctx.restore();
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
      const speed = (90 * dt) / Math.hypot(dx, dy);
      x = Math.max(0.06, Math.min(0.94, x + (dx * speed) / stage.width));
      y = Math.max(0.33, Math.min(0.94, y + (dy * speed) / stage.height));
    }
    if (!paused) clock += dt;
    paint();
    raf = requestAnimationFrame(tick);
  }
  stage.onkeydown = (e) => {
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
  $("spriteFrames").onchange = (e) => {
    clipFrames = Number(e.target.value);
    if (actor) remember((p) => (p.frames = clipFrames));
    refresh();
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
  function editMotion(editing) {
    const owner = actor, projectFolder = folder;
    nameInput.readOnly = editing;
    nameInput.value = editing ? motion : "";
    promptInput.value = editing ? animations.get(motion) || "" : "";
    nameInput.setCustomValidity("");
    promptInput.setCustomValidity("");
    $("spriteMotionTitle").textContent = editing ? "Edit animation prompt" : "Add animation";
    $("spriteMotionSave").textContent = editing ? "Save prompt" : "Add animation";
    $("spriteMotionForm").onsubmit = (e) => {
      e.preventDefault();
      if (folder !== projectFolder || actor !== owner) return motionDialog.close();
      const name = nameInput.value.trim(), prompt = promptInput.value.trim();
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
        if (existing) existing.prompt = prompt;
        else definitions.push({ id: name, prompt });
        if (editing) {
          for (const clip of project().clips) {
            if (actorOf(clip, clip.cast) === actor && clip.sprite?.animation === name) clip.action = prompt;
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

  $("spritePropBtn").onclick = () => {
    if (!project()) return;
    uploadedBaseAsset = null;
    $("spritePropBaseStatus").textContent = "";
    propStatus.textContent = "";
    propSize.value = CANVASES.includes(canvas) ? canvas : "square";
    populatePropRefs();
    syncPropDialogKind();
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
  $("spritePlay").onclick = () => {
    paused = !paused;
    refresh();
  };
  $("spriteStep").onclick = () => {
    paused = true;
    const fps = current()?.take.sprite.fps || 12;
    clock = (Math.floor(clock * fps) + 1.001) / fps;
    refresh();
  };
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
      if (!clip) {
        clip = editor.makeClip(actor, roster().get(actor).name, camera, motion, dir, animations.get(motion), {
          canvas,
          frames: clipFrames,
        });
      } else if (
        clip.sprite &&
        ((clip.sprite.canvas || "portrait") !== canvas || (clip.sprite.frames || 39) !== clipFrames)
      ) {
        editor.checkpoint();
        clip.sprite.canvas = canvas;
        if (clipFrames !== 39) clip.sprite.frames = clipFrames;
        else delete clip.sprite.frames;
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
