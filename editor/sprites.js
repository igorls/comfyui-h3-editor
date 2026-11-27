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

export function fitZoomFor(stageW, stageH, baseW = 64, baseH = 64, { minZoom = 0.5, maxZoom = 12, padding = 28 } = {}) {
  const availW = Math.max(32, stageW - padding * 2);
  const availH = Math.max(32, stageH - padding * 2);
  const raw = Math.min(availW / Math.max(1, baseW), availH / Math.max(1, baseH));
  const rounded = Math.round(raw * 20) / 20;
  return Math.max(minZoom, Math.min(maxZoom, rounded));
}

export function clampPan(panX, panY, zoom, stageW, stageH, baseW = 64, baseH = 64) {
  const halfDrawW = (baseW * zoom) / 2;
  const halfDrawH = (baseH * zoom) / 2;
  const maxX = Math.max(stageW * 0.35, halfDrawW + stageW * 0.25);
  const maxY = Math.max(stageH * 0.35, halfDrawH + stageH * 0.25);
  return {
    panX: Math.max(-maxX, Math.min(maxX, panX)),
    panY: Math.max(-maxY, Math.min(maxY, panY)),
  };
}

export function zoomAtPoint(
  cam,
  nextZoom,
  cursorX,
  cursorY,
  stageW,
  stageH,
  { minZoom = 0.5, maxZoom = 12, baseW = 64, baseH = 64 } = {},
) {
  const oldZoom = Math.max(minZoom, Number(cam?.zoom) || 1);
  const clamped = Math.max(minZoom, Math.min(maxZoom, Math.round(nextZoom * 100) / 100));
  const ratio = clamped / oldZoom;
  const ox = cursorX - stageW / 2;
  const oy = cursorY - stageH / 2;
  const rawX = ox - (ox - (Number(cam?.panX) || 0)) * ratio;
  const rawY = oy - (oy - (Number(cam?.panY) || 0)) * ratio;
  const { panX, panY } = clampPan(rawX, rawY, clamped, stageW, stageH, baseW, baseH);
  return { zoom: clamped, panX, panY, fit: false };
}

export function shouldUseHd(hdPref, zoom) {
  if (hdPref === "on") return true;
  if (hdPref === "off") return false;
  return Number(zoom) >= 2;
}

export function screenToTilePoint(mx, my, boxX, boxY, drawW, drawH, maskW, maskH) {
  const safeW = Math.max(1e-6, Number(drawW) || 0);
  const safeH = Math.max(1e-6, Number(drawH) || 0);
  const mw = Math.max(1, Number(maskW) || 1);
  const mh = Math.max(1, Number(maskH) || 1);
  const u = (mx - boxX) / safeW;
  const v = (my - boxY) / safeH;
  const inside = u >= 0 && u <= 1 && v >= 0 && v <= 1;
  const x = Math.max(0, Math.min(mw, u * mw));
  const y = Math.max(0, Math.min(mh, v * mh));
  return { x, y, u, v, inside };
}

export function draftSettingsSummary({ canvas = "square", keyColor = "white", frames = 39, artStylePreset = "", artStyleCustom = "" } = {}) {
  const canvasLabel = canvas === "portrait" ? "256×384" : canvas === "landscape" ? "384×256" : "512²";
  const keyLabel = keyColor === "green" ? "Green key" : keyColor === "blue" ? "Blue key" : "White key";
  const styleLabel =
    artStylePreset === "pixel"
      ? "Pixel art"
      : artStylePreset === "chibi"
        ? "Chibi"
        : artStylePreset === "custom"
          ? artStyleCustom?.trim() ? "Custom style" : "Custom"
          : "Match refs";
  return `${canvasLabel} · ${frames}f · ${keyLabel} · ${styleLabel}`;
}

export function compassStatusFor(row, pending = false) {
  if (pending) return { kind: "rendering", label: "Rendering" };
  if (!row) return { kind: "empty", label: "Empty" };
  if (row.mirror) return { kind: "mirror", label: "Mirrored" };
  if (row.take?.quality === "final") return { kind: "final", label: "Final" };
  const cfg = row.take?.saved?.sprite || row.clip?.sprite;
  if (cfg?.self_pin) return { kind: "loop", label: "Loop closed" };
  return { kind: "draft", label: "Draft" };
}

export function referenceForDirection(cast, direction) {
  if (!cast) return null;
  const all = [cast, ...(cast.pictures || [])].filter((a) => a?.file);
  if (!all.length) return null;
  const want =
    direction === "N" || direction === "NW" || direction === "NE"
      ? ["back", "side", "front"]
      : direction === "W"
        ? ["left", "side", "front"]
        : direction === "E"
          ? ["side", "left", "front"]
          : ["front", "side"];
  for (const view of want) {
    const idx = all.findIndex((a) => a.view === view);
    if (idx !== -1) return { index: idx, asset: all[idx], label: all[idx].view || "Main" };
  }
  return { index: 0, asset: all[0], label: all[0].view || "Main" };
}

function formatZoomLabel(z) {
  const rounded = Math.round(z * 10) / 10;
  return (Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1)) + "×";
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
  const ghostCanvas = document.createElement("canvas");
  const ghostCtx = ghostCanvas.getContext("2d");
  const aiMaskCanvas = document.createElement("canvas");
  const aiMaskCtx = aiMaskCanvas.getContext("2d");
  const eraseMaskCanvas = document.createElement("canvas");
  const eraseMaskCtx = eraseMaskCanvas.getContext("2d");
  const restoreMaskCanvas = document.createElement("canvas");
  const restoreMaskCtx = restoreMaskCanvas.getContext("2d");
  const exportMaskCanvas = document.createElement("canvas");
  const exportMaskCtx = exportMaskCanvas.getContext("2d");
  aiMaskCanvas.width = aiMaskCanvas.height = 128;
  eraseMaskCanvas.width = eraseMaskCanvas.height = 128;
  restoreMaskCanvas.width = restoreMaskCanvas.height = 128;
  exportMaskCanvas.width = exportMaskCanvas.height = 128;
  $("spriteReferencePanel").open = !matchMedia("(max-width: 720px)").matches;
  const keys = new Set(),
    pictures = new Map();
  let walkPointer = null;
  let panDrag = null;
  let paintDrag = null;
  let hoverPoint = null;
  let spaceHeld = false;
  let lastDrawBox = null;
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
    stageMode = "inspect",
    x = 0.5,
    y = 0.68,
    cam = { zoom: 4, panX: 0, panY: 0, fit: true },
    speed = 1,
    backdrop = "matte",
    showPivot = false,
    showOnion = false,
    showRef = false,
    hdPref = "auto",
    ground = null,
    busy = false,
    exporting = false;
  let fixActive = false,
    fixMode = "ai",
    fixAiTool = "paint",
    fixAlphaTool = "erase",
    fixBrushSize = 10,
    fixBusy = false,
    fixBusyLabel = "",
    fixPoseInfo = null,
    fixPoseInfoKey = "",
    fixPoseInfoReq = 0,
    fixTargetKey = "",
    aiMaskDirty = false,
    eraseMaskDirty = false,
    restoreMaskDirty = false;
  const hdSheets = new Map();
  let poseStripKey = "";
  let poseButtons = [];
  let lastPaintedPose = -1;
  let audition = null;
  const MIN_ZOOM = 0.5, MAX_ZOOM = 12;
  const open = () => !room.hidden;
  const project = () => editor.state.project;
  const chosen = (direction = dir) => pickedSprite(slots, prefs, camera, motion, direction);
  const current = () => audition || chosen();
  const style = () => $("spriteStyle").value;
  function sheetUrl(row, overrideSheet = null) {
    const sh = overrideSheet || row.take.sprite;
    const base = editor.viewUrl(sh[style()], sh.subfolder || row.take.sprite.subfolder || row.take.subfolder);
    const rev = sh.rev ?? row.take.sprite?.rev;
    if (rev) return base + (base.includes("?") ? "&" : "?") + "rev=" + encodeURIComponent(String(rev));
    return base;
  }
  function posePatchInfo(sheet, poseIdx) {
    const map = sheet?.patched_poses;
    if (!map || typeof map !== "object") return null;
    return map[String(poseIdx)] || null;
  }
  function invalidateTakeImages(row) {
    if (!row) return;
    const key = hdKey(row);
    const oldHd = hdSheets.get(key);
    hdSheets.delete(key);
    const files = new Set(
      [row.take.sprite?.illustrated, row.take.sprite?.indexed, oldHd?.sheet?.illustrated, oldHd?.sheet?.indexed].filter(Boolean),
    );
    for (const url of [...pictures.keys()]) {
      for (const file of files) {
        if (url.includes(encodeURIComponent(file)) || url.includes(file)) {
          pictures.delete(url);
          break;
        }
      }
    }
    poseStripKey = "";
  }
  function ensureMaskSize(baseW, baseH) {
    const scale = Math.max(1, Math.round(128 / Math.max(1, baseW, baseH)));
    const mw = Math.max(64, Math.round(baseW * scale));
    const mh = Math.max(64, Math.round(baseH * scale));
    if (aiMaskCanvas.width === mw && aiMaskCanvas.height === mh) return;
    for (const c of [aiMaskCanvas, eraseMaskCanvas, restoreMaskCanvas, exportMaskCanvas]) {
      c.width = mw;
      c.height = mh;
    }
    aiMaskDirty = eraseMaskDirty = restoreMaskDirty = false;
  }
  function clearFixMasks(scope = "all") {
    if (scope === "all" || scope === "ai") {
      aiMaskCtx?.clearRect(0, 0, aiMaskCanvas.width, aiMaskCanvas.height);
      aiMaskDirty = false;
    }
    if (scope === "all" || scope === "alpha") {
      eraseMaskCtx?.clearRect(0, 0, eraseMaskCanvas.width, eraseMaskCanvas.height);
      restoreMaskCtx?.clearRect(0, 0, restoreMaskCanvas.width, restoreMaskCanvas.height);
      eraseMaskDirty = false;
      restoreMaskDirty = false;
    }
  }
  function hasMaskPixels(mCtx, mCanvas) {
    if (!mCtx) return false;
    const data = mCtx.getImageData(0, 0, mCanvas.width, mCanvas.height).data;
    for (let i = 3; i < data.length; i += 4) {
      if (data[i] > 0) return true;
    }
    return false;
  }
  function exportMaskPng(maskCanvas) {
    if (!exportMaskCtx) return null;
    exportMaskCanvas.width = maskCanvas.width;
    exportMaskCanvas.height = maskCanvas.height;
    exportMaskCtx.globalCompositeOperation = "source-over";
    exportMaskCtx.fillStyle = "#000000";
    exportMaskCtx.fillRect(0, 0, exportMaskCanvas.width, exportMaskCanvas.height);
    exportMaskCtx.drawImage(maskCanvas, 0, 0);
    return exportMaskCanvas.toDataURL("image/png");
  }
  function strokeOnMask(mCtx, x0, y0, x1, y1, diameter, erase = false) {
    if (!mCtx) return;
    mCtx.save();
    mCtx.globalCompositeOperation = erase ? "destination-out" : "source-over";
    mCtx.fillStyle = "#ffffff";
    mCtx.strokeStyle = "#ffffff";
    mCtx.lineWidth = diameter;
    mCtx.lineCap = "round";
    mCtx.lineJoin = "round";
    if (Math.hypot(x1 - x0, y1 - y0) < 0.25) {
      mCtx.beginPath();
      mCtx.arc(x0, y0, diameter / 2, 0, Math.PI * 2);
      mCtx.fill();
    } else {
      mCtx.beginPath();
      mCtx.moveTo(x0, y0);
      mCtx.lineTo(x1, y1);
      mCtx.stroke();
    }
    mCtx.restore();
  }
  function paintMaskStroke(x0, y0, x1, y1, baseW) {
    const maskScale = aiMaskCanvas.width / Math.max(1, baseW);
    const diameter = Math.max(1, fixBrushSize * maskScale);
    if (fixMode === "ai") {
      strokeOnMask(aiMaskCtx, x0, y0, x1, y1, diameter, fixAiTool === "unpaint");
      if (fixAiTool === "paint") aiMaskDirty = true;
    } else if (fixMode === "alpha") {
      if (fixAlphaTool === "erase") {
        strokeOnMask(eraseMaskCtx, x0, y0, x1, y1, diameter, false);
        strokeOnMask(restoreMaskCtx, x0, y0, x1, y1, diameter, true);
        eraseMaskDirty = true;
      } else {
        strokeOnMask(restoreMaskCtx, x0, y0, x1, y1, diameter, false);
        strokeOnMask(eraseMaskCtx, x0, y0, x1, y1, diameter, true);
        restoreMaskDirty = true;
      }
    }
  }
  function drawMaskOverlay(maskCanvas, color, alpha, dx, dy, drawW, drawH) {
    if (!ghostCtx) return;
    const mw = maskCanvas.width, mh = maskCanvas.height;
    if (ghostCanvas.width !== mw || ghostCanvas.height !== mh) {
      ghostCanvas.width = mw;
      ghostCanvas.height = mh;
    }
    ghostCtx.clearRect(0, 0, mw, mh);
    ghostCtx.globalCompositeOperation = "source-over";
    ghostCtx.drawImage(maskCanvas, 0, 0);
    ghostCtx.globalCompositeOperation = "source-in";
    ghostCtx.fillStyle = color;
    ghostCtx.fillRect(0, 0, mw, mh);
    ghostCtx.globalCompositeOperation = "source-over";
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(ghostCanvas, 0, 0, mw, mh, dx, dy, drawW, drawH);
    ctx.restore();
  }
  function ensurePoseInfo(row, pose) {
    if (!fixActive || !row || row.mirror || !editor.poseInfo) return;
    const key = [folder, row.clip.id, row.take.id, pose, row.take.sprite?.rev || 0].join(":");
    if (key === fixPoseInfoKey) return;
    fixPoseInfoKey = key;
    fixPoseInfo = null;
    const req = ++fixPoseInfoReq;
    editor.poseInfo(row.clip, row.take, pose).then(
      (info) => {
        if (req !== fixPoseInfoReq || !fixActive) return;
        fixPoseInfo = info;
        syncFixUi(current(), current() ? currentPoseIndex(current().take.sprite) : 0);
      },
      () => {
        if (req !== fixPoseInfoReq) return;
      },
    );
  }
  function setFixActive(next) {
    const show = current();
    const canFix = Boolean(show && !show.mirror);
    const want = Boolean(next) && canFix;
    if (fixActive === want) return;
    fixActive = want;
    paintDrag = null;
    hoverPoint = null;
    spaceHeld = false;
    stage.classList.remove("fix-paint", "pan-ready");
    clearFixMasks();
    if (fixActive) {
      paused = true;
      if (stageMode !== "inspect") {
        stageMode = "inspect";
        keys.clear();
        walkPointer = null;
        panDrag = null;
        stage.classList.remove("panning");
        resetViewToFit();
      }
      const { baseW, baseH } = baseTileDims(show);
      ensureMaskSize(baseW, baseH);
      fixTargetKey = [folder, show.clip.id, show.take.id, currentPoseIndex(show.take.sprite)].join(":");
      fixPoseInfoKey = "";
      ensurePoseInfo(show, currentPoseIndex(show.take.sprite));
    } else {
      fixPoseInfo = null;
      fixPoseInfoKey = "";
      fixTargetKey = "";
    }
    syncStageUi();
    paint();
  }
  function syncFixUi(show, pose) {
    const toggle = $("spriteFixToggle");
    const bar = $("spriteFixBar");
    if (!toggle || !bar) return;
    const canFix = Boolean(show && !show.mirror);
    if (fixActive && !canFix) {
      fixActive = false;
      paintDrag = null;
      hoverPoint = null;
      spaceHeld = false;
      clearFixMasks();
    }
    toggle.disabled = !canFix;
    toggle.setAttribute("aria-pressed", String(fixActive && canFix));
    toggle.title = !show
      ? "Render or select a take to fix a pose (F)"
      : show.mirror
        ? `Mirrored takes inherit fixes from ${opposite(dir)} — switch to ${opposite(dir)} to fix`
        : "Fix a bad pose with AI inpaint, alpha brush, or ±1 frame nudge (F)";

    bar.hidden = !fixActive || !canFix;
    stage.classList.toggle("fix-paint", Boolean(fixActive && canFix && fixMode !== "nudge"));
    if (!fixActive || !canFix) return;

    const targetKey = [folder, show.clip.id, show.take.id, pose].join(":");
    if (targetKey !== fixTargetKey) {
      fixTargetKey = targetKey;
      clearFixMasks();
      const { baseW, baseH } = baseTileDims(show);
      ensureMaskSize(baseW, baseH);
    }
    ensurePoseInfo(show, pose);

    $("spriteFixTabAi").setAttribute("aria-selected", String(fixMode === "ai"));
    $("spriteFixTabAlpha").setAttribute("aria-selected", String(fixMode === "alpha"));
    $("spriteFixTabNudge").setAttribute("aria-selected", String(fixMode === "nudge"));
    $("spriteFixAiTools").hidden = fixMode !== "ai";
    $("spriteFixAlphaTools").hidden = fixMode !== "alpha";
    $("spriteFixNudgeTools").hidden = fixMode !== "nudge";

    $("spriteFixAiPaint").setAttribute("aria-pressed", String(fixAiTool === "paint"));
    $("spriteFixAiUnpaint").setAttribute("aria-pressed", String(fixAiTool === "unpaint"));
    $("spriteFixAiPaint").disabled = fixBusy;
    $("spriteFixAiUnpaint").disabled = fixBusy;
    $("spriteFixBrushAi").value = String(fixBrushSize);
    $("spriteFixBrushAiVal").textContent = `${fixBrushSize} px`;
    $("spriteFixBrushAi").disabled = fixBusy;
    $("spriteFixAiClear").disabled = fixBusy || !aiMaskDirty;
    $("spriteFixPrompt").disabled = fixBusy;
    $("spriteFixStrength").disabled = fixBusy;
    $("spriteFixApplyAi").disabled = fixBusy;
    $("spriteFixApplyAi").textContent = fixBusy && fixMode === "ai" ? "Fixing…" : "Fix pose";

    const hasRaw = fixPoseInfo?.has_raw_video !== false;
    $("spriteFixAlphaErase").setAttribute("aria-pressed", String(fixAlphaTool === "erase"));
    $("spriteFixAlphaRestore").setAttribute("aria-pressed", String(fixAlphaTool === "restore"));
    $("spriteFixAlphaErase").disabled = fixBusy;
    $("spriteFixAlphaRestore").disabled = fixBusy || !hasRaw;
    $("spriteFixAlphaRestore").title = hasRaw
      ? "Restore clipped pixels from raw video"
      : "Raw source video unavailable for Restore";
    if (!hasRaw && fixAlphaTool === "restore") fixAlphaTool = "erase";
    $("spriteFixBrushAlpha").value = String(fixBrushSize);
    $("spriteFixBrushAlphaVal").textContent = `${fixBrushSize} px`;
    $("spriteFixBrushAlpha").disabled = fixBusy;
    $("spriteFixAlphaClear").disabled = fixBusy || (!eraseMaskDirty && !restoreMaskDirty);
    $("spriteFixApplyAlpha").disabled = fixBusy || (!eraseMaskDirty && !restoreMaskDirty);
    $("spriteFixApplyAlpha").textContent = fixBusy && fixMode === "alpha" ? "Applying…" : "Apply alpha";

    const totalFrames = fixPoseInfo?.total_frames ?? show.take.frames ?? null;
    const layoutIdx = show.take.sprite?.layout?.indices?.[pose];
    const sourceIdx = fixPoseInfo?.source_index ?? (Number.isInteger(layoutIdx) ? layoutIdx : null);
    const readoutEl = $("spriteFixFrameReadout");
    if (sourceIdx != null && totalFrames != null) {
      readoutEl.textContent = `Frame ${sourceIdx} / ${totalFrames}`;
      readoutEl.title = fixPoseInfo
        ? `Source video frame ${sourceIdx} of ${totalFrames} (original: ${fixPoseInfo.orig_index})`
        : `Source video frame ${sourceIdx} of ${totalFrames}`;
    } else if (totalFrames != null) {
      readoutEl.textContent = `Frame — / ${totalFrames}`;
      readoutEl.title = "";
    } else {
      readoutEl.textContent = "Frame …";
      readoutEl.title = "";
    }
    $("spriteFixNudgeBack").disabled = fixBusy || !hasRaw || (sourceIdx != null && sourceIdx <= 0);
    $("spriteFixNudgeFwd").disabled = fixBusy || !hasRaw || (sourceIdx != null && totalFrames != null && sourceIdx >= totalFrames - 1);

    const isPatched = Boolean(posePatchInfo(show.take.sprite, pose) || fixPoseInfo?.patched);
    const revertBtn = $("spriteFixRevert");
    revertBtn.hidden = !isPatched;
    revertBtn.disabled = fixBusy || !isPatched;
    $("spriteFixDone").disabled = fixBusy;
  }
  function hdKey(row) {
    return [folder, row.clip.id, row.take.id].join(":");
  }
  function baseTileDims(row = current()) {
    const sh = row?.take?.sprite;
    return {
      baseW: sh?.tile_size?.[0] || sh?.tile || 64,
      baseH: sh?.tile_size?.[1] || sh?.tile || 64,
    };
  }
  function stageSize() {
    return {
      w: Math.max(1, stage.clientWidth || 600),
      h: Math.max(1, stage.clientHeight || 320),
    };
  }
  function effectiveZoom(row = current()) {
    if (cam.fit) {
      const { w, h } = stageSize();
      const { baseW, baseH } = baseTileDims(row);
      return fitZoomFor(w, h, baseW, baseH, { minZoom: MIN_ZOOM, maxZoom: MAX_ZOOM });
    }
    return cam.zoom;
  }
  function resetViewToFit() {
    const { w, h } = stageSize();
    const { baseW, baseH } = baseTileDims();
    cam = {
      zoom: fitZoomFor(w, h, baseW, baseH, { minZoom: MIN_ZOOM, maxZoom: MAX_ZOOM }),
      panX: 0,
      panY: 0,
      fit: true,
    };
  }
  function activeDisplaySheet(row) {
    if (!row) return null;
    if (shouldUseHd(hdPref, effectiveZoom(row))) {
      const cached = hdSheets.get(hdKey(row));
      if (cached && !cached.loading && !cached.error && cached.sheet) return cached.sheet;
    }
    return row.take.sprite;
  }
  async function ensureHd(row, { force = false, explicit = false } = {}) {
    if (!row || !editor.inspectSprite) return;
    if (!force && !shouldUseHd(hdPref, effectiveZoom(row))) return;
    const key = hdKey(row);
    const baseTile = row.take.sprite.tile_size?.[0] || row.take.sprite.tile || 64;
    if (baseTile >= 256) return;
    const existing = hdSheets.get(key);
    if (existing) {
      if (existing.loading) return;
      if (!existing.error) return;
      if (!force) return;
    }
    hdSheets.set(key, { loading: true });
    syncStageUi();
    try {
      const sheet = await editor.inspectSprite(row.clip, row.take, 256);
      hdSheets.set(key, { loading: false, sheet });
    } catch (error) {
      const message = error.message || "HD preview unavailable";
      hdSheets.set(key, { loading: false, error: message, silent: !explicit && hdPref === "auto" });
      if (hdPref === "on") hdPref = "off";
      if (explicit) editor.toast("HD 256 preview unavailable: " + message, true);
    }
    syncStageUi();
    paint();
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
    walkPointer = null;
    panDrag = null;
    paintDrag = null;
    hoverPoint = null;
    spaceHeld = false;
    stage.classList.remove("panning", "fix-paint", "pan-ready");
    if (!enabled) {
      fixActive = false;
      clearFixMasks();
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
    lastPaintedPose = -1;
    keys.clear();
    walkPointer = null;
    panDrag = null;
    paintDrag = null;
    hoverPoint = null;
    spaceHeld = false;
    fixActive = false;
    fixBusy = false;
    fixBusyLabel = "";
    fixPoseInfo = null;
    fixPoseInfoKey = "";
    fixTargetKey = "";
    clearFixMasks();
    stageMode = "inspect";
    x = 0.5;
    y = 0.68;
    cam = { zoom: 4, panX: 0, panY: 0, fit: true };
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

  function updateStageRefCard() {
    const btn = $("spriteStageRef");
    if (!btn) return;
    const cast = project()?.cast?.find((p) => p.id === actor);
    const ref = showRef ? referenceForDirection(cast, dir) : null;
    if (!ref) {
      btn.hidden = true;
      btn.onclick = null;
      return;
    }
    const img = $("spriteStageRefImg");
    const tag = $("spriteStageRefTag");
    const url = editor.viewUrl(ref.asset.file, ref.asset.subfolder);
    if (img.getAttribute("src") !== url) img.src = url;
    img.alt = `${cast.name} (${ref.label}) reference`;
    tag.textContent = ref.label;
    btn.hidden = false;
    btn.onclick = () => editor.inspectCast?.(cast.id, ref.index);
  }

  function updateArenaBadge(show, hdEntry, pose) {
    const badge = $("spriteArenaBadge");
    if (!badge) return;
    const sh = show?.take?.sprite;
    let text = "";
    let tone = "";
    if (fixBusy && fixBusyLabel) {
      text = fixBusyLabel;
      tone = "info";
    } else if (hdEntry?.loading) {
      text = "Loading HD 256…";
      tone = "info";
    } else if (hdEntry?.error && !hdEntry.silent && hdPref !== "off") {
      text = "HD 256 unavailable · click HD 256 to retry";
      tone = "warn";
    } else if (audition?.mirror) {
      text = `Mirror preview (${opposite(dir)}) · click Use mirrored to keep`;
      tone = "accent";
    } else if (showOnion && sh && sh.poses > 1 && (pose === 0 || pose === sh.poses - 1)) {
      text = `Loop seam · Pose ${sh.poses} ↔ 1`;
      tone = "accent";
    }
    badge.hidden = !text;
    badge.textContent = text;
    badge.dataset.tone = tone;
  }

  function syncStageUi() {
    if (!open() || !project() || !slots) return;
    const show = current();
    const zoomNow = effectiveZoom(show);
    const hdWanted = shouldUseHd(hdPref, zoomNow);
    const hdEntry = show ? hdSheets.get(hdKey(show)) : null;
    const hdActive = Boolean(show && hdWanted && activeDisplaySheet(show) !== show.take.sprite);
    const hdErrored = Boolean(hdEntry?.error);

    const inspectBtn = $("spriteModeInspect");
    const walkBtn = $("spriteModeWalk");
    if (inspectBtn) inspectBtn.setAttribute("aria-pressed", String(stageMode === "inspect"));
    if (walkBtn) walkBtn.setAttribute("aria-pressed", String(stageMode === "walk"));
    stage.classList.toggle("walk", stageMode === "walk");

    const hasGround = Boolean(project()?.sprite_ground?.file);
    $("spriteGround").hidden = stageMode !== "walk";
    $("spritePlain").hidden = stageMode !== "walk" || !hasGround;
    $("spritePlain").disabled = !hasGround;

    const zoomReset = $("spriteZoomReset");
    if (zoomReset) {
      zoomReset.textContent = cam.fit ? "Fit" : formatZoomLabel(zoomNow);
      zoomReset.setAttribute("aria-label", cam.fit ? `Fit (${formatZoomLabel(zoomNow)})` : `Zoom ${formatZoomLabel(zoomNow)}`);
    }
    $("spriteZoomOut").disabled = zoomNow <= MIN_ZOOM + 0.01;
    $("spriteZoomIn").disabled = zoomNow >= MAX_ZOOM - 0.01;

    for (const sw of $("spriteSurfaceSwatches")?.querySelectorAll("button[data-backdrop]") || []) {
      sw.setAttribute("aria-pressed", String(sw.dataset.backdrop === backdrop));
    }

    const hdBtn = $("spriteHd");
    const hdVisibleErr = Boolean(hdErrored && !hdEntry?.silent);
    hdBtn.disabled = !show;
    hdBtn.classList.toggle("is-error", hdVisibleErr && !hdEntry?.loading);
    hdBtn.setAttribute("aria-pressed", String(Boolean(show && (hdActive || hdEntry?.loading))));
    if (hdEntry?.loading) hdBtn.textContent = "HD…";
    else if (hdVisibleErr) hdBtn.textContent = "Retry HD";
    else hdBtn.textContent = "HD 256";
    hdBtn.title = hdErrored
      ? `HD 256 unavailable (${hdEntry.error}) · Click to retry (H)`
      : hdPref === "auto"
        ? "Auto HD 256 when zoomed ≥ 2× · Click to toggle Base 64 / HD 256 (H)"
        : hdPref === "on"
          ? "HD 256 pinned on · Click to switch to Base 64 (H)"
          : "Base 64 pixel strip · Click to enable HD 256 (H)";

    $("spriteOnion").setAttribute("aria-pressed", String(showOnion));
    $("spriteOnion").disabled = !show || (show.take.sprite.poses || 1) < 2;
    $("spritePivot").setAttribute("aria-pressed", String(showPivot));
    $("spritePivot").disabled = !show;
    const cast = project()?.cast?.find((p) => p.id === actor);
    const hasRefPic = Boolean(referenceForDirection(cast, dir));
    $("spriteRefToggle").disabled = !hasRefPic;
    $("spriteRefToggle").setAttribute("aria-pressed", String(showRef && hasRefPic));
    updateStageRefCard();

    $("spritePlay").textContent = paused ? "Play" : "Pause";
    $("spritePlay").disabled = !show;
    $("spritePrev").disabled = !show;
    $("spriteStep").disabled = !show;
    $("spriteSpeed").disabled = !show;
    $("spriteSpeed").value = String(speed);
    $("spriteCenter").textContent = stageMode === "inspect" ? "Fit" : "Center";
    $("spriteCenter").title = stageMode === "inspect" ? "Reset view to Fit (0)" : "Center character on floor (0)";
    $("spriteCenter").disabled = !show;

    if (show && hdWanted) void ensureHd(show);
    syncPoseStrip(show);
    const pose = show ? currentPoseIndex(show.take.sprite) : 0;
    syncFixUi(show, pose);
    updateArenaBadge(show, hdEntry, pose);
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
        b.ondblclick = () => {
          if (project().cast?.some((c) => c.id === p.id)) editor.editCast(p.id);
        };
        return b;
      }),
    );
    $("spriteTitle").textContent = person?.name || "Add a character reference to start";
    const refs = $("spriteRefs");
    $("spriteReferencePanel").hidden = !cast;
    refs.replaceChildren();
    if (cast) {
      const castAssets = [cast, ...(cast.pictures || [])].filter((a) => a.file);
      castAssets.forEach((asset, idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "sprite-ref-item sprite-ref-btn";
        btn.title = `Inspect ${cast.name}${asset.view ? " (" + asset.view + ")" : ""} reference`;
        btn.onclick = () => editor.inspectCast?.(cast.id, idx);
        const img = document.createElement("img");
        img.alt = cast.name + (asset.view ? " (" + asset.view + ")" : "") + " reference";
        img.src = editor.viewUrl(asset.file, asset.subfolder);
        btn.append(img);
        if (asset.view) {
          const tag = document.createElement("span");
          tag.className = "sprite-ref-tag";
          tag.textContent = asset.view;
          btn.append(tag);
        }
        refs.append(btn);
      });
      if (cast.refmod && editor.refmodVisual) {
        const wrap = document.createElement("div");
        wrap.className = "sprite-ref-item sprite-ref-mod";
        wrap.title = "RefMod: " + cast.refmod.file;
        wrap.append(editor.refmodVisual(cast.refmod.file, { alt: cast.name + " RefMod" }));
        refs.append(wrap);
      }
      const edit = button("Edit references", () => editor.editCast(cast.id));
      edit.className = "quiet sprite-refs-edit";
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
    $("spriteKeyColor").value = keyColor;
    $("spriteFrames").value = String(clipFrames);
    $("spriteArtStyle").value = artStylePreset;
    $("spriteArtStyleCustom").value = artStyleCustom;
    $("spriteArtStyleCustomField").hidden = artStylePreset !== "custom";
    $("spriteDraftSummary").textContent = draftSettingsSummary({
      canvas,
      keyColor,
      frames: clipFrames,
      artStylePreset,
      artStyleCustom,
    });
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
      label.className = "compass-dir";
      label.textContent = h;
      const row = chosen(h);
      const slotPending = Boolean(slots[h].clips.find((c) => editor.jobFor(c.id)));
      const st = compassStatusFor(row, slotPending);
      b.dataset.spriteFocus = "direction:" + h;
      b.classList.toggle("empty", !row && !slotPending);
      if (row) {
        filled++;
        b.append(frame(row));
        b.title = `${h} · ${st.label} · ${cutoutLabel(row.take.sprite)}`;
      } else {
        b.title = slotPending ? `${h} · Rendering` : `${h} · No take yet`;
      }
      b.append(label);
      if (st.kind !== "empty") {
        const dot = document.createElement("i");
        dot.className = "c-dot " + st.kind;
        dot.setAttribute("aria-hidden", "true");
        if (st.kind === "mirror") dot.textContent = "⇄";
        b.append(dot);
      }
      b.setAttribute("aria-label", `${h} ${st.label.toLowerCase()}`);
      compass.append(b);
    }
    const centerCell = document.createElement("div");
    centerCell.className = "sprite-compass-center";
    centerCell.setAttribute("aria-hidden", "true");
    centerCell.innerHTML = `<b>${filled}/${count}</b><span>dirs</span>`;
    compass.append(centerCell);

    $("spriteCoverage").textContent = `${filled} of ${count} directions ready`;

    const show = current(),
      picked = chosen(),
      pending = slots[dir].clips.find((c) => editor.jobFor(c.id));
    const partner = opposite(dir);
    const hasPartnerTakes = Boolean(partner !== dir && slots[partner]?.takes.length);

    const takes = $("spriteTakes");
    takes.replaceChildren();
    const rows = [...slots[dir].takes];
    if (show?.mirror) rows.push(show);
    for (const row of rows) {
      const selected = picked?.take.id === row.take.id && !!picked?.mirror === !!row.mirror;
      const b = button("", () => select(row), selected);
      b.className = "sprite-take-card";
      b.dataset.spriteFocus = (row.mirror ? "mirror:" : "take:") + row.clip.id + ":" + row.take.id;

      const head = document.createElement("div");
      head.className = "sprite-take-label";
      const pill = document.createElement("span");
      const savedCfg = row.take.saved?.sprite || row.clip.sprite;
      if (row.mirror) {
        pill.className = "sprite-pill mirror";
        pill.textContent = "Mirrored " + opposite(dir);
      } else if (row.take.quality === "final") {
        pill.className = "sprite-pill final";
        pill.textContent = "Final";
      } else {
        pill.className = "sprite-pill draft";
        pill.textContent = "Draft";
      }
      head.append(pill);
      if (savedCfg?.self_pin) {
        const loopPill = document.createElement("span");
        loopPill.className = "sprite-pill loop";
        loopPill.textContent = "Loop";
        head.append(loopPill);
      }
      const seedEl = document.createElement("span");
      seedEl.className = "sprite-take-seed";
      seedEl.textContent = "#" + row.take.seed;
      head.append(seedEl);

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
      const meta = [cutoutLabel(row.take.sprite)];
      if (savedCfg?.canvas) meta.push(savedCfg.canvas);
      if (row.take.frames && row.take.frames !== 39) meta.push(row.take.frames + "f");
      method.textContent = meta.join(" · ");
      b.append(frame(row), head, method, film);
      takes.append(b);
    }

    const draftDisabled = !actor || !!pending || busy || (motion === "custom" && !slots[dir].clips.length);
    if (!rows.length) {
      const empty = document.createElement("div");
      empty.className = "sprite-empty-well";
      const msg = document.createElement("span");
      msg.textContent = pending ? `Rendering ${dir}…` : `No takes for ${dir} yet.`;
      empty.append(msg);
      if (!pending && !draftDisabled) {
        const quickDraft = document.createElement("button");
        quickDraft.type = "button";
        quickDraft.className = "quiet";
        quickDraft.textContent = `New ${dir} draft`;
        quickDraft.onclick = () => $("spriteDraft").click();
        empty.append(quickDraft);
      }
      if (!pending && hasPartnerTakes) {
        const quickMirror = document.createElement("button");
        quickMirror.type = "button";
        quickMirror.className = "quiet";
        quickMirror.textContent = `Preview mirrored ${partner}`;
        quickMirror.onclick = () => $("spriteMirror").click();
        empty.append(quickMirror);
      }
      takes.append(empty);
    }

    const canFinish = Boolean(show && show.take.quality === "draft" && !editor.jobFor(show.clip.id) && !busy);
    $("spriteTakesTitle").textContent = `${dir} takes`;
    $("spriteDraft").textContent = pending ? `Rendering ${dir}…` : `New ${dir} draft`;
    $("spriteDraft").disabled = draftDisabled;
    $("spriteDraft").className = !rows.length && !pending ? "primary" : "quiet";

    $("spriteSelfPin").disabled =
      !show || !!show.mirror || motion === "idle" || !!editor.jobFor(show.clip.id) || busy;
    $("spriteEdit").disabled = !show && !slots[dir].clips.length;
    $("spriteFinish").disabled = !canFinish;
    $("spriteFinish").className = canFinish ? "primary" : "quiet";

    $("spriteMirror").hidden = partner === dir;
    $("spriteMirror").textContent = audition?.mirror
      ? "Use mirrored " + partner
      : "Preview mirrored " + partner;
    $("spriteMirror").disabled = !hasPartnerTakes;

    const ready = exportPicks().length;
    $("spriteAtlas").disabled = !actor || exporting || !ready;
    $("spriteAtlas").textContent = ready
      ? "Export atlas · " + ready + (ready === 1 ? " loop" : " loops")
      : "Export atlas";
    $("spriteStatus").textContent = pending
      ? "Queued or rendering"
      : slots[dir].clips.find((c) => c.error)?.error ||
        (audition?.mirror ? "Previewing mirror" : audition ? "Previewing editor take" : "");
    if (!exporting && $("spriteExportStatus")) $("spriteExportStatus").textContent = "";

    syncStageUi();
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
    syncStageUi();
    paint();
  }
  function jumpToPose(index) {
    const row = current();
    if (!row) return;
    const sh = row.take.sprite;
    if (index < 0 || index >= sh.poses) return;
    paused = true;
    clock = (index + 0.01) / (sh.fps || 12);
    syncStageUi();
    paint();
  }
  function syncPoseStrip(row) {
    const strip = $("spritePoseStrip");
    if (!strip) return;
    if (!row) {
      poseStripKey = "";
      poseButtons = [];
      lastPaintedPose = -1;
      strip.replaceChildren();
      return;
    }
    const baseSh = row.take.sprite;
    const sh = activeDisplaySheet(row);
    const isHd = sh !== baseSh;
    const url = sheetUrl(row, sh);
    const patchSig = JSON.stringify(baseSh.patched_poses || null);
    const key = [row.clip.id, row.take.id, row.mirror ? "m" : "", url, backdrop, sh.poses, isHd ? "hd" : "sd", patchSig].join("|");
    if (key !== poseStripKey) {
      poseStripKey = key;
      poseButtons = [];
      lastPaintedPose = -1;
      strip.replaceChildren();
      const backdropClass = backdrop !== "matte" ? " backdrop-" + backdrop : "";
      for (let idx = 0; idx < sh.poses; idx++) {
        const patched = Boolean(posePatchInfo(baseSh, idx));
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "sprite-pose-tile" + backdropClass + (isHd ? " is-hd" : "") + (patched ? " is-patched" : "");
        btn.title = `Inspect pose ${idx + 1} of ${sh.poses} (key ${idx + 1})${patched ? " · Patched" : ""}`;
        btn.setAttribute("aria-label", `Pose ${idx + 1} of ${sh.poses}${patched ? ", patched" : ""}`);
        btn.onclick = () => jumpToPose(idx);
        const cell = document.createElement("i");
        cell.style.backgroundImage = "url(" + url + ")";
        cell.style.backgroundSize = sh.poses * 64 + "px 64px";
        cell.style.backgroundPositionX = -idx * 64 + "px";
        if (row.mirror) cell.style.transform = "scaleX(-1)";
        const num = document.createElement("span");
        num.textContent = String(idx + 1);
        btn.append(cell, num);
        if (patched) {
          const dot = document.createElement("i");
          dot.className = "sprite-pose-patched";
          dot.setAttribute("aria-hidden", "true");
          btn.append(dot);
        }
        poseButtons.push(btn);
        strip.append(btn);
      }
    }
    const active = currentPoseIndex(baseSh);
    if (active !== lastPaintedPose) {
      lastPaintedPose = active;
      for (let i = 0; i < poseButtons.length; i++) {
        poseButtons[i].setAttribute("aria-pressed", String(i === active));
      }
    }
  }
  function drawBackdrop(w, h) {
    if (stageMode === "walk" && ground?.naturalWidth) {
      ctx.fillStyle = "#484f57";
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
      ctx.fillStyle = "#484f57";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "#2d3239";
      for (let yy = 0; yy < h; yy += sq) {
        for (let xx = ((yy / sq) % 2) * sq; xx < w; xx += sq * 2) {
          ctx.fillRect(xx, yy, sq, sq);
        }
      }
    } else {
      ctx.fillStyle = "#484f57";
      ctx.fillRect(0, 0, w, h);
    }
  }

  function drawTintedGhost(img, srcX, srcW, srcH, dx, dy, drawW, drawH, mirror, tint, alpha, smooth) {
    if (!ghostCtx) return;
    if (ghostCanvas.width !== srcW || ghostCanvas.height !== srcH) {
      ghostCanvas.width = srcW;
      ghostCanvas.height = srcH;
    }
    ghostCtx.clearRect(0, 0, srcW, srcH);
    ghostCtx.globalCompositeOperation = "source-over";
    ghostCtx.drawImage(img, srcX, 0, srcW, srcH, 0, 0, srcW, srcH);
    ghostCtx.globalCompositeOperation = "source-atop";
    ghostCtx.fillStyle = tint;
    ghostCtx.fillRect(0, 0, srcW, srcH);
    ghostCtx.globalCompositeOperation = "source-over";

    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.imageSmoothingEnabled = smooth;
    if (mirror) {
      ctx.translate(dx + drawW, dy);
      ctx.scale(-1, 1);
      ctx.drawImage(ghostCanvas, 0, 0, srcW, srcH, 0, 0, drawW, drawH);
    } else {
      ctx.drawImage(ghostCanvas, 0, 0, srcW, srcH, dx, dy, drawW, drawH);
    }
    ctx.restore();
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
    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const targetW = Math.round(w * dpr),
      targetH = Math.round(h * dpr);
    if (stage.width !== targetW || stage.height !== targetH) {
      stage.width = targetW;
      stage.height = targetH;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawBackdrop(w, h);
    const row = current();
    const readout = $("spritePoseReadout");
    if (!row) {
      if (readout) {
        readout.textContent = stageMode === "inspect"
          ? "No take yet · Render a draft or mirror an opposite direction"
          : "Walk mode · WASD or arrow keys";
      }
      return;
    }
    const baseSh = row.take.sprite;
    const sh = activeDisplaySheet(row);
    const url = sheetUrl(row, sh);
    if (!pictures.has(url)) {
      const img = new Image();
      img.onload = () => paint();
      img.src = url;
      pictures.set(url, img);
    }
    const img = pictures.get(url);
    const pose = currentPoseIndex(baseSh);
    if (pose !== lastPaintedPose) {
      lastPaintedPose = pose;
      for (let i = 0; i < poseButtons.length; i++) {
        poseButtons[i].setAttribute("aria-pressed", String(i === pose));
      }
      if (fixActive) syncFixUi(row, pose);
      if (showOnion) updateArenaBadge(row, hdSheets.get(hdKey(row)), pose);
    }
    const zoomNow = effectiveZoom(row);
    if (readout) {
      const hdTag = sh !== baseSh ? " · HD " + (sh.tile_size?.[0] || sh.tile) : " · Base " + (baseSh.tile_size?.[0] || baseSh.tile);
      const speedTag = speed !== 1 ? ` (${speed}×)` : "";
      const zoomTag = cam.fit ? `Fit (${formatZoomLabel(zoomNow)})` : formatZoomLabel(zoomNow);
      readout.textContent = `Pose ${pose + 1}/${baseSh.poses} · ${baseSh.fps} fps${speedTag} · ${zoomTag}${hdTag}`;
    }
    if (!img.naturalWidth) return;
    const srcW = sh.tile_size?.[0] || sh.tile,
      srcH = sh.tile_size?.[1] || sh.tile,
      baseW = baseSh.tile_size?.[0] || baseSh.tile,
      baseH = baseSh.tile_size?.[1] || baseSh.tile,
      drawW = Math.round(baseW * zoomNow),
      drawH = Math.round(baseH * zoomNow),
      pivX = Math.round(baseSh.pivot[0] * zoomNow),
      pivY = Math.round(baseSh.pivot[1] * zoomNow);

    let boxX, boxY, pivotScreenX, pivotScreenY;
    if (stageMode === "inspect") {
      const centerX = w / 2 + (cam.fit ? 0 : cam.panX);
      const centerY = h / 2 + (cam.fit ? 0 : cam.panY);
      boxX = Math.round(centerX - drawW / 2);
      boxY = Math.round(centerY - drawH / 2);
      pivotScreenX = boxX + (row.mirror ? drawW - pivX : pivX);
      pivotScreenY = boxY + pivY;
    } else {
      pivotScreenX = Math.round(x * w);
      pivotScreenY = Math.round(y * h);
      boxX = pivotScreenX - (row.mirror ? drawW - pivX : pivX);
      boxY = pivotScreenY - pivY;
    }

    const smooth = sh !== baseSh && srcW > drawW;

    if (showOnion && baseSh.poses > 1) {
      const prevPose = (pose - 1 + baseSh.poses) % baseSh.poses;
      const nextPose = (pose + 1) % baseSh.poses;
      drawTintedGhost(img, prevPose * srcW, srcW, srcH, boxX, boxY, drawW, drawH, row.mirror, "rgba(88, 178, 255, 0.52)", 0.28, smooth);
      if (nextPose !== prevPose) {
        drawTintedGhost(img, nextPose * srcW, srcW, srcH, boxX, boxY, drawW, drawH, row.mirror, "rgba(240, 179, 84, 0.52)", 0.22, smooth);
      }
    }

    ctx.save();
    ctx.imageSmoothingEnabled = smooth;
    if (row.mirror) {
      ctx.translate(boxX + drawW, boxY);
      ctx.scale(-1, 1);
      ctx.drawImage(img, pose * srcW, 0, srcW, srcH, 0, 0, drawW, drawH);
    } else {
      ctx.drawImage(img, pose * srcW, 0, srcW, srcH, boxX, boxY, drawW, drawH);
    }
    ctx.restore();

    lastDrawBox = { boxX, boxY, drawW, drawH, baseW, baseH };

    if (fixActive && !row.mirror && fixMode !== "nudge") {
      if (!showPivot) {
        ctx.save();
        ctx.strokeStyle = "rgba(236, 232, 223, 0.28)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        ctx.strokeRect(boxX + 0.5, boxY + 0.5, drawW - 1, drawH - 1);
        ctx.restore();
      }
      if (fixMode === "ai" && aiMaskDirty) {
        drawMaskOverlay(aiMaskCanvas, "#e58a7a", 0.42, boxX, boxY, drawW, drawH);
      } else if (fixMode === "alpha") {
        if (eraseMaskDirty) {
          drawMaskOverlay(eraseMaskCanvas, "#e58a7a", 0.44, boxX, boxY, drawW, drawH);
        }
        if (restoreMaskDirty) {
          drawMaskOverlay(restoreMaskCanvas, "#57bdb0", 0.44, boxX, boxY, drawW, drawH);
        }
      }
      if (hoverPoint && !panDrag && !spaceHeld) {
        const radius = Math.max(2, (fixBrushSize * zoomNow) / 2);
        const strokeColor =
          fixMode === "alpha" && fixAlphaTool === "restore"
            ? "#57bdb0"
            : fixMode === "ai" && fixAiTool === "unpaint"
              ? "#ece8df"
              : "#e58a7a";
        ctx.save();
        ctx.beginPath();
        ctx.arc(hoverPoint.mx, hoverPoint.my, radius, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(20, 22, 25, 0.72)";
        ctx.lineWidth = 3;
        ctx.stroke();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.restore();
      }
    }

    if (showPivot) {
      ctx.save();
      ctx.strokeStyle = "rgba(224, 162, 74, 0.65)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(boxX + 0.5, boxY + 0.5, drawW - 1, drawH - 1);
      ctx.setLineDash([]);
      ctx.strokeStyle = "#e0a24a";
      ctx.beginPath();
      ctx.moveTo(pivotScreenX - 10, pivotScreenY + 0.5);
      ctx.lineTo(pivotScreenX + 10, pivotScreenY + 0.5);
      ctx.moveTo(pivotScreenX + 0.5, pivotScreenY - 10);
      ctx.lineTo(pivotScreenX + 0.5, pivotScreenY + 10);
      ctx.stroke();
      ctx.restore();
    }
  }
  function tick(now) {
    if (!open()) return;
    const dt = last ? Math.min(0.05, (now - last) / 1000) : 0;
    last = now;
    if (stageMode === "walk") {
      const { w, h } = stageSize();
      let dx = Number(keys.has("E")) - Number(keys.has("W")),
        dy = Number(keys.has("S")) - Number(keys.has("N"));
      if (!dx && !dy && walkPointer) {
        const gapX = (walkPointer.x - x) * w,
          gapY = (walkPointer.y - y) * h;
        dx = Math.abs(gapX) > 4 ? Math.sign(gapX) : 0;
        dy = Math.abs(gapY) > 4 ? Math.sign(gapY) : 0;
      }
      if (count === 4 && dx && dy) dx = 0;
      if (dx || dy) {
        const heading = (dy < 0 ? "N" : dy > 0 ? "S" : "") + (dx < 0 ? "W" : dx > 0 ? "E" : "");
        if (heading !== dir) {
          dir = heading;
          audition = null;
          refresh();
        }
        const moveSpeed = (90 * dt) / Math.hypot(dx, dy);
        x = Math.max(0.06, Math.min(0.94, x + (dx * moveSpeed) / w));
        y = Math.max(0.33, Math.min(0.94, y + (dy * moveSpeed) / h));
      }
    }
    if (!paused) clock += dt * speed;
    paint();
    raf = requestAnimationFrame(tick);
  }

  function applyZoom(nextZoom, cursorX = null, cursorY = null) {
    const { w, h } = stageSize();
    const { baseW, baseH } = baseTileDims();
    const baseCam = cam.fit ? { zoom: effectiveZoom(), panX: 0, panY: 0, fit: false } : cam;
    const cx = cursorX === null ? w / 2 : cursorX;
    const cy = cursorY === null ? h / 2 : cursorY;
    cam = zoomAtPoint(baseCam, nextZoom, cx, cy, w, h, { minZoom: MIN_ZOOM, maxZoom: MAX_ZOOM, baseW, baseH });
    syncStageUi();
    paint();
  }

  function stepZoom(direction, cursorX = null, cursorY = null) {
    const factor = direction > 0 ? 1.25 : 1 / 1.25;
    applyZoom(effectiveZoom() * factor, cursorX, cursorY);
  }

  stage.addEventListener("wheel", (e) => {
    e.preventDefault();
    const bounds = stage.getBoundingClientRect();
    const mx = e.clientX - bounds.left;
    const my = e.clientY - bounds.top;
    const factor = e.deltaY < 0 ? 1.22 : 1 / 1.22;
    applyZoom(effectiveZoom() * factor, mx, my);
  }, { passive: false });

  stage.addEventListener("dblclick", (e) => {
    const row = current();
    if (!row) return;
    if (fixActive && fixMode !== "nudge") return;
    if (stageMode !== "inspect") {
      x = 0.5;
      y = 0.68;
      paint();
      return;
    }
    const bounds = stage.getBoundingClientRect();
    const mx = e.clientX - bounds.left;
    const my = e.clientY - bounds.top;
    const { w, h } = stageSize();
    const { baseW, baseH } = baseTileDims(row);
    const fitZ = fitZoomFor(w, h, baseW, baseH, { minZoom: MIN_ZOOM, maxZoom: MAX_ZOOM });
    if (cam.fit || effectiveZoom(row) < fitZ * 1.45) {
      const targetZ = Math.min(MAX_ZOOM, Math.max(8, Math.round(fitZ * 2.2)));
      applyZoom(targetZ, mx, my);
    } else {
      resetViewToFit();
      syncStageUi();
      paint();
    }
  });

  const walkPoint = (e) => {
    const bounds = stage.getBoundingClientRect();
    walkPointer = { x: (e.clientX - bounds.left) / bounds.width, y: (e.clientY - bounds.top) / bounds.height };
  };

  stage.onpointerdown = (e) => {
    stage.focus({ preventScroll: true });
    const row = current();
    const shouldPan =
      e.button === 1 ||
      (fixActive && spaceHeld && e.button === 0) ||
      (stageMode === "inspect" && !(fixActive && !row?.mirror && fixMode !== "nudge"));
    if (shouldPan) {
      e.preventDefault();
      stage.setPointerCapture(e.pointerId);
      const zNow = effectiveZoom();
      if (cam.fit) cam = { zoom: zNow, panX: 0, panY: 0, fit: false };
      panDrag = {
        pointerId: e.pointerId,
        startX: e.clientX,
        startY: e.clientY,
        panX: cam.panX,
        panY: cam.panY,
      };
      stage.classList.add("panning");
      return;
    }
    if (fixActive && !row?.mirror && fixMode !== "nudge" && e.button === 0) {
      if (fixBusy || !lastDrawBox) return;
      e.preventDefault();
      const bounds = stage.getBoundingClientRect();
      const mx = e.clientX - bounds.left;
      const my = e.clientY - bounds.top;
      hoverPoint = { mx, my };
      ensureMaskSize(lastDrawBox.baseW, lastDrawBox.baseH);
      const pt = screenToTilePoint(
        mx,
        my,
        lastDrawBox.boxX,
        lastDrawBox.boxY,
        lastDrawBox.drawW,
        lastDrawBox.drawH,
        aiMaskCanvas.width,
        aiMaskCanvas.height,
      );
      stage.setPointerCapture(e.pointerId);
      paintDrag = { pointerId: e.pointerId, lastX: pt.x, lastY: pt.y };
      paintMaskStroke(pt.x, pt.y, pt.x, pt.y, lastDrawBox.baseW);
      syncFixUi(row, currentPoseIndex(row.take.sprite));
      paint();
      return;
    }
    if (stageMode === "walk" && e.button === 0) {
      stage.setPointerCapture(e.pointerId);
      walkPoint(e);
    }
  };
  stage.onpointermove = (e) => {
    if (fixActive && fixMode !== "nudge") {
      const bounds = stage.getBoundingClientRect();
      hoverPoint = { mx: e.clientX - bounds.left, my: e.clientY - bounds.top };
    }
    if (panDrag && e.pointerId === panDrag.pointerId) {
      const { w, h } = stageSize();
      const { baseW, baseH } = baseTileDims();
      const clamped = clampPan(
        panDrag.panX + (e.clientX - panDrag.startX),
        panDrag.panY + (e.clientY - panDrag.startY),
        cam.zoom,
        w,
        h,
        baseW,
        baseH,
      );
      cam = { ...cam, panX: clamped.panX, panY: clamped.panY, fit: false };
      paint();
      return;
    }
    if (paintDrag && e.pointerId === paintDrag.pointerId && lastDrawBox) {
      const pt = screenToTilePoint(
        hoverPoint.mx,
        hoverPoint.my,
        lastDrawBox.boxX,
        lastDrawBox.boxY,
        lastDrawBox.drawW,
        lastDrawBox.drawH,
        aiMaskCanvas.width,
        aiMaskCanvas.height,
      );
      paintMaskStroke(paintDrag.lastX, paintDrag.lastY, pt.x, pt.y, lastDrawBox.baseW);
      paintDrag.lastX = pt.x;
      paintDrag.lastY = pt.y;
      paint();
      return;
    }
    if (walkPointer) walkPoint(e);
    else if (fixActive && fixMode !== "nudge") paint();
  };
  stage.onpointerup = stage.onpointercancel = (e) => {
    if (panDrag && e.pointerId === panDrag.pointerId) {
      panDrag = null;
      stage.classList.remove("panning");
    }
    if (paintDrag && e.pointerId === paintDrag.pointerId) {
      paintDrag = null;
      if (fixMode === "ai") aiMaskDirty = hasMaskPixels(aiMaskCtx, aiMaskCanvas);
      else if (fixMode === "alpha") {
        eraseMaskDirty = hasMaskPixels(eraseMaskCtx, eraseMaskCanvas);
        restoreMaskDirty = hasMaskPixels(restoreMaskCtx, restoreMaskCanvas);
      }
      const row = current();
      if (row) syncFixUi(row, currentPoseIndex(row.take.sprite));
      paint();
    }
    walkPointer = null;
  };
  stage.onpointerleave = () => {
    if (hoverPoint && !paintDrag) {
      hoverPoint = null;
      if (fixActive) paint();
    }
  };
  const stopMoving = () => {
    keys.clear();
    walkPointer = null;
    panDrag = null;
    paintDrag = null;
    spaceHeld = false;
    stage.classList.remove("panning", "pan-ready");
  };
  stage.onblur = stopMoving;
  window.addEventListener("blur", stopMoving);

  // Workspace-level keyboard shortcuts when Sprites workspace is active.
  window.addEventListener("keydown", (e) => {
    if (!open() || e.defaultPrevented) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (/INPUT|TEXTAREA|SELECT/.test(e.target.tagName) || e.target.isContentEditable) return;
    if (document.querySelector("dialog[open], [popover]:popover-open")) return;
    if (e.code === "Space" && e.target.matches("button:focus-visible, summary:focus-visible, a:focus-visible")) return;

    const k = e.key;
    const kl = k.toLowerCase();
    if (k === "?") {
      e.preventDefault();
      editor.openShortcuts?.();
      return;
    }
    if (e.code === "Space") {
      if (fixActive && fixMode !== "nudge") {
        e.preventDefault();
        if (!spaceHeld) {
          spaceHeld = true;
          stage.classList.add("pan-ready");
          paint();
        }
        return;
      }
      if (!current()) return;
      e.preventDefault();
      paused = !paused;
      syncStageUi();
      paint();
      return;
    }
    if (k === "Escape" && fixActive) {
      if (fixBusy) return;
      e.preventDefault();
      setFixActive(false);
      return;
    }
    if (k === "[" || k === ",") {
      e.preventDefault();
      stepPose(-1);
      return;
    }
    if (k === "]" || k === ".") {
      e.preventDefault();
      stepPose(1);
      return;
    }
    if (k === "+" || k === "=") {
      e.preventDefault();
      stepZoom(1);
      return;
    }
    if (k === "-") {
      e.preventDefault();
      stepZoom(-1);
      return;
    }
    if (k === "0") {
      e.preventDefault();
      if (stageMode === "inspect") resetViewToFit();
      else { x = 0.5; y = 0.68; }
      syncStageUi();
      paint();
      return;
    }
    if (/^[1-9]$/.test(k)) {
      const idx = Number(k) - 1;
      const row = current();
      if (row && idx < row.take.sprite.poses) {
        e.preventDefault();
        jumpToPose(idx);
      }
      return;
    }
    if (kl === "h") {
      if (!current()) return;
      e.preventDefault();
      $("spriteHd").click();
      return;
    }
    if (kl === "o") {
      if ($("spriteOnion").disabled) return;
      e.preventDefault();
      showOnion = !showOnion;
      syncStageUi();
      paint();
      return;
    }
    if (kl === "p") {
      if (!current()) return;
      e.preventDefault();
      showPivot = !showPivot;
      syncStageUi();
      paint();
      return;
    }
    if (kl === "r") {
      if ($("spriteRefToggle").disabled) return;
      e.preventDefault();
      showRef = !showRef;
      syncStageUi();
      return;
    }
    if (kl === "f") {
      if ($("spriteFixToggle").disabled) return;
      e.preventDefault();
      setFixActive(!fixActive);
      return;
    }
    if (kl === "m") {
      e.preventDefault();
      setStageMode(stageMode === "inspect" ? "walk" : "inspect");
      return;
    }
    if (stageMode === "walk") {
      const h = inputs[kl];
      if (h) {
        e.preventDefault();
        keys.add(h);
      }
    }
  });
  window.addEventListener("keyup", (e) => {
    if (!open()) return;
    if (e.code === "Space" && spaceHeld) {
      spaceHeld = false;
      stage.classList.remove("pan-ready");
      if (fixActive) paint();
    }
    const h = inputs[e.key.toLowerCase()];
    if (h) keys.delete(h);
  });

  function setStageMode(nextMode) {
    if (stageMode === nextMode) return;
    stageMode = nextMode;
    if (stageMode === "walk" && fixActive) {
      fixActive = false;
      paintDrag = null;
      hoverPoint = null;
      spaceHeld = false;
      stage.classList.remove("fix-paint", "pan-ready");
      clearFixMasks();
    }
    keys.clear();
    walkPointer = null;
    panDrag = null;
    stage.classList.remove("panning");
    if (stageMode === "inspect") {
      resetViewToFit();
    } else {
      x = 0.5;
      y = 0.68;
      cam = { zoom: Math.min(4, effectiveZoom()), panX: 0, panY: 0, fit: false };
    }
    syncStageUi();
    paint();
  }

  // Position and track Next draft settings popover
  const draftPopover = $("spriteDraftPopover");
  const draftBtn = $("spriteDraftSettingsBtn");
  function positionDraftPopover() {
    if (!draftPopover || !draftBtn) return;
    const r = draftBtn.getBoundingClientRect();
    const popW = Math.min(420, window.innerWidth - 16);
    const left = Math.max(8, Math.min(window.innerWidth - popW - 8, r.right - popW));
    const top = Math.min(window.innerHeight - 80, r.bottom + 6);
    draftPopover.style.left = `${left}px`;
    draftPopover.style.top = `${top}px`;
  }
  draftPopover?.addEventListener("toggle", (e) => {
    const isOpen = e.newState === "open";
    draftBtn?.setAttribute("aria-expanded", String(isOpen));
    if (isOpen) positionDraftPopover();
  });
  window.addEventListener("resize", () => {
    if (draftPopover?.matches(":popover-open")) positionDraftPopover();
  });

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
  $("spriteKeyColor").onchange = (e) => {
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
    $("spriteDraftSummary").textContent = draftSettingsSummary({
      canvas,
      keyColor,
      frames: clipFrames,
      artStylePreset,
      artStyleCustom,
    });
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
        const url = editor.viewUrl(item.file, item.subfolder);
        const imgBtn = document.createElement("button");
        imgBtn.type = "button";
        imgBtn.className = "sprite-prop-thumb";
        imgBtn.title = "Inspect " + item.file;
        imgBtn.onclick = () => editor.inspectImage?.(item.file, url);
        const img = document.createElement("img");
        img.src = url;
        img.alt = item.file;
        imgBtn.append(img);
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
        dl.href = url;
        dl.download = item.file;
        dl.textContent = "PNG";
        actions.append(baseBtn, dl);
        card.append(imgBtn, nameEl, actions);
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

  $("spriteModeInspect").onclick = () => setStageMode("inspect");
  $("spriteModeWalk").onclick = () => setStageMode("walk");
  $("spriteZoomOut").onclick = () => stepZoom(-1);
  $("spriteZoomIn").onclick = () => stepZoom(1);
  $("spriteZoomReset").onclick = () => {
    if (cam.fit) applyZoom(4);
    else if (effectiveZoom() < 7.5) applyZoom(8);
    else {
      resetViewToFit();
      syncStageUi();
      paint();
    }
  };
  $("spriteSurfaceSwatches")?.addEventListener("click", (e) => {
    const sw = e.target.closest("button[data-backdrop]");
    if (!sw) return;
    backdrop = sw.dataset.backdrop || "matte";
    syncStageUi();
    paint();
  });
  $("spritePivot").onclick = () => {
    showPivot = !showPivot;
    syncStageUi();
    paint();
  };
  $("spriteOnion").onclick = () => {
    showOnion = !showOnion;
    syncStageUi();
    paint();
  };
  $("spriteRefToggle").onclick = () => {
    showRef = !showRef;
    syncStageUi();
  };
  $("spriteFixToggle").onclick = () => setFixActive(!fixActive);
  $("spriteFixDone").onclick = () => setFixActive(false);
  for (const tabBtn of $("spriteFixBar")?.querySelectorAll("button[data-fix-mode]") || []) {
    tabBtn.onclick = () => {
      if (fixBusy) return;
      fixMode = tabBtn.dataset.fixMode || "ai";
      paintDrag = null;
      syncStageUi();
      paint();
    };
  }
  $("spriteFixAiPaint").onclick = () => {
    fixAiTool = "paint";
    syncStageUi();
    paint();
  };
  $("spriteFixAiUnpaint").onclick = () => {
    fixAiTool = "unpaint";
    syncStageUi();
    paint();
  };
  $("spriteFixAlphaErase").onclick = () => {
    fixAlphaTool = "erase";
    syncStageUi();
    paint();
  };
  $("spriteFixAlphaRestore").onclick = () => {
    fixAlphaTool = "restore";
    syncStageUi();
    paint();
  };
  const syncBrushSize = (val) => {
    fixBrushSize = Math.max(4, Math.min(32, Math.round(Number(val) || 10)));
    const row = current();
    if (row) syncFixUi(row, currentPoseIndex(row.take.sprite));
    paint();
  };
  $("spriteFixBrushAi").oninput = (e) => syncBrushSize(e.target.value);
  $("spriteFixBrushAlpha").oninput = (e) => syncBrushSize(e.target.value);
  $("spriteFixAiClear").onclick = () => {
    clearFixMasks("ai");
    syncStageUi();
    paint();
  };
  $("spriteFixAlphaClear").onclick = () => {
    clearFixMasks("alpha");
    syncStageUi();
    paint();
  };
  $("spriteFixPrompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("spriteFixApplyAi").click();
    } else if (e.key === "Escape" && !fixBusy) {
      e.preventDefault();
      setFixActive(false);
    }
  });
  $("spriteFixApplyAi").onclick = async () => {
    const row = current();
    if (!row || row.mirror || fixBusy) return;
    const promptText = $("spriteFixPrompt").value.trim();
    if (!promptText && !aiMaskDirty) {
      editor.toast("Paint the defect area or describe what to fix.");
      $("spriteFixPrompt").focus();
      return;
    }
    const pose = currentPoseIndex(row.take.sprite);
    const mask = aiMaskDirty ? exportMaskPng(aiMaskCanvas) : null;
    const strength = $("spriteFixStrength").value || "full";
    fixBusy = true;
    fixBusyLabel = "Fixing pose with H3Take…";
    syncStageUi();
    try {
      const res = await editor.fixPoseAi(row.clip, row.take, pose, { prompt: promptText, mask, strength });
      clearFixMasks("ai");
      invalidateTakeImages(row);
      fixPoseInfoKey = "";
      editor.toast("Pose " + (pose + 1) + " repaired" + (res?.seed != null ? " (#" + res.seed + ")" : "") + ".");
      refresh();
    } catch (error) {
      editor.toast(error.message || "Could not repair pose.", true);
    } finally {
      fixBusy = false;
      fixBusyLabel = "";
      syncStageUi();
      paint();
    }
  };
  $("spriteFixApplyAlpha").onclick = async () => {
    const row = current();
    if (!row || row.mirror || fixBusy) return;
    if (!eraseMaskDirty && !restoreMaskDirty) {
      editor.toast("Paint an Erase or Restore stroke on the sprite first.");
      return;
    }
    const pose = currentPoseIndex(row.take.sprite);
    const erase_mask = eraseMaskDirty ? exportMaskPng(eraseMaskCanvas) : null;
    const restore_mask = restoreMaskDirty ? exportMaskPng(restoreMaskCanvas) : null;
    fixBusy = true;
    fixBusyLabel = "Applying alpha…";
    syncStageUi();
    try {
      await editor.fixPoseAlpha(row.clip, row.take, pose, { erase_mask, restore_mask });
      clearFixMasks("alpha");
      invalidateTakeImages(row);
      fixPoseInfoKey = "";
      editor.toast("Alpha updated on pose " + (pose + 1) + ".");
      refresh();
    } catch (error) {
      editor.toast(error.message || "Could not apply alpha fix.", true);
    } finally {
      fixBusy = false;
      fixBusyLabel = "";
      syncStageUi();
      paint();
    }
  };
  async function runNudge(delta) {
    const row = current();
    if (!row || row.mirror || fixBusy) return;
    const pose = currentPoseIndex(row.take.sprite);
    fixBusy = true;
    fixBusyLabel = delta < 0 ? "Nudging −1f…" : "Nudging +1f…";
    syncStageUi();
    try {
      const res = await editor.nudgePose(row.clip, row.take, pose, delta);
      clearFixMasks();
      invalidateTakeImages(row);
      if (fixPoseInfo && Number.isInteger(res?.source_index)) {
        fixPoseInfo = { ...fixPoseInfo, source_index: res.source_index, patched: posePatchInfo(row.take.sprite, pose) || { kind: "nudge" } };
      }
      fixPoseInfoKey = "";
      refresh();
    } catch (error) {
      editor.toast(error.message || "Could not nudge pose.", true);
    } finally {
      fixBusy = false;
      fixBusyLabel = "";
      syncStageUi();
      paint();
    }
  }
  $("spriteFixNudgeBack").onclick = () => void runNudge(-1);
  $("spriteFixNudgeFwd").onclick = () => void runNudge(1);
  $("spriteFixRevert").onclick = async () => {
    const row = current();
    if (!row || row.mirror || fixBusy) return;
    const pose = currentPoseIndex(row.take.sprite);
    fixBusy = true;
    fixBusyLabel = "Reverting pose…";
    syncStageUi();
    try {
      await editor.revertPose(row.clip, row.take, pose);
      clearFixMasks();
      invalidateTakeImages(row);
      fixPoseInfoKey = "";
      editor.toast("Pose " + (pose + 1) + " reverted to original.");
      refresh();
    } catch (error) {
      editor.toast(error.message || "Could not revert pose.", true);
    } finally {
      fixBusy = false;
      fixBusyLabel = "";
      syncStageUi();
      paint();
    }
  };
  $("spriteHd").onclick = () => {
    const row = current();
    if (!row) return;
    const key = hdKey(row);
    const existing = hdSheets.get(key);
    if (existing?.error) {
      hdSheets.delete(key);
      hdPref = "on";
      syncStageUi();
      void ensureHd(row, { force: true, explicit: true });
      return;
    }
    const currentlyHd = shouldUseHd(hdPref, effectiveZoom(row));
    hdPref = currentlyHd ? "off" : "on";
    syncStageUi();
    if (hdPref === "on") void ensureHd(row, { force: true, explicit: true });
    paint();
  };
  $("spriteSpeed").onchange = (e) => {
    speed = Number(e.target.value) || 1;
    syncStageUi();
    paint();
  };
  $("spriteCenter").onclick = () => {
    if (stageMode === "inspect") resetViewToFit();
    else {
      x = 0.5;
      y = 0.68;
    }
    syncStageUi();
    paint();
  };
  $("spritePlay").onclick = () => {
    paused = !paused;
    syncStageUi();
    paint();
  };
  $("spritePrev").onclick = () => stepPose(-1);
  $("spriteStep").onclick = () => stepPose(1);
  $("spriteKeysBtn").onclick = () => editor.openShortcuts?.();
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
      editor.toast("Could not load that floor image.", true);
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
    if ($("spriteExportStatus")) {
      $("spriteExportStatus").textContent = "Packing " + picks.length + " animation slots…";
    }
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
