import { app } from "/scripts/app.js";

const STYLE = `
.h3ed { box-sizing: border-box; display: block; padding: 8px 12px 10px; color: #f4efe6; background: #161410; font: 13px/1.35 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; user-select: none; overflow: hidden; }
.h3ed * { box-sizing: border-box; }
.h3ed button, .h3ed select { font: inherit; color: inherit; }
.h3head { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; min-width: 0; }
.h3time { font-variant-numeric: tabular-nums; font-weight: 650; letter-spacing: -0.02em; flex: none; }
.h3budget { margin-left: auto; color: #d2c6b6; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }
.h3ed select { flex: none; background: #221f1b; border: 1px solid #4a433b; border-radius: 4px; padding: 2px 6px; }
.h3ed select:focus, .h3ed button:focus-visible { outline: 2px solid #e4c27a; outline-offset: 2px; }
.h3ruler { position: relative; height: 16px; margin-top: 2px; cursor: ew-resize; }
.h3tick { position: absolute; top: 0; color: #d2c6b6; font-size: 11px; font-variant-numeric: tabular-nums; }
.h3lane { position: relative; height: 44px; margin-top: 4px; background: #221f1b; border-radius: 4px; touch-action: none; }
.h3block { position: absolute; top: 6px; height: 32px; border-radius: 4px; padding: 0 10px; overflow: hidden; cursor: grab; border: 1px solid transparent; color: #f4efe6; display: flex; align-items: center; gap: 6px; white-space: nowrap; }
.h3block:active { cursor: grabbing; }
.h3block.generate { background: #3d3428; }
.h3block.hold { background: #243034; }
.h3block.selected { border-color: #e4c27a; }
.h3block b { font-weight: 650; }
.h3block span { color: #e7d7bc; font-size: 12px; font-variant-numeric: tabular-nums; }
.h3edge { position: absolute; top: 0; width: 8px; height: 100%; cursor: ew-resize; }
.h3edge.left { left: 0; }
.h3edge.right { right: 0; }
.h3pass { position: absolute; top: 2px; height: 16px; border-radius: 3px; font-size: 11px; line-height: 16px; padding: 0 6px; color: #f4efe6; background: #3d3428; overflow: hidden; white-space: nowrap; }
.h3pass.holdish { background: #2c3a34; }
.h3passes { position: relative; height: 20px; margin-top: 4px; }
.h3play { position: absolute; top: 0; bottom: 0; width: 2px; background: #f2e6c8; pointer-events: none; }
.h3note { margin-top: 8px; color: #d2c6b6; }
.h3note.warn { color: #e6c07a; }
.h3hint { margin-top: 4px; color: #d2c6b6; font-size: 12px; }
`;

function injectStyle() {
  const tag = document.getElementById("h3ed-style") || document.createElement("style");
  tag.id = "h3ed-style";
  tag.textContent = STYLE;
  if (!tag.parentNode) document.head.appendChild(tag);
}

function seconds(frames) {
  const s = Math.max(0, frames) / 24;
  return Math.floor(s / 60) + ":" + (s % 60).toFixed(1).padStart(4, "0");
}

function capFor(gb) {
  if (gb < 24) return 124;
  if (gb < 48) return 243;
  return 362;
}

function snapLegal(n) {
  n = Math.round(n);
  if (n < 5) return null;
  return 5 + 17 * Math.floor((n - 5) / 17);
}

function snapHold(n) {
  n = Math.max(39, Math.round(n));
  const k = Math.round((n - 39) / 51);
  return 39 + 51 * Math.max(0, k);
}

function splitSpan(total, maxBlock) {
  const cap = snapLegal(maxBlock);
  const legal = snapLegal(total);
  const overlap = 39;
  if (!cap || !legal) return { blocks: [], snapped: legal, error: "too short to split" };
  if (legal <= cap) return { blocks: [{ start: 0, length: legal, hold: 0, pass: 1 }], snapped: legal, error: null };
  const blocks = [];
  let covered = 0;
  for (let guard = 0; covered < legal && guard < 24; guard += 1) {
    if (!blocks.length) {
      blocks.push({ start: 0, length: Math.min(cap, legal), hold: 0, pass: 1 });
      covered = blocks[0].length;
      continue;
    }
    const remain = legal - covered;
    const length = snapLegal(Math.min(cap, remain + overlap));
    if (length == null || length - overlap < 5) {
      return { blocks, snapped: legal, error: "cannot fit the remaining " + remain + " frames" };
    }
    const start = covered - overlap;
    const next = start + length;
    if (next <= covered) return { blocks, snapped: legal, error: "pass did not advance" };
    blocks.push({ start, length, hold: overlap, pass: blocks.length + 1 });
    covered = next;
  }
  if (covered < legal) return { blocks, snapped: legal, error: "stopped before the region ended" };
  return { blocks, snapped: legal, error: null };
}

function mountTimeline(node) {
  injectStyle();
  const store = node.widgets?.find((w) => w.name === "timeline_data");
  if (store) {
    store.computeSize = () => [0, -4];
    store.hidden = true;
  }
  const root = document.createElement("div");
  root.className = "h3ed";
  const widget = node.addDOMWidget("timeline_ui", "timeline", root, {
    serialize: false,
    getMinHeight: () => 168,
  });
  widget.serialize = false;
  widget.computeSize = (width) => [width, 168];
  node.setSize([720, 230]);
  root.style.width = "700px";

  let state = {
    vram_gb: 16,
    selected: "after",
    play: 80,
    hold: { start: 124, length: 124, label: "Source" },
    generates: [
      { id: "before", start: 0, length: 124, label: "Before" },
      { id: "after", start: 248, length: 243, label: "After" },
    ],
  };
  try {
    const saved = JSON.parse(store?.value || "{}");
    if (saved.generates) state = { ...state, ...saved };
  } catch (_) {}
  for (const gen of state.generates) {
    const snapped = snapLegal(gen.length);
    if (snapped) gen.length = snapped;
  }

  let dragging = false;
  let laid = 0;

  function fitWidth() {
    const inner = Math.max(320, Math.round((node.size?.[0] || 720) - 20));
    root.style.width = inner + "px";
    return inner - 24;
  }

  function viewFrames() {
    const ends = [state.hold.start + state.hold.length, ...state.generates.map((g) => g.start + g.length), state.play];
    return Math.ceil(Math.max(248, Math.max(...ends) + 24) / 120) * 120;
  }

  function write() {
    if (!store) return;
    store.value = JSON.stringify({
      fps: 24,
      vram_gb: state.vram_gb,
      vram_set: !!state.vram_set,
      hold: state.hold,
      generates: state.generates,
    });
  }

  function targetOf(id) {
    if (id === "hold") return state.hold;
    return state.generates.find((g) => g.id === id);
  }

  function render() {
    if (dragging) return;
    const lanePx = fitWidth();
    const span = viewFrames();
    const px = (frames) => (frames / span) * lanePx;
    const cap = capFor(state.vram_gb);
    root.replaceChildren();

    const head = document.createElement("div");
    head.className = "h3head";
    const time = document.createElement("span");
    time.className = "h3time";
    time.textContent = seconds(state.play);
    const budget = document.createElement("span");
    budget.className = "h3budget";
    budget.textContent = state.vram_gb + " GB · " + cap + "f" + (state.device ? " · " + state.device : "");
    const select = document.createElement("select");
    select.setAttribute("aria-label", "Memory budget");
    for (const gb of [16, 24, 48, 96]) {
      const opt = document.createElement("option");
      opt.value = String(gb);
      opt.textContent = gb + " GB";
      if (gb === state.vram_gb) opt.selected = true;
      select.appendChild(opt);
    }
    select.addEventListener("pointerdown", (event) => event.stopPropagation());
    select.addEventListener("change", () => {
      state.vram_gb = Number(select.value);
      state.vram_set = true;
      write();
      render();
    });
    head.append(time, budget, select);
    root.appendChild(head);

    const ruler = document.createElement("div");
    ruler.className = "h3ruler";
    const step = span > 720 ? 10 : 5;
    for (let s = 0; s <= span / 24; s += step) {
      if (px(s * 24) > lanePx - 22) continue;
      const tick = document.createElement("span");
      tick.className = "h3tick";
      tick.style.left = px(s * 24) + "px";
      tick.textContent = s + "s";
      ruler.appendChild(tick);
    }
    ruler.addEventListener("pointerdown", (event) => {
      event.stopPropagation();
      const rect = ruler.getBoundingClientRect();
      state.play = Math.max(0, Math.min(span, Math.round(((event.clientX - rect.left) / lanePx) * span)));
      write();
      render();
    });
    root.appendChild(ruler);

    const lane = document.createElement("div");
    lane.className = "h3lane";
    const items = [
      ...state.generates.map((b) => ({ ...b, kind: "generate" })),
      { ...state.hold, id: "hold", kind: "hold" },
    ];
    for (const block of items) {
      const el = document.createElement("button");
      el.type = "button";
      el.className = "h3block " + block.kind + (state.selected === block.id ? " selected" : "");
      el.style.left = px(block.start) + "px";
      el.style.width = Math.max(48, px(block.length) - 3) + "px";
      const name = document.createElement("b");
      name.textContent = block.label;
      const sub = document.createElement("span");
      sub.textContent = block.kind === "hold" ? "held" : seconds(block.length);
      const left = document.createElement("i");
      left.className = "h3edge left";
      const right = document.createElement("i");
      right.className = "h3edge right";
      el.append(name, sub, left, right);
      bindDrag(el, block.id, lanePx, span);
      lane.appendChild(el);
    }
    const play = document.createElement("div");
    play.className = "h3play";
    play.style.left = px(state.play) + "px";
    lane.appendChild(play);
    root.appendChild(lane);

    const passes = document.createElement("div");
    passes.className = "h3passes";
    let anySplit = false;
    for (const gen of state.generates) {
      const split = splitSpan(gen.length, cap);
      if (split.blocks.length < 2) continue;
      anySplit = true;
      for (const part of split.blocks) {
        const chip = document.createElement("div");
        chip.className = "h3pass" + (part.hold ? " holdish" : "");
        chip.style.left = px(gen.start + part.start) + "px";
        chip.style.width = Math.max(36, px(part.length) - 3) + "px";
        chip.textContent = "Pass " + part.pass;
        passes.appendChild(chip);
      }
    }
    if (anySplit) root.appendChild(passes);

    const selected = targetOf(state.selected) || state.generates[1];
    const note = document.createElement("div");
    note.className = "h3note";
    if (state.selected === "hold") {
      note.textContent = "Source is held for " + seconds(state.hold.length) + ". Those frames are not resampled.";
    } else {
      const split = splitSpan(selected.length, cap);
      if (split.error) {
        note.className += " warn";
        note.textContent = selected.label + " could not be split: " + split.error + ".";
      } else if (split.blocks.length > 1) {
        note.textContent = selected.label + " is " + split.snapped + " frames. At " + state.vram_gb + " GB it runs as " + split.blocks.length + " passes. Each later pass holds the last 39 frames.";
      } else {
        note.textContent = selected.label + " fits in one pass. The source in the middle is held.";
      }
    }
    root.appendChild(note);

    const hint = document.createElement("div");
    hint.className = "h3hint";
    hint.textContent = "Drag a block to move it. Drag an edge to trim.";
    root.appendChild(hint);
    write();
    laid = lanePx;
  }

  function bindDrag(el, id, lanePx, span) {
    el.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const block = targetOf(id);
      if (!block) return;
      const edge = event.target.classList?.contains("left") ? "left" : event.target.classList?.contains("right") ? "right" : "move";
      dragging = true;
      state.selected = id;
      el.classList.add("selected");
      const origin = { x: event.clientX, start: block.start, length: block.length };
      const move = (ev) => {
        const delta = Math.round(((ev.clientX - origin.x) / lanePx) * span);
        if (edge === "right") {
          block.length = (id === "hold" ? snapHold(origin.length + delta) : snapLegal(origin.length + delta)) || (id === "hold" ? 39 : 22);
        } else if (edge === "left") {
          const end = origin.start + origin.length;
          const start = Math.max(0, origin.start + delta);
          const next = id === "hold" ? snapHold(end - start) : snapLegal(end - start);
          if (!next) return;
          block.length = next;
          block.start = Math.max(0, end - next);
        } else {
          block.start = Math.max(0, origin.start + delta);
        }
        el.style.left = ((block.start / span) * lanePx) + "px";
        el.style.width = Math.max(48, (block.length / span) * lanePx - 3) + "px";
      };
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        dragging = false;
        write();
        render();
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    });
  }

  root.addEventListener("pointerdown", (event) => event.stopPropagation());
  const prevResize = node.onResize;
  node.onResize = function (size) {
    prevResize?.apply(this, arguments);
    const inner = Math.max(320, Math.round(size[0] - 20));
    if (Math.abs(inner - 24 - laid) < 8) return;
    root.style.width = inner + "px";
    render();
  };

  render();
  if (!state.vram_set) {
    fetch("/system_stats")
      .then((response) => response.json())
      .then((stats) => {
        const dev = (stats.devices || [])[0];
        if (!dev || !dev.vram_total || state.vram_set) return;
        state.vram_gb = Math.max(1, Math.round(dev.vram_total / (1024 ** 3)));
        const raw = String(dev.name || "");
        const named = raw.match(/RTX PRO \d+/) || raw.match(/NVIDIA [^:]+/);
        state.device = named ? named[0].trim() : "";
        write();
        render();
      })
      .catch(() => {});
  }
}

app.registerExtension({
  name: "h3take.timeline",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "H3Timeline") return;
    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onCreated?.apply(this, arguments);
      try {
        mountTimeline(this);
      } catch (err) {
        console.error("H3 Timeline failed to mount", err);
      }
      return result;
    };
  },
});
