// Source coordinates are fractions, so moving a placement from draft to final keeps its framing.
// Fit the cut's aspect ratio inside the requested region, exactly as crop then cover does at export.
export function cropWindow(source, output, crop = null) {
  const { x, y, w, h } = crop || { x: 0, y: 0, w: 1, h: 1 };
  const ratio = output[0] * source[1] / (output[1] * source[0]);
  const width = Math.min(w, h * ratio);
  const height = Math.min(h, w / ratio);
  return { x: x + (w - width) / 2, y: y + (h - height) / 2, w: width, h: height };
}

export async function editCrop(url, seconds, source, output, original) {
  const get = (id) => document.getElementById(id);
  const dialog = get("cropper"), stage = get("cropStage"), box = get("cropBox");
  const video = get("cropSource"), zoom = get("cropZoom"), value = get("cropValue");
  const base = cropWindow(source, output);
  let rect = cropWindow(source, output, original);
  let result = original || null;
  const events = new AbortController();
  const listen = (el, type, callback) => el.addEventListener(type, callback, { signal: events.signal });
  const clamp = (v, max) => Math.max(0, Math.min(max, v));
  function paint() {
    Object.assign(box.style, { left: rect.x * 100 + "%", top: rect.y * 100 + "%",
      width: rect.w * 100 + "%", height: rect.h * 100 + "%" });
    zoom.value = String(base.w / rect.w);
    value.textContent = Math.round(100 * base.w / rect.w) + "%";
  }
  function place(x, y, w, h) {
    rect = { x: clamp(x, 1 - w), y: clamp(y, 1 - h), w, h };
    result = { ...rect };
    paint();
  }
  function resize(factor) {
    const w = base.w / factor, h = base.h / factor;
    place(rect.x + (rect.w - w) / 2, rect.y + (rect.h - h) / 2, w, h);
  }
  stage.style.setProperty("--source-ratio", String(source[0] / source[1]));
  zoom.max = String(Math.max(8, base.w / rect.w));
  listen(zoom, "input", () => resize(Number(zoom.value)));
  let drag = null;
  listen(box, "pointerdown", (event) => {
    if (event.button !== 0 || drag) return;
    const bounds = stage.getBoundingClientRect();
    drag = { id: event.pointerId, x: event.clientX, y: event.clientY, bounds, start: rect,
      resize: event.target === get("cropHandle") };
    box.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  listen(box, "pointermove", (event) => {
    if (event.pointerId !== drag?.id) return;
    const dx = (event.clientX - drag.x) / drag.bounds.width;
    const dy = (event.clientY - drag.y) / drag.bounds.height;
    const { x, y, w, h } = drag.start;
    if (drag.resize) {
      const pw = w * drag.bounds.width, ph = h * drag.bounds.height;
      const factor = 1 + ((event.clientX - drag.x) * pw + (event.clientY - drag.y) * ph) / (pw * pw + ph * ph);
      const width = Math.max(base.w / Number(zoom.max), Math.min(1 - x, (1 - y) * w / h, w * factor));
      place(x, y, width, width * h / w);
    } else place(x + dx, y + dy, w, h);
  });
  listen(box, "lostpointercapture", () => { drag = null; });
  listen(box, "keydown", (event) => {
    const delta = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[event.key];
    if (!delta) return;
    event.preventDefault();
    const step = event.shiftKey ? 10 : 1;
    place(rect.x + delta[0] * step / source[0], rect.y + delta[1] * step / source[1], rect.w, rect.h);
  });
  listen(get("cropReset"), "click", () => { rect = base; result = null; paint(); });
  listen(video, "loadedmetadata", () => { video.currentTime = seconds; });
  listen(video, "error", () => { get("cropError").hidden = false; });
  get("cropError").hidden = true;
  video.src = url;
  paint();
  dialog.returnValue = "";
  dialog.showModal();
  await new Promise((resolve) => dialog.addEventListener("close", resolve, { once: true }));
  events.abort();
  video.removeAttribute("src");
  video.load();
  return dialog.returnValue === "apply" ? result : undefined;
}
