// Runs a VFX clip's module, away from the editor.
//
// A VFX clip is code the user or an agent wrote, so it is kept apart twice over. It runs in a worker, which has no
// page: it cannot navigate anywhere, open a window or touch a document. That worker lives in a sandboxed frame with
// a policy that allows no network at all, so it cannot fetch, post or load anything either. All it has is what the
// editor hands it over one message port: the module's text, the frame size, its control values, its named pictures
// and data, and for each frame the picture below. It answers with a bitmap. A module that does not answer in time
// is stopped by throwing the whole frame away.
//
// The module's contract:
//   export const meta = { controls: { name: { type, default, min?, max?, step?, options?, label? } } }   (optional)
//   export async function setup(env) {}                                                                  (optional)
//   export function frame(env, k) {}      draws frame k; must depend only on k, env.params and env.below
// env: canvas (an OffscreenCanvas: take a 2d or a WebGL context from it), width, height, fps, frames, params,
//      below (ImageBitmap or null), inputs (the clip's named pictures, as ImageBitmaps), data (the clip's JSON data),
//      random(seed) (a repeatable 0..1 generator), and for modules that mention them: THREE (three.js) and
//      belowTexture (the picture below as a THREE.Texture, the right way up, updated before every frame).

const WORKER = String.raw`
let port = null, mod = null, env = null, THREE = null, wantsTexture = false, held = null;
const answer = (id, body, transfer) => port.postMessage(Object.assign({ id }, body), transfer || []);
const seeded = (seed) => { let a = (seed >>> 0) || 1; return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; };
const withDefaults = (controls, params) => {
  const out = {};
  for (const [key, c] of Object.entries(controls || {})) out[key] = params && key in params ? params[key] : c.default;
  for (const [key, v] of Object.entries(params || {})) if (!(key in out)) out[key] = v;
  return out;
};
async function handle(m) {
  try {
    if (m.type === 'load') {
      if (m.three && !THREE) {
        const core = URL.createObjectURL(new Blob([m.three.core], { type: 'text/javascript' }));
        const main = URL.createObjectURL(new Blob([m.three.module.split('./three.core.js').join(core)], { type: 'text/javascript' }));
        THREE = await import(main);
      }
      const canvas = new OffscreenCanvas(m.width, m.height);
      const url = URL.createObjectURL(new Blob([m.source], { type: 'text/javascript' }));
      try { mod = await import(url); } finally { URL.revokeObjectURL(url); }
      if (typeof mod.frame !== 'function') throw new Error('the module must export a function frame(env, k)');
      const meta = mod.meta && typeof mod.meta === 'object' ? mod.meta : {};
      env = { canvas, width: m.width, height: m.height, fps: m.fps, frames: m.frames, params: withDefaults(meta.controls, m.params),
              below: null, belowTexture: null, inputs: m.inputs || {}, data: m.data === undefined ? null : m.data, THREE,
              random: seeded, k: 0, t: 0 };
      wantsTexture = Boolean(THREE) && m.source.includes('belowTexture');
      if (wantsTexture) { env.belowTexture = new THREE.Texture(); env.belowTexture.colorSpace = THREE.SRGBColorSpace; }
      if (typeof mod.setup === 'function') await mod.setup(env);
      answer(m.id, { type: 'ready', meta: JSON.parse(JSON.stringify({ controls: meta.controls || {} })) });
    } else if (m.type === 'params') {
      env.params = withDefaults((mod.meta || {}).controls, m.params);
      answer(m.id, { type: 'ok' });
    } else if (m.type === 'frame') {
      env.below = m.below || null; env.k = m.k; env.t = m.k / env.fps;
      if (wantsTexture && m.below) {
        // A bitmap reaches WebGL upside down, so the texture gets a copy turned the right way up.
        const turned = await createImageBitmap(m.below, { imageOrientation: 'flipY' });
        if (held) held.close();
        held = turned;
        env.belowTexture.image = turned;
        env.belowTexture.needsUpdate = true;
      }
      await mod.frame(env, m.k);
      const bitmap = await createImageBitmap(env.canvas);
      if (m.below && m.below.close) m.below.close();
      env.below = null;
      answer(m.id, { type: 'drawn', bitmap }, [bitmap]);
    }
  } catch (error) {
    const where = /blob:[^)\s]*:(\d+):(\d+)/.exec(String(error && error.stack || ''));
    answer(m.id, { type: 'error', message: String(error && error.message || error), line: where ? Number(where[1]) : null });
  }
}
self.onmessage = (event) => {
  if (port || !event.data || event.data.type !== 'port') return;
  port = event.ports[0];
  port.onmessage = (e) => handle(e.data);
  port.postMessage({ type: 'boot' });
};
`;

// The frame's own page runs none of the module: it only starts the worker and passes it the editor's port.
// A sandboxed frame may not start a worker from a blob, so the worker's own code travels as a data URL.
const BOOT = "const worker = new Worker('data:text/javascript;charset=utf-8,' + encodeURIComponent(" + JSON.stringify(WORKER) + "), { type: 'module' });\n"
  + "addEventListener('message', (event) => {\n"
  + "  if (event.data && event.data.type === 'port') worker.postMessage({ type: 'port' }, [event.ports[0]]);\n"
  + '});\n'
  + "parent.postMessage({ type: 'frame' }, '*');\n";

const PAGE = '<!doctype html><meta charset="utf-8">'
  + '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; script-src \'unsafe-inline\' blob: data:; worker-src data:; '
  + 'img-src blob: data:; style-src \'unsafe-inline\'; font-src data:; connect-src \'none\'; form-action \'none\'; base-uri \'none\'; webrtc \'block\'">'
  + '<script type="module">' + BOOT.split('</').join('<\\/') + '</' + 'script>';

let threeText = null;

/** The text of three.js, fetched once, for modules that use it. */
async function three() {
  if (!threeText) {
    const get = async (file) => {
      const reply = await fetch('/h3take/editor-files/vendor/' + file);
      if (!reply.ok) throw new Error('three.js is missing from the editor (' + file + ')');
      return reply.text();
    };
    threeText = Promise.all([get('three.core.js'), get('three.module.js')]).then(([core, module]) => ({ core, module }));
  }
  return threeText;
}

export class VfxError extends Error {
  constructor(message, line) {
    super(message);
    this.line = line;
  }
}

export class VfxRuntime {
  constructor() {
    this.frame_ = document.createElement('iframe');
    this.frame_.setAttribute('sandbox', 'allow-scripts');
    this.frame_.setAttribute('aria-hidden', 'true');
    this.frame_.style.cssText = 'position:fixed;width:1px;height:1px;left:-10px;top:-10px;border:0;visibility:hidden';
    this.waiting = new Map();
    this.next = 1;
    this.dead = null;
    this.loads = 0;
    this.port = null;
    this.booted = new Promise((resolve, reject) => { this.boot = resolve; this.fail = reject; });
    this.booted.catch(() => {});
    this.listen = (event) => {
      if (event.source !== this.frame_.contentWindow || !event.data || event.data.type !== 'frame' || this.port) return;
      const channel = new MessageChannel();
      this.port = channel.port1;
      this.port.onmessage = (e) => this.heard(e.data || {});
      this.frame_.contentWindow.postMessage({ type: 'port' }, '*', [channel.port2]);
    };
    addEventListener('message', this.listen);
    // The frame loads once. A second load means its page went somewhere else, which the worker cannot do and the
    // page has no reason to: stop everything.
    this.frame_.addEventListener('load', () => {
      this.loads += 1;
      if (this.loads > 1) this.kill('The VFX sandbox left its page and was stopped.');
    });
    this.frame_.srcdoc = PAGE;
    document.body.append(this.frame_);
  }

  heard(m) {
    if (m.type === 'boot') return this.boot();
    const wait = this.waiting.get(m.id);
    if (!wait) return;
    this.waiting.delete(m.id);
    clearTimeout(wait.timer);
    if (m.type === 'error') wait.reject(new VfxError(m.message, m.line));
    else wait.resolve(m);
  }

  /** Stop the code for good: the frame and its worker go, and everything still waiting fails with the reason. */
  kill(reason) {
    if (this.dead) return;
    this.dead = reason;
    removeEventListener('message', this.listen);
    for (const wait of this.waiting.values()) {
      clearTimeout(wait.timer);
      wait.reject(new VfxError(reason, null));
    }
    this.waiting.clear();
    this.fail(new VfxError(reason, null));
    if (this.port) this.port.close();
    this.frame_.remove();
  }

  call(message, transfer = [], seconds = 20) {
    if (this.dead) return Promise.reject(new VfxError(this.dead, null));
    return this.booted.then(() => new Promise((resolve, reject) => {
      const id = this.next++;
      const timer = setTimeout(() => {
        // A worker stuck in a loop never answers again, so it is thrown away, not waited for.
        this.kill('The code did not answer in ' + seconds + ' seconds, so it was stopped. It may be stuck in a loop.');
      }, seconds * 1000);
      this.waiting.set(id, { resolve, reject, timer });
      this.port.postMessage(Object.assign({ id }, message), transfer);
    }));
  }

  /** Load a module at a size. Resolves to its meta ({ controls }). */
  async load({ source, width, height, fps, frames, params, inputs = {}, data = null }) {
    const uses = /\bTHREE\b/.test(source);
    const reply = await this.call({ type: 'load', source, width, height, fps, frames, params, inputs, data, three: uses ? await three() : null },
      Object.values(inputs), 40);
    return reply.meta;
  }

  setParams(params) {
    return this.call({ type: 'params', params });
  }

  /** Frame k, drawn over `below` (an ImageBitmap, handed over to the code, or null). Resolves to an ImageBitmap. */
  async frame(k, below = null) {
    const reply = await this.call({ type: 'frame', k, below }, below ? [below] : []);
    return reply.bitmap;
  }

  dispose() {
    this.kill('The VFX runtime was closed.');
  }
}
