// Plays a video while enough of it is on screen and `allowed()` says so, and pauses it when it leaves.
function whileVisible(video, allowed) {
  if (!("IntersectionObserver" in window)) { if (allowed()) video.play().catch(() => {}); return; }
  new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting && allowed()) video.play().catch(() => {});
      else if (!entry.isIntersecting && !video.paused) video.pause();
    }
  }, { threshold: 0.35 }).observe(video);
}

// The hero: a tangled ComfyUI graph for one H3 shot folds into the editor, which then plays the cut.
// Everything is laid out in the editor capture's own 1440 x 880 space and scaled to the frame.
(() => {
  const W = 1440, H = 880;
  const box = document.getElementById("morph-box");
  const stage = document.getElementById("morph-stage");
  const layer = document.getElementById("morph-nodes");
  const svg = document.getElementById("morph-wires");
  const grid = stage.querySelector(".morph-grid");
  const end = stage.querySelector(".morph-end");
  const reel = document.getElementById("reel");
  const sound = document.getElementById("reel-sound");
  const replay = document.getElementById("morph-replay");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ComfyUI's colours for the types a wire carries.
  const TYPE = { MODEL: "#b39ddb", CLIP: "#ffd500", CONDITIONING: "#ffa931", LATENT: "#ff9cf9", VAE: "#ff6e6e", IMAGE: "#64b5f6", MASK: "#81c784", AUDIO: "#64b5f6", VIDEO: "#64b5f6", NOISE: "#b0b0b0", GUIDER: "#66ffff", SAMPLER: "#ecb4b4", SIGMAS: "#cdffcd" };
  // Real ComfyUI core nodes, with H3's own files in their fields.
  const SPECS = [
    ["Load Diffusion Model", [], ["MODEL"], [["unet_name", "minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot"], ["weight_dtype", "default"]]],
    ["Load CLIP", [], ["CLIP"], [["clip_name", "qwen3vl_32b_minimax_h3_nvfp4_awq"]]],
    ["CLIP Text Encode (Prompt)", ["clip"], ["CONDITIONING"], [["text", "@Chef studies the plate. Then her eyes lift"]]],
    ["CLIP Text Encode (Prompt)", ["clip"], ["CONDITIONING"], [["text", "blurry, low quality, text"]]],
    ["Load VAE", [], ["VAE"], [["vae_name", "minimax_h3_video_vae_fp16"]]],
    ["Load VAE", [], ["VAE"], [["vae_name", "minimax_h3_audio_vae_fp32"]]],
    ["LoraLoaderModelOnly", ["model"], ["MODEL"], [["lora_name", "MMH3-Realism-V2"], ["strength_model", "1.20"]]],
    ["ModelSamplingSD3", ["model"], ["MODEL"], [["shift", "12.00"]]],
    ["Load Image", [], ["IMAGE", "MASK"], [["image", "chef.png"]]],
    ["VAE Encode", ["pixels", "vae"], ["LATENT"], []],
    ["Empty Latent Image", [], ["LATENT"], [["width", "1344"], ["height", "768"], ["length", "93"]]],
    ["RandomNoise", [], ["NOISE"], [["noise_seed", "1541804686"]]],
    ["CFGGuider", ["model", "positive", "negative"], ["GUIDER"], [["cfg", "1.0"]]],
    ["KSamplerSelect", [], ["SAMPLER"], [["sampler_name", "euler"]]],
    ["BasicScheduler", ["model"], ["SIGMAS"], [["scheduler", "simple"], ["steps", "6"]]],
    ["SamplerCustomAdvanced", ["noise", "guider", "sampler", "sigmas", "latent_image"], ["LATENT", "LATENT"], []],
    ["Upscale Latent", ["samples"], ["LATENT"], [["upscale_method", "bislerp"], ["width", "1920"], ["height", "1088"]]],
    ["SamplerCustomAdvanced", ["noise", "guider", "sampler", "sigmas", "latent_image"], ["LATENT", "LATENT"], []],
    ["VAE Decode", ["samples", "vae"], ["IMAGE"], []],
    ["VAE Decode Audio", ["samples", "vae"], ["AUDIO"], []],
    ["Create Video", ["images", "audio"], ["VIDEO"], [["fps", "24"]]],
    ["Save Video", ["video"], [], [["filename_prefix", "video/shot"], ["codec", "h264"]]],
    ["Preview Image", ["images"], [], []],
    ["ConditioningZeroOut", ["conditioning"], ["CONDITIONING"], []],
    ["LoraLoaderModelOnly", ["model"], ["MODEL"], [["lora_name", "h3_turbo_lora"], ["strength_model", "1.00"]]],
    ["Load Image", [], ["IMAGE", "MASK"], [["image", "last_frame.png"]]],
  ];
  // Where each node lands: a region of the editor, measured on the capture behind it.
  const TARGETS = [
    [12, 8, 220, 31, "#2a2d33"], [280, 8, 94, 31, "#2d3239"], [1000, 8, 142, 31, "#2d3239"], [1160, 8, 117, 31, "#2d3239"], [1285, 8, 63, 31, "#e0a24a"],
    [12, 102, 239, 66, "#2d3239"], [12, 172, 239, 66, "#2d3239"], [12, 242, 239, 66, "#3a3022"], [12, 312, 239, 66, "#2d3239"],
    [12, 469, 74, 94, "#2d3239"], [285, 81, 754, 427, "#000000"], [400, 535, 520, 32, "#22262c"],
    [1061, 48, 379, 39, "#22262c"], [1071, 99, 317, 30, "#1b1e23"], [1077, 226, 347, 132, "#1b1e23"], [1077, 397, 243, 31, "#1b1e23"],
    [1332, 397, 92, 31, "#1b1e23"], [1077, 469, 79, 36, "#2d3239"], [1077, 492, 168, 32, "#2d3239"], [1256, 492, 168, 32, "#e0a24a"],
    [12, 586, 700, 30, "#22262c"], [104, 692, 263, 114, "#2d3239"], [367, 692, 263, 114, "#2d3239"], [630, 692, 324, 114, "#3a3022"],
    [954, 692, 446, 114, "#2d3239"], [104, 820, 1296, 52, "#1f3a37"],
  ];

  // A seeded shuffle of positions, so the mess is the same mess on every visit.
  let seed = 7;
  const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };

  const nodes = SPECS.map(([title, ins, outs, fields], i) => {
    const rows = Math.max(ins.length, outs.length, 1);
    const w = 190 + Math.round(rnd() * 70);
    const h = 26 + rows * 20 + fields.length * 22 + 10;
    const el = document.createElement("div");
    el.className = "gnode";
    const inner = document.createElement("div");
    inner.className = "gnode-in";
    const head = document.createElement("div");
    head.className = "gnode-head";
    head.textContent = title;
    inner.append(head);
    for (let r = 0; r < rows; r++) {
      const row = document.createElement("div");
      row.className = "gnode-row";
      const a = document.createElement("span"), b = document.createElement("span");
      a.textContent = ins[r] || "";
      b.textContent = (outs[r] || "");
      if (ins[r]) { const dot = document.createElement("i"); dot.className = "l"; dot.style.background = "#9a9a9a"; row.append(dot); }
      if (outs[r]) { const dot = document.createElement("i"); dot.className = "r"; dot.style.background = TYPE[outs[r]] || "#9a9a9a"; row.append(dot); }
      row.append(a, b);
      inner.append(row);
    }
    for (const [k, v] of fields) {
      const f = document.createElement("div");
      f.className = "gnode-field";
      const key = document.createElement("span"), val = document.createElement("b");
      key.textContent = k; val.textContent = v;
      f.append(key, val);
      inner.append(f);
    }
    el.append(inner);
    layer.append(el);
    // Scattered over the whole frame, piled on each other, a little askew.
    const x = 30 + rnd() * (W - w - 60), y = 20 + rnd() * (H - h - 40);
    return { el, inner, ins, outs, x, y, w, h, tilt: (rnd() - 0.5) * 5, z: Math.floor(rnd() * 30) };
  });

  // Each target goes to the node that starts nearest it, so the fold reads as order coming out of the mess.
  const free = TARGETS.map((t, i) => i);
  const centre = (x, y, w, h) => [x + w / 2, y + h / 2];
  const order = nodes.map((n, i) => i).sort((a, b) => nodes[b].w * nodes[b].h - nodes[a].w * nodes[a].h);
  const bigFirst = TARGETS.map((t, i) => i).sort((a, b) => TARGETS[b][2] * TARGETS[b][3] - TARGETS[a][2] * TARGETS[a][3]);
  order.forEach((ni, k) => {
    const n = nodes[ni];
    const [cx, cy] = centre(n.x, n.y, n.w, n.h);
    const pool = k < 4 ? bigFirst.filter((t) => free.includes(t)).slice(0, 6) : free;
    let best = pool[0], bestD = Infinity;
    for (const t of pool) {
      const [tx, ty] = centre(...TARGETS[t].slice(0, 4));
      const d = Math.hypot(tx - cx, ty - cy);
      if (d < bestD) { bestD = d; best = t; }
    }
    free.splice(free.indexOf(best), 1);
    n.t = TARGETS[best];
    n.delay = 0.006 * k + rnd() * 0.06;   // at most about a fifth of the move apart, so every node lands
  });

  // Wires: from each output to an input of some other node, in the output's colour.
  const wires = [];
  nodes.forEach((n, i) => {
    n.outs.forEach((type, r) => {
      const targets = nodes.filter((m) => m !== n && m.ins.length);
      const links = 1 + Math.floor(rnd() * 2);
      for (let k = 0; k < links; k++) {
        const m = targets[Math.floor(rnd() * targets.length)];
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("stroke", TYPE[type] || "#9a9a9a");
        svg.append(path);
        wires.push({ a: n, ar: r, b: m, br: Math.floor(rnd() * m.ins.length), path });
      }
    });
  });

  const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
  const clamp01 = (x) => Math.max(0, Math.min(1, x));
  const span = (t, a, b) => clamp01((t - a) / (b - a));
  const mix = (a, b, k) => a + (b - a) * k;

  // One frame of the fold, from t = 0 (the graph) to t = 1 (the editor).
  function draw(t) {
    const hold = span(t, 0, 0.3);                  // the graph drifts a little while it is read
    const pull = span(t, 0.28, 0.46);              // the wires go as the nodes start to move
    const zoom = mix(1.05, 1, ease(hold));
    layer.style.transform = svg.style.transform = "scale(" + zoom + ")";
    layer.style.transformOrigin = svg.style.transformOrigin = "50% 50%";
    for (const n of nodes) {
      const k = ease(span(t, 0.3 + n.delay, 0.6 + n.delay));
      const [tx, ty, tw, th, colour] = n.t;
      n.cx = mix(n.x, tx, k); n.cy = mix(n.y, ty, k); n.cw = mix(n.w, tw, k); n.ch = mix(n.h, th, k);
      const s = n.el.style;
      s.left = n.cx + "px"; s.top = n.cy + "px"; s.width = n.cw + "px"; s.height = n.ch + "px";
      s.transform = "rotate(" + n.tilt * (1 - k) + "deg)";
      s.zIndex = k > 0.02 ? 100 + n.z : n.z;
      s.background = k > 0.8 ? colour : "";
      s.borderRadius = mix(8, 6, k) + "px";
      s.opacity = 1 - span(t, 0.8, 0.94);
      n.inner.style.opacity = 1 - span(k, 0.1, 0.55);
    }
    for (const w of wires) {
      const ax = w.a.cx + w.a.cw, ay = w.a.cy + 26 + 10 + w.ar * 20;
      const bx = w.b.cx, by = w.b.cy + 26 + 10 + w.br * 20;
      const dx = Math.max(60, Math.abs(bx - ax) * 0.5);
      w.path.setAttribute("d", "M" + ax + " " + ay + " C" + (ax + dx) + " " + ay + " " + (bx - dx) + " " + by + " " + bx + " " + by);
    }
    svg.style.opacity = 1 - pull;
    grid.style.opacity = 1 - span(t, 0.5, 0.78);
    end.style.opacity = span(t, 0.72, 0.9);
    reel.style.opacity = span(t, 0.82, 0.96);
  }

  const DURATION = 4200;
  let started = 0, frame = 0;
  function run(now) {
    if (!started) started = now;
    const t = Math.min(1, (now - started) / DURATION);
    draw(t);
    if (t < 1) frame = requestAnimationFrame(run);
    else { started = 0; reel.play().catch(() => {}); }
  }
  function play() {
    cancelAnimationFrame(frame);
    started = 0;
    reel.pause();
    reel.currentTime = 0;
    draw(0);
    frame = requestAnimationFrame((now) => { started = now + 700; frame = requestAnimationFrame(run); });
  }

  function fit() { stage.style.transform = "scale(" + box.clientWidth / W + ")"; }
  fit();
  new ResizeObserver(fit).observe(box);

  if (reduceMotion) { draw(1); }
  else {
    draw(0);
    let seen = false;
    new IntersectionObserver((entries) => {
      for (const entry of entries) if (entry.isIntersecting && !seen) { seen = true; play(); }
    }, { threshold: 0.3 }).observe(box);
  }
  replay.addEventListener("click", () => { if (reduceMotion) { draw(1); return; } play(); });

  sound.addEventListener("click", () => {
    reel.muted = !reel.muted;
    sound.setAttribute("aria-pressed", String(!reel.muted));
    if (!reel.muted && reel.paused && reel.style.opacity === "1") reel.play().catch(() => {});
  });
  // Off screen the cut pauses; back on screen it carries on once the fold has finished.
  new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) reel.pause();
      else if (reel.style.opacity === "1" && (!reduceMotion || !reel.muted)) reel.play().catch(() => {});
    }
  }, { threshold: 0.35 }).observe(box);
})();

// The showcase: choosing a render swaps the clip and the words that made it, together.
(() => {
  const SHOTS = {
    "harbor": {
      label: "Midday sun on a fisherman's face", seed: 1248160305, seconds: "3.8 s",
      prompt: "A weathered fisherman in his sixties stands on a harbor dock at midday. Hard sun from straight above lights his sunburnt face, with deep lines, grey stubble, and sweat and salt drying on his skin. He squints toward the sea, then turns his head slowly toward the camera. Fishing boats sit out of focus behind him. Camera: Close-up, handheld, slight drift. Sound: Gulls, water lapping against the dock, a distant boat engine.",
    },
    "golden-hour": {
      label: "A woman in a wheat field at golden hour", seed: 155121007, seconds: "3.8 s",
      prompt: "A young woman with red hair and freckles walks through a wheat field at golden hour. The low sun is behind her and lights the fine hair along her cheek and the edges of her face. She looks back over her shoulder toward the camera and smiles. Camera: Medium close-up, tracking beside her. Sound: Wind through the wheat, birdsong.",
    },
    "neon-rain": {
      label: "A woman in neon light and rain", seed: 308467067, seconds: "3.8 s",
      prompt: "A woman in her thirties stands on a city street at night in the rain. A pink neon sign on one side and a cyan one on the other light her face in two colours, and raindrops run down her cheeks and catch both colours. She blinks the rain away and looks up into it. Camera: Close-up, slow push in. Sound: Steady rain, distant traffic, the hum of neon signs.",
    },
    "candlelight": {
      label: "An old woman by one candle, speaking", seed: 177889935, seconds: "5.2 s",
      prompt: "Tight close-up of an old woman's face in her eighties, filling the frame from forehead to chin. A single candle burns just beside her face, its warm light falling across the deep wrinkles of one cheek and leaving the other side in shadow. Her silver hair is pinned back. She looks up from a letter she holds below the frame and speaks softly, with a small, tired smile. Camera: Tight close-up, locked off, shallow focus. Speaker: The old woman, soft and warm. Dialogue: \"He wrote that he would be home before the snow.\" Sound: A candle flame fluttering, a clock ticking in the room.",
    },
    "boxer": {
      label: "A boxer between rounds under a gym lamp", seed: 606441076, seconds: "3.8 s",
      prompt: "Close-up of a boxer in his late twenties sitting in the corner of a dim gym between rounds, his face and shoulders filling the frame. A single hanging lamp above lights the sweat on his forehead, cheekbones and shoulders. He breathes hard, then lifts his head and looks toward the camera, jaw set. The background is a dark, blurred gym wall. Camera: Close-up, handheld. Sound: Heavy breathing, a muffled crowd, a bell far off.",
    },
  };
  const FACTS = (shot) => [
    ["Model", "HyperFlow H3 Flash"],
    ["LoRA", "Realism Slider V2, strength 1.2"],
    ["Final", "1920 × 1088, " + shot.seconds + ", 24 fps"],
    ["Seed", String(shot.seed)],
  ];
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const video = document.getElementById("show-video");
  const stage = video.parentElement;
  const prompt = document.getElementById("show-prompt");
  const facts = document.getElementById("show-facts");
  const sound = document.getElementById("show-sound");
  const picks = [...document.querySelectorAll(".pick")];

  function words(text) {
    prompt.textContent = "";
    prompt.classList.remove("in");
    text.split(/(\s+)/).forEach((part, i) => {
      if (/^\s+$/.test(part)) { prompt.append(part); return; }
      const word = document.createElement("span");
      word.className = "w";
      word.textContent = part;
      word.style.transitionDelay = Math.min(i * 6, 700) + "ms";
      prompt.append(word);
    });
    requestAnimationFrame(() => requestAnimationFrame(() => prompt.classList.add("in")));
  }

  function choose(pick, focus) {
    const id = pick.dataset.shot;
    const shot = SHOTS[id];
    picks.forEach((each) => {
      const on = each === pick;
      each.setAttribute("aria-selected", String(on));
      each.tabIndex = on ? 0 : -1;
    });
    if (focus) pick.focus();
    facts.textContent = "";
    for (const [name, value] of FACTS(shot)) {
      const dt = document.createElement("dt");
      dt.textContent = name;
      const dd = document.createElement("dd");
      dd.textContent = value;
      facts.append(dt, dd);
    }
    words(shot.prompt);
    const swap = () => {
      video.src = "media/realism/" + id + ".mp4";
      video.poster = "media/realism/" + id + ".jpg";
      video.setAttribute("aria-label", shot.label);
      video.play().catch(() => {});
      stage.classList.remove("swapping");
    };
    if (reduceMotion) swap();
    else { stage.classList.add("swapping"); setTimeout(swap, 220); }
  }

  picks.forEach((pick, i) => {
    pick.addEventListener("click", () => { if (pick.getAttribute("aria-selected") !== "true") choose(pick, false); });
    pick.addEventListener("keydown", (event) => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (step) { event.preventDefault(); choose(picks[(i + step + picks.length) % picks.length], true); }
    });
  });
  sound.addEventListener("click", () => {
    video.muted = !video.muted;
    sound.setAttribute("aria-pressed", String(!video.muted));
    if (!video.muted && video.paused) video.play().catch(() => {});
  });

  // First words without the reveal, so they are there before the visitor scrolls to them.
  const first = SHOTS.harbor;
  prompt.textContent = first.prompt;
  for (const [name, value] of FACTS(first)) {
    const dt = document.createElement("dt");
    dt.textContent = name;
    const dd = document.createElement("dd");
    dd.textContent = value;
    facts.append(dt, dd);
  }
  whileVisible(video, () => !reduceMotion || !video.muted);
})();

// The launch film starts from the page's own button, with sound; the browser's controls come once it plays.
(() => {
  const film = document.getElementById("film-video");
  const button = document.getElementById("film-play");
  button.addEventListener("click", () => {
    film.controls = true;
    film.parentElement.classList.add("playing");
    film.play().catch(() => {});
    film.focus();
  });
})();

// The cut drives the timeline and the shot pane, the way the editor's playhead does.
(() => {
  const FPS = 24;
  const TOTAL = 360;
  const PROJECT = "The_pass";

  // The four shots of "The pass", with the fields they were rendered from. `cast` is who the shot keeps the same.
  const CLIPS = [
    {
      id: "c822cf892", name: "Wide", from: 0, to: 73, seconds: 3.04, strip: "media/wide.jpg", cast: "Chef",
      action: "@Chef wipes the rim of a white plate at the pass. Behind her, a flame flares up and dies. The pass is long brushed steel under a row of glowing heat lamps; the flame rises tall and orange from a pan on the range, lights the whole kitchen for a moment, then is gone. Steam drifts through the lamp light.",
      camera: "Wide shot at eye level from the dining side of the pass, slow push in",
      speaker: "",
      dialogue: "",
      sound: "The whoosh and roar of a gas flame flaring up, a pan hissing, the low hum of extractor fans. No music, no voices.",
    },
    {
      id: "cc2eecf3a", name: "Detail", from: 73, to: 146, seconds: 3.04, strip: "media/detail.jpg", cast: "",
      action: "Tweezers lower one green leaf onto a seared scallop. The scallop sits on a swipe of golden puree on a white plate under a warm heat lamp; a steady hand holds the steel tweezers, places the small herb leaf and lets go. Steam curls up through the light. The dark kitchen behind is empty and far out of focus.",
      camera: "Macro close-up, locked off, very shallow depth of field",
      speaker: "",
      dialogue: "",
      sound: "A soft metallic clink of tweezers touching porcelain, a pan sizzling quietly in the background, the low hum of extractor fans. No music, no voices.",
    },
    {
      id: "caa4b5e11", name: "Face", from: 146, to: 236, seconds: 3.75, strip: "media/face.jpg", cast: "Chef",
      action: "@Chef studies the plate. Then her eyes lift, and she nods. Only her face is in the frame, from forehead to chin, warm light falling on it from above, the dark kitchen a soft blur behind her. Her eyes are lowered, completely focused; then she raises her eyes to look straight ahead and gives one small, satisfied nod.",
      camera: "Extreme close-up of her face, forehead to chin filling the whole frame, locked off, 85mm lens, very shallow depth of field",
      speaker: "",
      dialogue: "",
      sound: "The low hum of extractor fans and a distant pan hissing. No music, no voices.",
    },
    {
      id: "c23560c00", name: "Bell", from: 236, to: 360, seconds: 5.17, strip: "media/bell.jpg", cast: "Chef",
      action: "@Chef slides the plate across the pass and rings the bell. A small dome-shaped brass call bell sits on the brushed steel in the foreground, in sharp focus. From the right, her hand pushes a white plate with a seared scallop on golden puree, topped with one green herb leaf, across the steel toward the camera until it stops beside the bell. After a beat her palm comes down and taps the button on top of the bell once.",
      camera: "Low angle along the surface of the steel pass under the heat lamps, the bell and the plate sharp in the foreground, the kitchen dark and soft behind",
      speaker: "",
      dialogue: "",
      sound: "Porcelain sliding across steel, a short silence, then one clear bright ding of a brass service bell that rings out over the low hum of extractor fans. No music, no voices.",
    },
  ];

  // Loudest sample in each of 240 slices of the cut's sound, 0 to 100.
  const PEAKS = [17,14,10,13,14,13,12,11,10,9,21,46,66,91,90,48,52,38,29,24,18,18,23,64,19,15,14,16,11,13,13,15,9,11,12,10,13,13,16,19,12,12,11,14,13,10,11,12,12,4,5,5,5,5,7,5,7,9,4,7,7,7,5,6,6,10,5,5,5,6,7,7,6,7,7,7,6,9,6,6,8,5,5,6,8,6,5,7,7,7,6,6,8,6,7,5,6,13,11,12,9,10,12,12,12,13,14,17,12,13,12,12,10,13,13,14,14,16,16,14,14,12,15,17,17,16,14,16,17,20,14,14,15,14,19,18,18,14,17,18,15,12,15,14,13,15,12,16,13,15,17,14,14,11,11,17,22,2,3,3,3,4,2,3,3,3,2,2,3,5,7,6,5,6,6,6,6,6,5,7,6,7,8,8,6,6,5,7,7,9,7,6,7,8,7,7,6,5,6,5,7,6,6,6,7,7,6,7,7,6,6,10,12,6,4,4,3,3,3,3,3,3,3,3,3,3,3,3,63,100,72,56,36,33,38,22,22,21,19,20];

  const $ = (id) => document.getElementById(id);
  const video = $("cut");
  const track = $("track");
  const playhead = $("playhead");
  const clock = $("clock");
  const announce = $("announce");
  const pad = (n) => String(n).padStart(2, "0");
  const timecode = (frame) => "00:" + pad(Math.floor(frame / FPS)) + ":" + pad(frame % FPS);
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* Timeline */

  const ruler = $("ruler");
  for (let second = 0; second * FPS < TOTAL; second++) {
    const tick = document.createElement("span");
    tick.className = "tick";
    tick.style.left = (second * FPS / TOTAL) * 100 + "%";
    tick.textContent = "0:" + pad(second);
    ruler.append(tick);
  }

  const blocks = CLIPS.map((clip) => {
    const block = document.createElement("div");
    block.className = "block";
    block.style.left = (clip.from / TOTAL) * 100 + "%";
    block.style.width = ((clip.to - clip.from) / TOTAL) * 100 + "%";
    block.style.backgroundImage = "url(" + clip.strip + ")";
    const label = document.createElement("span");
    label.textContent = clip.name;
    block.append(label);
    $("picture").append(block);
    return block;
  });

  const wave = $("wave");
  const svgNS = "http://www.w3.org/2000/svg";
  PEAKS.forEach((peak, i) => {
    const height = Math.max(1.2, Math.sqrt(peak / 100) * 40);
    const bar = document.createElementNS(svgNS, "rect");
    bar.setAttribute("x", i + 0.15);
    bar.setAttribute("width", 0.7);
    bar.setAttribute("y", (40 - height) / 2);
    bar.setAttribute("height", height);
    wave.append(bar);
  });

  /* The shot pane */

  const fieldIds = ["action", "camera", "speaker", "dialogue", "sound"];
  const empty = { speaker: "No one", dialogue: "Nothing", sound: "Nothing" };

  function opFor(clip) {
    const fields = { name: clip.name };
    if (clip.cast) fields.cast = ["$" + clip.cast.toLowerCase()];
    fields.action = clip.action;
    fields.camera = clip.camera;
    if (clip.speaker) fields.speaker = clip.speaker;
    if (clip.dialogue) fields.dialogue = clip.dialogue;
    fields.sound = clip.sound;
    fields.seconds = clip.seconds;
    return fields;
  }

  // One block per line, so a line that wraps hangs under its own text instead of the prompt.
  function line(code, parts) {
    const row = document.createElement("span");
    row.className = "line";
    for (const [cls, text] of parts) {
      const span = document.createElement("span");
      if (cls) span.className = cls;
      span.textContent = text;
      row.append(span);
    }
    code.append(row);
  }

  function renderAgent(clip) {
    const code = $("term");
    code.textContent = "";
    line(code, [["p", "$ "], ["", "python h3.py ops " + PROJECT + " shots.json"]]);
    line(code, [["p", "$ "], ["", "python h3.py render " + PROJECT + " "], ["nowrap", clip.id + " --final"]]);
    line(code, [["p", "$ "], ["", "python h3.py export " + PROJECT]]);
    const file = $("file");
    file.textContent = "";
    line(file, [["", "{\"op\": \"add_clip\", \"fields\": {"]]);
    const fields = Object.entries(opFor(clip));
    fields.forEach(([key, value], i) => {
      const last = i === fields.length - 1;
      const shown = typeof value === "number" ? [["n", String(value)]] : [["s", JSON.stringify(value).replace(/","/g, '", "')]];
      line(file, [["", "  "], ["k", JSON.stringify(key) + ": "], ...shown, ["", last ? "}}" : ","]]);
    });
    $("file-name").textContent = clip.name;
  }

  function renderHand(clip) {
    $("shot-name").textContent = clip.name;
    $("shot-meta").textContent = "Final, " + (Math.round(clip.seconds * 10) / 10) + " s";
    for (const id of fieldIds) {
      const cell = $("f-" + id);
      cell.textContent = clip[id] || empty[id];
      cell.classList.toggle("empty", !clip[id]);
      if (id === "action") cell.scrollTop = 0;
    }
    // Who the shot keeps the same: the cast member's picture and name, as in the editor.
    $("cast-who").hidden = !clip.cast;
    $("cast-none").hidden = !!clip.cast;
    // A shot without a speaker shows no empty speech fields.
    $("speech").hidden = !clip.speaker && !clip.dialogue;
  }

  let shown = -1;
  function showClip(index) {
    if (index === shown) return;
    shown = index;
    const clip = CLIPS[index];
    renderHand(clip);
    renderAgent(clip);
    blocks.forEach((block, i) => block.classList.toggle("on", i === index));
    if (!reduceMotion) {
      for (const pane of document.querySelectorAll(".pane")) {
        pane.classList.remove("swap");
        void pane.offsetWidth;
        pane.classList.add("swap");
      }
    }
  }

  /* Playhead */

  const frameNow = () => Math.max(0, Math.min(TOTAL - 1, Math.floor(video.currentTime * FPS + 0.001)));
  const clipAt = (frame) => Math.max(0, CLIPS.findIndex((clip) => frame >= clip.from && frame < clip.to));

  function sync() { show(frameNow()); }
  function show(frame) {
    const index = clipAt(frame);
    playhead.style.left = (frame / TOTAL) * 100 + "%";
    clock.textContent = timecode(frame);
    track.setAttribute("aria-valuenow", frame);
    track.setAttribute("aria-valuetext", timecode(frame) + ", " + CLIPS[index].name);
    showClip(index);
  }

  let ticking = false;
  function tick() {
    sync();
    if (!video.paused) requestAnimationFrame(tick);
    else ticking = false;
  }
  function startTicking() {
    if (!ticking) { ticking = true; requestAnimationFrame(tick); }
  }

  function seek(frame) {
    const clamped = Math.max(0, Math.min(TOTAL - 1, Math.round(frame)));
    video.currentTime = (clamped + 0.5) / FPS;
    sync();
  }

  function frameFromPointer(event) {
    const box = track.getBoundingClientRect();
    if (!box.width) return frameNow();
    return ((event.clientX - box.left) / box.width) * TOTAL;
  }
  let dragging = false;
  // A touch that turns into a vertical scroll is cancelled by the browser, so touches seek on release or drag.
  track.addEventListener("pointerdown", (event) => {
    dragging = true;
    track.setPointerCapture(event.pointerId);
    if (event.pointerType !== "touch") seek(frameFromPointer(event));
  });
  track.addEventListener("pointermove", (event) => { if (dragging) seek(frameFromPointer(event)); });
  track.addEventListener("pointerup", (event) => {
    if (dragging && event.pointerType === "touch") seek(frameFromPointer(event));
    dragging = false;
  });
  track.addEventListener("pointercancel", () => { dragging = false; });
  track.addEventListener("keydown", (event) => {
    const step = event.shiftKey ? FPS : 1;
    const here = frameNow();
    const moves = {
      ArrowLeft: here - step, ArrowRight: here + step, ArrowDown: here - step, ArrowUp: here + step,
      Home: 0, End: TOTAL - 1, PageUp: here + FPS, PageDown: here - FPS,
    };
    if (event.key in moves) { event.preventDefault(); seek(moves[event.key]); }
    else if (event.key === " ") { event.preventDefault(); togglePlay(); }
  });

  /* Transport */

  const room = document.querySelector(".stage");
  const play = $("play");
  function showPlaying() {
    room.classList.toggle("paused", video.paused);
    play.setAttribute("aria-label", video.paused ? "Play" : "Pause");
  }
  function togglePlay() {
    if (video.paused) video.play().catch(() => {});
    else video.pause();
  }
  play.addEventListener("click", togglePlay);
  video.addEventListener("click", togglePlay);
  $("start").addEventListener("click", () => seek(0));
  $("prev").addEventListener("click", () => seek(frameNow() - 1));
  $("next").addEventListener("click", () => seek(frameNow() + 1));
  video.addEventListener("play", () => { showPlaying(); startTicking(); });
  video.addEventListener("pause", () => { showPlaying(); sync(); });
  video.addEventListener("seeked", sync);
  video.addEventListener("timeupdate", sync); // keeps up where animation frames are paused

  const sound = $("sound");
  sound.addEventListener("click", () => {
    video.muted = !video.muted;
    sound.setAttribute("aria-pressed", String(!video.muted));
    if (!video.muted && video.paused) video.play().catch(() => {});
  });

  // The cut opens on the flame at its tallest, the poster's frame, so poster, playhead and panes agree
  // before the video loads and whenever the browser refuses to play it on its own. It plays only while
  // the editor is on screen, and stays paused once the visitor pauses it.
  const START = 36;
  let heldByVisitor = false;
  play.addEventListener("click", () => { heldByVisitor = video.paused; });
  function begin() { video.currentTime = (START + 0.5) / FPS; }
  if (video.readyState >= 1) begin();
  else video.addEventListener("loadedmetadata", begin, { once: true });
  showPlaying();
  show(START);
  whileVisible(video, () => !reduceMotion && !heldByVisitor);

  /* The two ways to make the shot */

  const tabs = [$("tab-hand"), $("tab-agent")];
  function choose(tab, focus) {
    for (const each of tabs) {
      const on = each === tab;
      each.setAttribute("aria-selected", String(on));
      each.tabIndex = on ? 0 : -1;
      $(each.getAttribute("aria-controls")).hidden = !on;
    }
    if (focus) tab.focus();
  }
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => choose(tab, false));
    tab.addEventListener("keydown", (event) => {
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        event.preventDefault();
        choose(tabs[(i + 1) % tabs.length], true);
      }
    });
  });

  /* Interactive Auto-Playing Idle Battler Showcase (Aurelia vs. Void Sentinel + MiniMax H3 VFX) */

  const sprCanvas = $("spr-canvas");
  if (sprCanvas) {
    const sctx = sprCanvas.getContext("2d");
    const sprReadout = $("spr-readout");
    const sprAutoBtn = $("spr-auto");
    const sprBackdropBtn = $("spr-backdrop");
    const sprPivotBtn = $("spr-pivot");
    const sprPylonBtn = $("spr-pylon");
    const actBtns = [...document.querySelectorAll("[data-spr-act]")];
    const litPicks = [...document.querySelectorAll("[data-spr-lit]")];

    const BACKDROPS = ["dark", "checker", "light"];
    const assets = {
      aurelia: { manifest: null, img: null },
      sentinel: { manifest: null, img: null },
      vfx: { manifest: null, img: null },
      pylonDormant: new Image(),
      pylonAwakened: new Image(),
    };
    assets.pylonDormant.src = "media/sprites/pylon-dormant.webp";
    assets.pylonAwakened.src = "media/sprites/pylon-awakened.webp";

    const loadAtlas = (slot, jsonPath) => {
      fetch(jsonPath)
        .then((r) => r.json())
        .then((data) => {
          assets[slot].manifest = data;
          const img = new Image();
          img.onload = () => {
            assets[slot].img = img;
            drawStage();
            ensureTicking();
          };
          img.src = "media/sprites/" + data.image;
        })
        .catch(() => {});
    };
    loadAtlas("aurelia", "media/sprites/aurelia-atlas.json");
    loadAtlas("sentinel", "media/sprites/sentinel-atlas.json");
    loadAtlas("vfx", "media/sprites/vfx-atlas.json");

    let autoBattle = !reduceMotion;
    let backdropIdx = 0;
    let showPivot = false;
    let pylonManual = null;
    let pylonFlash = 0;

    let wave = 1;
    let heroHp = 1000, heroMax = 1000, heroVisHp = 1000;
    let enemyHp = 1000, enemyMax = 1000, enemyVisHp = 1000;
    let novaCharge = 0;
    let turnStep = 0;
    let aiCooldown = 0.7;
    let respawnTimer = 0;
    let enemyFade = 1;
    let heroFlash = 0;
    let shake = 0;
    let statusBanner = "";
    let statusTimer = 0;

    const hero = { anim: "idle", t: 0, fired: false, lunge: 0 };
    const enemy = { anim: "idle", t: 0, fired: false, lunge: 0, alive: true };
    const vfxList = [];
    const projectiles = [];
    const popups = [];
    const embers = Array.from({ length: 22 }, (_, i) => ({
      x: (i * 0.047 + 0.02) % 1,
      y: (i * 0.13 + 0.1) % 1,
      r: 1.1 + (i % 3) * 0.7,
      vy: 0.025 + (i % 4) * 0.012,
      vx: ((i % 5) - 2) * 0.006,
      hue: i % 3 === 0 ? "violet" : "gold",
    }));

    let clock = 0;
    let lastTs = 0;
    let visible = false;
    let rafId = 0;

    const isPylonAwakened = () => (pylonManual !== null ? pylonManual : pylonFlash > 0);

    function syncControls() {
      if (sprAutoBtn) sprAutoBtn.setAttribute("aria-pressed", String(autoBattle));
      sprBackdropBtn.textContent = "Backdrop: " + BACKDROPS[backdropIdx][0].toUpperCase() + BACKDROPS[backdropIdx].slice(1);
      sprPivotBtn.setAttribute("aria-pressed", String(showPivot));
      const lit = isPylonAwakened();
      if (sprPylonBtn) {
        sprPylonBtn.setAttribute("aria-pressed", String(lit));
        sprPylonBtn.textContent = lit ? "Pylon: Awakened" : "Pylon: Dormant";
      }
      for (const b of litPicks) b.setAttribute("aria-pressed", String(String(lit) === b.dataset.sprLit));
      if (sprReadout) {
        const moveLabel =
          hero.anim === "nova" ? "Astral Nova" :
          hero.anim === "cast" ? "Solar Cast" :
          enemy.anim === "attack" ? "Void Slash" :
          autoBattle ? "Auto Battle" : "Paused";
        sprReadout.textContent = "Wave " + wave + " · " + moveLabel;
      }
    }

    function spawnVfx(key, nx, ny, scale, delay) {
      vfxList.push({ key, nx, ny, scale: scale || 1, t: -(delay || 0) });
    }

    function spawnPopup(text, nx, ny, color, big) {
      popups.push({ text, nx, ny, color, big: !!big, t: 0, life: big ? 1.25 : 0.95 });
    }

    function triggerHeroCast() {
      if (respawnTimer > 0) return;
      hero.anim = "cast";
      hero.t = 0;
      hero.fired = false;
      syncControls();
    }

    function triggerHeroNova() {
      if (respawnTimer > 0) return;
      hero.anim = "nova";
      hero.t = 0;
      hero.fired = false;
      pylonFlash = 2.2;
      novaCharge = 0;
      syncControls();
    }

    function triggerEnemyAttack() {
      if (!enemy.alive || respawnTimer > 0) return;
      enemy.anim = "attack";
      enemy.t = 0;
      enemy.fired = false;
      syncControls();
    }

    function damageEnemy(amount, crit) {
      if (!enemy.alive) return;
      enemyHp = Math.max(0, enemyHp - amount);
      enemy.anim = "hit";
      enemy.t = 0;
      enemy.fired = false;
      shake = crit ? 9 : 4.5;
      spawnVfx("solar_burst", 0.74, 0.56, crit ? 1.48 : 1.08, 0);
      if (crit) spawnVfx("solar_burst", 0.71, 0.48, 1.15, 0.14);
      spawnPopup(crit ? "-" + amount + " CRIT!" : "-" + amount, 0.73, 0.36, crit ? "#ffd37a" : "#f4b95e", crit);
      if (enemyHp <= 0) {
        enemy.alive = false;
        respawnTimer = 1.65;
        pylonFlash = 1.9;
        statusBanner = "WAVE " + wave + " CLEARED";
        statusTimer = 1.5;
        heroHp = Math.min(heroMax, heroHp + 180);
        syncControls();
      }
    }

    function damageHero(amount) {
      heroHp = Math.max(120, heroHp - amount);
      heroFlash = 0.4;
      shake = 3.5;
      spawnVfx("void_slash", 0.33, 0.57, 1.12, 0);
      spawnPopup("-" + amount, 0.28, 0.39, "#d8a4ff", false);
    }

    function animDuration(slot, key) {
      const a = assets[slot].manifest?.animations?.[key];
      if (!a || !a.frames?.length) return 0.9;
      return a.frames.length / (a.fps || 12);
    }

    function drawAnimSprite(slot, key, timeSec, cx, groundY, drawSize, alpha, showBox) {
      const m = assets[slot].manifest;
      const img = assets[slot].img;
      if (!m || !img || !img.naturalWidth) return;
      const a = m.animations?.[key] || m.animations?.idle;
      if (!a || !a.frames?.length) return;
      const rawIdx = Math.floor(Math.max(0, timeSec) * (a.fps || 12));
      const idx = a.loop ? rawIdx % a.frames.length : Math.min(a.frames.length - 1, rawIdx);
      const f = a.frames[idx];
      const scale = drawSize / f.h;
      const dw = Math.round(f.w * scale);
      const dh = Math.round(f.h * scale);
      const pivX = Math.round((a.pivot?.[0] ?? f.w / 2) * scale);
      const pivY = Math.round((a.pivot?.[1] ?? f.h * 0.92) * scale);
      const dx = Math.round(cx - pivX);
      const dy = Math.round(groundY - pivY);

      sctx.save();
      sctx.globalAlpha = alpha;
      sctx.drawImage(img, f.x, f.y, f.w, f.h, dx, dy, dw, dh);
      sctx.restore();

      if (showBox) {
        sctx.save();
        sctx.strokeStyle = "rgba(224, 162, 74, 0.55)";
        sctx.lineWidth = 1;
        sctx.setLineDash([4, 3]);
        sctx.strokeRect(dx + 0.5, dy + 0.5, dw - 1, dh - 1);
        sctx.setLineDash([]);
        sctx.strokeStyle = "#e0a24a";
        sctx.beginPath();
        sctx.moveTo(cx - 10, groundY + 0.5);
        sctx.lineTo(cx + 10, groundY + 0.5);
        sctx.moveTo(cx + 0.5, groundY - 10);
        sctx.lineTo(cx + 0.5, groundY + 10);
        sctx.stroke();
        sctx.restore();
      }
    }

    function drawBar(x, y, bw, bh, ratio, trailRatio, fillCss, alignRight, label, sub, pips, mode) {
      const light = mode === "light";
      sctx.save();
      sctx.font = "600 11px 'Geist', system-ui, sans-serif";
      sctx.textBaseline = "bottom";
      sctx.fillStyle = light ? "rgba(18,18,22,0.88)" : "rgba(244,244,245,0.92)";
      sctx.textAlign = alignRight ? "right" : "left";
      sctx.fillText(label, alignRight ? x + bw : x, y - 5);
      if (sub) {
        sctx.font = "500 10.5px 'Geist Mono', ui-monospace, monospace";
        sctx.fillStyle = light ? "rgba(18,18,22,0.55)" : "rgba(161,161,170,0.9)";
        sctx.textAlign = alignRight ? "left" : "right";
        sctx.fillText(sub, alignRight ? x : x + bw, y - 5);
      }
      sctx.fillStyle = light ? "rgba(18,18,22,0.14)" : "rgba(9,9,11,0.72)";
      sctx.beginPath();
      sctx.roundRect(x, y, bw, bh, 4);
      sctx.fill();
      sctx.strokeStyle = light ? "rgba(18,18,22,0.16)" : "rgba(255,255,255,0.12)";
      sctx.lineWidth = 1;
      sctx.stroke();

      const tw = Math.max(0, Math.min(1, trailRatio)) * (bw - 2);
      const fw = Math.max(0, Math.min(1, ratio)) * (bw - 2);
      if (tw > fw) {
        sctx.fillStyle = "rgba(255, 255, 255, 0.45)";
        sctx.beginPath();
        sctx.roundRect(alignRight ? x + bw - 1 - tw : x + 1, y + 1, tw, bh - 2, 3);
        sctx.fill();
      }
      if (fw > 0.5) {
        sctx.fillStyle = fillCss;
        sctx.beginPath();
        sctx.roundRect(alignRight ? x + bw - 1 - fw : x + 1, y + 1, fw, bh - 2, 3);
        sctx.fill();
      }
      if (pips !== null) {
        const pipW = 14, pipH = 3.5, gap = 4;
        for (let i = 0; i < 3; i++) {
          const px = x + i * (pipW + gap);
          const py = y + bh + 6;
          sctx.fillStyle = i < pips ? "#f2b85c" : (light ? "rgba(18,18,22,0.14)" : "rgba(255,255,255,0.14)");
          sctx.beginPath();
          sctx.roundRect(px, py, pipW, pipH, 2);
          sctx.fill();
        }
      }
      sctx.restore();
    }

    function drawStage() {
      const w = Math.max(1, sprCanvas.clientWidth);
      const h = Math.max(1, sprCanvas.clientHeight);
      if (sprCanvas.width !== w || sprCanvas.height !== h) {
        sprCanvas.width = w;
        sprCanvas.height = h;
      }
      const mode = BACKDROPS[backdropIdx];
      const lit = isPylonAwakened();

      sctx.save();
      if (shake > 0.2) {
        sctx.translate((Math.random() - 0.5) * shake, (Math.random() - 0.5) * shake * 0.7);
      }

      // 1. Backdrop
      if (mode === "light") {
        sctx.fillStyle = "#eceae3";
        sctx.fillRect(0, 0, w, h);
      } else if (mode === "checker") {
        const sq = 16;
        sctx.fillStyle = "#3a4049";
        sctx.fillRect(0, 0, w, h);
        sctx.fillStyle = "#262a31";
        for (let yy = 0; yy < h; yy += sq) {
          for (let xx = ((yy / sq) % 2) * sq; xx < w; xx += sq * 2) sctx.fillRect(xx, yy, sq, sq);
        }
      } else {
        const bg = sctx.createRadialGradient(w * 0.5, h * 0.52, 16, w * 0.5, h * 0.56, Math.max(w, h) * 0.7);
        bg.addColorStop(0, lit ? "#241d17" : "#181722");
        bg.addColorStop(0.55, "#111118");
        bg.addColorStop(1, "#09090d");
        sctx.fillStyle = bg;
        sctx.fillRect(0, 0, w, h);

        // Subtle arena horizon & floor perspective lines
        const floorTop = h * 0.65;
        const floorGrad = sctx.createLinearGradient(0, floorTop, 0, h);
        floorGrad.addColorStop(0, "rgba(255,255,255,0.035)");
        floorGrad.addColorStop(1, "rgba(0,0,0,0.35)");
        sctx.fillStyle = floorGrad;
        sctx.fillRect(0, floorTop, w, h - floorTop);
        sctx.strokeStyle = "rgba(255,255,255,0.06)";
        sctx.lineWidth = 1;
        sctx.beginPath();
        sctx.moveTo(0, floorTop + 0.5);
        sctx.lineTo(w, floorTop + 0.5);
        for (let i = -4; i <= 4; i++) {
          sctx.moveTo(w * 0.5 + i * (w * 0.08), floorTop);
          sctx.lineTo(w * 0.5 + i * (w * 0.22), h);
        }
        sctx.stroke();

        // Ambient golden & violet arena glows
        const gL = sctx.createRadialGradient(w * 0.27, h * 0.62, 6, w * 0.27, h * 0.62, w * 0.24);
        gL.addColorStop(0, "rgba(224, 162, 74, 0.16)");
        gL.addColorStop(1, "rgba(224, 162, 74, 0)");
        sctx.fillStyle = gL;
        sctx.fillRect(0, 0, w, h);

        if (enemyFade > 0.05) {
          const gR = sctx.createRadialGradient(w * 0.74, h * 0.6, 6, w * 0.74, h * 0.6, w * 0.24);
          gR.addColorStop(0, "rgba(155, 89, 255, " + (0.17 * enemyFade).toFixed(3) + ")");
          gR.addColorStop(1, "rgba(155, 89, 255, 0)");
          sctx.fillStyle = gR;
          sctx.fillRect(0, 0, w, h);
        }

        // Ambient floating sparks
        for (const e of embers) {
          sctx.fillStyle = e.hue === "violet" ? "rgba(182, 120, 255, 0.32)" : "rgba(240, 182, 86, 0.34)";
          sctx.beginPath();
          sctx.arc(e.x * w, e.y * h, e.r, 0, Math.PI * 2);
          sctx.fill();
        }
      }

      const groundY = Math.round(h * 0.82);
      const charSize = Math.round(Math.min(w * 0.42, h * 0.72));
      const pylonSize = Math.round(charSize * 0.58);
      const pylonX = Math.round(w * 0.52);
      const pylonY = Math.round(groundY - h * 0.065);

      // 2. Paired 4 MP Arcane Astrolabe Pylon (upstage center)
      if (lit && mode === "dark") {
        const pg = sctx.createRadialGradient(pylonX, pylonY - pylonSize * 0.66, 4, pylonX, pylonY - pylonSize * 0.56, pylonSize * 0.95);
        pg.addColorStop(0, "rgba(255, 200, 92, 0.34)");
        pg.addColorStop(0.55, "rgba(224, 146, 56, 0.10)");
        pg.addColorStop(1, "rgba(224, 146, 56, 0)");
        sctx.fillStyle = pg;
        sctx.fillRect(0, 0, w, h);
      }
      const pImg = lit ? assets.pylonAwakened : assets.pylonDormant;
      if (pImg.naturalWidth) {
        sctx.fillStyle = mode === "light" ? "rgba(0,0,0,0.12)" : "rgba(0,0,0,0.34)";
        sctx.beginPath();
        sctx.ellipse(pylonX, pylonY, pylonSize * 0.24, 6, 0, 0, Math.PI * 2);
        sctx.fill();
        sctx.drawImage(pImg, Math.round(pylonX - pylonSize / 2), Math.round(pylonY - pylonSize * 0.92), pylonSize, pylonSize);
      }

      // 3. Ground contact shadows for both fighters
      const heroX = Math.round(w * 0.27 + hero.lunge * w);
      const enemyX = Math.round(w * 0.74 - enemy.lunge * w);
      const shadowFill = mode === "light" ? "rgba(0,0,0,0.14)" : "rgba(0,0,0,0.42)";
      sctx.fillStyle = shadowFill;
      sctx.beginPath();
      sctx.ellipse(heroX, groundY + 2, charSize * 0.26, 8, 0, 0, Math.PI * 2);
      sctx.fill();
      if (enemyFade > 0.05) {
        sctx.save();
        sctx.globalAlpha = enemyFade;
        sctx.fillStyle = shadowFill;
        sctx.beginPath();
        sctx.ellipse(enemyX, groundY + 2, charSize * 0.28, 9, 0, 0, Math.PI * 2);
        sctx.fill();
        sctx.restore();
      }

      // 4. Fighters (real H3 Editor BiRefNet-cutout WebP atlases)
      drawAnimSprite("aurelia", hero.anim, hero.t, heroX, groundY, charSize, 1, showPivot);
      if (heroFlash > 0) {
        sctx.save();
        sctx.globalCompositeOperation = "lighter";
        const wg = sctx.createRadialGradient(heroX + charSize * 0.06, groundY - charSize * 0.46, 6, heroX, groundY - charSize * 0.46, charSize * 0.44);
        wg.addColorStop(0, "rgba(255, 218, 128, " + (heroFlash * 0.9).toFixed(3) + ")");
        wg.addColorStop(1, "rgba(255, 190, 90, 0)");
        sctx.fillStyle = wg;
        sctx.beginPath();
        sctx.arc(heroX, groundY - charSize * 0.46, charSize * 0.44, 0, Math.PI * 2);
        sctx.fill();
        sctx.restore();
      }
      if (enemyFade > 0.02) {
        drawAnimSprite("sentinel", enemy.anim, enemy.t, enemyX, groundY, charSize, enemyFade, showPivot);
      }

      // 5. Golden solar spell projectiles streaking from Aurelia's staff to Void Sentinel
      for (const p of projectiles) {
        const px = p.x * w;
        const py = p.y * h;
        sctx.save();
        if (mode === "dark") sctx.globalCompositeOperation = "lighter";
        const grad = sctx.createRadialGradient(px, py, 2, px, py, p.r * 2.8);
        grad.addColorStop(0, "#fff8d6");
        grad.addColorStop(0.35, "rgba(255, 192, 74, 0.92)");
        grad.addColorStop(1, "rgba(224, 118, 32, 0)");
        sctx.fillStyle = grad;
        sctx.beginPath();
        sctx.arc(px, py, p.r * 2.8, 0, Math.PI * 2);
        sctx.fill();
        sctx.restore();
      }

      // 6. Real MiniMax H3 additive VFX spritesheets (solar_burst & void_slash)
      const vMan = assets.vfx.manifest;
      const vImg = assets.vfx.img;
      if (vMan && vImg && vImg.naturalWidth) {
        for (const v of vfxList) {
          if (v.t < 0) continue;
          const a = vMan.animations?.[v.key];
          if (!a || !a.frames?.length) continue;
          const idx = Math.floor(v.t * (a.fps || 20));
          if (idx >= a.frames.length) continue;
          const f = a.frames[idx];
          const vs = Math.round(charSize * 0.72 * v.scale);
          const vx = Math.round(v.nx * w - vs / 2);
          const vy = Math.round(v.ny * h - vs / 2);
          sctx.save();
          if (mode === "dark") sctx.globalCompositeOperation = "lighter";
          sctx.drawImage(vImg, f.x, f.y, f.w, f.h, vx, vy, vs, vs);
          sctx.restore();
        }
      }

      // 7. Floating damage numbers & wave banner
      for (const pop of popups) {
        const p = Math.min(1, pop.t / pop.life);
        const alpha = p < 0.7 ? 1 : Math.max(0, 1 - (p - 0.7) / 0.3);
        const px = pop.nx * w;
        const py = pop.ny * h - p * 34;
        sctx.save();
        sctx.globalAlpha = alpha;
        sctx.font = (pop.big ? "700 20px" : "700 15px") + " 'Geist', system-ui, sans-serif";
        sctx.textAlign = "center";
        sctx.textBaseline = "middle";
        sctx.lineWidth = 3.5;
        sctx.strokeStyle = "rgba(9, 9, 11, 0.85)";
        sctx.strokeText(pop.text, px, py);
        sctx.fillStyle = pop.color;
        sctx.fillText(pop.text, px, py);
        sctx.restore();
      }
      if (statusTimer > 0 && statusBanner) {
        const alpha = Math.min(1, statusTimer * 2);
        sctx.save();
        sctx.globalAlpha = alpha;
        sctx.font = "600 12px 'Geist Mono', ui-monospace, monospace";
        sctx.textAlign = "center";
        sctx.textBaseline = "middle";
        sctx.fillStyle = "#f2c272";
        sctx.fillText(statusBanner, w * 0.5, h * 0.24);
        sctx.restore();
      }

      // 8. Minimalist top HP bars
      const barPad = Math.max(12, Math.round(w * 0.03));
      const barW = Math.min(200, Math.round(w * 0.40));
      const compact = w < 460;
      drawBar(barPad, 28, barW, 8, heroHp / heroMax, heroVisHp / heroMax, "#e0a24a", false, "AURELIA", Math.round(heroHp) + (compact ? "" : " HP"), novaCharge, mode);
      drawBar(w - barPad - barW, 28, barW, 8, enemyHp / enemyMax, enemyVisHp / enemyMax, "#a864ff", true, compact ? "SENTINEL" : "VOID SENTINEL", (compact ? "W" : "WAVE ") + wave, null, mode);

      sctx.restore();
    }

    function step(now) {
      if (!visible) { rafId = 0; lastTs = 0; return; }
      const dt = lastTs ? Math.min(0.05, (now - lastTs) / 1000) : 0;
      lastTs = now;

      if (!reduceMotion) {
        clock += dt;
        hero.t += dt;
        enemy.t += dt;
        shake = Math.max(0, shake - dt * 26);
        heroFlash = Math.max(0, heroFlash - dt);
        statusTimer = Math.max(0, statusTimer - dt);
        if (pylonFlash > 0) {
          const was = pylonFlash > 0;
          pylonFlash = Math.max(0, pylonFlash - dt);
          if (was && pylonFlash === 0 && pylonManual === null) syncControls();
        }
        heroVisHp += (heroHp - heroVisHp) * Math.min(1, dt * 7);
        enemyVisHp += (enemyHp - enemyVisHp) * Math.min(1, dt * 7);
        hero.lunge += (0 - hero.lunge) * Math.min(1, dt * 8);
        enemy.lunge += (0 - enemy.lunge) * Math.min(1, dt * 8);

        for (const e of embers) {
          e.y -= e.vy * dt;
          e.x += e.vx * dt;
          if (e.y < 0.05) { e.y = 0.96; e.x = (e.x + 0.37) % 1; }
        }

        // Aurelia animation events
        if (hero.anim === "cast") {
          const dur = animDuration("aurelia", "cast");
          if (!hero.fired && hero.t >= dur * 0.34) {
            hero.fired = true;
            hero.lunge = 0.025;
            projectiles.push({ x: 0.36, y: 0.54, tx: 0.73, ty: 0.56, r: 10, dmg: 245 + wave * 18, crit: false });
          }
          if (hero.t >= dur) {
            hero.anim = "idle";
            hero.t = 0;
            novaCharge = Math.min(3, novaCharge + 1);
            syncControls();
          }
        } else if (hero.anim === "nova") {
          const dur = animDuration("aurelia", "nova");
          if (!hero.fired && hero.t >= dur * 0.42) {
            hero.fired = true;
            spawnVfx("solar_burst", 0.30, 0.42, 0.92, 0);
            projectiles.push({ x: 0.34, y: 0.46, tx: 0.73, ty: 0.55, r: 15, dmg: 590 + wave * 45, crit: true });
          }
          if (hero.t >= dur) {
            hero.anim = "idle";
            hero.t = 0;
            syncControls();
          }
        }

        // Void Sentinel animation events
        if (enemy.anim === "attack") {
          const dur = animDuration("sentinel", "attack");
          if (!enemy.fired && enemy.t >= dur * 0.38) {
            enemy.fired = true;
            enemy.lunge = 0.055;
            damageHero(44 + wave * 6);
          }
          if (enemy.t >= dur) {
            enemy.anim = "idle";
            enemy.t = 0;
            syncControls();
          }
        } else if (enemy.anim === "hit") {
          const dur = animDuration("sentinel", "hit");
          if (enemy.t >= dur) {
            enemy.anim = "idle";
            enemy.t = 0;
          }
        }

        // Projectiles travel & impact
        for (let i = projectiles.length - 1; i >= 0; i--) {
          const p = projectiles[i];
          const dx = p.tx - p.x;
          const dy = p.ty - p.y;
          const dist = Math.hypot(dx, dy);
          const spd = p.crit ? 1.35 : 1.55;
          if (dist <= spd * dt + 0.015) {
            damageEnemy(p.dmg, p.crit);
            projectiles.splice(i, 1);
          } else {
            p.x += (dx / dist) * spd * dt;
            p.y += (dy / dist) * spd * dt;
          }
        }

        // Advance VFX strips & popups
        for (let i = vfxList.length - 1; i >= 0; i--) {
          vfxList[i].t += dt;
          const a = assets.vfx.manifest?.animations?.[vfxList[i].key];
          const dur = a ? a.frames.length / (a.fps || 20) : 0.8;
          if (vfxList[i].t > dur + 0.05) vfxList.splice(i, 1);
        }
        for (let i = popups.length - 1; i >= 0; i--) {
          popups[i].t += dt;
          if (popups[i].t >= popups[i].life) popups.splice(i, 1);
        }

        // Wave respawn & Auto-Battler AI cadence
        if (respawnTimer > 0) {
          respawnTimer = Math.max(0, respawnTimer - dt);
          enemyFade = respawnTimer > 0.7 ? Math.max(0, (respawnTimer - 0.7) / 0.95) : 0;
          if (respawnTimer === 0) {
            wave += 1;
            enemyMax = 1000 + (wave - 1) * 180;
            enemyHp = enemyMax;
            enemyVisHp = enemyMax;
            enemy.alive = true;
            enemy.anim = "idle";
            enemy.t = 0;
            aiCooldown = 0.65;
            syncControls();
          }
        } else {
          enemyFade = Math.min(1, enemyFade + dt * 2.8);
          if (autoBattle && hero.anim === "idle" && enemy.anim === "idle" && projectiles.length === 0) {
            aiCooldown -= dt;
            if (aiCooldown <= 0) {
              if (turnStep === 0) {
                triggerHeroCast();
                turnStep = 1;
                aiCooldown = 0.45;
              } else if (turnStep === 1) {
                triggerEnemyAttack();
                turnStep = 2;
                aiCooldown = 0.42;
              } else if (turnStep === 2) {
                triggerHeroCast();
                turnStep = 3;
                aiCooldown = 0.48;
              } else {
                triggerHeroNova();
                turnStep = 0;
                aiCooldown = 0.65;
              }
            }
          }
        }
      }

      drawStage();
      rafId = requestAnimationFrame(step);
    }

    function ensureTicking() {
      if (visible && !rafId) rafId = requestAnimationFrame(step);
    }

    if (sprAutoBtn) {
      sprAutoBtn.addEventListener("click", () => {
        autoBattle = !autoBattle;
        if (autoBattle) aiCooldown = 0.2;
        syncControls();
      });
    }
    actBtns.forEach((b) => b.addEventListener("click", () => {
      const act = b.dataset.sprAct;
      if (act === "cast") triggerHeroCast();
      else if (act === "nova") triggerHeroNova();
      else if (act === "enemy") triggerEnemyAttack();
    }));
    sprBackdropBtn.addEventListener("click", () => {
      backdropIdx = (backdropIdx + 1) % BACKDROPS.length;
      syncControls();
      drawStage();
    });
    sprPivotBtn.addEventListener("click", () => {
      showPivot = !showPivot;
      syncControls();
      drawStage();
    });
    const setPylon = (val) => {
      pylonManual = val === undefined ? !isPylonAwakened() : val;
      syncControls();
      drawStage();
    };
    if (sprPylonBtn) sprPylonBtn.addEventListener("click", () => setPylon());
    litPicks.forEach((b) => b.addEventListener("click", () => setPylon(b.dataset.sprLit === "true")));

    sprCanvas.addEventListener("pointerdown", (e) => {
      sprCanvas.focus();
      const r = sprCanvas.getBoundingClientRect();
      const nx = (e.clientX - r.left) / r.width;
      const ny = (e.clientY - r.top) / r.height;
      // Clicking the center astrolabe pylon toggles Dormant <-> Awakened
      if (Math.abs(nx - 0.52) < 0.085 && ny > 0.30 && ny < 0.76) {
        setPylon();
        return;
      }
      if (novaCharge >= 2) triggerHeroNova();
      else triggerHeroCast();
    });
    sprCanvas.addEventListener("keydown", (e) => {
      if (e.key === " " || e.key === "Enter") {
        e.preventDefault();
        triggerHeroNova();
      } else if (e.key.toLowerCase() === "l") {
        e.preventDefault();
        setPylon();
      }
    });

    new IntersectionObserver((entries) => {
      visible = entries[0]?.isIntersecting;
      if (visible) ensureTicking();
    }, { threshold: 0.05 }).observe(sprCanvas);
    syncControls();
  }

  /* Scroll-Driven + Interactive 3-Act Motion Transfer Showcase */

  const mtScroll = $("mt-scroll");
  const mtRoom = $("mt-room");
  if (mtScroll && mtRoom) {
    const mtViewport = $("mt-viewport");
    const vSrc = $("mt-v-src");
    const vRep = $("mt-v-rep");
    const vRes = $("mt-v-res");
    const videos = [vSrc, vRep, vRes];
    const layerRep = $("mt-layer-rep");
    const layerRes = $("mt-layer-res");
    const curtain1 = $("mt-curtain-1");
    const curtain2 = $("mt-curtain-2");
    const badge = $("mt-badge");
    const hint = $("mt-hint");
    const playBtn = $("mt-play");
    const scrub = $("mt-scrub");
    const clockEl = $("mt-clock");
    const stepBtns = [...document.querySelectorAll("[data-mt-step]")];
    const viewBtns = [...document.querySelectorAll("[data-mt-view]")];

    let progress = 0;
    let targetProgress = 0;
    let manualOverride = false;
    let dragging = false;
    let userPaused = false;
    let visible = false;
    let viewMode = "wipe";
    let rafId = 0;
    let lastScrollY = window.scrollY;
    let scrubbing = false;

    const clamp01 = (v) => Math.max(0, Math.min(1, v));

    function readScrollProgress() {
      const rect = mtScroll.getBoundingClientRect();
      const span = mtScroll.offsetHeight - window.innerHeight;
      if (span <= 80) return null;
      return clamp01(-rect.top / span);
    }

    function applyVisuals(p) {
      const w1 = clamp01((p - 0.08) / 0.36);
      const w2 = clamp01((p - 0.56) / 0.36);
      const left1 = (1 - w1) * 100;
      const left2 = (1 - w2) * 100;

      layerRep.style.clipPath = `inset(0 0 0 ${left1.toFixed(2)}%)`;
      layerRes.style.clipPath = `inset(0 0 0 ${left2.toFixed(2)}%)`;

      curtain1.style.left = `${left1.toFixed(2)}%`;
      curtain1.classList.toggle("on", w1 > 0.015 && w1 < 0.985 && w2 < 0.02);
      curtain2.style.left = `${left2.toFixed(2)}%`;
      curtain2.classList.toggle("on", w2 > 0.015 && w2 < 0.985);
      // Keep each handle's label inside the picture when its line nears an edge.
      const vw = mtViewport.clientWidth;
      for (const [c, left] of [[curtain1, left1], [curtain2, left2]]) {
        const knob = c.firstElementChild;
        if (!knob || !vw) continue;
        const x = (left / 100) * vw;
        const half = knob.offsetWidth / 2 + 6;
        const shift = Math.max(half, Math.min(vw - half, x)) - x;
        knob.style.transform = `translate(calc(-50% + ${shift.toFixed(1)}px), -50%)`;
      }

      const actIdx = w2 >= 0.45 ? 2 : (w1 >= 0.45 ? 1 : 0);
      for (const b of stepBtns) {
        b.setAttribute("aria-pressed", String(Number(b.dataset.mtStep) === actIdx));
      }
      if (badge) {
        badge.textContent =
          w2 > 0.02 && w2 < 0.98 ? "Wiping · Replace → New scene" :
          w1 > 0.02 && w1 < 0.98 ? "Wiping · Source → Replace in scene" :
          actIdx === 2 ? "03 · New scene, same motion" :
          actIdx === 1 ? "02 · Replace in scene (SAM 3.1 + plate)" :
          "01 · Source performance";
      }
    }

    function syncVideos() {
      if (!vSrc.duration || Number.isNaN(vSrc.duration)) return;
      const t = vSrc.currentTime;
      for (const v of [vRep, vRes]) {
        if (v.readyState >= 2 && Math.abs(v.currentTime - t) > 0.045) {
          try { v.currentTime = t; } catch (_) {}
        }
      }
      if (!scrubbing) {
        scrub.value = String(Math.round((t / vSrc.duration) * 1000));
      }
      if (clockEl) {
        clockEl.textContent = `${t.toFixed(1)} / ${vSrc.duration.toFixed(1)} s`;
      }
    }

    function setPlaying(play) {
      for (const v of videos) {
        if (play) v.play().catch(() => {});
        else v.pause();
      }
      if (playBtn) {
        playBtn.textContent = play ? "Pause" : "Play";
        playBtn.setAttribute("aria-pressed", String(play));
      }
    }

    function tick() {
      if (!visible) { rafId = 0; return; }
      if (!manualOverride && !dragging && viewMode === "wipe") {
        const sp = readScrollProgress();
        if (sp !== null) targetProgress = sp;
      }
      const diff = targetProgress - progress;
      progress = Math.abs(diff) < 0.001 ? targetProgress : progress + diff * (dragging ? 0.35 : 0.18);
      applyVisuals(progress);
      syncVideos();
      rafId = requestAnimationFrame(tick);
    }

    function ensureTick() {
      if (visible && !rafId) rafId = requestAnimationFrame(tick);
    }

    window.addEventListener("scroll", () => {
      const dy = Math.abs(window.scrollY - lastScrollY);
      lastScrollY = window.scrollY;
      if (dy > 14 && !dragging && viewMode === "wipe") {
        const sp = readScrollProgress();
        if (sp !== null) {
          manualOverride = false;
          targetProgress = sp;
        }
      }
    }, { passive: true });

    const setFromPointer = (e) => {
      const r = mtViewport.getBoundingClientRect();
      const nx = clamp01((e.clientX - r.left) / Math.max(1, r.width));
      manualOverride = true;
      targetProgress = 1 - nx;
      if (hint) hint.textContent = "Dragging the wipe";
      ensureTick();
    };

    mtViewport.addEventListener("pointerdown", (e) => {
      if (viewMode !== "wipe") return;
      dragging = true;
      mtViewport.setPointerCapture(e.pointerId);
      setFromPointer(e);
    });
    mtViewport.addEventListener("pointermove", (e) => {
      if (dragging) setFromPointer(e);
    });
    const endDrag = () => { dragging = false; };
    mtViewport.addEventListener("pointerup", endDrag);
    mtViewport.addEventListener("pointercancel", endDrag);
    mtViewport.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight") {
        e.preventDefault(); manualOverride = true; targetProgress = clamp01(targetProgress + 0.25); ensureTick();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault(); manualOverride = true; targetProgress = clamp01(targetProgress - 0.25); ensureTick();
      }
    });

    stepBtns.forEach((btn) => btn.addEventListener("click", () => {
      const idx = Number(btn.dataset.mtStep);
      manualOverride = true;
      targetProgress = idx === 0 ? 0 : (idx === 1 ? 0.5 : 1);
      if (viewMode !== "wipe") {
        viewMode = "wipe";
        mtRoom.classList.remove("triple");
        for (const vb of viewBtns) vb.setAttribute("aria-pressed", String(vb.dataset.mtView === "wipe"));
      }
      if (hint) hint.textContent = "Scroll or drag to wipe";
      ensureTick();
    }));

    viewBtns.forEach((btn) => btn.addEventListener("click", () => {
      viewMode = btn.dataset.mtView;
      mtRoom.classList.toggle("triple", viewMode === "triple");
      for (const vb of viewBtns) vb.setAttribute("aria-pressed", String(vb.dataset.mtView === viewMode));
      ensureTick();
    }));

    if (playBtn) {
      playBtn.addEventListener("click", () => {
        userPaused = !userPaused;
        setPlaying(!userPaused);
      });
    }

    if (scrub) {
      scrub.addEventListener("input", () => {
        scrubbing = true;
        if (!vSrc.duration || Number.isNaN(vSrc.duration)) return;
        const target = (Number(scrub.value) / 1000) * vSrc.duration;
        for (const v of videos) {
          try { v.currentTime = target; } catch (_) {}
        }
        if (clockEl) clockEl.textContent = `${target.toFixed(1)} / ${vSrc.duration.toFixed(1)} s`;
      });
      scrub.addEventListener("change", () => { scrubbing = false; });
    }

    new IntersectionObserver((entries) => {
      visible = Boolean(entries[0]?.isIntersecting);
      if (visible) {
        if (!userPaused && !reduceMotion) setPlaying(true);
        ensureTick();
      } else {
        for (const v of videos) v.pause();
      }
    }, { threshold: 0.05 }).observe(mtScroll);

    applyVisuals(0);
  }

  /* Copy buttons */

  for (const button of document.querySelectorAll("[data-copy]")) {
    const label = button.querySelector("span");
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(button.dataset.copy);
        label.textContent = "Copied";
        button.classList.add("done");
        announce.textContent = "Copied to the clipboard";
      } catch (_) {
        label.textContent = "Select and copy";
      }
      setTimeout(() => {
        label.textContent = "Copy";
        button.classList.remove("done");
        announce.textContent = "";
      }, 1800);
    });
  }
})();
