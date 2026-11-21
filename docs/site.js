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

  /* Interactive Spritesheet Showcase (Vesper 8-way atlas + paired 4 MP lantern) */

  const sprCanvas = $("spr-canvas");
  if (sprCanvas) {
    const sctx = sprCanvas.getContext("2d");
    const sprReadout = $("spr-readout");
    const sprBackdropBtn = $("spr-backdrop");
    const sprPivotBtn = $("spr-pivot");
    const sprLanternBtn = $("spr-lantern");
    const motionBtns = [...document.querySelectorAll("[data-spr-motion]")];
    const dirBtns = [...document.querySelectorAll("[data-spr-dir]")];
    const litPicks = [...document.querySelectorAll("[data-spr-lit]")];

    const BACKDROPS = ["dark", "checker", "light"];
    const KEY_MAP = { w: "N", arrowup: "N", s: "S", arrowdown: "S", a: "W", arrowleft: "W", d: "E", arrowright: "E" };
    const keys = new Set();
    let pointer = null;
    let manifest = null;
    let atlasImg = null;
    const lanternImgs = { unlit: new Image(), lit: new Image() };
    lanternImgs.unlit.src = "media/sprites/lantern-unlit.webp";
    lanternImgs.lit.src = "media/sprites/lantern-lit.webp";

    let motion = "run";
    let dir = "S";
    let backdropIdx = 0;
    let showPivot = false;
    let lanternLit = true;
    let px = 0.42;
    let py = 0.66;
    const lx = 0.72;
    const ly = 0.66;
    let clock = 0;
    let lastTs = 0;
    let visible = false;
    let rafId = 0;

    function resolveAnim(m, d) {
      if (!manifest) return null;
      const anims = manifest.animations || {};
      if (anims[m + "_" + d]) return { key: m + "_" + d, anim: anims[m + "_" + d] };
      // Idle has 4 cardinal directions; snap diagonals to nearest cardinal.
      const fallback = d.length === 2 ? d[0] : "S";
      if (anims[m + "_" + fallback]) return { key: m + "_" + fallback, anim: anims[m + "_" + fallback] };
      const firstKey = Object.keys(anims)[0];
      return firstKey ? { key: firstKey, anim: anims[firstKey] } : null;
    }

    function syncControls() {
      for (const b of motionBtns) b.setAttribute("aria-pressed", String(b.dataset.sprMotion === motion));
      for (const b of dirBtns) {
        const d = b.dataset.sprDir;
        const disabled = motion === "idle" && d.length === 2;
        b.disabled = disabled;
        b.setAttribute("aria-pressed", String(d === dir));
      }
      sprBackdropBtn.textContent = "Backdrop: " + BACKDROPS[backdropIdx][0].toUpperCase() + BACKDROPS[backdropIdx].slice(1);
      sprPivotBtn.setAttribute("aria-pressed", String(showPivot));
      sprLanternBtn.setAttribute("aria-pressed", String(lanternLit));
      sprLanternBtn.textContent = lanternLit ? "Lantern: Lit" : "Lantern: Unlit";
      for (const b of litPicks) b.setAttribute("aria-pressed", String(String(lanternLit) === b.dataset.sprLit));
      const cur = resolveAnim(motion, dir);
      if (sprReadout && cur) {
        sprReadout.textContent = (motion === "run" ? "Run" : "Idle") + " · " + dir + " · " + cur.anim.fps + " fps";
      }
    }

    function drawStage() {
      const w = Math.max(1, sprCanvas.clientWidth);
      const h = Math.max(1, sprCanvas.clientHeight);
      if (sprCanvas.width !== w || sprCanvas.height !== h) {
        sprCanvas.width = w;
        sprCanvas.height = h;
      }
      const mode = BACKDROPS[backdropIdx];
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
        const g = sctx.createRadialGradient(w * 0.52, h * 0.56, 20, w * 0.52, h * 0.56, Math.max(w, h) * 0.65);
        g.addColorStop(0, lanternLit ? "#211c19" : "#191c22");
        g.addColorStop(1, "#0d0e12");
        sctx.fillStyle = g;
        sctx.fillRect(0, 0, w, h);
        // Subtle 42-deg isometric floor grid
        sctx.strokeStyle = "rgba(255, 255, 255, 0.045)";
        sctx.lineWidth = 1;
        sctx.beginPath();
        for (let gx = -h; gx < w + h; gx += 48) {
          sctx.moveTo(gx, 0);
          sctx.lineTo(gx + h * 1.4, h);
          sctx.moveTo(gx + h * 1.4, 0);
          sctx.lineTo(gx, h);
        }
        sctx.stroke();
      }

      const lanternX = Math.round(lx * w);
      const lanternY = Math.round(ly * h);
      const vesperX = Math.round(px * w);
      const vesperY = Math.round(py * h);

      // Warm lantern glow pool on dark ground
      if (lanternLit && mode === "dark") {
        const glow = sctx.createRadialGradient(lanternX, lanternY - 56, 8, lanternX, lanternY - 24, 150);
        glow.addColorStop(0, "rgba(232, 138, 66, 0.28)");
        glow.addColorStop(0.55, "rgba(224, 96, 54, 0.08)");
        glow.addColorStop(1, "rgba(224, 96, 54, 0)");
        sctx.fillStyle = glow;
        sctx.fillRect(0, 0, w, h);
      }

      const drawLantern = () => {
        const img = lanternLit ? lanternImgs.lit : lanternImgs.unlit;
        if (!img.naturalWidth) return;
        const size = 148;
        sctx.save();
        sctx.fillStyle = mode === "light" ? "rgba(0,0,0,0.14)" : "rgba(0,0,0,0.34)";
        sctx.beginPath();
        sctx.ellipse(lanternX, lanternY + 2, 26, 9, 0, 0, Math.PI * 2);
        sctx.fill();
        sctx.imageSmoothingEnabled = false;
        sctx.drawImage(img, lanternX - size / 2, lanternY - size + 8, size, size);
        sctx.restore();
      };

      const drawVesper = () => {
        const cur = resolveAnim(motion, dir);
        if (!cur || !atlasImg || !atlasImg.naturalWidth) return;
        const { anim } = cur;
        const frames = anim.frames || [];
        if (!frames.length) return;
        const idx = Math.floor(clock * (anim.fps || 12)) % frames.length;
        const f = frames[idx];
        const scale = 1.4;
        const dw = Math.round(f.w * scale);
        const dh = Math.round(f.h * scale);
        const pivX = Math.round((anim.pivot?.[0] ?? f.w / 2) * scale);
        const pivY = Math.round((anim.pivot?.[1] ?? f.h - 10) * scale);

        sctx.save();
        sctx.fillStyle = mode === "light" ? "rgba(0,0,0,0.14)" : "rgba(0,0,0,0.34)";
        sctx.beginPath();
        sctx.ellipse(vesperX, vesperY + 2, 26, 9, 0, 0, Math.PI * 2);
        sctx.fill();
        sctx.imageSmoothingEnabled = false;
        sctx.drawImage(atlasImg, f.x, f.y, f.w, f.h, vesperX - pivX, vesperY - pivY, dw, dh);
        sctx.restore();

        if (showPivot) {
          sctx.save();
          sctx.strokeStyle = "rgba(224, 162, 74, 0.55)";
          sctx.lineWidth = 1;
          sctx.setLineDash([4, 3]);
          sctx.strokeRect(vesperX - pivX + 0.5, vesperY - pivY + 0.5, dw - 1, dh - 1);
          sctx.setLineDash([]);
          sctx.strokeStyle = "#e0a24a";
          sctx.beginPath();
          sctx.moveTo(vesperX - 10, vesperY + 0.5);
          sctx.lineTo(vesperX + 10, vesperY + 0.5);
          sctx.moveTo(vesperX + 0.5, vesperY - 10);
          sctx.lineTo(vesperX + 0.5, vesperY + 10);
          sctx.stroke();
          sctx.restore();
        }
      };

      // Y-sort Vesper and the paired lantern prop cleanly!
      if (vesperY < lanternY) { drawVesper(); drawLantern(); }
      else { drawLantern(); drawVesper(); }
    }

    function step(now) {
      if (!visible) { rafId = 0; lastTs = 0; return; }
      const dt = lastTs ? Math.min(0.05, (now - lastTs) / 1000) : 0;
      lastTs = now;
      let dx = Number(keys.has("E")) - Number(keys.has("W"));
      let dy = Number(keys.has("S")) - Number(keys.has("N"));
      if (!dx && !dy && pointer) {
        const gx = (pointer.x - px) * sprCanvas.width;
        const gy = (pointer.y - py) * sprCanvas.height;
        dx = Math.abs(gx) > 8 ? Math.sign(gx) : 0;
        dy = Math.abs(gy) > 8 ? Math.sign(gy) : 0;
      }
      if (dx || dy) {
        if (motion !== "run") { motion = "run"; syncControls(); }
        const h = (dy < 0 ? "N" : dy > 0 ? "S" : "") + (dx < 0 ? "W" : dx > 0 ? "E" : "");
        if (h && h !== dir) { dir = h; syncControls(); }
        const spd = (135 * dt) / Math.hypot(dx, dy);
        px = Math.max(0.09, Math.min(0.91, px + (dx * spd) / sprCanvas.width));
        py = Math.max(0.32, Math.min(0.86, py + (dy * spd) / sprCanvas.height));
      }
      if (!reduceMotion) clock += dt;
      drawStage();
      rafId = requestAnimationFrame(step);
    }

    function ensureTicking() {
      if (visible && !rafId) rafId = requestAnimationFrame(step);
    }

    motionBtns.forEach((b) => b.addEventListener("click", () => {
      motion = b.dataset.sprMotion;
      if (motion === "idle" && dir.length === 2) dir = dir[0];
      syncControls();
      drawStage();
    }));
    dirBtns.forEach((b) => b.addEventListener("click", () => {
      if (b.disabled) return;
      dir = b.dataset.sprDir;
      syncControls();
      drawStage();
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
    const toggleLantern = (val) => {
      lanternLit = val === undefined ? !lanternLit : val;
      syncControls();
      drawStage();
    };
    sprLanternBtn.addEventListener("click", () => toggleLantern());
    litPicks.forEach((b) => b.addEventListener("click", () => toggleLantern(b.dataset.sprLit === "true")));

    const pointerCoords = (e) => {
      const r = sprCanvas.getBoundingClientRect();
      return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
    };
    sprCanvas.addEventListener("pointerdown", (e) => {
      sprCanvas.focus();
      const pt = pointerCoords(e);
      // Clicking near the lantern toggles Unlit <-> Lit directly!
      const distX = Math.abs(pt.x - lx) * sprCanvas.clientWidth;
      const distY = Math.abs(pt.y - (ly - 0.14)) * sprCanvas.clientHeight;
      if (distX < 44 && distY < 76) {
        toggleLantern();
        return;
      }
      sprCanvas.setPointerCapture(e.pointerId);
      pointer = pt;
    });
    sprCanvas.addEventListener("pointermove", (e) => { if (pointer) pointer = pointerCoords(e); });
    const stopPtr = () => { pointer = null; };
    sprCanvas.addEventListener("pointerup", stopPtr);
    sprCanvas.addEventListener("pointercancel", stopPtr);
    sprCanvas.addEventListener("blur", () => { keys.clear(); pointer = null; });
    sprCanvas.addEventListener("keydown", (e) => {
      if (e.key.toLowerCase() === "l") { e.preventDefault(); toggleLantern(); return; }
      const h = KEY_MAP[e.key.toLowerCase()];
      if (h) { e.preventDefault(); keys.add(h); }
    });
    sprCanvas.addEventListener("keyup", (e) => {
      const h = KEY_MAP[e.key.toLowerCase()];
      if (h) { e.preventDefault(); keys.delete(h); }
    });

    fetch("media/sprites/vesper-atlas.json")
      .then((r) => r.json())
      .then((data) => {
        manifest = data;
        atlasImg = new Image();
        atlasImg.onload = () => { syncControls(); drawStage(); ensureTicking(); };
        atlasImg.src = "media/sprites/" + data.image;
        syncControls();
      })
      .catch(() => {});

    new IntersectionObserver((entries) => {
      visible = entries[0]?.isIntersecting;
      if (visible) ensureTicking();
    }, { threshold: 0.05 }).observe(sprCanvas);
    syncControls();
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
