// The teaser drives the timeline and the shot pane, the way the editor's playhead does.
(() => {
  const FPS = 24;
  const TOTAL = 236;
  const PROJECT = "Cafe_teaser";

  // The café teaser's three clips, with the fields they were rendered from.
  const CLIPS = [
    {
      id: "c68d69408", name: "Door", from: 0, to: 90, seconds: 3.75, strip: "media/door.jpg",
      action: "Seen from the quiet street through the café's glass door. A single small green sign hangs on a cord from a hook on the inside of the glass, at chest height, showing its Closed side. A young barista with short dark hair, in a white T-shirt and a dark green apron, steps up to the door and turns the key in the lock, then takes the hanging sign in one hand and turns it around so its OPEN side faces the street, the word OPEN in large white capital letters, and smiles out at the morning. The narrator is heard early in the shot.",
      camera: "Medium shot from the street at eye level, the barista's face clearly visible through the glass, slow push in",
      speaker: "@Narrator, off-screen",
      dialogue: "Every morning starts somewhere.",
      sound: "Keys jingling, the lock clicking open, a quiet street, birds in the distance",
    },
    {
      id: "c06f7138e", name: "Espresso", from: 90, to: 146, seconds: 2.33, strip: "media/espresso.jpg",
      action: "Inside the café, steam rises in soft curls from a chrome espresso machine as a thin stream of dark espresso pours into a white ceramic cup, filling it with golden crema.",
      camera: "Close-up, slight slow tilt down to the cup",
      speaker: "",
      dialogue: "",
      sound: "Hiss of steam and the gurgle of the espresso machine",
    },
    {
      id: "c7d209ad3", name: "First sip", from: 146, to: 236, seconds: 3.75, strip: "media/sip.jpg",
      action: "A woman in her thirties in a cream sweater sits at the café's window seat, morning sunlight on her face. She lifts a white espresso cup, takes the first sip, lowers it and smiles warmly. The narrator is heard at the very end of the shot, as she smiles.",
      camera: "Medium close-up, side-on against the bright window, gentle handheld",
      speaker: "@Narrator, off-screen",
      dialogue: "Make it yours.",
      sound: "Soft café murmur, a cup set down on a saucer",
    },
  ];

  // Loudest sample in each of 240 slices of the teaser's sound, 0 to 100.
  const PEAKS = [1,2,2,2,2,2,2,2,1,2,2,1,2,2,2,2,2,2,2,2,2,6,4,3,2,1,2,1,1,2,2,2,1,3,2,2,1,1,1,1,2,1,2,2,95,97,71,72,100,97,88,91,98,97,96,97,88,86,99,92,20,7,3,97,96,91,88,4,2,3,4,90,97,62,38,35,46,37,33,18,6,10,1,1,1,1,1,1,1,1,2,2,0,0,0,1,0,1,1,2,1,1,2,2,2,2,2,2,2,2,2,1,2,3,2,1,2,2,3,1,2,2,2,1,1,2,1,3,2,1,2,2,2,2,1,1,1,1,1,1,2,1,1,1,1,1,1,2,1,0,1,1,0,1,1,0,1,1,0,0,1,0,1,0,1,1,1,1,0,1,1,1,1,0,1,0,0,0,1,1,1,1,1,1,2,2,1,2,2,4,5,13,5,3,6,2,1,1,0,1,1,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,42,91,98,98,13,95,98,9,24,71,92,85,60,28,9,5,14,3,2,2,0,0,1];

  const $ = (id) => document.getElementById(id);
  const video = $("teaser");
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
    const fields = { name: clip.name, action: clip.action, camera: clip.camera };
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
      const shown = typeof value === "number" ? [["n", String(value)]] : [["s", JSON.stringify(value)]];
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

  // The cut opens on First sip, the poster's frame, so poster, playhead and panes agree
  // before the video loads and whenever the browser refuses to play it on its own.
  const START = 175;
  function begin() {
    video.currentTime = (START + 0.5) / FPS;
    if (!reduceMotion) video.play().catch(() => showPlaying());
  }
  if (video.readyState >= 1) begin();
  else video.addEventListener("loadedmetadata", begin, { once: true });
  showPlaying();
  show(START);

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
