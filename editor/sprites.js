const ids = ["NW", "N", "NE", "W", "", "E", "SW", "S", "SE"];
const motions = ["idle", "run", "custom"];
const $ = (id) => document.getElementById(id);
const opposite = (direction) => direction.replace(/[EW]/, (c) => (c === "E" ? "W" : "E"));
const directions = (count) => ids.filter((d) => d && (count === 8 || d.length === 1));
const motionOf = (s) => (s.motion === "idle" ? "idle" : s.guide || s.motion === "prompt" ? "custom" : "run");
const actorOf = (clip, cast) => cast?.[0] || clip.sprite_group || "clip-" + clip.id;
const slotKey = (camera, motion, direction) => [camera, motion, direction].join("/");
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
    motion = "run",
    dir = "S",
    count = 8,
    slots,
    prefs,
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
    count = prefs.count || 8;
    motion = "run";
    dir = "S";
    audition = null;
  }
  function focusClip(clip) {
    const t = clip?.takes?.find((t) => t.id === clip.active && t.sprite);
    const config = t?.saved?.sprite || clip?.sprite;
    if (!config) return;
    pickActor(actorOf(clip, t?.saved?.cast ?? clip.cast));
    camera = config.camera;
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
      ...Array.from(people.values(), (p) =>
        button(
          p.name,
          () => {
            pickActor(p.id);
            refresh();
          },
          p.id === actor,
        ),
      ),
    );
    $("spriteTitle").textContent = person?.name || "Add a character reference to start";
    const refs = $("spriteRefs");
    refs.replaceChildren();
    if (cast) {
      for (const asset of [cast, ...(cast.pictures || [])].filter((a) => a.file)) {
        const img = document.createElement("img");
        img.alt = cast.name + " reference";
        img.src = editor.viewUrl(asset.file, asset.subfolder);
        refs.append(img);
      }
      refs.append(button("Edit references", () => editor.editCast(cast.id)));
    }
    slots = spriteSlots(project(), actor, camera, motion);
    frames = [];
    $("spriteMotion").value = motion;
    $("spriteCamera").value = camera;
    $("spriteCount").value = String(count);
    const compass = $("spriteCompass");
    compass.replaceChildren();
    for (const [pos, h] of ids.entries()) {
      if (!directions(count).includes(h)) continue;
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
      if (row) b.append(frame(row));
      b.append(label);
      b.setAttribute("aria-label", h + (row?.mirror ? " mirrored" : row ? "" : " empty"));
      compass.append(b);
    }
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
      const b = button(label, () => select(row), selected);
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
      b.prepend(frame(row));
      b.append(film);
      takes.append(b);
    }
    if (!rows.length) takes.textContent = pending ? "Rendering…" : "No takes for this direction.";
    $("spriteTakesTitle").textContent = dir + " takes";
    $("spriteDraft").textContent = pending ? "Rendering " + dir + "…" : "New " + dir + " draft";
    $("spriteDraft").disabled =
      !actor || !!pending || busy || (motion === "custom" && !slots[dir].clips.length);
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
    paint();
  }
  function exportPicks() {
    const out = [];
    for (const m of motions) {
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
      tile = sh.tile,
      pose = Math.floor(clock * sh.fps) % sh.poses;
    ctx.save();
    ctx.imageSmoothingEnabled = false;
    ctx.translate(Math.round(x * w), Math.round(y * h));
    if (row.mirror) ctx.scale(-1, 1);
    ctx.drawImage(img, pose * tile, 0, tile, tile, -sh.pivot[0], -sh.pivot[1], tile, tile);
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
    audition = null;
    if (actor) remember((p) => (p.camera = camera));
    refresh();
  };
  $("spriteMotion").onchange = (e) => {
    motion = e.target.value;
    audition = null;
    refresh();
  };
  $("spriteCount").onchange = (e) => {
    count = Number(e.target.value);
    if (count === 4 && dir.length === 2) dir = "S";
    audition = null;
    if (actor) remember((p) => (p.count = count));
    refresh();
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
      if (!clip) clip = editor.makeClip(actor, roster().get(actor).name, camera, motion, dir);
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
    exporting = true;
    refresh();
    $("spriteStatus").textContent = "Packing " + picks.length + " animation slots…";
    try {
      await editor.download(name, { tile: Number($("spriteTile").value), style: style(), picks });
    } catch (error) {
      editor.toast(error.message, true);
    } finally {
      exporting = false;
      refresh();
    }
  };
  return { refresh, openClip: (clip) => flipMode(true, clip), isOpen: open };
}
