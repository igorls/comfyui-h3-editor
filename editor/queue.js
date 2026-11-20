// Managed Render Queue UI for H3 Editor (works in full-screen across Video, Sprites, Props and ComfyUI jobs).

const $ = (id) => document.getElementById(id);

export function relativeTime(atSeconds, nowSeconds = Date.now() / 1000) {
  if (!atSeconds) return "";
  const delta = Math.max(0, Math.round(nowSeconds - atSeconds));
  if (delta < 10) return "just now";
  if (delta < 60) return delta + "s ago";
  if (delta < 3600) return Math.floor(delta / 60) + "m ago";
  if (delta < 86400) return Math.floor(delta / 3600) + "h ago";
  return Math.floor(delta / 86400) + "d ago";
}

export function formatJobMeta(job, currentFolder = null) {
  const tags = [];
  if (job.kind === "comfyui") {
    tags.push("ComfyUI");
    return tags;
  }
  if (job.kind === "prop") {
    tags.push("4 MP " + (job.quality || "prop"));
  } else if (job.kind === "sprite") {
    tags.push(job.quality === "finish" || job.quality === "final" ? "Final sprite" : "Sprite draft");
    if (job.sprite?.direction) tags.push(job.sprite.direction);
  } else {
    const q = job.quality === "upscale"
      ? (job.upscaler ? job.upscaler.toUpperCase() : "Upscale")
      : job.quality === "finish" ? "Finish draft"
      : job.quality === "final" ? "Final" : "Draft";
    tags.push(job.part ? q + " · partial" : job.edit ? q + " · edit" : q);
  }
  if (Array.isArray(job.size) && job.size.length === 2) tags.push(job.size[0] + "×" + job.size[1]);
  if (job.frames && job.frames > 1) tags.push(job.frames + "f");
  if (job.folder && currentFolder && job.folder !== currentFolder) {
    tags.push(job.project || job.folder);
  }
  return tags;
}

export function mergeQueueState(serverSnapshot, localJobs, runningJob, currentFolder = null, clipLookup = null) {
  const snap = serverSnapshot || { running: [], pending: [], history: [] };
  const locals = localJobs instanceof Map ? localJobs : new Map(Object.entries(localJobs || {}));
  const seen = new Set();

  const enrich = (item, fallbackState) => {
    seen.add(item.id);
    const local = locals.get(item.id);
    const clip = (!item.name || item.name === item.clip) && item.clip && clipLookup ? clipLookup(item.clip) : null;
    const isRunning = fallbackState === "running" || (runningJob && runningJob.id === item.id);
    return {
      ...item,
      name: item.name || clip?.name || item.clip || "Render",
      state: isRunning ? "running" : fallbackState,
      stage: local?.stage || (isRunning ? "Running" : "Waiting in queue"),
      step: local?.step || 0,
      max: local?.max || 0,
      progress: local?.progress || 0,
      preview: local?.preview || null,
      localOnly: false,
    };
  };

  const running = (snap.running || []).map((item) => enrich(item, "running"));
  const pending = (snap.pending || []).map((item) => enrich(item, "queued"));

  // Include optimistically-queued local jobs before the next server poll confirms them.
  for (const [id, local] of locals) {
    if (seen.has(id)) continue;
    const clip = local.clip && clipLookup ? clipLookup(local.clip) : null;
    const spriteCfg = local.saved?.sprite || clip?.sprite;
    const item = {
      id,
      kind: spriteCfg ? "sprite" : "clip",
      folder: currentFolder,
      project: null,
      clip: local.clip || null,
      name: clip?.name || local.clip || "Clip",
      quality: local.quality || "draft",
      size: local.size,
      frames: local.frames,
      part: local.part,
      edit: Boolean(local.edit),
      upscaler: local.upscaler,
      sprite: spriteCfg ? { direction: spriteCfg.direction, camera: spriteCfg.camera, motion: spriteCfg.animation || spriteCfg.motion || "run" } : null,
      state: runningJob && runningJob.id === id ? "running" : "queued",
      stage: local.stage || (runningJob && runningJob.id === id ? "Running" : "Waiting in queue"),
      step: local.step || 0,
      max: local.max || 0,
      progress: local.progress || 0,
      preview: local.preview || null,
      position: pending.length + 1,
      localOnly: true,
    };
    if (item.state === "running") running.push(item);
    else pending.push(item);
  }

  return {
    running,
    pending,
    history: Array.isArray(snap.history) ? snap.history : [],
  };
}

export function summarizeQueue(merged) {
  const runningCount = merged.running.length;
  const pendingCount = merged.pending.length;
  const total = runningCount + pendingCount;
  const failedCount = merged.history.filter((h) => h.state === "failed").length;
  const front = merged.running[0] || merged.pending[0] || null;
  const activeProgress = front && front.state === "running" ? Math.max(0, Math.min(1, front.progress || 0)) : 0;

  if (!total) {
    return {
      busy: false,
      runningCount: 0,
      pendingCount: 0,
      failedCount,
      activeProgress: 0,
      label: failedCount ? "Queue · " + failedCount + " failed" : "Queue",
      summary: "Idle",
    };
  }

  const parts = [];
  if (merged.running[0]) {
    const pct = Math.round(activeProgress * 100);
    parts.push(merged.running[0].name + (pct > 0 ? " · " + pct + "%" : ""));
  } else if (pendingCount) {
    parts.push(merged.pending[0].name + " · queued");
  }
  const extra = total - 1;
  if (extra > 0) parts.push("+" + extra + " queued");

  const sumParts = [];
  if (runningCount) sumParts.push(runningCount + " running");
  if (pendingCount) sumParts.push(pendingCount + " waiting");
  return {
    busy: true,
    runningCount,
    pendingCount,
    failedCount,
    activeProgress,
    label: parts.join(" · "),
    summary: sumParts.join(" · "),
  };
}

export function installQueue(editor) {
  const button = $("queueButton");
  const label = $("queueLabel");
  const bar = $("queueButtonBar");
  const panel = $("queuePanel");
  const summaryEl = $("queueSummary");
  const clearPendingBtn = $("queueClearPending");
  const stopAllBtn = $("queueStopAll");
  const activeList = $("queueActive");
  const historyWrap = $("queueHistoryWrap");
  const historyList = $("queueHistory");
  const clearHistoryBtn = $("queueClearHistory");

  let snapshot = { running: [], pending: [], history: [] };
  let timer = 0;
  let fetching = false;
  let busyCancel = false;

  const isOpen = () => Boolean(panel && panel.matches(":popover-open"));

  async function refreshFromServer() {
    if (fetching) return;
    fetching = true;
    try {
      const data = await editor.fetchQueue();
      if (data && typeof data === "object") {
        snapshot = {
          running: Array.isArray(data.running) ? data.running : [],
          pending: Array.isArray(data.pending) ? data.pending : [],
          history: Array.isArray(data.history) ? data.history : [],
        };
      }
      render();
    } catch (_) {
      // Keep showing local job state even if server poll temporarily fails.
      render();
    } finally {
      fetching = false;
    }
  }

  function schedulePoll() {
    clearTimeout(timer);
    timer = 0;
    if (!isOpen()) return;
    const hasWork = snapshot.running.length + snapshot.pending.length + editor.state.jobs.size > 0;
    timer = setTimeout(async () => {
      if (!isOpen()) return;
      await refreshFromServer();
      schedulePoll();
    }, hasWork ? 2000 : 5000);
  }

  async function cancel(payload, successToast) {
    if (busyCancel) return;
    busyCancel = true;
    render();
    try {
      const res = await editor.cancelQueue(payload);
      if (res?.queue) snapshot = res.queue;
      await editor.afterQueueCancel(res?.cancelled || []);
      if (successToast) editor.toast(successToast);
    } catch (error) {
      editor.toast(error.message || "Could not cancel render.", true);
    } finally {
      busyCancel = false;
      render();
    }
  }

  function renderItem(item, isHistory = false) {
    const row = document.createElement("div");
    row.className = "queue-item " + (item.state || "queued");

    const head = document.createElement("div");
    head.className = "queue-item-head";

    const dot = document.createElement("i");
    dot.className = "queue-dot";
    dot.setAttribute("aria-hidden", "true");

    const titleWrap = document.createElement("div");
    titleWrap.className = "queue-item-title";
    const nameEl = document.createElement("b");
    nameEl.textContent = (!isHistory && item.state === "queued" && item.position ? "#" + item.position + " " : "") + item.name;
    titleWrap.append(nameEl);

    const tags = formatJobMeta(item, editor.state.folder);
    if (tags.length) {
      const meta = document.createElement("span");
      meta.className = "queue-item-meta";
      meta.textContent = tags.join(" · ");
      titleWrap.append(meta);
    }

    const actions = document.createElement("div");
    actions.className = "queue-item-actions";

    if (item.clip && item.folder === editor.state.folder && editor.clipOf(item.clip)) {
      const openBtn = document.createElement("button");
      openBtn.type = "button";
      openBtn.className = "quiet";
      openBtn.textContent = "Open";
      openBtn.onclick = () => {
        panel.hidePopover();
        editor.openJobClip(item.clip);
      };
      actions.append(openBtn);
    }

    if (!isHistory) {
      const stopBtn = document.createElement("button");
      stopBtn.type = "button";
      stopBtn.className = "quiet danger";
      stopBtn.disabled = busyCancel;
      stopBtn.textContent = item.state === "running" ? "Stop" : "Remove";
      stopBtn.title = item.state === "running" ? "Interrupt this active render" : "Remove from queue";
      stopBtn.onclick = () => cancel({ id: item.id }, item.state === "running" ? "Render stopped." : "Removed from queue.");
      actions.append(stopBtn);
    } else {
      const badge = document.createElement("span");
      badge.className = "queue-state-pill " + item.state;
      badge.textContent = item.state === "done" ? "Done" : item.state === "failed" ? "Failed" : "Stopped";
      actions.append(badge);
    }

    head.append(dot, titleWrap, actions);
    row.append(head);

    if (!isHistory && item.state === "running") {
      const sub = document.createElement("div");
      sub.className = "queue-item-sub";
      const pct = Math.round((item.progress || 0) * 100);
      const stepText = item.max ? ` · step ${item.step}/${item.max} (${pct}%)` : pct > 0 ? ` · ${pct}%` : "";
      const stageSpan = document.createElement("span");
      stageSpan.textContent = (item.stage || "Running") + stepText;
      sub.append(stageSpan);
      row.append(sub);

      const track = document.createElement("div");
      track.className = "queue-progress";
      const fill = document.createElement("i");
      fill.style.width = pct + "%";
      if (!pct) fill.classList.add("indeterminate");
      track.append(fill);
      row.append(track);
    } else if (isHistory) {
      const sub = document.createElement("div");
      sub.className = "queue-item-sub";
      const age = relativeTime(item.at);
      sub.textContent = item.error ? item.error + (age ? " · " + age : "") : age;
      if (item.error) sub.classList.add("error");
      if (sub.textContent) row.append(sub);
    }

    return row;
  }

  function render() {
    if (!button || !panel) return;
    const merged = mergeQueueState(snapshot, editor.state.jobs, editor.state.running, editor.state.folder, editor.clipOf);
    const sum = summarizeQueue(merged);

    button.classList.toggle("queue-chip", sum.busy);
    button.classList.toggle("quiet", !sum.busy);
    button.classList.toggle("has-failed", !sum.busy && sum.failedCount > 0);
    label.textContent = sum.label;
    bar.style.width = sum.busy ? Math.round(sum.activeProgress * 100) + "%" : "0%";
    bar.hidden = !sum.busy;

    const projectJobs = [...editor.state.jobs.values()];
    if ($("cancelRenders")) $("cancelRenders").disabled = !projectJobs.length;
    if ($("cancelFinals")) $("cancelFinals").disabled = !projectJobs.some((job) => job.quality === "final");

    if (!isOpen()) return;

    summaryEl.textContent = sum.summary;
    clearPendingBtn.disabled = busyCancel || !merged.pending.length;
    stopAllBtn.disabled = busyCancel || !(merged.running.length + merged.pending.length);

    activeList.replaceChildren();
    if (!merged.running.length && !merged.pending.length) {
      const empty = document.createElement("p");
      empty.className = "queue-empty";
      empty.textContent = "No active renders. Drafts, finals, sprites, and 4 MP props queued from H3 Editor appear here.";
      activeList.append(empty);
    } else {
      for (const item of merged.running) activeList.append(renderItem(item, false));
      for (const item of merged.pending) activeList.append(renderItem(item, false));
    }

    historyWrap.hidden = !merged.history.length;
    historyList.replaceChildren(...merged.history.map((item) => renderItem(item, true)));
  }

  function positionPanel() {
    panel.style.left = "0px";
    panel.style.top = "0px";
    const at = button.getBoundingClientRect();
    panel.style.left = Math.max(8, Math.min(at.right - panel.offsetWidth, window.innerWidth - panel.offsetWidth - 8)) + "px";
    panel.style.top = (at.bottom + 6) + "px";
  }

  window.addEventListener("resize", () => {
    if (isOpen()) positionPanel();
  });

  panel.addEventListener("toggle", () => {
    const opened = isOpen();
    button.setAttribute("aria-expanded", String(opened));
    if (opened) {
      render();
      positionPanel();
      refreshFromServer().then(() => {
        if (isOpen()) positionPanel();
        schedulePoll();
      });
    } else {
      clearTimeout(timer);
      timer = 0;
    }
  });

  clearPendingBtn.onclick = () => cancel({ scope: "pending" }, "Cleared waiting renders.");
  stopAllBtn.onclick = () => cancel({ scope: "all" }, "Stopped all renders.");
  clearHistoryBtn.onclick = async () => {
    try {
      const res = await editor.clearQueueHistory();
      if (res?.queue) snapshot = res.queue;
      else snapshot.history = [];
      render();
    } catch (error) {
      editor.toast(error.message || "Could not clear history.", true);
    }
  };

  return {
    render,
    refresh: refreshFromServer,
  };
}
