// A sound's gain points belong to its source file. Trimming can hide a point and reveal it again.
// Fades follow the current ends of its timeline placement.
export function curveLevel(sound, sourceFrame) {
  const points = sound.gain_points;
  if (!points?.length) return 1;
  let from = { frame: 0, gain: 1 };
  const length = Math.max(1, Math.round(sound.seconds * 24));
  for (const to of points) {
    if (sourceFrame === to.frame) return to.gain;
    if (sourceFrame < to.frame) return from.gain + (to.gain - from.gain) * (sourceFrame - from.frame) / (to.frame - from.frame);
    from = to;
  }
  if (from.frame >= length) return from.gain;
  return from.gain + (1 - from.gain) * Math.min(1, (sourceFrame - from.frame) / (length - from.frame));
}

export function soundEnvelope(sound, frame) {
  const length = sound.frames;
  const at = Math.max(0, Math.min(length, frame));
  const fadeIn = Math.min(length, sound.fade_in || 0);
  const fadeOut = Math.min(length, sound.fade_out || 0);
  return curveLevel(sound, at + (sound.offset || 0) * 24)
    * (fadeIn ? Math.min(1, at / fadeIn) : 1)
    * (fadeOut ? Math.min(1, (length - at) / fadeOut) : 1);
}

let mixer;
const connections = new WeakMap();

// Create or resume the audio graph from the Play gesture so the browser allows it to sound.
export function resumeSoundMixer() {
  if (!globalThis.AudioContext) return;
  mixer ??= new AudioContext();
  if (mixer.state === "suspended") mixer.resume().catch(() => {});
}

export function setSoundLevel(player, sound, frame, monitor) {
  const base = (sound.gain ?? 1) * monitor;
  const level = base * soundEnvelope(sound, frame);
  let link = connections.get(player);
  if (!link && mixer?.state === "running") {
    const source = mixer.createMediaElementSource(player);
    const gain = mixer.createGain();
    source.connect(gain).connect(mixer.destination);
    link = { source, gain };
    connections.set(player, link);
  }
  if (!link) {
    player.volume = Math.min(1, level);
    return;
  }
  player.volume = 1;
  const param = link.gain.gain, now = mixer.currentTime;
  param.cancelScheduledValues(now);
  param.setValueAtTime(level, now);
  // Ramp between picture frames as well, including on slow display refreshes.
  param.linearRampToValueAtTime(base * soundEnvelope(sound, frame + 1), now + 1 / 24);
}

export function disconnectSound(player) {
  const link = connections.get(player);
  if (link) {
    link.source.disconnect();
    link.gain.disconnect();
    connections.delete(player);
  }
}
