import { app } from "/scripts/app.js";

const PROMPT_ROOM = 150;

// H3 Take has a control for every kind of shot. Only the ones that do something right now are shown.
function relevant(node) {
  const value = (name) => node.widgets?.find((w) => w.name === name)?.value;
  const linked = (prefix) => (node.inputs || []).some((input) => input.name.startsWith(prefix) && input.link != null);
  const holding = value("video_mode") !== "off" || linked("carry");
  const extending = holding && value("video_mode") !== "v2v";
  return {
    mid_frames: linked("mid_images."),
    hold_frames: extending,
    align: extending,
    video_feather: extending,
    strength: holding,
    keep: holding && value("strength") === "custom",
    audio_feather: value("audio_hold") !== "off",
    ref_image_size: linked("ref_images.") || value("anchor") !== "none",
  };
}

function refresh(node) {
  const show = relevant(node);
  let changed = false;
  for (const [name, visible] of Object.entries(show)) {
    const widget = node.widgets?.find((w) => w.name === name);
    if (!widget || widget.hidden === !visible) continue;
    widget.hidden = !visible;
    changed = true;
  }
  if (!changed) return;
  node.setSize([node.size[0], node.computeSize()[1] + PROMPT_ROOM]);
  node.graph?.setDirtyCanvas?.(true, true);
}

function mount(node) {
  for (const name of ["video_mode", "strength", "audio_hold", "anchor"]) {
    const widget = node.widgets?.find((w) => w.name === name);
    if (!widget) continue;
    const previous = widget.callback;
    widget.callback = function () {
      const result = previous?.apply(this, arguments);
      refresh(node);
      return result;
    };
  }
  const connections = node.onConnectionsChange;
  node.onConnectionsChange = function () {
    const result = connections?.apply(this, arguments);
    refresh(node);
    return result;
  };
  const configured = node.onConfigure;
  node.onConfigure = function () {
    const result = configured?.apply(this, arguments);
    refresh(node);
    return result;
  };
  node.setSize([Math.max(node.size[0], 440), node.size[1]]);
  refresh(node);
}

app.registerExtension({
  name: "h3take.take",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "H3Take") return;
    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      try {
        mount(this);
      } catch (err) {
        console.error("H3 Take controls failed to mount", err);
      }
      return result;
    };
  },
});
