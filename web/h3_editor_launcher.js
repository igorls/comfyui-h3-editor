import { app } from "/scripts/app.js";

const FRAME_ID = "h3-editor-frame";

function closeEditor() {
  document.getElementById(FRAME_ID)?.remove();
}

function openEditor() {
  if (document.getElementById(FRAME_ID)) return;
  const frame = document.createElement("iframe");
  frame.id = FRAME_ID;
  frame.src = "/h3take/editor";
  frame.title = "H3 Editor";
  frame.style.cssText = "position:fixed;inset:0;width:100vw;height:100vh;border:0;z-index:10000;background:#1b1e23";
  document.body.appendChild(frame);
}

window.addEventListener("message", (event) => {
  if (event.origin === location.origin && event.data === "h3-editor-close") closeEditor();
});

app.registerExtension({
  name: "h3take.editor",
  commands: [{ id: "H3Take.OpenEditor", label: "Open H3 Editor", icon: "pi pi-video", function: openEditor }],
  menuCommands: [{ path: ["View"], commands: ["H3Take.OpenEditor"] }],
  setup() {
    app.extensionManager?.registerSidebarTab?.({
      id: "h3-editor",
      icon: "pi pi-video",
      title: "H3 Editor",
      tooltip: "Open H3 Editor",
      type: "custom",
      render(element) {
        const button = document.createElement("button");
        button.textContent = "Open H3 Editor";
        button.style.cssText = "margin:16px;padding:10px 16px;border-radius:6px;border:0;cursor:pointer;background:#e0a24a;color:#1b1e23;font:600 14px system-ui";
        button.addEventListener("click", openEditor);
        element.replaceChildren(button);
        openEditor();
      },
    });
  },
});
