// Writing with a language model: rewrite one clip, or write the shots of a video from a description.
//
// Calls go from this browser straight to the endpoint the user names, through universal-llm-client
// (vendor/universal-llm-client.js), or to this ComfyUI, which runs a text model with its own Generate
// Text node. Only text is sent: the project's look and language, the cast's names and descriptions, and
// the clips' fields. The settings, key included, stay in this browser's storage, never in a project.

const STORE = "h3take.writer";
const LIBRARY = "./vendor/universal-llm-client.js";

// The text model Comfy recommends for writing prompts for MiniMax H3, and where it comes from.
export const COMFY_MODEL = "qwen3.5_9b_w4a8_gptq.safetensors";
const COMFY_SOURCE = "Comfy-Org/MiniMax-H3-Image";

// Where a rewrite can be asked for: this ComfyUI, or one of the four wire protocols the client speaks.
// An empty address takes the client's own default.
export const PROVIDERS = [
  { id: "comfyui", title: "This ComfyUI", address: "", key: false, local: true,
    hint: "Runs a text model here, with ComfyUI's Generate Text node: nothing leaves this machine, and no key is needed. Comfy recommends "
      + COMFY_MODEL + " (6.8 GB, from " + COMFY_SOURCE + " on Hugging Face, in models/text_encoders). A shot takes seconds once the model is loaded; it waits for a render that is running." },
  { id: "openai", title: "OpenAI-compatible", address: "https://api.openai.com/v1", key: true,
    hint: "Any endpoint that speaks OpenAI Chat Completions: OpenAI, OpenRouter, LM Studio, vLLM, a llama.cpp server." },
  { id: "anthropic", title: "Anthropic", address: "https://api.anthropic.com", key: true, hint: "Anthropic's Messages API." },
  { id: "google", title: "Gemini", address: "", key: true, hint: "Google's Gemini API." },
  { id: "ollama", title: "Ollama", address: "http://localhost:11434", key: false,
    hint: "A local Ollama. It answers a web page only when OLLAMA_ORIGINS allows this address." },
];

export const FIELDS = ["action", "camera", "speaker", "dialogue", "sound", "music"];
export const LABELS = { name: "Name", action: "What happens", camera: "Camera", speaker: "Who speaks", dialogue: "What they say", sound: "Sound", music: "Music", seconds: "Seconds" };
export const LIMITS = { shots: [1, 12], seconds: [2, 8] };

export function settings() {
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem(STORE) || "{}") || {};
  } catch (_) {}
  const type = PROVIDERS.some((p) => p.id === saved.type) ? saved.type : "comfyui";
  return { type, url: String(saved.url || ""), model: String(saved.model || ""), key: String(saved.key || "") };
}

export function saveSettings(next) {
  try {
    localStorage.setItem(STORE, JSON.stringify({ type: next.type, url: next.url.trim(), model: next.model.trim(), key: next.key.trim() }));
    return true;
  } catch (_) {
    return false;
  }
}

export const provider = (type) => PROVIDERS.find((p) => p.id === type) || PROVIDERS[0];

// Whether a rewrite can be asked for: a model is named, and a key is set where the protocol needs one.
// A local OpenAI-compatible server needs no key, so only the two hosted protocols insist on it.
export function ready(given = settings()) {
  if (!given.model.trim()) return false;
  return given.key.trim() !== "" || !["anthropic", "google"].includes(given.type);
}

async function client(given) {
  const { AIModel } = await import(LIBRARY);
  const entry = { type: given.type };
  if (given.url.trim()) entry.url = given.url.trim().replace(/\/+$/, "");
  if (given.key.trim()) entry.apiKey = given.key.trim();
  // Anthropic answers a web page only when the page says it means to call from one.
  if (given.type === "anthropic") entry.headers = { "anthropic-dangerous-direct-browser-access": "true" };
  return new AIModel({ model: given.model.trim() || "none", providers: [entry], retries: 0, timeout: 180000 });
}

// What a failed call means for someone who is not reading the console.
function explain(error, given) {
  const text = String(error?.message || error || "The request failed.");
  if (/(401|403)|authentication|unauthorized|api[ _-]?key/i.test(text)) {
    return "The endpoint refused the key. Check the key and the model's name." + (text.length < 200 ? " It said: " + text : "");
  }
  if (/failed to fetch|networkerror|load failed|econnrefused|unable to connect/i.test(text)) {
    return "No answer from " + (given.url.trim() || provider(given.type).title) + ". Check the address, and that the endpoint accepts requests from a web page"
      + (given.type === "ollama" ? " (set OLLAMA_ORIGINS for Ollama)." : ".");
  }
  return text.length > 400 ? text.slice(0, 400) + "..." : text;
}

// One answer from a text model run by this ComfyUI: a small graph of its own Generate Text node, put at the front
// of the queue. The settings follow Comfy's own prompt-enhancer template, without its thinking: a 9B model asked
// to rewrite a long shot can think through all of its tokens and answer nothing, and without it answers in seconds.
async function askComfy(given, system, user) {
  const graph = {
    "1": { class_type: "CLIPLoader", inputs: { clip_name: given.model.trim(), type: "minimax", device: "default" } },
    "2": { class_type: "PrimitiveStringMultiline", inputs: { value: system } },
    "3": { class_type: "TextGenerateV2", inputs: {
      clip: ["1", 0], prompt: user, max_length: 4096, sampling_mode: "on",
      "sampling_mode.temperature": 0.7, "sampling_mode.top_k": 20, "sampling_mode.top_p": 0.95, "sampling_mode.min_p": 0,
      "sampling_mode.repetition_penalty": 1, "sampling_mode.presence_penalty": 0, "sampling_mode.seed": Math.floor(Math.random() * 2 ** 31),
      thinking: false, use_default_template: true, mtp: "auto", system_prompt: ["2", 0] } },
    "4": { class_type: "PreviewAny", inputs: { source: ["3", 0] } },
  };
  let reply;
  try {
    reply = await fetch("/prompt", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: graph, front: true }) });
  } catch (_) {
    throw new Error("ComfyUI did not answer. Is it still running?");
  }
  const sent = await reply.json().catch(() => ({}));
  if (!reply.ok || !sent.prompt_id) {
    const said = JSON.stringify(sent.node_errors || sent.error || sent);
    if (/TextGenerateV2|PrimitiveStringMultiline|PreviewAny/.test(said) && /not found|does not exist|invalid prompt/i.test(said)) {
      throw new Error("This ComfyUI has no Generate Text node. Update ComfyUI, or choose another provider.");
    }
    if (/clip_name|value_not_in_list/i.test(said)) {
      throw new Error(given.model.trim() + " is not in ComfyUI's models/text_encoders. Comfy recommends " + COMFY_MODEL + ", from " + COMFY_SOURCE + " on Hugging Face.");
    }
    throw new Error("ComfyUI refused the request: " + said.slice(0, 300));
  }
  const until = Date.now() + 15 * 60 * 1000;
  while (Date.now() < until) {
    await new Promise((done) => setTimeout(done, 1000));
    const history = await fetch("/history/" + sent.prompt_id).then((r) => r.json()).catch(() => ({}));
    const entry = history[sent.prompt_id];
    if (!entry) continue;
    if (entry.status?.status_str === "error") {
      const last = (entry.status.messages || []).find((row) => row[0] === "execution_error")?.[1];
      throw new Error("The text model failed in ComfyUI: " + String(last?.exception_message || "see ComfyUI's log").slice(0, 300));
    }
    const text = entry.outputs?.["4"]?.text?.[0];
    if (typeof text === "string") {
      if (!text.trim()) throw new Error("The model answered with nothing. Try again, or another model.");
      return text;
    }
  }
  throw new Error("The text model had not answered after 15 minutes. ComfyUI may be busy with renders.");
}

async function ask(given, system, user) {
  if (given.type === "comfyui") return askComfy(given, system, user);
  const model = await client(given);
  try {
    const reply = await model.chat([{ role: "system", content: system }, { role: "user", content: user }]);
    const content = reply?.message?.content;
    const text = typeof content === "string" ? content
      : Array.isArray(content) ? content.map((part) => (typeof part === "string" ? part : part?.text || "")).join("") : "";
    if (!text.trim()) throw new Error("The model answered with nothing. Try again, or another model.");
    return text;
  } catch (error) {
    throw new Error(explain(error, given));
  } finally {
    try { await model.dispose?.(); } catch (_) {}
  }
}

// The models an endpoint lists, for the Model field. Not every endpoint lists any.
export async function listModels(given) {
  if (given.type === "comfyui") {
    // ComfyUI's text encoders, the recommended one first. Many are encoders only and cannot write.
    const found = await fetch("/models/text_encoders").then((r) => r.json()).catch(() => null);
    if (!Array.isArray(found)) throw new Error("ComfyUI did not list its text encoders.");
    return found.filter((name) => typeof name === "string").sort((a, b) => (b === COMFY_MODEL) - (a === COMFY_MODEL) || a.localeCompare(b));
  }
  const model = await client(given);
  try {
    return (await model.getModels()).filter((name) => typeof name === "string" && name).sort();
  } catch (error) {
    throw new Error(explain(error, given));
  } finally {
    try { await model.dispose?.(); } catch (_) {}
  }
}

export async function test(given) {
  const text = await ask(given, "You answer in one short sentence.", "Say that you are ready to write video shots.");
  return text.trim().split("\n")[0].slice(0, 200);
}

// The first JSON value in a model's answer, which may come wrapped in a code fence or in a sentence.
export function parseJson(text) {
  const cleaned = String(text).replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  const starts = [cleaned.indexOf("{"), cleaned.indexOf("[")].filter((at) => at >= 0);
  if (!starts.length) throw new Error("The model did not answer in the form asked for. Try again, or another model.");
  const start = Math.min(...starts);
  const close = cleaned[start] === "{" ? "}" : "]";
  for (let end = cleaned.lastIndexOf(close); end > start; end = cleaned.lastIndexOf(close, end - 1)) {
    try {
      return JSON.parse(cleaned.slice(start, end + 1));
    } catch (_) {}
  }
  throw new Error("The model did not answer in the form asked for. Try again, or another model.");
}

const RULES = `You write shots for MiniMax H3, a model that makes one video clip with sound from a text description.

A shot has these fields:
- action: what we see and hear, in the order it happens. One continuous shot, with no cuts inside it. Plain present tense. Be concrete about the people, the place, the light and the movement. Always in English, whatever language people speak.
- camera: the framing and the camera's movement, in a few words, such as "Close-up, handheld" or "Slow dolly out".
- speaker: who talks and how, such as "The barista, cheerful". Empty when nobody speaks.
- dialogue: the exact words spoken, in the language they are spoken in. Empty when nobody speaks. At most 2.5 words for each second of the shot.
- sound: the soundscape, such as "Rain on glass, distant thunder".
- music: music under the shot. Leave it empty unless music is asked for.

Rules:
- The project's look is added to every shot by the editor. Do not repeat it in action.
- A cast member is placed by writing @Name in action, exactly as the name is given. When a cast member speaks, speaker starts with @Name. Never invent a cast name.
- Text that must appear in the picture is spelled out, such as: a sign with the word OPEN in large white capital letters. Keep it to a few short words.
- No shot lists, no numbered steps, no camera cuts, no markdown.
- Answer with JSON only, with no words before or after it.`;

const clean = (value, most = 4000) => String(value ?? "").replace(/\s+\n/g, "\n").trim().slice(0, most);

function about(project) {
  const lines = [
    "Format: " + (project.format || "16:9"),
    "Language people speak: " + (project.language || "English"),
  ];
  if (clean(project.style)) lines.push("Look, added to every shot by the editor: " + clean(project.style, 1500));
  const cast = (project.cast || []).filter((person) => clean(person.name));
  if (cast.length) {
    lines.push("Cast:");
    for (const person of cast) {
      const kind = person.file || person.refmod ? (person.voice ? "appears and has a voice" : "appears") : "a voice only, never seen";
      lines.push("- @" + clean(person.name, 80) + " (" + kind + ")" + (clean(person.description) ? ": " + clean(person.description, 600) : ""));
    }
  } else lines.push("Cast: none. Describe people in words.");
  return lines.join("\n");
}

const shotOf = (clip) => Object.fromEntries(FIELDS.map((key) => [key, clean(clip[key])]));

function fieldsFrom(value, seconds) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("The model did not answer in the form asked for. Try again, or another model.");
  const made = {};
  for (const key of FIELDS) made[key] = clean(value[key]);
  if (!made.action) throw new Error("The model left What happens empty. Try again, or another model.");
  if (seconds) {
    const wanted = Number(value.seconds);
    made.seconds = Number.isFinite(wanted) ? Math.max(LIMITS.seconds[0], Math.min(LIMITS.seconds[1], Math.round(wanted * 2) / 2)) : seconds;
  }
  return made;
}

// One clip written again. before and after are the clips on either side of it in the cut, when there are any.
export async function rewrite({ project, clip, before = null, after = null, directions = "" }, given = settings()) {
  const parts = [about(project), ""];
  if (before && clean(before.action)) parts.push("The shot before this one: " + clean(before.action, 1200), "");
  if (after && clean(after.action)) parts.push("The shot after this one: " + clean(after.action, 1200), "");
  parts.push(
    "The shot runs " + (Number(clip.seconds) || 5) + " seconds.",
    clean(clip.action) || clean(clip.dialogue) ? "The shot as it is now:\n" + JSON.stringify(shotOf(clip), null, 1) : "The shot is empty. Write it.",
    "",
    clean(directions) ? "What to change: " + clean(directions, 2000) : "Make it a stronger H3 shot. Keep what it is about, who is in it and what is said.",
    "",
    'Answer with one JSON object with exactly these keys: "action", "camera", "speaker", "dialogue", "sound", "music". Use "" for a field that stays empty.',
  );
  return fieldsFrom(parseJson(await ask(given, RULES, parts.join("\n"))), 0);
}

// The shots of a video, from a description of it.
export async function writeShots({ project, brief, count, seconds }, given = settings()) {
  const shots = Math.max(LIMITS.shots[0], Math.min(LIMITS.shots[1], Math.round(Number(count)) || 4));
  const each = Math.max(LIMITS.seconds[0], Math.min(LIMITS.seconds[1], Number(seconds) || 5));
  const parts = [
    about(project), "",
    "Write a video of " + shots + (shots === 1 ? " shot" : " shots") + ", each about " + each + " seconds, between " + LIMITS.seconds[0] + " and " + LIMITS.seconds[1] + ".",
    "Each shot is rendered by itself and knows nothing of the others. So each one describes its people and its place in full, in the same words every time, and never says \"the same\", \"again\" or \"as before\". Change the framing from one shot to the next.",
    "", "The video: " + clean(brief, 4000), "",
    'Answer with one JSON array of ' + shots + ' objects, each with exactly these keys: "name" (two or three words), "seconds" (a number), "action", "camera", "speaker", "dialogue", "sound", "music". Use "" for a field that stays empty.',
  ];
  let rows = parseJson(await ask(given, RULES, parts.join("\n")));
  if (rows && !Array.isArray(rows) && Array.isArray(rows.shots)) rows = rows.shots;
  if (!Array.isArray(rows) || !rows.length) throw new Error("The model did not answer with a list of shots. Try again, or another model.");
  return rows.slice(0, LIMITS.shots[1]).map((row, index) => ({ name: clean(row?.name, 60) || "Shot " + (index + 1), ...fieldsFrom(row, each) }));
}
