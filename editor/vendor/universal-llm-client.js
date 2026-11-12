// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/auditor.js
class NoopAuditor {
  record(_event) {}
}

class ConsoleAuditor {
  prefix;
  constructor(prefix = "[LLM]") {
    this.prefix = prefix;
  }
  record(event) {
    const parts = [
      this.prefix,
      event.type.toUpperCase(),
      event.provider ? `[${event.provider}]` : "",
      event.model ? `(${event.model})` : ""
    ].filter(Boolean);
    switch (event.type) {
      case "request":
        console.log(parts.join(" "), "→");
        break;
      case "response":
        console.log(parts.join(" "), event.duration ? `${event.duration}ms` : "", event.usage ? `${event.usage.totalTokens} tokens` : "");
        break;
      case "stream_start":
        console.log(parts.join(" "), "streaming...");
        break;
      case "stream_end":
        console.log(parts.join(" "), "done", event.duration ? `${event.duration}ms` : "");
        break;
      case "tool_call":
        console.log(parts.join(" "), event.toolExecution?.tool_call_id ?? "");
        break;
      case "tool_result":
        console.log(parts.join(" "), event.toolExecution?.error ? "❌" : "✅", event.toolExecution?.duration ? `${event.toolExecution.duration}ms` : "");
        break;
      case "error":
        console.error(parts.join(" "), event.error ?? "Unknown error");
        break;
      case "retry":
        console.warn(parts.join(" "), event.error ?? "", event.metadata ?? "");
        break;
      case "failover":
        console.warn(parts.join(" "), "→", event.metadata?.["nextProvider"] ?? "");
        break;
      case "structured_request":
        console.log(parts.join(" "), `schema=${event.schemaName ?? "unknown"}`, "→");
        break;
      case "structured_response":
        console.log(parts.join(" "), event.duration ? `${event.duration}ms` : "", `schema=${event.schemaName ?? "unknown"}`);
        break;
      case "structured_validation_error":
        console.error(parts.join(" "), `schema=${event.schemaName ?? "unknown"}`, event.error ?? "Validation failed", event.rawOutput ? `raw=${event.rawOutput.slice(0, 50)}...` : "");
        break;
    }
  }
}

class BufferedAuditor {
  events = [];
  maxBufferSize;
  onFlush;
  constructor(options = {}) {
    this.maxBufferSize = options.maxBufferSize ?? 1000;
    this.onFlush = options.onFlush;
  }
  record(event) {
    this.events.push(event);
    if (this.events.length >= this.maxBufferSize) {
      this.flush().catch(() => {});
    }
  }
  getEvents() {
    return this.events;
  }
  async flush() {
    if (this.events.length === 0)
      return;
    const batch = this.events.splice(0);
    if (this.onFlush) {
      await this.onFlush(batch);
    }
  }
  clear() {
    this.events.length = 0;
  }
}

// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/errors.js
function truncate(text, max = 500) {
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

class LLMHttpError extends Error {
  status;
  body;
  url;
  constructor(status, body, url) {
    super(`HTTP ${status}: ${truncate(body)}`);
    this.name = "LLMHttpError";
    this.status = status;
    this.body = body;
    this.url = url;
  }
}

class LLMProviderError extends Error {
  provider;
  retryable;
  constructor(provider, message, opts) {
    super(message);
    this.name = "LLMProviderError";
    this.provider = provider;
    this.retryable = opts?.retryable ?? false;
  }
}
function extractProviderErrorMessage(payload) {
  if (!payload || typeof payload !== "object")
    return null;
  const err = payload.error;
  if (typeof err === "string")
    return err.trim() ? err : null;
  if (err && typeof err === "object" && typeof err.message === "string") {
    const message = err.message;
    return message.trim() ? message : null;
  }
  return null;
}
function looksLikeErrorPayload(content) {
  const trimmed = content.trim();
  if (!(trimmed.startsWith("{") && trimmed.endsWith("}")))
    return false;
  try {
    return extractProviderErrorMessage(JSON.parse(trimmed)) !== null;
  } catch {
    return false;
  }
}
function classifyFailure(error) {
  if (error instanceof LLMHttpError) {
    if (error.status >= 500 || error.status === 408)
      return { retry: true, cooldown: false };
    if (error.status === 429 || error.status === 401 || error.status === 403 || error.status === 404) {
      return { retry: false, cooldown: true };
    }
    return { retry: false, cooldown: false };
  }
  if (error instanceof LLMProviderError) {
    return { retry: error.retryable, cooldown: !error.retryable };
  }
  if (isConnectivityFailure(error))
    return { retry: false, cooldown: true };
  return { retry: true, cooldown: false };
}
function errorCode(error) {
  if (error && typeof error === "object") {
    const e = error;
    if (typeof e.code === "string")
      return e.code;
    if (e.cause && typeof e.cause === "object" && typeof e.cause.code === "string") {
      return e.cause.code;
    }
  }
  return "";
}
function isConnectivityFailure(error) {
  const message = (error instanceof Error ? error.message : String(error)).toLowerCase();
  if (/timeout|fetch failed|failed to fetch|network|socket|unreachable|unable to connect|connection (refused|reset|closed|timed out|error)|econnrefused|econnreset|econnaborted|enotfound|ehostunreach|enetunreach|etimedout/.test(message)) {
    return true;
  }
  const code = errorCode(error).toUpperCase();
  return /^(ECONNREFUSED|ECONNRESET|ECONNABORTED|ENOTFOUND|EHOSTUNREACH|ENETUNREACH|ETIMEDOUT|CONNECTIONREFUSED|CONNECTIONCLOSED|CONNECTIONERROR)$/.test(code);
}

// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/structured-output.js
class StructuredOutputError extends Error {
  rawOutput;
  cause;
  constructor(message, options) {
    super(message);
    this.rawOutput = options.rawOutput;
    this.cause = options.cause;
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, StructuredOutputError);
    }
  }
}
function isStructuredOutputSuccess(result) {
  return result.ok === true;
}
function isStructuredOutputFailure(result) {
  return result.ok === false;
}
function normalizeJsonSchema(schema) {
  return JSON.parse(JSON.stringify(schema));
}
function getJsonSchema(options) {
  if (options.schemaConfig) {
    return normalizeJsonSchema(options.schemaConfig.jsonSchema);
  }
  if (options.jsonSchema) {
    return normalizeJsonSchema(options.jsonSchema);
  }
  throw new Error("Either schemaConfig or jsonSchema must be provided");
}
function getJsonSchemaFromConfig(config) {
  return normalizeJsonSchema(config.jsonSchema);
}
var GOOGLE_UNSUPPORTED_FEATURES = [
  "pattern",
  "minLength",
  "maxLength",
  "minimum",
  "maximum",
  "exclusiveMinimum",
  "exclusiveMaximum",
  "additionalProperties"
];
function stripUnsupportedFeatures(schema, provider) {
  if (provider !== "google") {
    return schema;
  }
  const result = JSON.parse(JSON.stringify(schema));
  for (const feature of GOOGLE_UNSUPPORTED_FEATURES) {
    delete result[feature];
  }
  if (result.properties) {
    for (const key of Object.keys(result.properties)) {
      if (result.properties[key]) {
        result.properties[key] = stripUnsupportedFeatures(result.properties[key], provider);
      }
    }
  }
  if (result["items"]) {
    if (Array.isArray(result["items"])) {
      result["items"] = result["items"].map((item) => stripUnsupportedFeatures(item, provider));
    } else {
      result["items"] = stripUnsupportedFeatures(result["items"], provider);
    }
  }
  for (const key of ["oneOf", "anyOf", "allOf"]) {
    const schemas = result[key];
    if (Array.isArray(schemas)) {
      result[key] = schemas.map((s) => stripUnsupportedFeatures(s, provider));
    }
  }
  return result;
}
function convertToProviderSchema(provider, options) {
  const jsonSchema = getJsonSchema(options);
  const schema = stripUnsupportedFeatures(jsonSchema, provider);
  const name = options.name ?? options.schemaConfig?.name ?? "response";
  return {
    schema,
    name,
    description: options.description ?? options.schemaConfig?.description
  };
}
function stripJsonFences(rawOutput) {
  if (!rawOutput.includes("```"))
    return rawOutput;
  const fenceMatch = rawOutput.match(/```(?:[a-zA-Z0-9_-]*)[ \t]*\n?([\s\S]*?)\n?[ \t]*```/);
  if (fenceMatch && fenceMatch[1] !== undefined) {
    return fenceMatch[1].trim();
  }
  return rawOutput;
}
function parseStructured(config, rawOutput) {
  let parsed;
  let parseError;
  try {
    parsed = JSON.parse(rawOutput);
  } catch (error) {
    parseError = error instanceof SyntaxError ? error : new SyntaxError(String(error));
    const fenceStripped = stripJsonFences(rawOutput);
    if (fenceStripped !== rawOutput) {
      try {
        parsed = JSON.parse(fenceStripped);
        parseError = undefined;
      } catch {}
    }
  }
  if (parseError) {
    throw new StructuredOutputError(`Failed to parse JSON: ${parseError.message}`, { rawOutput, cause: parseError });
  }
  if (config.validate) {
    try {
      return config.validate(parsed);
    } catch (error) {
      const validationError = error instanceof Error ? error : new Error(String(error));
      throw new StructuredOutputError(`Validation failed: ${validationError.message}`, { rawOutput, cause: validationError });
    }
  }
  return parsed;
}
function tryParseStructured(config, rawOutput) {
  try {
    const value = parseStructured(config, rawOutput);
    return { ok: true, value };
  } catch (error) {
    if (error instanceof StructuredOutputError) {
      return {
        ok: false,
        error,
        rawOutput
      };
    }
    throw error;
  }
}
function validateStructuredOutput(config, data, rawOutput) {
  if (config.validate) {
    try {
      return config.validate(data);
    } catch (error) {
      const rawData = rawOutput ?? JSON.stringify(data);
      const validationError = error instanceof Error ? error : new Error(String(error));
      throw new StructuredOutputError(`Validation failed: ${validationError.message}`, { rawOutput: rawData, cause: validationError });
    }
  }
  return data;
}

class StreamingJsonParser {
  buffer = "";
  validateFn;
  constructor(config) {
    this.validateFn = config.validate;
  }
  feed(chunk) {
    this.buffer += chunk;
    try {
      const parsed = JSON.parse(this.buffer);
      if (this.validateFn) {
        try {
          const validated = this.validateFn(parsed);
          return { partial: validated, complete: true };
        } catch {}
      }
      return { partial: parsed, complete: true };
    } catch {}
    const partialResult = this.tryParsePartial();
    return { partial: partialResult, complete: false };
  }
  getBuffer() {
    return this.buffer;
  }
  reset() {
    this.buffer = "";
  }
  tryParsePartial() {
    let braceCount = 0;
    let bracketCount = 0;
    let inString = false;
    let escaped = false;
    for (let i = 0;i < this.buffer.length; i++) {
      const char = this.buffer[i];
      if (escaped) {
        escaped = false;
        continue;
      }
      if (char === "\\" && inString) {
        escaped = true;
        continue;
      }
      if (char === '"') {
        inString = !inString;
        continue;
      }
      if (inString)
        continue;
      if (char === "{")
        braceCount++;
      else if (char === "}")
        braceCount--;
      else if (char === "[")
        bracketCount++;
      else if (char === "]")
        bracketCount--;
    }
    let closing = "";
    while (bracketCount > 0) {
      closing += "]";
      bracketCount--;
    }
    while (braceCount > 0) {
      closing += "}";
      braceCount--;
    }
    const candidate = this.buffer + closing;
    try {
      const parsed = JSON.parse(candidate);
      if (this.validateFn) {
        try {
          return this.validateFn(parsed);
        } catch {}
      }
      return parsed;
    } catch {}
    return;
  }
}

// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/router.js
class Router {
  providers = [];
  health = new Map;
  metrics = new Map;
  poolMetrics = new Map;
  slotWaiters = [];
  auditor;
  config;
  constructor(config = {}) {
    this.auditor = config.auditor ?? new NoopAuditor;
    this.config = {
      retriesPerProvider: config.retriesPerProvider ?? 2,
      maxFailures: config.maxFailures ?? 3,
      cooldownMs: config.cooldownMs ?? 30000,
      spillAfterMs: config.spillAfterMs ?? 750,
      coldProbeTimeoutMs: config.coldProbeTimeoutMs ?? 2500,
      coldProbeAfterMs: config.coldProbeAfterMs ?? 60000
    };
  }
  addProvider(entry) {
    this.providers.push(entry);
    this.health.set(entry.id, {
      healthy: true,
      consecutiveFailures: 0
    });
    this.metrics.set(entry.id, {
      inflight: 0,
      requests: 0,
      failures: 0
    });
    this.providers.sort((a, b) => a.priority - b.priority);
  }
  removeProvider(id) {
    this.providers = this.providers.filter((p) => p.id !== id);
    this.health.delete(id);
    this.metrics.delete(id);
  }
  setAuditor(auditor) {
    this.auditor = auditor;
  }
  getClients() {
    return this.providers.map((p) => p.client);
  }
  getStatus() {
    return this.providers.map((p) => {
      const m = this.metrics.get(p.id);
      return {
        id: p.id,
        healthy: this.isAvailable(p.id),
        active: true,
        consecutiveFailures: this.health.get(p.id)?.consecutiveFailures ?? 0,
        cooldownUntil: this.health.get(p.id)?.cooldownUntil,
        model: p.modelOverride ?? p.client.model,
        url: p.client.url,
        priority: p.priority,
        inflight: m?.inflight ?? 0,
        maxConcurrent: p.maxConcurrent,
        requests: m?.requests ?? 0,
        failures: m?.failures ?? 0,
        avgLatencyMs: m?.avgLatencyMs,
        lastLatencyMs: m?.lastLatencyMs
      };
    });
  }
  getPoolStatus() {
    const pools = new Map;
    for (const p of this.providers) {
      let pool = pools.get(p.priority);
      if (!pool) {
        const pm = this.poolMetrics.get(p.priority);
        pool = {
          priority: p.priority,
          nodes: [],
          inflight: 0,
          spills: pm?.spills ?? 0,
          queueWaits: pm?.queueWaits ?? 0
        };
        pools.set(p.priority, pool);
      }
      pool.nodes.push(p.id);
      pool.inflight += this.metrics.get(p.id)?.inflight ?? 0;
    }
    return [...pools.values()].sort((a, b) => a.priority - b.priority);
  }
  async execute(fn, context = "execute", route) {
    if (this.getAvailableProviders().length === 0) {
      throw new Error("No available LLM providers. All providers are unhealthy or in cooldown.");
    }
    let lastError;
    const tried = new Set;
    for (const priority of this.poolPriorities()) {
      while (true) {
        const provider = await this.pickFromPool(priority, route?.sessionKey, tried);
        if (!provider)
          break;
        tried.add(provider.id);
        if (this.needsColdProbe(provider) && !await this.coldNodeAlive(provider, context)) {
          lastError = new Error(`cold-node liveness probe timed out: ${provider.id}`);
          continue;
        }
        this.acquireSlot(provider.id);
        const started = Date.now();
        let disposition = { retry: true, cooldown: false };
        try {
          for (let attempt = 0;attempt <= this.config.retriesPerProvider; attempt++) {
            try {
              if (attempt > 0) {
                this.auditor.record({
                  timestamp: Date.now(),
                  type: "retry",
                  provider: provider.id,
                  model: provider.modelOverride ?? provider.client.model,
                  metadata: { attempt, context }
                });
              }
              const result = await fn(provider.client);
              this.recordSuccess(provider.id, Date.now() - started);
              return result;
            } catch (error) {
              lastError = error instanceof Error ? error : new Error(String(error));
              disposition = classifyFailure(lastError);
              this.auditor.record({
                timestamp: Date.now(),
                type: "error",
                provider: provider.id,
                model: provider.modelOverride ?? provider.client.model,
                error: lastError.message,
                metadata: { attempt, context, retryable: disposition.retry }
              });
              if (!disposition.retry)
                break;
            }
          }
        } finally {
          this.releaseSlot(provider.id);
        }
        this.recordFailure(provider.id, { cooldown: disposition.cooldown });
        const next = this.peekNextCandidate(tried);
        if (next) {
          this.auditor.record({
            timestamp: Date.now(),
            type: "failover",
            provider: provider.id,
            metadata: {
              from: provider.id,
              nextProvider: next.id,
              context,
              reason: lastError?.message
            }
          });
        }
      }
    }
    throw lastError ?? new Error("All providers failed");
  }
  async* executeStream(fn, context = "stream", route) {
    if (this.getAvailableProviders().length === 0) {
      throw new Error("No available LLM providers for streaming.");
    }
    let lastError;
    const tried = new Set;
    for (const priority of this.poolPriorities()) {
      while (true) {
        const provider = await this.pickFromPool(priority, route?.sessionKey, tried);
        if (!provider)
          break;
        tried.add(provider.id);
        if (this.needsColdProbe(provider) && !await this.coldNodeAlive(provider, context)) {
          lastError = new Error(`cold-node liveness probe timed out: ${provider.id}`);
          continue;
        }
        let emitted = false;
        this.acquireSlot(provider.id);
        const started = Date.now();
        try {
          const stream = fn(provider.client);
          let returnValue;
          while (true) {
            const result = await stream.next();
            if (result.done) {
              returnValue = result.value;
              break;
            }
            emitted = true;
            yield result.value;
          }
          this.recordSuccess(provider.id, Date.now() - started);
          return returnValue;
        } catch (error) {
          lastError = error instanceof Error ? error : new Error(String(error));
          this.recordFailure(provider.id, { cooldown: classifyFailure(lastError).cooldown });
          if (emitted) {
            this.auditor.record({
              timestamp: Date.now(),
              type: "error",
              provider: provider.id,
              error: lastError.message,
              metadata: { context, phase: "mid-stream", failedOver: false }
            });
            throw lastError;
          }
          this.auditor.record({
            timestamp: Date.now(),
            type: "failover",
            provider: provider.id,
            error: lastError.message,
            metadata: { context, phase: "pre-stream" }
          });
        } finally {
          this.releaseSlot(provider.id);
        }
      }
    }
    throw lastError ?? new Error("All providers failed for streaming");
  }
  validateOutputAndTools(_options) {}
  getSchemaFromOutput(output) {
    if (output.schema) {
      return {
        config: output.schema,
        name: output.name,
        description: output.description
      };
    }
    if (output.jsonSchema) {
      return {
        jsonSchema: output.jsonSchema,
        name: output.name,
        description: output.description
      };
    }
    throw new Error("output must have either schema or jsonSchema");
  }
  async chat(messages, options) {
    this.validateOutputAndTools(options);
    if (options?.output) {
      return this.chatWithStructuredOutput(messages, options);
    }
    return this.execute((client) => client.chat(messages, options), "chat", { sessionKey: options?.sessionKey });
  }
  async chatWithStructuredOutput(messages, options) {
    const { output } = options;
    const schemaInfo = this.getSchemaFromOutput(output);
    const schemaName = schemaInfo.name ?? "response";
    this.auditor.record({
      timestamp: Date.now(),
      type: "structured_request",
      provider: "router",
      schemaName
    });
    const { output: _, ...restOptions } = options;
    const structuredOptions = {
      ...restOptions,
      jsonSchema: "config" in schemaInfo ? getJsonSchemaFromConfig(schemaInfo.config) : schemaInfo.jsonSchema,
      schemaName: schemaInfo.name,
      schemaDescription: schemaInfo.description
    };
    const start = Date.now();
    const response = await this.execute((client) => client.chat(messages, structuredOptions), "chatWithStructuredOutput");
    if (response.message.tool_calls && response.message.tool_calls.length > 0) {
      return response;
    }
    const content = typeof response.message.content === "string" ? response.message.content : response.message.content.filter((part) => part.type === "text").map((part) => part.text).join("");
    const schemaConfig = "config" in schemaInfo ? schemaInfo.config : null;
    if (!schemaConfig || !schemaConfig.validate) {
      try {
        const structured = JSON.parse(content);
        this.auditor.record({
          timestamp: Date.now(),
          type: "structured_response",
          provider: response.provider ?? "router",
          model: response.message.role,
          duration: Date.now() - start,
          schemaName,
          usage: response.usage
        });
        return {
          ...response,
          structured
        };
      } catch (error) {
        const rawOutput = content;
        this.auditor.record({
          timestamp: Date.now(),
          type: "structured_validation_error",
          provider: response.provider ?? "router",
          schemaName,
          error: error instanceof Error ? error.message : "JSON parse failed",
          rawOutput
        });
        throw new StructuredOutputError(`Failed to parse JSON: ${rawOutput}`, { rawOutput, cause: error instanceof Error ? error : undefined });
      }
    }
    try {
      const validated = parseStructured(schemaConfig, content);
      this.auditor.record({
        timestamp: Date.now(),
        type: "structured_response",
        provider: response.provider ?? "router",
        duration: Date.now() - start,
        schemaName,
        usage: response.usage
      });
      return {
        ...response,
        structured: validated
      };
    } catch (error) {
      const rawOutput = content;
      this.auditor.record({
        timestamp: Date.now(),
        type: "structured_validation_error",
        provider: response.provider ?? "router",
        schemaName,
        error: error instanceof Error ? error.message : "Validation failed",
        rawOutput
      });
      throw error;
    }
  }
  async chatWithTools(messages, options) {
    return this.execute((client) => client.chatWithTools(messages, options), "chatWithTools", { sessionKey: options?.sessionKey });
  }
  async* chatStream(messages, options) {
    if (options?.output) {
      throw new Error('The "output" parameter is not supported with chatStream(). ' + "Use generateStructuredStream() for streaming structured output.");
    }
    return yield* this.executeStream((client) => client.chatStream(messages, options), "chatStream", { sessionKey: options?.sessionKey });
  }
  async embed(text) {
    return this.execute((client) => client.embed(text), "embed");
  }
  async embedArray(texts) {
    return this.execute((client) => client.embedArray(texts), "embedArray");
  }
  async getModels() {
    const allModels = [];
    for (const provider of this.providers) {
      try {
        const models = await provider.client.getModels();
        allModels.push(...models);
      } catch {}
    }
    return [...new Set(allModels)];
  }
  async getModelInfo() {
    return this.execute((client) => client.getModelInfo(), "getModelInfo");
  }
  async generateStructured(config, messages, options) {
    const jsonSchema = getJsonSchemaFromConfig(config);
    const schemaName = options?.schemaName ?? config.name ?? "response";
    this.auditor.record({
      timestamp: Date.now(),
      type: "structured_request",
      provider: "router",
      schemaName
    });
    const structuredOptions = {
      ...options,
      jsonSchema
    };
    const start = Date.now();
    const response = await this.execute((client) => client.chat(messages, structuredOptions), "generateStructured");
    const content = typeof response.message.content === "string" ? response.message.content : response.message.content.filter((part) => part.type === "text").map((part) => part.text).join("");
    try {
      const result = parseStructured(config, content);
      this.auditor.record({
        timestamp: Date.now(),
        type: "structured_response",
        provider: response.provider ?? "router",
        duration: Date.now() - start,
        schemaName,
        usage: response.usage
      });
      return result;
    } catch (error) {
      this.auditor.record({
        timestamp: Date.now(),
        type: "structured_validation_error",
        provider: response.provider ?? "router",
        schemaName,
        error: error instanceof Error ? error.message : "Validation failed",
        rawOutput: content
      });
      throw error;
    }
  }
  async tryParseStructured(config, messages, options) {
    try {
      const value = await this.generateStructured(config, messages, options);
      return { ok: true, value };
    } catch (error) {
      if (error instanceof Error && "rawOutput" in error) {
        return {
          ok: false,
          error,
          rawOutput: error.rawOutput
        };
      }
      throw error;
    }
  }
  async* generateStructuredStream(config, messages, options) {
    const jsonSchema = getJsonSchemaFromConfig(config);
    const schemaName = options?.schemaName ?? config.name ?? "response";
    this.auditor.record({
      timestamp: Date.now(),
      type: "structured_request",
      provider: "router",
      schemaName
    });
    const structuredOptions = {
      ...options,
      jsonSchema
    };
    const start = Date.now();
    const stream = this.executeStream((client) => client.chatStream(messages, structuredOptions), "generateStructuredStream");
    const parser = new StreamingJsonParser(config);
    let fullContent = "";
    let lastYielded;
    try {
      for await (const event of stream) {
        if (event.type !== "text")
          continue;
        fullContent += event.content;
        const result = parser.feed(event.content);
        if (result.partial !== undefined) {
          if (lastYielded === undefined || JSON.stringify(result.partial) !== JSON.stringify(lastYielded)) {
            lastYielded = result.partial;
            yield result.partial;
          }
        }
      }
      const complete = parseStructured(config, fullContent);
      this.auditor.record({
        timestamp: Date.now(),
        type: "structured_response",
        provider: "router",
        schemaName,
        duration: Date.now() - start
      });
      return complete;
    } catch (error) {
      this.auditor.record({
        timestamp: Date.now(),
        type: "structured_validation_error",
        provider: "router",
        schemaName,
        error: error instanceof Error ? error.message : "Validation failed",
        rawOutput: fullContent
      });
      throw error;
    }
  }
  registerTool(name, description, parameters, handler) {
    for (const provider of this.providers) {
      provider.client.registerTool(name, description, parameters, handler);
    }
  }
  registerTools(tools) {
    for (const provider of this.providers) {
      provider.client.registerTools(tools);
    }
  }
  isAvailable(id) {
    const h = this.health.get(id);
    if (!h)
      return false;
    if (h.healthy)
      return true;
    if (h.cooldownUntil && Date.now() >= h.cooldownUntil) {
      h.healthy = true;
      h.consecutiveFailures = 0;
      h.cooldownUntil = undefined;
      return true;
    }
    return false;
  }
  getAvailableProviders() {
    return this.providers.filter((p) => this.isAvailable(p.id));
  }
  poolPriorities() {
    return [...new Set(this.providers.map((p) => p.priority))].sort((a, b) => a - b);
  }
  inflight(id) {
    return this.metrics.get(id)?.inflight ?? 0;
  }
  hasCapacity(p) {
    return this.inflight(p.id) < (p.maxConcurrent ?? Infinity);
  }
  static affinityScore(sessionKey, nodeId) {
    let h = 2166136261;
    const s = `${sessionKey}\x00${nodeId}`;
    for (let i = 0;i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }
  poolCandidates(priority, sessionKey, tried) {
    const nodes = this.providers.filter((p) => p.priority === priority && !tried.has(p.id) && this.isAvailable(p.id));
    if (sessionKey && nodes.length > 1) {
      return [...nodes].sort((a, b) => Router.affinityScore(sessionKey, b.id) - Router.affinityScore(sessionKey, a.id));
    }
    return [...nodes].sort((a, b) => this.inflight(a.id) - this.inflight(b.id));
  }
  async pickFromPool(priority, sessionKey, tried) {
    let candidates = this.poolCandidates(priority, sessionKey, tried);
    if (candidates.length === 0)
      return null;
    const withSlot = candidates.find((p) => this.hasCapacity(p));
    if (withSlot)
      return withSlot;
    this.bumpPool(priority, "queueWaits");
    const deadline = Date.now() + this.config.spillAfterMs;
    while (Date.now() < deadline) {
      const woke = await this.waitForSlot(deadline - Date.now());
      if (!woke)
        break;
      candidates = this.poolCandidates(priority, sessionKey, tried);
      if (candidates.length === 0)
        return null;
      const freed = candidates.find((p) => this.hasCapacity(p));
      if (freed)
        return freed;
    }
    this.bumpPool(priority, "spills");
    this.auditor.record({
      timestamp: Date.now(),
      type: "failover",
      provider: `pool:${priority}`,
      metadata: {
        phase: "spill",
        reason: `pool saturated for ${this.config.spillAfterMs}ms`,
        nodes: candidates.map((p) => p.id)
      }
    });
    return null;
  }
  peekNextCandidate(tried) {
    return this.providers.find((p) => !tried.has(p.id) && this.isAvailable(p.id));
  }
  acquireSlot(id) {
    const m = this.metrics.get(id);
    if (!m)
      return;
    m.inflight++;
    m.requests++;
  }
  releaseSlot(id) {
    const m = this.metrics.get(id);
    if (m && m.inflight > 0)
      m.inflight--;
    const waiter = this.slotWaiters.shift();
    if (waiter)
      waiter();
  }
  waitForSlot(ms) {
    if (ms <= 0)
      return Promise.resolve(false);
    return new Promise((resolve) => {
      const waiter = () => {
        clearTimeout(timer);
        resolve(true);
      };
      const timer = setTimeout(() => {
        const idx = this.slotWaiters.indexOf(waiter);
        if (idx >= 0)
          this.slotWaiters.splice(idx, 1);
        resolve(false);
      }, ms);
      this.slotWaiters.push(waiter);
    });
  }
  bumpPool(priority, counter) {
    let pm = this.poolMetrics.get(priority);
    if (!pm) {
      pm = { spills: 0, queueWaits: 0 };
      this.poolMetrics.set(priority, pm);
    }
    pm[counter]++;
  }
  needsColdProbe(provider) {
    if (this.config.coldProbeTimeoutMs <= 0)
      return false;
    const h = this.health.get(provider.id);
    return Date.now() - (h?.lastSuccessAt ?? 0) >= this.config.coldProbeAfterMs;
  }
  async coldNodeAlive(provider, context) {
    let timer;
    const outcome = await Promise.race([
      provider.client.getModels().then(() => "ok", () => "error"),
      new Promise((resolve) => {
        timer = setTimeout(() => resolve("timeout"), this.config.coldProbeTimeoutMs);
      })
    ]).finally(() => clearTimeout(timer));
    if (outcome !== "timeout")
      return true;
    this.auditor.record({
      timestamp: Date.now(),
      type: "error",
      provider: provider.id,
      model: provider.modelOverride ?? provider.client.model,
      error: `cold-node liveness probe timed out after ${this.config.coldProbeTimeoutMs}ms`,
      metadata: { context, probe: true, retryable: false }
    });
    this.recordFailure(provider.id, { cooldown: true });
    return false;
  }
  recordSuccess(id, latencyMs) {
    const h = this.health.get(id);
    if (h) {
      h.healthy = true;
      h.consecutiveFailures = 0;
      h.cooldownUntil = undefined;
      h.lastSuccessAt = Date.now();
    }
    if (latencyMs !== undefined) {
      const m = this.metrics.get(id);
      if (m) {
        m.lastLatencyMs = latencyMs;
        m.avgLatencyMs = m.avgLatencyMs === undefined ? latencyMs : Math.round(m.avgLatencyMs * 0.8 + latencyMs * 0.2);
      }
    }
  }
  recordFailure(id, opts) {
    const m = this.metrics.get(id);
    if (m)
      m.failures++;
    const h = this.health.get(id);
    if (!h)
      return;
    h.consecutiveFailures++;
    h.lastFailure = Date.now();
    if (opts?.cooldown || h.consecutiveFailures >= this.config.maxFailures) {
      h.healthy = false;
      h.cooldownUntil = Date.now() + this.config.cooldownMs;
    }
  }
}

// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/tool-arg-coercion.js
function parseToolArguments(raw) {
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    return raw;
  }
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    if (!trimmed)
      return {};
    try {
      const parsed = JSON.parse(trimmed);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed;
      }
    } catch {}
  }
  return {};
}
function normalizeTypes(type) {
  if (typeof type === "string")
    return [type];
  if (Array.isArray(type))
    return type.filter((t) => typeof t === "string");
  return [];
}
function propSchema(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value;
  }
  return;
}
function coerceValue(value, types) {
  const jsType = value === null ? "null" : Array.isArray(value) ? "array" : typeof value;
  const allows = (t) => types.includes(t);
  if (types.length === 0)
    return { value, coerced: false, uncoercible: false };
  if (jsType === "array" && allows("array") || jsType === "null" && allows("null") || jsType === "object" && allows("object") || jsType === "string" && allows("string") || jsType === "number" && (allows("number") || allows("integer")) || jsType === "boolean" && allows("boolean")) {
    return { value, coerced: false, uncoercible: false };
  }
  if ((allows("number") || allows("integer")) && typeof value === "string") {
    const trimmed = value.trim();
    if (trimmed !== "" && Number.isFinite(Number(trimmed))) {
      const n = Number(trimmed);
      return { value: allows("integer") && !allows("number") ? Math.trunc(n) : n, coerced: true, uncoercible: false };
    }
  }
  if (allows("boolean")) {
    if (value === "true" || value === 1 || value === "1")
      return { value: true, coerced: true, uncoercible: false };
    if (value === "false" || value === 0 || value === "0")
      return { value: false, coerced: true, uncoercible: false };
  }
  if (allows("object") && typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return { value: parsed, coerced: true, uncoercible: false };
      }
    } catch {}
  }
  if (allows("array")) {
    if (typeof value === "string") {
      try {
        const parsed = JSON.parse(value);
        if (Array.isArray(parsed))
          return { value: parsed, coerced: true, uncoercible: false };
      } catch {}
    }
    if (value !== undefined && value !== null)
      return { value: [value], coerced: true, uncoercible: false };
  }
  if (allows("string") && (typeof value === "number" || typeof value === "boolean")) {
    return { value: String(value), coerced: true, uncoercible: false };
  }
  return { value, coerced: false, uncoercible: true };
}
function editDistance(a, b) {
  const m = a.length;
  const n = b.length;
  if (m === 0)
    return n;
  if (n === 0)
    return m;
  let prev = Array.from({ length: n + 1 }, (_, i) => i);
  let curr = new Array(n + 1);
  for (let i = 1;i <= m; i++) {
    curr[0] = i;
    for (let j = 1;j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost);
    }
    [prev, curr] = [curr, prev];
  }
  return prev[n];
}
function describeProperty(name, schema) {
  if (!schema)
    return name;
  if (schema.enum && schema.enum.length > 0) {
    return `${name} (one of ${schema.enum.map((v) => String(v)).join("|")})`;
  }
  const types = normalizeTypes(schema.type);
  return types.length > 0 ? `${name} (${types.join("|")})` : name;
}
function coerceAndValidateToolArgs(rawArgs, schema, toolName) {
  const args = parseToolArguments(rawArgs);
  const notes = [];
  const properties = schema?.properties;
  if (!properties || typeof properties !== "object") {
    return { args, corrected: false, notes };
  }
  const out = { ...args };
  let corrected = false;
  const enumViolations = [];
  const typeViolations = [];
  for (const [key, rawProp] of Object.entries(properties)) {
    if (!(key in out))
      continue;
    const prop = propSchema(rawProp);
    if (!prop)
      continue;
    const value = out[key];
    const types = normalizeTypes(prop.type);
    if (types.length > 0 && value !== undefined) {
      const c = coerceValue(value, types);
      if (c.coerced) {
        out[key] = c.value;
        corrected = true;
        notes.push(`coerced "${key}" to ${types.join("|")}`);
      } else if (c.uncoercible) {
        typeViolations.push(`${key} must be ${types.join("|")}`);
      }
    }
    if (prop.enum && prop.enum.length > 0) {
      const v = out[key];
      if (!prop.enum.some((e) => e === v)) {
        enumViolations.push(`${key} must be one of ${prop.enum.map((e) => String(e)).join("|")}`);
      }
    }
  }
  const required = schema?.required ?? [];
  const missing = required.filter((k) => {
    const v = out[k];
    return v === undefined || v === null || typeof v === "string" && v.trim() === "";
  });
  if (missing.length === 0 && enumViolations.length === 0 && typeViolations.length === 0) {
    return { args: out, corrected, notes };
  }
  const expected = Object.entries(properties).map(([name, p]) => {
    const isReq = required.includes(name);
    return `${isReq ? "" : "optional "}${describeProperty(name, propSchema(p))}`;
  }).join(", ");
  const sentKeys = Object.keys(args);
  const parts = [`Invalid arguments for ${toolName}.`];
  if (missing.length > 0)
    parts.push(`Missing required: ${missing.join(", ")}.`);
  if (enumViolations.length > 0)
    parts.push(enumViolations.join("; ") + ".");
  if (typeViolations.length > 0)
    parts.push(typeViolations.join("; ") + ".");
  parts.push(`Expected: ${expected}.`);
  parts.push(sentKeys.length > 0 ? `You sent keys: [${sentKeys.join(", ")}].` : "You sent no arguments.");
  const knownKeys = new Set(Object.keys(properties));
  const hints = [];
  for (const sent of sentKeys) {
    if (knownKeys.has(sent))
      continue;
    for (const want of missing) {
      const d = editDistance(sent.toLowerCase(), want.toLowerCase());
      if (d > 0 && d <= Math.max(1, Math.floor(want.length / 3))) {
        hints.push(`"${sent}" → "${want}"?`);
      }
    }
  }
  if (hints.length > 0)
    parts.push(`Did you mean ${hints.join(" ")}`);
  parts.push("Call the tool again with the corrected arguments.");
  return { args: out, corrected, notes, error: parts.join(" ") };
}

// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/client.js
class BaseLLMClient {
  options;
  toolRegistry = {};
  auditor;
  debug;
  constructor(options, auditor) {
    this.options = options;
    this.auditor = auditor ?? new NoopAuditor;
    this.debug = options.debug ?? false;
  }
  async embedArray(texts) {
    return Promise.all(texts.map((t) => this.embed(t)));
  }
  async getModelInfo(_modelName) {
    return { contextLength: 8192 };
  }
  sanitizeToolName(name) {
    return name.replace(/^@[^:]+:/, "").replace(/[^a-zA-Z0-9_-]/g, "_").replace(/_+/g, "_").replace(/^_|_$/g, "");
  }
  registerTool(name, description, parameters, handler) {
    const safeName = this.sanitizeToolName(name);
    this.toolRegistry[name] = {
      definition: { name: safeName, description, parameters },
      handler
    };
    if (safeName !== name && !this.toolRegistry[safeName]) {
      this.toolRegistry[safeName] = this.toolRegistry[name];
    }
    this.debugLog(`Registered tool: ${name} (LLM name: ${safeName})`);
  }
  registerTools(tools) {
    for (const tool of tools) {
      this.registerTool(tool.name, tool.description, tool.parameters, tool.handler);
    }
  }
  getToolDefinitions() {
    const seen = new Set;
    const defs = [];
    for (const { definition } of Object.values(this.toolRegistry)) {
      if (seen.has(definition.name))
        continue;
      seen.add(definition.name);
      defs.push({ type: "function", function: definition });
    }
    return defs;
  }
  async executeTool(toolCall) {
    const toolName = toolCall.function.name;
    const start = Date.now();
    let tool = this.toolRegistry[toolName];
    if (!tool) {
      const entries = Object.entries(this.toolRegistry);
      const bySuffix = entries.find(([k]) => k.endsWith(`:${toolName}`));
      if (bySuffix) {
        tool = bySuffix[1];
        this.debugLog(`Fuzzy tool match: "${toolName}" → "${bySuffix[0]}"`);
      }
      if (!tool) {
        const byPrefix = entries.filter(([k]) => k.startsWith(`${toolName}:`));
        if (byPrefix.length === 1) {
          tool = byPrefix[0][1];
          this.debugLog(`Fuzzy tool match (single): "${toolName}" → "${byPrefix[0][0]}"`);
        }
      }
    }
    if (!tool) {
      const result = {
        tool_call_id: toolCall.id,
        output: null,
        error: `Unknown tool: ${toolName}`,
        duration: Date.now() - start
      };
      this.auditor.record({
        timestamp: Date.now(),
        type: "tool_result",
        toolExecution: result,
        error: result.error
      });
      return result;
    }
    this.auditor.record({
      timestamp: Date.now(),
      type: "tool_call",
      metadata: { toolName, arguments: toolCall.function.arguments }
    });
    try {
      const coercion = coerceAndValidateToolArgs(toolCall.function.arguments, tool.definition.parameters, tool.definition.name);
      if (coercion.error) {
        const result = {
          tool_call_id: toolCall.id,
          output: null,
          error: coercion.error,
          duration: Date.now() - start
        };
        this.auditor.record({
          timestamp: Date.now(),
          type: "tool_result",
          toolExecution: result,
          error: result.error
        });
        return result;
      }
      const output = await tool.handler(coercion.args);
      const result = {
        tool_call_id: toolCall.id,
        output,
        duration: Date.now() - start
      };
      this.auditor.record({
        timestamp: Date.now(),
        type: "tool_result",
        toolExecution: result
      });
      return result;
    } catch (error) {
      const result = {
        tool_call_id: toolCall.id,
        output: null,
        error: error instanceof Error ? error.message : String(error),
        duration: Date.now() - start
      };
      this.auditor.record({
        timestamp: Date.now(),
        type: "tool_result",
        toolExecution: result,
        error: result.error
      });
      return result;
    }
  }
  async executeTools(toolCalls) {
    return Promise.all(toolCalls.map((tc) => this.executeTool(tc)));
  }
  async chatWithTools(messages, options) {
    const maxIterations = options?.maxIterations ?? 10;
    const conversationMessages = [...messages];
    const allToolExecutions = [];
    let iterations = 0;
    while (iterations < maxIterations) {
      const response = await this.chat(conversationMessages, {
        ...options,
        tools: this.getToolDefinitions()
      });
      if (!response.message.tool_calls?.length) {
        return {
          ...response,
          toolExecutions: allToolExecutions.length > 0 ? allToolExecutions : undefined
        };
      }
      conversationMessages.push(response.message);
      const toolResults = await this.executeTools(response.message.tool_calls);
      allToolExecutions.push(...toolResults);
      for (const result of toolResults) {
        const content = result.error ? result.output == null ? result.error : `${result.error}
${typeof result.output === "string" ? result.output : JSON.stringify(result.output)}` : typeof result.output === "string" ? result.output : JSON.stringify(result.output);
        conversationMessages.push({
          role: "tool",
          content,
          tool_call_id: result.tool_call_id
        });
      }
      iterations++;
    }
    const finalResponse = await this.chat(conversationMessages);
    return {
      ...finalResponse,
      toolExecutions: allToolExecutions
    };
  }
  setModel(modelName) {
    this.options.model = modelName;
    this.debugLog(`Model switched to: ${modelName}`);
  }
  get model() {
    return this.options.model;
  }
  get url() {
    return this.options.url;
  }
  setAuditor(auditor) {
    this.auditor = auditor;
  }
  debugLog(message, data) {
    if (this.debug) {
      console.log(`[LLM:${this.options.model}] ${message}`, data ?? "");
    }
  }
  generateToolCallId() {
    return `call_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
  }
  extractSchemaOptions(options) {
    if (!options)
      return null;
    if (options.schema) {
      return {
        schemaConfig: options.schema,
        name: options.schemaName,
        description: options.schemaDescription
      };
    }
    if (options.jsonSchema) {
      const config = {
        jsonSchema: options.jsonSchema
      };
      return {
        schemaConfig: config,
        name: options.schemaName,
        description: options.schemaDescription
      };
    }
    return null;
  }
  validateStructuredResponse(content, config) {
    if (!content) {
      throw new StructuredOutputError("Empty response from LLM", { rawOutput: content });
    }
    let parsed;
    try {
      parsed = JSON.parse(content);
    } catch (error) {
      const syntaxError = error instanceof SyntaxError ? error : new SyntaxError(String(error));
      throw new StructuredOutputError(`Failed to parse JSON: ${syntaxError.message}`, { rawOutput: content, cause: syntaxError });
    }
    if (config.validate) {
      try {
        config.validate(parsed);
      } catch (error) {
        const validationError = error instanceof Error ? error : new Error(String(error));
        throw new StructuredOutputError(`Validation failed: ${validationError.message}`, { rawOutput: content, cause: validationError });
      }
    }
  }
}

// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/thinking.js
var LEVELS = ["minimal", "low", "medium", "high"];
function isLevel(v) {
  return typeof v === "string" && LEVELS.includes(v);
}
function resolveThinking(perCall, config) {
  const value = perCall ?? config;
  if (value === undefined)
    return;
  if (value === false)
    return { enabled: false };
  if (value === true)
    return { enabled: true };
  if (isLevel(value))
    return { enabled: true, level: value };
  return;
}
function isOpenAIReasoningModel(model) {
  return /^(o\d|gpt-5)/i.test(model);
}
var STRICT_OPENAI_COMPAT_HOSTS = [
  "api.openai.com",
  "api.cerebras.ai",
  "api.groq.com",
  "api.fireworks.ai",
  "api.together.xyz",
  "api.together.ai",
  "api.mistral.ai",
  "api.deepseek.com",
  "api.perplexity.ai",
  "openrouter.ai",
  "api.x.ai"
];
function supportsChatTemplateKwargs(url) {
  const u = (url ?? "").toLowerCase();
  return !STRICT_OPENAI_COMPAT_HOSTS.some((host) => u.includes(host));
}
function isStrictOpenAICompatHost(url) {
  const u = (url ?? "").toLowerCase();
  return STRICT_OPENAI_COMPAT_HOSTS.some((host) => u.includes(host));
}
function geminiThinkingBudget(level) {
  switch (level) {
    case "minimal":
      return 512;
    case "low":
      return 2048;
    case "medium":
      return 8192;
    case "high":
      return 24576;
    default:
      return -1;
  }
}
function anthropicThinkingBudget(level, maxTokens) {
  const base = level === "high" ? 16384 : level === "medium" ? 4096 : level === "low" ? 1024 : level === "minimal" ? 1024 : 2048;
  return Math.max(1024, Math.min(base, maxTokens - 1024));
}

// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/http.js
async function httpRequest(url, options = {}) {
  const { method = "GET", headers = {}, body, timeout = 30000, signal } = options;
  const controller = new AbortController;
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  const combinedSignal = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;
  try {
    const response = await fetch(url, {
      method,
      headers: {
        "Content-Type": "application/json",
        ...headers
      },
      body: body ? JSON.stringify(body) : undefined,
      signal: combinedSignal
    });
    clearTimeout(timeoutId);
    if (!response.ok) {
      const errorText = await response.text().catch(() => "Unknown error");
      throw new LLMHttpError(response.status, errorText, url);
    }
    const data = await response.json();
    return {
      ok: response.ok,
      status: response.status,
      data,
      headers: response.headers
    };
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(`Request timeout after ${timeout}ms: ${url}`);
    }
    throw error;
  }
}
async function* httpStream(url, options = {}) {
  const { method = "POST", headers = {}, body, timeout = 120000, signal } = options;
  const controller = new AbortController;
  let idleTimeoutId = null;
  const resetIdleTimeout = () => {
    if (idleTimeoutId)
      clearTimeout(idleTimeoutId);
    idleTimeoutId = setTimeout(() => controller.abort(), timeout);
  };
  const clearIdleTimeout = () => {
    if (idleTimeoutId)
      clearTimeout(idleTimeoutId);
    idleTimeoutId = null;
  };
  resetIdleTimeout();
  const combinedSignal = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;
  try {
    const response = await fetch(url, {
      method,
      headers: {
        "Content-Type": "application/json",
        ...headers
      },
      body: body ? JSON.stringify(body) : undefined,
      signal: combinedSignal
    });
    if (!response.ok) {
      clearIdleTimeout();
      const errorText = await response.text().catch(() => "Unknown error");
      throw new LLMHttpError(response.status, errorText, url);
    }
    if (!response.body) {
      clearIdleTimeout();
      throw new Error("No response body for streaming");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder;
    try {
      while (true) {
        resetIdleTimeout();
        const { done, value } = await reader.read();
        if (done)
          break;
        yield decoder.decode(value, { stream: true });
      }
    } finally {
      clearIdleTimeout();
      reader.releaseLock();
    }
  } catch (error) {
    clearIdleTimeout();
    if (error instanceof Error && error.name === "AbortError") {
      if (signal?.aborted)
        throw error;
      throw new Error(`Stream idle timeout after ${timeout}ms: ${url}`);
    }
    throw error;
  }
}
async function* parseNDJSON(stream) {
  let buffer = "";
  for await (const chunk of stream) {
    buffer += chunk;
    const lines = buffer.split(`
`);
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed)
        continue;
      try {
        yield JSON.parse(trimmed);
      } catch {}
    }
  }
  if (buffer.trim()) {
    try {
      yield JSON.parse(buffer);
    } catch {}
  }
}
async function* parseSSE(stream) {
  let buffer = "";
  for await (const chunk of stream) {
    buffer += chunk;
    const events = buffer.split(`

`);
    buffer = events.pop() ?? "";
    for (const event of events) {
      const lines = event.split(`
`);
      let eventType;
      const dataLines = [];
      for (const line of lines) {
        if (line.startsWith("event:")) {
          eventType = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trim());
        }
      }
      const data = dataLines.join(`
`);
      if (data && data !== "[DONE]") {
        yield { event: eventType, data };
      }
    }
  }
}
function buildHeaders(options) {
  const headers = {
    "Content-Type": "application/json"
  };
  if (options.apiKey) {
    const headerName = options.authHeader || "Authorization";
    const defaultPrefix = headerName.toLowerCase() === "authorization" ? "Bearer " : "";
    const prefix = options.authPrefix !== undefined ? options.authPrefix : defaultPrefix;
    headers[headerName] = `${prefix}${options.apiKey}`.trim();
  }
  if (options.extraHeaders) {
    Object.assign(headers, options.extraHeaders);
  }
  return headers;
}

// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/gemma-channel.js
var GEMMA_THOUGHT_BLOCK = /<\|channel\|?>\s*'?\s*(?:thought|think)\s*'?\s*\r?\n?([\s\S]*?)<channel\|>/gi;
var GEMMA_COMPACT_THOUGHT_BLOCK = /<\|thought\s*\r?\n?([\s\S]*?)\|>/gi;
var GEMMA_BARE_MARKER = /<\|?channel\|?>?\s*'?\s*(?:thought|think|analysis|final|response|answer)?\s*'?\s*(?:\r?\n|$)/gi;
function stripGemmaChannelMarkers(text) {
  if (!text || !text.includes("channel"))
    return text;
  return text.replace(GEMMA_BARE_MARKER, "").replace(/^<channel\|>\s*/gm, "");
}
function extractGemmaThoughtChannels(input) {
  if (!input)
    return { content: input, reasoning: "", found: false };
  const reasoningParts = [];
  let found = false;
  const content = input.replace(GEMMA_THOUGHT_BLOCK, (_match, thought) => {
    found = true;
    const normalized = normalizeGemmaThought(thought);
    if (normalized)
      reasoningParts.push(normalized);
    return "";
  }).replace(GEMMA_COMPACT_THOUGHT_BLOCK, (_match, thought) => {
    found = true;
    const normalized = normalizeGemmaThought(thought);
    if (normalized)
      reasoningParts.push(normalized);
    return "";
  });
  return {
    content,
    reasoning: reasoningParts.join(`

`),
    found
  };
}
function normalizeGemmaThought(thought) {
  return thought.replace(/^\s+/, "").replace(/\s+$/, "");
}

// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/gemma-diffusion.js
function isGemmaDiffusionModel(model) {
  return /diffusion[-_]?gemma/i.test(model);
}
var TOOL_CALL_BLOCK = /<\|tool_call>\s*call:([a-zA-Z0-9_.-]+)\s*(?:\(\s*)?\{([\s\S]*?)\}(?:\s*\))?\s*<tool_call\|>/g;
var RESIDUAL_SPECIAL = /<\|?(?:turn|think|image|audio|video|tool_response|tool_call|tool|channel)\b[^>]*?\|?>|<(?:turn|channel|tool_response|tool_call|tool)\|>/g;
var QUOTE_TOKEN = '<|"|>';
function gemmaArgsToJson(body) {
  const src = `{${body}}`;
  let i = 0;
  const n = src.length;
  function skipWs() {
    while (i < n && /\s/.test(src[i]))
      i++;
  }
  function parseQuoted() {
    i += QUOTE_TOKEN.length;
    const end = src.indexOf(QUOTE_TOKEN, i);
    const raw = end === -1 ? src.slice(i) : src.slice(i, end);
    i = end === -1 ? n : end + QUOTE_TOKEN.length;
    return raw;
  }
  function parseStringQuote() {
    const quote = src[i];
    i++;
    let out = "";
    let escaped = false;
    while (i < n) {
      const ch = src[i];
      i++;
      if (escaped) {
        out += ch;
        escaped = false;
      } else if (ch === "\\") {
        escaped = true;
      } else if (ch === quote) {
        break;
      } else {
        out += ch;
      }
    }
    return out;
  }
  function parseBare(stops) {
    const start = i;
    while (i < n && !stops.includes(src[i]) && !src.startsWith(QUOTE_TOKEN, i))
      i++;
    return src.slice(start, i).trim();
  }
  function parseValue() {
    skipWs();
    if (src.startsWith(QUOTE_TOKEN, i))
      return JSON.stringify(parseQuoted());
    const c = src[i];
    if (c === '"' || c === "'")
      return JSON.stringify(parseStringQuote());
    if (c === "{")
      return parseObject();
    if (c === "[")
      return parseArray();
    const bare = parseBare(",}]");
    if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(bare))
      return bare;
    if (bare === "true" || bare === "false" || bare === "null")
      return bare;
    return JSON.stringify(bare);
  }
  function parseObject() {
    i++;
    const parts = [];
    skipWs();
    while (i < n && src[i] !== "}") {
      skipWs();
      const key = src.startsWith(QUOTE_TOKEN, i) ? parseQuoted() : parseBare(":");
      skipWs();
      if (src[i] === ":")
        i++;
      const value = parseValue();
      parts.push(`${JSON.stringify(key.trim())}:${value}`);
      skipWs();
      if (src[i] === ",")
        i++;
      skipWs();
    }
    i++;
    return `{${parts.join(",")}}`;
  }
  function parseArray() {
    i++;
    const parts = [];
    skipWs();
    while (i < n && src[i] !== "]") {
      parts.push(parseValue());
      skipWs();
      if (src[i] === ",")
        i++;
      skipWs();
    }
    i++;
    return `[${parts.join(",")}]`;
  }
  skipWs();
  return parseObject();
}
function parseGemmaToolCallBody(body) {
  const m = body.match(/^\s*call:([a-zA-Z0-9_.-]+)\s*\{([\s\S]*)\}\s*$/) ?? body.match(/^\s*call:([a-zA-Z0-9_.-]+)\s*\(\s*\{([\s\S]*)\}\s*\)\s*$/);
  if (!m)
    return null;
  return { name: m[1], argumentsJson: gemmaInnerArgsToJson(m[2]) };
}
function gemmaInnerArgsToJson(inner) {
  try {
    return JSON.stringify(JSON.parse(`{${inner}}`));
  } catch {
    return gemmaArgsToJson(inner);
  }
}
function parseGemmaDiffusionOutput(raw) {
  if (!raw)
    return { content: raw, reasoning: "", toolCalls: [] };
  const toolCalls = [];
  let text = raw.replace(TOOL_CALL_BLOCK, (_m, name, args) => {
    toolCalls.push({ name, argumentsJson: gemmaInnerArgsToJson(args) });
    return "";
  });
  const channels = extractGemmaThoughtChannels(text);
  text = channels.content;
  let reasoning = channels.reasoning;
  const danglingThought = text.match(/<\|channel>\s*thought\s*\r?\n?([\s\S]*)$/i);
  if (danglingThought) {
    reasoning = reasoning ? `${reasoning}

${danglingThought[1].trim()}` : danglingThought[1].trim();
    text = text.slice(0, danglingThought.index);
  }
  text = text.replace(RESIDUAL_SPECIAL, "");
  return {
    content: text.trim(),
    reasoning,
    toolCalls
  };
}

// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/stream-decoder.js
class PassthroughDecoder {
  content = "";
  callback;
  constructor(callback) {
    this.callback = callback;
  }
  push(token) {
    this.content += token;
    this.callback({ type: "text", content: token });
  }
  flush() {}
  getCleanContent() {
    return this.content;
  }
  getReasoning() {
    return;
  }
}
function parseJsonObject(text) {
  try {
    const parsed = JSON.parse(text);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed;
    }
  } catch {}
  return null;
}
function parseLooseArguments(rawArgs) {
  const trimmed = rawArgs.trim();
  if (!trimmed)
    return "{}";
  const jsonArgs = parseJsonObject(trimmed);
  if (jsonArgs)
    return JSON.stringify(jsonArgs);
  const unwrapped = trimmed.startsWith("{") && trimmed.endsWith("}") ? trimmed.slice(1, -1).trim() : trimmed;
  try {
    return gemmaArgsToJson(unwrapped);
  } catch {
    return null;
  }
}
function suffixThatPrefixes(value, prefix) {
  const max = Math.min(value.length, prefix.length - 1);
  for (let len = max;len > 0; len--) {
    const suffix = value.slice(-len);
    if (prefix.startsWith(suffix))
      return suffix;
  }
  return "";
}
function parseBareCallAtStart(text) {
  if (!text.startsWith("call:"))
    return { status: "invalid" };
  const nameStart = "call:".length;
  if (text.length <= nameStart)
    return { status: "incomplete" };
  const first = text[nameStart];
  if (!/[@A-Za-z_]/u.test(first))
    return { status: "invalid" };
  let pos = nameStart + 1;
  while (pos < text.length && /[@A-Za-z0-9_.:-]/u.test(text[pos]))
    pos++;
  const name = text.slice(nameStart, pos);
  while (pos < text.length && /\s/u.test(text[pos]))
    pos++;
  if (pos >= text.length)
    return { status: "incomplete" };
  const opener = text[pos];
  if (opener !== "(" && opener !== "{")
    return { status: "invalid" };
  const primaryClose = opener === "(" ? ")" : "}";
  const toleratedClose = opener === "(" ? "}" : undefined;
  let depth = 0;
  let quote = null;
  let escaped = false;
  for (let i = pos;i < text.length; i++) {
    const ch = text[i];
    if (quote) {
      if (escaped) {
        escaped = false;
      } else if (ch === "\\") {
        escaped = true;
      } else if (ch === quote) {
        quote = null;
      }
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
    } else if (ch === opener) {
      depth++;
    } else if (ch === primaryClose) {
      depth--;
      if (depth === 0) {
        const segment = text.slice(0, i + 1);
        const rawArgs = text.slice(pos + 1, i).trim();
        const argumentsJson = parseLooseArguments(rawArgs);
        if (!argumentsJson)
          return { status: "invalid" };
        return { status: "complete", segment, name, argumentsJson };
      }
    } else if (depth === 1 && toleratedClose !== undefined && ch === toleratedClose) {
      const rawArgs = text.slice(pos + 1, i).trim();
      if (rawArgs.startsWith("{"))
        continue;
      const argumentsJson = parseLooseArguments(rawArgs);
      if (!argumentsJson)
        continue;
      return { status: "complete", segment: text.slice(0, i + 1), name, argumentsJson };
    }
  }
  return { status: "incomplete" };
}
function checkOpenerMatch(candidate) {
  if ("<progress>".startsWith(candidate)) {
    return {
      matches: true,
      fullMatch: candidate === "<progress>" ? { type: "progress", close: "</progress>" } : undefined
    };
  }
  if ("<think>".startsWith(candidate)) {
    return {
      matches: true,
      fullMatch: candidate === "<think>" ? { type: "think", close: "</think>" } : undefined
    };
  }
  if ("<thinking>".startsWith(candidate)) {
    return {
      matches: true,
      fullMatch: candidate === "<thinking>" ? { type: "think", close: "</thinking>" } : undefined
    };
  }
  if ("<tool_call|>".startsWith(candidate)) {
    return {
      matches: true,
      fullMatch: candidate === "<tool_call|>" ? { type: "tool-call", close: "<|tool_response>" } : undefined
    };
  }
  if ("<|tool_response>".startsWith(candidate)) {
    return {
      matches: true,
      fullMatch: candidate === "<|tool_response>" ? { type: "tool-response", close: "" } : undefined
    };
  }
  if ("<|tool_call>".startsWith(candidate)) {
    return {
      matches: true,
      fullMatch: candidate === "<|tool_call>" ? { type: "gemma-tool-call", close: "<tool_call|>" } : undefined
    };
  }
  if ("<channel|>".startsWith(candidate)) {
    return {
      matches: true,
      fullMatch: candidate === "<channel|>" ? { type: "discard", close: "" } : undefined
    };
  }
  if ("<turn|>".startsWith(candidate)) {
    return {
      matches: true,
      fullMatch: candidate === "<turn|>" ? { type: "discard", close: "" } : undefined
    };
  }
  if ("<|thought".startsWith(candidate)) {
    return {
      matches: true,
      fullMatch: candidate === "<|thought" ? { type: "gemma-thought", close: "|>" } : undefined
    };
  }
  if (candidate.startsWith("<")) {
    if ("<|channel>".startsWith(candidate)) {
      return { matches: true };
    }
    if (candidate.startsWith("<|channel>")) {
      const suffix = candidate.slice("<|channel>".length);
      const suffixRegex = /^\s*(t(h(o(u(g(h(t)?)?)?)?)?)?)?$/i;
      if (suffixRegex.test(suffix)) {
        const isFullMatch = /^\s*thought$/i.test(suffix);
        return {
          matches: true,
          fullMatch: isFullMatch ? { type: "gemma-thought", close: "<channel|>" } : undefined
        };
      }
    }
  }
  return { matches: false };
}

class StandardChatDecoder {
  content = "";
  reasoning = "";
  callback;
  knownToolNames;
  tagBuffer = "";
  pendingText = "";
  pendingReasoning = "";
  inProgressTag = false;
  progressBody = "";
  inGemmaThought = false;
  gemmaThoughtBody = "";
  gemmaThoughtClose = "";
  inThinkTag = false;
  thinkBody = "";
  thinkClose = "";
  inToolCallTag = false;
  toolCallBody = "";
  toolCallClose = "";
  toolCallKind = "json";
  constructor(callback, options = {}) {
    this.callback = callback;
    this.knownToolNames = options.knownToolNames;
  }
  push(token) {
    let pos = 0;
    while (pos < token.length) {
      if (this.inGemmaThought) {
        this.gemmaThoughtBody += token.slice(pos);
        const closeIdx = this.gemmaThoughtBody.indexOf(this.gemmaThoughtClose);
        if (closeIdx !== -1) {
          const body = this.gemmaThoughtBody.slice(0, closeIdx);
          const remainder = this.gemmaThoughtBody.slice(closeIdx + this.gemmaThoughtClose.length);
          this.pushDecodedReasoning(normalizeGemmaThought(body));
          this.inGemmaThought = false;
          this.gemmaThoughtBody = "";
          this.gemmaThoughtClose = "";
          if (remainder)
            this.push(remainder);
        }
        return;
      }
      if (this.inThinkTag) {
        this.thinkBody += token.slice(pos);
        const closeIdx = this.thinkBody.indexOf(this.thinkClose);
        if (closeIdx !== -1) {
          const body = this.thinkBody.slice(0, closeIdx);
          const remainder = this.thinkBody.slice(closeIdx + this.thinkClose.length);
          this.pushDecodedReasoning(normalizeGemmaThought(body));
          this.inThinkTag = false;
          this.thinkBody = "";
          this.thinkClose = "";
          if (remainder)
            this.push(remainder);
        }
        return;
      }
      if (this.inToolCallTag) {
        this.toolCallBody += token.slice(pos);
        const closeIdx = this.toolCallBody.indexOf(this.toolCallClose);
        if (closeIdx !== -1) {
          const body = this.toolCallBody.slice(0, closeIdx);
          const remainder = this.toolCallBody.slice(closeIdx + this.toolCallClose.length);
          if (this.toolCallKind === "gemma") {
            const parsed = parseGemmaToolCallBody(body);
            if (parsed) {
              this.callback({
                type: "tool_call",
                calls: [
                  {
                    id: `gemma_${Date.now()}_${Math.random().toString(36).slice(2)}`,
                    type: "function",
                    function: { name: parsed.name, arguments: parsed.argumentsJson }
                  }
                ]
              });
            }
          } else if (body.trim()) {
            try {
              const normalizedJson = body.trim().replace(/'/g, '"').replace(/True/g, "true").replace(/False/g, "false").replace(/None/g, "null");
              const parsed = JSON.parse(normalizedJson);
              const calls = Array.isArray(parsed) ? parsed : [parsed];
              const validatedCalls = [];
              for (const call of calls) {
                if (call && typeof call === "object" && call.name) {
                  validatedCalls.push({
                    id: call.id || `recovered_${Date.now()}_${Math.random().toString(36).slice(2)}`,
                    type: "function",
                    function: {
                      name: call.name,
                      arguments: typeof call.arguments === "string" ? call.arguments : JSON.stringify(call.arguments ?? {})
                    }
                  });
                }
              }
              if (validatedCalls.length > 0) {
                this.callback({ type: "tool_call", calls: validatedCalls });
              }
            } catch {}
          }
          this.inToolCallTag = false;
          this.toolCallBody = "";
          this.toolCallClose = "";
          if (remainder)
            this.push(remainder);
        }
        return;
      }
      if (this.inProgressTag) {
        this.progressBody += token.slice(pos);
        const closeIdx = this.progressBody.indexOf("</progress>");
        if (closeIdx !== -1) {
          const body = this.progressBody.slice(0, closeIdx);
          const remainder = this.progressBody.slice(closeIdx + "</progress>".length);
          if (body) {
            this.callback({ type: "progress", content: body });
          }
          this.inProgressTag = false;
          this.progressBody = "";
          if (remainder)
            this.push(remainder);
        }
        return;
      }
      if (this.tagBuffer.length > 0) {
        const ch = token[pos];
        pos++;
        this.tagBuffer += ch;
        const matchResult = checkOpenerMatch(this.tagBuffer);
        if (matchResult.matches) {
          if (matchResult.fullMatch) {
            const fm = matchResult.fullMatch;
            if (fm.type === "progress") {
              this.inProgressTag = true;
              this.progressBody = "";
            } else if (fm.type === "think") {
              this.inThinkTag = true;
              this.thinkBody = "";
              this.thinkClose = fm.close;
            } else if (fm.type === "gemma-thought") {
              this.inGemmaThought = true;
              this.gemmaThoughtBody = "";
              this.gemmaThoughtClose = fm.close;
            } else if (fm.type === "tool-call") {
              this.inToolCallTag = true;
              this.toolCallBody = "";
              this.toolCallClose = fm.close;
              this.toolCallKind = "json";
            } else if (fm.type === "gemma-tool-call") {
              this.inToolCallTag = true;
              this.toolCallBody = "";
              this.toolCallClose = fm.close;
              this.toolCallKind = "gemma";
            }
            this.tagBuffer = "";
          }
        } else {
          this.pushDecodedText(this.tagBuffer);
          this.tagBuffer = "";
        }
        continue;
      }
      const ltIdx = token.indexOf("<", pos);
      if (ltIdx === -1) {
        this.pushDecodedText(token.slice(pos));
        return;
      }
      if (ltIdx > pos) {
        this.pushDecodedText(token.slice(pos, ltIdx));
      }
      this.tagBuffer = "<";
      pos = ltIdx + 1;
    }
  }
  pushDecodedText(text) {
    this.pendingText += text;
    this.drainBareCallBuffer("text", false);
  }
  pushDecodedReasoning(text) {
    this.pendingReasoning += text;
    this.drainBareCallBuffer("reasoning", false);
  }
  emitVisibleText(text) {
    if (!text)
      return;
    this.content += text;
    this.callback({ type: "text", content: text });
  }
  emitReasoningText(content) {
    if (!content)
      return;
    this.reasoning += content;
    this.callback({ type: "thinking", content });
  }
  emitRecoveredBareCall(name, argumentsJson) {
    this.callback({
      type: "tool_call",
      calls: [
        {
          id: `recovered_${Date.now()}_${Math.random().toString(36).slice(2)}`,
          type: "function",
          function: { name, arguments: argumentsJson }
        }
      ]
    });
  }
  drainBareCallBuffer(kind, final) {
    const emit = kind === "text" ? (text) => this.emitVisibleText(text) : (text) => this.emitReasoningText(text);
    const knownToolNames = this.knownToolNames;
    if (!knownToolNames?.size) {
      const buffered = kind === "text" ? this.pendingText : this.pendingReasoning;
      if (kind === "text")
        this.pendingText = "";
      else
        this.pendingReasoning = "";
      emit(buffered);
      return;
    }
    let buffer = kind === "text" ? this.pendingText : this.pendingReasoning;
    while (buffer.length > 0) {
      const callIdx = buffer.indexOf("call:");
      if (callIdx < 0) {
        const keep = final ? "" : suffixThatPrefixes(buffer, "call:");
        const ready = keep ? buffer.slice(0, -keep.length) : buffer;
        emit(ready);
        buffer = keep;
        break;
      }
      if (callIdx > 0) {
        emit(buffer.slice(0, callIdx));
        buffer = buffer.slice(callIdx);
        continue;
      }
      const parsed = parseBareCallAtStart(buffer);
      if (parsed.status === "incomplete") {
        if (!final)
          break;
        emit(buffer);
        buffer = "";
        break;
      }
      if (parsed.status === "invalid" || !parsed.segment || !parsed.name || !parsed.argumentsJson) {
        emit(buffer.slice(0, "call:".length));
        buffer = buffer.slice("call:".length);
        continue;
      }
      if (knownToolNames.has(parsed.name)) {
        this.emitRecoveredBareCall(parsed.name, parsed.argumentsJson);
      } else {
        emit(parsed.segment);
      }
      buffer = buffer.slice(parsed.segment.length);
    }
    if (kind === "text")
      this.pendingText = buffer;
    else
      this.pendingReasoning = buffer;
  }
  pushReasoning(content) {
    this.pushDecodedReasoning(content);
  }
  pushToolCalls(calls) {
    this.callback({ type: "tool_call", calls });
  }
  flush() {
    if (this.tagBuffer) {
      this.pushDecodedText(this.tagBuffer);
      this.tagBuffer = "";
    }
    this.drainBareCallBuffer("text", true);
    if (this.inGemmaThought) {
      this.pushDecodedReasoning(normalizeGemmaThought(this.gemmaThoughtBody));
      this.inGemmaThought = false;
      this.gemmaThoughtBody = "";
      this.gemmaThoughtClose = "";
    }
    if (this.inThinkTag) {
      this.pushDecodedReasoning(normalizeGemmaThought(this.thinkBody));
      this.inThinkTag = false;
      this.thinkBody = "";
      this.thinkClose = "";
    }
    if (this.inProgressTag) {
      if (this.progressBody) {
        this.pushDecodedText("<progress>" + this.progressBody);
      }
      this.inProgressTag = false;
      this.progressBody = "";
    }
    this.drainBareCallBuffer("reasoning", true);
    if (this.inToolCallTag) {
      this.inToolCallTag = false;
      this.toolCallBody = "";
      this.toolCallClose = "";
    }
  }
  getCleanContent() {
    return this.content;
  }
  getReasoning() {
    return this.reasoning || undefined;
  }
}

class InterleavedReasoningDecoder {
  buffer = "";
  content = "";
  reasoning = "";
  callback;
  inThink = false;
  inProgress = false;
  constructor(callback) {
    this.callback = callback;
  }
  push(token) {
    this.buffer += token;
    this.processBuffer();
  }
  flush() {
    if (this.buffer.length > 0) {
      if (this.inThink) {
        this.reasoning += this.buffer;
        this.callback({ type: "thinking", content: this.buffer });
      } else if (this.inProgress) {
        this.callback({ type: "progress", content: this.buffer });
      } else {
        this.content += this.buffer;
        this.callback({ type: "text", content: this.buffer });
      }
      this.buffer = "";
    }
  }
  getCleanContent() {
    return this.content;
  }
  getReasoning() {
    return this.reasoning || undefined;
  }
  processBuffer() {
    let safety = 0;
    while (this.buffer.length > 0 && safety++ < 200) {
      if (this.inThink) {
        const closeIdx = this.buffer.indexOf("</think>");
        if (closeIdx === -1) {
          if (this.buffer.endsWith("<") || this.buffer.endsWith("</") || this.buffer.endsWith("</t") || this.buffer.endsWith("</th") || this.buffer.endsWith("</thi") || this.buffer.endsWith("</thin") || this.buffer.endsWith("</think")) {
            return;
          }
          this.reasoning += this.buffer;
          this.callback({ type: "thinking", content: this.buffer });
          this.buffer = "";
          return;
        }
        const thinkContent = this.buffer.slice(0, closeIdx);
        if (thinkContent) {
          this.reasoning += thinkContent;
          this.callback({ type: "thinking", content: thinkContent });
        }
        this.buffer = this.buffer.slice(closeIdx + 8);
        this.inThink = false;
        continue;
      }
      if (this.inProgress) {
        const closeIdx = this.buffer.indexOf("</progress>");
        if (closeIdx === -1) {
          if (this.couldBePartialTag(this.buffer, "</progress>"))
            return;
          this.callback({ type: "progress", content: this.buffer });
          this.buffer = "";
          return;
        }
        const progressContent = this.buffer.slice(0, closeIdx);
        if (progressContent) {
          this.callback({ type: "progress", content: progressContent });
        }
        this.buffer = this.buffer.slice(closeIdx + 11);
        this.inProgress = false;
        continue;
      }
      const thinkIdx = this.buffer.indexOf("<think>");
      const progressIdx = this.buffer.indexOf("<progress>");
      const nextTag = this.findEarliest(thinkIdx, progressIdx);
      if (nextTag === -1) {
        const lastAngle = this.buffer.lastIndexOf("<");
        if (lastAngle >= 0 && lastAngle > this.buffer.length - 12) {
          const textBefore = this.buffer.slice(0, lastAngle);
          if (textBefore) {
            this.content += textBefore;
            this.callback({ type: "text", content: textBefore });
          }
          this.buffer = this.buffer.slice(lastAngle);
          return;
        }
        this.content += this.buffer;
        this.callback({ type: "text", content: this.buffer });
        this.buffer = "";
        return;
      }
      const textBefore = this.buffer.slice(0, nextTag);
      if (textBefore) {
        this.content += textBefore;
        this.callback({ type: "text", content: textBefore });
      }
      if (nextTag === thinkIdx) {
        this.buffer = this.buffer.slice(nextTag + 7);
        this.inThink = true;
      } else {
        this.buffer = this.buffer.slice(nextTag + 10);
        this.inProgress = true;
      }
    }
  }
  findEarliest(a, b) {
    if (a === -1)
      return b;
    if (b === -1)
      return a;
    return Math.min(a, b);
  }
  couldBePartialTag(buffer, tag) {
    for (let i = 1;i < tag.length; i++) {
      if (buffer.endsWith(tag.slice(0, i)))
        return true;
    }
    return false;
  }
}
var decoderRegistry = new Map;
function registerDecoder(type, factory) {
  decoderRegistry.set(type, factory);
}
function getRegisteredDecoders() {
  return Array.from(decoderRegistry.keys());
}
registerDecoder("passthrough", (cb) => new PassthroughDecoder(cb));
registerDecoder("standard-chat", (cb, options) => new StandardChatDecoder(cb, options));
registerDecoder("interleaved-reasoning", (cb) => new InterleavedReasoningDecoder(cb));
function createDecoder(type, callback, options) {
  const factory = decoderRegistry.get(type);
  if (!factory) {
    const available = Array.from(decoderRegistry.keys()).join(", ");
    throw new Error(`Unknown decoder type: "${type}". Available: ${available}`);
  }
  return factory(callback, options);
}

// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/stream-guard.js
class StreamLoopGuard {
  tail = "";
  totalChars = 0;
  pushes = 0;
  detected = null;
  checkIntervalPushes;
  minRepeats;
  maxPatternLen;
  tailWindow;
  minLoopSpan;
  maxChars;
  minParagraphRepeats;
  constructor(options = {}) {
    this.checkIntervalPushes = options.checkIntervalPushes ?? 50;
    this.minRepeats = options.minRepeats ?? 8;
    this.maxPatternLen = options.maxPatternLen ?? 192;
    this.tailWindow = options.tailWindow ?? 2048;
    this.minLoopSpan = options.minLoopSpan ?? 600;
    this.maxChars = options.maxChars ?? 400000;
    this.minParagraphRepeats = options.minParagraphRepeats ?? 4;
  }
  get detection() {
    return this.detected;
  }
  push(text) {
    if (this.detected)
      return this.detected;
    if (!text)
      return null;
    this.totalChars += text.length;
    this.tail = (this.tail + text).slice(-this.tailWindow);
    this.pushes++;
    if (this.totalChars > this.maxChars) {
      this.detected = { reason: "max_chars", totalChars: this.totalChars };
      return this.detected;
    }
    if (this.pushes % this.checkIntervalPushes !== 0)
      return null;
    const found = this.detectRepetition() ?? this.detectParagraphLoop();
    if (found)
      this.detected = found;
    return this.detected;
  }
  detectParagraphLoop() {
    const paragraphs = this.tail.split(/\n{2,}/).map((p) => p.trim()).filter((p) => p.length >= 8);
    if (paragraphs.length < this.minParagraphRepeats)
      return null;
    let bestRun = 1;
    let bestParagraph = "";
    let run = 1;
    for (let i = 1;i < paragraphs.length; i++) {
      run = paragraphs[i] === paragraphs[i - 1] ? run + 1 : 1;
      if (run > bestRun) {
        bestRun = run;
        bestParagraph = paragraphs[i];
      }
    }
    if (bestRun >= this.minParagraphRepeats && bestRun * bestParagraph.length >= Math.min(this.minLoopSpan, 240)) {
      return {
        reason: "paragraph_loop",
        pattern: bestParagraph.slice(0, 40),
        repeats: bestRun,
        totalChars: this.totalChars
      };
    }
    return null;
  }
  detectRepetition() {
    const tail = this.tail;
    if (tail.length < this.minLoopSpan)
      return null;
    return detectCharPatternLoop(tail, {
      minRepeats: this.minRepeats,
      maxPatternLen: this.maxPatternLen,
      minLoopSpan: this.minLoopSpan,
      totalChars: this.totalChars
    });
  }
}
function detectCharPatternLoop(tail, opts) {
  for (let pLen = 3;pLen <= Math.min(opts.maxPatternLen, Math.floor(tail.length / opts.minRepeats)); pLen++) {
    const pattern = tail.slice(-pLen);
    let repeats = 0;
    let pos = tail.length - pLen;
    while (pos >= 0 && tail.slice(pos, pos + pLen) === pattern) {
      repeats++;
      pos -= pLen;
    }
    if (repeats >= opts.minRepeats && repeats * pLen >= opts.minLoopSpan) {
      return {
        reason: "repetition",
        pattern: pattern.trim().slice(0, 40),
        repeats,
        totalChars: opts.totalChars
      };
    }
  }
  return null;
}
function collapseRepeatedRuns(text, minRun = 3) {
  let collapsed = 0;
  const collapseUnits = (units, joiner) => {
    const out = [];
    let i = 0;
    while (i < units.length) {
      const unit = units[i];
      let run = 1;
      while (i + run < units.length && units[i + run].trim() === unit.trim() && unit.trim().length >= 8) {
        run++;
      }
      if (run >= minRun) {
        out.push(unit, `[… repeated ${run}× — collapsed]`);
        collapsed += run - 1;
      } else {
        for (let k = 0;k < run; k++)
          out.push(units[i + k]);
      }
      i += run;
    }
    return out;
  };
  const paragraphs = collapseUnits(text.split(/\n{2,}/), `

`).join(`

`);
  const lines = collapseUnits(paragraphs.split(`
`), `
`).join(`
`);
  return { text: lines, collapsed };
}
function collapseRepeatsInToolArguments(argsJson) {
  try {
    const parsed = JSON.parse(argsJson);
    let collapsed = 0;
    const walk = (value) => {
      if (typeof value === "string" && value.length >= 200) {
        const result = collapseRepeatedRuns(value);
        collapsed += result.collapsed;
        return result.text;
      }
      if (Array.isArray(value))
        return value.map(walk);
      if (value && typeof value === "object") {
        const out = {};
        for (const [k, v] of Object.entries(value))
          out[k] = walk(v);
        return out;
      }
      return value;
    };
    const rewritten = walk(parsed);
    return collapsed > 0 ? { argsJson: JSON.stringify(rewritten), collapsed } : { argsJson, collapsed: 0 };
  } catch {
    return { argsJson, collapsed: 0 };
  }
}

// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/providers/ollama.js
class OllamaClient extends BaseLLMClient {
  constructor(options, auditor) {
    super({
      ...options,
      url: (options.url || "http://localhost:11434").replace(/\/+$/, "")
    }, auditor);
  }
  async chat(messages, options) {
    const url = `${this.options.url}/api/chat`;
    const model = options?.model ?? this.options.model;
    const tools = options?.tools ?? (Object.keys(this.toolRegistry).length > 0 ? this.getToolDefinitions() : undefined);
    const body = {
      model,
      messages: this.convertMessages(messages),
      stream: false,
      options: this.buildOllamaOptions(options)
    };
    this.hoistKeepAlive(body);
    if (tools?.length) {
      body["tools"] = this.convertToolsToOllama(tools);
    }
    body["think"] = resolveThinking(options?.thinking, this.options.thinking)?.enabled ?? true;
    const schemaOptions = this.extractSchemaOptions(options);
    if (schemaOptions) {
      body["format"] = this.buildFormatParameter(schemaOptions);
    } else if (options?.responseFormat) {
      body["format"] = "json";
    }
    const start = Date.now();
    this.auditor.record({
      timestamp: start,
      type: "request",
      provider: "ollama",
      model
    });
    const response = await httpRequest(url, {
      method: "POST",
      headers: buildHeaders(this.options),
      body,
      timeout: this.options.timeout ?? 30000
    });
    const data = response.data;
    const providerError = extractProviderErrorMessage(data);
    if (providerError) {
      throw new LLMProviderError("ollama", `Ollama error: ${providerError}`);
    }
    if (!data.message) {
      throw new LLMProviderError("ollama", `Ollama returned no message: ${JSON.stringify(data).slice(0, 300)}`);
    }
    const usage = data.prompt_eval_count || data.eval_count ? {
      inputTokens: data.prompt_eval_count ?? 0,
      outputTokens: data.eval_count ?? 0,
      totalTokens: (data.prompt_eval_count ?? 0) + (data.eval_count ?? 0),
      durationMs: data.total_duration ? data.total_duration / 1e6 : undefined,
      tokensPerSecond: data.eval_duration && data.eval_count ? data.eval_count / (data.eval_duration / 1e9) : undefined
    } : undefined;
    const toolCalls = data.message.tool_calls?.map((tc) => this.normalizeToolCall(tc));
    const gemmaContent = extractGemmaThoughtChannels(data.message.content || "");
    const reasoning = [data.message.thinking, gemmaContent.reasoning].filter(Boolean).join(`

`) || undefined;
    const result = {
      message: {
        role: "assistant",
        content: gemmaContent.content,
        tool_calls: toolCalls
      },
      finishReason: data.done_reason,
      reasoning,
      usage,
      provider: "ollama"
    };
    this.auditor.record({
      timestamp: Date.now(),
      type: "response",
      provider: "ollama",
      model,
      duration: Date.now() - start,
      usage
    });
    return result;
  }
  async* chatStream(messages, options) {
    const url = `${this.options.url}/api/chat`;
    const model = options?.model ?? this.options.model;
    const tools = options?.tools ?? (Object.keys(this.toolRegistry).length > 0 ? this.getToolDefinitions() : undefined);
    const body = {
      model,
      messages: this.convertMessages(messages),
      stream: true,
      options: this.buildOllamaOptions(options)
    };
    this.hoistKeepAlive(body);
    if (tools?.length) {
      body["tools"] = this.convertToolsToOllama(tools);
    }
    body["think"] = resolveThinking(options?.thinking, this.options.thinking)?.enabled ?? true;
    const schemaOptions = this.extractSchemaOptions(options);
    if (schemaOptions) {
      body["format"] = this.buildFormatParameter(schemaOptions);
    } else if (options?.responseFormat) {
      body["format"] = "json";
    }
    const start = Date.now();
    this.auditor.record({
      timestamp: start,
      type: "stream_start",
      provider: "ollama",
      model
    });
    const decoderEvents = [];
    const decoder = new StandardChatDecoder((event) => decoderEvents.push(event));
    let lastResponse;
    const streamedToolCalls = [];
    const streamTimeout = Math.max(this.options.timeout ?? 300000, 300000);
    const loopGuard = new StreamLoopGuard;
    const loopAbort = new AbortController;
    const effectiveSignal = options?.signal ? AbortSignal.any([options.signal, loopAbort.signal]) : loopAbort.signal;
    const abortOnLoop = () => {
      const d = loopGuard.detection;
      if (!d)
        return false;
      console.warn(`[ollama] Stream loop guard triggered (${d.reason}${d.pattern ? `: "${d.pattern}" ×${d.repeats}` : ""}) after ${d.totalChars} chars — aborting generation`);
      loopAbort.abort();
      return true;
    };
    const stream = httpStream(url, {
      method: "POST",
      headers: buildHeaders(this.options),
      body,
      timeout: streamTimeout,
      signal: effectiveSignal
    });
    try {
      for await (const chunk of parseNDJSON(stream)) {
        const streamError = extractProviderErrorMessage(chunk);
        if (streamError) {
          throw new LLMProviderError("ollama", `Ollama error: ${streamError}`);
        }
        lastResponse = chunk;
        if (chunk.message?.thinking) {
          decoder.pushReasoning(chunk.message.thinking);
          const pending = decoderEvents.splice(0);
          for (const event of pending) {
            yield event;
          }
          if (loopGuard.push(chunk.message.thinking) && abortOnLoop())
            break;
        }
        if (chunk.message?.content) {
          decoder.push(chunk.message.content);
          const pending = decoderEvents.splice(0);
          for (const event of pending) {
            yield event;
          }
          if (loopGuard.push(chunk.message.content) && abortOnLoop())
            break;
        }
        if (chunk.message?.tool_calls?.length) {
          const normalized = chunk.message.tool_calls.map((tc) => this.normalizeToolCall(tc));
          streamedToolCalls.push(...normalized);
          yield { type: "tool_call", calls: normalized };
          for (const tc of normalized) {
            if (loopGuard.push(tc.function.arguments) && abortOnLoop())
              break;
          }
          if (loopGuard.detection)
            break;
        }
      }
    } catch (error) {
      if (!loopGuard.detection)
        throw error;
    }
    decoder.flush();
    const pending = decoderEvents.splice(0);
    for (const event of pending) {
      yield event;
    }
    const usage = lastResponse?.prompt_eval_count ? {
      inputTokens: lastResponse.prompt_eval_count ?? 0,
      outputTokens: lastResponse.eval_count ?? 0,
      totalTokens: (lastResponse.prompt_eval_count ?? 0) + (lastResponse.eval_count ?? 0),
      durationMs: lastResponse.total_duration ? lastResponse.total_duration / 1e6 : undefined,
      tokensPerSecond: lastResponse.eval_duration && lastResponse.eval_count ? lastResponse.eval_count / (lastResponse.eval_duration / 1e9) : undefined
    } : undefined;
    this.auditor.record({
      timestamp: Date.now(),
      type: "stream_end",
      provider: "ollama",
      model,
      duration: Date.now() - start,
      usage
    });
    const rawReasoning = decoder.getReasoning();
    return {
      message: {
        role: "assistant",
        content: stripGemmaChannelMarkers(decoder.getCleanContent()),
        tool_calls: !loopGuard.detection && streamedToolCalls.length > 0 ? streamedToolCalls : undefined
      },
      finishReason: loopGuard.detection ? "degeneration" : lastResponse?.done_reason,
      reasoning: rawReasoning ? stripGemmaChannelMarkers(rawReasoning) : rawReasoning,
      usage,
      provider: "ollama"
    };
  }
  normalizeToolCall(toolCall) {
    return {
      ...toolCall,
      id: toolCall.id || this.generateToolCallId(),
      type: "function",
      function: {
        ...toolCall.function,
        name: toolCall.function?.name || "",
        arguments: this.normalizeToolArguments(toolCall.function?.arguments)
      }
    };
  }
  normalizeToolArguments(args) {
    if (typeof args === "string") {
      return args.trim().length > 0 ? args : "{}";
    }
    if (args == null) {
      return "{}";
    }
    return JSON.stringify(args) ?? "{}";
  }
  async embed(text) {
    const url = `${this.options.url}/api/embed`;
    const response = await httpRequest(url, {
      method: "POST",
      headers: buildHeaders(this.options),
      body: { model: this.options.model, input: text },
      timeout: this.options.timeout ?? 30000
    });
    return response.data.embeddings[0] ?? [];
  }
  async embedArray(texts) {
    const url = `${this.options.url}/api/embed`;
    const response = await httpRequest(url, {
      method: "POST",
      headers: buildHeaders(this.options),
      body: { model: this.options.model, input: texts },
      timeout: this.options.timeout ?? 30000
    });
    return response.data.embeddings;
  }
  async getModels() {
    const url = `${this.options.url}/api/tags`;
    const response = await httpRequest(url, {
      timeout: 5000
    });
    return response.data.models.map((m) => m.name);
  }
  async getModelInfo(modelName) {
    const url = `${this.options.url}/api/show`;
    try {
      const targetModel = modelName ?? this.options.model;
      const response = await httpRequest(url, {
        method: "POST",
        body: { name: targetModel },
        timeout: 5000
      });
      const modelInfo = response.data["model_info"];
      if (!modelInfo)
        return { contextLength: 8192 };
      const arch = modelInfo["general.architecture"];
      let contextLength = 8192;
      if (arch) {
        const ctxKey = `${arch}.context_length`;
        const ctxValue = modelInfo[ctxKey];
        if (ctxValue)
          contextLength = ctxValue;
      }
      try {
        const psResponse = await httpRequest(`${this.options.url}/api/ps`, { timeout: 5000 });
        const liveModel = psResponse.data.models?.find((model) => model.name?.toLowerCase() === targetModel.toLowerCase());
        if (liveModel?.context_length && liveModel.context_length > 0) {
          contextLength = Math.min(contextLength, liveModel.context_length);
        }
      } catch {}
      const paramCountRaw = modelInfo["general.parameter_count"];
      const capabilities = response.data["capabilities"];
      return {
        model: targetModel,
        contextLength,
        architecture: arch,
        parameterCount: paramCountRaw,
        capabilities
      };
    } catch {
      return { contextLength: 8192 };
    }
  }
  async ensureReady() {
    try {
      await this.getModelInfo();
    } catch {
      this.debugLog(`Model not found, attempting pull: ${this.options.model}`);
      await httpRequest(`${this.options.url}/api/pull`, {
        method: "POST",
        body: { name: this.options.model },
        timeout: 300000
      });
    }
  }
  convertMessages(messages) {
    return messages.map((msg) => {
      const converted = { role: msg.role };
      if (Array.isArray(msg.content)) {
        const textParts = [];
        const images = [];
        for (const part of msg.content) {
          if (part.type === "text") {
            textParts.push(part.text);
          } else if (part.type === "audio") {
            this.debugLog("Ollama: skipping audio content (not supported)");
          } else if (part.type === "image_url" && part.image_url?.url) {
            const url = part.image_url.url;
            if (url.startsWith("data:")) {
              const base64Data = url.split(",")[1];
              if (base64Data)
                images.push(base64Data);
            } else if (url.startsWith("http")) {
              this.debugLog("Ollama vision: skipping URL image, use base64 instead");
            } else {
              images.push(url);
            }
          }
        }
        converted["content"] = textParts.join(`
`);
        if (images.length > 0) {
          converted["images"] = images;
        }
      } else {
        converted["content"] = msg.content ?? "";
      }
      if (msg.tool_calls?.length) {
        converted["tool_calls"] = msg.tool_calls.map((tc) => ({
          ...tc,
          function: {
            ...tc.function,
            arguments: typeof tc.function.arguments === "string" ? (() => {
              try {
                return JSON.parse(tc.function.arguments);
              } catch {
                return tc.function.arguments;
              }
            })() : tc.function.arguments
          }
        }));
      }
      if (msg.tool_call_id) {
        converted["tool_call_id"] = msg.tool_call_id;
      }
      return converted;
    });
  }
  convertToolsToOllama(tools) {
    return tools.map((t) => ({
      type: "function",
      function: {
        name: t.function.name,
        description: t.function.description,
        parameters: t.function.parameters
      }
    }));
  }
  hoistKeepAlive(body) {
    const params = body["options"];
    if (!params || params["keep_alive"] === undefined)
      return;
    const raw = params["keep_alive"];
    delete params["keep_alive"];
    body["keep_alive"] = typeof raw === "string" && /^-?\d+$/.test(raw.trim()) ? Number(raw.trim()) : raw;
  }
  buildOllamaOptions(options) {
    const params = {
      ...this.options.defaultParameters,
      ...options?.parameters
    };
    if (options?.temperature !== undefined)
      params["temperature"] = options.temperature;
    if (options?.maxTokens !== undefined && params["num_predict"] === undefined) {
      params["num_predict"] = options.maxTokens;
    }
    const contextLength = options?.contextLength ?? this.options.contextLength;
    if (contextLength !== undefined && params["num_ctx"] === undefined) {
      params["num_ctx"] = contextLength;
    }
    return params;
  }
  buildFormatParameter(options) {
    if (options.jsonSchema) {
      return normalizeJsonSchema(options.jsonSchema);
    }
    if (options.schemaConfig) {
      return getJsonSchemaFromConfig(options.schemaConfig);
    }
    return "json";
  }
}

// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/providers/openai.js
var VLLM_AUTO_TOOL_CHOICE_HINT = "vLLM rejected automatic tool choice. Retrying with text-level tool calling. To use native tool_calls, start vLLM with --enable-auto-tool-choice and --tool-call-parser <parser>.";
function shouldOmitResponseFormatWire(options) {
  return options.omitResponseFormat === true;
}
var CEREBRAS_MODEL_CONTEXT_LENGTHS = {
  "gemma-4-31b": 131072
};
function isCerebrasEndpoint(url) {
  return (url ?? "").toLowerCase().includes("api.cerebras.ai");
}
function normalizeModelId(model) {
  return model.toLowerCase();
}
function inferOpenAICompatCapabilities(model) {
  const m = normalizeModelId(model);
  const caps = new Set(["completion"]);
  const vision = /gemma-?4/.test(m) || /gemma-?3(?!\s*embedding)/.test(m) || /\bgpt-4o\b/.test(m) || /\bgpt-4\.1\b/.test(m) || /\bgpt-5\b/.test(m) || /o[1-4](-|$)/.test(m) || /claude-(3|4|sonnet|opus|haiku)/.test(m) || /llava|bakllava|moondream|pixtral|molmo|internvl|minicpm-v|phi-4-multimodal|phi-3\.5-vision/.test(m) || /qwen[-_.]?(2\.5-)?vl|qwen3-vl|qwen.*-vl|vl.*qwen/.test(m) || /llama-?4/.test(m) || /mistral-small.*vision|mistral-medium/.test(m) || /glm-4v|glm-ocr|step-1v|aria\b|idefics|paligemma/.test(m);
  if (vision)
    caps.add("vision");
  const tools = /gemma-?4/.test(m) || /\bgpt-4|\bgpt-5|\bo[1-4]/.test(m) || /claude|qwen|llama|mistral|glm|kimi|deepseek|command-r|minimax/.test(m);
  if (tools)
    caps.add("tools");
  return [...caps];
}
function knownOpenAICompatModelInfo(url, model) {
  if (!isCerebrasEndpoint(url))
    return;
  const contextLength = CEREBRAS_MODEL_CONTEXT_LENGTHS[normalizeModelId(model)];
  if (!contextLength)
    return;
  return {
    model,
    contextLength,
    capabilities: inferOpenAICompatCapabilities(model)
  };
}
function withInferredCapabilities(model, info) {
  if (info.capabilities?.length)
    return info;
  return { ...info, capabilities: inferOpenAICompatCapabilities(model) };
}
function applyMaxTokensParam(params, maxTokens, url) {
  if (maxTokens === undefined)
    return;
  if (isCerebrasEndpoint(url)) {
    if (params["max_completion_tokens"] === undefined && params["max_tokens"] === undefined) {
      params["max_completion_tokens"] = maxTokens;
    }
    return;
  }
  if (params["max_tokens"] === undefined)
    params["max_tokens"] = maxTokens;
}
function cerebrasReasoningEffort(level) {
  if (level === "high")
    return "high";
  if (level === "low" || level === "minimal")
    return "low";
  return "medium";
}
function usesGemmaCerebrasReasoningDefault(model) {
  return normalizeModelId(model) === "gemma-4-31b";
}
function isGemmaModelId(model) {
  return /gemma/i.test(model);
}
function applyGemmaDualModeRequestDefaults(input) {
  if (!isGemmaModelId(input.model))
    return;
  if (isCerebrasEndpoint(input.url))
    return;
  const { body, thinking } = input;
  const thinkingOn = thinking?.enabled === true;
  if (body["temperature"] === undefined)
    body["temperature"] = 1;
  if (body["top_p"] === undefined)
    body["top_p"] = 0.95;
  if (body["top_k"] === undefined)
    body["top_k"] = 64;
  if (body["repetition_penalty"] === undefined) {
    body["repetition_penalty"] = thinkingOn ? 1.05 : 1.1;
  }
  if (supportsChatTemplateKwargs(input.url)) {
    const existing = body["chat_template_kwargs"] ?? {};
    if (existing["enable_thinking"] === undefined) {
      body["chat_template_kwargs"] = {
        ...existing,
        enable_thinking: thinkingOn
      };
    }
  }
}
function normalizeMessagesForOpenAICompat(messages) {
  let sawNonSystem = false;
  return messages.map((message) => {
    if (message.role !== "system") {
      sawNonSystem = true;
      return toOpenAICompatMessage(message, message.content);
    }
    if (!sawNonSystem) {
      return toOpenAICompatMessage(message, message.content);
    }
    return toOpenAICompatMessage({
      role: "user",
      content: `[SYSTEM MESSAGE]
${stringifyMessageContent(message.content)}`
    });
  });
}
function isValidToolArgumentsJson(args) {
  if (typeof args !== "string")
    return false;
  try {
    JSON.parse(args);
    return true;
  } catch {
    return false;
  }
}
function sanitizeOutboundToolCalls(toolCalls) {
  if (!toolCalls?.length)
    return toolCalls;
  if (toolCalls.every((tc) => isValidToolArgumentsJson(tc.function.arguments)))
    return toolCalls;
  return toolCalls.map((tc) => isValidToolArgumentsJson(tc.function.arguments) ? tc : {
    ...tc,
    function: {
      ...tc.function,
      arguments: JSON.stringify({ _invalid: "arguments were truncated/malformed and stubbed" })
    }
  });
}
function toOpenAICompatMessage(message, content = message.content) {
  const normalized = {
    role: message.role,
    content: content ?? ""
  };
  if (message.role === "assistant" && message.tool_calls) {
    normalized.tool_calls = sanitizeOutboundToolCalls(message.tool_calls);
  }
  if (message.role === "tool" && message.tool_call_id) {
    normalized.tool_call_id = message.tool_call_id;
  }
  return normalized;
}
function stringifyMessageContent(content) {
  if (typeof content === "string")
    return content;
  return content.map((part) => {
    if (part.type === "text")
      return part.text;
    if (part.type === "image_url")
      return `[Image: ${part.image_url.url}]`;
    if (part.type === "audio")
      return `[Audio: ${part.audio.mimeType}]`;
    return "";
  }).filter(Boolean).join(`
`);
}
function hasToolDefinitions(body) {
  const tools = body["tools"];
  return Array.isArray(tools) && tools.length > 0;
}
function isVllmAutoToolChoiceError(value) {
  const text = value instanceof Error ? value.message : typeof value === "string" ? value : JSON.stringify(value ?? "");
  const normalized = text.toLowerCase();
  return normalized.includes("auto") && normalized.includes("tool choice requires --enable-auto-tool-choice") && normalized.includes("--tool-call-parser");
}
async function requestWithVllmToolFallback(url, request, tools, onFallback) {
  try {
    return await httpRequest(url, {
      method: "POST",
      headers: request.headers,
      body: request.body,
      timeout: request.timeout
    });
  } catch (error) {
    if (tools?.length && hasToolDefinitions(request.body) && isVllmAutoToolChoiceError(error)) {
      onFallback();
      return httpRequest(url, {
        method: "POST",
        headers: request.headers,
        body: withoutNativeTools(request.body, tools),
        timeout: request.timeout
      });
    }
    throw error;
  }
}
function parseJsonObject2(text) {
  try {
    const parsed = JSON.parse(text);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed;
    }
  } catch {}
  return null;
}
function normalizeLooseArguments(rawArgs) {
  const trimmed = rawArgs.trim();
  if (!trimmed)
    return "{}";
  const jsonArgs = parseJsonObject2(trimmed);
  if (jsonArgs)
    return JSON.stringify(jsonArgs);
  const unwrapped = trimmed.startsWith("{") && trimmed.endsWith("}") ? trimmed.slice(1, -1).trim() : trimmed;
  try {
    return gemmaArgsToJson(unwrapped);
  } catch {
    return null;
  }
}
function sanitizeToolCallName(raw) {
  return raw.trim().replace(/[({\s]+$/u, "");
}
function recoverLooseToolArguments(rawArgs) {
  const trimmed = rawArgs.trim();
  if (!trimmed)
    return "{}";
  const stripped = trimmed.replace(/[)\s]+$/u, "");
  const candidates = stripped !== trimmed && stripped.endsWith("}") ? [stripped, trimmed] : [trimmed];
  for (const candidate of candidates) {
    const obj = parseJsonObject2(candidate);
    if (obj)
      return JSON.stringify(obj);
  }
  const loose = stripped.endsWith("}") ? stripped : trimmed;
  if (loose.startsWith("{") && loose.endsWith("}") && !loose.startsWith('{"')) {
    try {
      return gemmaArgsToJson(loose.slice(1, -1).trim());
    } catch {
      return null;
    }
  }
  return null;
}
function parseTextToolCallBody(content) {
  const body = content.trim();
  if (!body)
    return [];
  try {
    const parsed = JSON.parse(body);
    const rawCalls = Array.isArray(parsed) ? parsed : [parsed];
    const calls = [];
    for (const rawCall of rawCalls) {
      if (!rawCall || typeof rawCall !== "object")
        continue;
      const record = rawCall;
      const name = record["name"];
      if (typeof name !== "string" || !name)
        continue;
      const args = record["arguments"] ?? record["parameters"] ?? record["args"] ?? {};
      calls.push({
        name,
        arguments: typeof args === "string" ? JSON.stringify(parseJsonObject2(args) ?? {}) : JSON.stringify(args ?? {})
      });
    }
    if (calls.length > 0)
      return calls;
  } catch {}
  const bareGemmaCallMatch = /^call:([@A-Za-z_][@A-Za-z0-9_.:-]*)\s*[\({]([\s\S]*)[\)}]\s*$/u.exec(body);
  if (bareGemmaCallMatch) {
    const rawArgs = bareGemmaCallMatch[2].trim();
    const argumentsJson = normalizeLooseArguments(rawArgs);
    if (argumentsJson)
      return [{ name: bareGemmaCallMatch[1], arguments: argumentsJson }];
  }
  const functionCallMatch = /^([@A-Za-z_][@A-Za-z0-9_.:-]*)\s*\(([\s\S]*)\)\s*$/u.exec(body);
  if (functionCallMatch) {
    const rawArgs = functionCallMatch[2].trim();
    const argumentsJson = normalizeLooseArguments(rawArgs);
    if (argumentsJson)
      return [{ name: functionCallMatch[1], arguments: argumentsJson }];
  }
  const calls = [];
  const funcPattern = /<function=([@A-Za-z_][@A-Za-z0-9_.:-]*)>([\s\S]*?)<\/function>/g;
  let fMatch;
  while ((fMatch = funcPattern.exec(body)) !== null) {
    const args = {};
    const paramPattern = /<parameter=([A-Za-z_][A-Za-z0-9_-]*)>([\s\S]*?)<\/parameter>/g;
    let pMatch;
    while ((pMatch = paramPattern.exec(fMatch[2] ?? "")) !== null) {
      args[pMatch[1]] = pMatch[2].trim();
    }
    calls.push({ name: fMatch[1], arguments: JSON.stringify(args) });
  }
  return calls;
}
function findBareGemmaCallSegments(content) {
  const segments = [];
  const startPattern = /call:[@A-Za-z_][@A-Za-z0-9_.:-]*\s*[\({]/gu;
  let match;
  while ((match = startPattern.exec(content)) !== null) {
    const start = match.index;
    const opener = content[match.index + match[0].length - 1];
    const openIdx = match.index + match[0].length - 1;
    const primaryClose = opener === "(" ? ")" : "}";
    const toleratedClose = opener === "(" ? "}" : undefined;
    let depth = 0;
    let quote = null;
    let escaped = false;
    for (let i = openIdx;i < content.length; i++) {
      const ch = content[i];
      if (quote) {
        if (escaped) {
          escaped = false;
        } else if (ch === "\\") {
          escaped = true;
        } else if (ch === quote) {
          quote = null;
        }
        continue;
      }
      if (ch === '"' || ch === "'") {
        quote = ch;
      } else if (ch === opener) {
        depth++;
      } else if (ch === primaryClose) {
        depth--;
        if (depth === 0) {
          segments.push(content.slice(start, i + 1));
          startPattern.lastIndex = i + 1;
          break;
        }
      } else if (depth === 1 && toleratedClose !== undefined && ch === toleratedClose) {
        const rawArgs = content.slice(openIdx + 1, i).trim();
        if (rawArgs.startsWith("{"))
          continue;
        if (!normalizeLooseArguments(rawArgs))
          continue;
        segments.push(content.slice(start, i + 1));
        startPattern.lastIndex = i + 1;
        break;
      }
    }
  }
  return segments;
}
function recoverToolCallsFromText(content, knownToolNames, generateId) {
  if (!content || content.length < 10)
    return null;
  const calls = [];
  let cleanContent = content;
  const isKnownTool = (name) => knownToolNames.has(name);
  const toolCallPattern = /<tool_call>([\s\S]*?)<\/tool_call>/g;
  let tcMatch;
  while ((tcMatch = toolCallPattern.exec(content)) !== null) {
    const parsedCalls = parseTextToolCallBody(tcMatch[1]);
    let matched = false;
    for (const parsed of parsedCalls) {
      if (!isKnownTool(parsed.name))
        continue;
      matched = true;
      calls.push({
        id: generateId(),
        type: "function",
        function: { name: parsed.name, arguments: parsed.arguments }
      });
    }
    if (matched)
      cleanContent = cleanContent.replace(tcMatch[0], "");
  }
  for (const segment of findBareGemmaCallSegments(content)) {
    const parsedCalls = parseTextToolCallBody(segment);
    let matched = false;
    for (const parsed of parsedCalls) {
      if (!isKnownTool(parsed.name))
        continue;
      matched = true;
      calls.push({
        id: generateId(),
        type: "function",
        function: { name: parsed.name, arguments: parsed.arguments }
      });
    }
    if (matched)
      cleanContent = cleanContent.replace(segment, "");
  }
  if (calls.length === 0)
    return null;
  return { calls, cleanContent: cleanContent.trim() };
}
function toolFallbackInstruction(tools) {
  const toolLines = tools.map((tool) => {
    const fn = tool.function;
    return `- ${fn.name}: ${fn.description}
  parameters JSON schema: ${JSON.stringify(fn.parameters)}`;
  });
  return {
    role: "system",
    content: "The server does not support native OpenAI tool parsing for this request. " + `Use this text tool protocol instead.

` + `When you need a tool, respond with exactly one or more tool calls and no prose:
` + `<tool_call>tool_name({"argument":"value"})</tool_call>

` + `After tool results are provided, answer the user normally. Available tools:
` + toolLines.join(`
`)
  };
}
function withTextToolFallbackMessages(messages, tools) {
  return [toolFallbackInstruction(tools), ...messages];
}
function withoutNativeTools(body, tools) {
  const fallbackBody = { ...body };
  delete fallbackBody["tools"];
  delete fallbackBody["tool_choice"];
  fallbackBody["messages"] = withTextToolFallbackMessages(body["messages"] ?? [], tools);
  return fallbackBody;
}

class OpenAICompatibleClient extends BaseLLMClient {
  warnedVllmToolFallback = false;
  get gemmaNative() {
    return this.options.gemmaNativeProtocol ?? isGemmaDiffusionModel(this.options.model);
  }
  buildUrl(suffix) {
    const raw = this.options.url.replace(/\/+$/, "");
    const qIdx = raw.indexOf("?");
    const basePath = (qIdx === -1 ? raw : raw.slice(0, qIdx)).replace(/\/+$/, "");
    const existingQuery = qIdx === -1 ? "" : raw.slice(qIdx + 1);
    const path = suffix.startsWith("/") ? suffix : "/" + suffix;
    const search = new URLSearchParams(existingQuery);
    const qp = this.options.queryParams;
    if (qp) {
      for (const [k, v] of Object.entries(qp)) {
        if (v != null)
          search.set(k, String(v));
      }
    }
    const qs = search.toString();
    return basePath + path + (qs ? `?${qs}` : "");
  }
  constructor(options, auditor) {
    let base = (options.url || "https://api.openai.com").replace(/\/+$/, "");
    const desired = options.apiBasePath;
    const shouldAppend = desired !== "" && desired !== "/";
    if (shouldAppend) {
      const basePath = ("/" + (desired || "/v1").replace(/^\/+/, "")).replace(/\/+$/, "");
      if (!base.endsWith(basePath)) {
        base += basePath;
      }
    }
    super({ ...options, url: base }, auditor);
  }
  warnVllmToolFallback() {
    if (this.warnedVllmToolFallback)
      return;
    this.warnedVllmToolFallback = true;
    console.warn(`[OpenAI] ${VLLM_AUTO_TOOL_CHOICE_HINT}`);
  }
  async chat(messages, options) {
    const url = this.buildUrl("/chat/completions");
    const tools = options?.tools;
    const body = {
      model: this.options.model,
      messages: this.convertMessages(messages),
      ...this.buildRequestParams(options)
    };
    const omitWireFormat = shouldOmitResponseFormatWire({
      omitResponseFormat: this.options.omitResponseFormat,
      url: this.options.url,
      model: this.options.model
    });
    if (!omitWireFormat) {
      const schemaOptions = this.extractSchemaOptions(options);
      if (schemaOptions) {
        body["response_format"] = this.buildResponseFormat(schemaOptions);
      } else if (options?.responseFormat) {
        body["response_format"] = options.responseFormat;
      }
    }
    if (tools?.length) {
      body["tools"] = tools;
      if (options?.toolChoice) {
        body["tool_choice"] = options.toolChoice;
      }
    }
    if (this.gemmaNative) {
      body["skip_special_tokens"] = false;
      if (tools?.length)
        body["tool_choice"] = "none";
      if (body["max_tokens"] == null)
        body["max_tokens"] = 4096;
    }
    this.applyFamilySamplingDefaults(body, this.resolveCallThinking(options));
    await this.applyDefaultOutputBound(body, messages, tools);
    const start = Date.now();
    this.auditor.record({
      timestamp: start,
      type: "request",
      provider: "openai",
      model: this.options.model
    });
    const response = await requestWithVllmToolFallback(url, {
      headers: buildHeaders(this.options),
      body,
      timeout: this.options.timeout ?? 60000
    }, tools, () => this.warnVllmToolFallback());
    const data = response.data;
    const choice = data.choices[0];
    if (!choice) {
      throw new Error("No choices returned from OpenAI API");
    }
    const durationMs = Date.now() - start;
    const usage = data.usage ? {
      inputTokens: data.usage.prompt_tokens,
      outputTokens: data.usage.completion_tokens,
      totalTokens: data.usage.total_tokens,
      cachedTokens: data.usage.prompt_tokens_details?.cached_tokens,
      reasoningTokens: data.usage.completion_tokens_details?.reasoning_tokens,
      durationMs,
      tokensPerSecond: durationMs > 0 ? data.usage.completion_tokens / (durationMs / 1000) : undefined
    } : undefined;
    let toolCalls = choice.message.tool_calls?.map((tc) => this.normalizeToolCall(tc));
    let content = choice.message.content || "";
    let reasoning;
    const serverReasoning = [choice.message.reasoning, choice.message.reasoning_content].filter((value) => typeof value === "string" && value.length > 0).join(`

`) || undefined;
    if (typeof serverReasoning === "string" && serverReasoning.length > 0) {
      reasoning = serverReasoning;
    }
    if (this.gemmaNative && content) {
      const parsed = parseGemmaDiffusionOutput(content);
      content = parsed.content;
      if (parsed.reasoning)
        reasoning = parsed.reasoning;
      if (!toolCalls?.length && parsed.toolCalls.length) {
        toolCalls = parsed.toolCalls.map((tc) => ({
          id: this.generateToolCallId(),
          type: "function",
          function: { name: tc.name, arguments: tc.argumentsJson }
        }));
      }
    } else if (isGemmaModelId(this.options.model) && content) {
      const extracted = extractGemmaThoughtChannels(content);
      if (extracted.found) {
        content = extracted.content.trim();
        if (extracted.reasoning) {
          reasoning = reasoning ? `${reasoning}

${extracted.reasoning}` : extracted.reasoning;
        }
      }
      if (/<\|channel/i.test(content) || /<\|thought/i.test(content)) {
        const unclosed = content.match(/<\|channel\|?>\s*'?\s*(?:thought|think)\s*'?\s*\r?\n?([\s\S]*)$/i);
        if (unclosed) {
          const thought = (unclosed[1] ?? "").trim();
          if (thought) {
            reasoning = reasoning ? `${reasoning}

${thought}` : thought;
          }
          content = content.slice(0, unclosed.index).trim();
        } else {
          content = stripGemmaChannelMarkers(content).trim();
        }
      }
      if (reasoning)
        reasoning = stripGemmaChannelMarkers(reasoning);
    }
    if (!toolCalls?.length && tools?.length && content) {
      const knownToolNames = new Set(tools.map((tool) => tool.function.name));
      const recovered = recoverToolCallsFromText(content, knownToolNames, () => this.generateToolCallId());
      if (recovered) {
        toolCalls = recovered.calls;
        content = recovered.cleanContent;
      }
    }
    const result = {
      message: {
        role: "assistant",
        content,
        tool_calls: toolCalls
      },
      ...reasoning !== undefined && { reasoning },
      usage,
      finishReason: choice.finish_reason,
      provider: "openai"
    };
    this.auditor.record({
      timestamp: Date.now(),
      type: "response",
      provider: "openai",
      model: this.options.model,
      duration: Date.now() - start,
      usage
    });
    return result;
  }
  async* chatStream(messages, options) {
    const url = this.buildUrl("/chat/completions");
    const tools = options?.tools;
    const knownToolNames = new Set(tools?.map((tool) => tool.function.name) ?? []);
    const body = {
      model: this.options.model,
      messages: this.convertMessages(messages),
      stream: true,
      stream_options: { include_usage: true },
      ...this.buildRequestParams(options)
    };
    if (tools?.length) {
      body["tools"] = tools;
      if (options?.toolChoice) {
        body["tool_choice"] = options.toolChoice;
      }
    }
    if (this.gemmaNative) {
      body["skip_special_tokens"] = false;
      if (tools?.length)
        body["tool_choice"] = "none";
      if (body["max_tokens"] == null)
        body["max_tokens"] = 4096;
    }
    this.applyFamilySamplingDefaults(body, this.resolveCallThinking(options));
    await this.applyDefaultOutputBound(body, messages, tools);
    const start = Date.now();
    this.auditor.record({
      timestamp: start,
      type: "stream_start",
      provider: "openai",
      model: this.options.model
    });
    const loopGuard = new StreamLoopGuard;
    const loopAbort = new AbortController;
    const effectiveSignal = options?.signal ? AbortSignal.any([options.signal, loopAbort.signal]) : loopAbort.signal;
    const abortOnLoop = () => {
      const d = loopGuard.detection;
      if (!d)
        return false;
      console.warn(`[openai] Stream loop guard triggered (${d.reason}${d.pattern ? `: "${d.pattern}" ×${d.repeats}` : ""}) after ${d.totalChars} chars — aborting generation`);
      loopAbort.abort();
      return true;
    };
    const decoderEvents = [];
    const decoderOption = options?.decoder;
    const decoderInstanceProvided = Boolean(decoderOption && typeof decoderOption === "object");
    const yieldDecoderEvents = !decoderInstanceProvided && (this.gemmaNative || typeof decoderOption === "string" || knownToolNames.size > 0);
    const decoder = typeof decoderOption === "string" ? createDecoder(decoderOption, (e) => decoderEvents.push(e), { knownToolNames }) : decoderInstanceProvided ? decoderOption : new StandardChatDecoder(yieldDecoderEvents ? (e) => decoderEvents.push(e) : () => {}, { knownToolNames });
    const decoderToolCalls = [];
    const takeDecoderEvents = function* () {
      while (decoderEvents.length) {
        const event = decoderEvents.shift();
        if (event.type === "tool_call") {
          decoderToolCalls.push(...event.calls);
          yield event;
        } else if (yieldDecoderEvents) {
          yield event;
        }
      }
    };
    const toolCallAccum = new Map;
    let activeBody = body;
    let retriedWithTextTools = false;
    let emittedDeltas = 0;
    let lastFinishReason;
    let usage;
    let reasoningBuffer = "";
    let thinkingHeadPending = "";
    while (true) {
      const stream = httpStream(url, {
        method: "POST",
        headers: buildHeaders(this.options),
        body: activeBody,
        timeout: Math.max(this.options.timeout ?? 300000, 300000),
        signal: effectiveSignal
      });
      try {
        for await (const { data } of parseSSE(stream)) {
          try {
            const parsed = JSON.parse(data);
            if (parsed.usage) {
              usage = {
                inputTokens: parsed.usage.prompt_tokens,
                outputTokens: parsed.usage.completion_tokens,
                totalTokens: parsed.usage.total_tokens,
                cachedTokens: parsed.usage.prompt_tokens_details?.cached_tokens,
                reasoningTokens: parsed.usage.completion_tokens_details?.reasoning_tokens
              };
            }
            const delta = parsed.choices?.[0]?.delta;
            if (!delta)
              continue;
            if (delta.content || delta.reasoning || delta.reasoning_content || delta.tool_calls) {
              emittedDeltas++;
            }
            const reasoningDelta = delta.reasoning ?? delta.reasoning_content;
            if (reasoningDelta) {
              reasoningBuffer += reasoningDelta;
              decoder.pushReasoning?.(reasoningDelta);
              if (yieldDecoderEvents) {
                for (const event of takeDecoderEvents())
                  yield event;
              } else {
                if (thinkingHeadPending !== null) {
                  thinkingHeadPending += reasoningDelta;
                  if (thinkingHeadPending.includes(`
`) || thinkingHeadPending.length >= 96) {
                    const cleaned = stripGemmaChannelMarkers(thinkingHeadPending);
                    thinkingHeadPending = null;
                    if (cleaned)
                      yield { type: "thinking", content: cleaned };
                  }
                } else {
                  yield { type: "thinking", content: reasoningDelta };
                }
              }
              if (loopGuard.push(reasoningDelta) && abortOnLoop())
                break;
            }
            if (delta.content) {
              decoder.push(delta.content);
              if (yieldDecoderEvents) {
                for (const event of takeDecoderEvents())
                  yield event;
              } else {
                yield { type: "text", content: delta.content };
              }
              if (loopGuard.push(delta.content) && abortOnLoop())
                break;
            }
            if (delta.tool_calls) {
              for (const tc of delta.tool_calls) {
                const existing = toolCallAccum.get(tc.index);
                if (!existing) {
                  toolCallAccum.set(tc.index, {
                    id: tc.id || this.generateToolCallId(),
                    type: "function",
                    function: {
                      name: tc.function?.name || "",
                      arguments: tc.function?.arguments || ""
                    }
                  });
                } else {
                  if (tc.function?.arguments) {
                    existing.function.arguments += tc.function.arguments;
                  }
                  if (tc.function?.name) {
                    existing.function.name += tc.function.name;
                  }
                }
                if (tc.function?.arguments && loopGuard.push(tc.function.arguments) && abortOnLoop()) {
                  break;
                }
              }
              if (loopGuard.detection)
                break;
            }
            if (parsed.choices?.[0]?.finish_reason) {
              lastFinishReason = parsed.choices[0].finish_reason;
            }
            if (parsed.choices?.[0]?.finish_reason === "tool_calls" || parsed.choices?.[0]?.finish_reason === "stop") {
              if (toolCallAccum.size > 0) {
                const calls = Array.from(toolCallAccum.values()).map((tc) => this.normalizeToolCall(tc));
                yield { type: "tool_call", calls };
              }
            }
          } catch {}
        }
        break;
      } catch (error) {
        if (loopGuard.detection)
          break;
        if (!retriedWithTextTools && emittedDeltas === 0 && tools?.length && hasToolDefinitions(activeBody) && isVllmAutoToolChoiceError(error)) {
          this.warnVllmToolFallback();
          activeBody = withoutNativeTools(activeBody, tools);
          retriedWithTextTools = true;
          continue;
        }
        throw error;
      }
    }
    decoder.flush();
    if (yieldDecoderEvents) {
      for (const event of takeDecoderEvents())
        yield event;
    }
    if (thinkingHeadPending) {
      const cleaned = stripGemmaChannelMarkers(thinkingHeadPending);
      thinkingHeadPending = null;
      if (cleaned)
        yield { type: "thinking", content: cleaned };
    }
    if (usage) {
      const durationMs = Date.now() - start;
      usage = {
        ...usage,
        durationMs,
        tokensPerSecond: durationMs > 0 ? usage.outputTokens / (durationMs / 1000) : undefined
      };
    }
    this.auditor.record({
      timestamp: Date.now(),
      type: "stream_end",
      provider: "openai",
      model: this.options.model,
      duration: Date.now() - start,
      usage
    });
    let finalToolCalls = loopGuard.detection ? undefined : toolCallAccum.size > 0 ? Array.from(toolCallAccum.values()).map((tc) => this.normalizeToolCall(tc)) : decoderToolCalls.length > 0 ? decoderToolCalls : undefined;
    if (finalToolCalls?.length) {
      const valid = finalToolCalls.filter((tc) => isValidToolArgumentsJson(tc.function.arguments));
      if (valid.length < finalToolCalls.length) {
        console.warn(`[openai] Dropped ${finalToolCalls.length - valid.length} tool call(s) with malformed arguments (finish_reason=${lastFinishReason ?? "unknown"})`);
        finalToolCalls = valid.length > 0 ? valid : undefined;
      }
    }
    let cleanContent = stripGemmaChannelMarkers(decoder.getCleanContent());
    const decodedReasoning = decoder.getReasoning();
    let reasoning = decodedReasoning !== undefined ? decodedReasoning : yieldDecoderEvents ? undefined : reasoningBuffer;
    if (reasoning)
      reasoning = stripGemmaChannelMarkers(reasoning);
    if (this.gemmaNative) {
      const parsed = parseGemmaDiffusionOutput(cleanContent);
      cleanContent = parsed.content;
      if (parsed.reasoning) {
        reasoning = reasoning ? `${reasoning}

${parsed.reasoning}` : parsed.reasoning;
      }
      if (!finalToolCalls?.length && parsed.toolCalls.length) {
        finalToolCalls = parsed.toolCalls.map((tc) => ({
          id: this.generateToolCallId(),
          type: "function",
          function: { name: tc.name, arguments: tc.argumentsJson }
        }));
        yield { type: "tool_call", calls: finalToolCalls };
      }
    }
    if (!finalToolCalls?.length && tools?.length && cleanContent && !loopGuard.detection) {
      const knownToolNames = new Set(tools.map((tool) => tool.function.name));
      const recovered = recoverToolCallsFromText(cleanContent, knownToolNames, () => this.generateToolCallId());
      if (recovered) {
        finalToolCalls = recovered.calls;
        cleanContent = recovered.cleanContent;
        yield { type: "tool_call", calls: finalToolCalls };
      }
    }
    return {
      message: {
        role: "assistant",
        content: cleanContent,
        tool_calls: finalToolCalls
      },
      reasoning,
      usage,
      provider: "openai",
      ...loopGuard.detection ? { finishReason: "degeneration" } : lastFinishReason ? { finishReason: lastFinishReason } : {}
    };
  }
  normalizeToolCall(toolCall) {
    const name = sanitizeToolCallName(toolCall.function?.name || "");
    let args = this.normalizeToolArguments(toolCall.function?.arguments);
    if (!isValidToolArgumentsJson(args)) {
      const recovered = recoverLooseToolArguments(args);
      if (recovered)
        args = recovered;
    }
    return {
      ...toolCall,
      id: toolCall.id || this.generateToolCallId(),
      type: "function",
      function: {
        ...toolCall.function,
        name,
        arguments: args
      }
    };
  }
  normalizeToolArguments(args) {
    if (typeof args === "string") {
      return args.trim().length > 0 ? args : "{}";
    }
    if (args == null) {
      return "{}";
    }
    return JSON.stringify(args) ?? "{}";
  }
  async embed(text) {
    const url = this.buildUrl("/embeddings");
    const response = await httpRequest(url, {
      method: "POST",
      headers: buildHeaders(this.options),
      body: {
        model: this.options.model,
        input: text
      },
      timeout: this.options.timeout ?? 30000
    });
    return response.data.data[0]?.embedding ?? [];
  }
  async getModels() {
    const url = this.buildUrl("/models");
    try {
      const response = await httpRequest(url, {
        headers: buildHeaders(this.options),
        timeout: 5000
      });
      return response.data.data.map((m) => m.id);
    } catch {
      return [];
    }
  }
  modelInfoCache = new Map;
  knownWindowCache = new Map;
  async resolveKnownContextLength(model) {
    const cached = this.knownWindowCache.get(model);
    if (cached !== undefined)
      return cached ?? undefined;
    const probed = await this.probeContextLength(model);
    this.knownWindowCache.set(model, probed ?? null);
    return probed ?? undefined;
  }
  async probeContextLength(model) {
    const known = knownOpenAICompatModelInfo(this.options.url, model);
    if (known?.contextLength)
      return known.contextLength;
    try {
      const response = await httpRequest(this.buildUrl("/models"), {
        headers: buildHeaders(this.options),
        timeout: 5000
      });
      if (response.ok) {
        const cards = response.data.data ?? [];
        const card = cards.find((m) => m.id === model) ?? cards[0];
        const ctx = card?.max_model_len ?? card?.meta?.n_ctx_train;
        if (typeof ctx === "number" && ctx > 0)
          return ctx;
      }
    } catch {}
    return;
  }
  applyFamilySamplingDefaults(body, thinking) {
    applyGemmaDualModeRequestDefaults({
      model: this.options.model,
      url: this.options.url,
      body,
      thinking
    });
  }
  async applyDefaultOutputBound(body, messages, tools) {
    if (body["max_tokens"] != null || body["max_completion_tokens"] != null)
      return;
    const window = await this.resolveKnownContextLength(this.options.model);
    if (!window)
      return;
    const promptChars = JSON.stringify(messages).length + (tools?.length ? JSON.stringify(tools).length : 0);
    const promptEstimate = Math.ceil(promptChars / 4);
    const MARGIN = 512;
    const thinkingOn = body["chat_template_kwargs"]?.enable_thinking === true;
    const CAP = thinkingOn ? 16384 : 8192;
    const FLOOR = thinkingOn ? 2048 : 256;
    const bound = Math.max(FLOOR, Math.min(CAP, window - promptEstimate - MARGIN));
    if (isOpenAIReasoningModel(this.options.model) && isStrictOpenAICompatHost(this.options.url)) {
      body["max_completion_tokens"] = bound;
    } else {
      body["max_tokens"] = bound;
    }
  }
  async getModelInfo(modelName) {
    const model = modelName ?? this.options.model;
    const known = knownOpenAICompatModelInfo(this.options.url, model);
    if (known)
      return known;
    const cached = this.modelInfoCache.get(model);
    if (cached)
      return cached;
    const probed = await this.probeContextLength(model);
    const info = withInferredCapabilities(model, {
      model,
      contextLength: probed ?? 8192
    });
    this.modelInfoCache.set(model, info);
    return info;
  }
  convertMessages(messages) {
    return normalizeMessagesForOpenAICompat(messages);
  }
  buildRequestParams(options) {
    const params = {
      ...this.options.defaultParameters,
      ...options?.parameters
    };
    if (options?.temperature !== undefined)
      params["temperature"] = options.temperature;
    applyMaxTokensParam(params, options?.maxTokens, this.options.url);
    const thinking = resolveThinking(options?.thinking, this.options.thinking);
    if (isCerebrasEndpoint(this.options.url)) {
      if (params["reasoning_effort"] === undefined && params["disable_reasoning"] === undefined) {
        if (thinking) {
          params["reasoning_effort"] = thinking.enabled ? cerebrasReasoningEffort(thinking.level) : "none";
        } else if (usesGemmaCerebrasReasoningDefault(this.options.model)) {
          params["reasoning_effort"] = "medium";
        }
      }
    } else if (thinking) {
      if (isOpenAIReasoningModel(this.options.model)) {
        if (params["reasoning_effort"] === undefined) {
          params["reasoning_effort"] = thinking.enabled ? thinking.level ?? "medium" : "minimal";
        }
      } else if (supportsChatTemplateKwargs(this.options.url)) {
        const existing = params["chat_template_kwargs"] ?? {};
        params["chat_template_kwargs"] = { enable_thinking: thinking.enabled, ...existing };
      }
    }
    if (isOpenAIReasoningModel(this.options.model) && isStrictOpenAICompatHost(this.options.url)) {
      if (params["max_tokens"] !== undefined && params["max_completion_tokens"] === undefined) {
        params["max_completion_tokens"] = params["max_tokens"];
      }
      delete params["max_tokens"];
      delete params["temperature"];
      if ((options?.tools?.length ?? 0) > 0) {
        params["reasoning_effort"] = "none";
      }
    }
    return params;
  }
  resolveCallThinking(options) {
    return resolveThinking(options?.thinking, this.options.thinking);
  }
  buildResponseFormat(options) {
    let jsonSchema;
    let name;
    let description;
    if (options.jsonSchema) {
      jsonSchema = normalizeJsonSchema(options.jsonSchema);
      name = options.name || "response";
      description = options.description;
    } else if (options.schemaConfig) {
      jsonSchema = getJsonSchemaFromConfig(options.schemaConfig);
      name = options.name || options.schemaConfig.name || "response";
      description = options.description || options.schemaConfig.description;
    } else {
      throw new Error("Either schemaConfig or jsonSchema must be provided");
    }
    return {
      type: "json_schema",
      json_schema: {
        name,
        ...description && { description },
        schema: jsonSchema,
        strict: options.strict ?? true
      }
    };
  }
}

// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/providers/google.js
class GoogleClient extends BaseLLMClient {
  isVertex;
  apiVersion;
  constructor(options, auditor) {
    super(options, auditor);
    this.isVertex = options.apiType === "vertex";
    this.apiVersion = options.apiVersion ?? "v1beta";
  }
  getBaseUrl() {
    if (this.isVertex) {
      const region = this.options.region ?? "us-central1";
      return `https://${region}-aiplatform.googleapis.com/${this.apiVersion}/projects/-/locations/${region}/publishers/google/models/${this.options.model}`;
    }
    if (this.options.url)
      return this.options.url.replace(/\/+$/, "");
    return `https://generativelanguage.googleapis.com/${this.apiVersion}/models/${this.options.model}`;
  }
  getChatUrl() {
    const base = this.getBaseUrl();
    if (this.isVertex) {
      return `${base}:generateContent`;
    }
    return `${base}:generateContent?key=${this.options.apiKey}`;
  }
  getStreamUrl() {
    const base = this.getBaseUrl();
    if (this.isVertex) {
      return `${base}:streamGenerateContent?alt=sse`;
    }
    return `${base}:streamGenerateContent?alt=sse&key=${this.options.apiKey}`;
  }
  getEmbedUrl() {
    if (this.isVertex) {
      const region = this.options.region ?? "us-central1";
      return `https://${region}-aiplatform.googleapis.com/${this.apiVersion}/projects/-/locations/${region}/publishers/google/models/${this.options.model}:embedContent`;
    }
    return `https://generativelanguage.googleapis.com/${this.apiVersion}/models/${this.options.model}:embedContent?key=${this.options.apiKey}`;
  }
  getHeaders() {
    const headers = {
      "Content-Type": "application/json"
    };
    if (this.isVertex && this.options.apiKey) {
      headers["Authorization"] = `Bearer ${this.options.apiKey}`;
    }
    return headers;
  }
  async chat(messages, options) {
    const url = this.getChatUrl();
    const body = this.buildRequestBody(messages, options);
    const tier = options?.serviceTier;
    const effectiveTimeout = tier === "flex" ? Math.max(this.options.timeout ?? 60000, 600000) : this.options.timeout ?? 60000;
    const start = Date.now();
    this.auditor.record({
      timestamp: start,
      type: "request",
      provider: this.isVertex ? "vertex" : "google",
      model: this.options.model
    });
    const reqOptions = {
      method: "POST",
      headers: this.getHeaders(),
      body,
      timeout: effectiveTimeout
    };
    const response = tier === "flex" ? await this.fetchWithFlexRetry(url, reqOptions) : await httpRequest(url, reqOptions);
    const result = this.parseGoogleResponse(response.data);
    const resolvedTier = response.headers?.get("x-gemini-service-tier");
    if (resolvedTier) {
      result.serviceTier = resolvedTier.toLowerCase();
    }
    this.auditor.record({
      timestamp: Date.now(),
      type: "response",
      provider: this.isVertex ? "vertex" : "google",
      model: this.options.model,
      duration: Date.now() - start,
      usage: result.usage
    });
    return result;
  }
  async* chatStream(messages, options) {
    const url = this.getStreamUrl();
    const body = this.buildRequestBody(messages, options);
    const tier = options?.serviceTier;
    const effectiveTimeout = tier === "flex" ? Math.max(this.options.timeout ?? 120000, 600000) : this.options.timeout ?? 120000;
    const start = Date.now();
    this.auditor.record({
      timestamp: start,
      type: "stream_start",
      provider: this.isVertex ? "vertex" : "google",
      model: this.options.model
    });
    const decoder = new StandardChatDecoder(() => {});
    let usage;
    const allToolCalls = [];
    const stream = httpStream(url, {
      method: "POST",
      headers: this.getHeaders(),
      body,
      timeout: effectiveTimeout
    });
    let buffer = "";
    let reasoningBuffer = "";
    for await (const chunk of stream) {
      buffer += chunk;
      const lines = buffer.split(`
`);
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.startsWith("data: "))
          continue;
        const jsonStr = line.slice(6).trim();
        if (!jsonStr || jsonStr === "[DONE]")
          continue;
        try {
          const data = JSON.parse(jsonStr);
          if (data.usageMetadata) {
            usage = {
              inputTokens: data.usageMetadata.promptTokenCount ?? 0,
              outputTokens: data.usageMetadata.candidatesTokenCount ?? 0,
              totalTokens: data.usageMetadata.totalTokenCount ?? 0,
              cachedTokens: data.usageMetadata.cachedContentTokenCount,
              reasoningTokens: data.usageMetadata.thoughtsTokenCount
            };
          }
          const candidate = data.candidates?.[0];
          if (!candidate?.content?.parts)
            continue;
          for (const part of candidate.content.parts) {
            if (part.text) {
              if (part.thought) {
                reasoningBuffer += part.text;
                yield { type: "thinking", content: part.text };
              } else {
                decoder.push(part.text);
                yield { type: "text", content: part.text };
              }
            }
            if (part.functionCall) {
              const toolCall = this.convertFunctionCallToToolCall(part.functionCall, part.thoughtSignature);
              allToolCalls.push(toolCall);
              yield { type: "tool_call", calls: [toolCall] };
            }
          }
        } catch {}
      }
    }
    decoder.flush();
    this.auditor.record({
      timestamp: Date.now(),
      type: "stream_end",
      provider: this.isVertex ? "vertex" : "google",
      model: this.options.model,
      duration: Date.now() - start,
      usage
    });
    return {
      message: {
        role: "assistant",
        content: decoder.getCleanContent(),
        tool_calls: allToolCalls.length > 0 ? allToolCalls : undefined
      },
      reasoning: reasoningBuffer || decoder.getReasoning(),
      usage,
      provider: this.isVertex ? "vertex" : "google"
    };
  }
  supportsDeepResearch() {
    return !this.isVertex;
  }
  interactionsBase() {
    if (this.isVertex) {
      throw new Error("Deep Research is only available via Google AI Studio, not Vertex AI.");
    }
    return `https://generativelanguage.googleapis.com/${this.apiVersion}/interactions`;
  }
  deepResearchHeaders() {
    return {
      "Content-Type": "application/json",
      "x-goog-api-key": this.options.apiKey ?? "",
      "Api-Revision": "2026-05-20"
    };
  }
  buildInteractionBody(input, opts, background) {
    return {
      input,
      agent: opts.agent ?? "deep-research-preview-04-2026",
      background,
      agent_config: {
        type: "deep-research",
        thinking_summaries: opts.thinkingSummaries ?? "auto"
      },
      ...opts.tools?.length ? { tools: opts.tools.map((t) => ({ type: t })) } : {},
      ...opts.previousInteractionId ? { previous_interaction_id: opts.previousInteractionId } : {}
    };
  }
  toDeepResearchResult(i) {
    const obj = i ?? {};
    const steps = obj["steps"];
    let report = obj["output_text"] ?? obj["outputText"] ?? obj["output"];
    if (!report && Array.isArray(steps)) {
      const text = steps.flatMap((s) => Array.isArray(s.content) ? s.content : []).map((c) => c && typeof c === "object" && typeof c.text === "string" ? c.text : "").filter(Boolean).join(`

`);
      report = text || undefined;
    }
    return {
      id: obj["id"] ?? "",
      status: obj["status"] ?? "in_progress",
      report,
      steps,
      error: obj["error"],
      raw: obj
    };
  }
  async drRequest(url, init, retries = 3) {
    let lastErr;
    for (let attempt = 0;attempt <= retries; attempt++) {
      try {
        const res = await httpRequest(url, init);
        return res.data;
      } catch (e) {
        lastErr = e;
        if (attempt < retries)
          await this.delay(1500 * (attempt + 1), init.signal);
      }
    }
    throw lastErr;
  }
  async deepResearch(input, opts = {}) {
    const base = this.interactionsBase();
    const headers = this.deepResearchHeaders();
    const pollInterval = opts.pollIntervalMs ?? 5000;
    const deadline = Date.now() + (opts.timeoutMs ?? 600000);
    let interaction = await this.drRequest(base, {
      method: "POST",
      headers,
      body: this.buildInteractionBody(input, opts, true),
      timeout: this.options.timeout ?? 60000,
      signal: opts.signal
    });
    const id = interaction?.["id"];
    if (!id)
      return this.toDeepResearchResult(interaction);
    while ((interaction?.["status"] ?? "in_progress") === "in_progress") {
      if (Date.now() > deadline)
        break;
      await this.delay(pollInterval, opts.signal);
      try {
        interaction = await this.drRequest(`${base}/${id}`, { method: "GET", headers, timeout: this.options.timeout ?? 60000, signal: opts.signal }, 2);
      } catch {}
    }
    return this.toDeepResearchResult(interaction);
  }
  async* deepResearchStream(input, opts = {}) {
    const base = this.interactionsBase();
    const headers = this.deepResearchHeaders();
    const stream = httpStream(base, {
      method: "POST",
      headers,
      body: { ...this.buildInteractionBody(input, opts, true), stream: true },
      timeout: opts.timeoutMs ?? 600000,
      signal: opts.signal
    });
    let last;
    for await (const { data } of parseSSE(stream)) {
      if (!data || data === "[DONE]")
        continue;
      let parsed;
      try {
        parsed = JSON.parse(data);
      } catch {
        continue;
      }
      last = parsed;
      const delta = parsed["delta"] ?? parsed["step"]?.["delta"];
      if (delta) {
        const dtype = delta["type"];
        if (dtype === "thought")
          yield { type: "thought", content: String(delta["text"] ?? delta["content"] ?? "") };
        else if (dtype === "text")
          yield { type: "text", content: String(delta["text"] ?? delta["content"] ?? "") };
        else if (dtype === "image")
          yield { type: "image", content: delta["image"] ?? delta["content"] };
      }
      if (typeof parsed["status"] === "string")
        yield { type: "status", status: parsed["status"] };
    }
    return this.toDeepResearchResult(last);
  }
  delay(ms, signal) {
    return new Promise((resolve, reject) => {
      if (signal?.aborted)
        return reject(new Error("aborted"));
      const t = setTimeout(resolve, ms);
      signal?.addEventListener("abort", () => {
        clearTimeout(t);
        reject(new Error("aborted"));
      }, { once: true });
    });
  }
  async embed(text) {
    const url = this.getEmbedUrl();
    const response = await httpRequest(url, {
      method: "POST",
      headers: this.getHeaders(),
      body: {
        content: {
          parts: [{ text }]
        }
      },
      timeout: this.options.timeout ?? 30000
    });
    return response.data.embedding.values;
  }
  async getModels() {
    const baseUrl = this.isVertex ? `https://${this.options.region ?? "us-central1"}-aiplatform.googleapis.com/${this.apiVersion}/models` : `https://generativelanguage.googleapis.com/${this.apiVersion}/models?key=${this.options.apiKey}`;
    try {
      const response = await httpRequest(baseUrl, {
        headers: this.getHeaders(),
        timeout: 1e4
      });
      return response.data.models.map((m) => m.name.replace(/^models\//, ""));
    } catch {
      return [];
    }
  }
  buildRequestBody(messages, options) {
    const isGemma = this.options.model.toLowerCase().includes("gemma");
    const { systemInstruction, contents } = this.convertToGoogleMessages(messages, isGemma);
    const tools = options?.tools ?? (Object.keys(this.toolRegistry).length > 0 ? this.getToolDefinitions() : undefined);
    const body = {
      contents,
      generationConfig: this.buildGenerationConfig(options)
    };
    if (systemInstruction && !isGemma) {
      body.systemInstruction = {
        parts: [{ text: systemInstruction }]
      };
    }
    if (tools?.length) {
      body.tools = [{
        functionDeclarations: tools.map((t) => this.convertToGoogleTool(t))
      }];
    }
    const tier = options?.serviceTier;
    if (tier && tier !== "standard") {
      body.service_tier = tier.toUpperCase();
    }
    return body;
  }
  buildGenerationConfig(options) {
    const config = {
      ...this.options.defaultParameters,
      ...options?.parameters
    };
    if (options?.temperature !== undefined)
      config["temperature"] = options.temperature;
    if (options?.maxTokens !== undefined)
      config["maxOutputTokens"] = options.maxTokens;
    const thinking = resolveThinking(options?.thinking, this.options.thinking);
    if (thinking && config["thinkingConfig"] === undefined) {
      if (/gemini-3/i.test(this.options.model)) {
        const tc = {};
        if (!thinking.enabled) {
          tc["thinkingLevel"] = "MINIMAL";
        } else {
          if (thinking.level)
            tc["thinkingLevel"] = thinking.level.toUpperCase();
          tc["includeThoughts"] = true;
        }
        config["thinkingConfig"] = tc;
      } else {
        config["thinkingConfig"] = thinking.enabled ? { thinkingBudget: geminiThinkingBudget(thinking.level), includeThoughts: true } : { thinkingBudget: 0 };
      }
    }
    const schemaOptions = this.extractSchemaOptions(options);
    if (schemaOptions) {
      config["responseMimeType"] = "application/json";
      let jsonSchema;
      if (schemaOptions.jsonSchema) {
        jsonSchema = normalizeJsonSchema(schemaOptions.jsonSchema);
      } else if (schemaOptions.schemaConfig) {
        jsonSchema = getJsonSchemaFromConfig(schemaOptions.schemaConfig);
      } else {
        throw new Error("Either schemaConfig or jsonSchema must be provided");
      }
      const googleSchema = stripUnsupportedFeatures(jsonSchema, "google");
      config["responseSchema"] = googleSchema;
    }
    return config;
  }
  convertToGoogleMessages(messages, isGemma) {
    let systemInstruction;
    const contents = [];
    for (const msg of messages) {
      if (msg.role === "system") {
        if (isGemma) {
          systemInstruction = typeof msg.content === "string" ? msg.content : msg.content.filter((p) => p.type === "text").map((p) => p.text).join("");
        } else {
          systemInstruction = typeof msg.content === "string" ? msg.content : msg.content.filter((p) => p.type === "text").map((p) => p.text).join("");
        }
        continue;
      }
      if (msg.role === "tool") {
        let responseData;
        try {
          responseData = typeof msg.content === "string" ? JSON.parse(msg.content) : { result: msg.content };
        } catch {
          responseData = { result: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content) };
        }
        contents.push({
          role: "function",
          parts: [{
            functionResponse: {
              name: msg.tool_call_id ?? "unknown",
              response: responseData
            }
          }]
        });
        continue;
      }
      if (msg.role === "assistant") {
        const parts = [];
        const textContent = typeof msg.content === "string" ? msg.content : "";
        if (textContent)
          parts.push({ text: textContent });
        if (msg.tool_calls) {
          for (const tc of msg.tool_calls) {
            const part = {
              functionCall: {
                name: tc.function.name,
                args: this.parseToolArguments(tc.function.arguments)
              }
            };
            if (tc.thoughtSignature) {
              part.thoughtSignature = tc.thoughtSignature;
            }
            parts.push(part);
          }
        }
        contents.push({ role: "model", parts });
        continue;
      }
      const parts = this.convertContentToGoogleParts(msg.content);
      if (isGemma && systemInstruction && contents.length === 0) {
        const systemParts = [{ text: `[System Instructions]
${systemInstruction}

[User Message]
` }];
        contents.push({
          role: "user",
          parts: [...systemParts, ...parts]
        });
        systemInstruction = undefined;
      } else {
        contents.push({ role: "user", parts });
      }
    }
    return { systemInstruction, contents };
  }
  convertContentToGoogleParts(content) {
    if (typeof content === "string") {
      return [{ text: content }];
    }
    return content.map((part) => {
      if (part.type === "text") {
        return { text: part.text };
      }
      if (part.type === "audio") {
        return {
          inlineData: {
            mimeType: part.audio.mimeType,
            data: part.audio.data
          }
        };
      }
      const url = part.image_url.url;
      if (url.startsWith("data:")) {
        const match = url.match(/^data:([^;]+);base64,(.+)$/);
        if (match) {
          return {
            inlineData: {
              mimeType: match[1],
              data: match[2]
            }
          };
        }
      }
      return { text: `[Image: ${url}]` };
    });
  }
  convertToGoogleTool(tool) {
    return {
      name: tool.function.name,
      description: tool.function.description,
      parameters: {
        type: "object",
        properties: tool.function.parameters.properties ?? {},
        required: tool.function.parameters.required
      }
    };
  }
  convertFunctionCallToToolCall(fc, thoughtSignature) {
    const toolCall = {
      id: this.generateToolCallId(),
      type: "function",
      function: {
        name: fc.name || "",
        arguments: JSON.stringify(fc.args ?? {})
      }
    };
    if (thoughtSignature) {
      toolCall.thoughtSignature = thoughtSignature;
    }
    return toolCall;
  }
  parseToolArguments(args) {
    if (typeof args !== "string") {
      return args ?? {};
    }
    if (args.length === 0) {
      return {};
    }
    try {
      const parsed = JSON.parse(args);
      return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
    } catch {
      return {};
    }
  }
  parseGoogleResponse(data) {
    const candidate = data.candidates?.[0];
    if (!candidate?.content?.parts) {
      return {
        message: { role: "assistant", content: "" },
        provider: this.isVertex ? "vertex" : "google"
      };
    }
    let textContent = "";
    let reasoningText = "";
    const toolCalls = [];
    for (const part of candidate.content.parts) {
      if (part.text) {
        if (part.thought)
          reasoningText += part.text;
        else
          textContent += part.text;
      }
      if (part.functionCall) {
        toolCalls.push(this.convertFunctionCallToToolCall(part.functionCall, part.thoughtSignature));
      }
    }
    const usage = data.usageMetadata ? {
      inputTokens: data.usageMetadata.promptTokenCount,
      outputTokens: data.usageMetadata.candidatesTokenCount,
      totalTokens: data.usageMetadata.totalTokenCount,
      cachedTokens: data.usageMetadata.cachedContentTokenCount,
      reasoningTokens: data.usageMetadata.thoughtsTokenCount
    } : undefined;
    return {
      message: {
        role: "assistant",
        content: textContent,
        tool_calls: toolCalls.length > 0 ? toolCalls : undefined
      },
      reasoning: reasoningText || undefined,
      usage,
      provider: this.isVertex ? "vertex" : "google"
    };
  }
  async fetchWithFlexRetry(url, reqOptions, maxRetries = 3, baseDelay = 5000) {
    for (let attempt = 0;attempt < maxRetries; attempt++) {
      try {
        return await httpRequest(url, reqOptions);
      } catch (error) {
        const isRetryable = error instanceof Error && (error.message.includes("HTTP 503") || error.message.includes("HTTP 429"));
        if (!isRetryable || attempt >= maxRetries - 1)
          throw error;
        const delay = baseDelay * 2 ** attempt;
        await new Promise((r) => setTimeout(r, delay));
      }
    }
    throw new Error("Unreachable");
  }
}

// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/providers/anthropic.js
class AnthropicClient extends BaseLLMClient {
  baseUrl;
  constructor(options, auditor) {
    const url = (options.url || "https://api.anthropic.com").replace(/\/+$/, "");
    super({ ...options, url }, auditor);
    this.baseUrl = url;
  }
  buildAnthropicHeaders() {
    const headers = {
      "Content-Type": "application/json",
      "anthropic-version": "2023-06-01"
    };
    if (this.options.apiKey) {
      headers["x-api-key"] = this.options.apiKey;
    }
    return headers;
  }
  async chat(messages, options) {
    const url = `${this.baseUrl}/v1/messages`;
    const body = this.buildRequestBody(messages, options, false);
    const start = Date.now();
    this.auditor.record({
      timestamp: start,
      type: "request",
      provider: "anthropic",
      model: this.options.model
    });
    const response = await httpRequest(url, {
      method: "POST",
      headers: this.buildAnthropicHeaders(),
      body,
      timeout: this.options.timeout ?? 60000
    });
    const data = response.data;
    const result = this.parseAnthropicResponse(data);
    this.auditor.record({
      timestamp: Date.now(),
      type: "response",
      provider: "anthropic",
      model: this.options.model,
      duration: Date.now() - start,
      usage: result.usage
    });
    return result;
  }
  async* chatStream(messages, options) {
    const url = `${this.baseUrl}/v1/messages`;
    const body = this.buildRequestBody(messages, options, true);
    const start = Date.now();
    this.auditor.record({
      timestamp: start,
      type: "stream_start",
      provider: "anthropic",
      model: this.options.model
    });
    const decoder = new StandardChatDecoder(() => {});
    const contentBlocks = new Map;
    let usage;
    let inputTokens = 0;
    const stream = httpStream(url, {
      method: "POST",
      headers: this.buildAnthropicHeaders(),
      body,
      timeout: this.options.timeout ?? 120000
    });
    for await (const { data } of parseSSE(stream)) {
      try {
        const event = JSON.parse(data);
        switch (event.type) {
          case "message_start": {
            inputTokens = event.message.usage?.input_tokens ?? 0;
            break;
          }
          case "content_block_start": {
            const block = event.content_block;
            if (block.type === "text") {
              contentBlocks.set(event.index, { type: "text", text: "" });
            } else if (block.type === "tool_use") {
              contentBlocks.set(event.index, {
                type: "tool_use",
                text: "",
                toolId: block.id,
                toolName: block.name,
                inputJson: ""
              });
            } else if (block.type === "thinking") {
              contentBlocks.set(event.index, { type: "thinking", text: "", thinking: "" });
            }
            break;
          }
          case "content_block_delta": {
            const block = contentBlocks.get(event.index);
            if (!block)
              break;
            if (event.delta.type === "text_delta") {
              block.text += event.delta.text;
              decoder.push(event.delta.text);
              yield { type: "text", content: event.delta.text };
            } else if (event.delta.type === "input_json_delta") {
              block.inputJson = (block.inputJson ?? "") + event.delta.partial_json;
            } else if (event.delta.type === "thinking_delta") {
              block.thinking = (block.thinking ?? "") + event.delta.thinking;
              decoder.pushReasoning(event.delta.thinking);
              yield { type: "thinking", content: event.delta.thinking };
            } else if (event.delta.type === "signature_delta") {
              block.signature = event.delta.signature;
            }
            break;
          }
          case "content_block_stop": {
            const block = contentBlocks.get(event.index);
            if (block?.type === "tool_use" && block.toolId && block.toolName) {
              const toolCall = {
                id: block.toolId,
                type: "function",
                function: {
                  name: block.toolName,
                  arguments: block.inputJson ?? "{}"
                }
              };
              yield { type: "tool_call", calls: [toolCall] };
            }
            break;
          }
          case "message_delta": {
            const outputTokens = event.usage?.output_tokens ?? 0;
            usage = {
              inputTokens,
              outputTokens,
              totalTokens: inputTokens + outputTokens
            };
            break;
          }
          case "error": {
            throw new Error(`Anthropic stream error: ${event.error.type} — ${event.error.message}`);
          }
        }
      } catch (e) {
        if (e instanceof Error && e.message.startsWith("Anthropic stream error")) {
          throw e;
        }
      }
    }
    decoder.flush();
    this.auditor.record({
      timestamp: Date.now(),
      type: "stream_end",
      provider: "anthropic",
      model: this.options.model,
      duration: Date.now() - start,
      usage
    });
    const toolCalls = [];
    for (const block of contentBlocks.values()) {
      if (block.type === "tool_use" && block.toolId && block.toolName) {
        toolCalls.push({
          id: block.toolId,
          type: "function",
          function: {
            name: block.toolName,
            arguments: block.inputJson ?? "{}"
          }
        });
      }
    }
    return {
      message: {
        role: "assistant",
        content: decoder.getCleanContent(),
        tool_calls: toolCalls.length > 0 ? toolCalls : undefined
      },
      reasoning: decoder.getReasoning(),
      usage,
      provider: "anthropic"
    };
  }
  async embed(_text) {
    throw new Error("Anthropic does not support embeddings. Use a different provider.");
  }
  async getModels() {
    const url = `${this.baseUrl}/v1/models`;
    try {
      const response = await httpRequest(url, {
        headers: this.buildAnthropicHeaders(),
        timeout: 5000
      });
      return response.data.data.map((m) => m.id);
    } catch {
      return [
        "claude-sonnet-4-20250514",
        "claude-haiku-4-20250514",
        "claude-opus-4-20250514"
      ];
    }
  }
  async getModelInfo(_modelName) {
    const model = _modelName ?? this.options.model;
    if (model.includes("claude-4") || model.includes("claude-opus") || model.includes("claude-sonnet") || model.includes("claude-haiku")) {
      return {
        model,
        contextLength: 200000,
        capabilities: ["tools", "vision", "thinking"]
      };
    }
    return {
      model,
      contextLength: 200000,
      capabilities: ["tools", "vision"]
    };
  }
  buildRequestBody(messages, options, stream) {
    const systemMessages = messages.filter((m) => m.role === "system");
    const nonSystemMessages = messages.filter((m) => m.role !== "system");
    const systemPrompt = systemMessages.length > 0 ? systemMessages.map((m) => typeof m.content === "string" ? m.content : this.extractText(m.content)).join(`

`) : undefined;
    let system = systemPrompt;
    if (options?.enablePromptCaching && systemPrompt) {
      system = [
        {
          type: "text",
          text: systemPrompt,
          cache_control: { type: "ephemeral" }
        }
      ];
    }
    const tools = options?.tools ?? (Object.keys(this.toolRegistry).length > 0 ? this.getToolDefinitions() : undefined);
    const anthropicTools = tools?.map((t) => this.convertToolDef(t));
    let toolChoice;
    if (options?.toolChoice === "required") {
      toolChoice = { type: "any" };
    } else if (options?.toolChoice === "none") {
      toolChoice = { type: "auto" };
    } else if (options?.toolChoice === "auto") {
      toolChoice = { type: "auto" };
    }
    const thinking = resolveThinking(options?.thinking, this.options.thinking);
    const thinkingOn = thinking?.enabled === true;
    const requestedMax = options?.maxTokens ?? 4096;
    const budget = thinkingOn ? anthropicThinkingBudget(thinking?.level, requestedMax) : 0;
    const maxTokens = thinkingOn ? Math.max(requestedMax, budget + 1024) : requestedMax;
    const body = {
      model: this.options.model,
      messages: this.convertMessages(nonSystemMessages),
      max_tokens: maxTokens,
      ...system && { system },
      ...anthropicTools?.length && { tools: anthropicTools },
      ...toolChoice && { tool_choice: toolChoice },
      ...stream && { stream: true },
      ...thinkingOn ? { thinking: { type: "enabled", budget_tokens: budget } } : options?.temperature !== undefined && { temperature: options.temperature }
    };
    return body;
  }
  convertMessages(messages) {
    const result = [];
    for (let i = 0;i < messages.length; i++) {
      const msg = messages[i];
      if (msg.role === "assistant") {
        const blocks = [];
        const text = typeof msg.content === "string" ? msg.content : this.extractText(msg.content);
        if (text) {
          blocks.push({ type: "text", text });
        }
        if (msg.tool_calls) {
          for (const tc of msg.tool_calls) {
            let input = {};
            try {
              input = JSON.parse(tc.function.arguments);
            } catch {}
            blocks.push({
              type: "tool_use",
              id: tc.id,
              name: tc.function.name,
              input
            });
          }
        }
        if (blocks.length > 0) {
          result.push({ role: "assistant", content: blocks });
        }
      } else if (msg.role === "tool") {
        const toolResultBlock = {
          type: "tool_result",
          tool_call_id: msg.tool_call_id ?? "",
          content: typeof msg.content === "string" ? msg.content : this.extractText(msg.content)
        };
        const toolResults = [toolResultBlock];
        while (i + 1 < messages.length && messages[i + 1].role === "tool") {
          i++;
          const nextMsg = messages[i];
          toolResults.push({
            type: "tool_result",
            tool_call_id: nextMsg.tool_call_id ?? "",
            content: typeof nextMsg.content === "string" ? nextMsg.content : this.extractText(nextMsg.content)
          });
        }
        result.push({ role: "user", content: toolResults });
      } else if (msg.role === "user") {
        const blocks = this.convertUserContent(msg.content);
        result.push({ role: "user", content: blocks });
      }
    }
    return this.ensureAlternating(result);
  }
  convertUserContent(content) {
    if (typeof content === "string") {
      return [{ type: "text", text: content }];
    }
    const blocks = [];
    for (const part of content) {
      if (part.type === "text") {
        blocks.push({ type: "text", text: part.text });
      } else if (part.type === "audio") {
        this.debugLog("[Anthropic] Audio content dropped — not supported");
      } else if (part.type === "image_url") {
        const url = part.image_url.url;
        if (url.startsWith("data:")) {
          const match = url.match(/^data:([^;]+);base64,(.+)$/);
          if (match) {
            blocks.push({
              type: "image",
              source: {
                type: "base64",
                media_type: match[1],
                data: match[2]
              }
            });
          }
        } else {
          blocks.push({
            type: "image",
            source: {
              type: "url",
              url
            }
          });
        }
      }
    }
    return blocks.length > 0 ? blocks : [{ type: "text", text: "" }];
  }
  ensureAlternating(messages) {
    if (messages.length <= 1)
      return messages;
    const merged = [messages[0]];
    for (let i = 1;i < messages.length; i++) {
      const current = messages[i];
      const last = merged[merged.length - 1];
      if (current.role === last.role) {
        const lastContent = Array.isArray(last.content) ? last.content : [{ type: "text", text: last.content }];
        const currentContent = Array.isArray(current.content) ? current.content : [{ type: "text", text: current.content }];
        merged[merged.length - 1] = {
          role: current.role,
          content: [...lastContent, ...currentContent]
        };
      } else {
        merged.push(current);
      }
    }
    return merged;
  }
  parseAnthropicResponse(data) {
    let textContent = "";
    let reasoning;
    const toolCalls = [];
    for (const block of data.content) {
      if (block.type === "text") {
        textContent += block.text;
      } else if (block.type === "tool_use") {
        toolCalls.push({
          id: block.id,
          type: "function",
          function: {
            name: block.name,
            arguments: JSON.stringify(block.input)
          }
        });
      } else if (block.type === "thinking") {
        reasoning = (reasoning ?? "") + block.thinking;
      }
    }
    const usage = {
      inputTokens: data.usage.input_tokens,
      outputTokens: data.usage.output_tokens,
      totalTokens: data.usage.input_tokens + data.usage.output_tokens
    };
    return {
      message: {
        role: "assistant",
        content: textContent,
        tool_calls: toolCalls.length > 0 ? toolCalls : undefined
      },
      reasoning,
      usage,
      provider: "anthropic"
    };
  }
  convertToolDef(tool) {
    return {
      name: tool.function.name,
      description: tool.function.description,
      input_schema: {
        type: "object",
        properties: tool.function.parameters.properties,
        required: tool.function.parameters.required
      }
    };
  }
  extractText(content) {
    if (typeof content === "string")
      return content;
    return content.filter((p) => p.type === "text").map((p) => p.text).join("");
  }
}

// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/ai-model.js
var DEFAULT_URLS = {
  ollama: "http://localhost:11434",
  openai: "https://api.openai.com",
  llamacpp: "http://localhost:8080",
  anthropic: "https://api.anthropic.com"
};

class AIModel {
  router;
  auditor;
  config;
  constructor(config) {
    this.config = config;
    this.auditor = config.auditor ?? new NoopAuditor;
    const routerConfig = {
      retriesPerProvider: config.retries ?? 2,
      ...config.spillAfterMs !== undefined ? { spillAfterMs: config.spillAfterMs } : {},
      auditor: this.auditor
    };
    this.router = new Router(routerConfig);
    for (let i = 0;i < config.providers.length; i++) {
      const providerConfig = config.providers[i];
      const client = this.createClient(providerConfig);
      const id = `${this.normalizeType(providerConfig.type)}-${i}`;
      this.router.addProvider({
        id,
        client,
        priority: providerConfig.priority ?? i,
        modelOverride: providerConfig.model,
        ...providerConfig.maxConcurrent !== undefined ? { maxConcurrent: providerConfig.maxConcurrent } : {}
      });
    }
  }
  async chat(messages, options) {
    return this.router.chat(messages, options);
  }
  async chatWithTools(messages, options) {
    return this.router.chatWithTools(messages, options);
  }
  async* chatStream(messages, options) {
    return yield* this.router.chatStream(messages, options);
  }
  async generateStructured(config, messages, options) {
    return this.router.generateStructured(config, messages, options);
  }
  async tryParseStructured(config, messages, options) {
    return this.router.tryParseStructured(config, messages, options);
  }
  async* generateStructuredStream(config, messages, options) {
    return yield* this.router.generateStructuredStream(config, messages, options);
  }
  async embed(text) {
    return this.router.embed(text);
  }
  async embedArray(texts) {
    return this.router.embedArray(texts);
  }
  getGoogleClient(method) {
    const googleClients = this.router.getClients().filter((c) => c instanceof GoogleClient);
    const aiStudio = googleClients.find((c) => c.supportsDeepResearch());
    if (aiStudio)
      return aiStudio;
    if (googleClients.length > 0) {
      throw new Error(`${method} requires an AI Studio Google provider (type: "google"); Vertex AI is not supported for Deep Research.`);
    }
    throw new Error(`${method} requires a Google provider (type: "google"). None is configured.`);
  }
  async deepResearch(input, options) {
    return this.getGoogleClient("deepResearch").deepResearch(input, options);
  }
  async* deepResearchStream(input, options) {
    return yield* this.getGoogleClient("deepResearchStream").deepResearchStream(input, options);
  }
  registerTool(name, description, parameters, handler) {
    this.router.registerTool(name, description, parameters, handler);
  }
  registerTools(tools) {
    this.router.registerTools(tools);
  }
  async getModels() {
    return this.router.getModels();
  }
  async getModelInfo() {
    return this.router.getModelInfo();
  }
  setModel(name) {
    this.config.model = name;
  }
  get model() {
    return this.config.model;
  }
  getProviderStatus() {
    return this.router.getStatus();
  }
  getPoolStatus() {
    return this.router.getPoolStatus();
  }
  async dispose() {
    await this.auditor.flush?.();
  }
  createClient(providerConfig) {
    const type = this.normalizeType(providerConfig.type);
    const modelName = providerConfig.model ?? this.config.model;
    const defaultParameters = providerConfig.defaultParameters ? { ...this.config.defaultParameters, ...providerConfig.defaultParameters } : this.config.defaultParameters;
    const clientOptions = {
      model: modelName,
      url: providerConfig.url ?? DEFAULT_URLS[type] ?? "",
      apiType: type,
      apiKey: providerConfig.apiKey,
      timeout: this.config.timeout ?? 30000,
      retries: this.config.retries ?? 2,
      debug: this.config.debug ?? false,
      defaultParameters,
      contextLength: providerConfig.contextLength ?? this.config.contextLength,
      thinking: this.config.thinking,
      region: providerConfig.region,
      apiVersion: providerConfig.apiVersion,
      extraHeaders: providerConfig.headers,
      queryParams: providerConfig.queryParams,
      authHeader: providerConfig.authHeader,
      authPrefix: providerConfig.authPrefix,
      apiBasePath: providerConfig.apiBasePath,
      omitResponseFormat: providerConfig.omitResponseFormat
    };
    switch (type) {
      case "ollama":
        return new OllamaClient(clientOptions, this.auditor);
      case "openai":
      case "llamacpp":
        return new OpenAICompatibleClient(clientOptions, this.auditor);
      case "google":
      case "vertex":
        return new GoogleClient(clientOptions, this.auditor);
      case "anthropic":
        return new AnthropicClient(clientOptions, this.auditor);
      default:
        throw new Error(`Unknown provider type: ${type}`);
    }
  }
  normalizeType(type) {
    return type.toLowerCase();
  }
}
// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/interfaces.js
var AIModelType;
(function(AIModelType) {
  AIModelType["Chat"] = "chat";
  AIModelType["Embedding"] = "embedding";
})(AIModelType || (AIModelType = {}));
var AIModelApiType;
(function(AIModelApiType) {
  AIModelApiType["Ollama"] = "ollama";
  AIModelApiType["OpenAI"] = "openai";
  AIModelApiType["Google"] = "google";
  AIModelApiType["Vertex"] = "vertex";
  AIModelApiType["LlamaCpp"] = "llamacpp";
  AIModelApiType["Anthropic"] = "anthropic";
})(AIModelApiType || (AIModelApiType = {}));
function textContent(text) {
  return { type: "text", text };
}
function imageContent(base64DataOrUrl, mimeType = "image/jpeg", detail) {
  const url = base64DataOrUrl.startsWith("data:") || base64DataOrUrl.startsWith("http") ? base64DataOrUrl : `data:${mimeType};base64,${base64DataOrUrl}`;
  return {
    type: "image_url",
    image_url: { url, detail }
  };
}
function multimodalMessage(text, images, mimeType = "image/jpeg") {
  const content = [
    textContent(text),
    ...images.map((img) => imageContent(img, mimeType))
  ];
  return { role: "user", content };
}
function extractTextContent(content) {
  if (typeof content === "string")
    return content;
  return content.filter((part) => part.type === "text").map((part) => part.text).join("");
}
function hasImages(content) {
  if (typeof content === "string")
    return false;
  return content.some((part) => part.type === "image_url");
}
function audioContent(base64Data, mimeType) {
  return {
    type: "audio",
    audio: { data: base64Data, mimeType }
  };
}
function hasAudio(content) {
  if (typeof content === "string")
    return false;
  return content.some((part) => part.type === "audio");
}
// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/tools.js
class ToolBuilder {
  name;
  desc = "";
  properties = {};
  required = [];
  constructor(name) {
    this.name = name;
  }
  description(desc) {
    this.desc = desc;
    return this;
  }
  addParameter(name, type, description, isRequired = false, extra) {
    this.properties[name] = {
      type,
      description,
      ...extra
    };
    if (isRequired) {
      this.required.push(name);
    }
    return this;
  }
  build() {
    return {
      type: "function",
      function: {
        name: this.name,
        description: this.desc,
        parameters: {
          type: "object",
          properties: this.properties,
          required: this.required.length > 0 ? this.required : undefined
        }
      }
    };
  }
  buildFunction() {
    return this.build().function;
  }
}

class ToolExecutor {
  static withTimeout(handler, timeoutMs) {
    return async (args) => {
      const result = await Promise.race([
        Promise.resolve(handler(args)),
        new Promise((_, reject) => setTimeout(() => reject(new Error(`Tool execution timeout after ${timeoutMs}ms`)), timeoutMs))
      ]);
      return result;
    };
  }
  static safe(handler) {
    return async (args) => {
      try {
        return await handler(args);
      } catch (error) {
        return {
          error: error instanceof Error ? error.message : String(error)
        };
      }
    };
  }
  static withValidation(handler, requiredFields) {
    return async (args) => {
      if (!args || typeof args !== "object") {
        throw new Error("Tool arguments must be an object");
      }
      const obj = args;
      for (const field of requiredFields) {
        if (obj[field] === undefined || obj[field] === null) {
          throw new Error(`Missing required argument: ${field}`);
        }
      }
      return handler(args);
    };
  }
  static timed(handler) {
    return async (args) => {
      const start = Date.now();
      const result = await handler(args);
      return {
        result,
        duration: Date.now() - start
      };
    };
  }
  static compose(handler, ...wrappers) {
    return wrappers.reduceRight((h, wrapper) => wrapper(h), handler);
  }
}
function createTimeTool() {
  return {
    name: "get_current_time",
    description: "Get the current date and time",
    parameters: {
      type: "object",
      properties: {
        timezone: {
          type: "string",
          description: 'IANA timezone (e.g. "America/New_York"). Defaults to UTC.'
        }
      }
    },
    handler: (args) => {
      const { timezone } = args ?? {};
      const now = new Date;
      try {
        return {
          iso: now.toISOString(),
          formatted: now.toLocaleString("en-US", {
            timeZone: timezone || "UTC",
            dateStyle: "full",
            timeStyle: "long"
          }),
          timezone: timezone || "UTC",
          timestamp: now.getTime()
        };
      } catch {
        return {
          iso: now.toISOString(),
          formatted: now.toUTCString(),
          timezone: "UTC",
          timestamp: now.getTime()
        };
      }
    }
  };
}
function createRandomNumberTool() {
  return {
    name: "random_number",
    description: "Generate a random number within a range",
    parameters: {
      type: "object",
      properties: {
        min: { type: "number", description: "Minimum value (default 0)" },
        max: { type: "number", description: "Maximum value (default 100)" }
      }
    },
    handler: (args) => {
      const { min = 0, max = 100 } = args ?? {};
      return {
        value: Math.floor(Math.random() * (max - min + 1)) + min,
        min,
        max
      };
    }
  };
}
// ../../C:/Users/igorl/AppData/Local/Temp/claude/72a3105a48ca607f/422c2632-2fd5-4896-8e78-0d5ebdb710e2/scratchpad/ulc/package/dist/mcp.js
class MCPToolBridge {
  config;
  auditor;
  clients = new Map;
  transports = new Map;
  discoveredTools = [];
  connected = false;
  constructor(config) {
    this.config = config;
    this.auditor = config.auditor ?? new NoopAuditor;
  }
  async connect() {
    if (this.connected)
      return;
    let sdk;
    try {
      sdk = await import("@modelcontextprotocol/sdk/client/index.js");
    } catch {
      throw new Error(`MCP integration requires @modelcontextprotocol/sdk.
Install it: bun add @modelcontextprotocol/sdk`);
    }
    const entries = Object.entries(this.config.servers);
    for (const [serverName, serverConfig] of entries) {
      try {
        const client = new sdk.Client({ name: "universal-llm-client", version: "3.0.0" }, { capabilities: {} });
        let transport;
        if (serverConfig.url) {
          if (!sdk.StreamableHTTPClientTransport) {
            const httpModule = await import("@modelcontextprotocol/sdk/client/streamableHttp.js").catch(() => null);
            if (!httpModule?.StreamableHTTPClientTransport) {
              throw new Error("StreamableHTTPClientTransport not available in this SDK version");
            }
            transport = new httpModule.StreamableHTTPClientTransport(new URL(serverConfig.url), serverConfig.headers ? { requestInit: { headers: serverConfig.headers } } : undefined);
          } else {
            transport = new sdk.StreamableHTTPClientTransport(new URL(serverConfig.url), serverConfig.headers ? { requestInit: { headers: serverConfig.headers } } : undefined);
          }
        } else if (serverConfig.command) {
          if (!sdk.StdioClientTransport) {
            const stdioModule = await import("@modelcontextprotocol/sdk/client/stdio.js").catch(() => null);
            if (!stdioModule?.StdioClientTransport) {
              throw new Error(`Stdio transport not available. This is expected in browser environments.
Use HTTP transport (url) instead, or run in Node/Bun/Deno.`);
            }
            transport = new stdioModule.StdioClientTransport({
              command: serverConfig.command,
              args: serverConfig.args,
              env: serverConfig.env
            });
          } else {
            transport = new sdk.StdioClientTransport({
              command: serverConfig.command,
              args: serverConfig.args,
              env: serverConfig.env
            });
          }
        } else {
          throw new Error(`MCP server "${serverName}" must have either "url" or "command" configured`);
        }
        await client.connect(transport);
        this.clients.set(serverName, client);
        this.transports.set(serverName, transport);
        const toolList = await client.listTools();
        for (const tool of toolList.tools) {
          this.discoveredTools.push({
            serverName,
            name: tool.name,
            qualifiedName: `${serverName}:${tool.name}`,
            description: tool.description ?? "",
            inputSchema: tool.inputSchema
          });
        }
        this.auditor.record({
          timestamp: Date.now(),
          type: "response",
          metadata: {
            event: "mcp_connected",
            server: serverName,
            toolCount: toolList.tools.length
          }
        });
      } catch (error) {
        this.auditor.record({
          timestamp: Date.now(),
          type: "error",
          error: error instanceof Error ? error.message : String(error),
          metadata: { event: "mcp_connect_failed", server: serverName }
        });
        throw error;
      }
    }
    this.connected = true;
  }
  async registerTools(model) {
    if (!this.connected) {
      await this.connect();
    }
    const tools = this.discoveredTools.map((mcpTool) => ({
      name: mcpTool.qualifiedName,
      description: mcpTool.description,
      parameters: this.convertInputSchema(mcpTool.inputSchema),
      handler: this.createToolHandler(mcpTool)
    }));
    model.registerTools(tools);
  }
  getTools() {
    return this.discoveredTools;
  }
  async disconnect() {
    for (const [serverName, client] of this.clients) {
      try {
        await client.close();
      } catch (error) {
        this.auditor.record({
          timestamp: Date.now(),
          type: "error",
          error: error instanceof Error ? error.message : String(error),
          metadata: { event: "mcp_disconnect_failed", server: serverName }
        });
      }
    }
    this.clients.clear();
    this.transports.clear();
    this.discoveredTools = [];
    this.connected = false;
  }
  createToolHandler(mcpTool) {
    return async (args) => {
      const client = this.clients.get(mcpTool.serverName);
      if (!client) {
        throw new Error(`MCP server "${mcpTool.serverName}" is not connected`);
      }
      const start = Date.now();
      this.auditor.record({
        timestamp: start,
        type: "tool_call",
        metadata: {
          event: "mcp_tool_call",
          server: mcpTool.serverName,
          tool: mcpTool.name
        }
      });
      try {
        const result = await client.callTool({
          name: mcpTool.name,
          arguments: args
        });
        const textParts = result.content.filter((c) => c.type === "text" && c.text).map((c) => c.text);
        const output = textParts.length === 1 ? textParts[0] : textParts.join(`
`);
        this.auditor.record({
          timestamp: Date.now(),
          type: "tool_result",
          metadata: {
            event: "mcp_tool_result",
            server: mcpTool.serverName,
            tool: mcpTool.name,
            duration: Date.now() - start
          }
        });
        return output;
      } catch (error) {
        this.auditor.record({
          timestamp: Date.now(),
          type: "error",
          error: error instanceof Error ? error.message : String(error),
          metadata: {
            event: "mcp_tool_error",
            server: mcpTool.serverName,
            tool: mcpTool.name
          }
        });
        throw error;
      }
    };
  }
  convertInputSchema(schema) {
    return {
      type: "object",
      properties: schema["properties"] ?? {},
      required: schema["required"]
    };
  }
}
export {
  AIModel,
  AIModelApiType,
  AIModelType,
  BufferedAuditor,
  ConsoleAuditor,
  InterleavedReasoningDecoder,
  LLMHttpError,
  LLMProviderError,
  MCPToolBridge,
  NoopAuditor,
  PassthroughDecoder,
  StandardChatDecoder,
  StreamLoopGuard,
  StreamingJsonParser,
  StructuredOutputError,
  ToolBuilder,
  ToolExecutor,
  applyGemmaDualModeRequestDefaults,
  audioContent,
  buildHeaders,
  classifyFailure,
  coerceAndValidateToolArgs,
  collapseRepeatedRuns,
  collapseRepeatsInToolArguments,
  convertToProviderSchema,
  createDecoder,
  createRandomNumberTool,
  createTimeTool,
  extractProviderErrorMessage,
  extractTextContent,
  gemmaArgsToJson,
  getJsonSchema,
  getJsonSchemaFromConfig,
  getRegisteredDecoders,
  hasAudio,
  hasImages,
  httpRequest,
  httpStream,
  imageContent,
  inferOpenAICompatCapabilities,
  isConnectivityFailure,
  isGemmaDiffusionModel,
  isGemmaModelId,
  isStructuredOutputFailure,
  isStructuredOutputSuccess,
  looksLikeErrorPayload,
  multimodalMessage,
  normalizeJsonSchema,
  parseGemmaDiffusionOutput,
  parseNDJSON,
  parseSSE,
  parseStructured,
  parseToolArguments,
  registerDecoder,
  shouldOmitResponseFormatWire,
  stripJsonFences,
  stripUnsupportedFeatures,
  textContent,
  tryParseStructured,
  validateStructuredOutput
};
