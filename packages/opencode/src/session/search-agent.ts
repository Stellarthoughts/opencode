import { Config } from "@/config/config"
import { Installation } from "@/installation"
import { MCP } from "@/mcp"
import { Provider } from "@/provider/provider"
import { ProviderTransform } from "@/provider/transform"
import { Log } from "@/util/log"
import { generateText, jsonSchema, tool, type ModelMessage, type Tool } from "ai"
import { mergeDeep } from "remeda"
import type { MessageV2 } from "./message-v2"

export namespace SearchAgent {
  const log = Log.create({ service: "session.search-agent" })
  const cache = new Map<string, string>()

  const fallback = {
    type: "object",
    additionalProperties: false,
    properties: {
      searchQueries: {
        type: "array",
        items: { type: "string" },
      },
      recallQueries: {
        type: "array",
        items: { type: "string" },
      },
      queries: {
        type: "array",
        items: {
          type: "object",
          additionalProperties: false,
          properties: {
            query: {
              type: "string",
            },
            mode: {
              type: "string",
              enum: ["search", "recall", "auto", "flash"],
            },
            depth: {
              type: "string",
              enum: ["snippet", "summary", "full"],
            },
            results: {
              type: "number",
            },
          },
          required: ["query", "mode"],
        },
      },
      searchMode: {
        type: "string",
        enum: ["auto", "search", "recall", "flash"],
      },
      results: {
        type: "number",
      },
      depth: {
        type: "string",
        enum: ["snippet", "summary", "full"],
      },
    },
  }

  export type Input = {
    sessionID: string
    messages: MessageV2.WithParts[]
    config?: Config.SearchAgent
  }

  export type Output = {
    tool: string
    input: Record<string, unknown>
    output: string
    metadata: Record<string, unknown>
  }

  type Query = {
    query: string
    mode: string
    depth?: string
    results?: number
  }

  export async function execute(input: Input): Promise<Output | undefined> {
    if (!input.config?.enabled) return
    return run(input).catch((error) => {
      log.warn("search agent failed", {
        sessionID: input.sessionID,
        error,
      })
      return undefined
    })
  }

  async function run(input: Input): Promise<Output | undefined> {
    const cfg = input.config
    if (!cfg) return
    const msgs = filter(input.messages, new Set(cfg.messageFilter.includeToolResults))
    if (msgs.length === 0) return
    log.info("search agent executing", {
      sessionID: input.sessionID,
      messages: msgs.length,
    })

    const parsed = Provider.parseModel(cfg.model)
    const model = await Provider.getModel(parsed.providerID, parsed.modelID)
    const language = await Provider.getLanguage(model)
    const provider = await Provider.getProvider(model.providerID)
    const system = await boot(input.sessionID, cfg)

    const base = ProviderTransform.options({
      model,
      sessionID: input.sessionID,
      providerOptions: provider?.options,
    })

    const options = mergeDeep(base, model.options)
    const disable =
      model.api.npm === "@ai-sdk/anthropic" ||
      model.api.npm === "@ai-sdk/google-vertex/anthropic" ||
      model.api.id.includes("claude")
    const timeout = Math.max(1, cfg.timeout)
    const signal = AbortSignal.timeout(timeout)

    const schema = cfg.tool.schema ?? fallback
    const toolName = cfg.tool.name
    const result = await generateText({
      model: language,
      messages: msgs,
      system,
      tools: {
        [toolName]: tool({
          description: cfg.tool.description,
          inputSchema: jsonSchema(schema as any),
        }),
      },
      toolChoice: {
        type: "tool",
        toolName,
      } as any,
      providerOptions: ProviderTransform.providerOptions(model, {
        ...options,
        ...(disable ? { disable_parallel_tool_use: true } : {}),
      }),
      maxOutputTokens: 500,
      maxRetries: 0,
      abortSignal: signal,
      headers: headers(model),
    })

    const call = result.toolCalls.find((x) => x.toolName === toolName)
    if (!call) return

    const args = parse(call.input)
    log.info("search agent queries", {
      sessionID: input.sessionID,
      queries: args.queries,
      legacyInput: args.legacy,
    })
    if (args.queries.length === 0) return

    const found = await search(args, input.sessionID, countUserMessages(input.messages), timeout)
    if (!found) return
    const output = found.text

    if (!output.trim()) return
    log.info("search agent mcp results", {
      sessionID: input.sessionID,
      outputSize: output.length,
    })

    return {
      tool: toolName,
      input: call.input as Record<string, unknown>,
      output,
      metadata: {
        model: cfg.model,
        timeout,
        inputFormat: args.legacy ? "legacy" : "queries",
        queries: args.queries,
      },
    }
  }

  async function boot(sessionID: string, cfg: Config.SearchAgent) {
    const hit = cache.get(sessionID)
    if (hit) {
      log.info("search agent boot cache hit", {
        sessionID,
        size: hit.length,
      })
      return hit
    }

    if (!cfg.model.toLowerCase().includes("haiku")) {
      log.warn("search agent model is not haiku", {
        sessionID,
        model: cfg.model,
      })
    }

    const tools = await MCP.tools()
    const rows = await Promise.all(
      cfg.bootTools.map(async (name) => {
        const pick = pickTool(tools, name)
        if (!pick?.execute) {
          return {
            text: `- ${name}: unavailable`,
            called: false,
          }
        }
        const result = await pick
          .execute({}, { toolCallId: `boot-${name}`, messages: [], abortSignal: AbortSignal.timeout(cfg.timeout) } as any)
          .catch((error: unknown) => ({ error }))
        return {
          text: [`<boot tool="${name}">`, toText(result), `</boot>`].join("\n"),
          called: true,
        }
      }),
    )
    const lines = rows.map((x) => x.text)
    const called = rows.filter((x) => x.called).length

    const system = [cfg.systemPromptPrefix, lines.join("\n\n")].filter((x) => x.trim()).join("\n\n")
    cache.set(sessionID, system)
    log.info("search agent boot computed", {
      sessionID,
      bootTools: called,
      size: system.length,
    })
    return system
  }

  function headers(model: Provider.Model) {
    const result: Record<string, string> = {
      ...model.headers,
      "User-Agent": `opencode/${Installation.VERSION}`,
    }
    const beta = result["anthropic-beta"]
    if (!beta) return result
    const value = beta
      .split(",")
      .map((x: string) => x.trim())
      .filter((x: string) => x && !x.includes("interleaved-thinking"))
      .join(",")
    if (value) {
      result["anthropic-beta"] = value
      return result
    }
    delete result["anthropic-beta"]
    return result
  }

  function filter(messages: MessageV2.WithParts[], names: Set<string>) {
    return messages
      .map((msg): ModelMessage | undefined => {
        if (msg.info.role === "user") {
          const text = msg.parts
            .filter((x): x is MessageV2.TextPart => x.type === "text" && !x.synthetic && !x.ignored)
            .map((x) => x.text.trim())
            .filter((x) => x)
            .join("\n\n")
          const hasFiles = msg.parts.some((x) => x.type === "file" && !x.ignored)
          if (!text && !hasFiles) return
          return {
            role: "user",
            content: text || "[User sent an image/file without text]",
          }
        }

        const text = msg.parts
          .filter((x): x is MessageV2.TextPart => x.type === "text")
          .map((x) => x.text.trim())
          .filter((x) => x)

        const tools = msg.parts
          .filter(
            (x): x is MessageV2.ToolPart & { state: MessageV2.ToolStateCompleted } =>
              x.type === "tool" && x.state.status === "completed" && hasTool(names, x.tool),
          )
          .map((x) => `<tool_result name="${x.tool}">\n${x.state.output}\n</tool_result>`)

        const content = [...text, ...tools].join("\n\n").trim()
        if (!content) return
        return {
          role: "assistant",
          content,
        }
      })
      .filter((x): x is ModelMessage => Boolean(x))
  }

  function hasTool(names: Set<string>, tool: string) {
    if (names.has(tool)) return true
    for (const name of names) {
      if (tool.endsWith(`_${name}`)) return true
      if (name.endsWith(`_${tool}`)) return true
    }
    return false
  }

  function parse(input: unknown) {
    const body = typeof input === "object" && input ? (input as Record<string, unknown>) : {}

    const queriesRaw = body["queries"]
    if (Array.isArray(queriesRaw)) {
      const queries = queriesRaw
        .filter((q): q is Record<string, unknown> => typeof q === "object" && q !== null)
        .map((q) => ({
          query: typeof q.query === "string" ? q.query.trim() : "",
          mode: typeof q.mode === "string" ? q.mode : "search",
          depth: typeof q.depth === "string" ? q.depth : undefined,
          results: typeof q.results === "number" ? q.results : undefined,
        }))
        .filter((q) => q.query.length > 0)

      if (queries.length > 0) {
        return { queries, legacy: false }
      }
    }

    const all = read(body["queries"])
    let search = read(body["searchQueries"])
    let recall = read(body["recallQueries"])
    if (search.length === 0 && recall.length === 0 && all.length > 0) {
      const mode = body["searchMode"]
      if (mode === "recall") recall = all
      else if (mode === "search" || mode === "flash") search = all
      else {
        search = all
        recall = all
      }
    }
    const results = typeof body["results"] === "number" ? body["results"] : undefined
    const depth =
      body["depth"] === "snippet" || body["depth"] === "summary" || body["depth"] === "full"
        ? body["depth"]
        : undefined

    const queries = [
      ...search.map((q) => ({ query: q, mode: "search", depth, results })),
      ...recall.map((q) => ({ query: q, mode: "recall", depth, results })),
    ]

    return {
      queries,
      legacy: true,
    }
  }

  function read(input: unknown) {
    if (typeof input === "string") {
      const text = input.trim()
      if (!text) return []
      return [text]
    }
    if (!Array.isArray(input)) return []
    return input
      .filter((x): x is string => typeof x === "string")
      .map((x) => x.trim())
      .filter((x) => x)
  }

  function countUserMessages(messages: MessageV2.WithParts[]) {
    return messages.filter((x) => x.info.role === "user").length
  }

  async function search(
    input: { queries: Query[] },
    sessionID: string,
    turnNumber: number,
    timeout: number,
  ): Promise<{ text: string } | undefined> {
    const tools = await MCP.tools()
    const omniTool = pickTool(tools, "omni_tool")
    if (!omniTool?.execute) {
      log.warn("omni_tool not found, falling back to legacy search")
      return legacySearch(input, timeout)
    }

    try {
      const result = await omniTool.execute(
        {
          queries: input.queries,
          sessionId: sessionID,
          turnNumber,
        },
        { toolCallId: "search-agent-omni", messages: [], abortSignal: AbortSignal.timeout(timeout) } as any,
      )
      const text = toText(result)
      if (!text.trim()) return
      return { text }
    } catch (error) {
      log.warn("omni_tool failed, falling back to legacy search", { error })
      return legacySearch(input, timeout)
    }
  }

  async function legacySearch(input: { queries: Query[] }, timeout: number): Promise<{ text: string } | undefined> {
    const tools = await MCP.tools()
    const searchTool = pickTool(tools, "search_memory")
    const recallTool = pickTool(tools, "recall_context")
    const search = input.queries.filter((x) => x.mode !== "recall").map((x) => x.query)
    const recall = input.queries
      .filter((x) => x.mode === "recall" || x.mode === "auto")
      .map((x) => x.query)
    const results = input.queries.find((x) => x.results !== undefined)?.results
    const depth = input.queries.find((x) => x.depth !== undefined)?.depth
    const base = {
      ...(results !== undefined ? { results } : {}),
      ...(depth !== undefined ? { depth } : {}),
    }

    const run = async (
      id: string,
      tool: ReturnType<typeof pickTool>,
      queries: string[],
      callID: string,
    ): Promise<{ name: string; text: string; count: number } | undefined> => {
      if (queries.length === 0 || !tool?.execute) return
      return tool
        .execute(
          {
            ...base,
            queries,
          },
          { toolCallId: callID, messages: [], abortSignal: AbortSignal.timeout(timeout) } as any,
        )
        .then((result: unknown) => ({
          name: tool.id,
          text: toText(result),
          count: toCount(result),
        }))
        .catch((error: unknown) => {
          log.warn("search agent mcp search failed", {
            tool: id,
            error,
          })
          throw error
        })
    }

    const [searchResult, recallResult] = await Promise.all([
      run("search_memory", searchTool, search, "search-agent-search"),
      run("recall_context", recallTool, recall, "search-agent-recall"),
    ]).catch(() => {
      return [undefined, undefined] as const
    })

    if (searchResult === undefined && recallResult === undefined) return

    const text = [
      searchResult ? `# search_memory\n${searchResult.text}` : undefined,
      recallResult ? `# recall_context\n${recallResult.text}` : undefined,
    ]
      .filter((x) => x)
      .join("\n\n")
    if (!text.trim()) return
    return { text }
  }

  function toCount(input: unknown): number {
    if (!input) return 0
    if (Array.isArray(input)) return input.length
    if (typeof input === "string") {
      const text = input.trim()
      if (!text) return 0
      return 0
    }
    if (typeof input !== "object") return 0

    const obj = input as Record<string, unknown>
    for (const key of ["results", "items", "memories", "matches", "entries"]) {
      const val = obj[key]
      if (Array.isArray(val)) return val.length
    }

    if (!Array.isArray(obj.content)) return 0
    for (const item of obj.content) {
      if (!item || typeof item !== "object") continue
      const row = item as Record<string, unknown>
      if (row.type === "text" && typeof row.text === "string") {
        const count = toCount(row.text)
        if (count > 0) return count
      }
    }
    return obj.content.length
  }

  function pickTool(tools: Record<string, Tool>, name: string) {
    const safe = name.replace(/[^a-zA-Z0-9_-]/g, "_")
    for (const [key, item] of Object.entries(tools)) {
      if (key === name || key === safe) return { id: key, ...item }
      if (key.endsWith(`_${name}`) || key.endsWith(`_${safe}`)) return { id: key, ...item }
    }
  }

  function toText(input: unknown): string {
    if (typeof input === "string") return input
    if (!input || typeof input !== "object") return String(input)
    const result = input as { content?: Array<Record<string, unknown>> }
    if (!Array.isArray(result.content)) return JSON.stringify(input) ?? String(input)
    const lines = result.content
      .flatMap((item) => {
        if (item.type === "text" && typeof item.text === "string") return [item.text]
        if (item.type === "resource" && typeof item.resource === "object" && item.resource) {
          const resource = item.resource as Record<string, unknown>
          if (typeof resource.text === "string") return [resource.text]
        }
        return []
      })
      .join("\n\n")
    if (lines) return lines
    return JSON.stringify(input) ?? String(input)
  }
}
