# norma-e

norma-e is a Go SDK for building tool-using agents. It provides a streaming execution loop, host-controlled tool permissions, and conversation state behind `agentcore.Session`. It supports the Anthropic Messages, OpenAI Chat Completions, and OpenAI Responses APIs.

norma-e is a fork of [Norma](https://github.com/Autumn-27/Norma), originally created by [Autumn-27](https://github.com/Autumn-27). Original copyright © 2026 Autumn-27. See [LICENSE](LICENSE) for the Apache 2.0 license and copyright notice.

Build with Go 1.26.3 or later. Import SDK packages under `github.com/infinmalum/norma-e`, for example `github.com/infinmalum/norma-e/agentcore`.

Starting with v0.7.0, use this module path when upgrading from earlier releases. Update your Go imports and replace the old module requirement with `go get github.com/infinmalum/norma-e@v0.7.0`, then run `go mod tidy`.

## Packages

| Package | Responsibility |
|---|---|
| `llm` | Vendor-neutral message model (`Message`/`ContentBlock`/`StreamEvent`) + `Provider` interface. `NewProvider(Config{Format})` selects Anthropic Messages, OpenAI Chat Completions, or OpenAI Responses. Each adapter normalizes streaming and tool calls into the same event type. Image and document blocks (`ImageBlock`/`DocumentBlock`) are part of the message model; see [Multimodal input](#multimodal-input). |
| `tool` | `CoreTool` contract, fail-closed `Build` factory, `Registry`, a lightweight JSON-Schema validator, and the core tools: **Read, Write, Edit, MultiEdit, LS, Glob, Grep, Bash** (default set) plus opt-in **WebFetch** / **WebSearch** (network, permission-gated), **TodoWrite** (stateful planning), **AskUserQuestion** (structured mid-task questions), background-task tools (**TaskOutput/TaskStop/TaskList/Monitor**), and **deferred tools** — schemas withheld from the model to save context, discovered on demand via `SearchExtraTools` and invoked through `ExecuteExtraTool`. Read handles non-text inputs by content: images (PNG/JPEG/GIF/WEBP ≤5MB) and PDFs return native image/document blocks; docx/xlsx/pptx return extracted text (stdlib-only); UTF-16 text is transcoded; other binaries are reported, not shown. |
| `permission` | Evaluates allow and deny rules, tool policy, permission mode, and host approval. Deny rules take priority, including for deferred targets and nested agent calls. |
| `harness` | The loop: `Query` streams `Event`s (`iter.Seq2`), schedules tools that declare concurrency safety in parallel and other tools exclusively, pairs every `tool_use` with a `tool_result` (synthetic on deny/abort), enforces termination reasons, and threads cancellation. `QueryDeps` injects side effects for testing. |
| `compaction` | Context-window management: Snip / MicroCompact / Context Collapse / AutoCompact, a circuit breaker, reactive recovery from prompt-too-long, and non-destructive `compact_boundary` summaries. Invoked skills are re-injected verbatim after each boundary so their instructions survive summarization. |
| `noa` / `noaadapter` | Optional model-driven compression. The adapter supplies a projected request view, the Compress tool, and durable archives. |
| `hook` | Eight lifecycle hooks (PreToolUse, PostToolUse, Stop, …) as Go callbacks that can block, rewrite input, inject context, or gate the Stop transition. |
| `subagent` | An `Agent` tool that delegates to isolated child agents reusing the same loop, with a recursion-depth guard and the enclosing tool permission policy. |
| `mcp` | Model Context Protocol integration: a stdio JSON-RPC client and in-process servers, exposing tools as `mcp__<server>__<tool>`. |
| `plan` | Plan mode: read-only exploration → `ExitPlanMode` presents a plan for approval → the permission mode flips mid-loop so the agent can implement. |
| `memory` | Persistent long-term memory: markdown-with-frontmatter files + a `MEMORY.md` index, relevance selection, staleness warnings, and `RecallMemory`/`RecordMemory` tools. |
| `skill` | Named, reusable instruction packages loaded from `SKILL.md` files (YAML frontmatter — `name`/`description`/`license`/`compatibility`/`mcps` — plus a Markdown body). A `Skill` tool injects a chosen skill's procedure on demand as standing guidance that survives compaction, and can unlock a skill's declared MCP tools on invocation. |
| `coordinator` | Coordinator mode: a top-level agent that decomposes work and delegates to isolated `worker` subagents — LLM-driven (Agent tool + system segment) or programmatic (`RunParallel` fan-out). |
| `agentcore` | The public entry point: a stateful `Session` whose `Prompt` streams the loop, plus a one-shot `Run`. Aggregates everything above, and optionally persists an append-only JSONL transcript so a session can be resumed (`Session.Resume`). |

## Quick start

```go
prov, _ := llm.NewProvider(llm.Config{Format: llm.FormatAnthropic, Model: "your-model-name"})
// or:  llm.Config{Format: llm.FormatOpenAI, BaseURL: "https://api.deepseek.com/v1", Model: "deepseek-chat"}

sess := agentcore.NewSession(agentcore.Options{
    Provider:       prov,
    SystemPrompt:   []string{"You are a helpful coding agent."}, // the SDK injects NO preset prompt
    PermissionMode: permission.ModeDefault,
    CanUseTool:     myApprovalCallback, // gates mutating tools
})

defer sess.Close()

for ev, err := range sess.Prompt(ctx, "Add a /health endpoint and run the tests.") {
    if err != nil { return err }
    switch ev.Kind {
    case harness.KindText:
        fmt.Print(ev.Text)
    case harness.KindResult:
        return ev.Terminal.CompletionError()
    }
}
```

## CLI demo

```bash
go build -o norma-e ./cmd/agentcore

ANTHROPIC_API_KEY=... ./norma-e -provider anthropic -model your-model-name
OPENAI_API_KEY=...    ./norma-e -provider openai    -model gpt-4o
OPENAI_API_KEY=...    ./norma-e -provider openai-responses -model your-model-name
./norma-e -provider openai -model gpt-4o -yes -p "summarize the go files here"
```

Mutating tools (Write/Edit/Bash) prompt for approval unless `-yes` is set. Supply your own system prompt with `-system FILE`.

## Model API formats

| CLI `-provider` | SDK `llm.Config.Format` | Path appended to the base URL | Environment defaults |
|---|---|---|---|
| `anthropic` | `llm.FormatAnthropic` | `/v1/messages` | `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` (default `https://api.anthropic.com`) |
| `openai` | `llm.FormatOpenAI` | `/chat/completions` | `OPENAI_API_KEY`, `OPENAI_BASE_URL` (default `https://api.openai.com/v1`) |
| `openai-responses` | `llm.FormatOpenAIResponses` | `/responses` | `OPENAI_API_KEY`, `OPENAI_BASE_URL` (default `https://api.openai.com/v1`) |

Set `-base-url` in the CLI or `llm.Config.BaseURL` in the SDK for a compatible gateway. Include any API prefix the gateway requires, such as `/v1` for the default OpenAI base URL, and leave off the path in the table. Supply the model identifier accepted by that endpoint. All three SDK adapters support streaming and non-streaming requests. The CLI uses streaming. Implement `llm.Provider` to connect a different protocol.

## Multimodal input

The **Read** tool returns native image blocks for PNG, JPEG, GIF, and WebP and native document blocks for PDF, up to 5 MiB per file. It extracts paragraphs, cell values, and slide text from DOCX, XLSX, and PPTX on every provider. Office text extraction omits page layout, charts, and embedded images.

Hosts can also build media attachments in a user `llm.Message` and pass it through `llm.CompletionRequest.Messages` or `harness.QueryInput.Messages`. Use `llm.ImageBlock(mediaType, base64)` for inline images or `llm.DocumentBlock(mediaType, base64)` for inline documents. For a remote image or document, construct a media block with `Source: &llm.BlockSource{Type: "url", URL: url}`. To reference an existing file, use `Source: &llm.BlockSource{Type: "file", FileID: id}`. The URL must be accessible to the configured provider. File IDs must belong to that provider and be accessible to the credentials used for the request.

| Wire format | Image sources | PDF sources | Office originals passed by the host | Tool-result media |
|---|---|---|---|---|
| Anthropic Messages | Base64, URL, file ID | Base64, URL, file ID | Unsupported | Native blocks inside `tool_result` |
| OpenAI Chat Completions | Base64, URL | Base64, file ID | Unsupported | Media follows the tool replies in a user message |
| OpenAI Responses | Base64, URL, file ID | Base64, URL, file ID | DOCX, XLSX, PPTX via Base64, URL, or file ID | Native parts inside `function_call_output.output` |

Use `llm.MediaTypeDOCX`, `llm.MediaTypeXLSX`, or `llm.MediaTypePPTX` with `DocumentBlock` for inline Office files. Set `Source.Filename` to preserve a filename on OpenAI requests. The SDK supplies a default filename for inline PDF and these Office formats. Set `Source.Detail` for OpenAI image inputs or Responses PDF inputs. The accepted detail levels depend on the API and model. For PDF URL or file ID references, optionally set `Source.MediaType` to `llm.MediaTypePDF`. OpenAI-compatible gateways must support the listed wire shapes to accept these attachments.

`Session.Prompt`, the CLI, and subagent prompts accept text. Agents using those entry points can read local attachments through `Read`. PostToolUse hooks receive flattened text. MicroCompact clears stale media-bearing tool results, and the compaction summarizer sees media placeholders. MCP image results retain their image blocks. Other non-text MCP results become notices.

## Tool permissions

`AllowedTools` and `DisallowedTools` accept exact tool names or `*`. Argument selectors such as `Bash(git:*)` are unsupported and deny calls to the named tool. For argument-specific approval, evaluate the arguments in the host's `CanUseTool` callback and leave the tool subject to approval. The permission policy gates tool calls. Process and filesystem isolation are the host's responsibility.

Deferred tools omit their schemas from the model's tool list. The model can request their schemas through `SearchExtraTools` and invoke unlocked targets through `ExecuteExtraTool`. Both the wrapper and the target are checked. Child agents inherit the enclosing query's allow and deny rules and default permission mode. An explicitly configured child mode can further restrict calls.

## Run completion and plan approval

`Session.Prompt` streams text and reasoning as they arrive. Tool calls start after the model response finishes successfully. The harness assembles argument fragments by tool call and uses the same scheduling rules for streaming and non-streaming responses.

Read `KindResult` to determine the outcome. `Terminal.CompletionError()` returns the provider error when present or a typed `*harness.IncompleteError` with the terminal reason when the run stops early. `agentcore.Run` and `subagent.Run` return this error with any available final text.

Supply `PlanOptions.Approver` to authorize a transition out of read-only plan mode. With no approver, the session stays in plan mode. Set `PlanEachPrompt` to require a new approval for each user prompt. Supply `CanUseTool` separately for commands that still require individual approval.

A subagent explicitly configured with `permission.ModePlan` can be delegated to from a read-only parent. Every child tool call still passes the enclosing permission policy. Set `coordinator.Config.ConcurrentDelegation` to opt read-only worker calls into concurrent scheduling when the worker provider, tools, and hooks support concurrent use. The parent query's concurrency limit applies to model-driven delegation. `RunParallel` uses `coordinator.Config.MaxParallel`. Workers share the configured working directory.

## Context and session state

Use `agentcore.Options.Compaction` for the built-in automatic summarizer, or call `noaadapter.Enable` before creating the session to install model-driven compression. NOA requires a durable `ArchiveBaseDir`. Set `ModelContextLimit` to match the deployment. The adapter supplies its own request view and takes precedence over built-in compaction when both are configured.

`Session.Usage()` accumulates token counts across completed prompts, including counts restored by `Session.Resume`. The usage reported in a prompt's events describes that prompt. When a transcript write fails, the session continues in memory and exposes the error through `Session.TranscriptError()` and the optional `OnWarn` callback. Resume reads complete JSONL records. On the first append after recovery from an incomplete last record, the store saves the original log as a sibling `.recovery-*` file before trimming that record. A malformed complete record causes resume to return an error.

## Examples

- `examples/custom-llm` — three provider configs + a custom non-coding prompt + a custom tool.
- `examples/coding-agent` — full coding agent with interactive permissions, a PreToolUse hook, and compaction.
- `examples/security-audit` — read-only auditor that delegates to a subagent.
- `examples/planner-memory` — per-prompt plan approval, separate tool approval, and persistent memory.
- `examples/coordinator` — parallel read-only workers with host-driven and model-driven delegation.

## Tests

```bash
go test ./...
```

Tests cover provider adapters with local HTTP fixtures, tool execution and permission checks, transcript recovery, and simulated long sessions with NOA. The GitHub Actions workflow runs the suite on pushes and pull requests.
