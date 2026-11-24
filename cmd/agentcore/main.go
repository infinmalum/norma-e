// Command agentcore is a runnable REPL demonstrating the SDK. It wires a
// provider (Anthropic or OpenAI format), a host-supplied system prompt, the
// core toolset, and an interactive permission prompt into a streaming loop.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"

	"github.com/infinmalum/norma-e/agentcore"
	"github.com/infinmalum/norma-e/harness"
	"github.com/infinmalum/norma-e/llm"
	"github.com/infinmalum/norma-e/permission"
	"github.com/infinmalum/norma-e/tool"
)

// The SDK injects no system prompt, so this host supplies one. Swap it (or use
// -system FILE) to retarget the agent to any scenario.
const defaultSystem = `You are a helpful coding agent operating in a terminal. Use the available tools (Read, Write, Edit, MultiEdit, LS, Glob, Grep, Bash) to inspect and modify the project. Prefer dedicated tools over Bash equivalents. Before editing a file, read it. Report what you did concisely. Confirm before destructive or irreversible actions.`

func main() { os.Exit(run()) }

func run() int {
	var (
		providerName = flag.String("provider", env("AGENT_PROVIDER", "anthropic"), "anthropic | openai | openai-responses")
		model        = flag.String("model", env("AGENT_MODEL", ""), "model id (e.g. claude-opus-4-8, gpt-4o)")
		baseURL      = flag.String("base-url", "", "override API base URL")
		apiKey       = flag.String("api-key", "", "API key (else ANTHROPIC_API_KEY / OPENAI_API_KEY)")
		systemFile   = flag.String("system", "", "path to a system prompt file (else a built-in coding prompt)")
		workDir      = flag.String("dir", "", "working directory for tools (default cwd)")
		yes          = flag.Bool("yes", false, "auto-approve all tool calls (bypass permission)")
		maxTurns     = flag.Int("max-turns", 50, "max model round-trips per message")
		oneShot      = flag.String("p", "", "run a single prompt and exit")
	)
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "error: -model is required")
		return 2
	}
	wd := *workDir
	if wd == "" {
		wd, _ = os.Getwd()
	}
	system := defaultSystem
	if *systemFile != "" {
		b, err := os.ReadFile(*systemFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error reading system prompt: %v\n", err)
			return 1
		}
		system = string(b)
	}

	prov, err := llm.NewProvider(llm.Config{
		Format:  llm.Format(strings.ToLower(*providerName)),
		BaseURL: *baseURL,
		APIKey:  *apiKey,
		Model:   *model,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		return 2
	}

	mode := permission.ModeDefault
	if *yes {
		mode = permission.ModeBypass
	}

	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	sess := agentcore.NewSession(agentcore.Options{
		Provider:       prov,
		SystemPrompt:   []string{system},
		PermissionMode: mode,
		CanUseTool: func(_ context.Context, name string, input json.RawMessage, _ permission.Context) permission.Decision {
			return askUser(in, name, input)
		},
		AskUser: func(_ context.Context, q tool.Question) tool.Answer {
			return askUserQuestion(in, q)
		},
		WorkingDir: wd,
		MaxTurns:   *maxTurns,
	})

	defer sess.Close()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if *oneShot != "" {
		if err := runTurn(ctx, sess, *oneShot); err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			return 1
		}
		return 0
	}

	fmt.Printf("agentcore — provider=%s model=%s dir=%s\nType your request. Ctrl-D or /exit to quit.\n", *providerName, *model, wd)
	for {
		fmt.Print("\n\x1b[1m›\x1b[0m ")
		if !in.Scan() {
			fmt.Println()
			if err := in.Err(); err != nil {
				fmt.Fprintln(os.Stderr, "error:", err)
				return 1
			}
			return 0
		}
		line := strings.TrimSpace(in.Text())
		switch {
		case line == "":
			continue
		case line == "/exit" || line == "/quit":
			return 0
		case line == "/reset":
			sess.Reset()
			fmt.Println("(history cleared)")
		default:
			if err := runTurn(ctx, sess, line); err != nil {
				fmt.Fprintln(os.Stderr, "error:", err)
			}
		}
	}
}

func runTurn(ctx context.Context, sess *agentcore.Session, input string) error {
	for ev, err := range sess.Prompt(ctx, input) {
		if err != nil {
			return err
		}
		switch ev.Kind {
		case harness.KindText:
			fmt.Print(ev.Text)
		case harness.KindToolUse:
			fmt.Printf("\n\x1b[2m· %s %s\x1b[0m\n", ev.ToolUse.Name, summarize(ev.ToolUse.Input))
		case harness.KindToolResult:
			status := "ok"
			if ev.ToolResult.IsError {
				status = "error"
			}
			fmt.Printf("\x1b[2m  ↳ %s\x1b[0m\n", status)
		case harness.KindResult:
			fmt.Println()
			return ev.Terminal.CompletionError()
		}
	}
	return nil
}

// askUser is the interactive permission callback for mutating tool calls.
func askUser(in *bufio.Scanner, name string, input json.RawMessage) permission.Decision {
	fmt.Printf("\n\x1b[33mAllow %s %s ? [y/N] \x1b[0m", name, input)
	if in.Scan() {
		if answer := strings.ToLower(strings.TrimSpace(in.Text())); answer == "y" || answer == "yes" {
			return permission.Allowed()
		}
	}
	return permission.Denied("denied by user")
}

// askUserQuestion presents a structured question on the terminal and returns
// the user's choice(s).
func askUserQuestion(in *bufio.Scanner, q tool.Question) tool.Answer {
	fmt.Printf("\n\x1b[36m%s\x1b[0m\n", q.Question)
	for i, o := range q.Options {
		desc := ""
		if o.Description != "" {
			desc = " — " + o.Description
		}
		fmt.Printf("  %d) %s%s\n", i+1, o.Label, desc)
	}
	prompt := "Choose 1 option"
	if q.MultiSelect {
		prompt = "Choose one or more (comma-separated)"
	}
	fmt.Printf("\x1b[33m%s: \x1b[0m", prompt)
	if !in.Scan() {
		return tool.Answer{}
	}
	var sel []string
	for _, part := range strings.Split(strings.TrimSpace(in.Text()), ",") {
		part = strings.TrimSpace(part)
		if n, err := strconv.Atoi(part); err == nil && n >= 1 && n <= len(q.Options) {
			sel = append(sel, q.Options[n-1].Label)
		} else if part != "" {
			sel = append(sel, part) // free-form fallback
		}
	}
	return tool.Answer{Selected: sel}
}

func summarize(input json.RawMessage) string {
	s := string(input)
	if len(s) > 160 {
		s = s[:160] + "…"
	}
	return s
}

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
