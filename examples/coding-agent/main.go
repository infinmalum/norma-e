// Example coding-agent assembles a full coding agent: the core tools, TodoWrite
// for planning, a PreToolUse hook, automatic context compaction, and either an
// interactive REPL or a one-shot run (-p). It shows how a host wires the SDK
// into a CLI.
//
// Flags:
//
//	-provider anthropic|openai|openai-responses  (default openai)
//	-model    MODEL             (env AGENT_MODEL)
//	-dir      DIR               working directory (default cwd)
//	-yes                        auto-approve tool calls (for non-interactive runs)
//	-p        "PROMPT"          run a single prompt and exit
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/infinmalum/norma-e/agentcore"
	"github.com/infinmalum/norma-e/compaction"
	"github.com/infinmalum/norma-e/harness"
	"github.com/infinmalum/norma-e/hook"
	"github.com/infinmalum/norma-e/llm"
	"github.com/infinmalum/norma-e/permission"
	"github.com/infinmalum/norma-e/tool"
)

func main() { os.Exit(run()) }

func run() int {
	providerName := flag.String("provider", envOr("AGENT_PROVIDER", "openai"), "anthropic | openai | openai-responses")
	model := flag.String("model", envOr("AGENT_MODEL", ""), "model id")
	dir := flag.String("dir", "", "working directory (default cwd)")
	yes := flag.Bool("yes", false, "auto-approve all tool calls")
	oneShot := flag.String("p", "", "run a single prompt and exit")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "error: -model is required")
		return 2
	}
	prov, err := llm.NewProvider(llm.Config{Format: llm.Format(*providerName), Model: *model})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	wd := *dir
	if wd == "" {
		wd, _ = os.Getwd()
	}

	// A PreToolUse hook that logs shell commands (could also block them).
	hooks := hook.NewRegistry()
	hooks.On(hook.PreToolUse, func(_ context.Context, ev hook.Event) hook.Result {
		if ev.ToolName == "Bash" {
			fmt.Fprintf(os.Stderr, "\x1b[2m[hook] shell: %s\x1b[0m\n", string(ev.Input))
		}
		return hook.Result{}
	})

	// System prompt + a live environment section.
	env := fmt.Sprintf("# Environment\n- Working directory: %s\n- Platform: %s\n- Today's date: %s\n- You are powered by the model named %s.",
		wd, runtime.GOOS, time.Now().Format("2006-01-02"), *model)

	mode := permission.ModeDefault
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	canUse := func(_ context.Context, name string, input json.RawMessage, _ permission.Context) permission.Decision {
		return confirm(in, name, input)
	}
	if *yes {
		mode = permission.ModeBypass
		canUse = nil
	}

	todos := tool.NewTodoStore()
	sess := agentcore.NewSession(agentcore.Options{
		Provider:       prov,
		SystemPrompt:   []string{systemPrompt, env},
		PermissionMode: mode,
		CanUseTool:     canUse,
		Hooks:          hooks,
		Todos:          todos,
		Compaction:     &compaction.Config{ContextWindow: 200000},
		WorkingDir:     wd,
		MaxTurns:       80,
	})

	defer sess.Close()
	if *oneShot != "" {
		err := runTurn(sess, *oneShot)
		if items := todos.List(); len(items) > 0 {
			fmt.Fprintf(os.Stderr, "\n[todos: %d items, final plan]\n", len(items))
			for _, td := range items {
				fmt.Fprintf(os.Stderr, "  [%s] %s\n", td.Status, td.Content)
			}
		}
		printUsage(sess)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			return 1
		}
		return 0
	}

	fmt.Printf("coding-agent (%s). Type a request, Ctrl-D to quit.\n", *model)
	for {
		fmt.Print("\n› ")
		if !in.Scan() {
			if err := in.Err(); err != nil {
				fmt.Fprintln(os.Stderr, "error:", err)
				return 1
			}
			return 0
		}
		if line := strings.TrimSpace(in.Text()); line != "" {
			if err := runTurn(sess, line); err != nil {
				fmt.Fprintln(os.Stderr, "error:", err)
			}
			printUsage(sess)
		}
	}
}

// printUsage reports cumulative token usage tracked by the Session (populated
// from the provider's usage frames, accumulated across all turns).
func printUsage(sess *agentcore.Session) {
	u := sess.Usage()
	fmt.Fprintf(os.Stderr, "\n[token usage] input=%d  output=%d  cache_read=%d  total=%d\n",
		u.InputTokens, u.OutputTokens, u.CacheReadTokens, u.InputTokens+u.OutputTokens)
}

func runTurn(sess *agentcore.Session, input string) error {
	for ev, err := range sess.Prompt(context.Background(), input) {
		if err != nil {
			return err
		}
		switch ev.Kind {
		case harness.KindText:
			fmt.Print(ev.Text)
		case harness.KindToolUse:
			fmt.Printf("\n\x1b[2m· %s %s\x1b[0m\n", ev.ToolUse.Name, summarize(ev.ToolUse.Input))
		case harness.KindToolResult:
			if ev.ToolResult.IsError {
				fmt.Print("\x1b[31m  ↳ error\x1b[0m\n")
			}
		case harness.KindResult:
			fmt.Println()
			return ev.Terminal.CompletionError()
		}
	}
	return nil
}

func summarize(input json.RawMessage) string {
	s := string(input)
	if len(s) > 120 {
		s = s[:120] + "…"
	}
	return s
}

func confirm(in *bufio.Scanner, name string, input json.RawMessage) permission.Decision {
	fmt.Printf("\nAllow %s %s ? [y/N] ", name, input)
	if in.Scan() {
		if s := strings.ToLower(strings.TrimSpace(in.Text())); s == "y" || s == "yes" {
			return permission.Allowed()
		}
	}
	return permission.Denied("denied by user")
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
