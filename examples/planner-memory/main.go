// Example planner-memory shows the Phase-4 features: plan mode (the agent
// explores read-only, presents a plan for approval, then implements) and
// persistent memory (relevant memories are auto-injected, and the agent can
// record durable facts). Both layer onto the same Session with no change to the
// core loop.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/infinmalum/norma-e/agentcore"
	"github.com/infinmalum/norma-e/harness"
	"github.com/infinmalum/norma-e/llm"
	"github.com/infinmalum/norma-e/memory"
	"github.com/infinmalum/norma-e/permission"
)

func main() {
	model := envOr("AGENT_MODEL", "claude-opus-4-8")
	prov, err := llm.NewProvider(llm.Config{Format: llm.FormatAnthropic, Model: model})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	wd, _ := os.Getwd()

	// Memory persisted under the working directory.
	store := memory.NewStore(filepath.Join(wd, ".agent-memory"))

	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	sess := agentcore.NewSession(agentcore.Options{
		Provider:     prov,
		SystemPrompt: []string{"You are a coding agent. Plan before you build; record durable facts (preferences, project constraints) to memory."},
		WorkingDir:   wd,
		MaxTurns:     50,
		CanUseTool: func(_ context.Context, name string, input json.RawMessage, _ permission.Context) permission.Decision {
			fmt.Printf("\nAllow %s %s? [y/N] ", name, input)
			if in.Scan() {
				if answer := strings.ToLower(strings.TrimSpace(in.Text())); answer == "y" || answer == "yes" {
					return permission.Allowed()
				}
			}
			return permission.Denied("denied by user")
		},
		// Plan mode: start read-only; ask the terminal to approve the plan.
		Plan: &agentcore.PlanOptions{
			PlanEachPrompt: true,
			TargetMode:     permission.ModeAcceptEdits,
			Approver: func(_ context.Context, plan string) (bool, string) {
				return approvePlan(in, plan)
			},
		},
		// Memory: inject relevant memories per prompt and expose the index.
		Memory: &agentcore.MemoryOptions{
			Store:                store,
			AutoInject:           true,
			IncludeIndexInPrompt: true,
		},
	})

	fmt.Printf("planner-memory (%s) — plan mode + memory. Ctrl-D to quit.\n", model)
	defer sess.Close()
	for {
		fmt.Print("\n› ")
		if !in.Scan() {
			return
		}
		line := strings.TrimSpace(in.Text())
		if line == "" {
			continue
		}
		for ev, err := range sess.Prompt(context.Background(), line) {
			if err != nil {
				fmt.Fprintf(os.Stderr, "[error: %v]\n", err)
				break
			}
			switch ev.Kind {
			case harness.KindText:
				fmt.Print(ev.Text)
			case harness.KindToolUse:
				fmt.Printf("\n· %s\n", ev.ToolUse.Name)
			case harness.KindResult:
				fmt.Println()
				if err := ev.Terminal.CompletionError(); err != nil {
					fmt.Fprintln(os.Stderr, "error:", err)
				}
			}
		}
	}
}

// approvePlan prints the plan and asks for terminal approval.
func approvePlan(in *bufio.Scanner, plan string) (bool, string) {
	fmt.Printf("\n--- Proposed plan ---\n%s\n---------------------\nApprove this plan? [y/N] ", plan)
	if !in.Scan() {
		return false, ""
	}
	if s := strings.ToLower(strings.TrimSpace(in.Text())); s == "y" || s == "yes" {
		return true, ""
	}
	fmt.Print("Feedback (optional): ")
	if in.Scan() {
		return false, strings.TrimSpace(in.Text())
	}
	return false, ""
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
