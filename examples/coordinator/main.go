// Example coordinator shows multi-agent orchestration two ways: a programmatic
// fan-out (RunParallel spawns one worker per task concurrently) and an
// LLM-driven coordinator Session (the model decomposes the request and spawns
// workers via the Agent tool), with a skill registry available throughout.
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/infinmalum/norma-e/agentcore"
	"github.com/infinmalum/norma-e/coordinator"
	"github.com/infinmalum/norma-e/harness"
	"github.com/infinmalum/norma-e/llm"
	"github.com/infinmalum/norma-e/permission"
	"github.com/infinmalum/norma-e/skill"
	"github.com/infinmalum/norma-e/tool"
)

func main() {
	model := envOr("AGENT_MODEL", "gpt-4o")
	prov, err := llm.NewProvider(llm.Config{Format: llm.FormatOpenAI, Model: model})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	wd, _ := os.Getwd()

	coord := &coordinator.Config{
		Provider:             prov,
		WorkerTools:          tool.NewRegistry(tool.NewRead(), tool.NewGlob(), tool.NewGrep()),
		PermissionMode:       permission.ModePlan,
		ConcurrentDelegation: true,
		WorkingDir:           wd,
		MaxParallel:          4,
	}

	// 1) Programmatic fan-out: each task runs in its own worker, concurrently.
	fmt.Println("== programmatic fan-out ==")
	failed := false
	for _, r := range coord.RunParallel(context.Background(), []string{
		"Summarize the Go files in the llm package.",
		"List the exported types in the tool package.",
	}) {
		if r.Err != nil {
			fmt.Printf("- %q failed: %v\n", r.Task, r.Err)
			failed = true
			continue
		}
		fmt.Printf("- %q →\n%s\n\n", r.Task, r.Output)
	}

	// 2) LLM-driven coordinator Session with a skill available.
	skills := skill.NewRegistry(skill.Skill{
		Name:         "audit",
		WhenToUse:    "when asked to review code for security issues",
		Instructions: "Search for injection, hardcoded secrets, and unsafe deserialization; report file:line findings.",
	})
	sess := agentcore.NewSession(agentcore.Options{
		Provider:               prov,
		SystemPrompt:           []string{"You are a coordinator."},
		AppendSystemPrompt:     []string{coord.SystemSegment()},
		Tools:                  []tool.CoreTool{coord.AgentTool()}, // coordinator only delegates
		Skills:                 skills,
		WorkingDir:             wd,
		MaxTurns:               30,
		DisableBackgroundTasks: true,
		PermissionMode:         permission.ModePlan,
	})

	fmt.Println("== LLM-driven coordinator ==")
	for ev, err := range sess.Prompt(context.Background(), "Investigate and summarize the agent loop and its tests.") {
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			failed = true
			break
		}
		switch ev.Kind {
		case harness.KindResult:
			if err := ev.Terminal.CompletionError(); err != nil {
				fmt.Fprintln(os.Stderr, "error:", err)
				failed = true
			}
		case harness.KindText:
			fmt.Print(ev.Text)
		case harness.KindToolUse:
			fmt.Printf("\n· delegating: %s\n", ev.ToolUse.Name)
		}
	}
	fmt.Println()
	if failed {
		os.Exit(1)
	}
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
