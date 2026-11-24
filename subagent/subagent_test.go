package subagent

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"strings"
	"testing"

	"github.com/infinmalum/norma-e/harness"
	"github.com/infinmalum/norma-e/llm"
	"github.com/infinmalum/norma-e/permission"
	"github.com/infinmalum/norma-e/tool"
)

// fakeProvider always answers with a fixed line of text and ends the turn.
type fakeProvider struct{ reply string }

func (f fakeProvider) Stream(_ context.Context, _ llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		yield(llm.StreamEvent{Type: llm.SETextDelta, Text: f.reply}, nil)
		yield(llm.StreamEvent{Type: llm.SEMessageDelta, StopReason: "end_turn"}, nil)
		yield(llm.StreamEvent{Type: llm.SEMessageStop}, nil)
	}
}

func (f fakeProvider) Complete(ctx context.Context, req llm.CompletionRequest) (llm.Message, string, llm.Usage, error) {
	acc := llm.NewAccumulator()
	for ev, err := range f.Stream(ctx, req) {
		if err != nil {
			return llm.Message{}, "", llm.Usage{}, err
		}
		acc.Add(ev)
	}
	return acc.Message(), acc.StopReason, acc.Usage, nil
}

func TestSubagentDispatch(t *testing.T) {
	agentTool := NewTool(Config{
		Provider:    fakeProvider{reply: "subagent answer"},
		ParentTools: tool.NewRegistry(tool.DefaultTools()...),
		Agents: map[string]Definition{
			"researcher": {Description: "researches", Prompt: []string{"You research."}, PermissionMode: permission.ModeBypass},
		},
	})
	res, err := agentTool.Call(context.Background(), []byte(`{"agent_type":"researcher","prompt":"find x"}`), &tool.ToolContext{})
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	if !strings.Contains(res.Flatten(), "subagent answer") {
		t.Fatalf("subagent result: %q", res.Flatten())
	}
}

func TestUnknownAgentType(t *testing.T) {
	agentTool := NewTool(Config{Provider: fakeProvider{}, Agents: map[string]Definition{}})
	res, _ := agentTool.Call(context.Background(), []byte(`{"agent_type":"nope","prompt":"x"}`), &tool.ToolContext{})
	if !res.IsError {
		t.Fatal("expected error for unknown agent_type")
	}
}

func TestRecursionDepthGuard(t *testing.T) {
	agentTool := NewTool(Config{
		Provider: fakeProvider{reply: "x"},
		MaxDepth: 2,
		Agents:   map[string]Definition{"a": {Prompt: []string{"p"}, PermissionMode: permission.ModeBypass}},
	})
	// Simulate being already at max depth.
	ctx := context.WithValue(context.Background(), depthKey, 2)
	res, _ := agentTool.Call(ctx, []byte(`{"agent_type":"a","prompt":"x"}`), &tool.ToolContext{})
	if !res.IsError || !strings.Contains(res.Flatten(), "recursion depth") {
		t.Fatalf("expected depth guard, got %q", res.Flatten())
	}
}

type loopingProvider struct{ fakeProvider }

func (loopingProvider) Stream(context.Context, llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		for _, event := range []llm.StreamEvent{
			{Type: llm.SEToolUseStart, ToolID: "a", ToolName: "Read"},
			{Type: llm.SEToolInputJSON, Text: `{"file_path":"sample.txt"}`},
			{Type: llm.SEMessageStop},
		} {
			if !yield(event, nil) {
				return
			}
		}
	}
}

func TestWorkerReportsTurnLimit(t *testing.T) {
	_, err := Run(t.Context(), Config{
		Provider: loopingProvider{},
		Agents:   map[string]Definition{"worker": {MaxTurns: 1}},
	}, "worker", "continue", t.TempDir())
	var incomplete *harness.IncompleteError
	if !errors.As(err, &incomplete) || incomplete.Reason != harness.ReasonMaxTurns {
		t.Fatalf("err=%v, want max_turns", err)
	}
}

func TestPlanModeDelegatesOnlyToReadOnlyAgents(t *testing.T) {
	at := NewTool(Config{
		Provider: fakeProvider{reply: "reviewed"},
		Agents: map[string]Definition{
			"reader": {PermissionMode: permission.ModePlan, Concurrent: true},
			"editor": {PermissionMode: permission.ModeAcceptEdits, Concurrent: true},
		},
	})
	input := func(agent string) json.RawMessage {
		return json.RawMessage(`{"agent_type":"` + agent + `","prompt":"review"}`)
	}
	if !at.IsReadOnly(input("reader")) || !at.IsConcurrencySafe(input("reader")) {
		t.Fatal("read-only worker should allow concurrent delegation")
	}
	if at.IsReadOnly(input("editor")) || at.IsConcurrencySafe(input("editor")) {
		t.Fatal("editable worker should require exclusive dispatch and a non-plan parent")
	}
	turn := 0
	deps := harness.QueryDeps{CallModel: func(context.Context, llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
		turn++
		return func(yield func(llm.StreamEvent, error) bool) {
			if turn == 1 {
				for _, event := range []llm.StreamEvent{
					{Type: llm.SEToolUseStart, ToolID: "r", ToolName: "Agent"},
					{Type: llm.SEToolInputJSON, Text: string(input("reader"))},
					{Type: llm.SEToolUseStart, ToolID: "w", ToolName: "Agent"},
					{Type: llm.SEToolInputJSON, Text: string(input("editor"))},
					{Type: llm.SEMessageStop},
				} {
					if !yield(event, nil) {
						return
					}
				}
				return
			}
			yield(llm.StreamEvent{Type: llm.SETextDelta, Text: "done"}, nil)
		}
	}}
	results := map[string]llm.ContentBlock{}
	for event, _ := range harness.Query(t.Context(), harness.QueryInput{
		Tools: tool.NewRegistry(at), PermissionMode: permission.ModePlan,
	}, deps) {
		if event.Kind == harness.KindToolResult {
			results[event.ToolResult.ToolUseID] = *event.ToolResult
		}
	}
	if len(results) != 2 || results["r"].IsError || !results["w"].IsError {
		t.Fatalf("results=%+v", results)
	}
}
