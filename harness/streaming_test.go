package harness

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"sync/atomic"
	"testing"

	"github.com/infinmalum/norma-e/llm"
	"github.com/infinmalum/norma-e/permission"
	"github.com/infinmalum/norma-e/tool"
)

// Verify interleaved argument deltas are routed to the right call and no tool
// starts until the model turn ends.
func TestStreamCollectsToolCallsBeforeDispatch(t *testing.T) {
	ran := make(chan string, 4)
	probe := tool.Build(tool.Spec{
		Name: "probe",
		Schema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"id": map[string]any{"type": "string"}},
		},
		ReadOnly:   func(json.RawMessage) bool { return true },
		Concurrent: func(json.RawMessage) bool { return true },
		Permissions: func(context.Context, json.RawMessage, permission.Context) permission.Decision {
			return permission.Allowed()
		},
		Run: func(_ context.Context, in json.RawMessage, _ *tool.ToolContext) (tool.Result, error) {
			var v struct {
				ID string `json:"id"`
			}
			json.Unmarshal(in, &v)
			ran <- v.ID
			return tool.Text("ok"), nil
		},
	})

	var turn int32
	callModel := func(_ context.Context, _ llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
		n := atomic.AddInt32(&turn, 1)
		return func(yield func(llm.StreamEvent, error) bool) {
			if n > 1 { // second turn: finish.
				yield(llm.StreamEvent{Type: llm.SETextDelta, Text: "done"}, nil)
				yield(llm.StreamEvent{Type: llm.SEMessageDelta, StopReason: "end_turn"}, nil)
				yield(llm.StreamEvent{Type: llm.SEMessageStop}, nil)
				return
			}
			yield(llm.StreamEvent{Type: llm.SEToolUseStart, ToolID: "a", ToolName: "probe", ToolIndex: 0}, nil)
			yield(llm.StreamEvent{Type: llm.SEToolInputJSON, Text: `{"id":`, ToolIndex: 0, HasToolIndex: true}, nil)
			yield(llm.StreamEvent{Type: llm.SEToolUseStart, ToolID: "b", ToolName: "probe", ToolIndex: 1}, nil)
			yield(llm.StreamEvent{Type: llm.SEToolInputJSON, Text: `{"id":"b"}`, ToolIndex: 1, HasToolIndex: true}, nil)
			yield(llm.StreamEvent{Type: llm.SEToolInputJSON, Text: `"a"}`, ToolIndex: 0, HasToolIndex: true}, nil)
			if len(ran) != 0 {
				t.Error("tool ran while the provider was still streaming")
			}
			yield(llm.StreamEvent{Type: llm.SEMessageDelta, StopReason: "tool_use"}, nil)
			yield(llm.StreamEvent{Type: llm.SEMessageStop}, nil)
		}
	}

	in := QueryInput{
		Messages:       []llm.Message{llm.UserText("go")},
		Tools:          tool.NewRegistry(probe),
		PermissionMode: permission.ModeBypass,
	}

	term := drain(t, in, QueryDeps{CallModel: callModel})
	if term.Reason != ReasonCompleted {
		t.Fatalf("terminal=%+v", term)
	}
	results := term.Messages[2].Content
	if len(results) != 2 || results[0].ToolUseID != "a" || results[1].ToolUseID != "b" {
		t.Fatalf("results=%+v", results)
	}
	close(ran)
	got := map[string]bool{}
	for id := range ran {
		got[id] = true
	}
	if len(got) != 2 || !got["a"] || !got["b"] {
		t.Fatalf("executed arguments=%v", got)
	}
}

func TestStreamErrorDoesNotDispatchTools(t *testing.T) {
	dropped := errors.New("stream dropped")
	var runs atomic.Int32
	probe := tool.Build(tool.Spec{Name: "probe", Run: func(context.Context, json.RawMessage, *tool.ToolContext) (tool.Result, error) {
		runs.Add(1)
		return tool.Text("ran"), nil
	}})
	call := func(context.Context, llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
		return func(yield func(llm.StreamEvent, error) bool) {
			for _, event := range toolTurn("a", "probe", "{}")[:2] {
				if !yield(event, nil) {
					return
				}
			}
			yield(llm.StreamEvent{}, dropped)
		}
	}
	in := QueryInput{Messages: []llm.Message{llm.UserText("go")}, Tools: tool.NewRegistry(probe), PermissionMode: permission.ModeBypass}
	for event, _ := range Query(t.Context(), in, QueryDeps{CallModel: call}) {
		if event.Kind == KindToolUse || event.Kind == KindToolResult {
			t.Fatalf("unexpected tool event: %+v", event)
		}
		if event.Kind == KindResult && (!errors.Is(event.Terminal.Err, dropped) || len(event.Terminal.Messages) != 1) {
			t.Fatalf("terminal=%+v", event.Terminal)
		}
	}
	if runs.Load() != 0 {
		t.Fatal("tool ran after a failed model turn")
	}
}
