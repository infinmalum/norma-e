package coordinator

import (
	"context"
	"iter"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/infinmalum/norma-e/harness"
	"github.com/infinmalum/norma-e/llm"
	"github.com/infinmalum/norma-e/permission"
	"github.com/infinmalum/norma-e/tool"
)

// echoProvider answers each worker with text derived from the task's last user
// message, so we can verify distinct workers ran.
type echoProvider struct{ count int32 }

func (p *echoProvider) Stream(_ context.Context, req llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	atomic.AddInt32(&p.count, 1)
	last := ""
	if n := len(req.Messages); n > 0 {
		last = req.Messages[n-1].Text()
	}
	return func(yield func(llm.StreamEvent, error) bool) {
		yield(llm.StreamEvent{Type: llm.SETextDelta, Text: "did: " + last}, nil)
		yield(llm.StreamEvent{Type: llm.SEMessageDelta, StopReason: "end_turn"}, nil)
		yield(llm.StreamEvent{Type: llm.SEMessageStop}, nil)
	}
}

func (p *echoProvider) Complete(ctx context.Context, req llm.CompletionRequest) (llm.Message, string, llm.Usage, error) {
	return accumulateStream(ctx, p, req)
}

// accumulateStream drains a provider's own Stream to satisfy the non-streaming
// Complete method for test mocks.
func accumulateStream(ctx context.Context, p llm.Provider, req llm.CompletionRequest) (llm.Message, string, llm.Usage, error) {
	acc := llm.NewAccumulator()
	for ev, err := range p.Stream(ctx, req) {
		if err != nil {
			return llm.Message{}, "", llm.Usage{}, err
		}
		acc.Add(ev)
	}
	return acc.Message(), acc.StopReason, acc.Usage, nil
}

func TestRunParallelFansOut(t *testing.T) {
	prov := &echoProvider{}
	c := &Config{
		Provider:       prov,
		WorkerTools:    tool.NewRegistry(tool.DefaultTools()...),
		PermissionMode: permission.ModeBypass,
	}
	tasks := []string{"task A", "task B", "task C"}
	results := c.RunParallel(context.Background(), tasks)
	if len(results) != 3 {
		t.Fatalf("results=%d", len(results))
	}
	for i, r := range results {
		if r.Err != nil {
			t.Fatalf("task %d err: %v", i, r.Err)
		}
		if !strings.Contains(r.Output, tasks[i]) {
			t.Fatalf("result %d mismatched: task=%q output=%q", i, tasks[i], r.Output)
		}
	}
	if atomic.LoadInt32(&prov.count) != 3 {
		t.Fatalf("expected 3 worker runs, got %d", prov.count)
	}
}

func TestAgentToolAndSystemSegment(t *testing.T) {
	c := &Config{
		Provider:    &echoProvider{},
		WorkerTools: tool.NewRegistry(tool.NewRead(), tool.NewBash()),
	}
	at := c.AgentTool()
	if at.Name() != "Agent" {
		t.Fatalf("agent tool name=%q", at.Name())
	}
	seg := c.SystemSegment()
	if !strings.Contains(seg, "Coordinator mode") || !strings.Contains(seg, "Read") || !strings.Contains(seg, "Bash") {
		t.Fatalf("system segment missing pieces:\n%s", seg)
	}
	// The Agent tool actually dispatches a worker.
	res, _ := at.Call(context.Background(), []byte(`{"agent_type":"worker","prompt":"find bugs"}`), &tool.ToolContext{})
	if !strings.Contains(res.Flatten(), "find bugs") {
		t.Fatalf("worker dispatch: %q", res.Flatten())
	}
}

type parallelProvider struct {
	echoProvider
	started atomic.Int32
	ready   chan struct{}
}

func (p *parallelProvider) Stream(ctx context.Context, req llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
	return func(yield func(llm.StreamEvent, error) bool) {
		if p.started.Add(1) == 2 {
			close(p.ready)
		}
		select {
		case <-p.ready:
		case <-ctx.Done():
			yield(llm.StreamEvent{}, ctx.Err())
			return
		}
		p.echoProvider.Stream(ctx, req)(yield)
	}
}

func TestReadOnlyModelDelegationsOverlap(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Second)
	defer cancel()
	worker := &parallelProvider{ready: make(chan struct{})}
	cfg := &Config{Provider: worker, PermissionMode: permission.ModePlan, ConcurrentDelegation: true}
	turn := 0
	deps := harness.QueryDeps{CallModel: func(context.Context, llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
		turn++
		return func(yield func(llm.StreamEvent, error) bool) {
			if turn > 1 {
				yield(llm.StreamEvent{Type: llm.SETextDelta, Text: "done"}, nil)
				return
			}
			for _, task := range []string{"a", "b"} {
				if !yield(llm.StreamEvent{Type: llm.SEToolUseStart, ToolID: task, ToolName: "Agent"}, nil) ||
					!yield(llm.StreamEvent{Type: llm.SEToolInputJSON, Text: `{"agent_type":"worker","prompt":"` + task + `"}`}, nil) {
					return
				}
			}
			yield(llm.StreamEvent{Type: llm.SEMessageStop}, nil)
		}
	}}
	var results int
	for event, err := range harness.Query(ctx, harness.QueryInput{
		Tools: tool.NewRegistry(cfg.AgentTool()), PermissionMode: permission.ModePlan, MaxConcurrency: 2,
	}, deps) {
		if err != nil {
			t.Fatal(err)
		}
		if event.Kind == harness.KindToolResult {
			if event.ToolResult.IsError {
				t.Fatalf("worker failed: %+v", event.ToolResult)
			}
			results++
		}
	}
	if results != 2 {
		t.Fatalf("got %d worker results, want 2", results)
	}
}
