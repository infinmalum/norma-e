package main

import (
	"bufio"
	"context"
	"errors"
	"iter"
	"strings"
	"testing"

	"github.com/infinmalum/norma-e/agentcore"
	"github.com/infinmalum/norma-e/harness"
	"github.com/infinmalum/norma-e/llm"
	"github.com/infinmalum/norma-e/permission"
)

func TestRunTurnReturnsModelFailure(t *testing.T) {
	want := errors.New("model fixture failure")
	session := agentcore.NewSession(agentcore.Options{
		DisableBackgroundTasks: true,
		Deps: harness.QueryDeps{CallModel: func(context.Context, llm.CompletionRequest) iter.Seq2[llm.StreamEvent, error] {
			return func(yield func(llm.StreamEvent, error) bool) {
				yield(llm.StreamEvent{}, want)
			}
		}},
	})
	defer session.Close()
	if err := runTurn(session, "hello"); !errors.Is(err, want) {
		t.Fatalf("err=%v, want %v", err, want)
	}
}

func TestToolApprovalSharesPromptInput(t *testing.T) {
	in := bufio.NewScanner(strings.NewReader("first task\nyes\nsecond task\n"))
	if !in.Scan() || in.Text() != "first task" {
		t.Fatal("first prompt missing")
	}
	if decision := confirm(in, "Write", []byte(`{"file_path":"note.txt","content":"hello"}`)); decision.Behavior != permission.Allow {
		t.Fatalf("decision=%+v", decision)
	}
	if !in.Scan() || in.Text() != "second task" {
		t.Fatal("approval consumed the next prompt")
	}
}
