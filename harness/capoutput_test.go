package harness

import (
	"strings"
	"testing"

	"github.com/infinmalum/norma-e/llm"
	"github.com/infinmalum/norma-e/tool"
)

// The global post-tool net caps an oversized text block regardless of the tool.
func TestCapOutputCapsOversizedTextBlock(t *testing.T) {
	tc := &tool.ToolContext{MaxOutputChars: 100} // no OutputDir → head+tail truncation
	long := strings.Repeat("z", 5000)
	res := tool.Result{Content: []llm.ContentBlock{llm.TextBlock(long)}}

	out := capOutput(tc, res)
	got := out.Content[0].Text
	if len(got) >= 5000 || !strings.Contains(got, "characters truncated]") {
		t.Fatalf("capOutput did not cap oversized block: len=%d", len(got))
	}
}

// Output already within budget is passed through untouched.
func TestCapOutputLeavesSmallOutputIntact(t *testing.T) {
	tc := &tool.ToolContext{MaxOutputChars: 100}
	res := tool.Result{Content: []llm.ContentBlock{llm.TextBlock("fine")}}

	if out := capOutput(tc, res); out.Content[0].Text != "fine" {
		t.Fatalf("small output changed: %q", out.Content[0].Text)
	}
}

func TestCapOutputIgnoresMarkersAndSharesBudget(t *testing.T) {
	tc := &tool.ToolContext{MaxOutputChars: 100}
	res := tool.Result{Content: []llm.ContentBlock{
		llm.TextBlock("<persisted-output>" + strings.Repeat("a", 65)),
		llm.TextBlock(strings.Repeat("b", 90)),
	}}
	out := capOutput(tc, res)
	if len(out.Flatten()) >= len(res.Flatten()) || !strings.Contains(out.Flatten(), "characters truncated]") {
		t.Fatalf("total output escaped cap: %q", out.Flatten())
	}
	if second := capOutput(tc, out); second.Flatten() != out.Flatten() {
		t.Fatal("capture metadata was not preserved")
	}
}
