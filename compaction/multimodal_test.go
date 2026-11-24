package compaction

import (
	"strings"
	"testing"

	"github.com/infinmalum/norma-e/llm"
)

// Image/document blocks carry the fixed media cost (2000 tokens), not their
// byte length — a multi-megabyte base64 payload must not distort the estimate.
func TestEstimateTokensMediaBlocks(t *testing.T) {
	big := strings.Repeat("x", 1<<20) // ~1MB of base64
	msgs := []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.ImageBlock(llm.MediaTypePNG, big)}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.DocumentBlock(llm.MediaTypePDF, big)}},
	}
	if got := EstimateTokens(msgs); got != 2*mediaBlockChars/4 {
		t.Fatalf("EstimateTokens = %d, want %d (fixed per media block)", got, 2*mediaBlockChars/4)
	}

	// Nested inside a tool_result the cost recurses the same way.
	msgs = []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{{
			Type:    llm.BlockToolResult,
			Content: []llm.ContentBlock{llm.ImageBlock(llm.MediaTypeJPEG, big)},
		}}},
	}
	if got := EstimateTokens(msgs); got != mediaBlockChars/4 {
		t.Fatalf("nested EstimateTokens = %d, want %d", got, mediaBlockChars/4)
	}
}

// The summarizer replay must carry media placeholders, never base64 payloads,
// and must drop boundary markers as before.
func TestStrippedForSummaryReplacesMedia(t *testing.T) {
	payload := "QUJDREVGRw=="
	msgs := []llm.Message{
		llm.BoundaryMessage(llm.BoundaryMeta{}), // must be dropped
		{Role: llm.RoleUser, Content: []llm.ContentBlock{
			llm.TextBlock("see this"),
			llm.ImageBlock(llm.MediaTypePNG, payload),
		}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{{
			Type:    llm.BlockToolResult,
			Content: []llm.ContentBlock{llm.TextBlock("out"), llm.DocumentBlock(llm.MediaTypePDF, payload)},
		}}},
	}
	out := strippedForSummary(msgs)
	if len(out) != 2 {
		t.Fatalf("messages = %d, want 2 (boundary dropped)", len(out))
	}
	for _, m := range out {
		for _, b := range m.Content {
			if b.Type == llm.BlockImage || b.Type == llm.BlockDocument {
				t.Fatalf("media block survived: %+v", b)
			}
			if strings.Contains(b.Text, payload) {
				t.Fatalf("payload leaked into summary replay: %q", b.Text)
			}
		}
	}
	if out[0].Content[1].Type != llm.BlockText || !strings.Contains(out[0].Content[1].Text, "[image image/png omitted") {
		t.Fatalf("image placeholder missing: %+v", out[0].Content[1])
	}
	nested := out[1].Content[0].Content
	if len(nested) != 2 || nested[1].Type != llm.BlockText || !strings.Contains(nested[1].Text, "[document application/pdf omitted") {
		t.Fatalf("nested media not replaced: %+v", nested)
	}
	if nested[0].Text != "out" {
		t.Fatalf("sibling text must survive: %+v", nested[0])
	}
}
