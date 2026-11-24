package noaadapter

import (
	"strings"
	"testing"

	"github.com/infinmalum/norma-e/llm"
	"github.com/infinmalum/norma-e/noa"
)

// Media must round-trip the noa pipeline: projection yields stub cores with
// distinct identities, and reassembly restores the original payloads — top-level
// image blocks and media nested inside tool results alike.
func TestProjectMediaRoundTrip(t *testing.T) {
	history := []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{
			llm.TextBlock("look"),
			llm.ImageBlock(llm.MediaTypePNG, "cG5nLWJ5dGVz"),
		}},
		{Role: llm.RoleAssistant, Content: []llm.ContentBlock{
			{Type: llm.BlockToolUse, ID: "c1", Name: "Read", Input: []byte(`{"file_path":"x.pdf"}`)},
		}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{{
			Type:      llm.BlockToolResult,
			ToolUseID: "c1",
			Content: []llm.ContentBlock{
				llm.TextBlock("out"),
				llm.DocumentBlock(llm.MediaTypePDF, "cGRmLWJ5dGVz"),
			},
		}}},
	}

	cores, sc := Project(history)

	var imgCore, docCore, resultCore *noa.CoreMessage
	for i := range cores {
		c := cores[i]
		switch c.ContentType {
		case noa.CTImage:
			imgCore = &cores[i]
		case noa.CTDocument:
			docCore = &cores[i]
		case noa.CTToolResult:
			resultCore = &cores[i]
		}
	}
	if imgCore == nil || docCore == nil || resultCore == nil {
		t.Fatalf("media cores missing: %+v", cores)
	}
	// The stub carries media type + payload hash, never the bytes themselves.
	if !strings.Contains(imgCore.Text, "[image image/png") || strings.Contains(imgCore.Text, "cG5nLWJ5dGVz") {
		t.Fatalf("image stub wrong: %q", imgCore.Text)
	}
	if docCore.Role != noa.RoleTool {
		t.Fatalf("nested media should ride the tool result: %+v", docCore)
	}
	if o := sc.Origins[docCore.ID]; o.nested < 0 {
		t.Fatalf("nested origin not recorded: %+v", o)
	}

	out := Reassemble(cores, history, sc, ReassembleOptions{})
	if len(out) != 3 {
		t.Fatalf("reassembled %d messages, want 3: %+v", len(out), out)
	}
	img := out[0].Content[1]
	if img.Type != llm.BlockImage || img.Source == nil || img.Source.Data != "cG5nLWJ5dGVz" {
		t.Fatalf("image payload not restored: %+v", img)
	}
	res := out[2].Content[0]
	if res.Type != llm.BlockToolResult || res.ToolUseID != "c1" || len(res.Content) != 1 || res.Content[0].Text != "out" {
		t.Fatalf("tool result shape wrong: %+v", res)
	}
	// The nested document is re-attached as a sibling block of its tool_result
	// (the pipeline's rebuild replaces nested content; the payload rides at the
	// top level of the same message).
	doc := out[2].Content[1]
	if doc.Type != llm.BlockDocument || doc.Source == nil || doc.Source.Data != "cGRmLWJ5dGVz" {
		t.Fatalf("nested document not restored: %+v", doc)
	}
}

// Distinct payloads must project to distinct ids (the stub embeds a payload
// hash), while identical payloads cluster like identical text does.
func TestMediaStubIdentity(t *testing.T) {
	media := llm.ImageBlock(llm.MediaTypePNG, "AAA")
	other := llm.ImageBlock(llm.MediaTypePNG, "BBB")
	cores, _ := Project([]llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{media}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{other}},
		{Role: llm.RoleUser, Content: []llm.ContentBlock{media}},
	})
	if len(cores) != 3 {
		t.Fatalf("cores = %d, want 3", len(cores))
	}
	if cores[0].ID == cores[1].ID {
		t.Fatalf("distinct payloads share an id: %s", cores[0].ID)
	}
	if cores[0].ID == cores[2].ID {
		t.Fatal("identical payloads must cluster, but got duplicate ids")
	}
	if !strings.HasPrefix(cores[2].ID, cores[0].ID) {
		t.Fatalf("clustered id lost its base hash: %s vs %s", cores[2].ID, cores[0].ID)
	}
}

// The estimator charges media blocks the fixed token cost.
func TestEstimateMessagesMedia(t *testing.T) {
	msgs := []llm.Message{
		{Role: llm.RoleUser, Content: []llm.ContentBlock{llm.ImageBlock(llm.MediaTypePNG, "x"), llm.TextBlock("hello")}},
	}
	want := noaMediaTokens + len("hello")/4
	if got := estimateMessages(msgs); got != want {
		t.Fatalf("estimate = %d, want %d", got, want)
	}
}
