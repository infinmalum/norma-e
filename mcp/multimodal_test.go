package mcp

import (
	"strings"
	"testing"

	"github.com/infinmalum/norma-e/llm"
)

// Text-only responses keep the exact old shape: one text block.
func TestMcpContentToResultTextOnly(t *testing.T) {
	res := mcpContentToResult([]mcpContentBlock{
		{Type: "text", Text: "hello "},
		{Type: "text", Text: "world"},
	}, false)
	if res.IsError {
		t.Fatal("isError must stay false")
	}
	if len(res.Content) != 1 || res.Content[0].Type != llm.BlockText || res.Content[0].Text != "hello world" {
		t.Fatalf("content = %+v", res.Content)
	}
}

// Image blocks pass through as native image blocks in their original position,
// with surrounding text split around them.
func TestMcpContentToResultImagePassthrough(t *testing.T) {
	res := mcpContentToResult([]mcpContentBlock{
		{Type: "text", Text: "before"},
		{Type: "image", MimeType: "image/png", Data: "cG5n"},
		{Type: "text", Text: "after"},
	}, false)
	if len(res.Content) != 3 {
		t.Fatalf("blocks = %d, want [text, image, text]: %+v", len(res.Content), res.Content)
	}
	if res.Content[0].Text != "before" {
		t.Fatalf("block0 = %+v", res.Content[0])
	}
	img := res.Content[1]
	if img.Type != llm.BlockImage || img.Source == nil || img.Source.MediaType != "image/png" || img.Source.Data != "cG5n" {
		t.Fatalf("block1 = %+v", img)
	}
	if res.Content[2].Text != "after" {
		t.Fatalf("block2 = %+v", res.Content[2])
	}
}

// Unrecognized typed content degrades to a notice instead of vanishing.
func TestMcpContentToResultUnknownType(t *testing.T) {
	res := mcpContentToResult([]mcpContentBlock{{Type: "audio", MimeType: "audio/wav"}}, true)
	if !res.IsError {
		t.Fatal("isError must propagate")
	}
	if len(res.Content) != 1 || !strings.Contains(res.Content[0].Text, "[audio content omitted") {
		t.Fatalf("content = %+v", res.Content)
	}
}

// An empty content array still yields a single (empty) text block, matching
// the previous string-accumulation shape.
func TestMcpContentToResultEmpty(t *testing.T) {
	res := mcpContentToResult(nil, false)
	if len(res.Content) != 1 || res.Content[0].Type != llm.BlockText {
		t.Fatalf("content = %+v", res.Content)
	}
}
