package tool

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf16"

	"github.com/infinmalum/norma-e/llm"
)

// NewRead builds the Read tool: reads a file with cat -n style line numbers and
// optional offset/limit. Images and PDFs (up to maxMediaBytes) are returned as
// native image/document content blocks for multimodal providers; Office
// documents (docx/xlsx/pptx) are returned as extracted text; UTF-16 text is
// transcoded to UTF-8; other binaries are reported, not shown.
func NewRead() CoreTool {
	return Build(Spec{
		Name:        "Read",
		Description: "Reads a file from the local filesystem and returns its contents with line numbers. Use offset/limit for large files. Always read a file before editing it. PNG/JPEG/GIF/WEBP images and PDFs up to 5MB come back as native image/document blocks (supported providers only); docx/xlsx/pptx come back as extracted text; other binary files are reported, not shown.",
		Prompt:      "file_path may be absolute or relative to the working directory. Output lines are prefixed with their 1-based line number. Media payloads do not carry line numbers.",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"file_path": map[string]any{"type": "string", "description": "Path to the file to read."},
				"offset":    map[string]any{"type": "integer", "description": "1-based line to start from."},
				"limit":     map[string]any{"type": "integer", "description": "Max lines to read (default 2000)."},
			},
			"required": []any{"file_path"},
		},
		ReadOnly:    always,
		Concurrent:  always,
		Permissions: allowReadOnly,
		Run:         runRead,
	})
}

const maxLineLen = 2000

func runRead(_ context.Context, input json.RawMessage, tc *ToolContext) (Result, error) {
	var in struct {
		FilePath string `json:"file_path"`
		Offset   int    `json:"offset"`
		Limit    int    `json:"limit"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return Result{}, err
	}
	path := resolvePath(tc, in.FilePath)
	info, err := os.Stat(path)
	if err != nil {
		return Errorf(fmt.Sprintf("Error reading %s: %v", in.FilePath, err)), nil
	}
	if info.IsDir() {
		return Errorf(fmt.Sprintf("Error: %s is a directory, not a file. Use LS or Glob to list its contents.", in.FilePath)), nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return Errorf(fmt.Sprintf("Error reading %s: %v", in.FilePath, err)), nil
	}
	// UTF-16 text carries a NUL byte in the high half of every ASCII character
	// and would trip the binary heuristic below; transcode to UTF-8 first so
	// Excel's "Unicode text" exports and Windows UTF-16 files read normally.
	if s, ok := decodeUTF16(data); ok {
		data = []byte(s)
	}
	if res, handled := readMedia(tc, in.FilePath, data); handled {
		return res, nil
	}
	if isBinary(data) {
		return Text(fmt.Sprintf("[binary file %s not shown — %d bytes]", in.FilePath, len(data))), nil
	}
	lines := strings.Split(string(data), "\n")
	start := 0
	if in.Offset > 0 {
		start = in.Offset - 1
	}
	if start > len(lines) {
		start = len(lines)
	}
	limit := in.Limit
	if limit <= 0 {
		limit = 2000
	}
	end := start + limit
	if end > len(lines) {
		end = len(lines)
	}
	var b strings.Builder
	for i := start; i < end; i++ {
		line := lines[i]
		if len(line) > maxLineLen {
			line = line[:maxLineLen] + "… [line truncated]"
		}
		fmt.Fprintf(&b, "%6d\t%s\n", i+1, line)
	}
	if b.Len() == 0 {
		if len(data) == 0 {
			return Text("(file is empty)"), nil
		}
		return Text("(no lines in the requested range)"), nil
	}
	res := Text(readCapture(tc, b.String(), start))
	// Read already limits its output and supplies an exact continuation offset.
	res.cappedAt = maxOut(tc)
	return res, nil
}

// readOutputBonus lifts the Read tool's output budget above the shared default
// (ToolContext.MaxOutputChars): Read gets a little more room before truncating.
const readOutputBonus = 1000

// readCapture bounds Read's line-numbered output. When it exceeds the Read budget
// (MaxToolOutputChars + readOutputBonus) it cuts at a line boundary and tells the
// model the exact offset to continue from — the source file is already on disk, so
// there is no need to spill a copy: the model just pages on with Read(offset=…).
// start is the 0-based line the output began at.
func readCapture(tc *ToolContext, s string, start int) string {
	max := maxOut(tc) + readOutputBonus
	if len(s) <= max {
		return s
	}
	head, headLines := headWithinBudget(s, max)
	nextOffset := start + headLines + 1 // 1-based file line to continue from
	return head + fmt.Sprintf(
		"\n... [Read truncated to fit the output limit. Continue reading the file with Read offset=%d (and an optional limit), or narrow with Grep/`sed -n`.] ...",
		nextOffset)
}

// headWithinBudget returns the largest whole-line prefix of s within max bytes and
// the number of lines it contains. Falls back to a hard cut if a single line
// already exceeds the budget (no newline within reach).
func headWithinBudget(s string, max int) (string, int) {
	if len(s) <= max {
		return s, strings.Count(s, "\n")
	}
	cut := strings.LastIndexByte(s[:max], '\n')
	if cut < 0 {
		return cutHead(s, max), 0
	}
	head := s[:cut+1]
	return head, strings.Count(head, "\n")
}

// isBinary reports whether data looks like a binary (non-text) file, using the
// common heuristic of a NUL byte in the first chunk.
func isBinary(data []byte) bool {
	n := len(data)
	if n > 8000 {
		n = 8000
	}
	for i := 0; i < n; i++ {
		if data[i] == 0 {
			return true
		}
	}
	return false
}

// maxMediaBytes caps the raw payload Read will return as an image or document
// content block. Anthropic accepts ~5MB per image and 32MB per PDF; 5MB for
// both keeps the base64 expansion (~1.33x) and the request size bounded. Media
// results self-cap (cappedAt), so they bypass the shared text budget.
const maxMediaBytes = 5 << 20

// sniffMedia reports the media type of an image or PDF by magic bytes — never
// by file extension, which the model may have gotten wrong.
func sniffMedia(data []byte) string {
	switch {
	case len(data) >= 8 && bytes.Equal(data[:8], []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}):
		return llm.MediaTypePNG
	case len(data) >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF:
		return llm.MediaTypeJPEG
	case len(data) >= 6 && (bytes.Equal(data[:6], []byte("GIF87a")) || bytes.Equal(data[:6], []byte("GIF89a"))):
		return llm.MediaTypeGIF
	case len(data) >= 12 && bytes.Equal(data[:4], []byte("RIFF")) && bytes.Equal(data[8:12], []byte("WEBP")):
		return llm.MediaTypeWEBP
	case len(data) >= 5 && bytes.Equal(data[:5], []byte("%PDF-")):
		return llm.MediaTypePDF
	}
	return ""
}

// decodeUTF16 transcodes a UTF-16 byte order mark'd payload to UTF-8. ok=false
// for everything else (including plain ASCII, which has no BOM).
func decodeUTF16(data []byte) (string, bool) {
	if len(data) < 4 || len(data)%2 != 0 {
		return "", false
	}
	littleEndian := false
	switch {
	case data[0] == 0xFF && data[1] == 0xFE:
		littleEndian = true
	case data[0] == 0xFE && data[1] == 0xFF:
	default:
		return "", false
	}
	u16 := make([]uint16, 0, (len(data)-2)/2)
	for i := 2; i+1 < len(data); i += 2 {
		if littleEndian {
			u16 = append(u16, uint16(data[i])|uint16(data[i+1])<<8)
		} else {
			u16 = append(u16, uint16(data[i])<<8|uint16(data[i+1]))
		}
	}
	return string(utf16.Decode(u16)), true
}

// readMedia handles Read's non-plain-text inputs by content: images and PDFs
// become native image/document blocks; OOXML Office documents are decompressed
// to text. handled=false falls through to the binary placeholder. Media results
// set cappedAt so the shared text budget never truncates their payload.
func readMedia(tc *ToolContext, display string, data []byte) (Result, bool) {
	media := sniffMedia(data)
	if media == "" {
		return readOffice(tc, display, data)
	}
	if len(data) > maxMediaBytes {
		return Errorf(fmt.Sprintf(
			"Error: %s is a %d-byte %s; Read returns media up to %d bytes as content blocks. Extract the needed content another way.",
			display, len(data), media, maxMediaBytes)), true
	}
	enc := base64.StdEncoding.EncodeToString(data)
	var res Result
	if media == llm.MediaTypePDF {
		res = Result{Content: []llm.ContentBlock{llm.DocumentBlock(media, enc)}}
	} else {
		res = Result{Content: []llm.ContentBlock{llm.ImageBlock(media, enc)}}
	}
	res.cappedAt = maxOut(tc)
	return res, true
}

// readOffice extracts text from OOXML Office documents (docx/xlsx/pptx, which
// are zip archives of XML parts). A zip that is not an Office document is not
// handled, so it falls through to the binary placeholder.
func readOffice(tc *ToolContext, display string, data []byte) (Result, bool) {
	if len(data) < 4 || data[0] != 'P' || data[1] != 'K' {
		return Result{}, false
	}
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return Result{}, false // corrupt zip: treat as an ordinary binary
	}
	kind, ok := sniffOffice(zr)
	if !ok {
		return Result{}, false
	}
	text, err := extractOffice(zr, kind)
	if err != nil {
		return Errorf(fmt.Sprintf("Error reading %s: %s document could not be read: %v", display, kind, err)), true
	}
	body := fmt.Sprintf("[Office %s: extracted text of %s (%d source bytes)]\n%s", kind, display, len(data), text)
	// Extracted text is synthetic, so offset paging (the regular Read path)
	// does not apply: bound it with the head+tail truncation instead, and mark
	// the result self-capped like the other Read outputs.
	body = truncate(body, maxOut(tc)+readOutputBonus)
	res := Text(body)
	res.cappedAt = maxOut(tc)
	return res, true
}

// NewWrite builds the Write tool: creates or overwrites a file.
func NewWrite() CoreTool {
	return Build(Spec{
		Name:        "Write",
		Description: "Writes a file to the local filesystem, creating parent directories and overwriting any existing file. Prefer Edit for modifying existing files.",
		Prompt:      "Provide the complete file contents — Write replaces the whole file.",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"file_path": map[string]any{"type": "string", "description": "Path of the file to write."},
				"content":   map[string]any{"type": "string", "description": "Full contents to write."},
			},
			"required": []any{"file_path", "content"},
		},
		Run: runWrite,
	})
}

func runWrite(_ context.Context, input json.RawMessage, tc *ToolContext) (Result, error) {
	var in struct {
		FilePath string `json:"file_path"`
		Content  string `json:"content"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return Result{}, err
	}
	path := resolvePath(tc, in.FilePath)
	existed := false
	if info, err := os.Stat(path); err == nil {
		if info.IsDir() {
			return Errorf(fmt.Sprintf("Error: %s is an existing directory, not a file.", in.FilePath)), nil
		}
		existed = true
	}
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return Errorf(fmt.Sprintf("Error creating directory: %v", err)), nil
		}
	}
	if err := os.WriteFile(path, []byte(in.Content), 0o644); err != nil {
		return Errorf(fmt.Sprintf("Error writing %s: %v", in.FilePath, err)), nil
	}
	verb := "Created"
	if existed {
		verb = "Overwrote"
	}
	return Text(fmt.Sprintf("%s %s (%d bytes)", verb, in.FilePath, len(in.Content))), nil
}

// NewEdit builds the Edit tool: exact string replacement with uniqueness check.
func NewEdit() CoreTool {
	return Build(Spec{
		Name:        "Edit",
		Description: "Performs an exact string replacement in a file. old_string must match exactly once unless replace_all is set. Read the file first so old_string matches precisely.",
		Prompt:      "old_string must match the file exactly (whitespace included) and be unique unless replace_all is true. new_string must differ from old_string.",
		Schema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"file_path":   map[string]any{"type": "string", "description": "Path of the file to edit."},
				"old_string":  map[string]any{"type": "string", "description": "Exact text to replace."},
				"new_string":  map[string]any{"type": "string", "description": "Replacement text."},
				"replace_all": map[string]any{"type": "boolean", "description": "Replace all occurrences."},
			},
			"required": []any{"file_path", "old_string", "new_string"},
		},
		Run: runEdit,
	})
}

func runEdit(_ context.Context, input json.RawMessage, tc *ToolContext) (Result, error) {
	var in struct {
		FilePath   string `json:"file_path"`
		OldString  string `json:"old_string"`
		NewString  string `json:"new_string"`
		ReplaceAll bool   `json:"replace_all"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return Result{}, err
	}
	if in.OldString == in.NewString {
		return Errorf("Error: old_string and new_string are identical"), nil
	}
	path := resolvePath(tc, in.FilePath)
	data, err := os.ReadFile(path)
	if err != nil {
		return Errorf(fmt.Sprintf("Error reading %s: %v", in.FilePath, err)), nil
	}
	content := string(data)
	count := strings.Count(content, in.OldString)
	if count == 0 {
		return Errorf(fmt.Sprintf("Error: old_string not found in %s", in.FilePath)), nil
	}
	if count > 1 && !in.ReplaceAll {
		return Errorf(fmt.Sprintf("Error: old_string is not unique in %s (%d matches); add context or set replace_all", in.FilePath, count)), nil
	}
	var updated string
	if in.ReplaceAll {
		updated = strings.ReplaceAll(content, in.OldString, in.NewString)
	} else {
		updated = strings.Replace(content, in.OldString, in.NewString, 1)
	}
	if err := os.WriteFile(path, []byte(updated), 0o644); err != nil {
		return Errorf(fmt.Sprintf("Error writing %s: %v", in.FilePath, err)), nil
	}
	return Text(fmt.Sprintf("Edited %s (%d replacement(s))", in.FilePath, count)), nil
}
