package tool

import (
	"archive/zip"
	"bytes"
	"encoding/base64"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/infinmalum/norma-e/llm"
)

// A real 1x1 PNG; everything the Read path sees is magic bytes + payload.
const pngOnePx = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

func writeTestFile(t *testing.T, dir, name string, data []byte) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
	return name
}

func writeTestZip(t *testing.T, dir, name string, files map[string]string) string {
	t.Helper()
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for fn, body := range files {
		w, err := zw.Create(fn)
		if err != nil {
			t.Fatalf("zip create %s: %v", fn, err)
		}
		if _, err := w.Write([]byte(body)); err != nil {
			t.Fatalf("zip write %s: %v", fn, err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatalf("zip close: %v", err)
	}
	return writeTestFile(t, dir, name, buf.Bytes())
}

// PNG (by magic bytes, not extension) is returned as a native image block.
func TestReadReturnsImageBlock(t *testing.T) {
	dir := t.TempDir()
	data, _ := base64.StdEncoding.DecodeString(pngOnePx)
	name := writeTestFile(t, dir, "logo.png", data)
	res := run(t, NewRead(), dir, map[string]any{"file_path": name})
	if len(res.Content) != 1 {
		t.Fatalf("content blocks = %d, want 1", len(res.Content))
	}
	b := res.Content[0]
	if b.Type != llm.BlockImage || b.Source == nil {
		t.Fatalf("block = %+v", b)
	}
	if b.Source.MediaType != llm.MediaTypePNG || b.Source.Data != pngOnePx || b.Source.Type != "base64" {
		t.Fatalf("source = %+v", b.Source)
	}
}

// Extension lies are ignored: a PNG payload named .txt is still an image; a
// text payload named .png is still text.
func TestReadSniffsContentNotExtension(t *testing.T) {
	dir := t.TempDir()
	data, _ := base64.StdEncoding.DecodeString(pngOnePx)
	res := run(t, NewRead(), dir, map[string]any{"file_path": writeTestFile(t, dir, "actually-image.txt", data)})
	if res.Content[0].Type != llm.BlockImage {
		t.Fatalf("png named .txt → %s", res.Content[0].Type)
	}
	res = run(t, NewRead(), dir, map[string]any{"file_path": writeTestFile(t, dir, "actually-text.png", []byte("just text"))})
	if res.Content[0].Type != llm.BlockText || !strings.Contains(res.Content[0].Text, "just text") {
		t.Fatalf("text named .png → %+v", res.Content[0])
	}
}

// JPEG/GIF/WEBP/PDF map to their media types; other binaries stay placeholders.
func TestReadMagicByteDetection(t *testing.T) {
	dir := t.TempDir()
	cases := []struct {
		name     string
		magic    []byte
		wantType llm.BlockType
		wantMT   string
	}{
		{"a.jpg", []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00}, llm.BlockImage, llm.MediaTypeJPEG},
		{"b.gif", []byte("GIF89a\x00\x00"), llm.BlockImage, llm.MediaTypeGIF},
		{"c.webp", []byte("RIFF\x00\x00\x00\x00WEBP"), llm.BlockImage, llm.MediaTypeWEBP},
		{"d.pdf", []byte("%PDF-1.4\nrest"), llm.BlockDocument, llm.MediaTypePDF},
		{"e.bin", []byte{0x00, 0x01, 0x02}, "", ""},
	}
	for _, c := range cases {
		res := run(t, NewRead(), dir, map[string]any{"file_path": writeTestFile(t, dir, c.name, c.magic)})
		got := res.Content[0]
		if c.wantType == "" {
			if got.Type != llm.BlockText || !strings.Contains(got.Text, "binary file") {
				t.Fatalf("%s: want binary placeholder, got %+v", c.name, got)
			}
			continue
		}
		if got.Type != c.wantType || got.Source == nil || got.Source.MediaType != c.wantMT {
			t.Fatalf("%s: got %+v, want %s/%s", c.name, got, c.wantType, c.wantMT)
		}
	}
}

// Oversized media is refused with a hint instead of silently truncating.
func TestReadMediaTooLarge(t *testing.T) {
	dir := t.TempDir()
	data := append([]byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}, make([]byte, maxMediaBytes+1)...)
	res := run(t, NewRead(), dir, map[string]any{"file_path": writeTestFile(t, dir, "big.png", data)})
	if !res.IsError || !strings.Contains(res.Content[0].Text, "up to") {
		t.Fatalf("want size error, got %+v", res.Content[0])
	}
}

func TestReadDocxExtractsParagraphs(t *testing.T) {
	dir := t.TempDir()
	name := writeTestZip(t, dir, "doc.docx", map[string]string{
		"word/document.xml": `<w:document xmlns:w="urn:x"><w:body>` +
			`<w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:t> World</w:t></w:r></w:p>` +
			`<w:p><w:r><w:t>Second</w:t></w:r><w:br/></w:p>` +
			`</w:body></w:document>`,
	})
	res := run(t, NewRead(), dir, map[string]any{"file_path": name})
	text := res.Content[0].Text
	if !strings.Contains(text, "[Office docx: extracted text") {
		t.Fatalf("missing header: %q", text)
	}
	if !strings.Contains(text, "Hello World\nSecond") {
		t.Fatalf("paragraphs wrong: %q", text)
	}
}

func TestReadXlsxExtractsSheets(t *testing.T) {
	dir := t.TempDir()
	name := writeTestZip(t, dir, "book.xlsx", map[string]string{
		"xl/workbook.xml": `<workbook xmlns="urn:main" xmlns:r="urn:rel">` +
			`<sheets><sheet name="Data" sheetId="1" r:id="rId1"/></sheets></workbook>`,
		"xl/_rels/workbook.xml.rels": `<Relationships xmlns="urn:rel">` +
			`<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>`,
		"xl/sharedStrings.xml": `<sst xmlns="urn:main">` +
			`<si><t>Alpha</t></si><si><r><t>Beta</t></r><r><t>2</t></r></si></sst>`,
		"xl/worksheets/sheet1.xml": `<worksheet xmlns="urn:main"><sheetData>` +
			`<row r="1"><c r="A1" t="s"><v>0</v></c><c r="C1" t="s"><v>1</v></c></row>` +
			`<row r="2"><c r="A2"><v>42</v></c></row>` +
			`</sheetData></worksheet>`,
	})
	res := run(t, NewRead(), dir, map[string]any{"file_path": name})
	text := res.Content[0].Text
	if !strings.Contains(text, "## Data") {
		t.Fatalf("sheet header missing: %q", text)
	}
	if !strings.Contains(text, "Alpha\t\tBeta2") {
		t.Fatalf("row 1 wrong (gap A1→C1 must pad): %q", text)
	}
	if !strings.Contains(text, "42") {
		t.Fatalf("row 2 wrong: %q", text)
	}
}

func TestReadPptxExtractsSlidesInOrder(t *testing.T) {
	dir := t.TempDir()
	name := writeTestZip(t, dir, "deck.pptx", map[string]string{
		"ppt/slides/slide2.xml": `<p:sld xmlns:p="urn:x"><p:txBody><a:p><a:r><a:t>Second</a:t></a:r></a:p></p:txBody></p:sld>`,
		"ppt/slides/slide10.xml": `<p:sld xmlns:p="urn:x"><p:txBody><a:p><a:r><a:t>Tenth</a:t></a:r></a:p></p:txBody></p:sld>`,
		"ppt/slides/slide1.xml":  `<p:sld xmlns:p="urn:x"><p:txBody><a:p><a:r><a:t>First</a:t></a:r></a:p></p:txBody></p:sld>`,
	})
	res := run(t, NewRead(), dir, map[string]any{"file_path": name})
	text := res.Content[0].Text
	i1, i2, i10 := strings.Index(text, "First"), strings.Index(text, "Second"), strings.Index(text, "Tenth")
	if i1 < 0 || i2 < 0 || i10 < 0 {
		t.Fatalf("slide text missing: %q", text)
	}
	if !(i1 < i2 && i2 < i10) {
		t.Fatalf("slides out of numeric order (slide10 must be last): %q", text)
	}
}

// A zip that is not an Office document falls through to the binary placeholder.
func TestReadNonOfficeZipIsBinary(t *testing.T) {
	dir := t.TempDir()
	name := writeTestZip(t, dir, "arc.zip", map[string]string{"random/file.txt": "x"})
	res := run(t, NewRead(), dir, map[string]any{"file_path": name})
	if !strings.Contains(res.Content[0].Text, "binary file") {
		t.Fatalf("want placeholder, got %q", res.Content[0].Text)
	}
}

// UTF-16 (BOM'd) text must be transcoded, not mistaken for binary.
func TestReadUTF16LE(t *testing.T) {
	dir := t.TempDir()
	raw := []byte{0xFF, 0xFE, 'h', 0, 'e', 0, 'l', 0, 'l', 0, 'o', 0}
	res := run(t, NewRead(), dir, map[string]any{"file_path": writeTestFile(t, dir, "u.txt", raw)})
	if !strings.Contains(res.Content[0].Text, "hello") {
		t.Fatalf("utf-16 not transcoded: %q", res.Content[0].Text)
	}
}

// Media blocks are charged their fixed cost, so a payload far beyond the text
// budget passes through CapResult intact.
func TestCapResultMediaFixedCost(t *testing.T) {
	tc := &ToolContext{MaxOutputChars: 30000}
	big := strings.Repeat("A", 100000)
	res := Result{Content: []llm.ContentBlock{llm.ImageBlock(llm.MediaTypePNG, big)}}
	got := CapResult(tc, res)
	if len(got.Content) != 1 || got.Content[0].Type != llm.BlockImage || got.Content[0].Source.Data != big {
		t.Fatalf("media result was altered: %+v", got.Content)
	}
}

// Over budget only because of text: the text is bounded; the media block rides
// along inside the JSON fallback.
func TestCapResultTextOverBudgetWithMedia(t *testing.T) {
	tc := &ToolContext{MaxOutputChars: 1000}
	big := strings.Repeat("A", 5000)
	res := Result{Content: []llm.ContentBlock{llm.TextBlock(big), llm.ImageBlock(llm.MediaTypePNG, "dA==")}}
	got := CapResult(tc, res)
	if len(got.Content) != 1 || got.Content[0].Type != llm.BlockText {
		t.Fatalf("over-budget result = %+v", got.Content)
	}
	if len(got.Content[0].Text) > 1000+200 {
		t.Fatalf("text not bounded: %d", len(got.Content[0].Text))
	}
}
