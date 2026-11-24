package tool

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/infinmalum/norma-e/llm"
)

const defaultMaxOutput = 30000

func maxOut(tc *ToolContext) int {
	if tc != nil && tc.MaxOutputChars > 0 {
		return tc.MaxOutputChars
	}
	return defaultMaxOutput
}

var spillSeq atomic.Int64

// Capture bounds a tool's textual output for the model. When the output exceeds
// the limit (ToolContext.MaxOutputChars, default 30000) and the host configured
// an OutputDir, the FULL output is written to a file there and a tight head + a
// pointer to that file is returned — so the model can read slices on demand
// (grep/sed) instead of losing the overflow. Without an OutputDir it falls back
// to a head+tail truncation (the overflow is discarded). Tools that produce
// potentially large output can return CapResult(tc, Text(output)) to preserve
// capture metadata when the harness checks the result again.
//
// The pointer is always an ABSOLUTE path: a relative one only resolves from the
// working directory it was computed against, and the reader is often somewhere
// else — a later command after a `cd`, a resumed session with a different
// WorkingDir, or an out-of-band tool (grep/sed) the model runs itself. Heads cut
// by Capture and tail-fallbacks in truncate() are aligned to UTF-8 rune
// boundaries, so a bounded result never carries a half-encoded character.
func Capture(tc *ToolContext, s string) string {
	max := maxOut(tc)
	if len(s) <= max {
		return s
	}
	if tc != nil && tc.OutputDir != "" {
		if path, err := spillOutput(tc.OutputDir, s); err == nil {
			lines := strings.Count(s, "\n") + 1
			return cutHead(s, max) + fmt.Sprintf(
				"\n\n... <persisted-output>[Output too large: full %d bytes / %d lines.Full output saved to  %s </persisted-output>",
				len(s), lines, path)
		}
	}
	return truncate(s, max)
}

// CaptureOnce bounds output without interpreting its text as capture metadata.
// Deprecated: use Capture for strings or CapResult for tool results.
func CaptureOnce(tc *ToolContext, s string) string { return Capture(tc, s) }

// mediaFixedChars is the budget cost of one image/document block. The payload
// is base64 and can far exceed the whole text budget on its own, but its model
// cost is fixed (~2000 tokens ≈ 8000 chars, the same convention compaction
// uses), so it is charged the fixed cost instead of its wire length. Charging
// the raw JSON length would push any media-bearing result past the budget and
// destroy the image by replacing the whole result with truncated JSON text.
const mediaFixedChars = 8000

// CapResult applies a shared budget across all content blocks. Capture may add
// a short truncation notice or an absolute spill path outside the body budget.
// Extra messages are host-provided conversation instructions and have their own
// contract (for example, Skill injects instructions verbatim).
func CapResult(tc *ToolContext, res Result) Result {
	limit := maxOut(tc)
	if res.cappedAt > 0 && res.cappedAt <= limit {
		return res
	}
	remaining := limit
	textOnly := true
	for _, block := range res.Content {
		if block.Type == llm.BlockText {
			remaining -= 1 + len(block.Text)
		} else {
			textOnly = false
			if block.Type == llm.BlockImage || block.Type == llm.BlockDocument {
				remaining -= mediaFixedChars
				continue
			}
			encoded, err := json.Marshal(block)
			if err != nil {
				return Errorf("Error: invalid tool output")
			}
			remaining -= len(encoded)
		}
	}
	if remaining >= 0 {
		return res
	}
	var body string
	if textOnly {
		var joined strings.Builder
		for i, block := range res.Content {
			if i > 0 {
				joined.WriteByte('\n')
			}
			joined.WriteString(block.Text)
		}
		body = joined.String()
	} else {
		encoded, err := json.Marshal(res.Content)
		if err != nil {
			return Errorf("Error: invalid tool output")
		}
		body = string(encoded)
	}
	res.Content = []llm.ContentBlock{llm.TextBlock(Capture(tc, body))}
	res.cappedAt = limit
	return res
}

func spillOutput(dir, s string) (string, error) {
	// Resolve the directory first: the returned pointer is what the model will
	// read, so it must not depend on anyone's current working directory.
	//
	// A failure here is reported rather than papered over with the relative
	// path. filepath.Abs only fails when dir is relative and os.Getwd fails —
	// i.e. the process's cwd has been deleted, which is precisely when a
	// relative pointer is most useless. Capture's caller falls back to
	// truncate(), which hands the model bounded output instead of a path it
	// cannot resolve.
	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(abs, 0o755); err != nil {
		return "", err
	}
	name := fmt.Sprintf("output-%d-%d.txt", time.Now().Unix(), spillSeq.Add(1))
	path := filepath.Join(abs, name)
	if err := os.WriteFile(path, []byte(s), 0o644); err != nil {
		return "", err
	}
	return path, nil
}

// resolvePath joins a relative path against the working directory.
func resolvePath(tc *ToolContext, p string) string {
	if filepath.IsAbs(p) || tc == nil || tc.WorkingDir == "" {
		return p
	}
	return filepath.Join(tc.WorkingDir, p)
}

// truncate trims s to max bytes, keeping head and tail. Both cuts are aligned to
// rune boundaries (see cutHead/cutTail): slicing bytes blindly can split a
// multi-byte character, and the mojibake then travels into the model's context
// (and into transcripts) as invalid UTF-8.
func truncate(s string, max int) string {
	if max <= 0 || len(s) <= max {
		return s
	}
	half := max / 2
	return cutHead(s, half) + fmt.Sprintf("\n\n... [%d characters truncated] ...\n\n", len(s)-max) + cutTail(s, half)
}

// byteSeq is string or []byte: both index to a byte, so the rune-boundary
// primitives below serve either without converting (and allocating) between them.
type byteSeq interface{ string | []byte }

// runeSafeLen returns the largest n <= max with b[:n] ending on a rune boundary.
func runeSafeLen[T byteSeq](b T, max int) int {
	if max <= 0 {
		return 0
	}
	if len(b) <= max {
		return len(b)
	}
	for max > 0 && !utf8.RuneStart(b[max]) {
		max--
	}
	return max
}

// runeSafeStart returns the smallest n with b[n:] starting on a rune boundary,
// i.e. the number of leading UTF-8 continuation bytes to drop. Use it when the
// length is already right and only the boundary is in question — after slicing a
// ring buffer, or seeking a file to an arbitrary byte offset.
func runeSafeStart[T byteSeq](b T) int {
	i := 0
	for i < len(b) && !utf8.RuneStart(b[i]) {
		i++
	}
	return i
}

// cutHead returns the longest prefix of s that is at most max bytes AND ends on a
// rune boundary. max<len(s) is assumed (callers check the length first).
func cutHead(s string, max int) string { return s[:runeSafeLen(s, max)] }

// cutTail returns the longest suffix of s that is at most max bytes AND starts on
// a rune boundary.
func cutTail(s string, max int) string {
	if max <= 0 {
		return ""
	}
	if len(s) <= max {
		return s
	}
	s = s[len(s)-max:]
	return s[runeSafeStart(s):]
}
