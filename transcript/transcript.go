// Package transcript persists a conversation to an append-only JSONL log on
// disk and reconstructs it for resume. The main session is written to
// <dir>/<sessionID>.jsonl; each
// subagent is written to its own sidechain file under
// <dir>/<sessionID>/subagents/agent-<agentID>.jsonl, so subagent tokens are
// counted and stored separately (aggregate them offline if needed).
//
// The log is the full-fidelity history: every user/assistant/tool message is
// recorded as it is born. When compaction inserts a boundary, the post-boundary
// working set — the boundary marker, the summary, and the kept recent tail — is
// ALSO persisted as messages, so a cold-start resume restores the COMPACTED view
// (via llm.MessagesForAPI, which slices from the last boundary) instead of
// replaying raw history and re-summarizing. The kept tail
// is re-recorded after the boundary (it also exists earlier as births); only the
// post-boundary copy reaches the model. Lightweight working-set edits that do not
// add a boundary (MicroCompact tool-result clearing) stay ephemeral and
// are not persisted. A separate Type=="boundary" record is written for
// observability and is skipped during reconstruction.
package transcript

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/infinmalum/norma-e/llm"
)

// Record is one line of a JSONL transcript.
type Record struct {
	UUID        string            `json:"uuid"`
	ParentUUID  string            `json:"parent_uuid,omitempty"`
	SessionID   string            `json:"session_id"`
	AgentID     string            `json:"agent_id,omitempty"`
	IsSidechain bool              `json:"is_sidechain,omitempty"`
	Timestamp   string            `json:"timestamp"`
	Type        string            `json:"type"` // "message" | "boundary"
	Message     *llm.Message      `json:"message,omitempty"`
	Usage       *llm.Usage        `json:"usage,omitempty"`    // assistant turns only
	Boundary    *llm.BoundaryMeta `json:"boundary,omitempty"` // Type=="boundary"
}

// Store is a directory of session transcripts.
type Store struct {
	dir string
	mu  sync.Mutex // serializes appends across writers in this process
}

// NewStore returns a Store rooted at dir (created lazily on first write).
func NewStore(dir string) *Store { return &Store{dir: dir} }

// MainPath is the JSONL file for a top-level session.
func (s *Store) MainPath(sessionID string) string {
	return filepath.Join(s.dir, sessionID+".jsonl")
}

// AgentPath is the sidechain JSONL file for a subagent of a session.
func (s *Store) AgentPath(sessionID, agentID string) string {
	return filepath.Join(s.dir, sessionID, "subagents", "agent-"+agentID+".jsonl")
}

// Append writes records to path as JSONL, creating parent directories. It is
// safe for concurrent use within the process.
func (s *Store) Append(path string, recs ...Record) error {
	if len(recs) == 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	if err := prepareAppend(f, w); err != nil {
		return err
	}
	for _, r := range recs {
		b, err := json.Marshal(r)
		if err != nil {
			return err
		}
		if _, err := w.Write(b); err != nil {
			return err
		}
		if err := w.WriteByte('\n'); err != nil {
			return err
		}
	}
	return w.Flush()
}

// Load reads all records from a transcript file. A missing file yields no
// records and no error (a fresh session). An incomplete final JSON record
// without a newline is ignored. A malformed complete line is an error.
func (s *Store) Load(path string) ([]Record, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	recs, _, err := readRecords(f)
	return recs, err
}

func readRecords(r io.Reader) (recs []Record, partialAt int64, err error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 64*1024*1024)
	// Preserve delimiters so a torn write can be distinguished from a bad
	// complete line, including when that complete line is the last line.
	sc.Split(func(data []byte, atEOF bool) (int, []byte, error) {
		if i := bytes.IndexByte(data, '\n'); i >= 0 {
			return i + 1, data[:i+1], nil
		}
		if atEOF && len(data) > 0 {
			return len(data), data, nil
		}
		return 0, nil, nil
	})
	var offset int64
	for sc.Scan() {
		line := sc.Bytes()
		start := offset
		offset += int64(len(line))
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var record Record
		if err := json.Unmarshal(line, &record); err != nil {
			if line[len(line)-1] != '\n' && isPartialRecord(line) {
				return recs, start, nil
			}
			return nil, -1, fmt.Errorf("transcript: corrupt record at byte %d: %w", start, err)
		}
		recs = append(recs, record)
	}
	return recs, -1, sc.Err()
}

func isPartialRecord(line []byte) bool {
	var record json.RawMessage
	return errors.Is(json.NewDecoder(bytes.NewReader(line)).Decode(&record), io.ErrUnexpectedEOF)
}

// Before extending an interrupted log, preserve the original file in a private
// backup. Only discard bytes after the last complete record once the backup
// has been written and synced.
func prepareAppend(f *os.File, w *bufio.Writer) error {
	info, err := f.Stat()
	if err != nil || info.Size() == 0 {
		return err
	}
	var last [1]byte
	if _, err := f.ReadAt(last[:], info.Size()-1); err != nil {
		return err
	}
	if last[0] == '\n' {
		return nil
	}
	_, partialAt, err := readRecords(f)
	if err != nil {
		return err
	}
	if partialAt < 0 {
		return w.WriteByte('\n')
	}
	backup, err := os.CreateTemp(filepath.Dir(f.Name()), filepath.Base(f.Name())+".recovery-*")
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(backup, io.NewSectionReader(f, 0, info.Size()))
	if err := errors.Join(copyErr, backup.Sync(), backup.Close()); err != nil {
		return fmt.Errorf("transcript: could not preserve original log in %s: %w", backup.Name(), err)
	}
	return f.Truncate(partialAt)
}

// Messages reconstructs the raw conversation and cumulative usage from records.
// Observability-only boundary records are skipped. The persisted boundary
// message and its working set remain available for replay.
func Messages(recs []Record) ([]llm.Message, llm.Usage) {
	var msgs []llm.Message
	var total llm.Usage
	for _, r := range recs {
		if r.Type == "boundary" || r.Message == nil {
			continue
		}
		msgs = append(msgs, *r.Message)
		if r.Usage != nil {
			total.Add(*r.Usage)
		}
	}
	return msgs, total
}

// LastUUID returns the UUID of the last record, for continuing the parent chain.
func LastUUID(recs []Record) string {
	if len(recs) == 0 {
		return ""
	}
	return recs[len(recs)-1].UUID
}

// Writer binds a Store, a file path, and a session/agent identity, assigning a
// UUID and parent link to each record. It satisfies harness.Recorder. Append
// errors are retained (see Err) rather than panicking, so a failing disk
// degrades to in-memory operation instead of crashing the agent loop.
type Writer struct {
	store       *Store
	path        string
	sessionID   string
	agentID     string
	isSidechain bool

	mu   sync.Mutex
	last string
	now  func() time.Time
	err  error
}

// NewWriter returns a Writer for a fresh transcript file. agentID == "" writes
// the main session file; a non-empty agentID writes a subagent sidechain.
func (s *Store) NewWriter(sessionID, agentID string) *Writer {
	w := &Writer{store: s, sessionID: sessionID, agentID: agentID, now: time.Now}
	if agentID == "" {
		w.path = s.MainPath(sessionID)
	} else {
		w.path = s.AgentPath(sessionID, agentID)
		w.isSidechain = true
	}
	return w
}

// ResumeWriter returns a Writer that continues an existing main-session file,
// chaining new records after lastUUID.
func (s *Store) ResumeWriter(sessionID, lastUUID string) *Writer {
	w := s.NewWriter(sessionID, "")
	w.last = lastUUID
	return w
}

// Err returns the first append error encountered, if any.
func (w *Writer) Err() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.err
}

func (w *Writer) write(r Record) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.err != nil {
		return // retain a consistent durable prefix after an append failure
	}
	r.SessionID = w.sessionID
	r.AgentID = w.agentID
	r.IsSidechain = w.isSidechain
	r.UUID = newID()
	r.ParentUUID = w.last
	r.Timestamp = w.now().UTC().Format(time.RFC3339Nano)
	if err := w.store.Append(w.path, r); err != nil {
		w.err = err
		return
	}
	w.last = r.UUID
}

// RecordMessage persists a conversation message. usage is attributed to this
// turn (zero for non-assistant messages).
func (w *Writer) RecordMessage(m llm.Message, usage llm.Usage) {
	rec := Record{Type: "message", Message: &m}
	if usage != (llm.Usage{}) {
		u := usage
		rec.Usage = &u
	}
	w.write(rec)
}

// RecordBoundary persists a compaction boundary for observability.
func (w *Writer) RecordBoundary(meta llm.BoundaryMeta) {
	m := meta
	w.write(Record{Type: "boundary", Boundary: &m})
}

// newID returns a random 128-bit hex identifier.
func newID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		// rand should never fail; fall back to a time-based id to stay unique-ish.
		return hex.EncodeToString([]byte(time.Now().UTC().Format(time.RFC3339Nano)))
	}
	return hex.EncodeToString(b[:])
}

// NewSessionID returns a fresh session identifier.
func NewSessionID() string { return newID() }

// NewAgentID returns a fresh subagent identifier.
func NewAgentID() string { return newID() }
