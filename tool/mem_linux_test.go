//go:build linux

package tool

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// holdOrphan backgrounds a memory holder from a subshell that exits at once:
// the holder is reparented to pid 1 and so leaves the task's ppid tree, but
// stays in its process group — the set treeKill actually kills. The trailing
// ":" in each branch stops the shell from exec-ing the final sleep in place,
// which would drop the allocation the test depends on.
const holdOrphan = `( { x=a; while [ ${#x} -lt 200000000 ]; do x="$x$x"; done; sleep 30; :; } & ) ; sleep 30; :`

// A leaker whose parent exits must still be caught. Measuring the ppid tree
// instead of the process group loses it the moment init adopts it: the walk
// reports the few MiB of the surviving shell while the orphan holds hundreds,
// and the guard never fires on the shape a shell script produces every time it
// starts its real work with "cmd &".
func TestMemGuardCatchesOrphanedDescendant(t *testing.T) {
	t.Setenv("AGENT_CORE_TASK_MEM_CAP", "64m")
	m := newTestManager(t)
	task, err := m.Spawn(SpawnSpec{Command: holdOrphan})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	m.Background(task.ID)
	info := waitTerminalIn(t, m, task.ID, 30*time.Second)
	if info.Status != TaskFailed || info.Reason != terminateMemory {
		t.Fatalf("status=%s reason=%q, want failed/memory — the orphan left the ppid tree but not the process group", info.Status, info.Reason)
	}
}

// The inverse: a parent that forks workers shares its pages, and summing RSS
// over the group multiplies them by the number of holders. Killing on that sum
// would take down ordinary multiprocessing work and tell the model to rewrite a
// command that never leaked, so a group over the cheap bound must be confirmed
// against Pss before it dies.
func TestMemGuardSparesSharedPages(t *testing.T) {
	if _, ok := procPSS(os.Getpid()); !ok {
		t.Skip("PSS is unavailable on this host")
	}

	const mappingSize = 32 << 20
	dir := t.TempDir()
	// Back the mapping with real pages even if the temporary directory is on tmpfs.
	if err := os.WriteFile(filepath.Join(dir, "shared"), make([]byte, mappingSize), 0o600); err != nil {
		t.Fatal(err)
	}
	testBinary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}

	t.Setenv("AGENT_CORE_TASK_MEM_CAP", "80m")
	limit := memCapBytes()
	m := newTestManager(t)
	task, err := m.Spawn(SpawnSpec{
		Command:    `for i in 1 2 3 4; do "$MEM_GUARD_TEST_BINARY" -test.run='^TestMemGuardSharedMappingHelper$' & done; wait`,
		WorkingDir: dir,
		Env:        []string{"MEM_GUARD_TEST_BINARY=" + testBinary},
	})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	defer m.Kill(task.ID)
	m.Background(task.ID)

	// Each worker signals after faulting in the same fixed-size mapping. The
	// fixture can reach the desired RSS/PSS split without growing private
	// Bash strings whose intermediate allocations can cross the cap.
	deadline := time.Now().Add(10 * time.Second)
	for {
		out, _ := m.Output(task.ID, 0)
		info, _ := m.Get(task.ID)
		if info.Status.Terminal() || time.Now().After(deadline) {
			t.Fatalf("shared mapping workers failed to start: %s/%s, output=%q", info.Status, info.Reason, out)
		}
		if strings.Count(out, "mapping ready") == 4 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	pgid := task.cmd.Process.Pid
	rss := pgrpRSS([]int{pgid})[pgid]
	pss, ok := pgrpPSS(pgid)
	if !ok || rss <= limit || pss >= limit {
		t.Fatalf("expected RSS > cap > PSS, got rss=%dMiB pss=%dMiB cap=%dMiB measured=%v",
			rss>>20, pss>>20, limit>>20, ok)
	}

	m.memGuardPass()
	if info, _ := m.Get(task.ID); info.Status != TaskRunning {
		t.Fatalf("shared mapping was terminated: %s/%s, rss=%dMiB pss=%dMiB",
			info.Status, info.Reason, rss>>20, pss>>20)
	}
}

// TestMemGuardSharedMappingHelper runs in each worker spawned by the manager.
func TestMemGuardSharedMappingHelper(t *testing.T) {
	if os.Getenv("MEM_GUARD_TEST_BINARY") == "" {
		return
	}
	f, err := os.Open("shared")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	mapping, err := syscall.Mmap(int(f.Fd()), 0, int(info.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(mapping)
	var sum byte
	for offset, pageSize := 0, os.Getpagesize(); offset < len(mapping); offset += pageSize {
		sum |= mapping[offset]
	}
	if sum != 0 {
		t.Fatalf("unexpected mapping contents: %d", sum)
	}
	if _, err := os.Stdout.WriteString("mapping ready\n"); err != nil {
		t.Fatal(err)
	}
	time.Sleep(30 * time.Second)
}

// pgrpRSS must see every member of the group, including one the task's own
// parent-child links no longer reach.
func TestPgrpRSSCoversWholeGroup(t *testing.T) {
	m := newTestManager(t)
	task, err := m.Spawn(SpawnSpec{Command: holdOrphan})
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	defer m.Kill(task.ID)
	pgid := task.cmd.Process.Pid

	var measured, want int64
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		measured = pgrpRSS([]int{pgid})[pgid]
		want = groupRSSViaProc(t, pgid)
		if want > 100<<20 {
			break
		}
		time.Sleep(200 * time.Millisecond)
	}
	if want <= 100<<20 {
		t.Skipf("holder never became resident on this host (%d MiB)", want>>20)
	}
	// Both read /proc a moment apart, so allow drift; the failure this guards
	// against is an order-of-magnitude miss, not a few MiB.
	if diff := want - measured; diff > 32<<20 || diff < -(32<<20) {
		t.Fatalf("pgrpRSS = %d MiB, independent group sum = %d MiB", measured>>20, want>>20)
	}
}

// groupRSSViaProc sums VmRSS over the process group, reading /proc/<pid>/status
// rather than the stat field pgrpRSS parses, so the two do not share a bug.
func groupRSSViaProc(t *testing.T, pgid int) int64 {
	t.Helper()
	entries, err := os.ReadDir("/proc")
	if err != nil {
		t.Fatalf("read /proc: %v", err)
	}
	var total int64
	for _, e := range entries {
		pid, err := strconv.Atoi(e.Name())
		if err != nil {
			continue
		}
		g, err := syscall.Getpgid(pid)
		if err != nil || g != pgid {
			continue
		}
		b, err := os.ReadFile(procFile(pid, "status"))
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(b), "\n") {
			if !strings.HasPrefix(line, "VmRSS:") {
				continue
			}
			f := strings.Fields(line)
			if len(f) >= 2 {
				if kb, err := strconv.ParseInt(f[1], 10, 64); err == nil {
					total += kb << 10
				}
			}
			break
		}
	}
	return total
}

// The derived default must be a usable cap, not a constant and not something
// that would kill ordinary work on the machine running the tests.
func TestDefaultMemCapFollowsHostBudget(t *testing.T) {
	budget, ok := hostMemBudget()
	if !ok {
		t.Skip("host memory budget unreadable")
	}
	if budget < minMemCap {
		t.Skipf("host budget %s is below the floor", humanBytes(budget))
	}
	got := defaultMemCap()
	if got > budget {
		t.Fatalf("cap %s exceeds the budget %s it is meant to protect", humanBytes(got), humanBytes(budget))
	}
	if got < minMemCap {
		t.Fatalf("cap %s is below the floor %s", humanBytes(got), humanBytes(minMemCap))
	}
}
