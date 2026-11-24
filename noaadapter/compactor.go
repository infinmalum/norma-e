package noaadapter

import (
	"context"

	"github.com/infinmalum/norma-e/compaction"
	"github.com/infinmalum/norma-e/llm"
)

// Compactor adapts a Session to the harness's context-manager interfaces.
//
// It satisfies harness.Compactor and harness.ContextView. Implementing the
// latter is what takes over request construction from llm.MessagesForAPI.
type Compactor struct {
	sess *Session
	// overflow tracks what a prompt-too-long error taught us.
	overflow overflowState
}

// Pre is a no-op on the history.
//
// The built-in compaction rewrites messages here; noa must not. Message
// identity is a content hash over the stored text, and a rewrite would change
// every id derived from it, invalidating every ref the model holds.
func (c *Compactor) Pre(_ context.Context, msgs []llm.Message, lastInputTokens int) []llm.Message {
	if lastInputTokens > 0 {
		c.sess.noteProviderTokens(lastInputTokens)
	}
	return msgs
}

// View builds the request's message array.
func (c *Compactor) View(_ context.Context, msgs []llm.Message) []llm.Message {
	return c.sess.View(msgs)
}

// IsOverflow reports whether err is a context-overflow rejection. The built-in
// compaction already recognises every provider's phrasing, so the detection is
// borrowed rather than duplicated.
func (c *Compactor) IsOverflow(err error) bool {
	if !compaction.IsOverflow(err) {
		return false
	}
	if window := ParseOverflowWindow(err.Error()); window > 0 {
		c.overflow.learn(min(window, c.sess.Config().ModelContextLimit))
	}
	return true
}

// Reactive recovers from a prompt-too-long error.
//
// noa cannot ask the model to compress here — the model is not in the loop at
// this point. What it can do is arm the mechanical valve and VERIFY the result:
// returning true without actually shrinking the view would produce a
// 413 → retry → 413 loop, which is worse than a clean failure. So the view is
// rebuilt and measured, and false is returned if it did not shrink.
func (c *Compactor) Reactive(ctx context.Context, msgs []llm.Message) ([]llm.Message, bool) {
	// Compare against the array the provider actually rejected, not the raw
	// projection. The projection is always larger than the sent view, so
	// measuring against it would show progress on every attempt and never admit
	// that arming changed nothing.
	before := c.sess.sentTokens()

	c.overflow.arm(c.sess.Config().ModelContextLimit)
	floor := c.overflow.armedFloor()
	c.sess.setEmergencyFloor(floor)

	after := estimateMessages(c.View(ctx, msgs))

	shrank := before == 0 || after < before
	fits := floor == 0 || after <= floor
	if !shrank || !fits {
		// Nothing more to give. Returning true here would promise a smaller
		// request that is not smaller, and the loop would retry into the same
		// rejection indefinitely; a clean prompt-too-long is the better outcome.
		c.sess.setEmergencyFloor(0)
		c.overflow.disarm()
		return msgs, false
	}
	return msgs, true
}

// noaMediaTokens is the fixed token cost charged per image/document block in
// estimates: media is size-independent for the model (~2000 tokens), while its
// Text carries only the short stub — charging len(Text) would price it ~0.
const noaMediaTokens = 2000

// estimateMessages sizes a rebuilt request.
func estimateMessages(msgs []llm.Message) int {
	total := 0
	for _, m := range msgs {
		for _, b := range m.Content {
			if b.Type == llm.BlockImage || b.Type == llm.BlockDocument {
				total += noaMediaTokens
				continue
			}
			total += len(b.Text) / 4
			total += len(b.Thinking) / 4
			total += len(b.Input) / 4
			for _, inner := range b.Content {
				if inner.Type == llm.BlockImage || inner.Type == llm.BlockDocument {
					total += noaMediaTokens
					continue
				}
				total += len(inner.Text) / 4
			}
		}
	}
	return total
}
