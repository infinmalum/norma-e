package harness

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/infinmalum/norma-e/llm"
	"github.com/infinmalum/norma-e/permission"
	"github.com/infinmalum/norma-e/tool"
)

// rawInputTool is the optional opt-out from schema validation. Discovered by
// assertion rather than declared on tool.CoreTool, so hosts implementing that
// interface are unaffected.
type rawInputTool interface{ AcceptsRawInput() bool }

func acceptsRawInput(t tool.CoreTool) bool {
	r, ok := t.(rawInputTool)
	return ok && r.AcceptsRawInput()
}

func acceptsRawInputFor(reg *tool.Registry, name string) bool {
	if reg == nil {
		return false
	}
	t, ok := reg.Get(name)
	return ok && acceptsRawInput(t)
}

// execOne preserves the model's tool-call ID, including for a deferred target.
func (l *loop) execOne(ctx context.Context, settling bool, use llm.ContentBlock, emitProgress func(tool.ProgressInfo)) (llm.ContentBlock, []llm.Message) {
	res := l.runTool(ctx, settling, use, emitProgress, false)
	content := res.Content
	if len(content) == 0 {
		content = []llm.ContentBlock{llm.TextBlock("(no output)")}
	}
	return llm.ContentBlock{Type: llm.BlockToolResult, ToolUseID: use.ID, Content: content, IsError: res.IsError}, res.Extra
}

func (l *loop) runTool(ctx context.Context, settling bool, use llm.ContentBlock, emitProgress func(tool.ProgressInfo), deferred bool) tool.Result {
	if err := ctx.Err(); err != nil {
		return tool.Errorf(err.Error())
	}
	if l.in.Tools == nil {
		return tool.Errorf(fmt.Sprintf("Error: unknown tool %q", use.Name))
	}
	t, ok := l.in.Tools.Get(use.Name)
	if !ok {
		return tool.Errorf(fmt.Sprintf("Error: unknown tool %q", use.Name))
	}
	if !deferred && slices.Contains(l.in.DeferredTools, use.Name) {
		return tool.Errorf("Error: invoke deferred tools through " + tool.ExecuteExtraToolName)
	}
	if settling && slices.Contains(l.in.Settlement.DisabledTools, use.Name) {
		return tool.Errorf("Tool call was not permitted during settlement")
	}
	input := bytes.Clone(use.Input)
	safe := t.IsConcurrencySafe(input)
	validate := func(input json.RawMessage) error {
		if !acceptsRawInput(t) {
			if len(input) > 0 && !json.Valid(input) {
				return fmt.Errorf("input is not valid JSON")
			}
			if err := tool.ValidateInput(t.InputSchema(), input); err != nil {
				return err
			}
		}
		if safe && !t.IsConcurrencySafe(input) {
			return fmt.Errorf("rewritten input requires exclusive scheduling")
		}
		return nil
	}
	if err := validate(input); err != nil {
		return tool.Errorf("Error: invalid tool input: " + err.Error())
	}

	mode := l.in.PermissionMode
	if l.in.PermissionModeFunc != nil {
		mode = l.in.PermissionModeFunc()
	}
	pc := permission.Context{
		Mode: mode, WorkingDir: l.in.WorkingDir, Allowed: l.in.Allowed, Disallowed: l.in.Disallowed,
	}
	if settling {
		pc.Disallowed = slices.Concat(pc.Disallowed, l.in.Settlement.DisabledTools)
	}
	authorize := func(input json.RawMessage) permission.Decision {
		return permission.Evaluate(ctx, permission.Request{
			ToolName: use.Name, Input: bytes.Clone(input), Ctx: pc, Ask: l.in.CanUseTool,
			IsReadOnly: t.IsReadOnly(input), ToolDecision: t.CheckPermissions(ctx, bytes.Clone(input), pc),
		})
	}
	dec := authorize(input)
	if dec.Behavior != permission.Allow {
		return tool.Errorf("Tool call was not permitted: " + dec.Message)
	}
	original := input
	if len(dec.UpdatedInput) > 0 {
		input = dec.UpdatedInput
	}
	if l.in.Hooks != nil {
		// A hook receives a copy so an in-place edit must be returned explicitly.
		if block, msg, updated := l.in.Hooks.PreToolUse(ctx, use.Name, bytes.Clone(input)); block {
			return tool.Errorf("Blocked by hook: " + msg)
		} else if len(updated) > 0 {
			input = updated
		}
	}
	if !bytes.Equal(original, input) {
		if err := validate(input); err != nil {
			return tool.Errorf("Error: invalid rewritten tool input: " + err.Error())
		}
		dec := authorize(input)
		if dec.Behavior != permission.Allow || (len(dec.UpdatedInput) > 0 && !bytes.Equal(input, dec.UpdatedInput)) {
			return tool.Errorf("Tool call was not permitted after rewriting input: " + dec.Message)
		}
	}
	if err := ctx.Err(); err != nil {
		return tool.Errorf(err.Error())
	}

	tc := &tool.ToolContext{
		WorkingDir: l.in.WorkingDir, AgentID: l.in.AgentID, ToolUseID: use.ID,
		Emit: emitProgress, OutputDir: l.in.ToolOutputDir, MaxOutputChars: l.in.MaxToolOutputChars,
		Tasks: l.in.Tasks, Env: l.in.BashEnv, DeferredTools: slices.Clone(l.in.DeferredTools),
	}
	if use.Name == tool.ExecuteExtraToolName {
		tc.DispatchDeferred = func(name string, input json.RawMessage) tool.Result {
			target := llm.ContentBlock{ID: use.ID, Name: name, Input: input}
			return l.runTool(ctx, settling, target, emitProgress, true)
		}
	}
	res, err := t.Call(permission.WithParentPolicy(ctx, pc, l.in.CanUseTool), input, tc)
	if err != nil {
		res = tool.Errorf("Error: " + err.Error())
	}
	res = capOutput(tc, res)

	if l.in.Hooks != nil {
		raw, _ := json.Marshal(res.Flatten())
		l.in.Hooks.PostToolUse(ctx, use.Name, input, raw, res.IsError)
	}
	return res
}

func capOutput(tc *tool.ToolContext, res tool.Result) tool.Result {
	return tool.CapResult(tc, res)
}
