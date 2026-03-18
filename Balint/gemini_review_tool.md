# Gemini Review Tool

This repo includes a local MCP server that lets Codex call Gemini CLI as a second-opinion reviewer for the current git diff inside [Balint](C:\Users\balin\PycharmProjects\NQS\Balint).

## Required setup

- Install Gemini CLI so the `gemini` command is available on `PATH`
- Sign in with the Google account that has your Gemini Pro subscription

The server uses Gemini CLI headless mode, so it reuses the CLI's cached login instead of requiring an API key.

Optional:

- `GEMINI_CLI_COMMAND`: defaults to `gemini`
- `GEMINI_MODEL`: passed through as `--model ...` when set
- `GEMINI_REVIEW_MAX_INPUT_CHARS`: defaults to `120000`
- `GEMINI_REVIEW_UNTRACKED_FILE_CHAR_LIMIT`: defaults to `24000`
- `GEMINI_REVIEW_DEBUG=1`: enables debug logging
- `GEMINI_REVIEW_LOG_MAX_BYTES`: defaults to `1000000`

## Register in Codex

From this workspace, register the MCP server as a local command:

```powershell
codex mcp add gemini-review -- python C:\Users\balin\PycharmProjects\NQS\Balint\gemini_review_server.py
```

If you want a fixed model override:

```powershell
codex mcp add gemini-review --env GEMINI_MODEL=gemini-2.5-flash -- python C:\Users\balin\PycharmProjects\NQS\Balint\gemini_review_server.py
```

If Gemini CLI is not on `PATH`, register the full CLI path too:

```powershell
codex mcp add gemini-review --env GEMINI_CLI_COMMAND=C:\Users\balin\AppData\Roaming\npm\gemini.cmd -- python C:\Users\balin\PycharmProjects\NQS\Balint\gemini_review_server.py
```

## Expected workflow

- Ask Codex to review changes normally
- Codex performs its own review first
- Codex calls `gemini_review_diff` for a second opinion
- If Gemini is unavailable, Codex should continue with its own review and note the tool failure briefly

## Tool inputs

`gemini_review_diff` can still be called with no arguments to review the local uncommitted diff discovered by the server.

When Codex already has better local context, pass it directly:

- `task_scope`: what the review should focus on
- `baseline_context`: relevant baseline behavior, profiler output, or measurements
- `uncommitted_diff`: the explicit diff to review instead of asking the server to rediscover it
- `critical_review_findings`: Codex's own critical findings, as a string or list of strings
- `review_focus`, `max_input_chars`, and `path_filters` still work as before

Preferred call pattern when Codex already did its own review:

1. Pass the task scope.
2. Pass the relevant baseline or profiler context.
3. Pass the exact uncommitted diff.
4. Pass Codex's own critical review findings.

Example payload shape:

```json
{
  "task_scope": "Review only the arithmetic helper change.",
  "baseline_context": "This path is hot in profiling and runs 10M times per step.",
  "uncommitted_diff": "diff --git a/foo.py b/foo.py\n...",
  "critical_review_findings": [
    "Potential arithmetic regression in foo.py",
    "Double-check subtraction vs addition"
  ]
}
```

## Common failures

- Gemini CLI missing from `PATH`: the tool returns `status: "unavailable"`
- Gemini CLI not logged in: the tool returns `status: "unavailable"`
- No local changes: the tool returns `status: "no_changes"`
- Malformed Gemini output: the tool returns `status: "invalid_response"`
- Large diffs: the review payload is truncated and `meta.truncated` is set to `true`

## Debugging

- Set `GEMINI_REVIEW_DEBUG=1` to write helper timing/debug logs
- The log defaults to `%TEMP%\\gemini_review_mcp.log`
- Set `GEMINI_REVIEW_LOG_MAX_BYTES` to control the automatic log truncation threshold
