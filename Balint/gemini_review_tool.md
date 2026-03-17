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

## Register in Codex

From this workspace, register the MCP server as a local command:

```powershell
codex mcp add gemini-review -- python -m codex_gemini_review.server
```

If you want a fixed model override:

```powershell
codex mcp add gemini-review --env GEMINI_MODEL=gemini-2.5-flash -- python -m codex_gemini_review.server
```

## Expected workflow

- Ask Codex to review changes normally
- Codex performs its own review first
- Codex calls `gemini_review_diff` for a second opinion
- If Gemini is unavailable, Codex should continue with its own review and note the tool failure briefly

## Common failures

- Gemini CLI missing from `PATH`: the tool returns `status: "unavailable"`
- Gemini CLI not logged in: the tool returns `status: "unavailable"`
- No local changes: the tool returns `status: "no_changes"`
- Malformed Gemini output: the tool returns `status: "invalid_response"`
- Large diffs: the review payload is truncated and `meta.truncated` is set to `true`
