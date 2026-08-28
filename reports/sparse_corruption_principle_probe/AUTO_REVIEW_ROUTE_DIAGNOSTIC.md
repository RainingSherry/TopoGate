# auto-review-loop route diagnostic

Date: 2026-08-18

## Finding

The earlier `review_unavailable_no_score` outcome was a tool-selection/configuration failure,
not a missing repository mount and not a Claude model limitation.

The active Claude-review bridge constructs Claude CLI commands with:

```text
--permission-mode plan
--tools ""
```

`--tools ""` disables all built-in Claude tools, including `Read`, `Glob`, and `Grep`. The
remaining project MCP tool exposed to the reviewer was `mcp__codegraph__*`, but the Claude
session rejected those calls in plan mode with `Cannot call ... while in plan mode`. The session
cwd was nevertheless `/home/luolie/ToPoGate`.

Evidence:

- `/data/luolie/aris_repo/mcp-servers/claude-review/server.py` and the configured
  `/home/luolie/.codex/mcp-servers/claude-review/server.py` are byte-identical.
- `server.py:39` defaults `CLAUDE_REVIEW_TOOLS` to an empty string.
- `server.py:247` unconditionally selects `--permission-mode plan`.
- `server.py:260-261` forwards the empty tool list as `--tools ""`.
- `/home/luolie/.codex/config.toml` did not define `CLAUDE_REVIEW_TOOLS`.
- The failed session trace shows `permissionMode: plan`, repeated rejected
  `mcp__codegraph__codegraph_files/status/search` calls, and no built-in `Read` tool.

## Successful alternatives

The MCP interface already exposes a per-call `tools` override. Two read-only probes succeeded
without changing repository files or leaving plan mode:

1. `tools="Read"` read `reports/sparse_corruption_principle_probe/PROTOCOL.md` and returned the
   exact first heading and table field.
2. `tools="Read,Glob,Grep"` located the protocol, confirmed `LEGAL_GPU_POOL=(1,2,3,4,5,6)`,
   `FORBIDDEN_GPU_IDS=(0,7)`, and found the review state file.

Using the same override, a substantive C0/C1/C2 review read the actual reports, scripts, tests and
JSON artifacts and returned `score=8`, `verdict=ready` for the implementation contract. Its scope
did not include the locked C2 GPU performance matrix.

## Recommended fix

Keep `--permission-mode plan` and enable only read-only built-ins by setting the MCP server
environment to:

```toml
CLAUDE_REVIEW_TOOLS = "Read,Glob,Grep"
```

The auto-review prompt should also tell Claude to prefer those built-ins and not call codegraph in
this route. Do not add `Bash`, `Edit`, or write-capable tools for this review workflow. A per-call
`tools` override is the least invasive immediate workaround; the environment setting is the
persistent fix for ordinary `auto-review-loop` calls, which currently omit the `tools` argument.

## Lower-confidence fallback

For a compact review packet, the bridge can carry the full text directly in the prompt (or via the
`system` argument). This can produce a critique, but it is weaker than direct file reading because
Claude cannot independently verify omitted files, hashes, or raw artifacts. It should be labelled
`review_packet_only`, not equivalent to a repository review.

## Boundaries

No repository performance run was launched. No default user-level configuration was changed in this
diagnostic. The successful score is evidence that the read-only route works; it is not a scientific
performance result and does not unlock C2/C3/adaptive stages.
