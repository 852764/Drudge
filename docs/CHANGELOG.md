# Changelog

## Unreleased

- Tighten release edge cases: reset authentication on same-ID Codex provider re-import, normalize native probe routing, require actual text/stream evidence in probes, and prevent Responses retries after tool/reasoning fragments while redacting stream errors.
- Add centralized provider presets and native Anthropic Messages JSON/SSE transport, preserving Chat/Responses/Codex OAuth paths. Scope persisted provider history to matching model endpoints, isolate auxiliary credentials, make generation options configurable, and add `/providers` plus offline protocol/tool-loop regressions.
- Document a pinned project-local installation of five official Codex skills and add shared Drudge execution notes for imported workflows, with offline activation/resume and host-permission regression tests. Skill resources remain locally installed rather than bundled in releases.
- Add an automatic slash-command menu with Chinese descriptions, prefix filtering, arrow-key navigation, Tab/Enter fill-only confirmation and static subcommand completion. Help and completion share one command catalog; non-interactive terminals retain simple input.
- Replace typed approval answers with an arrow-key menu: per-call/session/deny choices, default deny, scrollable details, Ctrl+C cancellation and paused status rendering. Session grants remain scoped to the same tool and risk level.
- Add an opt-in JSReverser-MCP browser preset and local YAML setup/probe helper, reusing host approvals and both model APIs.
- Bound large MCP messages, serialize stdio requests, enforce whole-request deadlines, and clean up owned process trees.
- Add MCP tool allowlists, optional environment isolation and host network gating; existing generic MCP defaults remain compatible.
- Document and avoid the tested upstream collector/second-browser issue in the default 26-tool preset; DOM inspection uses same-page script evaluation. Inline screenshot-to-model vision remains pending.

## 0.2.0b3 — Beta

This entry summarizes cumulative Beta capabilities, including earlier Beta releases
and subsequent fixes on the 0.2.0b3 development line; not every feature was introduced in b3.

### Available capabilities

- Responses API tool-call normalization alongside Chat Completions.
- Durable context checkpoints, deterministic/LLM compaction, interrupted-tool recovery, and session forks.
- Persistent plans, acceptance records, output receipts/pagination, batch file reads, and repository search completeness metadata.
- Host-controlled approvals, immutable `ToolContext`, Windows descendant cleanup, and offline release verification.

### Improved

- Shared transient-error retry policy for the compatible Chat Completions and Responses clients, with bounded numeric `Retry-After` handling.
- Heuristic context estimation and one-shot structured context-overflow recovery; estimates do not guarantee provider acceptance.
- Cross-platform display/canonical path separation, including macOS `/var` aliases and sibling-prefix containment checks.
- Atomic guarded edits, conflict-aware undo, bounded tool output, and explicit query exit states.
- CI diagnostics now persist unittest logs and surface failing test excerpts as annotations.

### Verification

- Python 3.10 and 3.13 CI matrix on Ubuntu and Windows, plus Python 3.13 on macOS.
- Offline release gate covers tests, compile checks, archives, installation into a new virtual environment, installed smoke tests, and CLI help/version. Runtime dependencies are reused from the host; this is not a fully isolated dependency-resolution test.

### Known limits

- This remains a Beta for human-reviewed local development workflows.
- Real provider latency, billing, account limits, long-running tasks, and broad repository benchmarks require separate live acceptance.
- Terminal approval is host policy, not an operating-system sandbox.
