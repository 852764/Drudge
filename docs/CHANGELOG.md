# Changelog

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
