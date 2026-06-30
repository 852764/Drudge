# Priority 1-3 Implementation Notes

This document records the second development pass after the initial Drudge/Codex OAuth integration.

## Priority 1: engineering foundation

Implemented:

- Removed tracked Python cache files from git index.
- Added `.gitattributes` to stabilize text/binary handling and reduce line-ending noise.
- Added `drudge doctor` for local diagnostics.
- Added credential hardening checks for `.drudge/auth.json` and `.codex/auth.json` reads.

Useful commands:

```powershell
python main.py doctor
$env:DRUDGE_HOME="F:\Drudge\.drudge\live"
python main.py --codex-oauth doctor
```

`doctor` prints provider, model API, toolsets, workspace, approval mode, storage path, auth status, registered tools, and git status. It does not print tokens.

## Priority 2: agent loop and tool stability

Implemented:

- Added a standard tool result envelope:

```json
{
  "ok": true,
  "content": "...",
  "error": null,
  "metadata": {},
  "blocked": true
}
```

- Kept legacy fields where useful so existing model/tool loops remain compatible.
- Added `approval_mode` under `security`.
- Added `apply_patch` as a first-class file tool for targeted edits.
- Added non-interactive safety policy hooks for file mutation, terminal, and network tools.

Configuration example:

```yaml
security:
  workspace_root: F:\Drudge
  allow_outside_workspace: false
  allow_terminal: true
  allow_network: true
  approval_mode: auto  # auto | on_request | never
```

Current behavior:

- `auto`: allow normal in-workspace operations and block dangerous command markers.
- `never`: block mutating file operations, terminal commands, and network requests.
- `on_request`: low-risk reads run directly; file mutations, network requests, and terminal commands pause for host-side approval. The user can allow once, allow the same tool/risk level for the session, or deny.

The registry enforces approval independently of the model. Calling a medium/high-risk tool directly without an approved host decision returns a blocked result.

## Next-stage priorities 1-2

Implemented:

- Tool risk levels: `low`, `medium`, `high`, and `critical`.
- Interactive approval UI through `--approval-mode on_request`.
- Streaming Chat Completions, Responses API, and Codex OAuth output.
- `Ctrl+C` cancellation without exiting the interactive session.
- Cancellable terminal subprocesses; timeout/cancel attempts to terminate the complete process tree.

See [Approval, streaming, and cancellation](APPROVAL_STREAMING.md) for behavior and integration details.

## Priority 3: context management

Implemented:

- Added deterministic repo map generation.
- Injected repo map into the system prompt.
- Added deterministic conversation compaction helper.
- Added config knobs:

```yaml
agent:
  compression_threshold: 0.8
  compact_keep_recent: 8
  repo_map_enabled: true
  repo_map_max_files: 80
```

The repo map excludes private/generated directories such as `.git`, `.drudge`, `__pycache__`, `.venv`, `node_modules`, `dist`, and `build`.

## Verification

```powershell
python -m unittest discover -s tests -v
python -m compileall -q agent config.py main.py prompt tools tests
python main.py doctor
```

The test suite covers:

- tool result envelope compatibility;
- `apply_patch` tool;
- approval-mode blocking;
- sensitive auth-file blocking;
- repo map private-dir exclusion;
- context compaction;
- CLI `doctor` parsing.
- risk classification and host approval;
- Chat Completions and Responses SSE parsing;
- Codex OAuth delta callbacks;
- terminal subprocess cancellation.
