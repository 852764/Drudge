# MCP Resources / Skills Workflows / Memory / Undo

This pass adds four practical capabilities on top of the existing Drudge runtime.

## 1. MCP resources and prompts

Drudge still supports MCP `tools/list` and `tools/call`, and now also exposes:

- `resources/list`
- `resources/read`
- `prompts/list`
- `prompts/get`

When an MCP server advertises these capabilities during `initialize`, Drudge creates extra callable tools such as:

- `mcp__demo__list_resources`
- `mcp__demo__read_resource`
- `mcp__demo__list_prompts`
- `mcp__demo__get_prompt`

`/mcp` now shows capability flags plus resource/prompt counts.

If an MCP server process exits unexpectedly, Drudge will automatically restart it on the next MCP call unless the server config sets:

```yaml
mcp_servers:
  demo:
    auto_restart: false
```

## 2. Skills workflow MVP

`SKILL.md` front matter now supports:

```yaml
---
name: review
description: Review source changes
references:
  - references/guide.md
scripts:
  run:
    - echo hello
  preflight:
    - python -m unittest tests.test_x -v
---
```

Behavior:

- `references` are loaded from files under the skill directory and injected into the skill render output.
- `scripts` define workflow phases such as `run`, `preflight`, or `postflight`.
- `/skill show <name>` shows workflow and reference metadata.
- `/skill run <name> [phase]` executes the declared phase commands through the existing terminal tool path.

This keeps skills host-controlled while moving them beyond plain prompt injection.

## 3. Durable memory

Drudge now persists long-term memories in SQLite.

Scopes:

- `project`: bound to the current workspace
- `user`: global to the current Drudge database

Commands:

```text
/memory list [project|user]
/memory add <project|user> <content>
/memory pin <id>
/memory unpin <id>
/memory rm <id>
```

Model tools:

- `memory_list`
- `memory_create`
- `memory_update`
- `memory_delete`

At the start of each turn, Drudge deterministically ranks stored memories against the user prompt and injects the top matches into the system prompt.

## 4. File checkpoints and undo

Every successful `write_file`, `patch`, and `apply_patch` operation now records a file revision in SQLite.

Each revision stores:

- path
- operation
- before content
- after content
- compact unified diff summary
- undo state

Commands:

```text
/changes
/undo
```

`/undo` reverts the most recent reversible file change in the active session.

Tool results for file mutations now include:

- `diff_summary`
- `checkpoint_created`

These values also flow into run trace data.

## Verification

```powershell
python -m unittest discover -s tests -v
python -m compileall -q agent config.py main.py prompt tools tests
```
