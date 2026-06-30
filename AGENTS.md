# Drudge Project Guidance

- Keep the runtime compatible with Python 3.10 and newer.
- Preserve both Chat Completions and Responses API behavior when changing the model loop.
- Keep tool authorization in host-controlled code; model arguments must never override `ToolContext`.
- Never read, print, or commit `.drudge/auth.json` or `.codex/auth.json`.
- Add offline tests for storage migrations, tool protocol changes, CLI options, and context-loading behavior.
- Before handing off a change, run:
  - `python -m unittest discover -s tests -v`
  - `python -m compileall -q agent config.py main.py prompt tools tests`
