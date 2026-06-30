# Drudge Codex OAuth

This document describes the experimental ChatGPT/Codex OAuth provider added to Drudge.

## What it does

Drudge can now authenticate with the same ChatGPT Codex entitlement style used by Codex-compatible clients, then call the Codex backend from Drudge's own agent loop.

This is not a wrapper around the Codex CLI or Codex SDK. Drudge owns:

- the CLI entrypoint;
- the agent loop;
- tool execution;
- token storage under Drudge's own auth file;
- HTTP calls to the Codex backend.

It also does not read or reuse `~/.codex/auth.json`.

## When to use this vs API key

Use `--codex-config` when you want OpenAI Platform API keys, relay keys, or an OpenAI-compatible gateway.

Use `--codex-oauth` when you want to test whether a ChatGPT/Codex login can drive Drudge directly without a Platform API key.

Important billing/quota boundary:

- ChatGPT Plus/Codex login is separate from OpenAI Platform API billing.
- An `OPENAI_API_KEY` still uses Platform API billing.
- The OAuth path uses ChatGPT/Codex account entitlement and quota, but it depends on an undocumented Codex backend surface and can change.

## Login

```powershell
drudge auth login
```

On headless machines or if you do not want Drudge to open a browser:

```powershell
drudge auth login --no-browser
```

The command prints a device URL and user code. Open the URL, log in with ChatGPT, enter the code, then return to the terminal.

## Status and logout

```powershell
drudge auth status
drudge auth logout
```

Status intentionally does not print access tokens or refresh tokens.

## Run Drudge with Codex OAuth

```powershell
drudge --codex-oauth -q "Inspect this repo and summarize the architecture"
```

During development you can call the Python entrypoint directly:

```powershell
python main.py --codex-oauth -q "Reply exactly DRUDGE_CODEX_OK"
```

`--models` is not supported for this provider because the ChatGPT Codex backend does not expose the same `/models` API as OpenAI Platform.

## Credential storage

By default Drudge stores credentials at:

```text
~/.drudge/auth.json
```

You can override the directory:

```powershell
$env:DRUDGE_HOME="F:\Drudge\.drudge\live"
drudge auth login --no-browser
```

Treat this file like a password. Do not commit it, paste it into issues, or share it with other tools.

## Limitations

- This provider is experimental.
- Device-code auth must be available for the ChatGPT/Codex account.
- The backend URL, request shape, model allowlist, quota behavior, and headers may change.
- Tokens are stored by Drudge separately from Codex CLI credentials.
- The current store is protected by an in-process lock, not a cross-process file lock.
- If the access token cannot be refreshed, run `drudge auth login` again.

## Implementation references

- OpenAI Codex authentication docs: https://developers.openai.com/codex/auth
- Hermes agent auth implementation: https://github.com/NousResearch/hermes-agent/blob/main/hermes_cli/auth.py
- Hermes agent Codex runtime credential resolver: https://github.com/NousResearch/hermes-agent/blob/main/agent/codex_runtime.py

