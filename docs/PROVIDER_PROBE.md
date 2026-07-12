# Provider Capability Probe

Drudge can verify whether a model is merely listed by a provider or is actually usable through each supported protocol.

## Usage

Probe the configured model:

```powershell
python main.py doctor --probe-model
```

Probe a specific model:

```powershell
python main.py doctor --probe-model gpt-5.5
```

Machine-readable output:

```powershell
python main.py doctor --probe-model gpt-5.5 --probe-json
```

Skip streaming checks:

```powershell
python main.py doctor --probe-model gpt-5.5 --no-probe-streaming
```

## Capability Matrix

The probe checks:

- whether `/models` lists the exact model ID;
- basic `/chat/completions` requests;
- Chat Completions tool schemas;
- Chat Completions streaming;
- basic `/responses` requests;
- Responses API tool schemas;
- Responses API streaming.

A listed model can still fail every inference endpoint because model listing, account permissions, and endpoint routing are separate provider concerns.

The tool probe only sends a harmless `probe_echo` schema. It does not execute a local Drudge tool. Probe output never includes API keys or authorization headers.

## Cost

A full probe performs one model-list request and up to six small inference requests with a maximum output size of 64 tokens. Use `--no-probe-streaming` to reduce this to four inference requests.

## Interpretation

- `chat.basic=yes`, `chat.tools=no`: use the model for plain Chat Completions but choose another model for agent tools.
- `responses.basic=yes`, `chat.basic=no`: configure `model.api: responses`.
- both basic probes fail while the model is listed: check account permissions or provider-side routing.
- streaming fails while basic succeeds: disable streaming for that provider or add a provider-specific SSE adapter.

