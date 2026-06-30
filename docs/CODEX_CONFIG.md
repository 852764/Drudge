# 在 Drudge 中使用 Codex 配置

Drudge 可以显式读取 Codex 的用户级 `config.toml`：

```powershell
drudge --codex-config
```

默认路径是 `$CODEX_HOME/config.toml`；未设置 `CODEX_HOME` 时使用 `~/.codex/config.toml`。也可以传入其他路径：

```powershell
drudge --codex-config C:\path\to\config.toml
```

## 支持的字段

Drudge 只读取模型调用需要的安全子集：

- `model`
- `model_provider`
- `openai_base_url`
- `profile` 和当前 `[profiles.<name>]`
- `[model_providers.<name>].base_url`
- `env_key`
- `wire_api = "responses"`
- `http_headers`
- `env_http_headers`
- `query_params`
- `request_max_retries`
- `model_context_window`

Drudge 的 `--codex-config` 不读取 Codex 的 `auth.json`，也不会复用 Codex CLI 的 ChatGPT 登录 Token。Codex command-backed provider auth 暂不执行；请改用 `env_key` 或 `env_http_headers`。

## 中转站示例

`~/.codex/config.toml`：

```toml
model = "gpt-5.5"
model_provider = "relay"

[model_providers.relay]
name = "Relay"
base_url = "https://relay.example.com/v1"
env_key = "RELAY_API_KEY"
wire_api = "responses"
```

PowerShell：

```powershell
$env:RELAY_API_KEY="中转站提供的 API Key"
drudge --codex-config
```

中转站必须支持 Responses API，包括函数调用的 `call_id`、`function_call_output` 和流式协议。Codex 当前自定义 provider 的 wire API 只支持 `responses`。参考 OpenAI 官方 [Custom model providers](https://developers.openai.com/codex/config-advanced#custom-model-providers)。

## 配置优先级

从低到高：

1. Drudge 默认值和环境变量
2. `--codex-config` 指定的 Codex TOML
3. `-c/--config` 指定的 Drudge YAML
4. `-m`、`--toolsets` 等 CLI 参数

因此可以复用 Codex provider，同时用 Drudge YAML 覆盖温度、工具集或安全策略：

```powershell
drudge --codex-config -c drudge.local.yaml
```

## 认证限制

如果 Codex 当前通过 ChatGPT 浏览器登录，而配置里没有 `env_key`，Drudge 只能读取模型和 provider，不能获得调用凭据。请使用 OpenAI API Key 或中转站单独提供的 API Key。OpenAI 官方也区分 ChatGPT 登录与 API Key 登录，参见 [Codex authentication](https://developers.openai.com/codex/auth)。

## Drudge Codex OAuth

`--codex-config` 和 `--codex-oauth` 是两条不同路径：

- `--codex-config` 读取 Codex TOML 的安全子集，仍然需要 API Key、中转站 Key 或兼容的 header/env 凭据。
- `--codex-oauth` 会执行一次 Drudge 自己的 ChatGPT/Codex device login，并把凭据写入 Drudge 自己的 auth store。
- Drudge 仍然不会读取或复用 `~/.codex/auth.json`。

登录和使用方式见 [Drudge Codex OAuth](CODEX_OAUTH.md)。
