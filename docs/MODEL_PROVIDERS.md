# 多模型与 Provider 配置

Drudge 将 provider 默认配置、认证和 API 协议适配分开处理。模型名称由用户明确指定，不根据名称猜测服务商或自动发送到其他服务商。

## 内置预设

在交互界面输入 `/providers` 查看预设，不发起网络请求；`/models` 查询当前服务商的模型列表。

| `model.provider` | 默认协议 | 默认密钥环境变量 |
| --- | --- | --- |
| `openai` | Responses | `OPENAI_API_KEY` |
| `anthropic` | Anthropic Messages | `ANTHROPIC_API_KEY` |
| `deepseek` | Chat Completions | `DEEPSEEK_API_KEY` |
| `openrouter` | Chat Completions | `OPENROUTER_API_KEY` |
| `gemini` | Google 的 OpenAI 兼容接口 | `GEMINI_API_KEY` |
| `dashscope` | DashScope OpenAI 兼容接口 | `DASHSCOPE_API_KEY` |
| `ollama` | 本地 Chat Completions，端口 11434 | 默认无密钥 |
| `lmstudio` | 本地 Chat Completions，端口 1234 | 默认无密钥 |

预设只给出地址、协议及部分请求默认值，不代表该服务商的每个模型都支持工具调用。DashScope 默认使用中国区地址；其他区域可显式覆盖 `base_url`。

## 最小配置

例如保存为 `config.anthropic.yaml`，将 `YOUR_MODEL_ID` 替换为账户实际可用的模型 ID：

```yaml
model:
  provider: anthropic
  name: YOUR_MODEL_ID
  api_key_env: ANTHROPIC_API_KEY
```

在终端环境设置对应密钥后启动：

```powershell
python main.py -c config.anthropic.yaml
```

本地模型示例：

```yaml
model:
  provider: ollama
  name: YOUR_INSTALLED_MODEL_ID
```

预设的 `base_url`、`api`、`api_key_env` 都可以显式覆盖。自定义网关继续使用：

```yaml
model:
  provider: custom
  name: YOUR_MODEL_ID
  base_url: http://127.0.0.1:8318/v1
  api: responses
  api_key_env: CPA_API_KEY
  reasoning_effort: high
  disable_response_storage: true
```

现有 CodeGo Codex TOML 导入及 `--codex-oauth` 路径保留；本轮没有修改个人配置、密钥或认证文件。默认本地 `custom` 配置仍使用 Responses。`custom` 切换到其他 provider 时，服务商默认值会重新应用；请同时确认模型 ID。

## 参数兼容

- `api` 支持 `chat`、`responses`、`anthropic`，以及已有的 `auto` / Codex OAuth 路径。`chat_completions`、`openai_chat`、`openai_responses`、`anthropic_messages`、`messages` 可作为对应协议的别名。
- `temperature: null` 表示完全不发送该字段，适用于不接受该参数的模型。OpenAI、Anthropic 预设默认省略；显式数值仍会发送。
- `max_tokens` 是 Drudge 的输出预算。Chat 使用 `chat_token_limit: max_tokens` 或 `max_completion_tokens` 指定字段名；Responses 使用 `max_output_tokens`；Anthropic 使用 `max_tokens`。
- `reasoning_effort` 在 Chat 中使用同名字段，在 Responses 中转换为 `reasoning.effort`；仅在模型支持时配置。Anthropic 的高级 thinking 参数尚未提供配置入口。
- `stream_usage: true` 为 Chat 请求增加 `stream_options.include_usage`；默认关闭，避免兼容服务因不识别该选项而拒绝请求。
- `base_url` 推荐填写 API 根路径，也可包含完整的 `/chat/completions`、`/responses`、`/messages` 后缀。认证信息、查询参数分别使用配置字段，不放进 URL。

Header 和查询参数示例：

```yaml
model:
  provider: custom
  name: YOUR_MODEL_ID
  base_url: https://gateway.example/v1
  api: chat
  api_key_env: GATEWAY_API_KEY
  temperature: null
  chat_token_limit: max_completion_tokens
  env_headers:
    X-Tenant-Token: GATEWAY_TENANT_TOKEN
  query_params:
    api-version: YOUR_API_VERSION
```

`api_key_env` 指向的环境变量存在时优先使用其值；不存在时可使用同一配置显式提供的 `api_key`。更换密钥环境变量名不会沿用先前配置的密钥。缺少 `env_headers` 引用的环境变量会明确报错。环境 Header 按大小写不敏感的名称覆盖静态 Header。

## 切换与历史兼容

- 更换 provider 时清除旧 provider 的地址、密钥、Header、查询参数和 OAuth 标记，再应用新预设及显式配置。
- `utility_model` 可继续继承同一服务商配置；切换 provider 或 URL origin 时清除继承的认证信息，需配置自己的密钥来源。
- 原生 Anthropic 的工具结果、系统消息和工具 schema 转换在适配器内完成；宿主仍处理工具权限和审批。
- Anthropic thinking/signature 块及 Chat `reasoning_content` 可随工具事务持久化，恢复会话或创建分支时保留。专用数据只回传给匹配的协议、模型和 API 根地址，不作为可见回答输出，也不进入跨模型摘要。
- `api: auto` 仅在 Chat 路由实际返回 HTTP 404 时尝试同一根地址的 Responses。401/403、一般参数错误及包含“404”字样的流内错误不会触发协议切换。
- 流式输出已经出现内容、工具片段或推理片段后，中断不会自动重放请求。

## 联网能力探测

以下命令会实际请求当前配置的服务商，可能产生 token 用量；不执行探测返回的工具调用：

```powershell
python main.py -c config.anthropic.yaml doctor --probe-model --probe-json
```

可使用 `--probe-model YOUR_MODEL_ID` 或 `--no-probe-streaming`。Anthropic 配置只探测原生 Messages；OpenAI 兼容配置保留 Chat/Responses 探测矩阵。Codex OAuth 使用已有独立路径，不参加该通用探测。

## 验证范围与后续边界

本轮使用离线 MockTransport 和真实本地工具回路验证配置、认证、请求参数、JSON/SSE、工具结果、恢复/分支、审批及错误恢复；不等同于各服务商生产端点的联网认证。

Gemini 当前接入兼容 API，而非原生 `generateContent`。Anthropic 当前聚焦文本与客户端函数工具；图像、服务端工具、高级 thinking 配置和模型列表分页尚待扩展。模型 ID、上下文长度及支持参数应按实际服务商设置，本轮没有引入远端模型目录或自动跨服务商降级。

## 设计参考

只借鉴分层设计，未复制上游实现：

- [Hermes provider 配置](https://github.com/NousResearch/hermes-agent/blob/57775e9e161087dbe55e096c038f512233c03381/hermes_cli/providers.py)：统一 provider 元数据、运行时解析；同版本 `runtime_provider.py` 和 `agent/anthropic_adapter.py` 用于对照认证隔离及原生协议适配。
- [CC Switch adapter](https://github.com/farion1231/cc-switch/blob/b4a079430ce85a604e10d97d4b7530774e00e112/src-tauri/src/proxy/providers/adapter.rs)：将认证、URL 和协议转换分离；同版本 `claude.rs`、`transform.rs` 用于对照消息和工具转换。
