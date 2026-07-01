# LLM 上下文摘要压缩

Drudge 的上下文压缩默认改为调用当前会话使用的 LLM 生成结构化摘要，不再把规则截断结果作为正常压缩内容。

## 双模型配置

正常回答和工具循环使用顶层 model；上下文摘要使用可选的 utility_model。这个客户端也可以继续承接后续低成本后台任务。没有配置 utility_model 时会直接复用主模型，旧配置无需修改。

同一供应商使用两个模型时，只需覆盖辅助模型名称，其余 URL、API 类型和认证配置会从主模型继承：

```yaml
model:
  name: strong-model
  base_url: https://api.example.com/v1
  api_key: your-key
  api: responses

utility_model:
  name: cheap-model
  temperature: 0.1
  max_tokens: 2048
```

如果主模型使用 Codex OAuth，而辅助模型走另一个 OpenAI-compatible 中转站，需要覆盖完整供应商配置：

```powershell
$env:UTILITY_API_KEY="your-relay-key"
```

```yaml
utility_model:
  provider: openai-compatible
  name: cheap-model
  base_url: https://relay.example.com/v1
  api: chat
  api_key_env: UTILITY_API_KEY
  temperature: 0.1
  max_tokens: 2048
```

此时可以继续用 python main.py --codex-oauth 启动：正常任务走 Codex OAuth，压缩摘要走中转站。api_key_env 会在运行时读取环境变量，密钥不需要写入配置文件。

## 行为

触发条件由以下配置控制：

```yaml
agent:
  compression_threshold: 0.80
  compact_keep_recent: 8
  context_summary_mode: llm
  context_summary_fallback: true
```

- 估算上下文超过模型窗口的 80% 时，Agent 自动压缩。
- /compact 可以手动触发同一套流程。
- 系统消息和最近 8 条非系统消息保留原文。
- 如果保留区从工具结果开始，会向前扩展到对应 assistant tool call，避免拆断工具事务。
- 更早的消息会以独立请求交给当前 LLM，摘要请求不携带工具定义。
- 摘要要求保留用户约束、工程决策、文件状态、测试结果、错误、未完成事项和精确路径等继续工作所需的信息。
- provider_items、加密 reasoning 等供应商私有状态不会进入摘要转录。
- 摘要中的 think 标签内容会被过滤，不会重新写入会话上下文。

压缩会消耗一次额外模型调用，其 token 会计入当前进程的 token 统计。/compact 会输出实际采用的模式和摘要调用 token：

```text
Context compacted: 24 -> 10 messages, ~18000 -> ~4200 tokens (mode=llm, summary_tokens=1350)
```

## 失败降级

默认 context_summary_fallback: true。摘要模型不可用、返回空文本或只返回 reasoning 时，Drudge 会退回原有的确定性摘要，保证当前会话仍能继续，并在 /compact 输出失败原因。

如果明确希望摘要失败就终止本轮，可以设置：

```yaml
agent:
  context_summary_fallback: false
```

也可以临时禁用 LLM 摘要：

```yaml
agent:
  context_summary_mode: deterministic
```

## 验证

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
python -m unittest tests.test_context_manager tests.test_agent_loop -v
python -m unittest discover -s tests -v
```
