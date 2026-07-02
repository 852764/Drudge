# 模型请求重试与错误恢复

## Codex OAuth

Codex OAuth 请求默认最多尝试 3 次：

```yaml
model:
  timeout: 300
  max_retries: 3
```

以下情况可以自动重试：

- 连接建立前或尚未收到输出时的 httpx transport error；
- HTTP 429；
- HTTP 500、502、503、504；
- 401 会先刷新 OAuth token，再重新请求。

重试使用指数退避。取消操作、明确的模型失败和非瞬时 HTTP 错误不会重试。

## 部分输出保护

如果已经收到文本 delta 或工具调用，再发生网络中断，Drudge 不会自动重试。原因是重复请求可能导致：

- 重复打印文本；
- 重复生成工具调用；
- 对有副作用的操作产生歧义。

此时错误会明确包含 partial stream/transport 信息，会话和已完成工具结果仍保存在 SQLite，可以直接继续当前会话。

## 空异常

部分网络异常的 str(error) 是空字符串。Drudge 会回退到异常类型，因此不会再只显示：

```text
LLM call failed (turn 5):
```

而会显示类似：

```text
LLM call failed (turn 5): CodexTransportError: Codex transport failed: ReadTimeout
```

Trace 中也保存格式化后的异常类型。

## 验证

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
python -m unittest tests.test_codex_client tests.test_agent_loop -v
python -m unittest discover -s tests -v
```
