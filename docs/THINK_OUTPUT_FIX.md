# `<think>` 刷屏与无最终答案防护

部分 OpenAI-compatible 中转站会把推理模型的内部思考放进普通 `content`：

```text
<think>...</think>
最终答案
```

长会话或输出空间不足时，模型可能只生成 `<think>` 内容，直到达到输出上限，完全没有最终答案。旧版 Drudge 会把这些 delta 原样打印，因此表现为持续刷 `<think>`，最后无回答。

现在 Drudge 会：

1. 在流式增量中识别跨 chunk 的 `<think>` 标签；
2. 隐藏 think 块，只向 CLI 输出最终可见文本；
3. 清理写入 SQLite 和 Responses `provider_items` 的可见推理标签；
4. 检测只有 think、没有最终答案的响应；
5. 自动压缩旧上下文，并追加“仅输出最终答案”的系统要求后重试；
6. 对重复嵌套标签或超长纯推理流提前停止，避免持续刷屏；
7. 重试仍失败时返回明确错误，不再静默结束。

手动压缩长会话：

```text
/compact
```

相关配置：

```yaml
display:
  hide_reasoning_tags: true

agent:
  reasoning_tag_max_chars: 12000
  reasoning_recovery_attempts: 1
```

如果所用模型必须展示原始 think 内容，可以设置 `hide_reasoning_tags: false`，此时 Drudge 不过滤也不触发 think-only 恢复。
