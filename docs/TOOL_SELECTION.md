# 动态工具选择

Drudge 默认不会无条件调用辅助模型。只有工具数量或完整 schema 体积达到阈值时，才执行工具预选。

## 配置

```yaml
tool_selection:
  enabled: true
  min_tools: 16
  min_schema_tokens: 3000
  max_selected: 12
  search_limit: 5
  sticky_recent: 4
  always_include: []
```

触发规则：

- 工具数达到 min_tools；或者
- 完整工具 schema 估算达到 min_schema_tokens。

未达到阈值时，主模型收到全部业务工具，不调用选择模型，也不附带 tool_search。

## 选择流程

达到阈值后，Drudge 将精简工具目录发送给 utility_model。目录只包含：

- 工具名称；
- 简短描述；
- category；
- risk。

不会把完整参数 schema 发送给选择模型。选择上下文还包括当前请求、少量最近对话、会话摘要、Active Skill 描述和最近实际调用的工具。

辅助模型必须返回：

```json
{"tools":["read_file","search_files"],"reason":"需要检查代码"}
```

主模型最终收到：

- 选择出的完整工具 schema；
- tool_search 的 schema；
- always_include 和近期 sticky 工具。

一次用户 turn 只调用一次选择模型。该 turn 内后续模型请求复用同一结果。

## 动态补充

如果主模型发现缺少能力，可以调用：

```text
tool_search(query="需要运行测试")
```

tool_search 使用本地确定性排序搜索完整目录，将匹配工具加入当前 turn。下一次主模型请求会附带新增工具的完整 schema，不会再次调用辅助模型。

## 失败处理

选择模型超时、返回非法 JSON、输出未知工具或只输出 reasoning 时，Drudge 使用本地 category/关键词排序。选择失败不会中断正常 Agent run。

工具选择只决定哪些 schema 对模型可见，不代表授权。最终调用仍经过 ToolProvider、风险分类和用户审批。

## Trace

tool_selection 和 tool_search 会写入 run_events；选择模型调用写入 model_calls，purpose 为 tool_selection。/status 会显示最近选择模式和工具数量。

## 验证

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
python -m unittest tests.test_tool_selection -v
python -m unittest discover -s tests -v
```
