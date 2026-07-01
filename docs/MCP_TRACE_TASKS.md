# MCP、运行 Trace 与持久化任务

本阶段增加三个能力：

1. 组合式 ToolProvider 和 MCP stdio；
2. SQLite 持久化运行 trace；
3. 可由用户和 Agent 共同维护的会话任务。

## MCP stdio

配置示例：

```yaml
mcp_servers:
  local-helper:
    command: python
    args:
      - path/to/server.py
    cwd: .
    timeout: 30
    risk: medium
    env:
      HELPER_MODE: safe
```

每个服务在一次 Agent run 开始时启动，完成或取消时关闭。Drudge 会执行 MCP initialize、tools/list 和 tools/call。暴露给模型的工具名带命名空间：

```text
mcp__local-helper__tool-name
```

当前支持 stdio transport。一个 run 内的工具调用仍按顺序执行，不支持 MCP sampling、resources 和 prompts。

安全规则：

- MCP 服务配置属于宿主配置，不能由模型修改；
- MCP 返回的风险描述不可信，Drudge 使用配置中的 risk；
- 默认 risk 为 medium；
- approval_mode=on_request 时必须由用户批准；
- approval_mode=never 会阻止 medium 及以上 MCP 工具；
- MCP stderr 只保留最近少量内容用于诊断。

交互命令：

```text
/mcp
/tools
```

## 持久化 Trace

SQLite 新增：

- runs：每次 Agent run 的状态、模型和起止时间；
- run_events：状态变化、工具调用、压缩和任务事件；
- model_calls：调用用途、模型、token、耗时和错误。

工具参数和结果写入 trace 前会限制长度，并按敏感字段名隐藏 token、API key 和密码。

```text
/runs
/trace
/trace <run_id>
```

## 持久化任务

任务属于 session，恢复会话后仍然存在。状态包括：

- pending
- in_progress
- completed
- cancelled

交互命令：

```text
/tasks
/tasks all
/task add 实现解析器
/task start 1
/task done 1
/task cancel 1
/task reopen 1
```

模型可调用以下宿主工具：

- task_list
- task_create
- task_update

未完成任务会注入后续轮次的 system context，任务工具只修改 Drudge 本地 SQLite 数据。

## 验证

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
python -m unittest tests.test_mcp_trace_tasks -v
python -m unittest discover -s tests -v
```
