# AgentRuntime 生命周期

AgentRuntime 为交互会话提供一个长期存在的异步运行环境。

## 生命周期

```python
from agent import Agent, AgentRuntime

agent = Agent(config)

async with AgentRuntime(agent) as runtime:
    first = await runtime.run_turn("检查项目")
    second = await runtime.run_turn("继续修复")
```

- start：初始化主模型和 ToolProvider，启动 MCP stdio 进程；
- run_turn：在同一个事件循环中串行执行一次用户 turn；
- cancel：取消当前模型或工具操作；
- close：关闭 MCP 进程并释放 provider；
- close 后 Runtime 不允许重新启动。

同一个 Runtime 内的多个 turn 会复用 MCP 连接。并发提交的 turn 通过异步锁串行执行，避免同时修改同一个消息历史。

## Agent 兼容行为

Agent 本身也提供 start 和 close：

```python
await agent.start()
try:
    await agent.run("first")
    await agent.run("second")
finally:
    await agent.close()
```

直接调用一次 agent.run 仍然兼容：如果 Agent 尚未启动，该次调用会临时启动 provider，并在结束后自动关闭。

## CLI

交互 CLI 现在只创建一个 asyncio event loop：

- prompt_toolkit 使用 prompt_async；
- slash 命令直接 await，不再嵌套 asyncio.run；
- MCP 服务在进入交互模式时启动，退出时关闭；
- /quit 和 EOF 会正常执行 Runtime.close。

单次 -q 查询仍使用一次性的 Agent 生命周期。

## 验证

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
python -m unittest tests.test_mcp_trace_tasks -v
python -m unittest discover -s tests -v
```
