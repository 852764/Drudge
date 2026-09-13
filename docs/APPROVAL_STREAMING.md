# 审批、流式输出与取消

本文记录下一阶段优先级 1、2 的实现结果。

## 1. 工具风险与审批

风险分为四级：

- `low`：只读本地操作，例如读取、搜索文件；
- `medium`：修改工作区文件、执行普通终端命令、只读网络请求；
- `high`：可能修改外部或系统状态，例如非 GET 网络请求、安装依赖、`git push`；
- `critical`：明显危险的系统级命令，始终由安全策略拦截。

启动交互审批：

```powershell
python main.py --codex-oauth --approval-mode on_request
```

审批选项：

- `y`：仅允许本次调用；
- `a`：本次会话内允许相同工具和风险等级；
- 回车或 `n`：拒绝。

从 `0.2.0b1` 起默认使用 `on_request`。审批在 Agent 主循环和工具注册表边界执行，不信任模型传入的参数。无交互终端时默认拒绝。`never` 会继续禁止终端、网络和文件修改；需要自动执行的可信流水线可显式配置 `auto`。非法审批模式在创建 `ToolContext` 时直接报错，避免拼写错误落入自动执行。

审批与危险命令字符串匹配不是操作系统沙箱。批准终端命令相当于允许其以当前用户权限运行；`workspace_root` 的文件路径约束不限制已批准的 shell 内部操作。审阅命令后再批准，处理不可信仓库时使用独立容器或低权限系统账户。

## 2. 流式输出

以下通道现在都会把文本增量直接发送到 CLI：

- OpenAI-compatible Chat Completions SSE；
- OpenAI Responses API 语义事件；
- Drudge 的 Codex OAuth Responses 后端。

Responses API 处理的主要事件包括 `response.output_text.delta`、`response.output_item.done`、`response.completed`、`response.incomplete`、`response.failed` 和 `error`。实现依据 OpenAI 官方的 [Streaming API responses](https://developers.openai.com/api/docs/guides/streaming-responses)。

工具参数仍会完整聚合后再交给工具注册表校验，不会执行未完成的参数片段。

## 3. 用户取消与子进程清理

模型生成或工具运行期间按 `Ctrl+C`：

- 当前 Agent run 进入 `cancelled` 状态；
- HTTP 流被关闭；
- 终端工具尝试终止整棵子进程树；
- 交互模式回到输入提示，不退出 Drudge。

终端超时也使用同一套进程清理逻辑。Windows 在启动 shell 前由独立 Python owner 加入 kill-on-close Job Object；退出或取消 owner 会关闭 Job，清理仍占用管道的后代进程。Job 初始化失败时不启动命令。Unix 使用独立进程组和 `SIGTERM`/`SIGKILL`。清理错误通过结果元数据或取消异常携带，重复取消会等待已有清理任务结束。

终端现在并发分块捕获 stdout/stderr，超时和取消可保留部分日志；这是输出捕获改进，不是终端实时渲染。大结果通过 `/outputs`、`/output` 分页查看。详见[工具状态与大输出分页](TOOL_OUTPUTS.md)，包括输出上限及限制性环境中的进程清理说明。

## 4. 验证

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
python -m unittest discover -s tests -v
python -m compileall -q agent config.py main.py prompt tools tests
python main.py --approval-mode on_request doctor
```

自动测试覆盖审批允许/拒绝、直接绕过审批的阻断、两种 SSE 协议、Codex OAuth 增量回调和终端取消。
