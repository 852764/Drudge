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

审批使用方向键菜单，不再要求输入 `yes/no`：

```text
  1. 仅本次允许
  2. 本会话允许此工具（同一风险等级）
> 3. 拒绝本次操作
```

- `↑/↓` 或 `Tab/Shift+Tab`：移动选项，`Enter` 确认；数字 `1/2/3` 也可定位选项，仍需回车确认。
- 每次新申请默认选中“拒绝”，不会沿用上次允许选项。`Esc`、`Ctrl+D` 或输入流结束均拒绝本次操作。
- `Ctrl+C` 取消当前任务，交互 CLI 返回输入提示；单次查询退出码为 `130`。
- 菜单显示工具、风险、操作及参数。长详情使用 `PgUp/PgDn` 翻页；常见密钥字段脱敏，控制字符按文本显示。命令字符串里的所有秘密并不能保证自动识别，审阅时仍需留意。
- 会话允许按**工具名称 + 风险等级**记录，后续参数可以不同，不代表批准其他工具或更高风险；新建、恢复会话或退出进程后不沿用。
- 审批期间暂停后台状态行，结束或取消后恢复，不遗留抢占下一次输入的后台 `input()` 线程。粘贴文本不会触发批准。

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

自动测试覆盖真实按键解析、三种选项、默认拒绝、EOF/取消、状态行恢复、会话授权范围、审批允许/拒绝、直接绕过审批的阻断、两种 SSE 协议、Codex OAuth 增量回调和终端取消。无交互输入/输出、纯文本终端或菜单初始化失败时拒绝本次操作；Windows 原生控制台不依赖 Unix `TERM` 能力标记。
