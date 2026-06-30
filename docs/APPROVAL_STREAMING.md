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

`on_request` 的审批在 Agent 主循环和工具注册表边界执行，不信任模型传入的参数。无交互终端时默认拒绝。`never` 会继续禁止终端、网络和文件修改；`auto` 保持原来的自动执行策略，但危险命令仍会被拦截。

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

终端超时也使用同一套进程清理逻辑。Windows 使用新进程组和 `taskkill /T /F`，Unix 使用独立进程组和 `SIGTERM`/`SIGKILL`。

## 4. 验证

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
python -m unittest discover -s tests -v
python -m compileall -q agent config.py main.py prompt tools tests
python main.py --approval-mode on_request doctor
```

自动测试覆盖审批允许/拒绝、直接绕过审批的阻断、两种 SSE 协议、Codex OAuth 增量回调和终端取消。
