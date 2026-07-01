# Drudge 状态与 Codex 限额

Drudge 现在支持类似 Codex `/status` 的状态视图。

## 使用

交互模式：

```text
/status
```

命令行：

```powershell
$env:DRUDGE_HOME="F:\Drudge\.drudge\live"
$env:DRUDGE_DB_PATH="F:\Drudge\.drudge\drudge.db"
python main.py --codex-oauth status
```

JSON 输出：

```powershell
python main.py --codex-oauth status --json
```

恢复指定会话后查看状态：

```powershell
python main.py --codex-oauth --resume <session_id> status
```

## 显示内容

本地状态：

- Session ID 和运行状态；
- 当前模型、Provider 和 API 类型；
- turn、消息数量以及本进程 token 数；
- 估算上下文占用和剩余空间；
- workspace、审批模式和已启用 Skills。

Codex OAuth 账户状态：

- ChatGPT 计划类型；
- 5 小时窗口已用/剩余百分比和重置时间；
- 周窗口已用/剩余百分比和重置时间；
- 额外模型窗口和 credits（账户返回时）。

Drudge 每次执行 `/status` 都读取实时快照，不使用历史 session 中可能过期的 rate-limit 数据。响应中的 email、user ID 和 account ID 不会进入状态对象或终端输出。

## Provider 边界

账户订阅限额只适用于 `--codex-oauth`。第三方中转站和普通 API Key 没有统一的账户限额协议，因此 Drudge 只显示本地状态。如果中转站以后提供自己的 usage endpoint，应通过独立 Provider adapter 接入，不能套用 ChatGPT 限额。

官方 Codex app-server 对应的稳定接口是 `account/rateLimits/read`，字段包含 `usedPercent`、`windowDurationMins` 和 `resetsAt`。Drudge 为避免启动 Codex 子进程，使用现有 Drudge OAuth 凭据直读当前 Codex 同源 usage 后端并归一化字段。该后端不是通用 OpenAI API 合约；如果上游发生变化，Drudge 会显示 account usage unavailable，而不会影响 Agent 对话。

参考：

- [Codex app-server Rate limits](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md#7-rate-limits-chatgpt)
- [Codex slash commands](https://developers.openai.com/codex/cli/slash-commands)
