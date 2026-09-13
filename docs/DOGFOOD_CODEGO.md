# CodeGo 真实试用与 0.2.0b2 改进

本轮使用用户配置的 CodeGo、本地 `http://127.0.0.1:8318/v1`、`gpt-6-astra`、`xhigh`，实际运行了 CLI 和编码任务，而非只使用模型替身。传输为 Responses HTTP/SSE；`supports_websockets` 不代表 Drudge 已启用 WebSocket。

## 实测范围与结果

编码任务是在新建的购物车样本中，将固定金额扣减改为百分比折扣，补齐 Decimal/ROUND_HALF_UP、无效输入与空购物车行为。允许改动三个指定文件，保留原测试，并要求实际执行 `python -m unittest discover -s tests -v`。前后使用相同任务、模型、推理强度和 **15 轮上限**。

| 指标 | 改进前 | 改进后 |
| --- | --- | --- |
| 主任务状态 | 15 轮耗尽，尚未执行测试 | 8 轮完成 |
| 主任务耗时 | 191.4 秒 | 160.3 秒 |
| 主任务 reported tokens | 106,158 | 70,965 |
| Agent 自行运行测试 | 未运行 | 17 项通过，终端退出码 0 |
| 独立宿主功能验收 | 8 项通过 | 8 项通过 |
| 原测试文件 | 未改动 | 未改动 |
| 计划恢复 | 成功，验证步骤尚未完成 | 成功，三个步骤均完成 |

这些是**各一次样本**，不是稳定吞吐量基准，也不是竞品对比。token 数是 provider 报告的累计 usage，不代表账单、去重输入量或缓存命中量。表中不含恢复追问；基线耗尽时任务尚未完成，不能把两段耗时解读为相同完成度的提速保证。

改进后还验证了压缩与恢复：工作上下文从 20 条消息压缩到 10 条，计划仍可恢复。批量读取中两个尚不存在的文件返回独立错误，其余四个文件的读取结果仍可使用。Agent 随后创建所需测试并完成验证。

另一次 CLI 连通性基线在 3.48 秒内返回 `CODEGO_READY`，退出码 0、1,337 reported tokens。未加载自定义模型指令文件。

本地证据保存在被 Git 忽略的 `build/dogfood/`：`cli-tljzpb6f`、`cart-k07syq9s`、`cart-11kn4y2t`。每个目录包含报告；编码任务还包含会话 SQLite。真实工具调用及退出码已从 SQLite 核对，不以模型自述代替证据。原始会话、测试数据库和密钥不随源码发布。

## 已修复

- **批量读取**：`read_files` 一次接受 1–8 个文件，逐项保留状态、行号与整文件 SHA-256。默认每文件读取 200 行、最多 500 行；每项 JSON 预览最多 6,000 字符，大结果可通过输出 receipt 分页。任一文件失败使批次 `ok=false`，成功文件仍保留。每个路径单独执行宿主权限校验。
- **收尾效率**：临时宿主提示告知剩余循环轮次，鼓励合并独立读取、在成功编辑后优先测试而非重复读回。预算提示不写入用户历史，也不扩大权限。
- **完整工具禁用**：`--no-tools` 现在同时禁用本地、计划、记忆、输出和 MCP 工具；不启动 MCP 子进程。内部宿主选项为 `agent.tools_enabled: false`。单独配置 `toolsets: []` 保留原来的元工具语义。
- **可靠失败状态**：不完整或异常终止的模型响应不会被标为成功，其中的函数调用不执行。可见的部分文本仍保留。单次查询退出码：成功 0、失败 1、轮次耗尽 2、取消 130。
- **超时生效**：普通 Chat/Responses 客户端使用 `model.timeout`；只接受正有限数字。
- **工作区准确性**：提示和新会话元数据使用实际工具工作区，不再误用 CLI 启动目录。宿主撤销文件工具后，撤销操作也遵循更新后的权限。

## 使用 CodeGo（不在文件中保存密钥）

创建 `codego.toml`：

```toml
model_provider = "CodeGo"
model = "gpt-6-astra"
model_reasoning_effort = "xhigh"

[model_providers.CodeGo]
name = "CodeGo"
base_url = "http://127.0.0.1:8318/v1"
env_key = "CPA_API_KEY"
wire_api = "responses"
```

从工作区目录运行，先在当前环境设置 `CPA_API_KEY`，不把值放进 TOML 或 Git：

```powershell
drudge --codex-config codego.toml --toolsets file,terminal -q "检查当前项目并运行测试"
```

单次非交互模式遇到需审批的操作会拒绝执行；交互审阅时省略 `-q`。可信独立样本的自动化可显式选择 `--approval-mode auto`，这不是操作系统沙箱。

Drudge 目前没有 provider 原生 web-search 工具；`web_search` 与 `supports_websockets` 不属于其 TOML 导入子集。选择 `--toolsets file,terminal` 不注册 `web_request`，但不限制已批准的终端命令内部联网。自定义模型指令文件和 Codex 项目信任设置不用于授予 Drudge 工具权限。

## 后续仍需测量

下一步应扩大真实仓库任务集、重复运行并记录分位数、非预期改动与费用。本轮未改变仓库搜索规模、后台任务或数据库总量保留策略，也未修订 provider 能力探测器；这些不应描述为已完成的能力。
