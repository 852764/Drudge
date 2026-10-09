# Drudge

Drudge 是一个 Python 3.10+ 终端编程 Agent，当前版本为 **0.2.0b3 Beta**，面向有人审阅的本地开发工作流。发布范围及已知边界见[发布说明](docs/RELEASE.md)。

项目目标是逐步演进成类似 Codex 的本地开发助手。

## 当前能力

- OpenAI-compatible Chat Completions 客户端
- Responses API 与函数工具调用适配
- Anthropic 原生 Messages，以及 OpenAI、DeepSeek、OpenRouter、Gemini 兼容接口、DashScope、Ollama、LM Studio 配置预设
- 终端、文件、Web 三类工具注册
- 单次查询与交互式 CLI
- 输入 `/` 弹出命令菜单、中文说明、上下选择及 Tab 补全
- YAML 配置文件与环境变量配置
- 不可变工具权限上下文和显式 Agent 运行状态
- SHA-256 编辑前置校验、原子文件写入、冲突检测撤销与 `/undo --dry-run` 预览
- 持久化压缩上下文、中断工具事务恢复，以及 `/fork` 会话分支
- 准确的终端退出状态、有界大输出捕获，以及 `/outputs` / `/output` 持久化分页
- 可恢复的计划、验收记录、并发修订冲突检测和 `/plan`
- 默认交互审批、Windows Job Object 后代进程清理及离线发布检查
- 批量读取、剩余轮次提示、完整的 `--no-tools` 禁用，以及可靠的单次查询退出码
- 统一的 `.gitignore` 仓库发现、无静默 200 文件截断的搜索，以及明确的搜索完整性与行列定位
- 可配置的 JSReverser-MCP 浏览器调试预设：页面、DOM 脚本、Console/Network、断点与截图

## 安装

```bash
pip install -e .
```

## 配置

多服务商可通过 `model.provider` 选择预设，使用环境变量提供密钥；交互命令 `/providers` 查看支持列表。参数差异、配置示例和能力探测见[多模型配置说明](docs/MODEL_PROVIDERS.md)。

默认读取环境变量：

```bash
set DRUDGE_API_KEY=your_local_proxy_api_key
set DRUDGE_MODEL=gpt-5.5
set DRUDGE_BASE_URL=http://127.0.0.1:8318/v1
```

也可以传入 YAML 配置：

```yaml
model:
  provider: custom
  name: gpt-5.5
  base_url: http://127.0.0.1:8318/v1
  api: responses
  reasoning_effort: high
  disable_response_storage: true
  api_key: your_api_key
toolsets:
  - terminal
  - file
  - web

agent:
  refusal_review_enabled: true
  refusal_review_notice: "[Drudge] 检测到模型可能拒绝了请求，正在进行安全二次处理..."
  # Optional: override the second-pass review model/provider.
  # refusal_review_model:
  #   name: gpt-4o-mini
```

### 默认浏览器调试（可选安装）

构建 JSReverser-MCP 后，将它注册到本地配置；后续每次使用此配置启动都会自动连接，保留交互审批：

```powershell
python scripts/configure_browser.py --server-path build/reference/JSReverser-MCP/build/src/index.js --enable-network --probe
drudge -c config.local.yaml
```

上游安装步骤、CodeGo 启动方式、可见浏览器选项和已知兼容问题见[浏览器调试说明](docs/BROWSER_MCP.md)。

## 使用

默认 `on_request` 会在修改文件或执行终端命令前询问；非交互运行不会自动批准。仅在信任任务和执行环境时显式使用 `--approval-mode auto`。审批不是操作系统沙箱。

```bash
drudge --version
drudge --help
drudge -q "列出当前目录文件"
drudge -c config.yaml -m gpt-4o-mini
drudge --codex-config -q "检查当前项目"
drudge --codex-config C:\path\to\config.toml
```

开发时也可以直接运行：

```bash
python main.py --version
python main.py --help
```

### 交互式命令菜单

在对话输入框键入 `/` 自动显示命令菜单，继续输入前缀可以筛选，例如 `/re` 对应 `/resume`。

- `↑` / `↓`：选择命令；`Shift+Tab`：向上选择。
- `Tab`：填入当前选项；未选择时填入第一项。
- 已选择选项时按 `Enter` 只填入命令，补齐参数后再按 `Enter` 执行；手动输入命令时直接按 `Enter` 执行。
- `Esc`：关闭菜单并恢复选择前输入的前缀。
- `/task `、`/skill `、`/memory `、`/undo ` 等支持子命令或选项补全。

例如：输入 `/re` → `Tab` → 填写会话 ID → `Enter` 恢复会话；会话 ID 可先通过 `/sessions` 查看。
候选菜单只使用内置命令元数据，不请求模型、不连接 MCP，也不自动执行命令。普通文本保留原有输入与历史行为。
管道输入、纯文本终端或缺少 `prompt-toolkit` 时使用简单输入模式，可输入 `/` 或 `/help` 查看完整帮助。

## 测试

测试完全离线，不需要 API Key：

```bash
python -m unittest discover -s tests -v
python -m compileall -q agent config.py main.py prompt tools tests
# 完整离线发布门槛（另需 setuptools、wheel）：
python scripts/release_check.py
```

## 开发文档

- [多模型、Provider 配置与协议兼容](docs/MODEL_PROVIDERS.md)
- [第一周实施说明](docs/WEEK1_IMPLEMENTATION.md)
- [使用 Codex 配置](docs/CODEX_CONFIG.md)
- [Drudge Codex OAuth](docs/CODEX_OAUTH.md)
- [Priority 1-3 Implementation Notes](docs/PRIORITY_1_2_3.md)
- [审批、流式输出与取消](docs/APPROVAL_STREAMING.md)
- [SQLite 会话恢复、AGENTS.md 与 Skills](docs/SESSIONS_AGENTS_SKILLS.md)
- [在 Drudge 中安装和使用 Codex skills](docs/CODEX_SKILLS.md)
- [Drudge 状态与 Codex 限额](docs/STATUS.md)
- [`<think>` 刷屏与无最终答案防护](docs/THINK_OUTPUT_FIX.md)
- [LLM 上下文摘要压缩](docs/LLM_CONTEXT_COMPACTION.md)
- [MCP、运行 Trace 与持久化任务](docs/MCP_TRACE_TASKS.md)
- [AgentRuntime 生命周期](docs/AGENT_RUNTIME.md)
- [动态工具选择](docs/TOOL_SELECTION.md)
- [模型请求重试与错误恢复](docs/LLM_RETRY_RECOVERY.md)
- [可靠文件编辑与撤销预览](docs/RELIABLE_FILE_EDITS.md)
- [长任务上下文恢复与会话分支](docs/DURABLE_CONTEXT.md)
- [工具状态与大输出分页](docs/TOOL_OUTPUTS.md)
- [持久化计划与验收记录](docs/PERSISTENT_PLANS.md)
- [Beta 发布门槛与运维说明](docs/RELEASE.md)
- [版本变更记录](docs/CHANGELOG.md)
- [CodeGo 真实试用记录与本轮改进](docs/DOGFOOD_CODEGO.md)
- [仓库搜索、忽略规则与扫描预算](docs/REPOSITORY_SEARCH.md)
- [默认浏览器调试与 JSReverser-MCP](docs/BROWSER_MCP.md)
