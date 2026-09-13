# Drudge

Drudge 是一个 Python 3.10+ 终端编程 Agent，当前版本为 **0.2.0b1 Beta**，面向有人审阅的本地开发工作流。发布范围及已知边界见[发布说明](docs/RELEASE.md)。

## 当前能力

- OpenAI-compatible Chat Completions 客户端
- Responses API 与函数工具调用适配
- 终端、文件、Web 三类工具注册
- 单次查询与交互式 CLI
- YAML 配置文件与环境变量配置
- 不可变工具权限上下文和显式 Agent 运行状态
- SHA-256 编辑前置校验、原子文件写入、冲突检测撤销与 `/undo --dry-run` 预览
- 持久化压缩上下文、中断工具事务恢复，以及 `/fork` 会话分支
- 准确的终端退出状态、有界大输出捕获，以及 `/outputs` / `/output` 持久化分页
- 可恢复的计划、验收记录、并发修订冲突检测和 `/plan`
- 默认交互审批、Windows Job Object 后代进程清理及离线发布检查

## 安装

```bash
pip install -e .
```

## 配置

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

## 测试

测试完全离线，不需要 API Key：

```bash
python -m unittest discover -s tests -v
python -m compileall -q agent config.py main.py prompt tools tests
# 完整离线发布门槛（另需 setuptools、wheel）：
python scripts/release_check.py
```

## 开发文档

- [第一周实施说明](docs/WEEK1_IMPLEMENTATION.md)
- [使用 Codex 配置](docs/CODEX_CONFIG.md)
- [Drudge Codex OAuth](docs/CODEX_OAUTH.md)
- [Priority 1-3 Implementation Notes](docs/PRIORITY_1_2_3.md)
- [审批、流式输出与取消](docs/APPROVAL_STREAMING.md)
- [SQLite 会话恢复、AGENTS.md 与 Skills](docs/SESSIONS_AGENTS_SKILLS.md)
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
