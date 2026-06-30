# SQLite 会话恢复、AGENTS.md 与 Skills

## 1. 为什么使用 SQLite

当前 Drudge 是单机 CLI Agent，SQLite 足够承担会话、消息和工具调用存储：

- 不需要部署额外数据库；
- 每条消息和工具结果可事务化写入；
- 可以用 WAL 模式提高读写稳定性；
- 后续可以继续增加 checkpoint、trace 和任务表。

默认数据库路径是 `~/.drudge/drudge.db`。也可以设置项目本地数据库：

```powershell
$env:DRUDGE_DB_PATH="F:\Drudge\.drudge\drudge.db"
```

数据库启动时会自动迁移旧的 `sessions` 表，增加会话元数据字段。

## 2. 恢复会话

交互模式：

```text
/sessions
/resume <session_id>
/history <session_id>
/new
```

启动时恢复：

```powershell
python main.py --codex-oauth --resume <session_id>
```

恢复内容包括：

- system、user、assistant、tool 消息；
- Function Calling 参数和 tool call ID；
- Responses API 的 `provider_items`；
- 会话启用的 Skills；
- 累计 Agent turn。

如果程序在工具调用完成前退出，恢复时会补入一个 `interrupted` tool result，保证 Chat Completions 和 Responses 的工具事务仍然完整。恢复使用当前工作区、当前模型配置以及最新的 `AGENTS.md`/Skill 内容。

## 3. AGENTS.md

Drudge 从 `security.workspace_root` 开始，按目录层级加载到当前工作目录之间的 `AGENTS.md`：

```text
project/AGENTS.md
project/src/AGENTS.md
project/src/feature/AGENTS.md
```

越深层的文件作用域越小，并在冲突时优先。示例：

```markdown
# AGENTS.md

- 修改代码后运行 `python -m unittest discover -s tests -v`。
- 不要修改 `.drudge/auth.json`。
- 新模块必须包含离线测试。
```

配置项：

```yaml
agent:
  instructions_enabled: true
  instructions_filename: AGENTS.md
  instructions_max_chars: 64000
```

运行 `python main.py doctor` 可以查看实际加载的文件。

## 4. Skills MVP

项目 Skill 目录：

```text
.drudge/
└── skills/
    └── code-review/
        ├── SKILL.md
        ├── scripts/
        └── references/
```

全局 Skill 目录是 `$DRUDGE_HOME/skills`，项目 Skill 会覆盖同名全局 Skill。

`SKILL.md` 示例：

```markdown
---
name: code-review
description: Review source changes and run focused tests
---

1. 检查变更涉及的调用链。
2. 优先运行相关测试，再运行完整测试。
3. 输出按严重程度排序的问题。
```

交互命令：

```text
/skills
/skill code-review
/skill show code-review
/skill off code-review
/skill clear
```

启动时启用一个或多个 Skill：

```powershell
python main.py --codex-oauth --skill code-review --skill tests
```

MVP 只会自动加载启用 Skill 的 `SKILL.md`。`references/` 和 `scripts/` 不会全部注入上下文，Agent 可以根据 Skill 指令通过文件/终端工具按需使用。

## 5. 验证

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
python -m unittest discover -s tests -v
python -m compileall -q agent config.py main.py prompt tools tests
python main.py doctor
```
