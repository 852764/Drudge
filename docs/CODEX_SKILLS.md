# 在 Drudge 中使用 Codex skills

Drudge 可直接发现 Codex 使用的 `SKILL.md` 格式。推荐把技能完整目录安装到当前项目的 `.drudge/skills/<name>/`，保留 `scripts/`、`references/`、许可证及其他资源。无需连接 Codex 服务，也无需修改 Codex 原来的 skills 目录。

## 本次选择

来源：[OpenAI 官方 skills 仓库](https://github.com/openai/skills)，固定版本 `49f948faa9258a0c61caceaf225e179651397431`。

| Skill | 用途 | 额外前提 |
| --- | --- | --- |
| `cli-creator` | CLI 命令设计、JSON 输出、配置与打包工作流 | 使用项目已安装的语言工具链 |
| `gh-fix-ci` | 排查 GitHub Actions 的 PR 检查失败 | GitHub CLI `gh`、登录及仓库访问权限 |
| `gh-address-comments` | 梳理并处理 PR review / issue 评论 | GitHub CLI `gh`、登录及仓库访问权限 |
| `security-best-practices` | Python、JavaScript/TypeScript、Go 安全审查 | 按项目选择适用的参考文件 |
| `security-threat-model` | 基于仓库证据梳理信任边界、威胁与缓解措施 | 明确审查范围及部署背景 |

这些技能是工作流说明，不等于自动安装了配套工具。`gh` 尚未安装或登录时，先完成 GitHub CLI 的安装与登录；不要把 token 粘贴到对话或提交到仓库。

## 使用

在 Drudge 输入框执行：

```text
/skills
/skill cli-creator
```

然后发送实际任务，例如“检查这个 CLI 的命令结构和错误输出，提出改进并补测试”。

```text
/skill show cli-creator
/skill off cli-creator
/skill gh-fix-ci
```

也支持启动参数 `--skill cli-creator`；已启用的技能随会话保存，`/resume <id>` 会恢复。安装在当前工作区后，下一次发现技能即可使用，无需重新安装 Drudge。

`/skill <name>` 加载工作流说明；`/skill run <name>` 专用于 Drudge front matter 中显式定义的 workflow commands。这组官方技能没有该字段，按其说明通过常规工具调用脚本，不要用 `/skill run` 代替激活。

## 重新安装到其他工作区

如果已安装 Codex 的 `skill-installer`，在目标工作区执行以下 PowerShell 命令。`CODEX_HOME` 未设置时采用 `$HOME/.codex`：

```powershell
$codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $HOME '.codex' }
$installer = Join-Path $codexHome 'skills/.system/skill-installer/scripts/install-skill-from-github.py'
python $installer --repo openai/skills `
  --ref 49f948faa9258a0c61caceaf225e179651397431 `
  --path skills/.curated/cli-creator skills/.curated/gh-fix-ci skills/.curated/gh-address-comments skills/.curated/security-best-practices skills/.curated/security-threat-model `
  --dest .drudge/skills --method download
```

安装器需要网络，目标目录已有同名技能时会停止，避免覆盖本地改动。也可以从已有 Codex skills 目录复制选定技能的完整目录；不要复制 `.codex` 整个目录或任何认证文件。

`.drudge/` 默认被 Git 和发布包排除，因此这次是**项目本地安装**，不是把官方技能内置进 wheel。克隆到另一台机器后需执行上述安装步骤；本文档保留了来源、固定版本和安装列表。

## 兼容性与权限

- `name`、`description` 和 Markdown 正文直接加载，Codex 的 `metadata` 等额外 front matter 可保留。
- `agents/openai.yaml` 是 Codex UI 元数据，不会变成 Drudge 的权限或工具配置。
- 未启用的技能仅进入名称/描述目录；正文按需启用，链接到的参考文件不会全部注入上下文。
- Drudge 在已加载技能后加入执行说明：相对资源路径从技能目录解析，Shell 示例适配当前平台，配套技能使用项目 `.drudge/skills`。
- 上游的 `sandbox_permissions=require_escalated` 等 Codex 专用参数不是 Drudge 工具参数；宿主的 `ToolContext`、审批和工作区限制保持不变。
- 启用技能不会自动执行脚本、安装工具、登录账户或批准外部写操作。也不会读取 `.drudge/auth.json` 或 `.codex/auth.json`。

离线回归覆盖 Codex front matter、按需加载、启用/停用、会话恢复和宿主权限保持不变；真实 GitHub 联网工作流需要单独满足依赖并验收。
