"""Local command metadata shared by help and interactive completion.

This module has no UI dependencies and never discovers tools or calls a model.
Command execution and authorization remain in the existing host handlers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlashCommand:
    name: str
    arguments: str
    description: str
    menu_description: str
    aliases: tuple[str, ...] = ()
    takes_input: bool = False

    @property
    def usage(self) -> str:
        return f"{self.name} {self.arguments}".rstrip()


SLASH_COMMANDS = (
    SlashCommand("/help", "", "Show this help", "查看命令帮助"),
    SlashCommand("/sessions", "", "List saved sessions", "列出已保存会话"),
    SlashCommand("/resume", "<id>", "Resume a saved session", "恢复会话，需填写会话 ID", takes_input=True),
    SlashCommand("/new", "", "Start a new session", "开始新会话"),
    SlashCommand("/history", "[id]", "Show saved messages", "查看会话历史"),
    SlashCommand("/fork", "[title]", "Branch current conversation (workspace files stay shared)", "创建会话分支，工作区文件仍共享"),
    SlashCommand("/status", "", "Show session, context, and account limits", "查看会话、上下文和用量状态", aliases=("/usage",)),
    SlashCommand("/compact", "", "Compact older conversation context", "压缩较早的对话上下文"),
    SlashCommand("/plan", "", "Show session plan, acceptance criteria and evidence", "查看计划与验收记录"),
    SlashCommand("/tools", "", "List available tools", "列出可用工具"),
    SlashCommand("/mcp", "", "Inspect configured MCP stdio servers", "查看 MCP 服务和工具"),
    SlashCommand("/config", "", "Show current config", "查看当前配置（敏感字段脱敏）"),
    SlashCommand("/models", "", "List provider models", "查询服务商模型列表"),
    SlashCommand("/providers", "", "List built-in provider configurations", "查看支持的 provider 配置与协议"),
    SlashCommand("/runs", "", "List recent runs", "查看最近运行记录"),
    SlashCommand("/trace", "[run_id]", "Show a persisted run trace", "查看运行轨迹"),
    SlashCommand("/tasks", "[all]", "List persistent session tasks", "查看任务，all 包含已关闭任务"),
    SlashCommand("/task", "add <title> | start|done|cancel|reopen <id>", "Create or update a persistent task", "创建或更新任务", takes_input=True),
    SlashCommand("/memory", "list [scope] | add <project|user> <content> | pin|unpin|rm <id>", "Manage durable project/user memories", "查看或管理持久记忆", takes_input=True),
    SlashCommand("/changes", "", "List reversible file changes", "查看可撤销的文件修改"),
    SlashCommand("/undo", "[--dry-run]", "Revert or preview the latest file change (conflict-checked)", "撤销最近修改；--dry-run 仅预览"),
    SlashCommand("/outputs", "", "List captured tool outputs in this session", "列出当前会话的工具输出"),
    SlashCommand("/output", "<id> [offset] [limit]", "Read an output page (up to 1000 characters)", "按 ID 分页读取工具输出", takes_input=True),
    SlashCommand("/skills", "", "List discovered skills", "查看已发现的技能"),
    SlashCommand("/skill", "<name> | off|show <name> | run <name> [phase] | clear", "Activate, inspect, run or deactivate a skill", "启用、查看、运行或停用技能", takes_input=True),
    SlashCommand("/clear", "", "Clear screen", "清空屏幕，保留会话"),
    SlashCommand("/quit", "", "Exit Drudge", "退出 Drudge", aliases=("/exit", "/q")),
)


@dataclass(frozen=True)
class CommandOption:
    name: str
    description: str
    takes_input: bool = False


# Only literal options supported by the host command handlers. Free-form text,
# session IDs, file paths and skill names are left untouched.
COMMAND_OPTIONS = {
    ("/tasks",): (CommandOption("all", "包含已完成和已取消的任务"),),
    ("/undo",): (CommandOption("--dry-run", "仅预览撤销，不修改文件"),),
    ("/task",): (
        CommandOption("add", "创建任务：add <标题>", True),
        CommandOption("start", "开始任务：start <ID>", True),
        CommandOption("done", "完成任务：done <ID>", True),
        CommandOption("cancel", "取消任务：cancel <ID>", True),
        CommandOption("reopen", "重新打开任务：reopen <ID>", True),
    ),
    ("/skill",): (
        CommandOption("show", "查看技能：show <名称>", True),
        CommandOption("off", "停用技能：off <名称>", True),
        CommandOption("run", "运行技能：run <名称> [阶段]", True),
        CommandOption("clear", "停用所有技能"),
    ),
    ("/memory",): (
        CommandOption("list", "列出记忆：list [project|user]"),
        CommandOption("add", "添加记忆：add <project|user> <内容>", True),
        CommandOption("pin", "固定记忆：pin <ID>", True),
        CommandOption("unpin", "取消固定：unpin <ID>", True),
        CommandOption("rm", "删除记忆：rm <ID>", True),
    ),
    ("/memory", "add"): (
        CommandOption("project", "当前项目记忆，后接记忆内容", True),
        CommandOption("user", "用户记忆，后接记忆内容", True),
    ),
    ("/memory", "list"): (
        CommandOption("project", "只列出当前项目记忆"),
        CommandOption("user", "只列出用户记忆"),
    ),
}


def command_help_lines() -> list[str]:
    lines = ["Type / for commands; Up/Down select, Tab/Enter fill, Esc dismiss."]
    for command in SLASH_COMMANDS:
        aliases = f" (aliases: {', '.join(command.aliases)})" if command.aliases else ""
        lines.append(f"{command.usage:<28} {command.description}{aliases}")
    return lines
