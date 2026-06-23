"""核心 牛马，其他的牛马都是这个牛马来的 — 工具调用 + LLM 交互 + 上下文管理"""

import json
import os
import asyncio
from typing import Any

from .llm import LLMClient, create_client
from .tools import registry
from .prompt import build_system_prompt
from .config import get_config, ConfigManager
from .utils import truncate_string

# 工具调用结果的最大字符数
MAX_TOOL_RESULT_CHARS = 10000


class Agent:
    """核心 Agent"""

    def __init__(self, config: ConfigManager | None = None):
        self.config = config or get_config()
        self.llm: LLMClient | None = None
        self._messages: list[dict] = []
        self._turn_count: int = 0
        self._total_tokens: int = 0

    def _init_llm(self) -> None:
        """初始化 LLM 客户端"""
        if self.llm is None:
            mc = self.config.get_model_config()
            self.llm = create_client(mc)

    def _build_initial_messages(
        self,
        user_prompt: str,
        memory_entries: list[str] | None = None,
        skills: list[str] | None = None,
    ) -> list[dict]:
        """构建初始消息列表"""
        toolsets = self.config.get_toolsets()
        system_content = build_system_prompt(toolsets, memory_entries, skills)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ]

    async def run(
        self,
        prompt: str,
        memory_entries: list[str] | None = None,
        skills: list[str] | None = None,
        stream_callback: Any = None,
    ) -> str:
        """
        执行 Agent 对话循环

        Args:
            prompt: 用户输入的提示词
            memory_entries: 记忆条目列表
            skills: 加载的技能文档列表
            stream_callback: 可选的回调函数，接收每一轮的文本增量

        Returns:
            最终响应文本
        """
        self._init_llm()
        self._turn_count = 0
        self._messages = self._build_initial_messages(prompt, memory_entries, skills)

        max_turns = self.config.get_agent_config().get("max_turns", 50)
        compression_threshold = self.config.get_agent_config().get("compression_threshold", 0.80)
        toolsets = self.config.get_toolsets()
        tool_schemas = registry.get_schemas(toolsets)

        final_response_parts: list[str] = []

        while self._turn_count < max_turns:
            self._turn_count += 1

            # 上下文压缩检查
            estimated = self.llm.estimate_tokens(self._messages)
            context_limit = self.config.get("model", "context_length")
            if estimated > context_limit * compression_threshold:
                self._compress_context()

            try:
                response = await self.llm.chat(
                    self._messages,
                    tools=tool_schemas if tool_schemas else None,
                )
            except Exception as e:
                error_msg = f"LLM call failed (turn {self._turn_count}): {e}"
                final_response_parts.append(error_msg)
                break

            # 统计 token 使用
            usage = response.get("usage", {})
            self._total_tokens += usage.get("total_tokens", 0)

            finish_reason = self.llm.extract_finish_reason(response)

            # 处理文本响应
            text = self.llm.extract_text(response)
            if text:
                final_response_parts.append(text)
                # 如果 LLM 决定停止且不是 tool_call，返回
                if finish_reason == "stop":
                    break

            # 处理工具调用
            tool_calls = self.llm.extract_tool_calls(response)
            if tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": tool_calls,
                }
                self._messages.append(assistant_msg)

                for tc in tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "")
                    tool_args_str = func.get("arguments", "{}")

                    try:
                        tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
                    except json.JSONDecodeError:
                        tool_args = {}

                    # 执行工具（支持 async handler）
                    tool_result = await registry.dispatch_async(tool_name, tool_args)
                    tool_result = truncate_string(tool_result, MAX_TOOL_RESULT_CHARS)

                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": tool_result,
                    }
                    self._messages.append(tool_msg)

                    # 日志（如果配置开启了）
                    if self.config.get("display", "show_tool_calls"):
                        self._log_tool_call(tool_name, tool_args, tool_result)

                # 继续下一轮
                continue

            # 无工具调用且 finish_reason 不是 stop — 将助手消息加入历史
            if text:
                self._messages.append({"role": "assistant", "content": text})

            # 如果 finish_reason 是 stop 或 length，退出
            if finish_reason in ("stop", "length"):
                break

        # 达到 max_turns
        if self._turn_count >= max_turns:
            final_response_parts.append(
                f"\n\n[Agent reached maximum turns ({max_turns}). Task may be incomplete.]"
            )

        return "\n".join(final_response_parts)

    def _compress_context(self) -> None:
        """压缩上下文：保留系统消息 + 最近 N 轮 + 会话摘要"""
        system_msg = None
        later_messages = []
        keep_recent = 6  # 保留最近 6 条消息（3 轮）

        for msg in self._messages:
            if msg["role"] == "system":
                system_msg = msg
            else:
                later_messages.append(msg)

        if len(later_messages) <= keep_recent:
            return

        # 保留系统消息 + 最近的消息
        recent = later_messages[-keep_recent:]
        old_messages = later_messages[:-keep_recent]

        # 摘要旧消息
        summary = self._summarize_messages(old_messages)
        summary_msg = {
            "role": "user",
            "content": f"[Previous conversation summary]\n{summary}\n[End summary]",
        }

        new_messages = [system_msg] if system_msg else []
        new_messages.append(summary_msg)
        new_messages.extend(recent)
        self._messages = new_messages

    def _summarize_messages(self, messages: list[dict]) -> str:
        """简单摘要：提取关键信息。

        TODO(Phase 1): 当前实现仅做简单截断（前 200 字符），Phase 1 应改为调用
        LLM 做摘要压缩，以更好地保留对话语义信息。当前 MVP 实现已满足基本可用性。
        """
        parts = []
        for msg in messages:
            if msg["role"] == "user":
                content = msg.get("content", "")
                if content and len(content) > 20:
                    parts.append(f"User asked: {content[:200]}")
            elif msg["role"] == "assistant":
                content = msg.get("content", "")
                if content:
                    parts.append(f"Agent responded: {content[:200]}")
            elif msg["role"] == "tool":
                content = msg.get("content", "")
                if "error" in content.lower():
                    parts.append(f"Tool error: {content[:100]}")
                else:
                    parts.append("Tool result received")
        return "\n".join(parts[:10])

    def _log_tool_call(self, name: str, args: dict, result: str) -> None:
        """记录工具调用（彩色输出）"""
        try:
            from colorama import Fore, Style, init
            init()
            print(f"{Fore.CYAN}[TOOL] {name}{Style.RESET_ALL}")
            if args:
                arg_str = json.dumps(args, ensure_ascii=False)
                if len(arg_str) > 150:
                    arg_str = arg_str[:150] + "..."
                print(f"  args: {arg_str}")
            if result:
                preview = result[:200].replace("\n", " ")
                print(f"  result: {preview}")
            print()
        except ImportError:
            pass

    def get_token_usage(self) -> dict:
        """获取 token 使用统计"""
        return {
            "total_tokens": self._total_tokens,
            "turns": self._turn_count,
        }

    def get_messages(self) -> list[dict]:
        """获取当前消息历史"""
        return list(self._messages)
