"""核心 牛马，其他的牛马都是这个牛马来的 — 工具调用 + LLM 交互 + 上下文管理"""

import json
import os
import asyncio
from typing import Any

from .llm import LLMClient, create_client
from .refusal import build_refusal_review_messages, is_refusal
from .storage import ConversationStore
from tools import registry
from prompt import build_system_prompt
from config import get_config, ConfigManager
from utils import truncate_string

# 工具调用结果的最大字符数
MAX_TOOL_RESULT_CHARS = 10000


class Agent:
    """核心 Agent"""

    def __init__(self, config: ConfigManager | None = None):
        self.config = config or get_config()
        self.llm: LLMClient | None = None
        self.refusal_llm: LLMClient | None = None
        self._messages: list[dict] = []
        self._turn_count: int = 0
        self._total_tokens: int = 0
        self.session_id: str | None = None
        self.store: ConversationStore | None = None
        self._init_store()

    def _init_store(self) -> None:
        storage_config = self.config.get_storage_config()
        if not storage_config.get("enabled", True):
            return
        self.store = ConversationStore(storage_config.get("path", "~/.drudge/drudge.db"))

    def _init_llm(self) -> None:
        """初始化 LLM 客户端"""
        if self.llm is None:
            mc = self.config.get_model_config()
            self.llm = create_client(mc)
        security = self.config.get_security_config()
        registry.set_runtime_defaults({
            "workspace": security.get("workspace_root", os.getcwd()),
            "allow_outside_workspace": security.get("allow_outside_workspace", False),
            "allow_terminal": security.get("allow_terminal", True),
        })

    def _init_refusal_llm(self) -> None:
        if self.refusal_llm is None:
            mc = dict(self.config.get_model_config())
            review_config = self.config.get("agent", "refusal_review_model", default=None)
            if isinstance(review_config, dict):
                mc.update(review_config)
            self.refusal_llm = create_client(mc)

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

    def _ensure_session(self, prompt: str, memory_entries: list[str] | None, skills: list[str] | None) -> None:
        if self._messages:
            self._messages.append({"role": "user", "content": prompt})
            self._persist_message("user", prompt)
            return

        self._messages = self._build_initial_messages(prompt, memory_entries, skills)
        if not self.store:
            return

        model = self.config.get("model", "name", default="unknown")
        self.session_id = self.store.create_session(prompt, model)
        for message in self._messages:
            self._persist_message(message["role"], message.get("content"))

    def _persist_message(
        self,
        role: str,
        content: str | None,
        *,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.store and self.session_id:
            self.store.append_message(
                self.session_id,
                role,
                content,
                tool_call_id=tool_call_id,
                metadata=metadata,
            )

    def _persist_tool_call(self, name: str, args: dict, result: str) -> None:
        if self.store and self.session_id:
            self.store.append_tool_call(self.session_id, self._turn_count, name, args, result)

    async def _review_refusal_if_needed(self, user_prompt: str, response_text: str) -> str:
        agent_config = self.config.get_agent_config()
        if not agent_config.get("refusal_review_enabled", True):
            return response_text
        if not is_refusal(response_text):
            return response_text

        notice = agent_config.get(
            "refusal_review_notice",
            "[Drudge] Detected a possible refusal. Running a safe second-pass review...",
        )
        self._persist_message("system", notice, metadata={"event": "refusal_review_started"})

        self._init_refusal_llm()
        review_messages = build_refusal_review_messages(user_prompt, response_text)
        review_response = await self.refusal_llm.chat(review_messages)
        usage = review_response.get("usage", {})
        self._total_tokens += usage.get("total_tokens", 0)
        review_text = self.refusal_llm.extract_text(review_response) or response_text
        self._messages.append({"role": "assistant", "content": review_text})
        self._persist_message("assistant", review_text, metadata={"refusal_review": True})
        return f"{notice}\n\n{review_text}"

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
        self._ensure_session(prompt, memory_entries, skills)

        max_turns = self.config.get_agent_config().get("max_turns", 50)
        compression_threshold = self.config.get_agent_config().get("compression_threshold", 0.80)
        toolsets = self.config.get_toolsets()
        tool_schemas = registry.get_schemas(toolsets)

        final_response_parts: list[str] = []

        request_turns = 0
        hit_max_turns = True
        while request_turns < max_turns:
            request_turns += 1
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
                    self._messages.append({"role": "assistant", "content": text})
                    self._persist_message("assistant", text)
                    hit_max_turns = False
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
                self._persist_message("assistant", text, metadata={"tool_calls": tool_calls})

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
                    self._persist_message("tool", tool_result, tool_call_id=tc.get("id", ""))
                    self._persist_tool_call(tool_name, tool_args, tool_result)

                    # 日志（如果配置开启了）
                    if self.config.get("display", "show_tool_calls"):
                        self._log_tool_call(tool_name, tool_args, tool_result)

                # 继续下一轮
                continue

            # 无工具调用且 finish_reason 不是 stop — 将助手消息加入历史
            if text:
                self._messages.append({"role": "assistant", "content": text})
                self._persist_message("assistant", text)

            # 如果 finish_reason 是 stop 或 length，退出
            if finish_reason in ("stop", "length"):
                hit_max_turns = False
                break

        # 达到 max_turns
        if hit_max_turns:
            final_response_parts.append(
                f"\n\n[Agent reached maximum turns ({max_turns}). Task may be incomplete.]"
            )

        final_text = "\n".join(final_response_parts)
        return await self._review_refusal_if_needed(prompt, final_text)

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
