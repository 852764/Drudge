"""工具模块 — 自注册工具集"""

from .registry import registry
from .context import ToolContext

# 导入触发工具注册
from . import terminal  # noqa: F401
from . import file_ops  # noqa: F401
from . import web       # noqa: F401

__all__ = ["registry", "ToolContext"]
