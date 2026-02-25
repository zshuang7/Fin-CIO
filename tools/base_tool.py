"""
tools/base_tool.py — Abstract base class for all FinAgent data-source tools.

HOW TO ADD A NEW DATA SOURCE IN 4 STEPS:
─────────────────────────────────────────
1. Create  tools/my_new_tool.py
2. Define  class MyNewTool(BaseTool):
               tool_name        = "my_new_tool"
               tool_description = "Fetches data from XYZ API"
3. In __init__, call super().__init__(name=self.tool_name)
   then self.register(self.my_function)
4. Done — the tool is auto-discovered at startup.

No changes needed in any other file.
"""

import os
from agno.tools import Toolkit
from agno.utils.log import logger


class BaseTool(Toolkit):
    """
    Parent class for every FinAgent tool.

    Subclass it, set tool_name / tool_description, register your methods
    in __init__, and the auto-discovery in tools/__init__.py will pick it up.
    """

    # ── Identity (override in each subclass) ────────────────────────────
    tool_name: str = ""
    tool_description: str = ""
    tool_version: str = "1.0.0"

    # ════════════════════════════════════════════════════════════════════
    # 在此处添加新的 API Key 环境变量（所有子工具可继承使用）
    #
    # ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    # FRED_API_KEY          = os.getenv("FRED_API_KEY", "")
    # BLOOMBERG_API_KEY     = os.getenv("BLOOMBERG_API_KEY", "")
    # POLYGON_API_KEY       = os.getenv("POLYGON_API_KEY", "")
    # FINNHUB_API_KEY       = os.getenv("FINNHUB_API_KEY", "")
    # QUANDL_API_KEY        = os.getenv("QUANDL_API_KEY", "")
    #
    # 使用方式示例（在子类 __init__ 里）:
    #   self.api_key = BaseTool.ALPHA_VANTAGE_API_KEY
    #   if not self.api_key:
    #       logger.warning("ALPHA_VANTAGE_API_KEY not set in .env")
    # ════════════════════════════════════════════════════════════════════

    def get_tool_info(self) -> dict:
        """Returns metadata about this tool (used by the tool registry UI)."""
        return {
            "name": self.tool_name or self.__class__.__name__,
            "description": self.tool_description,
            "version": self.tool_version,
            "functions": [fn.__name__ for fn in self.functions.values()],
        }

    def _safe_call(self, fn, *args, **kwargs):
        """
        Wraps any tool function call with structured error handling.
        Use this inside tool methods for consistent error messages.
        """
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.error(f"[{self.tool_name}] Error: {e}")
            return f"[{self.tool_name}] Error: {e}"
