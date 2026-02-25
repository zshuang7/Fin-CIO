"""
tools/__init__.py — Auto-discovery registry for all BaseTool subclasses.

HOW IT WORKS:
─────────────
On import this module scans every .py file in the tools/ directory,
finds classes that extend BaseTool, and registers them in _TOOL_REGISTRY.

ADDING A NEW TOOL:
──────────────────
1. Drop  tools/my_source.py  into the folder.
2. Define  class MySource(BaseTool):  with tool_name set.
3. Restart the app — the registry auto-detects it.

No edits to this file or any other file needed.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path

from tools.base_tool import BaseTool

# ── Private registry ──────────────────────────────────────────────────────────
_TOOL_REGISTRY: dict[str, type[BaseTool]] = {}

# Files that define infrastructure (not data tools)
_SKIP = {"base_tool"}


def _discover() -> None:
    tools_dir = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(tools_dir)]):
        if module_info.name in _SKIP:
            continue
        try:
            mod = importlib.import_module(f"tools.{module_info.name}")
            for _, cls in inspect.getmembers(mod, inspect.isclass):
                if (
                    issubclass(cls, BaseTool)
                    and cls is not BaseTool
                    and cls.__module__ == mod.__name__  # only classes defined IN this file
                ):
                    key = cls.tool_name or cls.__name__
                    _TOOL_REGISTRY[key] = cls
        except Exception as exc:
            import warnings
            warnings.warn(f"[tools] Could not load '{module_info.name}': {exc}")


_discover()


# ── Public API ────────────────────────────────────────────────────────────────

def get_tool_registry() -> dict[str, type[BaseTool]]:
    """Returns a copy of the full {tool_name: ToolClass} registry."""
    return dict(_TOOL_REGISTRY)


def get_tool_names() -> list[str]:
    """Returns a sorted list of all registered tool names."""
    return sorted(_TOOL_REGISTRY.keys())


def get_tool_instance(tool_name: str, **kwargs) -> BaseTool:
    """
    Instantiate a tool by name.  kwargs are forwarded to the constructor.
    Raises KeyError if the tool name is not registered.
    """
    if tool_name not in _TOOL_REGISTRY:
        raise KeyError(
            f"Tool '{tool_name}' not found. "
            f"Available: {get_tool_names()}"
        )
    return _TOOL_REGISTRY[tool_name](**kwargs)


def get_all_instances(**kwargs) -> list[BaseTool]:
    """Instantiate every registered tool.  kwargs forwarded to all constructors."""
    return [cls(**kwargs) for cls in _TOOL_REGISTRY.values()]


# ── Convenience re-exports ────────────────────────────────────────────────────
from tools.finance_engine import FinanceEngine   # noqa: E402
from tools.news_engine import NewsEngine          # noqa: E402
from tools.report_engine import ReportEngine      # noqa: E402

__all__ = [
    "BaseTool",
    "FinanceEngine",
    "NewsEngine",
    "ReportEngine",
    "get_tool_registry",
    "get_tool_names",
    "get_tool_instance",
    "get_all_instances",
]
