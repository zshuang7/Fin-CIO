"""
main.py — Conversational CLI entry point for FinAgent.

Usage:
    py main.py                   # interactive conversation loop
    py main.py "分析 TSLA"        # single question then exit
    py main.py --list-tools      # print registered tools and exit
"""

import sys
import argparse
from dotenv import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FinAgent — AI Investment Research Assistant"
    )
    parser.add_argument(
        "query", nargs="?", default=None,
        help="One-shot query (omit for interactive mode).",
    )
    parser.add_argument(
        "--list-tools", action="store_true",
        help="Print all auto-discovered tools and exit.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def list_tools():
    from tools import get_tool_registry
    registry = get_tool_registry()
    print("\n📦 Registered Tools:")
    print("─" * 40)
    for name, cls in registry.items():
        instance = cls()
        info = instance.get_tool_info()
        fns = ", ".join(info.get("functions", []))
        print(f"  ✅ {name}")
        print(f"     {info.get('description', '')}")
        print(f"     Functions: {fns}")
        print()


def run_once(query: str):
    from agents.team_config import cio_team
    print(f"\n{'─' * 60}")
    print(f"  Query: {query}")
    print(f"{'─' * 60}\n")
    cio_team.print_response(query, stream=False)
    print()


def conversation_loop():
    from agents.team_config import cio_team

    print("\n" + "=" * 60)
    print("  FinAgent CIO — AI Investment Research Assistant")
    print("  Powered by DeepSeek-R1 · LiteLLM · Agno")
    print("=" * 60)
    print("  Ask anything about stocks, macro, or valuation.")
    print("  Examples:")
    print("    分析 TSLA")
    print("    What is Apple's P/E ratio?")
    print("    帮我生成 NVDA 完整报告")
    print("    Compare MSFT and GOOGL")
    print("  Type 'tools' to list available tools.")
    print("  Type 'exit' to quit.\n")

    history: list[str] = []

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if cmd == "tools":
            list_tools()
            continue

        # Build context from recent history
        ctx = "\n".join(history[-6:])
        if ctx:
            full_query = f"[Recent conversation]\n{ctx}\n\n[Current question]\n{user_input}"
        else:
            full_query = user_input

        history.append(f"User: {user_input}")

        print()
        try:
            cio_team.print_response(full_query, stream=False)
            # Capture last response for context (approximate)
            history.append(f"CIO: (response provided above)")
        except Exception as e:
            print(f"Error: {e}")

        print()


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    if args.list_tools:
        list_tools()
        sys.exit(0)

    if args.query:
        run_once(args.query)
    else:
        conversation_loop()
