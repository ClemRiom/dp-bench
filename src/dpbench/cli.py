"""The ``dpbench`` command: ``dpbench dashboard`` opens the interface."""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

from dpbench import __version__


def cmd_dashboard(args: argparse.Namespace) -> int:
    import subprocess

    app = Path(__file__).resolve().parent / "dashboard" / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--theme.base",
        "light",
        "--theme.primaryColor",
        "#2a78d6",
    ]
    url = f"http://localhost:{args.port}"
    print(f"DP-Bench dashboard on {url}. Stop with Ctrl+C.")
    if not args.no_browser:
        threading.Timer(2.5, lambda: webbrowser.open(url)).start()
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dpbench",
        description="DP-Bench: benchmark differentially private algorithms.",
    )
    parser.add_argument("--version", action="version", version=f"dpbench {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = commands.add_parser("dashboard", help="open the interactive dashboard (Streamlit)")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    p.set_defaults(func=cmd_dashboard)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # never crash on a console that cannot print "ε"
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
