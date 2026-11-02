"""The ``dpbench`` command: ``dpbench dashboard`` opens the interface.

``dpbench run`` runs an experiment described by a YAML file: the dashboard uses it to run
the experiments built in its form in a background process.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

from dpbench import __version__

DEFAULT_RUN_DIRS = ("runs",)


def cmd_run(args: argparse.Namespace) -> int:
    from dpbench.core.config import load_config
    from dpbench.experiments.runner import plan_jobs, run_experiment

    config = load_config(args.config, args.overrides)
    jobs = plan_jobs(config)
    if args.dry_run:
        print(f"Configuration {config.name!r} is valid: {len(jobs)} trainings.")
        for job in jobs:
            print(f"  {job.index:>3}. {job.run_id}")
        return 0
    result = run_experiment(config, args.output_dir, run_dir=args.run_dir)
    failed = int((result.results["status"] != "ok").sum()) if not result.results.empty else 0
    return 1 if failed == len(result.results) and failed else 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    import subprocess

    app = Path(__file__).resolve().parent / "dashboard" / "app.py"
    dirs = args.runs_dir or list(DEFAULT_RUN_DIRS)
    env = dict(os.environ)
    env["DPBENCH_RUNS_DIRS"] = os.pathsep.join(str(Path(d).resolve()) for d in dirs)
    env["DPBENCH_STARTED_AT"] = str(time.time())  # the dashboard lists only the runs launched from now on
    if args.plugins:
        env["DPBENCH_PLUGINS"] = os.pathsep.join(
            str(Path(p).resolve()) if p.endswith(".py") else p for p in args.plugins
        )
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
    print(f"DP-Bench dashboard on {url} (runs from: {', '.join(dirs)}). Stop with Ctrl+C.")
    if not args.no_browser:
        threading.Timer(2.5, lambda: webbrowser.open(url)).start()
    try:
        return subprocess.call(command, env=env)
    except KeyboardInterrupt:
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dpbench",
        description="DP-Bench: benchmark differentially private algorithms (machine learning and queries).",
    )
    parser.add_argument("--version", action="version", version=f"dpbench {__version__}")
    parser.add_argument(
        "--workspace",
        metavar="DIR",
        help="workspace folder of the added datasets and algorithms (default: ./workspace)",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = commands.add_parser("run", help="run an experiment described by a YAML file (used by the dashboard)")
    p.add_argument("config", help="path of the YAML configuration")
    p.add_argument("overrides", nargs="*", help="dot-list overrides, e.g. training.epochs=2 seeds=[0]")
    p.add_argument("-o", "--output-dir", help="parent directory of the run (default: output_dir of the config)")
    p.add_argument("--run-dir", help="exact directory of the run (must not exist or be empty)")
    p.add_argument("--dry-run", action="store_true", help="validate the configuration and list the trainings")
    p.set_defaults(func=cmd_run)

    p = commands.add_parser("dashboard", help="open the interactive dashboard (Streamlit)")
    p.add_argument("--runs-dir", action="append", help="directory where the runs are written (default: runs)")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    p.add_argument("--plugins", nargs="*", help="external plugin modules or .py files to load in the dashboard")
    p.set_defaults(func=cmd_dashboard)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # never crash on a console that cannot print "ε"
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    from dpbench.core.config import ConfigError
    from dpbench.workspace import WORKSPACE_ENV

    if args.workspace:  # also seen by the dashboard and by experiments run in the background
        os.environ[WORKSPACE_ENV] = str(Path(args.workspace).expanduser().resolve())

    try:
        return int(args.func(args) or 0)
    except (ConfigError, FileNotFoundError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
