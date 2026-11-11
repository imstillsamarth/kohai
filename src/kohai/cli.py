"""Command line entry point. `kohai` with no arguments opens the TUI."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def _cmd_scan(as_json: bool) -> int:
    from rich.console import Console
    from rich.panel import Panel

    from . import render
    from .advisor import advise
    from .catalog import load_catalog
    from .sysinfo import scan

    info = scan()
    advice = advise(info, load_catalog())

    if as_json:
        out = {
            "system": info.to_dict(),
            "backend": advice.backend,
            "budget_gb": advice.budget_gb,
            "train_where": advice.train_where,
            "recommended": advice.recommended and advice.recommended.model.id,
            "stretch": advice.stretch and advice.stretch.model.id,
            "fast": advice.fast and advice.fast.model.id,
            "settings": advice.settings,
            "tips": advice.tips,
            "warnings": advice.warnings,
            "models": [
                {"id": v.model.id, "run": v.run, "train": v.train, "rl": v.rl, **vars(v.est)}
                for v in advice.verdicts
            ],
        }
        json.dump(out, sys.stdout, indent=2)
        print()
        return 0

    console = Console()
    console.print(Panel(render.system_panel(info), title="Your machine", expand=False))
    console.print(Panel(render.advice_panel(advice), title="Tofu's advice", expand=False))
    console.print(render.models_table(advice))
    return 0


def _cmd_download(model_id: str) -> int:
    from rich.console import Console
    from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

    from .catalog import load_catalog
    from .download import DownloadError, download

    console = Console()
    model = next((m for m in load_catalog() if m.id == model_id), None)
    if model is None:
        console.print(f"[red]Unknown model id {model_id!r}.[/] Run `kohai scan` to list ids.")
        return 2

    with Progress(
        "[progress.description]{task.description}", BarColumn(), DownloadColumn(), TransferSpeedColumn()
    ) as progress:
        task = progress.add_task(model.name, total=None)

        def on_progress(done: int, total: int) -> None:
            progress.update(task, completed=done, total=total or None)

        try:
            path = download(model, on_progress)
        except DownloadError as e:
            console.print(f"[red]{e}[/]")
            return 1
    console.print(f"[green]✔ Downloaded to[/] {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kohai", description="Turn a tiny local LLM into a specialist. Run with no command for the TUI."
    )
    parser.add_argument("--version", action="version", version=f"kohai {__version__}")
    sub = parser.add_subparsers(dest="cmd")
    scan_p = sub.add_parser("scan", help="detect hardware and recommend student models")
    scan_p.add_argument("--json", action="store_true", help="machine-readable output")
    dl_p = sub.add_parser("download", help="download a student model by id")
    dl_p.add_argument("model", help="model id, e.g. qwen2.5-1.5b")
    args = parser.parse_args(argv)

    if args.cmd == "scan":
        return _cmd_scan(args.json)
    if args.cmd == "download":
        return _cmd_download(args.model)

    from .tui import KohaiApp

    KohaiApp().run()
    return 0
