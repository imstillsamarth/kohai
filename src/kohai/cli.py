"""Command line entry point. `kohai` with no arguments opens the TUI."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from . import __version__

console = Console()


# ---------------------------------------------------------------------------
# scan / download (v0.1)
# ---------------------------------------------------------------------------


def _cmd_scan(as_json: bool) -> int:
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

    console.print(Panel(render.system_panel(info), title="Your machine", expand=False))
    console.print(Panel(render.advice_panel(advice), title="Tofu's advice", expand=False))
    console.print(render.models_table(advice))
    return 0


def _download_with_progress(model) -> bool:
    from rich.progress import DownloadColumn, TransferSpeedColumn

    from .download import DownloadError, download

    with Progress(
        "[progress.description]{task.description}", BarColumn(), DownloadColumn(), TransferSpeedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Downloading {model.name}", total=None)
        try:
            path = download(model, lambda d, t: progress.update(task, completed=d, total=t or None))
        except DownloadError as e:
            console.print(f"[red]{e}[/]")
            return False
    console.print(f"[green]✔[/] {model.name} → {path}")
    return True


def _cmd_download(model_id: str) -> int:
    from .catalog import load_catalog

    model = next((m for m in load_catalog() if m.id == model_id), None)
    if model is None:
        console.print(f"[red]Unknown model id {model_id!r}.[/] Run `kohai scan` to list ids.")
        return 2
    return 0 if _download_with_progress(model) else 1


# ---------------------------------------------------------------------------
# teachers / topics / runs
# ---------------------------------------------------------------------------


def _cmd_teachers(test: bool) -> int:
    from .teachers import API_PRESETS, INSTALLED, MISSING, READY, TeacherError, all_teachers

    t = Table(title="Teachers (senpai)", header_style="bold")
    t.add_column("spec")
    t.add_column("name")
    t.add_column("status")
    t.add_column("notes", style="dim")
    colors = {READY: "green", INSTALLED: "cyan", MISSING: "red"}
    for teacher in all_teachers():
        status, note = teacher.status()
        if test and status in (READY, INSTALLED):
            try:
                teacher.complete("Reply with just: OK", timeout=120)
                status, note = READY, "test call worked"
            except TeacherError as e:
                status, note = "failed", str(e)[:120]
        spec = f"{teacher.id}:<model>" if teacher.id in API_PRESETS else teacher.id
        t.add_row(spec, teacher.name, f"[{colors.get(status, 'yellow')}]{status}[/]", note)
    console.print(t)
    if not test:
        console.print("[dim]Run `kohai teachers --test` to make a tiny real call to each installed one.[/]")
    return 0


def _cmd_topics() -> int:
    from .topics import all_topics

    t = Table(title="Topic packs", header_style="bold")
    t.add_column("id")
    t.add_column("name")
    t.add_column("what the kohai learns")
    for topic in all_topics():
        t.add_row(topic.id, topic.name, topic.description)
    console.print(t)
    return 0


def _fmt_pct(x: float | None) -> str:
    return "–" if x is None else f"{x * 100:.0f}%"


def _cmd_runs() -> int:
    from .runs import list_runs

    runs = list_runs()
    if not runs:
        console.print("No runs yet. Start one with [b]kohai gen[/b].")
        return 0
    t = Table(title="Runs", header_style="bold")
    for col in ("id", "topic", "student", "teacher", "stage", "data", "base", "trained"):
        t.add_column(col)
    for r in runs:
        m = r.metrics()
        ev = m.get("eval", {})
        best = ev.get("rl") or ev.get("sft")
        data = m.get("data", {})
        t.add_row(
            r.id, r.topic, r.model_id, r.teacher, r.stage,
            f"{data.get('train', 0)}/{data.get('test', 0)}" if data else "–",
            _fmt_pct(ev.get("base", {}).get("solved")),
            _fmt_pct(best.get("solved") if best else None),
        )
    console.print(t)
    return 0


# ---------------------------------------------------------------------------
# gen / train / eval / arena
# ---------------------------------------------------------------------------


def _cmd_gen(topic_id: str, teacher_spec: str, model_id: str | None, n: int, workers: int,
             resume: str | None = None) -> int:
    from .advisor import advise
    from .catalog import load_catalog
    from .sysinfo import scan
    from .teachers import TeacherError, get_teacher
    from .topics import get_topic

    if resume:
        from .runs import load_run

        run = load_run(resume)
        topic, teacher = get_topic(run.topic), get_teacher(teacher_spec or run.teacher)
        console.print(f"[bold]Resuming[/] {run.id} · {len(run.read_jsonl('pool.jsonl'))} tasks already saved")
        return _gen_into(run, topic, teacher, n, workers)

    topic = get_topic(topic_id)
    teacher = get_teacher(teacher_spec or "claude")
    catalog = load_catalog()
    advice = advise(scan(), catalog)
    if model_id is None:
        if advice.recommended is None:
            console.print("[red]No recommended model for this machine; pass --model.[/]")
            return 2
        model_id = advice.recommended.model.id
    verdict = advice.verdict(model_id)
    if verdict is None:
        console.print(f"[red]Unknown model {model_id!r}.[/] Run `kohai scan` to list ids.")
        return 2
    settings = advice.settings_for(verdict)
    if not settings:
        console.print(f"[yellow]{verdict.model.name} can't be trained on this machine.[/]")
        return 2

    from .pipeline import create_run

    run = create_run(topic, teacher, verdict, settings)
    console.print(
        f"[bold]Run[/] {run.id}\n  topic [b]{topic.name}[/] · senpai [b]{teacher.name}[/] · "
        f"kohai [b]{verdict.model.name}[/] · {settings['method']}"
    )

    return _gen_into(run, topic, teacher, n, workers)


def _gen_into(run, topic, teacher, n: int, workers: int) -> int:
    from .generate import generate
    from .teachers import TeacherError

    with Progress(
        TextColumn("[bold]Senpai writing tasks"), BarColumn(), MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[info]}"), TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("gen", total=n, info="")

        def on_event(e: dict) -> None:
            if e["event"] == "gen":
                progress.update(
                    task, completed=min(e["accepted"], e["target"]), total=e["target"],
                    info=f"calls {e['calls']} · rejected {e['rejected']} · dupes {e['duplicates']} · errors {e['errors']}",
                )
            elif e["event"] == "teacher_error":
                progress.console.print(f"[yellow]teacher error:[/] {e['error'][:200]}")

        try:
            result = generate(run, topic, teacher, n, workers=workers, emit=on_event)
        except TeacherError as e:
            run.log_event({"event": "stage_failed", "stage": "gen", "error": str(e)})
            progress.stop()
            console.print(f"[red]{e}[/]")
            console.print(f"Resume with: [b]kohai gen --resume {run.id}[/]")
            return 1
        finally:
            run.set_active(None)
    console.print(
        f"[green]✔[/] {result['train']} training + {result['test']} held-out tasks "
        f"([dim]{result['rejected']} rejected by the checker[/])"
    )
    console.print(f"Next: [b]kohai train {run.id}[/]")
    return 0


def _stage_progress(run, stage: str, label: str, extra: list[str] | None = None) -> dict | None:
    from .pipeline import StageError, stream_stage

    last: dict | None = None
    with Progress(
        TextColumn(f"[bold]{label}"), BarColumn(), MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[info]}"), TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task(stage, total=None, info="loading model…")
        try:
            for e in stream_stage(run, stage, extra):
                kind = e.get("event")
                if kind == "start":
                    progress.update(task, total=e.get("total"), info="")
                elif kind == "step":
                    info = []
                    if "loss" in e and stage == "sft":
                        info.append(f"loss {e['loss']:.3f}")
                    if "reward" in e:
                        info.append(f"reward {e['reward']:.2f} · solved {e['solved'] * 100:.0f}%")
                    if "variant" in e:
                        info.append(e["variant"])
                    if e.get("vram_gb"):
                        info.append(f"vram {e['vram_gb']:.1f}G")
                    progress.update(task, completed=e["step"], total=e["total"], info=" · ".join(info))
                elif kind == "result":
                    progress.console.print(
                        f"  {e['variant']:>4}: solved [b]{e['solved'] * 100:.0f}%[/] · score {e['score'] * 100:.0f}%"
                    )
                last = e
        except StageError as err:
            progress.stop()
            console.print(f"[red]{err}[/]")
            return None
    return last


def _print_scoreboard(run) -> None:
    m = run.metrics()
    ev, arena = m.get("eval", {}), m.get("arena", {})
    if not ev and not arena:
        return
    t = Table(title=f"Scoreboard · {run.topic} · {len(run.read_jsonl('test.jsonl'))} held-out tasks",
              header_style="bold")
    t.add_column("model")
    t.add_column("solved", justify="right")
    t.add_column("partial score", justify="right")
    names = {"base": f"{run.model_id} (before)", "sft": f"{run.model_id} + SFT", "rl": f"{run.model_id} + SFT + RL"}
    rows = [(names[k], v) for k, v in ev.items()]
    for spec, v in arena.items():
        label = v.get("name", spec) + (" [dim](the teacher)[/]" if v.get("is_teacher") else "")
        rows.append((label, v))
    best = max(v["solved"] for _, v in rows)
    for name, v in rows:
        style = "bold green" if v["solved"] == best else ""
        t.add_row(name, f"[{style}]{v['solved'] * 100:.0f}%[/]" if style else f"{v['solved'] * 100:.0f}%",
                  f"{v['score'] * 100:.0f}%")
    console.print(t)


def _cmd_train(run_id: str, rl: bool, rl_steps: int, epochs: int | None) -> int:
    from .runs import load_run

    from . import state
    from .catalog import load_catalog

    run = load_run(run_id)
    if state.downloaded_path(run.model_id) is None:
        model = next((m for m in load_catalog() if m.id == run.model_id), None)
        if model and not _download_with_progress(model):
            return 1
    console.print(f"[bold]Training[/] {run.id} · {run.settings.get('method')}")
    console.print("[dim]Watch it live in the TUI: run `kohai`, then press w[/]")
    extra = ["--epochs", str(epochs)] if epochs else []
    try:
        if _stage_progress(run, "sft", "1/3 SFT: learning from senpai", extra) is None:
            return 1
        if rl and _stage_progress(run, "rl", "2/3 RL: practicing with rewards", ["--steps", str(rl_steps)]) is None:
            return 1
        if _stage_progress(run, "eval", "3/3 Eval: held-out test") is None:
            return 1
    finally:
        run.set_active(None)
    _print_scoreboard(load_run(run.id))
    console.print(f"Compare with a big model: [b]kohai arena {run.id} --vs opencode[/]")
    return 0


def _cmd_eval(run_id: str) -> int:
    from .runs import load_run

    run = load_run(run_id)
    if _stage_progress(run, "eval", "Eval: held-out test") is None:
        return 1
    _print_scoreboard(load_run(run.id))
    return 0


def _cmd_arena(run_id: str, vs: str, workers: int) -> int:
    from .pipeline import arena
    from .runs import load_run
    from .teachers import TeacherError, get_teacher
    from .topics import get_topic

    run = load_run(run_id)
    opponent = get_teacher(vs)
    if opponent.id == run.teacher:
        console.print("[yellow]Note: this opponent is the teacher that wrote the tests, so it has a home advantage.[/]")
    with Progress(
        TextColumn(f"[bold]Arena vs {opponent.name}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("arena", total=None)
        try:
            arena(run, get_topic(run.topic), opponent, workers=workers,
                  emit=lambda e: progress.update(task, completed=e["step"], total=e["total"]))
        except TeacherError as e:
            console.print(f"[red]{e}[/]")
            return 1
    _print_scoreboard(load_run(run.id))
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kohai", description="Your tiny model's senpai is a big one. Run with no command for the TUI."
    )
    parser.add_argument("--version", action="version", version=f"kohai {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("scan", help="detect hardware and recommend student models")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p = sub.add_parser("download", help="download a student model by id")
    p.add_argument("model", help="model id, e.g. qwen2.5-1.5b")

    p = sub.add_parser("teachers", help="list teachers (senpai) and whether they work")
    p.add_argument("--test", action="store_true", help="make a tiny real call to each installed teacher")
    sub.add_parser("topics", help="list topic packs")
    sub.add_parser("runs", help="list training runs")

    p = sub.add_parser("gen", help="create a run: the senpai writes verified training tasks")
    p.add_argument("--topic", default="regex")
    p.add_argument("--teacher", help="e.g. claude (default), opencode, ollama:qwen2.5:14b")
    p.add_argument("--resume", metavar="RUN", help="continue collecting tasks for an existing run")
    p.add_argument("--model", help="student model id (default: Tofu's pick for your hardware)")
    p.add_argument("-n", type=int, default=300, help="training tasks to collect (plus a held-out split)")
    p.add_argument("--workers", type=int, default=4, help="parallel teacher calls")

    p = sub.add_parser("train", help="SFT → RL → eval for a run")
    p.add_argument("run", nargs="?", default="latest")
    p.add_argument("--no-rl", action="store_true")
    p.add_argument("--rl-steps", type=int, default=80)
    p.add_argument("--epochs", type=int)

    p = sub.add_parser("eval", help="score base vs trained adapters on the held-out set")
    p.add_argument("run", nargs="?", default="latest")

    p = sub.add_parser("arena", help="score a big model on the same held-out set")
    p.add_argument("run", nargs="?", default="latest")
    p.add_argument("--vs", required=True, help="opponent, any teacher spec")
    p.add_argument("--workers", type=int, default=4)

    args = parser.parse_args(argv)
    try:
        if args.cmd == "scan":
            return _cmd_scan(args.json)
        if args.cmd == "download":
            return _cmd_download(args.model)
        if args.cmd == "teachers":
            return _cmd_teachers(args.test)
        if args.cmd == "topics":
            return _cmd_topics()
        if args.cmd == "runs":
            return _cmd_runs()
        if args.cmd == "gen":
            return _cmd_gen(args.topic, args.teacher, args.model, args.n, args.workers, args.resume)
        if args.cmd == "train":
            return _cmd_train(args.run, not args.no_rl, args.rl_steps, args.epochs)
        if args.cmd == "eval":
            return _cmd_eval(args.run)
        if args.cmd == "arena":
            return _cmd_arena(args.run, args.vs, args.workers)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]{e}[/]")
        return 2

    from .tui import KohaiApp

    KohaiApp().run()
    return 0
