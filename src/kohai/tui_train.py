"""Training screen + runs list.

TrainScreen has two modes:
- new run:  pick topic + senpai, then watch data → SFT → RL → exam live, with Tofu
- watch:    open an existing run (started here or from the CLI); its events.jsonl is replayed
            and followed live. Paused/unfinished runs can be continued from here.
"""

from __future__ import annotations

import time

from rich.markup import escape
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Input, Label, ProgressBar, RichLog, Select, Static

from . import render, state
from .advisor import Advice, Verdict
from .mascot import Mascot, SpeechBubble
from .runs import Run, list_runs, load_run
from .teachers import CLI_TEACHERS, INSTALLED, READY, get_teacher
from .topics import all_topics, get_topic

STAGES = [
    ("gen", "Senpai writes tasks"),
    ("sft", "SFT: learn from senpai"),
    ("rl", "RL: practice with rewards"),
    ("eval", "Exam: held-out tasks"),
]
PENDING, RUNNING, DONE, FAILED, SKIPPED = "pending", "running", "done", "failed", "skipped"
ICONS = {
    PENDING: ("○", "#6B6F80"),
    RUNNING: ("◉", "#7FD1B9"),
    DONE: ("✔", "#7BD88F"),
    FAILED: ("✕", "#F7768E"),
    SKIPPED: ("–", "#6B6F80"),
}

HERO_CSS = """
    {screen} #hero {{ height: 12; padding: 1 2 0 2; }}
    {screen} Mascot {{ margin: 0 2 0 0; }}
    {screen} #bubble-wrap {{ width: 1fr; height: 10; align: left middle; }}
    {screen} SpeechBubble {{
        width: 1fr; max-width: 80; height: auto;
        border: round $primary 60%; border-title-color: $primary; border-title-style: bold;
        padding: 0 1;
    }}
"""


def _pct(x) -> str:
    return "–" if x is None else f"{x * 100:.0f}%"


def _teacher_name(spec: str) -> str:
    try:
        return get_teacher(spec).name
    except ValueError:
        return spec


class TrainScreen(Screen):
    DEFAULT_CSS = HERO_CSS.format(screen="TrainScreen") + """
    TrainScreen { background: $background; }
    TrainScreen #tbody { height: 1fr; padding: 0 2; }
    TrainScreen #setup {
        width: 44; border: round $panel-lighten-2; border-title-color: $foreground; padding: 0 1;
    }
    TrainScreen #form { height: 1fr; }
    TrainScreen #form Label { color: $text-muted; text-style: bold; margin-top: 1; }
    TrainScreen #form Select, TrainScreen #form Input { width: 100%; }
    TrainScreen #custom { margin-top: 1; }
    TrainScreen #numbers { height: auto; }
    TrainScreen #numbers > Vertical { width: 1fr; height: auto; }
    TrainScreen #numbers > Vertical:first-child { margin-right: 2; }
    TrainScreen #start { dock: bottom; width: 100%; margin-bottom: 1; }
    TrainScreen #pipeline {
        width: 1fr; margin-left: 2; border: round $panel-lighten-2;
        border-title-color: $foreground; padding: 0 1;
    }
    TrainScreen .stage { height: 1; margin-top: 1; }
    TrainScreen .stage-bar { margin-left: 2; display: none; }
    TrainScreen .stage-bar.shown { display: block; }
    TrainScreen #scoreboard { height: auto; margin-top: 1; }
    TrainScreen #log { height: 1fr; min-height: 4; margin-top: 1; background: $surface; border: none; }
    """

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("c", "cancel", "Cancel training"),
    ]

    def __init__(self, verdict: Verdict, advice: Advice, run: Run | None = None) -> None:
        super().__init__()
        self.verdict = verdict
        self.advice = advice
        self.watch = run  # watch mode when set
        self.running = False  # a pipeline started from THIS screen is running
        self.cancelled = False
        self._proc = None
        self._status = {k: PENDING for k, _ in STAGES}
        self._info = {k: "" for k, _ in STAGES}
        self._last_say = 0.0
        self._offset = 0
        self._was_live = False

    # ---- layout ---------------------------------------------------------

    def compose(self) -> ComposeResult:
        m = self.verdict.model
        settings = self.watch.settings if self.watch else self.advice.settings_for(self.verdict)
        with Horizontal(id="hero"):
            yield Mascot(id="mascot")
            with Vertical(id="bubble-wrap"):
                yield SpeechBubble(id="bubble")
        with Horizontal(id="tbody"):
            with Vertical(id="setup"):
                label = "▶  Continue training" if self.watch else "🥋  Start training"
                yield Button(label, id="start", variant="primary")
                with VerticalScroll(id="form"):
                    yield Label("KOHAI (STUDENT)")
                    yield Static(
                        Text.assemble(
                            (render.short_name(m.name), "bold #7FD1B9"),
                            (f"\n{settings.get('method', '')} · {m.params_label}", "dim"),
                        )
                    )
                    if self.watch:
                        yield from self._compose_watch_form()
                    else:
                        yield from self._compose_new_form()
            with Vertical(id="pipeline"):
                for key, _ in STAGES:
                    yield Static(id=f"st-{key}", classes="stage")
                    yield ProgressBar(id=f"pb-{key}", classes="stage-bar", show_eta=False)
                yield Static(id="scoreboard")
                yield RichLog(id="log", markup=True, wrap=True, max_lines=500)
        yield Footer()

    def _compose_new_form(self) -> ComposeResult:
        yield Label("TOPIC")
        yield Select([(t.name, t.id) for t in all_topics()], value="regex", allow_blank=False,
                     id="topic", compact=True)
        yield Label("SENPAI (TEACHER)")
        yield Select(self._teacher_options(), allow_blank=False, id="teacher", compact=True)
        yield Input(placeholder="or spec: ollama:qwen2.5:14b", id="custom", compact=True)
        with Horizontal(id="numbers"):
            with Vertical():
                yield Label("TASKS")
                yield Input("300", type="integer", id="n", compact=True)
            with Vertical():
                yield Label("RL STEPS")
                yield Input("80", type="integer", id="rl-steps", compact=True)
        yield Static("[dim]RL steps 0 = skip RL[/]")

    def _compose_watch_form(self) -> ComposeResult:
        r = self.watch
        assert r
        yield Label("RUN")
        yield Static(f"[dim]{escape(r.id)}[/]")
        yield Label("TOPIC")
        yield Static(get_topic(r.topic).name)
        yield Label("SENPAI (TEACHER)")
        yield Static(escape(_teacher_name(r.teacher)))
        yield Label("RL STEPS (IF CONTINUING)")
        yield Input("80", type="integer", id="rl-steps", compact=True)

    def _teacher_options(self) -> list[tuple[str, str]]:
        opts = []
        for tid in CLI_TEACHERS:
            t = get_teacher(tid)
            status, _ = t.status()
            if status in (READY, INSTALLED):
                opts.append((t.name, tid))
        return opts or [("(none found: use a spec below)", "")]

    def on_mount(self) -> None:
        self.query_one("#bubble").border_title = "Tofu"
        self.query_one("#setup").border_title = "Run" if self.watch else "Setup"
        self.query_one("#pipeline").border_title = "Pipeline"
        for key, _ in STAGES:
            self._render_stage(key)
        if self.watch:
            self._attach_run()
            self.set_interval(0.5, self._poll)
            return
        name = escape(render.short_name(self.verdict.model.name))
        self._say(
            f"Training [b]{name}[/b]! Pick a topic and a senpai. The senpai writes tasks, "
            "I check every answer, then your kohai trains on the good ones.",
            "happy",
        )

    # ---- helpers --------------------------------------------------------

    def _say(self, markup: str, mood: str = "idle", instant: bool = False) -> None:
        self.query_one("#mascot", Mascot).mood = mood
        self.query_one("#bubble", SpeechBubble).say(markup, instant=instant)
        self._last_say = time.monotonic()

    def _say_throttled(self, markup: str, mood: str) -> None:
        if time.monotonic() - self._last_say > 1.5:
            self._say(markup, mood, instant=True)

    def _render_stage(self, key: str) -> None:
        icon, color = ICONS[self._status[key]]
        title = dict(STAGES)[key]
        t = Text()
        t.append(f"{icon} ", style=f"bold {color}")
        t.append(title, style="bold" if self._status[key] == RUNNING else "")
        if self._info[key]:
            t.append(f"   {self._info[key]}", style="dim")
        self.query_one(f"#st-{key}", Static).update(t)

    def _set(self, key: str, status: str | None = None, info: str | None = None) -> None:
        if status:
            self._status[key] = status
        if info is not None:
            self._info[key] = info
        self._render_stage(key)

    def _bar(self, key: str, done: float, total: float | None) -> None:
        bar = self.query_one(f"#pb-{key}", ProgressBar)
        bar.add_class("shown")
        bar.update(total=total, progress=done)

    def _log(self, markup: str) -> None:
        self.query_one("#log", RichLog).write(markup)

    def _lock_form(self, locked: bool) -> None:
        for w in self.query("#form Select, #form Input, #start"):
            w.disabled = locked

    # ---- watch mode -----------------------------------------------------

    def _attach_run(self) -> None:
        """Show what's on disk, then replay the event log, then follow it live."""
        run = self.watch
        assert run
        train, test = run.read_jsonl("train.jsonl"), run.read_jsonl("test.jsonl")
        if train:
            self._set("gen", DONE, f"{len(train)} train · {len(test)} held-out")
        elif run.path("pool.jsonl").exists():
            self._set("gen", RUNNING if run.active else PENDING,
                      f"{len(run.read_jsonl('pool.jsonl'))} verified so far")
        for key in ("sft", "rl"):
            if run.path(f"adapter-{key}").exists():
                self._set(key, DONE)
        ev = run.metrics().get("eval", {})
        if ev:
            self._set("eval", DONE)
            self.query_one("#scoreboard", Static).update(self._scoreboard(ev))

        events, self._offset = run.read_events(0)
        for e in events:
            self._apply(e, replay=True)
        self._update_watch_controls()
        live = run.active
        self._was_live = bool(live)
        if live:
            self._say(f"Watching this run live. It's on [b]{escape(str(live.get('stage')))}[/b] right now…",
                      "training" if live.get("stage") in ("sft", "rl") else "working")
        elif ev:
            before = ev.get("base", {}).get("solved")
            after = (ev.get("rl") or ev.get("sft") or {}).get("solved")
            self._say(f"This run is finished: [b]{_pct(before)}[/b] before → "
                      f"[b #7BD88F]{_pct(after)}[/] after training.", "happy")
        else:
            self._say("This run is paused. Press [b]Continue training[/b] to pick it up where it stopped.",
                      "thinking")

    def _update_watch_controls(self) -> None:
        if not self.watch:
            return
        btn = self.query_one("#start", Button)
        live = self.watch.active
        done = bool(self.watch.metrics().get("eval")) and self.watch.path("adapter-sft").exists()
        if self.running:
            btn.label, btn.disabled = "Training…", True
        elif live:
            btn.label, btn.disabled = "●  Live: running elsewhere", True
        elif done:
            btn.label, btn.disabled = "✔  Finished", True
        else:
            btn.label, btn.disabled = "▶  Continue training", False

    def _poll(self) -> None:
        if not self.watch or self.running:
            return
        run = self.watch
        events, self._offset = run.read_events(self._offset)
        for e in events:
            self._apply(e)
        live = run.active
        if self._status["gen"] in (PENDING, RUNNING) and not run.path("train.jsonl").exists():
            pool = run.path("pool.jsonl")
            if pool.exists() and not events:
                count = sum(1 for _ in pool.open(encoding="utf-8"))
                self._set("gen", RUNNING if live else PENDING, f"{count} verified so far")
        if self._was_live and not live:
            self._update_watch_controls()
        self._was_live = bool(live)

    def _apply(self, e: dict, replay: bool = False) -> None:
        stage, kind = e.get("stage"), e.get("event")
        if stage == "gen":
            if kind == "stage_start":
                self._set("gen", RUNNING, "asking senpai…")
            elif kind == "stage_failed":
                self._set("gen", FAILED, "stopped")
                self._log(f"[red]{escape(str(e.get('error')))}[/]")
            else:
                if self._status["gen"] in (PENDING, FAILED):
                    self._set("gen", RUNNING)
                self._on_gen(e)
            return
        if stage not in ("sft", "rl", "eval"):
            return
        if kind == "stage_start":
            self._set(stage, RUNNING, "loading model…")
            self._bar(stage, 0, None)
            if not replay:
                self._stage_intro(stage)
        elif kind == "stage_end":
            self._set(stage, DONE)
            if stage == "eval" and self.watch:
                self._finished(self.watch, celebrate=not replay)
        elif kind == "stage_failed":
            self._set(stage, FAILED)
            self._log(f"[red]{escape(str(e.get('error')))}[/]")
            if not replay:
                self._say(f"The {stage} stage failed. Details are in the log below.", "sad")
        else:
            self._on_stage(stage, e)

    # ---- actions --------------------------------------------------------

    def action_back(self) -> None:
        if self.running:
            self.notify("Training is running. Press c to cancel it first.", severity="warning")
            return
        self.app.pop_screen()

    def action_cancel(self) -> None:
        if not self.running:
            if self.watch and self.watch.active:
                self.notify("This run belongs to another window; stop it there.", severity="warning")
            return
        self.cancelled = True
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._say("Stopping… the current stage will end.", "sad")

    @on(Button.Pressed, "#start")
    def _start(self) -> None:
        if self.running:
            return
        try:
            rl_steps = max(0, int(self.query_one("#rl-steps", Input).value or 0))
        except ValueError:
            self.notify("RL steps must be a number.", severity="error")
            return

        if self.watch:
            run = self.watch
            if run.active:
                self.notify("This run is already being trained elsewhere.", severity="warning")
                return
            topic, teacher = get_topic(run.topic), get_teacher(run.teacher)
            n = run.metrics().get("data", {}).get("train", 200) or 200
        else:
            run = None
            custom = self.query_one("#custom", Input).value.strip()
            spec = custom or self.query_one("#teacher", Select).value
            try:
                teacher = get_teacher(str(spec))
            except ValueError as e:
                self.notify(str(e), severity="error")
                return
            topic = get_topic(str(self.query_one("#topic", Select).value))
            try:
                n = max(30, int(self.query_one("#n", Input).value or 300))
            except ValueError:
                self.notify("Tasks must be a number.", severity="error")
                return
            self.query_one("#log", RichLog).clear()
            self.query_one("#scoreboard", Static).update("")
            for key, _ in STAGES:
                self._set(key, PENDING, "")
                self.query_one(f"#pb-{key}", ProgressBar).remove_class("shown")

        self.running, self.cancelled = True, False
        self._lock_form(True)
        if rl_steps == 0 and self._status["rl"] != DONE:
            self._set("rl", SKIPPED, "skipped")
        self._update_watch_controls()
        self._pipeline(topic, teacher, n, rl_steps, run)

    # ---- the pipeline (worker thread) -------------------------------------

    @work(thread=True, exclusive=True, group="pipeline")
    def _pipeline(self, topic, teacher, n: int, rl_steps: int, run: Run | None = None) -> None:
        from .download import download
        from .generate import generate
        from .pipeline import StageError, create_run, stream_stage

        call = self.app.call_from_thread
        try:
            if run is None:
                settings = self.advice.settings_for(self.verdict)
                run = create_run(topic, teacher, self.verdict, settings)
                call(self._log, f"[b]run[/b] {run.id}")
            else:
                call(self._log, f"[b]continuing[/b] {run.id}")

            if state.downloaded_path(self.verdict.model.id) is None:
                call(self._set, "gen", RUNNING, "downloading the kohai first…")
                call(self._say, "First, fetching your kohai's weights…", "working")
                download(self.verdict.model, lambda d, t: call(self._bar, "gen", d, t or None))

            if not run.path("train.jsonl").exists():
                call(self._set, "gen", RUNNING, "asking senpai…")
                call(self._bar, "gen", 0, None)
                call(self._say, f"Asking [b]{escape(teacher.name)}[/b] to write {n} {topic.name.lower()} tasks…",
                     "working")
                generate(run, topic, teacher, n, emit=lambda e: call(self._on_gen, e))
                if self.cancelled:
                    raise StageError("cancelled")

            stages = []
            if not run.path("adapter-sft").exists():
                stages.append(("sft", []))
            if rl_steps and not run.path("adapter-rl").exists():
                stages.append(("rl", ["--steps", str(rl_steps)]))
            stages.append(("eval", []))
            for key, extra in stages:
                call(self._set, key, RUNNING, "loading model…")
                call(self._bar, key, 0, None)
                call(self._stage_intro, key)
                for e in stream_stage(run, key, extra, on_proc=lambda p: setattr(self, "_proc", p)):
                    call(self._on_stage, key, e)
                if self.cancelled:
                    raise StageError("cancelled")
                call(self._set, key, DONE)
            call(self._finished, run)
        except Exception as e:  # noqa: BLE001
            call(self._failed, str(e))
        finally:
            if run is not None:
                run.set_active(None)

    # ---- event handlers (UI thread) ---------------------------------------

    def _on_gen(self, e: dict) -> None:
        if e["event"] == "gen":
            self._bar("gen", min(e["accepted"], e["target"]), e["target"])
            info = f"{e['accepted']}/{e['target']} verified · {e['rejected']} rejected · {e['calls']} calls"
            if e.get("errors"):
                info += f" · {e['errors']} errors"
            self._set("gen", info=info)
            self._say_throttled(
                f"Senpai is writing… [b]{e['accepted']}[/b] tasks passed my checks, "
                f"{e['rejected']} didn't.",
                "working",
            )
        elif e["event"] == "teacher_error":
            self._log(f"[yellow]senpai error:[/] {escape(e['error'][:200])}")
        elif e["event"] == "gen_done":
            self._set("gen", DONE, f"{e['train']} train · {e['test']} held-out · {e['rejected']} rejected")

    def _stage_intro(self, key: str) -> None:
        lines = {
            "sft": ("Training time! Your kohai is copying the senpai's answers.", "training"),
            "rl": ("Now practice: it tries answers, I score them, good ones get rewarded.", "training"),
            "eval": ("Exam time. Questions it has never seen. Before vs after…", "thinking"),
        }
        self._say(*lines[key])

    def _on_stage(self, key: str, e: dict) -> None:
        kind = e.get("event")
        if kind == "start":
            self._bar(key, 0, e.get("total"))
            self._set(key, RUNNING, info="")
        elif kind == "step":
            self._bar(key, e["step"], e["total"])
            vram = f" · vram {e['vram_gb']:.1f}G" if e.get("vram_gb") else ""
            if key == "sft":
                self._set(key, info=f"step {e['step']}/{e['total']} · loss {e['loss']:.3f}{vram}")
                self._say_throttled(f"Learning… loss [b]{e['loss']:.3f}[/b] (lower is better)", "training")
            elif key == "rl":
                self._set(key, info=f"step {e['step']}/{e['total']} · reward {e['reward']:.2f} · "
                                    f"solved {e['solved'] * 100:.0f}%{vram}")
                self._say_throttled(
                    f"Practicing… solving [b]{e['solved'] * 100:.0f}%[/b] of attempts "
                    f"(reward {e['reward']:.2f})",
                    "training",
                )
                if e["step"] % 10 == 0 and e.get("sample"):
                    self._log(f"[dim]rl sample:[/] {escape(e['sample'])}")
            elif key == "eval":
                self._set(key, info=f"{e['step']}/{e['total']} · {e.get('variant', '')}")
                self._say_throttled(f"Exam in progress… grading the [b]{e.get('variant', '')}[/b] model", "thinking")
        elif kind == "result":
            self._log(f"exam [b]{e['variant']}[/b]: solved {e['solved'] * 100:.0f}% · score {e['score'] * 100:.0f}%")
        elif kind == "error":
            self._log(f"[red]{escape(str(e.get('error')))}[/]")

    def _finished(self, run, celebrate: bool = True) -> None:
        self.running = False
        self._lock_form(False)
        run = load_run(run.id)
        if self.watch:
            self.watch = run
            _, self._offset = run.read_events(self._offset)  # we already showed these
        ev = run.metrics().get("eval", {})
        self.query_one("#scoreboard", Static).update(self._scoreboard(ev))
        self._update_watch_controls()
        if not celebrate:
            return
        before = ev.get("base", {}).get("solved", 0)
        after = (ev.get("rl") or ev.get("sft") or {}).get("solved", 0)
        self.query_one("#mascot", Mascot).cheer(40)
        if after > before:
            self._say(
                f"Your kohai went from [b]{before * 100:.0f}%[/b] to [b #7BD88F]{after * 100:.0f}%[/] "
                f"on tasks it never saw! Try `kohai arena --vs opencode` to fight a big model.",
                "happy",
            )
        else:
            self._say(
                f"Done, but no gain ({before * 100:.0f}% → {after * 100:.0f}%). "
                "More tasks or more RL steps usually help.",
                "thinking",
            )

    def _failed(self, message: str) -> None:
        self.running = False
        self._lock_form(False)
        if self.watch:
            _, self._offset = self.watch.read_events(self._offset)
        self._update_watch_controls()
        for key, _ in STAGES:
            if self._status[key] == RUNNING:
                self._set(key, FAILED)
        if self.cancelled:
            self._say("Training cancelled. Your data is saved, so you can pick it up later.", "sad")
            return
        self._say(f"Oof, something broke: {escape(message.splitlines()[0][:160])}", "sad")
        self._log(f"[red]{escape(message)}[/]")

    @staticmethod
    def _scoreboard(ev: dict) -> Table:
        t = Table(title="Exam results (held-out)", title_style="bold", header_style="bold dim", expand=False)
        t.add_column("model")
        t.add_column("solved", justify="right")
        t.add_column("partial", justify="right")
        names = {"base": "before training", "sft": "+ SFT", "rl": "+ SFT + RL"}
        best = max((v["solved"] for v in ev.values()), default=0)
        for k, v in ev.items():
            style = "bold #7BD88F" if v["solved"] == best and k != "base" else ""
            t.add_row(names.get(k, k), Text(f"{v['solved'] * 100:.0f}%", style=style), f"{v['score'] * 100:.0f}%")
        return t


class RunsScreen(Screen):
    """Every training run, live ones marked. Enter opens it in watch mode."""

    DEFAULT_CSS = HERO_CSS.format(screen="RunsScreen") + """
    RunsScreen { background: $background; }
    RunsScreen #runs {
        height: 1fr; margin: 0 2; background: $background;
        border: round $primary 60%; border-title-color: $foreground;
        border-subtitle-color: $text-muted;
    }
    RunsScreen #runs > .datatable--header { background: $background; color: $text-muted; text-style: bold; }
    RunsScreen #runs > .datatable--cursor { background: $primary 22%; }
    """

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("enter", "open", "Open / watch"),
    ]

    def __init__(self, advice: Advice) -> None:
        super().__init__()
        self.advice = advice
        self._ids: list[str] = []

    def compose(self) -> ComposeResult:
        with Horizontal(id="hero"):
            yield Mascot(id="mascot")
            with Vertical(id="bubble-wrap"):
                yield SpeechBubble(id="bubble")
        yield DataTable(id="runs", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#bubble").border_title = "Tofu"
        table = self.query_one("#runs", DataTable)
        table.border_title = "Runs"
        table.border_subtitle = "● live  ·  enter: open / watch"
        for col in ("", "Run", "Topic", "Kohai", "Senpai", "Stage", "Data", "Before", "After"):
            table.add_column(col)
        self._fill()
        self.set_interval(2.0, self._fill)
        runs = list_runs()
        live = [r for r in runs if r.active]
        if not runs:
            self._say("No runs yet. Go back, pick a model and press [b]t[/b] to train one!", "thinking")
        elif live:
            self._say(f"[b]{len(live)}[/b] run(s) training right now. Open one to watch it live!", "training")
        else:
            self._say("Here's every kohai you've trained. Open one to see how it did.", "happy")
        table.focus()

    def _say(self, markup: str, mood: str) -> None:
        self.query_one("#mascot", Mascot).mood = mood
        self.query_one("#bubble", SpeechBubble).say(markup)

    def _fill(self) -> None:
        table = self.query_one("#runs", DataTable)
        keep = table.cursor_row
        runs = list(reversed(list_runs()))
        table.clear()
        self._ids = []
        for r in runs:
            m = r.metrics()
            ev, data = m.get("eval", {}), m.get("data", {})
            live = r.active
            if data:
                data_s = f"{data.get('train', 0)}/{data.get('test', 0)}"
            elif r.path("pool.jsonl").exists():
                data_s = f"{sum(1 for _ in r.path('pool.jsonl').open(encoding='utf-8'))}…"
            else:
                data_s = "–"
            after = (ev.get("rl") or ev.get("sft") or {}).get("solved")
            table.add_row(
                Text("●", style="bold #7FD1B9") if live else Text(""),
                Text(r.id, style="bold" if live else ""),
                r.topic,
                r.model_id,
                _teacher_name(r.teacher),
                Text(str(live["stage"]) if live else r.stage, style="#7FD1B9" if live else "dim"),
                data_s,
                _pct(ev.get("base", {}).get("solved")),
                Text(_pct(after), style="bold #7BD88F" if after is not None else ""),
                key=r.id,
            )
            self._ids.append(r.id)
        if table.row_count:
            table.move_cursor(row=min(keep, table.row_count - 1))

    def action_back(self) -> None:
        self.app.pop_screen()

    @on(DataTable.RowSelected)
    def _selected(self, event: DataTable.RowSelected) -> None:
        self.action_open()

    def action_open(self) -> None:
        table = self.query_one("#runs", DataTable)
        if not self._ids:
            return
        run = load_run(self._ids[table.cursor_row])
        verdict = self.advice.verdict(run.model_id)
        if verdict is None:
            self.notify(f"Model {run.model_id} isn't in the catalog anymore.", severity="error")
            return
        self.app.push_screen(TrainScreen(verdict, self.advice, run=run))
