"""The kohai TUI: Tofu scans your hardware, then helps you pick and download a student."""

from __future__ import annotations

import time

from rich.markup import escape
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.theme import Theme
from textual.widgets import Button, DataTable, Footer, ProgressBar, Static

from . import __version__, render, state
from .advisor import CLOUD, NO, OK, SLOW, TIGHT, Advice, Verdict, advise
from .catalog import load_catalog
from .mascot import Mascot, SpeechBubble
from .sysinfo import SystemInfo, scan

KOHAI_THEME = Theme(
    name="kohai",
    primary="#7FD1B9",
    secondary="#E5484D",
    accent="#F5C76B",
    foreground="#E6E1D6",
    background="#121218",
    surface="#181920",
    panel="#20212B",
    success="#7BD88F",
    warning="#F5C76B",
    error="#F7768E",
    dark=True,
)

HINTS = [
    "Press [b]t[/b] to train the highlighted model on a topic.",
    "Press [b]w[/b] to see your runs, and watch live training.",
    "Press [b]d[/b] to download the highlighted model.",
    "Press [b]f[/b] to show only models that can do RL on this machine.",
    "Small model + narrow topic + RL = beats models 10× bigger.",
    "Big model = senpai. Your small model = kohai. I'm just the sensei.",
    "Debug your dataset on the ✦ fast model first, then train ★.",
    "Teachers and training are coming next. Stay tuned, student.",
]


class KohaiApp(App):
    TITLE = "kohai"

    CSS = """
    Screen { background: $background; }

    #hero { height: 12; padding: 1 2 0 2; }
    Mascot { margin: 0 2 0 0; }
    #bubble-wrap { width: 1fr; height: 10; align: left middle; }
    SpeechBubble {
        width: 1fr;
        max-width: 72;
        height: auto;
        border: round $primary 60%;
        border-title-color: $primary;
        border-title-style: bold;
        padding: 0 1;
    }
    #syscard {
        width: 46;
        height: auto;
        margin: 1 0 0 2;
        border: round $secondary 50%;
        border-title-color: $secondary;
        border-title-style: bold;
        padding: 0 1;
    }

    #body { height: 1fr; padding: 0 2; }
    #models {
        width: 1fr;
        background: $background;
        border: round $panel-lighten-2;
        border-title-color: $foreground;
        border-subtitle-color: $text-muted;
    }
    #models:focus { border: round $primary 70%; }
    #models > .datatable--header { background: $background; color: $text-muted; text-style: bold; }
    #models > .datatable--cursor { background: $primary 22%; }
    #models > .datatable--hover { background: $panel; }

    #card {
        width: 54;
        margin-left: 2;
        border: round $panel-lighten-2;
        border-title-color: $foreground;
        padding: 0 1;
    }
    #card-body { height: auto; }
    #progress { display: none; margin-top: 1; }
    #progress.active { display: block; }
    #download { width: 100%; margin-top: 1; }

    Footer { background: $surface; }

    /* small terminals */
    .narrow #syscard { display: none; }
    .narrow #card { width: 44; }
    .narrow #hero { padding: 1 1 0 1; }
    .narrow #body { padding: 0 1; }
    """

    BINDINGS = [
        Binding("t", "train", "Train"),
        Binding("w", "runs", "Runs"),
        Binding("d", "download", "Download"),
        Binding("f", "toggle_fit", "RL-ready only"),
        Binding("r", "rescan", "Rescan"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.catalog = load_catalog()
        self.info: SystemInfo | None = None
        self.advice: Advice | None = None
        self.selected: Verdict | None = None
        self.fits_only = False
        self.downloading: str | None = None  # model id being downloaded
        self._ready_at = 0.0
        self._last_said = 0.0
        self._tips: list[str] = []
        self._tip_i = 0

    # ---- layout ---------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Horizontal(id="hero"):
            yield Mascot(id="mascot")
            with Vertical(id="bubble-wrap"):
                yield SpeechBubble(id="bubble")
            yield Static(id="syscard")
        with Horizontal(id="body"):
            yield DataTable(id="models", cursor_type="row", zebra_stripes=False)
            with VerticalScroll(id="card"):
                yield Static(id="card-body")
                yield ProgressBar(id="progress", show_eta=False)
                yield Button("Download", id="download", variant="primary", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self.register_theme(KOHAI_THEME)
        self.theme = "kohai"
        self.query_one("#bubble").border_title = "Tofu"
        self.query_one("#syscard").border_title = f"kohai v{__version__}"
        self.query_one("#models").border_title = "Students"
        self.query_one("#card").border_title = "Model"
        table = self.query_one("#models", DataTable)
        table.add_column(" ", key="badge", width=1)
        table.add_column("Model", key="model")
        table.add_column("Size", key="size")
        table.add_column("Run", key="run")
        table.add_column("Train", key="train")
        table.add_column("RL", key="rl")
        table.add_column(" ", key="dl", width=1)
        self.set_interval(14, self._next_tip)
        self.action_rescan()

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        # Main-screen actions are hidden and disabled while another screen (training) is open.
        if action in ("train", "runs", "download", "toggle_fit", "rescan") and len(self.screen_stack) > 1:
            return False
        return True

    def on_resize(self, event) -> None:
        self.screen.set_class(event.size.width < 125, "narrow")

    # ---- Tofu -----------------------------------------------------------

    def say(self, markup: str, mood: str = "idle", instant: bool = False) -> None:
        self.query_one("#mascot", Mascot).mood = mood
        self.query_one("#bubble", SpeechBubble).say(markup, instant=instant)
        self._last_said = time.monotonic()

    def _next_tip(self) -> None:
        if self.downloading or not self._tips or time.monotonic() - self._last_said < 12:
            return
        tip = self._tips[self._tip_i % len(self._tips)]
        self._tip_i += 1
        self.say(tip, "idle")

    def _greeting(self, info: SystemInfo, a: Advice) -> str:
        rec = escape(render.short_name(a.recommended.model.name)) if a.recommended else None
        gpu = info.best_gpu
        if a.backend == "cuda" and gpu:
            msg = (
                f"Your [b]{escape(render.short_gpu(gpu.name))}[/b] gives us "
                f"[b]{a.budget_gb:.1f} GB[/b] to train with."
            )
            if rec:
                msg += f" Your kohai: [b #F5C76B]★ {rec}[/]. Fine-tuning [i]and[/i] RL both fit."
            return msg
        if a.backend == "mps":
            return f"Apple Silicon! About [b]{a.budget_gb:.0f} GB[/b] to play with. I'd start with [b]{rec}[/b]."
        return (
            "No CUDA GPU here, so we'll train on a free Colab GPU. "
            f"Best student for that: [b #F5C76B]★ {rec}[/]."
        )

    def _comment(self, v: Verdict) -> tuple[str, str]:
        a = self.advice
        assert a
        name = f"[b]{escape(render.short_name(v.model.name))}[/b]"
        if a.recommended is v:
            if a.train_where == "colab":
                msg, mood = f"{name} is my pick for training on Colab.", "happy"
            else:
                msg, mood = f"{name} is my top pick for you! Fine-tuning and RL fit comfortably.", "happy"
        elif v.rl == OK:
            msg, mood = f"{name} can do fine-tuning [i]and[/i] RL right here. Nice.", "happy"
        elif v.rl == TIGHT:
            msg, mood = f"RL with {name} will be tight. Doable with small batches.", "thinking"
        elif v.train in (OK, TIGHT):
            msg, mood = f"You can fine-tune {name} here, but RL needs a bigger GPU.", "thinking"
        elif v.train == SLOW:
            msg, mood = f"{name} trains on CPU… slowly. Fine for testing the pipeline.", "thinking"
        else:
            msg, mood = f"{name} is too big to train here. That's a Colab job.", "sad"
        if v.model.gated:
            msg += " Heads up: it needs a license click on Hugging Face."
        return msg, mood

    # ---- scanning -------------------------------------------------------

    def action_rescan(self) -> None:
        self.say("Hmm… let me take a look at your machine…", "thinking")
        self.query_one("#syscard", Static).update(Text("scanning…", style="dim"))
        self._scan()

    @work(thread=True, exclusive=True, group="scan")
    def _scan(self) -> None:
        started = time.monotonic()
        info = scan()
        advice = advise(info, self.catalog)
        time.sleep(max(0.0, 1.2 - (time.monotonic() - started)))  # let Tofu think for a beat
        self.call_from_thread(self._show, info, advice)

    def _show(self, info: SystemInfo, advice: Advice) -> None:
        self.info, self.advice = info, advice
        self.query_one("#syscard", Static).update(render.system_card(info, advice))
        self._tips = [escape(t) for t in advice.tips] + [f"⚠ {escape(w)}" for w in advice.warnings] + HINTS
        self._ready_at = time.monotonic()
        self._fill_table()
        self.query_one("#mascot", Mascot).cheer(12)
        self.say(self._greeting(info, advice), "happy")

    def _visible(self) -> list[Verdict]:
        assert self.advice
        if not self.fits_only:
            return self.advice.verdicts
        return [v for v in self.advice.verdicts if v.rl in (OK, TIGHT) or v.train == SLOW]

    def _fill_table(self) -> None:
        if not self.advice:
            return
        a = self.advice
        table = self.query_one("#models", DataTable)
        keep = self.selected.model.id if self.selected else (a.recommended.model.id if a.recommended else None)
        table.clear()
        downloaded = state.downloads()
        focus_row = 0
        for i, v in enumerate(self._visible()):
            m = v.model
            badge = a.badge(m.id)
            if m.id == keep:
                focus_row = i
            name = Text(render.short_name(m.name), style="bold" if badge else "")
            if m.gated:
                name.append(" 🔒", style="dim")
            table.add_row(
                Text(badge, style="bold #F5C76B" if badge == "★" else "#7FD1B9"),
                name,
                Text(m.params_label, style="dim"),
                render.chip(v.run),
                render.chip(v.train),
                render.chip(v.rl),
                Text("✔" if m.id in downloaded else "", style="#7BD88F"),
                key=m.id,
            )
        if table.row_count:
            table.move_cursor(row=focus_row)
            table.focus()
        shown = "RL-ready" if self.fits_only else "all"
        table.border_subtitle = f"★ pick  ◆ stretch  ✦ fast  ·  {shown} ({len(self._visible())})"

    def action_toggle_fit(self) -> None:
        self.fits_only = not self.fits_only
        self._fill_table()
        if self.fits_only:
            self.say("Showing only students that can do [b]RL[/b] here. That's where the magic is.", "happy")
        else:
            self.say("Showing every student again.", "idle")

    # ---- selection ------------------------------------------------------

    @on(DataTable.RowHighlighted)
    def _highlighted(self, event: DataTable.RowHighlighted) -> None:
        if not self.advice or event.row_key.value is None:
            return
        v = self.advice.verdict(event.row_key.value)
        if v is None:
            return
        changed = self.selected is not v
        self.selected = v
        self._update_card()
        # Comment on the user's choice, but not on the initial auto-selection.
        if changed and not self.downloading and time.monotonic() - self._ready_at > 0.8:
            self.say(*self._comment(v))

    def _update_card(self) -> None:
        if not (self.selected and self.advice):
            return
        v = self.selected
        downloaded = state.downloaded_path(v.model.id) is not None
        self.query_one("#card-body", Static).update(render.model_card(v, self.advice, downloaded))
        btn = self.query_one("#download", Button)
        if self.downloading == v.model.id:
            btn.label, btn.variant, btn.disabled = "Downloading…", "default", True
        elif downloaded:
            btn.label, btn.variant, btn.disabled = "✔ On your disk", "success", True
        else:
            btn.label = f"⬇  Download  ·  {v.model.disk_gb} GB"
            btn.variant = "primary"
            btn.disabled = self.downloading is not None

    # ---- train ----------------------------------------------------------

    def action_train(self) -> None:
        if self.selected is None or self.advice is None or self.downloading:
            return
        v = self.selected
        if not self.advice.settings_for(v):
            self.say(f"[b]{escape(render.short_name(v.model.name))}[/b] is too big to train here. "
                     "Pick a smaller one.", "sad")
            return
        if self.advice.train_where == "colab":
            self.notify("No CUDA GPU detected; local training will be very slow.", severity="warning")
        from .tui_train import TrainScreen

        self.push_screen(TrainScreen(v, self.advice))

    def action_runs(self) -> None:
        if self.advice is None:
            return
        from .tui_train import RunsScreen

        self.push_screen(RunsScreen(self.advice))

    # ---- download -------------------------------------------------------

    @on(Button.Pressed, "#download")
    def _pressed(self) -> None:
        self.action_download()

    def action_download(self) -> None:
        if self.downloading or self.selected is None:
            return
        v = self.selected
        if state.downloaded_path(v.model.id):
            self.say(f"[b]{escape(render.short_name(v.model.name))}[/b] is already on your disk!", "happy")
            return
        if v.run == NO and v.train == CLOUD:
            self.notify(f"{v.model.name} won't run or train on this machine.", severity="warning")
        self.downloading = v.model.id
        bar = self.query_one("#progress", ProgressBar)
        bar.add_class("active")
        bar.update(total=None, progress=0)
        self._update_card()
        self.say(f"On it! Fetching [b]{escape(render.short_name(v.model.name))}[/b]…", "working")
        self._download(v)

    @work(thread=True, exclusive=True, group="download")
    def _download(self, v: Verdict) -> None:
        from .download import DownloadError, download

        def on_progress(done: int, total: int) -> None:
            self.call_from_thread(self._progress, v, done, total)

        try:
            path = download(v.model, on_progress)
        except DownloadError as e:
            self.call_from_thread(self._finished, v, None, str(e))
        else:
            self.call_from_thread(self._finished, v, str(path), None)

    def _progress(self, v: Verdict, done: int, total: int) -> None:
        self.query_one("#progress", ProgressBar).update(total=total or None, progress=done)
        if total:
            pct = done / total * 100
            self.say(
                f"Fetching [b]{escape(render.short_name(v.model.name))}[/b]… "
                f"{done / 1024**3:.2f} / {total / 1024**3:.2f} GB  ({pct:.0f}%)",
                "working",
                instant=True,
            )

    def _finished(self, v: Verdict, path: str | None, error: str | None) -> None:
        self.downloading = None
        self.query_one("#progress", ProgressBar).remove_class("active")
        if error:
            self.say(f"Oof. {escape(error)}", "sad")
            self.notify(error, title="Download failed", severity="error", timeout=12)
        else:
            self.query_one("#mascot", Mascot).cheer()
            self.say(
                f"Got it! [b]{escape(render.short_name(v.model.name))}[/b] is on your disk. "
                "Next up: a teacher (coming soon).",
                "happy",
            )
        self._fill_table()
        self._update_card()


if __name__ == "__main__":  # pragma: no cover
    KohaiApp().run()
