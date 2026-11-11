"""Rich renderables shared by the CLI and the TUI."""

from __future__ import annotations

from rich.console import Group
from rich.table import Table
from rich.text import Text

from . import state
from .advisor import CLOUD, NO, OK, RUN_CPU, RUN_GPU, SLOW, TIGHT, Advice, Verdict
from .sysinfo import SystemInfo

_FIT = {
    OK: ("✔ fits", "green"),
    TIGHT: ("~ tight", "yellow"),
    CLOUD: ("☁ cloud", "cyan"),
    SLOW: ("slow·cpu", "yellow"),
    NO: ("✘", "red"),
    RUN_GPU: ("✔ GPU", "green"),
    RUN_CPU: ("CPU", "yellow"),
}

_BADGE_LABEL = {"★": "recommended", "◆": "stretch", "✦": "fast iteration"}


def fit(value: str) -> Text:
    label, style = _FIT.get(value, (value, ""))
    return Text(label, style=style)


def system_panel(info: SystemInfo) -> Table:
    t = Table.grid(padding=(0, 1))
    t.add_column(style="bold cyan", no_wrap=True)
    t.add_column()
    t.add_row("OS", f"{info.os} {info.os_release}" + ("  [dim](WSL)[/]" if info.in_wsl else ""))
    t.add_row("CPU", f"{info.cpu}\n[dim]{info.cores} cores / {info.threads} threads[/]")
    t.add_row("RAM", f"{info.ram_gb:.1f} GB")
    if info.gpus:
        for g in info.gpus:
            details = []
            if g.vram_gb:
                details.append(f"{g.vram_gb:.1f} GB VRAM")
            if g.cuda:
                details.append(f"CUDA {g.cuda}")
            if g.driver:
                details.append(f"driver {g.driver}")
            line = f"[bold]{g.name}[/]"
            if details:
                line += "\n[dim]" + " · ".join(details) + "[/]"
            t.add_row("GPU", line)
    else:
        t.add_row("GPU", "[red]none detected[/]")
    t.add_row("Disk", f"{info.disk_free_gb:.0f} GB free [dim](model cache)[/]")
    t.add_row("Python", info.python)
    return t


def _pick_line(symbol: str, label: str, v: Verdict | None, why: str) -> Text:
    line = Text()
    line.append(f"{symbol} {label}: ", style="bold")
    if v is None:
        line.append("none", style="dim")
    else:
        line.append(v.model.name, style="bold magenta")
        line.append(f"\n   {why}", style="dim")
    return line


def advice_panel(a: Advice) -> Group:
    parts: list = []
    head = Text()
    names = {"cuda": "NVIDIA CUDA", "mps": "Apple Metal", "cpu": "CPU only"}
    head.append(names[a.backend], style="bold green" if a.backend != "cpu" else "bold yellow")
    if a.budget_gb:
        head.append(f" · {a.budget_gb:.1f} GB training budget")
    if a.train_where == "colab":
        head.append(" · picks below are for Colab", style="cyan")
    parts.append(head)
    parts.append(Text())

    rec = a.recommended
    parts.append(
        _pick_line(
            "★",
            "Recommended",
            rec,
            "Best student that fits SFT + RL comfortably" if rec and rec.rl == OK else "Best student that fits",
        )
    )
    if a.stretch:
        parts.append(_pick_line("◆", "Stretch", a.stretch, "Biggest you can still train; expect tuning"))
    if a.fast:
        parts.append(_pick_line("✦", "Fast iteration", a.fast, "Debug your dataset here first, then scale up"))

    if a.settings:
        s = a.settings
        parts.append(Text())
        parts.append(Text("Training settings", style="bold underline"))
        parts.append(
            Text(
                f"{s['method']} · seq {s['max_seq_len']} · batch {s['batch_size']}×{s['grad_accum']} accum\n"
                f"LoRA r={s['lora_rank']} · grad-ckpt {s['gradient_checkpointing']} · "
                f"RL group {s['rl_group_size']}"
            )
        )

    if a.tips:
        parts.append(Text())
        for tip in a.tips:
            parts.append(Text(f"• {tip}"))
    for w in a.warnings:
        parts.append(Text(f"⚠ {w}", style="bold yellow"))
    return Group(*parts)


def model_detail(v: Verdict, a: Advice) -> Group:
    m = v.model
    title = Text(m.name, style="bold magenta")
    badge = a.badge(m.id)
    if badge:
        title.append(f"  {badge} {_BADGE_LABEL[badge]}", style="bold yellow")

    meta = Text(f"{m.repo} · {m.license} · {m.context // 1024}k ctx · {m.params_label} params", style="dim")
    if m.gated:
        meta.append("  🔒 gated", style="yellow")

    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold")
    t.add_column(justify="right")
    t.add_column()
    t.add_row("Run (4-bit)", f"~{v.est.infer_gb} GB", fit(v.run))
    t.add_row("Fine-tune (QLoRA)", f"~{v.est.train_gb} GB", fit(v.train))
    t.add_row("RL (GRPO)", f"~{v.est.rl_gb} GB", fit(v.rl))
    cached = state.downloaded_path(m.id)
    t.add_row("Download", f"~{m.disk_gb} GB", Text("✔ downloaded", style="green") if cached else Text(""))

    return Group(title, meta, Text(), t, Text(), Text(m.notes), Text("tags: " + ", ".join(m.tags), style="dim"))


def models_table(a: Advice) -> Table:
    t = Table(title=f"Student models (budget {a.budget_gb:.1f} GB)", header_style="bold")
    t.add_column("", width=2)
    t.add_column("id")
    t.add_column("Model")
    t.add_column("Params", justify="right")
    t.add_column("Run")
    t.add_column("Fine-tune")
    t.add_column("RL")
    t.add_column("License")
    for v in a.verdicts:
        t.add_row(
            a.badge(v.model.id),
            v.model.id,
            v.model.name + (" 🔒" if v.model.gated else ""),
            v.model.params_label,
            fit(v.run),
            fit(v.train),
            fit(v.rl),
            v.model.license,
        )
    return t


# ---------------------------------------------------------------------------
# TUI cards
# ---------------------------------------------------------------------------

import re  # noqa: E402

from .advisor import COMFORT  # noqa: E402

GREEN, YELLOW, BLUE, RED, DIM, MINT = "#7BD88F", "#F5C76B", "#7AA2F7", "#F7768E", "#6B6F80", "#7FD1B9"

_CHIP = {
    OK: ("● fits", GREEN),
    TIGHT: ("◐ tight", YELLOW),
    CLOUD: ("○ cloud", BLUE),
    SLOW: ("◌ slow", YELLOW),
    NO: ("✕ no", RED),
    RUN_GPU: ("● gpu", GREEN),
    RUN_CPU: ("◐ cpu", YELLOW),
}


def chip(value: str) -> Text:
    label, color = _CHIP.get(value, (value, DIM))
    return Text(label, style=color)


def short_gpu(name: str) -> str:
    name = re.sub(r"^(NVIDIA\s+)?(GeForce\s+)?", "", name)
    return name.replace(" Laptop GPU", " Laptop")


def short_cpu(name: str) -> str:
    n = re.sub(r"\((R|TM)\)", "", name)
    n = re.sub(r"\b\d+(st|nd|rd|th) Gen\b", "", n)
    n = re.sub(r"\b(Intel Core|Intel|AMD|Processor|CPU)\b", "", n)
    n = n.split(" with ")[0].split("@")[0]
    return re.sub(r"\s+", " ", n).strip()


def short_name(model_name: str) -> str:
    return re.sub(r"\s+(Instruct|IT)$", "", model_name)


def bar(need: float, budget: float, width: int = 16) -> Text:
    if budget <= 0:
        return Text("─" * width, style=DIM)
    ratio = need / budget
    color = GREEN if ratio <= COMFORT else YELLOW if ratio <= 1 else RED
    filled = width if ratio > 1 else max(1, round(ratio * width))
    return Text("━" * filled, style=color) + Text("━" * (width - filled), style="#2E3040")


def system_card(info: SystemInfo, a: Advice) -> Text:
    t = Text()
    gpu = info.best_gpu
    if gpu and a.backend != "cpu":
        t.append(short_gpu(gpu.name), style="bold")
        t.append("\n")
        total = gpu.vram_gb or a.budget_gb
        label = "VRAM " if a.backend == "cuda" else "MEM  "
        t.append(label, style=DIM)
        t.append_text(bar(a.budget_gb, total, 14) if total else Text())
        t.append(f" {a.budget_gb:.1f}", style=f"bold {MINT}")
        t.append(f" / {total:.1f} GB usable\n", style=DIM)
    else:
        t.append(short_gpu(gpu.name) if gpu else "No GPU detected", style="bold")
        t.append("\n")
        t.append("Training → free Colab GPU\n", style=BLUE)
    t.append("RAM  ", style=DIM)
    t.append(f"{info.ram_gb:.0f} GB")
    t.append("  CPU ", style=DIM)
    t.append(short_cpu(info.cpu) + "\n")
    t.append("DISK ", style=DIM)
    t.append(f"{info.disk_free_gb:.0f} GB free")
    extra = [f"CUDA {gpu.cuda}"] if gpu and gpu.cuda else []
    extra.append(f"Py {info.python}")
    t.append("  " + " · ".join(extra), style=DIM)
    return t


def model_card(v: Verdict, a: Advice, downloaded: bool) -> Group:
    m = v.model
    parts: list = []

    title = Text(short_name(m.name), style=f"bold {MINT}")
    parts.append(title)
    badge = a.badge(m.id)
    if badge:
        parts.append(Text(f"{badge} {_BADGE_LABEL[badge]} for your machine", style=f"bold {YELLOW}"))
    parts.append(Text(m.repo, style=DIM))
    meta = Text(f"{m.params_label} params · {m.context // 1024}k ctx · {m.license}")
    if m.gated:
        meta.append("  🔒 license gate", style=YELLOW)
    parts.append(meta)
    parts.append(Text())

    where = f"your {a.budget_gb:.1f} GB" if a.budget_gb else "your machine (CPU)"
    parts.append(Text(f"MEMORY ON {where.upper()}", style=f"bold {DIM}"))
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="bold", width=6)
    grid.add_column(no_wrap=True)
    grid.add_column(justify="right", width=6)
    grid.add_column(no_wrap=True)
    for label, need, verdict in (
        ("Run", v.est.infer_gb, v.run),
        ("Train", v.est.train_gb, v.train),
        ("RL", v.est.rl_gb, v.rl),
    ):
        grid.add_row(label, bar(need, a.budget_gb), f"{need:.1f}G", chip(verdict))
    parts.append(grid)
    parts.append(Text())

    settings = a.settings_for(v)
    parts.append(Text("TRAIN WITH", style=f"bold {DIM}"))
    if settings:
        where_note = " (on Colab)" if a.train_where == "colab" else ""
        parts.append(
            Text(
                f"{settings['method']} · seq {settings['max_seq_len']} · "
                f"batch {settings['batch_size']}×{settings['grad_accum']} · r{settings['lora_rank']}{where_note}"
            )
        )
    else:
        parts.append(Text("Needs a bigger GPU (cloud) to train.", style=BLUE))
    parts.append(Text())
    parts.append(Text(m.notes))
    parts.append(Text(" ".join(f"#{t}" for t in m.tags), style=MINT))
    if downloaded:
        parts.append(Text())
        parts.append(Text("✔ downloaded", style=f"bold {GREEN}"))
    return Group(*parts)
