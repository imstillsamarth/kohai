"""Tofu: kohai's pixel sensei. A tofu cube with a red headband and a black belt.

Drawn with half-block characters: each terminal cell shows two vertical pixels
(top pixel = foreground of "▀", bottom pixel = background), so the sprite is
16x20 pixels in a 16x10 cell widget.
"""

from __future__ import annotations

import random

from rich.style import Style
from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget

PALETTE = {
    "O": "#5A4C68",  # outline
    "W": "#F7EEDC",  # tofu
    "S": "#E2D3B8",  # tofu shade
    "R": "#E5484D",  # headband
    "r": "#B8323A",  # headband tails / mouth
    "K": "#1F1B24",  # belt
    "E": "#1F1B24",  # eyes
    "P": "#F4A7B0",  # blush
    "B": "#8EC5FF",  # sweat drop
    "Y": "#FFD166",  # sparkle
    "h": "#262633",  # shadow
}

# 16 x 14 base sprite. Eyes, mouth, tails and effects are drawn per frame.
BASE = [
    "................",
    "...OOOOOOOOOO...",
    "..OWWWWWWWWWWO..",
    "..ORRRRRRRRRRO..",
    "..OWWWWWWWWWWO..",
    "..OWWWWWWWWWWO..",
    "..OWWWWWWWWWWO..",
    "..OWPWWWWWWPWO..",
    "..OWWWWWWWWWWO..",
    "..OWWWWWWWWWWO..",
    "..OKKKKKKKKKKO..",
    "..OSWWWKKWWWSO..",
    "..OSSWKWWKWSSO..",
    "...OOOOOOOOOO...",
    "...OO......OO...",
]

CANVAS_W, CANVAS_H = 16, 20
BASELINE = 3
JUMP = (0, 1, 2, 3, 3, 2, 1, 0)
EYES = (5, 10)

MOODS = ("idle", "happy", "thinking", "working", "sad")

_STYLES: dict[tuple[str, str | None], Style] = {}


def _style(fg: str, bg: str | None = None) -> Style:
    key = (fg, bg)
    if key not in _STYLES:
        _STYLES[key] = Style(color=PALETTE[fg], bgcolor=PALETTE[bg] if bg else None)
    return _STYLES[key]


def frame(tick: int, mood: str = "idle", blinking: bool = False, cheering: bool = False) -> list[list[str]]:
    """Build one animation frame as a CANVAS_H x CANVAS_W grid of palette keys ('.' = empty)."""
    g = [list(row) for row in BASE]

    def put(r: int, c: int, ch: str) -> None:
        g[r][c] = ch

    fast = mood == "working" or cheering

    # Headband tails flutter.
    put(3, 14, "r")
    put(3, 15, "r")
    if (tick // (2 if fast else 4)) % 2:
        put(2, 15, "r")
    else:
        put(4, 15, "r")

    # Eyes.
    if cheering or mood == "happy":
        for x in EYES:  # ^ ^
            put(5, x, "E")
            put(6, x - 1, "E")
            put(6, x + 1, "E")
    elif blinking:
        for x in EYES:
            put(6, x, "E")
    elif mood == "working":
        shift = 1 if (tick // 8) % 2 else -1
        for x in EYES:
            put(5, x + shift, "E")
            put(6, x + shift, "E")
    elif mood == "thinking":
        for x in EYES:
            put(5, x + 1, "E")
    elif mood == "sad":
        for x, lean in zip(EYES, (1, -1)):
            put(5, x + lean, "E")
            put(6, x, "E")
    else:
        for x in EYES:
            put(5, x, "E")
            put(6, x, "E")

    # Mouth.
    if cheering:
        for c in range(6, 10):
            put(7, c, "E")
        put(8, 7, "r")
        put(8, 8, "r")
    elif mood == "happy":
        put(7, 6, "E")
        put(7, 9, "E")
        put(8, 7, "E")
        put(8, 8, "E")
    elif mood == "sad":
        put(7, 7, "E")
        put(7, 8, "E")
        put(8, 6, "E")
        put(8, 9, "E")
    else:
        put(8, 7, "E")
        put(8, 8, "E")

    # Effects.
    if mood in ("working", "thinking") and (tick // 6) % 3:
        put(4, 1, "B")
        put(5, 1, "B")
    if cheering:
        if (tick // 2) % 2:
            put(0, 1, "Y")
            put(1, 14, "Y")
        else:
            put(1, 0, "Y")
            put(0, 14, "Y")

    # Position: breathe, bob faster when working, jump when cheering.
    if cheering:
        y = BASELINE - JUMP[tick % len(JUMP)] + 1
    elif mood == "working":
        y = BASELINE + (tick // 2) % 2
    else:
        y = BASELINE + (tick // 5) % 2

    canvas = [["."] * CANVAS_W for _ in range(CANVAS_H)]
    lift = BASELINE + 1 - y  # 0 on the ground
    shadow_w = max(8 - 2 * lift, 2)
    start = (CANVAS_W - shadow_w) // 2
    for c in range(start, start + shadow_w):
        canvas[CANVAS_H - 1][c] = "h"
    for r, row in enumerate(g):
        for c, ch in enumerate(row):
            if ch != "." and 0 <= y + r < CANVAS_H:
                canvas[y + r][c] = ch
    return canvas


def draw(tick: int, mood: str = "idle", blinking: bool = False, cheering: bool = False) -> Text:
    """Render one animation frame as Rich Text using half-block characters."""
    canvas = frame(tick, mood, blinking, cheering)
    text = Text(no_wrap=True, overflow="crop")
    for r in range(0, CANVAS_H, 2):
        for c in range(CANVAS_W):
            top, bot = canvas[r][c], canvas[r + 1][c]
            if top == "." and bot == ".":
                text.append(" ")
            elif top == ".":
                text.append("▄", _style(bot))
            elif bot == ".":
                text.append("▀", _style(top))
            else:
                text.append("▀", _style(top, bot))
        if r < CANVAS_H - 2:
            text.append("\n")
    return text


class Mascot(Widget):
    """Animated Tofu. Set `mood`, call `cheer()` to celebrate."""

    DEFAULT_CSS = """
    Mascot { width: 16; height: 10; }
    """

    mood: reactive[str] = reactive("idle")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tick = 0
        self._blink = 0
        self._next_blink = random.randint(15, 35)
        self._cheer = 0

    def on_mount(self) -> None:
        self.set_interval(0.12, self._step)

    def cheer(self, ticks: int = 24) -> None:
        self._cheer = ticks

    def _step(self) -> None:
        self.tick += 1
        if self._blink:
            self._blink -= 1
        elif self.tick >= self._next_blink:
            self._blink = 2
            self._next_blink = self.tick + random.randint(25, 60)
        if self._cheer:
            self._cheer -= 1
        self.refresh()

    def render(self) -> Text:
        return draw(self.tick, self.mood, blinking=self._blink > 0, cheering=self._cheer > 0)
