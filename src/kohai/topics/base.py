"""The Topic interface. A topic pack = task generator prompt + checker + reward."""

from __future__ import annotations

import json
import random
import re
from abc import ABC, abstractmethod


class Topic(ABC):
    id: str
    name: str
    description: str

    @abstractmethod
    def teacher_prompt(self, n: int, rng: random.Random) -> str:
        """Prompt asking the teacher for `n` tasks with reference solutions."""

    @abstractmethod
    def validate(self, item: dict) -> dict | None:
        """Normalize one raw teacher item; return None if malformed or its solution is wrong."""

    @abstractmethod
    def key(self, ex: dict) -> str:
        """Dedup key."""

    @abstractmethod
    def prompt(self, ex: dict) -> str:
        """What the student sees."""

    @abstractmethod
    def target(self, ex: dict) -> str:
        """The ideal student answer (for SFT)."""

    @abstractmethod
    def score(self, ex: dict, completion: str) -> float:
        """0..1 partial credit for a completion. 1.0 means solved."""

    def reward(self, ex: dict, completion: str) -> float:
        """RL reward: partial credit plus a bonus for fully solving."""
        s = self.score(ex, completion)
        return 0.5 * s + (0.5 if s >= 1.0 else 0.0)

    def parse(self, text: str) -> list[dict]:
        """Pull a JSON array of task objects out of a teacher's reply."""
        text = re.sub(r"```(?:json)?", "", text)
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data, list):
                    return [d for d in data if isinstance(d, dict)]
            except json.JSONDecodeError:
                pass
        items = []
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if line.startswith("{") and line.endswith("}"):
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return items
