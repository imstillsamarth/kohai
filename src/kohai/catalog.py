"""Student-model catalog (loaded from models.toml)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib import resources


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    repo: str
    family: str
    params_b: float
    license: str
    gated: bool
    context: int
    quality: int
    tags: tuple[str, ...]
    notes: str

    @property
    def disk_gb(self) -> float:
        """Approximate download size (bf16 safetensors)."""
        return round(self.params_b * 2.0 + 0.05, 1)

    @property
    def params_label(self) -> str:
        if self.params_b < 1:
            return f"{self.params_b * 1000:.0f}M"
        return f"{self.params_b:.1f}B"


def load_catalog() -> list[Model]:
    raw = resources.files("kohai").joinpath("models.toml").read_text(encoding="utf-8")
    data = tomllib.loads(raw)
    models = [
        Model(**{**entry, "tags": tuple(entry.get("tags", []))}) for entry in data.get("model", [])
    ]
    return sorted(models, key=lambda m: m.params_b)
