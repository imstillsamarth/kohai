"""A run = one kohai being trained: topic + teacher + student model + its data and adapters.

Layout of ~/.kohai/runs/<run-id>/:
    run.json        config (topic, teacher, model, settings)
    train.jsonl     verified training tasks
    test.jsonl      held-out tasks (never trained on)
    rejected.jsonl  teacher tasks that failed the checker
    adapter-sft/    LoRA after supervised fine-tuning
    adapter-rl/     LoRA after RL
    metrics.json    eval results (base vs sft vs rl, arena opponents)
    events.jsonl    live progress log (any UI can tail it)
    active.json     which process is working on this run right now
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import state


@dataclass
class Run:
    id: str
    topic: str
    teacher: str
    model_id: str
    model_repo: str
    params_b: float
    settings: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)

    @property
    def dir(self) -> Path:
        return runs_dir() / self.id

    def path(self, name: str) -> Path:
        return self.dir / name

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path("run.json").write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def read_jsonl(self, name: str) -> list[dict]:
        p = self.path(name)
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    def metrics(self) -> dict:
        try:
            return json.loads(self.path("metrics.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save_metrics(self, update: dict) -> None:
        data = self.metrics()
        data.update(update)
        self.path("metrics.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ---- live progress ---------------------------------------------------

    def log_event(self, event: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.path("events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": round(time.time(), 2), **event}, ensure_ascii=False) + "\n")

    def read_events(self, offset: int = 0) -> tuple[list[dict], int]:
        """New complete events since byte `offset`, and the new offset."""
        p = self.path("events.jsonl")
        if not p.exists():
            return [], offset
        with p.open("rb") as f:
            f.seek(offset)
            data = f.read()
        end = data.rfind(b"\n")
        if end == -1:
            return [], offset
        events = []
        for line in data[: end + 1].decode("utf-8", "replace").splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
        return events, offset + end + 1

    def set_active(self, stage: str | None) -> None:
        p = self.path("active.json")
        if stage is None:
            p.unlink(missing_ok=True)
        else:
            p.write_text(json.dumps({"pid": os.getpid(), "stage": stage, "t": time.time()}), encoding="utf-8")

    @property
    def active(self) -> dict | None:
        """{'pid', 'stage'} if a live process is working on this run."""
        try:
            data = json.loads(self.path("active.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        import psutil

        return data if psutil.pid_exists(int(data.get("pid", -1))) else None

    @property
    def stage(self) -> str:
        if self.path("adapter-rl").exists():
            return "rl"
        if self.path("adapter-sft").exists():
            return "sft"
        if self.path("train.jsonl").exists():
            return "data"
        if self.path("pool.jsonl").exists():
            return "collecting"
        return "new"


def runs_dir() -> Path:
    p = state.home() / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def new_run_id(topic: str, model_id: str) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{topic}-{model_id}"


def load_run(run_id: str) -> Run:
    """Load by full id, unique prefix, or 'latest'."""
    candidates = sorted(p.name for p in runs_dir().iterdir() if (p / "run.json").exists())
    if run_id == "latest":
        if not candidates:
            raise FileNotFoundError("No runs yet. Start one with `kohai gen`.")
        run_id = candidates[-1]
    matches = [c for c in candidates if c == run_id] or [c for c in candidates if c.startswith(run_id)]
    if len(matches) != 1:
        raise FileNotFoundError(f"No unique run matching {run_id!r}")
    data = json.loads((runs_dir() / matches[0] / "run.json").read_text(encoding="utf-8"))
    return Run(**data)


def list_runs() -> list[Run]:
    out = []
    for p in sorted(runs_dir().iterdir()):
        if (p / "run.json").exists():
            try:
                out.append(load_run(p.name))
            except (OSError, ValueError, TypeError):
                pass
    return out
