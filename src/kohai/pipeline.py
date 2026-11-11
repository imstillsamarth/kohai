"""Glue for running training stages as subprocesses and the arena against big models."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

from .runs import Run
from .teachers import Teacher, TeacherError
from .topics import Topic

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class StageError(Exception):
    pass


def stream_stage(
    run: Run,
    stage: str,
    extra: list[str] | None = None,
    on_proc: Callable[[subprocess.Popen], None] | None = None,
) -> Iterator[dict]:
    """Run `python -m kohai.trainer <stage> <run>` and yield its JSON events.

    `on_proc` receives the child process, so a UI can terminate it to cancel."""
    cmd = [sys.executable, "-m", "kohai.trainer", stage, run.id, *(extra or [])]
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_VERBOSITY": "error",
    }
    log_path = run.path("train.log")
    run.dir.mkdir(parents=True, exist_ok=True)
    run.set_active(stage)
    run.log_event({"event": "stage_start", "stage": stage})
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {stage} =====\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8",
            errors="replace", env=env, creationflags=_NO_WINDOW,
        )
        if on_proc:
            on_proc(proc)
        error = None
        assert proc.stdout
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "error":
                error = event.get("error")
            run.log_event({"stage": stage, **event})
            yield event
        code = proc.wait()
    if code != 0:
        run.log_event({"event": "stage_failed", "stage": stage, "error": error or f"exit code {code}"})
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:])
        raise StageError(f"{stage} failed: {error or f'exit code {code}'}\n--- train.log ---\n{tail}")
    run.log_event({"event": "stage_end", "stage": stage})


def arena(
    run: Run, topic: Topic, teacher: Teacher, workers: int = 4, emit: Callable[[dict], None] | None = None
) -> dict:
    """Score a big model (any teacher) on the run's held-out test set."""
    emit = emit or (lambda e: None)
    tests = run.read_jsonl("test.jsonl")
    scores: list[float] = []
    errors = 0

    def one(ex: dict) -> float | None:
        try:
            return topic.score(ex, teacher.complete(topic.prompt(ex), timeout=240))
        except TeacherError:
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, s in enumerate(pool.map(one, tests), 1):
            if s is None:
                errors += 1
            else:
                scores.append(s)
            emit({"event": "step", "stage": "arena", "step": i, "total": len(tests)})
    if not scores:
        raise TeacherError(f"{teacher.name} answered none of the {len(tests)} questions.")
    result = {
        "name": teacher.name,
        "solved": round(sum(s >= 1.0 for s in scores) / len(scores), 4),
        "score": round(sum(scores) / len(scores), 4),
        "n": len(scores),
        "errors": errors,
        "is_teacher": teacher.id == run.teacher,
    }
    arena_all = run.metrics().get("arena", {})
    arena_all[teacher.id] = result
    run.save_metrics({"arena": arena_all})
    return result


def create_run(topic: Topic, teacher: Teacher, verdict, settings: dict) -> Run:
    """Create and save a new run for a student model (an advisor Verdict)."""
    from .runs import new_run_id

    run = Run(
        id=new_run_id(topic.id, verdict.model.id),
        topic=topic.id,
        teacher=teacher.id,
        model_id=verdict.model.id,
        model_repo=verdict.model.repo,
        params_b=verdict.model.params_b,
        settings=settings,
    )
    run.save()
    return run
