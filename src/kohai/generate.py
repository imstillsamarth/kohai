"""Dataset generation: the senpai writes tasks, the topic's checker keeps only correct ones.

Verified tasks are appended to `pool.jsonl` the moment they pass, so a crash, a network drop
or Ctrl+C never loses work: running again on the same run resumes from the pool.
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .runs import Run
from .teachers import Teacher, TeacherError
from .topics import Topic

Emit = Callable[[dict], None]

MAX_CONSECUTIVE_ERRORS = 6
CALL_RETRIES = 2  # extra attempts per teacher call before it counts as an error


def split_size(n: int) -> int:
    return max(10, min(100, round(n * 0.15)))


def _load_pool(run: Run, topic: Topic) -> dict[str, dict]:
    pool: dict[str, dict] = {}
    for ex in run.read_jsonl("pool.jsonl"):
        pool.setdefault(topic.key(ex), ex)
    return pool


def finalize(run: Run, topic: Topic, n: int, seed: int | None = None) -> dict:
    """Split the pool into train / held-out test files."""
    examples = list(_load_pool(run, topic).values())
    if len(examples) < 20:
        raise TeacherError(f"Only {len(examples)} verified tasks so far; need at least 20.")
    random.Random(seed).shuffle(examples)
    n_test = min(split_size(n), len(examples) // 5)
    test, train = examples[:n_test], examples[n_test:]
    for name, rows in (("train.jsonl", train), ("test.jsonl", test)):
        with run.path(name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"train": len(train), "test": len(test)}


def generate(
    run: Run,
    topic: Topic,
    teacher: Teacher,
    n: int,
    batch: int = 10,
    workers: int = 4,
    emit: Emit | None = None,
    seed: int | None = None,
) -> dict:
    """Collect `n` verified tasks (plus a held-out test split) into the run directory."""
    ui_emit = emit or (lambda e: None)

    def emit(e: dict) -> None:
        run.log_event({"stage": "gen", **e})
        ui_emit(e)

    run.dir.mkdir(parents=True, exist_ok=True)
    run.set_active("gen")
    emit({"event": "stage_start"})
    rng = random.Random(seed)
    lock = threading.Lock()
    target = n + split_size(n)
    run.dir.mkdir(parents=True, exist_ok=True)

    accepted = _load_pool(run, topic)
    resumed = len(accepted)
    needed_calls = max(0, target - resumed) // batch + 1
    max_calls = max(4, needed_calls * 3)
    stats = {"calls": 0, "errors": 0, "accepted": len(accepted), "rejected": 0, "duplicates": 0,
             "resumed": resumed}
    consecutive_errors = 0
    last_error = ""
    emit({"event": "gen", "target": target, **stats})

    def one_call(prompt: str) -> str:
        for attempt in range(CALL_RETRIES + 1):
            try:
                return teacher.complete(prompt)
            except TeacherError:
                if attempt == CALL_RETRIES:
                    raise
                time.sleep(5 * (attempt + 1))
        return ""

    # Not a `with` block: on abort we must not wait for in-flight teacher calls to time out.
    pool = ThreadPoolExecutor(max_workers=workers)
    with (
        run.path("pool.jsonl").open("a", encoding="utf-8") as pool_f,
        run.path("rejected.jsonl").open("a", encoding="utf-8") as rej_f,
    ):
        try:
            pending = set()
            submitted = 0

            def submit() -> None:
                nonlocal submitted
                with lock:
                    prompt = topic.teacher_prompt(batch, rng)
                pending.add(pool.submit(one_call, prompt))
                submitted += 1

            while len(accepted) < target and submitted < max_calls and len(pending) < workers:
                submit()

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    pending.discard(fut)
                    stats["calls"] += 1
                    try:
                        text = fut.result()
                        consecutive_errors = 0
                    except TeacherError as e:
                        stats["errors"] += 1
                        consecutive_errors += 1
                        last_error = str(e)
                        emit({"event": "teacher_error", "error": last_error})
                        text = ""
                    for raw in topic.parse(text):
                        ex = topic.validate(raw)
                        if ex is None:
                            rej_f.write(json.dumps(raw, ensure_ascii=False, default=str) + "\n")
                            stats["rejected"] += 1
                            continue
                        k = topic.key(ex)
                        if k in accepted:
                            stats["duplicates"] += 1
                            continue
                        accepted[k] = ex
                        pool_f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    pool_f.flush()
                    rej_f.flush()
                    stats["accepted"] = len(accepted)
                    emit({"event": "gen", "target": target, **stats})

                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    raise TeacherError(
                        f"Senpai failed {consecutive_errors} times in a row ({len(accepted)} tasks saved, "
                        f"resume any time). Last error: {last_error}"
                    )
                while len(accepted) < target and submitted < max_calls and len(pending) < workers:
                    submit()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    split = finalize(run, topic, n, seed)
    result = {**stats, **split}
    run.save_metrics({"data": result})
    emit({"event": "gen_done", **result})
    return result
