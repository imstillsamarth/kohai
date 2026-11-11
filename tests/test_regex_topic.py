import random

from kohai.generate import generate
from kohai.runs import Run
from kohai.teachers import Teacher
from kohai.topics import get_topic
from kohai.topics.regex_tasks import _clean_answer, run_tests

TOPIC = get_topic("regex")
ZIP = {
    "description": "US ZIP codes, optionally with a 4-digit extension",
    "match": ["12345", "12345-6789", "00000", "99999-0000", "54321", "11111-2222"],
    "no_match": ["1234", "123456", "12345-678", "abcde", "12345-", "-12345"],
    "regex": r"\d{5}(-\d{4})?",
}


def test_validate_accepts_correct_and_rejects_wrong():
    assert TOPIC.validate(ZIP) is not None
    assert TOPIC.validate({**ZIP, "regex": r"\d{5}"}) is None  # misses extensions
    assert TOPIC.validate({**ZIP, "regex": "(("}) is None  # doesn't compile
    assert TOPIC.validate({"description": "x"}) is None


def test_clean_answer_handles_common_formats():
    assert _clean_answer(r"`\d+`") == r"\d+"
    assert _clean_answer("```regex\n^a+$\n```") == "^a+$"
    assert _clean_answer("Regex: ab*") == "ab*"
    assert _clean_answer(r"r'\w+'") == r"\w+"
    assert _clean_answer("/abc/") == "abc"


def test_score_and_reward():
    assert TOPIC.score(ZIP, r"\d{5}(-\d{4})?") == 1.0
    assert TOPIC.reward(ZIP, r"\d{5}(-\d{4})?") == 1.0
    partial = TOPIC.score(ZIP, r"\d{5}")
    assert 0 < partial < 1
    assert TOPIC.score(ZIP, "((") == 0.0
    assert TOPIC.score(ZIP, "") == 0.0


def test_catastrophic_regex_times_out_instead_of_hanging():
    passed, total = run_tests(r"(a+)+$", ["a" * 5], ["a" * 30 + "b"])
    assert total == 2


def test_prompt_hides_some_tests():
    p = TOPIC.prompt(ZIP)
    assert "12345-6789" in p and "11111-2222" not in p


class FakeTeacher(Teacher):
    def __init__(self):
        super().__init__(id="fake", name="Fake")
        self.i = 0
        self.tag = ""

    def status(self):
        return "ready", ""

    def complete(self, prompt, timeout=300):
        import json

        items = []
        for _ in range(10):
            self.i += 1
            k = self.i
            items.append({
                "description": f"{self.tag}exactly {k} letter a's",
                "match": ["a" * k, "a" * k, "a" * k, "a" * k],
                "no_match": ["a" * (k + 1), "b", "a" * (k + 2), "c" * k],
                "regex": f"a{{{k}}}",
            })
        items.append({"description": "broken", "match": ["x"] * 4, "no_match": ["y"] * 4, "regex": "z"})
        return json.dumps(items)


def test_generate_with_fake_teacher(tmp_path, monkeypatch):
    monkeypatch.setenv("KOHAI_HOME", str(tmp_path))
    run = Run(id="t", topic="regex", teacher="fake", model_id="m", model_repo="m/m", params_b=0.5)
    run.save()
    result = generate(run, TOPIC, FakeTeacher(), n=40, workers=2, seed=1)
    assert result["train"] + result["test"] >= 40
    assert result["rejected"] >= 1
    assert len(run.read_jsonl("train.jsonl")) == result["train"]
    keys = {TOPIC.key(e) for e in run.read_jsonl("train.jsonl")}
    assert not keys & {TOPIC.key(e) for e in run.read_jsonl("test.jsonl")}


class FlakyTeacher(FakeTeacher):
    """Fails every call after the first `ok_calls`."""

    def __init__(self, ok_calls):
        super().__init__()
        self.ok_calls = ok_calls
        self.calls = 0

    def complete(self, prompt, timeout=300):
        from kohai.teachers import TeacherError

        self.calls += 1
        if self.calls > self.ok_calls:
            raise TeacherError("network down")
        return super().complete(prompt, timeout)


def test_generation_survives_failure_and_resumes(tmp_path, monkeypatch):
    import pytest

    from kohai import generate as gen
    from kohai.teachers import TeacherError

    monkeypatch.setenv("KOHAI_HOME", str(tmp_path))
    monkeypatch.setattr(gen, "CALL_RETRIES", 0)
    monkeypatch.setattr(gen.time, "sleep", lambda s: None)
    run = Run(id="t2", topic="regex", teacher="fake", model_id="m", model_repo="m/m", params_b=0.5)
    run.save()
    with pytest.raises(TeacherError):
        generate(run, TOPIC, FlakyTeacher(ok_calls=2), n=60, workers=1, seed=1)
    saved = len(run.read_jsonl("pool.jsonl"))
    assert saved == 20  # nothing lost

    teacher = FakeTeacher()
    teacher.tag = "second batch: "  # new, non-duplicate tasks
    result = generate(run, TOPIC, teacher, n=60, workers=1, seed=1)
    assert result["resumed"] == 20
    assert result["train"] + result["test"] >= 60
