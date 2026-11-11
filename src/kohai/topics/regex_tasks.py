"""Regex topic: plain-English request + examples -> a regular expression.

Checked by running the regex against match / no-match strings (Python full-match
semantics). Uses the `regex` module so pathological patterns time out instead of
hanging training.
"""

from __future__ import annotations

import random

import regex

from .base import Topic

THEMES = [
    "dates", "times and durations", "email addresses", "URLs and domains", "IPv4/IPv6 addresses",
    "phone numbers", "hex colors", "file names and extensions", "semantic versions", "identifiers",
    "integers, decimals and signed numbers", "postal codes", "currency amounts", "usernames",
    "password rules", "log line fragments", "HTML-like tags", "UUIDs", "MAC addresses", "ISBNs",
    "card-number formats", "roman numerals", "binary/octal/hex literals", "git commit hashes",
    "CSS units", "URL slugs", "file paths", "coordinates", "repeated characters or words",
    "license plates", "product SKUs", "hashtags and mentions", "time zones", "percentages",
    "chess moves", "chemical formulas", "IBAN-like codes", "markdown syntax", "emoji shortcodes",
]

MIXES = [
    "mostly easy, a few medium",
    "mostly medium, a few hard",
    "an even spread of easy, medium and hard",
]

SHOWN = 3  # match / no-match examples shown to the student; the rest are hidden tests
TIMEOUT = 0.05


def _clean_answer(text: str) -> str:
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            block = parts[1]
            first_nl = block.find("\n")
            if first_nl != -1 and block[:first_nl].strip().isalpha():
                block = block[first_nl + 1 :]  # drop a language tag
            text = block.strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    ans = lines[0]
    for prefix in ("regex:", "pattern:", "answer:"):
        if ans.lower().startswith(prefix):
            ans = ans[len(prefix) :].strip()
    if len(ans) >= 2 and ans[0] == ans[-1] == "`":
        ans = ans[1:-1]
    if len(ans) >= 3 and ans[0] == "r" and ans[1] in "'\"" and ans[-1] == ans[1]:
        ans = ans[2:-1]
    elif len(ans) >= 2 and ans[0] == ans[-1] and ans[0] in "'\"":
        ans = ans[1:-1]
    if len(ans) >= 2 and ans[0] == ans[-1] == "/":
        ans = ans[1:-1]
    return ans


def run_tests(pattern: str, match: list[str], no_match: list[str]) -> tuple[int, int]:
    """Return (passed, total). Compile errors or timeouts count as failures."""
    total = len(match) + len(no_match)
    try:
        rx = regex.compile(pattern)
    except Exception:
        return 0, total
    passed = 0
    for s, want in [(s, True) for s in match] + [(s, False) for s in no_match]:
        try:
            got = rx.fullmatch(s, timeout=TIMEOUT) is not None
        except Exception:  # TimeoutError, etc.
            got = not want
        passed += got == want
    return passed, total


class RegexTopic(Topic):
    id = "regex"
    name = "Regex"
    description = "Write a regular expression from a plain-English request and examples."

    def teacher_prompt(self, n: int, rng: random.Random) -> str:
        themes = ", ".join(rng.sample(THEMES, 6))
        return f"""You are generating training data for a small model that learns to write regular expressions.

Create {n} diverse regex tasks as a JSON array. Each item has exactly these keys:
{{"description": "...", "match": [6 strings], "no_match": [6 strings], "regex": "..."}}

Rules:
- "regex" must FULLY match every "match" string and NONE of the "no_match" strings,
  using Python re.fullmatch semantics with no flags.
- "description" is a plain-English request a developer would write,
  e.g. "US ZIP codes, optionally followed by a dash and 4 more digits".
- Difficulty: {rng.choice(MIXES)}. Themes for this batch: {themes}.
- "no_match" strings must be tricky near-misses, not random junk.
- Every string is at most 40 characters with no newlines.
- Prefer clear, idiomatic patterns. No lookbehind.
- Output ONLY the JSON array. No commentary, no markdown fences."""

    def validate(self, item: dict) -> dict | None:
        try:
            desc = str(item["description"]).strip()
            pattern = str(item["regex"]).strip()
            match = [str(s) for s in item["match"]]
            no_match = [str(s) for s in item["no_match"]]
        except (KeyError, TypeError):
            return None
        if not desc or not pattern or len(match) < SHOWN + 1 or len(no_match) < SHOWN + 1:
            return None
        if any(len(s) > 60 or "\n" in s for s in match + no_match):
            return None
        if set(match) & set(no_match):
            return None
        passed, total = run_tests(pattern, match, no_match)
        if passed != total:
            return None
        return {"description": desc, "match": match, "no_match": no_match, "regex": pattern}

    def key(self, ex: dict) -> str:
        return " ".join(ex["description"].lower().split())

    def prompt(self, ex: dict) -> str:
        yes = "\n".join(f"- {s}" for s in ex["match"][:SHOWN])
        no = "\n".join(f"- {s}" for s in ex["no_match"][:SHOWN])
        return (
            f"Write a regular expression (Python syntax, must match the whole string) for: "
            f"{ex['description']}\n\nShould match:\n{yes}\n\nShould NOT match:\n{no}\n\n"
            f"Answer with only the regex."
        )

    def target(self, ex: dict) -> str:
        return ex["regex"]

    def extract(self, completion: str) -> str:
        return _clean_answer(completion)

    def score(self, ex: dict, completion: str) -> float:
        pattern = self.extract(completion)
        if not pattern:
            return 0.0
        passed, total = run_tests(pattern, ex["match"], ex["no_match"])
        return passed / total if total else 0.0
