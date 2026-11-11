"""Topic packs: what the kohai specializes in, and how answers are checked."""

from __future__ import annotations

from .base import Topic
from .regex_tasks import RegexTopic

_TOPICS: dict[str, Topic] = {t.id: t for t in (RegexTopic(),)}


def get_topic(topic_id: str) -> Topic:
    try:
        return _TOPICS[topic_id]
    except KeyError:
        raise ValueError(f"Unknown topic {topic_id!r}. Available: {', '.join(_TOPICS)}") from None


def all_topics() -> list[Topic]:
    return list(_TOPICS.values())


__all__ = ["Topic", "get_topic", "all_topics"]
