"""Stateful ORSO chat session with bounded conversation memory."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any



@dataclass
class Turn:
    role: str
    text: str


class ChatSession:
    """Small, dependency-free conversation buffer.

    The session stores text turns and renders only the newest turns that fit the
    configured character budget. The model's own context window remains the hard
    limit during generation.
    """

    def __init__(self, *, system_prompt: str = "", max_turns: int = 12, max_chars: int = 12000):
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        self.system_prompt = str(system_prompt)
        self.max_turns = int(max_turns)
        self.max_chars = int(max_chars)
        self.turns: list[Turn] = []

    def clear(self) -> None:
        self.turns.clear()

    def add(self, role: str, text: str) -> None:
        role = str(role).strip() or "user"
        text = str(text)
        self.turns.append(Turn(role, text))
        if len(self.turns) > self.max_turns:
            del self.turns[: len(self.turns) - self.max_turns]

    def prompt(self, user_text: str) -> str:
        pieces: list[str] = []
        if self.system_prompt:
            pieces.append(f"System: {self.system_prompt}")
        pieces.extend(f"{turn.role.capitalize()}: {turn.text}" for turn in self.turns)
        pieces.append(f"User: {user_text}")
        pieces.append("Assistant:")
        rendered = "\n".join(pieces)
        if len(rendered) <= self.max_chars:
            return rendered
        return rendered[-self.max_chars :]

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "max_turns": self.max_turns,
            "max_chars": self.max_chars,
            "turns": [{"role": t.role, "text": t.text} for t in self.turns],
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "ChatSession":
        session = cls(
            system_prompt=str(state.get("system_prompt", "")),
            max_turns=int(state.get("max_turns", 12)),
            max_chars=int(state.get("max_chars", 12000)),
        )
        for item in state.get("turns", []):
            if not isinstance(item, dict):
                continue
            session.add(str(item.get("role", "user")), str(item.get("text", "")))
        return session

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ChatSession":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
