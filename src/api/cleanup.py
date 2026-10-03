"""Deterministic reply cleanup: an ordered spec of small, pure, idempotent rules.

No model, no network, no guessing: every rule is a plain string transform that either
removes scraped-UI noise or fixes a well-known JSON slip. Two tiers:

TEXT_RULES   Applied to every reply before anything reads it. Remove what the browser
             scrape adds around ChatGPT's words; never rewrite the words themselves.
JSON_RULES   Applied only to a candidate tool-call/answer object that failed a strict
             parse. They can change content inside the failing object, which is
             acceptable because the alternative was a failed call.

repair_json() runs the JSON tier and returns text that strict json parses, or None.
It never invents arguments: unterminated strings and mismatched brackets are left
broken (None) rather than guessed at. The optional OpenRouter formatter only sees what
this spec could not fix.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Rule:
    name: str
    apply: Callable[[str], str]
    why: str


# ── Text tier ─────────────────────────────────────────────────────────

_INVISIBLE = re.compile("[​‌‍⁠﻿]")
_THINKING = re.compile(
    r"\A\s*Thought for (?:[0-9]+\s*(?:s|sec|secs|seconds?|m|min|mins|minutes?|h|hours?)\b\s*)+(?:\n+|\Z)",
    re.IGNORECASE,
)
_SPEAKER = re.compile(r"\A\s*(?:ChatGPT|You) said:\s*\n?", re.IGNORECASE)


def _strip_invisible(s: str) -> str:
    return _INVISIBLE.sub("", s)


def _normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def _strip_thinking_label(s: str) -> str:
    return _THINKING.sub("", s, count=1)


def _strip_speaker_label(s: str) -> str:
    return _SPEAKER.sub("", s, count=1)


TEXT_RULES: tuple[Rule, ...] = (
    Rule("strip_invisible", _strip_invisible, "zero-width characters and BOM break JSON and exact matches"),
    Rule("normalize_newlines", _normalize_newlines, "CRLF/CR from the clipboard become LF"),
    # Speaker label first: "ChatGPT said:" may precede "Thought for 6s", never the reverse.
    Rule("strip_speaker_label", _strip_speaker_label, 'scraped "ChatGPT said:" heading before the reply'),
    Rule("strip_thinking_label", _strip_thinking_label, 'scraped "Thought for 6s" line before the reply'),
)


def clean_text(text: str | None) -> str:
    """Apply the text tier, then trim. Idempotent; returns "" for None."""
    out = text or ""
    for rule in TEXT_RULES:
        out = rule.apply(out)
    return out.strip()


# ── JSON tier ─────────────────────────────────────────────────────────

# A double-quoted JSON string, so rules can skip string contents.
_STRING = r'"(?:\\.|[^"\\])*"'


def _normalize_spaces(s: str) -> str:
    return re.sub("[    ]", " ", s)


def _straighten_double_quotes(s: str) -> str:
    return re.sub("[“”„‟]", '"', s)


def _strip_comments(s: str) -> str:
    out: list[str] = []
    i, in_string = 0, False
    while i < len(s):
        c = s[i]
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < len(s):
                out.append(s[i + 1])
                i += 2
                continue
            in_string = c != '"'
            i += 1
        elif c == '"':
            in_string = True
            out.append(c)
            i += 1
        elif s.startswith("//", i):
            end = s.find("\n", i)
            i = len(s) if end < 0 else end
        elif s.startswith("/*", i):
            end = s.find("*/", i + 2)
            i = len(s) if end < 0 else end + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


_PY_LITERALS = {"True": "true", "False": "false", "None": "null"}


def _python_literals(s: str) -> str:
    return re.sub(
        rf"{_STRING}|\b(?:True|False|None)\b",
        lambda m: m.group(0) if m.group(0).startswith('"') else _PY_LITERALS[m.group(0)],
        s,
    )


def _trailing_commas(s: str) -> str:
    return re.sub(
        rf"{_STRING}|,(\s*[}}\]])",
        lambda m: m.group(0) if m.group(0).startswith('"') else m.group(1),
        s,
    )


def _close_brackets(s: str) -> str:
    """Append the closers a truncated object is missing. Never guesses: an unterminated
    string or a mismatched bracket returns the text unchanged."""
    stack: list[str] = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if in_string:
            if c == "\\":
                i += 1
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]":
            if not stack or stack.pop() != c:
                return s
        i += 1
    return s if in_string else s + "".join(reversed(stack))


def _escape_control_chars(s: str) -> str:
    """Raw newlines/tabs inside a string (a multi-line shell command) are invalid JSON."""
    return re.sub(
        _STRING,
        lambda m: m.group(0).replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"),
        s,
    )


# Order matters: closers are added before trailing commas are removed, so a truncated
# `[1,2,` ends up as `[1,2]`.
JSON_RULES: tuple[Rule, ...] = (
    Rule("normalize_spaces", _normalize_spaces, "non-breaking/thin spaces are not JSON whitespace"),
    Rule("straighten_double_quotes", _straighten_double_quotes, "typographic quotes used as string delimiters"),
    Rule("escape_control_chars", _escape_control_chars, "raw newline/tab characters inside a string"),
    Rule("strip_comments", _strip_comments, "// and /* */ comments outside strings"),
    Rule("python_literals", _python_literals, "True/False/None outside strings"),
    Rule("close_brackets", _close_brackets, "missing closers on a truncated object (never inside a string)"),
    Rule("trailing_commas", _trailing_commas, "a comma before } or ]"),
)


def _unfence(text: str) -> str:
    fenced = re.search(r"```[A-Za-z0-9_-]*[ \t]*\n?(.*?)```", text, re.DOTALL)
    if fenced and "{" in fenced.group(1):
        return fenced.group(1)
    return re.sub(r"\A\s*```[A-Za-z0-9_-]*[ \t]*\n?", "", text)  # an opening fence that never closed


def _object_at_start(s: str) -> str | None:
    """s up to the end of a strictly parseable JSON object that starts at s[0], else None.

    Only the object at the start counts: scanning on to a later "{" would return an inner
    object of a broken outer one (the arguments of a truncated tool call, say) as if it
    were the whole reply.
    """
    try:
        obj, end = json.JSONDecoder().raw_decode(s)
    except ValueError:
        return None
    return s[:end] if isinstance(obj, dict) else None


def _python_dict(s: str) -> str | None:
    """A Python-style dict literal ('single' quotes, True/None) as JSON text.

    ast.literal_eval only evaluates literals, so this cannot run code.
    """
    literal = s.replace("‘", "'").replace("’", "'")
    end = literal.rfind("}")
    if end <= 0:
        return None
    try:
        parsed = ast.literal_eval(literal[: end + 1])
        return json.dumps(parsed) if isinstance(parsed, dict) else None
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def repair_json(text: str | None, anchor: str | None = None) -> str | None:
    """Return one JSON object from text, repaired by JSON_RULES, as strictly valid JSON text.

    The object starts at the "{" that opens the object holding the key ``anchor`` (for
    example "tool_calls"), or at the first "{" without one. Prose before or after the
    object and a Markdown fence around it are ignored. Returns None when the spec cannot
    make a valid object without guessing.
    """
    if not text or "{" not in text:
        return None
    body = _unfence(text)
    start = -1
    if anchor and f'"{anchor}"' in body:
        start = body.rfind("{", 0, body.find(f'"{anchor}"'))
    if start < 0:
        start = body.find("{")
    if start < 0:
        return None
    body = body[start:]

    found = _object_at_start(body) or _python_dict(body)
    if found:
        return found
    cleaned = body
    for rule in JSON_RULES:
        cleaned = rule.apply(cleaned)
    return _object_at_start(cleaned)
