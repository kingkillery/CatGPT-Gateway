"""Model-picker helpers for the live ChatGPT web UI."""

from __future__ import annotations

from dataclasses import dataclass

from patchright.async_api import Page
from src.chatgpt.model_selector_scripts import (
    _MARK_OPENERS_SCRIPT,
    _MARK_OPTIONS_SCRIPT,
    _VISIBLE_FINGERPRINTS_SCRIPT,
)


@dataclass(frozen=True)
class ModelOption:
    """One visible model-picker option."""

    label: str
    selected: bool = False
    disabled: bool = False
    source: str = ""


@dataclass(frozen=True)
class ModelPickerState:
    """Visible state of the ChatGPT model picker."""

    opener_label: str = ""
    options: tuple[ModelOption, ...] = ()


@dataclass(frozen=True)
class ModelSelection:
    """Result from trying to select a ChatGPT model option."""

    matched: bool
    selected: str = ""
    reason: str = ""
    options: tuple[ModelOption, ...] = ()


INTENSITY_HINTS: dict[str, tuple[str, ...]] = {
    "fast": ("fast", "instant", "quick", "mini"),
    "normal": ("auto", "normal", "default", "balanced", "chatgpt"),
    "balanced": ("auto", "normal", "balanced", "chatgpt"),
    "deep": ("thinking", "reason", "deep", "pro"),
    "thinking": ("thinking", "reason", "deep"),
    "pro": ("pro", "thinking", "reason"),
}
PLAN_SUFFIX_TERMS: tuple[str, ...] = ("pro",)


def _coerce_options(raw_options: object) -> tuple[ModelOption, ...]:
    if not isinstance(raw_options, list):
        return ()
    options: list[ModelOption] = []
    for item in raw_options:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            continue
        options.append(
            ModelOption(
                label=label.strip(),
                selected=bool(item.get("selected")),
                disabled=bool(item.get("disabled")),
                source=str(item.get("source") or ""),
            )
        )
    return tuple(options)


def _normalize(value: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


def _intensity_hints(intensity: str | None) -> tuple[str, ...]:
    if not intensity:
        return ()
    normalized = intensity.strip().lower()
    return INTENSITY_HINTS.get(normalized, (normalized,))


def _strip_gpt_plan_suffix(words: list[str]) -> list[str]:
    if len(words) > 1 and words[0] == "gpt" and words[-1] in PLAN_SUFFIX_TERMS:
        return words[:-1]
    return words


def _query_aliases(query_norm: str, hint_words: tuple[str, ...]) -> tuple[str, ...]:
    words = query_norm.split()
    hint_set = set(hint_words)
    variants = [words]
    if hint_set:
        without_hints = [word for word in words if word not in hint_set]
        if without_hints != words:
            variants.append(without_hints)
    for variant in variants[:]:
        without_plan_suffix = _strip_gpt_plan_suffix(variant)
        if without_plan_suffix != variant:
            variants.append(without_plan_suffix)
    aliases: list[str] = []
    for variant in variants:
        alias = " ".join(variant)
        if alias and alias not in aliases:
            aliases.append(alias)
    return tuple(aliases)


def selection_query(model: str | None, intensity: str | None) -> str:
    """Build one natural model-picker query from optional model/intensity fields."""
    model_text = (model or "").strip()
    intensity_text = (intensity or "").strip()
    if model_text and intensity_text:
        return f"{model_text} {intensity_text}"
    return model_text or intensity_text


def _score_option(option: ModelOption, query: str, intensity: str | None) -> int:
    norm = _normalize(option.label)
    query_norm = _normalize(query)
    hint_words = tuple(
        word
        for hint in _intensity_hints(intensity)
        for word in _normalize(hint).split()
    )
    score = 0
    for index, alias_norm in enumerate(_query_aliases(query_norm, hint_words)):
        alias_words = alias_norm.split()
        exact_score = 120 if index == 0 else 110
        contains_score = 90 if index == 0 else 80
        words_score = 70 if index == 0 else 60
        if norm == alias_norm:
            score += exact_score
        if alias_norm and alias_norm in norm:
            score += contains_score
        if alias_words and all(word in norm.split() or word in norm for word in alias_words):
            score += words_score
    for hint in hint_words:
        if hint in norm.split() or hint in norm:
            score += 18
    if option.selected:
        score += 5
    if option.disabled:
        score -= 1000
    return score


async def _close_model_menu(page: Page) -> None:
    await page.keyboard.press("Escape")
    await page.wait_for_timeout(120)


async def _open_model_picker(page: Page) -> tuple[str, tuple[ModelOption, ...]]:
    candidates = await page.evaluate(_MARK_OPENERS_SCRIPT)
    if not isinstance(candidates, list):
        return "", ()
    fallback_label = ""
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        index = candidate.get("index")
        label = str(candidate.get("label") or "")
        if fallback_label == "":
            fallback_label = label
        await _close_model_menu(page)
        before = await page.evaluate(_VISIBLE_FINGERPRINTS_SCRIPT)
        opener = await page.query_selector(f'[data-catgpt-model-opener="{index}"]')
        if opener is None:
            continue
        await opener.click(timeout=2000)
        await page.wait_for_timeout(700)
        raw_options = await page.evaluate(_MARK_OPTIONS_SCRIPT, before)
        options = _coerce_options(raw_options)
        if options:
            return label, options
    await _close_model_menu(page)
    return fallback_label, ()


async def inspect_model_picker(page: Page) -> ModelPickerState:
    """Open the live model picker and return visible options."""
    opener_label, options = await _open_model_picker(page)
    await _close_model_menu(page)
    return ModelPickerState(opener_label=opener_label, options=options)


async def select_model_option(
    page: Page,
    *,
    model: str | None = None,
    intensity: str | None = None,
) -> ModelSelection:
    """Select the closest live ChatGPT model-picker option."""
    query = selection_query(model, intensity)
    if not query:
        return ModelSelection(matched=True, reason="no selection requested")
    opener_label, options = await _open_model_picker(page)
    if not options:
        opener_option = ModelOption(label=opener_label, selected=True)
        if _score_option(opener_option, query, intensity) >= 18:
            return ModelSelection(matched=True, selected=opener_label, options=(opener_option,))
        return ModelSelection(matched=False, reason="model picker options not found")
    ranked = sorted(
        enumerate(options),
        key=lambda item: _score_option(item[1], query, intensity),
        reverse=True,
    )
    best_index, best_option = ranked[0]
    if _score_option(best_option, query, intensity) < 18:
        await _close_model_menu(page)
        return ModelSelection(
            matched=False,
            reason=f"no model option matched '{query}'",
            options=options,
        )
    option_handle = await page.query_selector(f'[data-catgpt-model-option="{best_index}"]')
    if option_handle is None:
        await _close_model_menu(page)
        return ModelSelection(matched=False, reason="matched option disappeared", options=options)
    await option_handle.click(timeout=2000)
    await page.wait_for_timeout(800)
    return ModelSelection(matched=True, selected=best_option.label, options=options)
