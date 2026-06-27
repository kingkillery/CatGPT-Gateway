"""JavaScript snippets used to inspect the ChatGPT model picker."""

from __future__ import annotations

_MARK_OPENERS_SCRIPT = r"""
() => {
  const textOf = (el) => [
    el.innerText || '',
    el.getAttribute('aria-label') || '',
    el.getAttribute('title') || '',
    el.getAttribute('data-testid') || '',
  ].join(' ').replace(/\s+/g, ' ').trim();
  const visible = (el) => {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 0 && rect.height > 0;
  };
  const score = (el) => {
    const label = textOf(el).toLowerCase();
    const normalized = label.replace(/[^a-z0-9]+/g, ' ').trim();
    const testid = (el.getAttribute('data-testid') || '').toLowerCase();
    const tag = el.tagName.toLowerCase();
    const role = el.getAttribute('role') || '';
    const rect = el.getBoundingClientRect();
    let value = 0;
    if (tag === 'button' || role === 'button') value += 8;
    if (tag === 'a') value -= 60;
    if (testid.includes('model')) value += 90;
    if (testid.includes('selector') || testid.includes('switcher')) value += 35;
    if (/^(pro|auto|default|chatgpt|gpt|gpt [0-9a-z .-]+|o[0-9a-z .-]+)$/.test(normalized)) value += 110;
    if (normalized.includes('model')) value += 30;
    if (normalized.includes('gpt') && label.length < 80) value += 30;
    if (normalized.includes('reason') || normalized.includes('thinking')) value += 20;
    if (normalized.includes('auto') || normalized.includes('pro')) value += 14;
    if (rect.left < 260 && !/^(pro|auto|default|chatgpt|gpt|o[0-9])/.test(normalized)) value -= 45;
    if (/organize|home|profile|account|project|conversation|history|pin|temporary chat|voice|send|attach|tools/.test(normalized)) value -= 95;
    return value;
  };
  document.querySelectorAll('[data-catgpt-model-opener]').forEach((el) => el.removeAttribute('data-catgpt-model-opener'));
  const candidates = Array.from(document.querySelectorAll('button,[role="button"],a'))
    .filter(visible)
    .map((el) => ({ el, label: textOf(el), score: score(el) }))
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 12);
  candidates.forEach((item, idx) => item.el.setAttribute('data-catgpt-model-opener', String(idx)));
  return candidates.map((item, idx) => ({ index: idx, label: item.label, score: item.score }));
}
"""

_VISIBLE_FINGERPRINTS_SCRIPT = r"""
() => Array.from(document.querySelectorAll('button,[role="button"],[role="menuitem"],[role="option"],[role="radio"],a'))
  .filter((el) => {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 0 && rect.height > 0;
  })
  .map((el) => `${el.tagName}:${el.getAttribute('role') || ''}:${(el.innerText || el.getAttribute('aria-label') || '').replace(/\s+/g, ' ').trim()}`)
"""

_MARK_OPTIONS_SCRIPT = r"""
(before) => {
  const beforeSet = new Set(Array.isArray(before) ? before : []);
  const textOf = (el) => [
    el.innerText || '',
    el.getAttribute('aria-label') || '',
    el.getAttribute('title') || '',
    el.getAttribute('data-testid') || '',
  ].join(' ').replace(/\s+/g, ' ').trim();
  const visible = (el) => {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 0 && rect.height > 0;
  };
  const modelish = (label, role, testid) => {
    const lower = label.toLowerCase();
    return /gpt|model|auto|thinking|reason|pro|mini|default|o[0-9]/.test(lower)
      || /model/i.test(testid)
      || ['menuitem', 'option', 'radio'].includes(role);
  };
  document.querySelectorAll('[data-catgpt-model-option]').forEach((el) => el.removeAttribute('data-catgpt-model-option'));
  const seen = new Set();
  const options = [];
  for (const el of Array.from(document.querySelectorAll('button,[role="button"],[role="menuitem"],[role="option"],[role="radio"],a')).filter(visible)) {
    const label = textOf(el);
    if (!label || label.length > 220) continue;
    const role = el.getAttribute('role') || el.tagName.toLowerCase();
    const testid = el.getAttribute('data-testid') || '';
    const fingerprint = `${el.tagName}:${el.getAttribute('role') || ''}:${(el.innerText || el.getAttribute('aria-label') || '').replace(/\s+/g, ' ').trim()}`;
    const inPopup = Boolean(el.closest('[role="menu"],[role="listbox"],[role="dialog"],[data-radix-popper-content-wrapper]'));
    const isNew = !beforeSet.has(fingerprint);
    if (!(isNew || inPopup || ['menuitem', 'option', 'radio'].includes(role)) || !modelish(label, role, testid)) continue;
    const key = label.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    const index = options.length;
    el.setAttribute('data-catgpt-model-option', String(index));
    options.push({
      label,
      selected: el.getAttribute('aria-checked') === 'true' || el.getAttribute('aria-selected') === 'true',
      disabled: el.getAttribute('aria-disabled') === 'true' || el.disabled === true,
      source: `${role}${testid ? ':' + testid : ''}`,
      index,
    });
  }
  return options;
}
"""
