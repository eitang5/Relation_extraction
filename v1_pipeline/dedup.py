"""Module 6: event dedup via head-noun matching.

Fallback merge key for spans that don't fall into any coref cluster. Take the
last alphabetic word of the span as the head noun and lowercase it. Two spans
with the same head noun merge into one Event.

Crude (a "destruction" event and "the destruction of homes" event collapse),
but acceptable for v1. Real event coref ships in v2.
"""

import re

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def head_noun(text: str) -> str:
    words = _WORD_RE.findall(text)
    return words[-1].lower() if words else text.strip().lower()
