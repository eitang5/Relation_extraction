"""Module 1: sentence splitting.

Splits an article into sentences with character offsets into the original text.
Offsets are needed so downstream span outputs (from st2) can be mapped back to
article-global positions for entity-coref resolution.
"""

from functools import lru_cache

import spacy
from spacy.language import Language


@lru_cache(maxsize=1)
def _nlp() -> Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def split_sentences(text: str) -> list[dict]:
    """Split `text` into sentences with character offsets.

    Returns a list of {"text": str, "start": int, "end": int} where start/end
    are character offsets into the original `text` such that
    text[start:end] == sentence_text.
    """
    doc = _nlp()(text)
    return [
        {"text": sent.text, "start": sent.start_char, "end": sent.end_char}
        for sent in doc.sents
    ]


if __name__ == "__main__":
    sample = (
        "The United Nations passed a resolution against child soldiers. "
        "As a result, several countries ratified the new treaty. "
        "The vote was unanimous."
    )
    for s in split_sentences(sample):
        print(f"[{s['start']:>3}:{s['end']:>3}] {s['text']!r}")
