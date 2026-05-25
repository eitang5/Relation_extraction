"""Module 4: entity coref over the full article via fastcoref.

We only use coref to MERGE event nodes across sentences (two spans referring to
the same happening become one Event node). No Entity layer in v1 — see
project_neo4j memory for the schema decision.
"""

from functools import lru_cache

from . import config
from ._inference import pick_device


@lru_cache(maxsize=1)
def _load():
    device = pick_device(config.DEVICE)
    # fastcoref expects 'cuda' or 'cpu' — MPS isn't supported.
    if device == "mps":
        device = "cpu"
    if config.COREF_MODEL == "fcoref":
        from fastcoref import FCoref

        return FCoref(device=device)
    from fastcoref import LingMessCoref

    return LingMessCoref(device=device)


def coref_article(article_text: str) -> list[list[tuple[int, int]]]:
    """Run coref on a single article. Returns a list of clusters; each cluster
    is a list of (start_char, end_char) tuples into article_text."""
    model = _load()
    preds = model.predict(texts=[article_text])
    clusters = preds[0].get_clusters(as_strings=False)
    return [[tuple(span) for span in cluster] for cluster in clusters]
