"""Module 4: entity coref over the full article via fastcoref.

We only use coref to MERGE event nodes across sentences (two spans referring to
the same happening become one Event node). No Entity layer in v1 — see
project_neo4j memory for the schema decision.
"""

from functools import lru_cache

from . import config
from ._inference import pick_device


def _patch_longformer_eager_attention() -> None:
    """Force Longformer to use eager attention.

    fastcoref's coref models are Longformer-based, and recent transformers
    versions try SDPA by default — but Longformer has no SDPA implementation,
    so the load crashes with `ValueError: LongformerModel does not support an
    attention implementation through torch.nn.functional.scaled_dot_product_attention`.
    fastcoref doesn't expose `attn_implementation`, so we patch it on at the
    class level once.
    """
    from transformers import LongformerModel

    if getattr(LongformerModel, "_v1_eager_patched", False):
        return
    orig_from_pretrained = LongformerModel.from_pretrained

    @classmethod
    def patched(cls, *args, **kwargs):
        kwargs.setdefault("attn_implementation", "eager")
        return orig_from_pretrained(*args, **kwargs)

    LongformerModel.from_pretrained = patched
    LongformerModel._v1_eager_patched = True


@lru_cache(maxsize=1)
def _load():
    _patch_longformer_eager_attention()

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
