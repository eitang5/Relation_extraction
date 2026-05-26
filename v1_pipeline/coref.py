"""Module 4: entity coref over the full article via fastcoref.

We only use coref to MERGE event nodes across sentences (two spans referring to
the same happening become one Event node). No Entity layer in v1 — see
project_neo4j memory for the schema decision.
"""

from functools import lru_cache

from . import config
from ._inference import pick_device


def _patch_longformer_eager_attention() -> None:
    """Force Longformer (and any model lacking SDPA) to silently fall back to eager.

    fastcoref's coref models are Longformer-based, and recent transformers
    versions explicitly request SDPA for them — but Longformer has no SDPA
    implementation, so transformers raises `ValueError: LongformerModel does
    not support an attention implementation through ... sdpa`. Patching
    from_pretrained kwargs isn't enough because fastcoref's load path requests
    SDPA via the config. So we patch the check itself: when SDPA is requested
    but unsupported, silently downgrade to eager instead of raising.

    Also marks Longformer as `_supports_sdpa = True` so the hard check passes;
    Longformer's actual forward uses its custom sliding-window attention and
    ignores the SDPA flag, so the lie is harmless.
    """
    from transformers import LongformerModel, PreTrainedModel

    # Belt: lie about SDPA support so the hard_check_only branch doesn't raise.
    LongformerModel._supports_sdpa = True

    # Suspenders: patch the check to swallow the ValueError if it ever fires.
    if getattr(PreTrainedModel, "_v1_sdpa_patched", False):
        return
    orig_check = PreTrainedModel._check_and_enable_sdpa

    @classmethod
    def patched_check(cls, config, hard_check_only: bool = False):
        try:
            return orig_check.__func__(cls, config, hard_check_only)
        except (ValueError, ImportError):
            config._attn_implementation = "eager"
            return config

    PreTrainedModel._check_and_enable_sdpa = patched_check
    PreTrainedModel._v1_sdpa_patched = True


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
    if not preds:
        return []
    clusters = preds[0].get_clusters(as_strings=False)
    return [[tuple(span) for span in cluster] for cluster in clusters]
