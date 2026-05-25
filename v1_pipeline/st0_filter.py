"""Module 2: st0 binary causal filter.

Drops sentences that st0 classifies as label 0 (no_relation). Keeps those
classified as label 1 (causal). Per the trained model's config, label 1 is the
positive (causal) class.
"""

from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from . import config
from ._inference import batches, pick_device


@lru_cache(maxsize=1)
def _load():
    device = pick_device(config.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(config.ST0_CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(config.ST0_CKPT)
    model.eval().to(device)
    return tokenizer, model, device


def filter_sentences(sentences: list[dict], batch_size: int | None = None) -> list[dict]:
    """Return only the sentences st0 predicts as causal (label == 1).

    Input  : list of {"text", "start", "end", "idx"} dicts (from sentence_split).
    Output : same dicts, filtered.
    """
    if not sentences:
        return []
    tokenizer, model, device = _load()
    bs = batch_size or config.ST0_BATCH

    keep_mask: list[bool] = []
    for batch in batches(sentences, bs):
        texts = [s["text"] for s in batch]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_LEN,
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        preds = logits.argmax(-1).cpu().tolist()
        keep_mask.extend(p == 1 for p in preds)

    return [s for s, keep in zip(sentences, keep_mask) if keep]
