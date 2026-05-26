"""Module 3a: st1 relation type classifier.

For each causal sentence (from st0 filter), predict the relation type and a
confidence (softmax probability of the argmax class). Sentences classified as
"no_relation" are dropped from the output — they shouldn't happen often after
st0 already filtered, but the two models disagree sometimes.
"""

from functools import lru_cache

from . import config
from ._inference import batches, pick_device


@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = pick_device(config.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(config.ST1_CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(config.ST1_CKPT)
    model.eval().to(device)
    return tokenizer, model, device, torch


def classify_relations(sentences: list[dict], batch_size: int | None = None) -> list[dict]:
    """Run st1 on each sentence. Return one row per kept sentence:

        {
          "sentence": <original sentence dict>,
          "relation_type": "cause" | "enable" | "prevent" | "intend",
          "confidence": float in [0, 1]
        }

    Sentences predicted as "no_relation" are dropped.
    """
    if not sentences:
        return []
    tokenizer, model, device, torch = _load()
    bs = batch_size or config.ST1_BATCH
    id2label = model.config.id2label

    out: list[dict] = []
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
            probs = torch.softmax(logits, dim=-1)
        argmax = logits.argmax(-1)
        confs = probs.gather(1, argmax.unsqueeze(-1)).squeeze(-1).cpu().tolist()
        preds = argmax.cpu().tolist()

        for sentence, pred, conf in zip(batch, preds, confs):
            label = id2label[pred]
            if label == "no_relation":
                continue
            out.append(
                {
                    "sentence": sentence,
                    "relation_type": label,
                    "confidence": float(conf),
                }
            )
    return out
