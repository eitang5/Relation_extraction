"""Module 3b: st2 BIO span extractor for SUBJ and OBJ.

For each sentence, runs BERT-large-NER BIO tagger and walks the predicted tags
to recover SUBJ and OBJ spans. Returns char offsets within the SENTENCE (caller
is responsible for adding sentence.start to convert to article-global offsets).
"""

from functools import lru_cache

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from . import config
from ._inference import batches, pick_device


@lru_cache(maxsize=1)
def _load():
    device = pick_device(config.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(config.ST2_CKPT)
    model = AutoModelForTokenClassification.from_pretrained(config.ST2_CKPT)
    model.eval().to(device)
    return tokenizer, model, device


def extract_spans_batch(sentences: list[dict], batch_size: int | None = None) -> list[dict]:
    """Run st2 on each sentence. Returns parallel list:

        [{"sentence": <orig>, "subj_spans": [(s,e,text), ...], "obj_spans": [...]}, ...]

    Span offsets are within the sentence text (0 = first char of sentence).
    """
    if not sentences:
        return []
    tokenizer, model, device = _load()
    bs = batch_size or config.ST2_BATCH
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
            return_offsets_mapping=True,
        )
        offsets_batch = enc.pop("offset_mapping").tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            preds = model(**enc).logits.argmax(-1).cpu().tolist()

        for sentence, text, pred_seq, offsets in zip(batch, texts, preds, offsets_batch):
            labels = [id2label[p] for p in pred_seq]
            spans = _walk_bio(labels, offsets, text)
            out.append(
                {
                    "sentence": sentence,
                    "subj_spans": spans["SUBJ"],
                    "obj_spans": spans["OBJ"],
                }
            )
    return out


def _walk_bio(labels: list[str], offsets: list[tuple[int, int]], text: str) -> dict:
    """Walk BIO tags + token offsets to recover contiguous SUBJ/OBJ spans.

    Returns {"SUBJ": [(start, end, text), ...], "OBJ": [...]} with char offsets
    into `text`.
    """
    spans: dict[str, list] = {"SUBJ": [], "OBJ": []}
    cur_type: str | None = None
    cur_start: int | None = None
    cur_end: int | None = None

    def close():
        nonlocal cur_type, cur_start, cur_end
        if cur_type is not None and cur_start is not None and cur_end is not None:
            spans[cur_type].append((cur_start, cur_end, text[cur_start:cur_end]))
        cur_type = cur_start = cur_end = None

    for label, (off_s, off_e) in zip(labels, offsets):
        # Special tokens (CLS, SEP, PAD) get (0, 0) offsets — skip.
        if off_s == 0 and off_e == 0:
            continue
        if label.startswith("B-"):
            close()
            cur_type = label[2:]
            cur_start, cur_end = off_s, off_e
        elif label.startswith("I-") and cur_type == label[2:]:
            cur_end = off_e
        else:
            close()
    close()
    return spans
