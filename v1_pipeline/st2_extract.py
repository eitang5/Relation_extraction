"""Module 3b: REBEL event-pair extractor."""

from . import config
from ._inference import batches, pick_device


from functools import lru_cache


@lru_cache(maxsize=1)
def _load_rebel():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = pick_device(config.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(config.ST2_REBEL_CKPT)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.ST2_REBEL_CKPT)
    model.eval().to(device)
    return tokenizer, model, device, torch


def extract_triples_batch(sentences: list[dict], batch_size: int | None = None) -> list[dict]:
    """Run configured st2 backend. Returns one row per input sentence:

        [{"sentence": <orig>, "triples": [{"subject": str, "object": str,
          "raw_relation": str, "subject_span": (s,e) | None,
          "object_span": (s,e) | None}], ...}, ...]
    """
    backend = config.ST2_BACKEND.strip().lower()
    if backend == "rebel":
        return extract_rebel_triples_batch(sentences, batch_size=batch_size)
    raise ValueError(f"Unsupported ST2_BACKEND: {config.ST2_BACKEND!r}")


def extract_rebel_triples_batch(sentences: list[dict], batch_size: int | None = None) -> list[dict]:
    """Run REBEL and parse generated relation triples."""
    if not sentences:
        return []
    tokenizer, model, device, torch = _load_rebel()
    bs = batch_size or config.ST2_BATCH

    out: list[dict] = []
    for batch in batches(sentences, bs):
        texts = [s["text"] for s in batch]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.ST2_REBEL_MAX_LEN,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_new_tokens=config.ST2_REBEL_MAX_NEW_TOKENS,
                num_beams=3,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)

        for sentence, generated_text in zip(batch, decoded):
            triples = []
            seen = set()
            for triple in parse_rebel_triples(generated_text):
                subject_span = _find_text_span(sentence["text"], triple["subject"])
                object_span = _find_text_span(sentence["text"], triple["object"])
                key = (
                    _norm_text(triple["subject"]),
                    _norm_text(triple["object"]),
                    _norm_text(triple["raw_relation"]),
                )
                if key in seen:
                    continue
                seen.add(key)
                triples.append(
                    {
                        **triple,
                        "subject_span": subject_span,
                        "object_span": object_span,
                    }
                )
            out.append({"sentence": sentence, "triples": triples})
    return out


def parse_rebel_triples(text: str) -> list[dict]:
    """Parse REBEL output into subject/object/relation triples.

    REBEL linearizes as:
        <triplet> subject <subj> object <obj> relation
    and can repeat that block multiple times.
    """
    text = (
        text.replace("<s>", " ")
        .replace("</s>", " ")
        .replace("<pad>", " ")
        .strip()
    )
    triples: list[dict] = []
    current: str | None = None
    subject = ""
    obj = ""
    relation = ""

    def close() -> None:
        nonlocal subject, obj, relation
        subj_clean = _clean_generated_field(subject)
        obj_clean = _clean_generated_field(obj)
        rel_clean = _clean_generated_field(relation)
        if subj_clean and obj_clean:
            triples.append(
                {
                    "subject": subj_clean,
                    "object": obj_clean,
                    "raw_relation": rel_clean,
                }
            )
        subject = ""
        obj = ""
        relation = ""

    for token in text.split():
        if token == "<triplet>":
            close()
            current = "subject"
        elif token == "<subj>":
            current = "object"
        elif token == "<obj>":
            current = "relation"
        elif current == "subject":
            subject = f"{subject} {token}"
        elif current == "object":
            obj = f"{obj} {token}"
        elif current == "relation":
            relation = f"{relation} {token}"
    close()
    return triples


def _find_text_span(text: str, needle: str) -> tuple[int, int] | None:
    if not needle:
        return None
    start = text.lower().find(needle.lower())
    if start == -1:
        return None
    return start, start + len(needle)


def _clean_generated_field(value: str) -> str:
    return " ".join(value.replace("</triplet>", " ").split())


def _norm_text(value: str) -> str:
    return " ".join(value.casefold().split())
