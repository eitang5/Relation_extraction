"""Module 7a: assemble the article-level graph as a JSON-serializable dict.

Orchestrates modules 1-6 end-to-end on a single article and produces the
canonical output structure that gets written to disk and loaded into Neo4j.
"""

import hashlib
import time

from .coref import coref_article
from .sentence_split import split_sentences
from .st0_filter import filter_sentences
from .st1_classify import classify_relations
from .st2_extract import extract_triples_batch


_RELATION_TO_EDGE_TYPE = {
    "cause": "CAUSES",
    "enable": "ENABLES",
    "prevent": "PREVENTS",
    "intend": "INTENDS",
}


def build_article_graph(
    article_id: str,
    article_text: str,
    article_metadata: dict | None = None,
) -> dict:
    """Run the full per-article pipeline. Returns the article graph dict."""
    t0 = time.time()

    # 1. Sentence split (with idx attached for traceability).
    sentences = split_sentences(article_text)
    for i, s in enumerate(sentences):
        s["idx"] = i

    # 2. st0 filter.
    survivors = filter_sentences(sentences)

    # 3a. st1 relation type — drops sentences st1 considers no_relation.
    rel_results = classify_relations(survivors)

    # 3b. st2 event-pair extraction on the sentences that st1 kept.
    kept_sentences = [r["sentence"] for r in rel_results]
    triple_results = extract_triples_batch(kept_sentences)

    # 4. Coref over the FULL article (not just survivors — pronouns may have
    #    antecedents in sentences st0 dropped).
    clusters = coref_article(article_text)

    # 5+6+7. Resolve, dedup, build event nodes and causal edges.
    events: dict[str, dict] = {}
    edges: list[dict] = []
    n_skipped_missing_span = 0
    n_skipped_unmatched_span = 0
    n_st2_triples = 0

    for rel, st2_result in zip(rel_results, triple_results):
        sentence = rel["sentence"]
        sent_start = sentence["start"]
        triples = st2_result.get("triples", [])
        n_st2_triples += len(triples)

        # Need at least one subject/object pair to form a causal edge.
        if not triples:
            n_skipped_missing_span += 1
            continue

        for triple in triples:
            subj_span = triple.get("subject_span")
            obj_span = triple.get("object_span")
            if subj_span is None or obj_span is None:
                n_skipped_unmatched_span += 1

            subj_global = (
                (sent_start + subj_span[0], sent_start + subj_span[1])
                if subj_span is not None
                else None
            )
            obj_global = (
                (sent_start + obj_span[0], sent_start + obj_span[1])
                if obj_span is not None
                else None
            )

            subj_id, subj_name = _resolve_event(triple["subject"])
            obj_id, obj_name = _resolve_event(triple["object"])

            _register_event(events, subj_id, subj_name, article_id, subj_global, sentence["idx"])
            _register_event(events, obj_id, obj_name, article_id, obj_global, sentence["idx"])

            edges.append(
                {
                    "source": subj_id,
                    "target": obj_id,
                    "relation_type": _RELATION_TO_EDGE_TYPE[rel["relation_type"]],
                    "confidence": rel["confidence"],
                    "source_sentence": sentence["text"],
                    "sentence_idx": sentence["idx"],
                    "st2_subject": triple["subject"],
                    "st2_object": triple["object"],
                    "st2_raw_relation": triple.get("raw_relation", ""),
                }
            )

    processing_ms = int((time.time() - t0) * 1000)

    return {
        "article": {
            "id": article_id,
            "text": article_text,
            "metadata": article_metadata or {},
        },
        "sentences": sentences,
        "events": list(events.values()),
        "edges": edges,
        "stats": {
            "n_sentences": len(sentences),
            "n_after_st0": len(survivors),
            "n_after_st1": len(rel_results),
            "n_st2_triples": n_st2_triples,
            "n_skipped_missing_span": n_skipped_missing_span,
            "n_skipped_unmatched_span": n_skipped_unmatched_span,
            "n_events": len(events),
            "n_edges": len(edges),
            "n_coref_clusters": len(clusters),
            "processing_ms": processing_ms,
        },
    }


def _resolve_event(span_text: str) -> tuple[str, str]:
    """Return a global event id and canonical name.

    Event ids are based on normalized event text, not article id, so repeated
    substantive event phrases can merge across articles in Neo4j.
    """
    key = _normalise_event_text(span_text)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"event:{digest}", span_text


def _register_event(
    events: dict[str, dict],
    event_id: str,
    name: str,
    article_id: str,
    span: tuple[int, int] | None,
    sentence_idx: int,
) -> None:
    if event_id not in events:
        events[event_id] = {
            "id": event_id,
            "name": name,  # canonical name = first span text seen
            "canonical_key": _normalise_event_text(name),
            "article_id": article_id,
            "article_ids": [article_id],
            "spans": [],
            "sentence_indices": [],
        }
    if article_id not in events[event_id]["article_ids"]:
        events[event_id]["article_ids"].append(article_id)
    if span is not None:
        events[event_id]["spans"].append([span[0], span[1]])
    if sentence_idx not in events[event_id]["sentence_indices"]:
        events[event_id]["sentence_indices"].append(sentence_idx)


def _normalise_event_text(text: str) -> str:
    return " ".join(text.casefold().split())
