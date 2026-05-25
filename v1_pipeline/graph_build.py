"""Module 7a: assemble the article-level graph as a JSON-serializable dict.

Orchestrates modules 1-6 end-to-end on a single article and produces the
canonical output structure that gets written to disk and loaded into Neo4j.
"""

import hashlib
import time

from .coref import coref_article
from .dedup import head_noun
from .resolver import find_cluster_for_span
from .sentence_split import split_sentences
from .st0_filter import filter_sentences
from .st1_classify import classify_relations
from .st2_extract import extract_spans_batch


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

    # 3b. st2 BIO spans on the sentences that st1 kept.
    kept_sentences = [r["sentence"] for r in rel_results]
    span_results = extract_spans_batch(kept_sentences)

    # 4. Coref over the FULL article (not just survivors — pronouns may have
    #    antecedents in sentences st0 dropped).
    clusters = coref_article(article_text)

    # 5+6+7. Resolve, dedup, build event nodes and causal edges.
    events: dict[str, dict] = {}
    edges: list[dict] = []
    n_skipped_missing_span = 0

    for rel, spans in zip(rel_results, span_results):
        sentence = rel["sentence"]
        sent_start = sentence["start"]

        # Need at least one SUBJ and one OBJ to form a causal edge.
        if not spans["subj_spans"] or not spans["obj_spans"]:
            n_skipped_missing_span += 1
            continue

        # Take the first SUBJ and first OBJ as primary. st2 occasionally emits
        # multiple; for v1 we keep it simple.
        subj_local = spans["subj_spans"][0]  # (local_start, local_end, text)
        obj_local = spans["obj_spans"][0]

        subj_global = (sent_start + subj_local[0], sent_start + subj_local[1])
        obj_global = (sent_start + obj_local[0], sent_start + obj_local[1])

        subj_id, subj_name = _resolve_event(article_id, subj_local[2], subj_global, clusters)
        obj_id, obj_name = _resolve_event(article_id, obj_local[2], obj_global, clusters)

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
            "n_skipped_missing_span": n_skipped_missing_span,
            "n_events": len(events),
            "n_edges": len(edges),
            "n_coref_clusters": len(clusters),
            "processing_ms": processing_ms,
        },
    }


def _resolve_event(
    article_id: str,
    span_text: str,
    span_global: tuple[int, int],
    clusters: list[list[tuple[int, int]]],
) -> tuple[str, str]:
    """Return (event_id, canonical_name) for a span. Uses coref cluster index
    when available, else falls back to head-noun string match."""
    cluster_idx = find_cluster_for_span(span_global, clusters)
    if cluster_idx is not None:
        key = f"cluster:{cluster_idx}"
    else:
        key = f"noun:{head_noun(span_text)}"
    digest = hashlib.sha1(f"{article_id}:{key}".encode("utf-8")).hexdigest()[:12]
    return f"event:{article_id}:{digest}", span_text


def _register_event(
    events: dict[str, dict],
    event_id: str,
    name: str,
    article_id: str,
    span: tuple[int, int],
    sentence_idx: int,
) -> None:
    if event_id not in events:
        events[event_id] = {
            "id": event_id,
            "name": name,  # canonical name = first span text seen
            "article_id": article_id,
            "spans": [],
            "sentence_indices": [],
        }
    events[event_id]["spans"].append([span[0], span[1]])
    if sentence_idx not in events[event_id]["sentence_indices"]:
        events[event_id]["sentence_indices"].append(sentence_idx)
