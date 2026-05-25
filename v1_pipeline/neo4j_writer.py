"""Module 7b: write an article graph to Neo4j.

Reads the JSON produced by graph_build.build_article_graph and writes one
:Article node, one :Event node per event, a :CONTAINS edge from article to
each event, and a typed causal edge per triple. All writes use MERGE so
re-running on the same article is idempotent.

Neo4j property type limitation: properties cannot be lists of lists. The
events[].spans field (list of [start,end] pairs) is JSON-stringified when
written to the graph, so a cypher consumer reads back via apoc.convert.fromJsonList
or just stores the string.
"""

import json
from datetime import datetime, timezone

from .graph_storage import GraphStorage


def write_article_to_neo4j(article_graph: dict, storage: GraphStorage | None = None) -> dict:
    """Idempotently write one article's graph to Neo4j. Returns counts."""
    owns_storage = storage is None
    storage = storage or GraphStorage()

    written = {"articles": 0, "events": 0, "contains_edges": 0, "causal_edges": 0}
    try:
        article = article_graph["article"]
        article_id = article["id"]
        metadata = article.get("metadata") or {}
        article_node_id = f"article:{article_id}"

        storage.add_entity(
            article_node_id,
            label="Article",
            article_id=article_id,
            title=metadata.get("title", ""),
            url=metadata.get("url", ""),
            publication_date=metadata.get("publication_date", ""),
            ingested_at=datetime.now(timezone.utc).isoformat(),
            n_sentences=article_graph["stats"]["n_sentences"],
            n_events=article_graph["stats"]["n_events"],
            n_edges=article_graph["stats"]["n_edges"],
        )
        written["articles"] += 1

        for event in article_graph["events"]:
            storage.add_entity(
                event["id"],
                label="Event",
                name=event["name"],
                article_id=event["article_id"],
                spans_json=json.dumps(event["spans"]),
                sentence_indices=event["sentence_indices"],
            )
            written["events"] += 1

            storage.add_relationship(
                article_node_id,
                event["id"],
                "CONTAINS",
                source_label="Article",
                target_label="Event",
            )
            written["contains_edges"] += 1

        for edge in article_graph["edges"]:
            storage.add_relationship(
                edge["source"],
                edge["target"],
                edge["relation_type"],
                source_label="Event",
                target_label="Event",
                confidence=edge["confidence"],
                source_sentence=edge["source_sentence"],
                sentence_idx=edge["sentence_idx"],
                article_id=article_id,
            )
            written["causal_edges"] += 1

        return written
    finally:
        if owns_storage:
            storage.close()
