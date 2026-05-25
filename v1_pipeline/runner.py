"""Module 8: CLI runner.

Read a JSONL of articles, run the full pipeline on each, write one JSON file
per article. Optionally also write each article to Neo4j.

JSONL format (one article per line):
    {"id": "a1", "text": "...", "title": "...", "url": "...", "publication_date": "..."}

Only `id` and `text` are required; the rest become article metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .graph_build import build_article_graph


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m v1_pipeline.runner")
    parser.add_argument("--input", required=True, help="JSONL file (one article per line).")
    parser.add_argument("--output-dir", required=True, help="Directory for per-article JSON.")
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="Also write to Neo4j (requires NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD).",
    )
    parser.add_argument("--limit", type=int, help="Process at most N articles.")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    storage = None
    writer = None
    if args.neo4j:
        from .graph_storage import GraphStorage
        from .neo4j_writer import write_article_to_neo4j

        storage = GraphStorage()
        writer = write_article_to_neo4j

    total = 0
    total_ms = 0
    try:
        with open(args.input) as f:
            for line in f:
                if args.limit and total >= args.limit:
                    break
                line = line.strip()
                if not line:
                    continue

                article = json.loads(line)
                article_id = article["id"]
                article_text = article["text"]
                metadata = {k: v for k, v in article.items() if k not in ("id", "text")}

                graph = build_article_graph(article_id, article_text, metadata)

                out_path = out_dir / f"{article_id}.json"
                out_path.write_text(json.dumps(graph, indent=2))

                if writer is not None:
                    writer(graph, storage=storage)

                stats = graph["stats"]
                total_ms += stats["processing_ms"]
                total += 1
                print(
                    f"[{total}] {article_id}: "
                    f"{stats['n_sentences']}s → "
                    f"{stats['n_after_st0']} causal → "
                    f"{stats['n_events']} events / {stats['n_edges']} edges "
                    f"({stats['processing_ms']}ms)",
                    flush=True,
                )
    finally:
        if storage is not None:
            storage.close()

    throughput = (total / (total_ms / 1000.0)) if total_ms else 0.0
    print(
        f"\nDone. {total} articles → {out_dir} "
        f"({total_ms}ms total, {throughput:.1f} articles/sec).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
