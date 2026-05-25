"""Vendored from verdius/v0/core/extract.py — keep in sync.

Extended slightly from the v0 original: accepts a `label` parameter on
add_entity / add_relationship so v1 can write distinct :Article and :Event
labels instead of v0's hardcoded :Entity.
"""

from __future__ import annotations

import os
import re
from typing import Any

from neo4j import GraphDatabase


class GraphStorage:
    def __init__(self) -> None:
        self.database = os.environ.get("NEO4J_DATABASE")
        self.driver = GraphDatabase.driver(
            os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.environ.get("NEO4J_USERNAME", "neo4j"),
                os.environ["NEO4J_PASSWORD"],
            ),
        )

    def close(self) -> None:
        self.driver.close()

    def add_entity(self, entity_id: str, label: str = "Entity", **properties: Any) -> None:
        label = _safe_label(label)
        with self.driver.session(database=self.database) as session:
            session.run(
                f"""
                MERGE (n:{label} {{id: $entity_id}})
                SET n += $properties
                """,
                entity_id=entity_id,
                properties=properties,
            )

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        source_label: str = "Entity",
        target_label: str = "Entity",
        **properties: Any,
    ) -> None:
        relationship_type = _safe_relationship_type(relationship_type)
        source_label = _safe_label(source_label)
        target_label = _safe_label(target_label)

        with self.driver.session(database=self.database) as session:
            session.run(
                f"""
                MATCH (source:{source_label} {{id: $source_id}})
                MATCH (target:{target_label} {{id: $target_id}})
                MERGE (source)-[relationship:{relationship_type}]->(target)
                SET relationship += $properties
                """,
                source_id=source_id,
                target_id=target_id,
                properties=properties,
            )


def _safe_relationship_type(value: str) -> str:
    value = value.strip().upper().replace(" ", "_").replace("-", "_")
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", value):
        raise ValueError(f"Invalid relationship type: {value!r}")
    return value


def _safe_label(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid node label: {value!r}")
    return value
