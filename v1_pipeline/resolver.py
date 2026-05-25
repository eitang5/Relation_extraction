"""Module 5: match st2 spans to coref clusters by char-offset overlap.

Used to decide whether two st2 spans across different sentences refer to the
same event (they fall into the same coref cluster) or are distinct.
"""

from . import config


def find_cluster_for_span(
    span: tuple[int, int],
    clusters: list[list[tuple[int, int]]],
    threshold: float | None = None,
) -> int | None:
    """Return the index of the cluster that best overlaps `span`, or None if
    no cluster mention overlaps above the IoU threshold."""
    thresh = threshold if threshold is not None else config.SPAN_OVERLAP_THRESHOLD
    best_idx: int | None = None
    best_iou = 0.0
    for ci, cluster in enumerate(clusters):
        for mention in cluster:
            iou = _iou(span, mention)
            if iou > best_iou:
                best_iou = iou
                best_idx = ci
    return best_idx if best_iou >= thresh else None


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter_s = max(a[0], b[0])
    inter_e = min(a[1], b[1])
    if inter_e <= inter_s:
        return 0.0
    inter = inter_e - inter_s
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0
