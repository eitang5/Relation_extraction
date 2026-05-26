"""Shared inference helpers — device pick + a batching iterator."""


def pick_device(requested: str = "auto") -> str:
    """Resolve 'auto' to the best available device."""
    import torch

    if requested == "cuda":
        return "cuda"
    if requested == "mps":
        return "mps"
    if requested == "cpu":
        return "cpu"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
