"""Shared lightweight helpers."""

__all__ = ["normalise_chrom"]


def normalise_chrom(source: str, target: str) -> str:
    """Match the presence of a ``chr`` prefix in a target chromosome name."""
    source_has_prefix = source.startswith("chr")
    target_has_prefix = target.startswith("chr")

    if source_has_prefix and not target_has_prefix:
        return source[3:]
    if not source_has_prefix and target_has_prefix:
        return "chr" + source
    return source
