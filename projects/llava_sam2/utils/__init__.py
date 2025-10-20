"""Utility subpackage for llava_sam2.

Exports training hooks and helpers so MMEngine can import them via config.
"""

try:
    from .cma_warmup_hook import CMAWarmupHook  # noqa: F401
except Exception:
    # Allow environments where dependencies are partially missing;
    # config's custom_imports will still attempt direct module import.
    CMAWarmupHook = None  # type: ignore
