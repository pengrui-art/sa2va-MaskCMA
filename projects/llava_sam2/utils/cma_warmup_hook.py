from __future__ import annotations

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class CMAWarmupHook(Hook):
    """
    Gradually increases CMA residual strength during the early phase of training.

    This helps avoid early over-conditioning from cross-modal injection
    (especially with multi-token routing) that could hurt convergence on RefCOCO.

    Args:
        warmup_ratio: Fraction of total iters used for warmup (0~1).
    """

    def __init__(self, warmup_ratio: float = 0.1) -> None:
        assert 0.0 <= warmup_ratio <= 1.0
        self.warmup_ratio = warmup_ratio
        self._cma_modules = []

    def before_train(self, runner) -> None:
        # Discover SAM2TrainRunner modules once training starts
        try:
            from projects.llava_sam2.models.sam2_train import SAM2TrainRunner
        except Exception:
            SAM2TrainRunner = None

        if SAM2TrainRunner is None:
            return

        self._cma_modules.clear()
        for m in runner.model.modules():
            if isinstance(m, SAM2TrainRunner) and hasattr(m, "set_cma_warmup_scale"):
                self._cma_modules.append(m)

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        max_iters = getattr(runner, "max_iters", None)
        if not max_iters or max_iters <= 0:
            return
        warmup_iters = int(max_iters * self.warmup_ratio)
        cur_iter = runner.iter + 1  # 0-indexed in mmengine
        scale = 1.0 if warmup_iters <= 0 else min(1.0, cur_iter / warmup_iters)
        for mod in self._cma_modules:
            mod.set_cma_warmup_scale(scale)
