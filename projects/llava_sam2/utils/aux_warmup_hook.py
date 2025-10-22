from __future__ import annotations

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class AuxLossWarmupHook(Hook):
    """
    Linearly increases auxiliary loss strength (e.g., TMC/boundary) during the
    early phase of training to avoid destabilizing the main optimization.

    Args:
        warmup_ratio: Fraction of total iters used for warmup (0~1).
    """

    def __init__(self, warmup_ratio: float = 0.0) -> None:
        assert 0.0 <= warmup_ratio <= 1.0
        self.warmup_ratio = warmup_ratio
        self._targets = []

    def before_train(self, runner) -> None:
        if self.warmup_ratio <= 0:
            return
        # Discover VideoLLaVASAMModel-like modules once
        self._targets.clear()
        for m in runner.model.modules():
            if hasattr(m, "set_aux_loss_warmup_scale") and callable(
                m.set_aux_loss_warmup_scale
            ):
                self._targets.append(m)

    def after_train_iter(
        self, runner, batch_idx: int, data_batch=None, outputs=None
    ) -> None:
        if self.warmup_ratio <= 0 or not self._targets:
            return
        max_iters = getattr(runner, "max_iters", None)
        if not max_iters or max_iters <= 0:
            return
        warmup_iters = int(max_iters * self.warmup_ratio)
        cur_iter = runner.iter + 1
        scale = 1.0 if warmup_iters <= 0 else min(1.0, cur_iter / warmup_iters)
        for mod in self._targets:
            mod.set_aux_loss_warmup_scale(scale)
