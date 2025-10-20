import time
from typing import Any, Iterator

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


class _ProgressIter:
    def __init__(
        self, base_iter: Iterator, skip_target: int, log_every: int = 500, rank: int = 0
    ):
        self.base_iter = base_iter
        self.skip_target = max(int(skip_target), 0)
        self.log_every = max(int(log_every), 1)
        self.rank = rank
        self.count = 0
        self._start_t = None
        self._done = self.skip_target == 0

    def __iter__(self):
        return self

    def __next__(self) -> Any:
        if self._start_t is None:
            self._start_t = time.time()

        item = next(self.base_iter)
        self.count += 1

        if not self._done:
            if (
                self.count % self.log_every == 0
                or self.count == 1
                or self.count >= self.skip_target
            ):
                elapsed = max(time.time() - self._start_t, 1e-6)
                rate = self.count / elapsed
                remain = max(self.skip_target - self.count, 0)
                eta = remain / rate if rate > 0 else float("inf")
                pct = (
                    (self.count / self.skip_target * 100.0)
                    if self.skip_target > 0
                    else 100.0
                )
                print(
                    f"[Resume][rank{self.rank}] Advancing dataloader: {self.count}/{self.skip_target} "
                    f"({pct:.1f}%), {rate:.1f} it/s, ETA {eta/60:.1f} min"
                )
            if self.count >= self.skip_target:
                self._done = True

        return item


class _ProgressLoader:
    def __init__(
        self, base_loader, skip_target: int, rank: int = 0, log_every: int = 500
    ):
        self.base = base_loader
        self.skip_target = skip_target
        self.rank = rank
        self.log_every = log_every

    def __iter__(self):
        return _ProgressIter(
            iter(self.base), self.skip_target, self.log_every, self.rank
        )

    def __len__(self):
        try:
            return len(self.base)
        except Exception:
            return 0

    # delegate attributes
    def __getattr__(self, name):
        return getattr(self.base, name)


@HOOKS.register_module()
class ResumeProgressHook(Hook):
    """Wrap the train dataloader to show progress while fast-forwarding on resume."""

    def __init__(self, log_every: int = 500):
        self.log_every = log_every

    def before_train(self, runner):
        # only rank0 prints by default; but our PerRankLogHook tees stdout per rank, so all ranks will log
        loop = runner.train_loop
        if getattr(loop, "_resume_progress_wrapped", False):
            return
        # skip target equals the resumed iteration count
        skip_target = int(getattr(runner, "iter", 0))
        if skip_target <= 0:
            return
        try:
            base_loader = loop.dataloader
        except Exception:
            return
        # Replace with progress loader
        loop.dataloader = _ProgressLoader(
            base_loader,
            skip_target=skip_target,
            rank=getattr(runner, "rank", 0),
            log_every=self.log_every,
        )
        loop._resume_progress_wrapped = True
