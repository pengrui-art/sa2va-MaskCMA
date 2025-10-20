import io
import logging
import os
import sys
import traceback
from datetime import datetime

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


class _Tee(io.TextIOBase):
    def __init__(self, stream, file_obj):
        self._stream = stream
        self._file = file_obj

    def write(self, data):
        try:
            self._stream.write(data)
            self._stream.flush()
        except Exception:
            pass
        try:
            self._file.write(data)
            self._file.flush()
        except Exception:
            pass
        return len(data)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass


@HOOKS.register_module()
class PerRankLogHook(Hook):
    """
    Hook to capture logs and uncaught exceptions per rank into dedicated files.

    It attaches a FileHandler to mmengine/root loggers, captures sys.stdout/stderr
    (optional), and installs a sys.excepthook that writes a crash report.
    """

    def __init__(
        self,
        log_subdir: str = "rank_logs",
        capture_stdout: bool = True,
        capture_stderr: bool = True,
        log_env_info: bool = True,
        log_cfg_snapshot: bool = False,
        log_level: int = logging.DEBUG,
    ) -> None:
        self.log_subdir = log_subdir
        self.capture_stdout = capture_stdout
        self.capture_stderr = capture_stderr
        self.log_env_info = log_env_info
        self.log_cfg_snapshot = log_cfg_snapshot
        self.log_level = log_level

        self._fh = None
        self._file = None
        self._orig_stdout = None
        self._orig_stderr = None
        self._orig_excepthook = None

    def _get_rank(self, runner):
        try:
            return runner.rank
        except Exception:
            return int(os.environ.get("RANK", 0))

    def _get_world_size(self, runner):
        try:
            return runner.world_size
        except Exception:
            return int(os.environ.get("WORLD_SIZE", 1))

    def before_run(self, runner):
        rank = self._get_rank(runner)
        world_size = self._get_world_size(runner)
        work_dir = getattr(runner, "work_dir", os.getcwd())
        log_dir = os.path.join(work_dir, self.log_subdir)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"rank{rank}.log")

        # Open append mode to keep prior runs
        self._file = open(log_path, "a", buffering=1)

        # Attach a file handler to mmengine logger and root logger
        fmt = logging.Formatter(
            fmt=f"%(asctime)s [rank={rank}/%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._fh = logging.FileHandler(log_path)
        self._fh.setLevel(self.log_level)
        self._fh.setFormatter(fmt)

        for name in ("mmengine", "current", None):  # None -> root
            logger = logging.getLogger(name)
            logger.setLevel(self.log_level)
            logger.addHandler(self._fh)

        # Capture stdout/stderr if requested
        if self.capture_stdout:
            self._orig_stdout = sys.stdout
            sys.stdout = _Tee(sys.stdout, self._file)
        if self.capture_stderr:
            self._orig_stderr = sys.stderr
            sys.stderr = _Tee(sys.stderr, self._file)

        # Install exception hook to dump crashes
        self._orig_excepthook = sys.excepthook

        def _hook(exc_type, exc_value, tb):
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._file.write(
                f"\n======== Uncaught Exception (rank {rank}) @ {ts} ========\n"
            )
            traceback.print_exception(exc_type, exc_value, tb, file=self._file)
            self._file.flush()
            if self._orig_excepthook is not None:
                try:
                    self._orig_excepthook(exc_type, exc_value, tb)
                except Exception:
                    pass

        sys.excepthook = _hook

        # Log environment/system info
        if self.log_env_info:
            self._log_env(rank, world_size, runner)

        # Optional: dump cfg snapshot
        if self.log_cfg_snapshot:
            cfg = getattr(runner, "cfg", None)
            if cfg is not None:
                try:
                    self._file.write("\n===== Config Snapshot =====\n")
                    self._file.write(str(cfg))
                    self._file.write("\n===========================\n")
                except Exception:
                    pass

    def _log_env(self, rank: int, world_size: int, runner):
        def w(line: str):
            try:
                self._file.write(line + "\n")
                self._file.flush()
            except Exception:
                pass

        w("===== Runtime/Environment Info =====")
        w(f"rank={rank}, world_size={world_size}")
        try:
            import mmengine, deepspeed, transformers
            import xtuner
        except Exception:
            mmengine = deepspeed = transformers = xtuner = None

        def _v(mod):
            return getattr(mod, "__version__", "unknown") if mod else "n/a"

        w(f"python: {sys.version.split()[0]}")
        w(f"torch: {torch.__version__}")
        w(f"cuda available: {torch.cuda.is_available()}")
        try:
            w(f"cuda runtime: {torch.version.cuda}")
            w(f"cudnn: {torch.backends.cudnn.version()}")
        except Exception:
            pass
        w(
            f"mmengine: {_v(mmengine)}  deepspeed: {_v(deepspeed)}  transformers: {_v(transformers)}  xtuner: {_v(xtuner)}"
        )

        # GPUs
        try:
            n = torch.cuda.device_count()
            w(f"num gpus: {n}")
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                w(f"gpu[{i}]: {props.name}, {props.total_memory/1024**3:.1f} GB")
        except Exception:
            pass

        # Key env vars
        for k in [
            "CUDA_VISIBLE_DEVICES",
            "NCCL_DEBUG",
            "NCCL_IB_DISABLE",
            "NCCL_P2P_DISABLE",
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
        ]:
            if k in os.environ:
                w(f"env {k}={os.environ[k]}")

        w("====================================")

    def after_run(self, runner):
        # Restore std streams and excepthook
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr
        if self._orig_excepthook is not None:
            sys.excepthook = self._orig_excepthook

        # Detach handler
        if self._fh is not None:
            for name in ("mmengine", "current", None):
                logger = logging.getLogger(name)
                try:
                    logger.removeHandler(self._fh)
                except Exception:
                    pass
            self._fh.close()
            self._fh = None
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
