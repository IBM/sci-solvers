"""Background pollers for peak host RAM (process RSS) and peak GPU memory.

Both degrade to no-ops when their library (``psutil`` / ``pynvml``) or a GPU is missing;
``peak_mb`` then stays ``None`` and the result simply omits that measurement.
"""

__copyright__ = """
(C) Copyright IBM Corp. 2026

SPDX-License-Identifier: Apache-2.0
"""

import threading
from typing import Any


class _Poller:
    """Sample a reading every ``interval`` seconds on a daemon thread; keep the maximum."""

    def __init__(self, interval: float) -> None:
        """Prepare the poller; subclasses set ``_available``."""
        self.interval = interval
        self.peak_mb: float | None = None
        self._available = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _read_mb(self) -> float:
        raise NotImplementedError

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                value = self._read_mb()
                if self.peak_mb is None or value > self.peak_mb:
                    self.peak_mb = value
            except Exception:  # noqa: BLE001 - a failed sample must not kill the run
                pass
            self._stop.wait(self.interval)

    def reset_peak(self) -> None:
        """Forget the peak so far; the next sample starts a new window."""
        if self._available:
            self.peak_mb = None

    def __enter__(self) -> "_Poller":
        """Start sampling."""
        if self._available:
            self._stop.clear()
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        """Stop sampling."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class RamPoller(_Poller):
    """Peak resident set size of this process, in MB."""

    def __init__(self, interval: float = 0.5) -> None:
        """Bind to the current process via ``psutil``."""
        super().__init__(interval)
        try:
            import psutil

            self._process = psutil.Process()
            self._available = True
        except ImportError:
            pass

    def _read_mb(self) -> float:
        return self._process.memory_info().rss / 1e6


class GpuRamPoller(_Poller):
    """Peak device-wide GPU memory in MB, as a delta above the usage at ``__enter__``.

    Device-wide (NVML) rather than per-allocator, so solvers that allocate outside
    PyTorch (e.g. sbd's CUDA backend) are measured the same way as PyTorch solvers.
    """

    def __init__(self, interval: float = 0.1, gpu_index: int = 0) -> None:
        """Initialize NVML for ``gpu_index``."""
        super().__init__(interval)
        self._baseline_mb = 0.0
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self._nvml = pynvml
            self._available = True
        except Exception:  # noqa: BLE001 - no NVML, no GPU, or init failure
            pass

    def _used_mb(self) -> float:
        return self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 1e6

    def _read_mb(self) -> float:
        return self._used_mb() - self._baseline_mb

    def __enter__(self) -> "GpuRamPoller":
        """Record the baseline, then start sampling."""
        if self._available:
            try:
                self._baseline_mb = self._used_mb()
            except Exception:  # noqa: BLE001
                self._baseline_mb = 0.0
        super().__enter__()
        return self
