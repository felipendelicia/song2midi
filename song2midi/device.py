"""Device selection and memory budgeting for the separation stage.

Demucs is the only stage that gets near a memory limit. The cost table below is
a measured-by-hand approximation, not a guarantee — treat it as a starting point
and tune it if separation OOMs on real hardware.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from song2midi.errors import CudaUnavailableError

# Torch words allocation failures differently per device, and matching only
# "out of memory" made the whole retry path dead code on CPU - the one device
# that actually needs it. CUDA says "CUDA out of memory"; the CPU allocator
# says "[enforce fail at alloc_cpu.cpp] DefaultCPUAllocator: can't allocate
# memory"; other backends have their own phrasings.
OOM_MARKERS = (
    "out of memory",
    "can't allocate memory",
    "cannot allocate memory",
    "defaultcpuallocator",
    "not enough memory",
    "bad_alloc",
)


def is_out_of_memory(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in OOM_MARKERS)


def warn(message: str) -> None:
    print(f"song2midi: {message}", file=sys.stderr)


def release_cuda(torch) -> None:
    """Give a smaller retry a chance of fitting where the previous one did not."""
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

# segment length in seconds -> peak memory in GB.
#
# CPU figures are measured: peak RSS separating a 4-minute mix on this
# project's locked torch, with model.segment actually set (before that fix the
# knob did nothing and every segment cost the same). The earlier guesses here
# were 4x too high, so a machine with 4.5 GB free was pushed to the smallest
# segment when the largest fits comfortably.
#
# Note the curve is shallow, not linear: a fixed cost dominates - the weights
# plus the full-length output tensor, which grows with the song rather than
# with the segment. Halving the segment buys much less than it looks like it
# should.
#
# CUDA figures are the CPU ones minus the output tensor, which lives in system
# RAM either way, and are the least trustworthy numbers in this file: they have
# not been measured on real hardware. They are deliberately conservative.
SEGMENT_COST_GB: dict[str, dict[float, float]] = {
    "cuda": {7.8: 1.2, 6.0: 1.0, 5.0: 0.9, 4.0: 0.8},
    "cpu": {7.8: 1.6, 6.0: 1.4, 5.0: 1.35, 4.0: 1.3},
}

SAFETY_MARGIN = 0.8
MIN_SEGMENT_SECONDS = 2.0
FALLBACK_AVAILABLE_GB = 2.0


@dataclass(frozen=True)
class DeviceBudget:
    device: str
    segment_seconds: float
    available_gb: float

    def halved(self) -> DeviceBudget:
        """Retry budget after an out-of-memory failure."""
        return DeviceBudget(
            device=self.device,
            segment_seconds=max(MIN_SEGMENT_SECONDS, self.segment_seconds / 2),
            available_gb=self.available_gb,
        )


def choose_segment(available_gb: float, device: str) -> float:
    costs = SEGMENT_COST_GB[device]
    usable = available_gb * SAFETY_MARGIN
    affordable = [seconds for seconds, cost in costs.items() if cost <= usable]
    return max(affordable) if affordable else min(costs)


def resolve(requested: str = "auto") -> DeviceBudget:
    device = _select_device(requested)
    available = _available_gb(device)
    return DeviceBudget(
        device=device,
        segment_seconds=choose_segment(available, device),
        available_gb=available,
    )


def _select_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise CudaUnavailableError(
                "CUDA was requested but torch is not installed"
            ) from None
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        raise CudaUnavailableError(_why_no_cuda(torch))
    return "cpu"


def _why_no_cuda(torch) -> str:
    """Name the actual cause.

    The project installs the CPU-only torch wheels, so a machine with a
    perfectly good NVIDIA card still reports no CUDA. Saying only "CUDA is not
    available" sends the user off to reinstall drivers that are fine.
    """
    if getattr(torch.version, "cuda", None) is None:
        return (
            "CUDA was requested, but this environment has the CPU-only build of "
            f"torch ({torch.__version__}), which cannot use a GPU no matter what "
            "hardware is present. Install a CUDA build of torch to use --device cuda."
        )
    return (
        "CUDA was requested but no usable GPU was found. torch was built for CUDA "
        f"{torch.version.cuda}; check that the driver is installed and the GPU is visible."
    )


def _available_gb(device: str) -> float:
    if device == "cuda":
        import torch

        free_bytes, _ = torch.cuda.mem_get_info()
        return free_bytes / 1024**3
    return _available_ram_gb()


def _available_ram_gb() -> float:
    """Available RAM in GB, or `FALLBACK_AVAILABLE_GB` when it cannot be read.

    This used to parse /proc/meminfo, which does not exist on Windows or macOS:
    both silently took the 2 GB fallback and got the smallest Demucs segment no
    matter how much memory the machine had. psutil is the cross-platform answer,
    so there is one code path rather than one per OS.
    """
    try:
        import psutil

        available = psutil.virtual_memory().available / 1024**3
    except Exception:  # psutil missing, or a platform it cannot read
        return FALLBACK_AVAILABLE_GB
    return available if available > 0.0 else FALLBACK_AVAILABLE_GB
