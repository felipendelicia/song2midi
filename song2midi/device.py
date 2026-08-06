"""Device selection and memory budgeting for the separation stage.

Demucs is the only stage that gets near a memory limit. The cost table below is
a measured-by-hand approximation, not a guarantee — treat it as a starting point
and tune it if separation OOMs on real hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

from song2midi.errors import CudaUnavailableError

# segment length in seconds -> peak memory in GB
SEGMENT_COST_GB: dict[str, dict[float, float]] = {
    "cuda": {7.8: 3.0, 6.0: 2.4, 5.0: 2.0, 4.0: 1.5},
    # CPU inference keeps activations in float32 without the kernel fusion CUDA
    # gets, so the same segment costs roughly twice as much RAM.
    "cpu": {7.8: 6.0, 6.0: 4.6, 5.0: 3.9, 4.0: 3.0},
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
