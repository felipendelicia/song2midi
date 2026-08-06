"""Device selection and memory budgeting for the separation stage.

Demucs is the only stage that gets near a memory limit. The cost table below is
a measured-by-hand approximation, not a guarantee — treat it as a starting point
and tune it if separation OOMs on real hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

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
            raise RuntimeError("CUDA was requested but torch is not installed") from None
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        raise RuntimeError("CUDA was requested but is not available")
    return "cpu"


def _available_gb(device: str) -> float:
    if device == "cuda":
        import torch

        free_bytes, _ = torch.cuda.mem_get_info()
        return free_bytes / 1024**3
    return _available_ram_gb()


def _available_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return FALLBACK_AVAILABLE_GB  # conservative default when we cannot tell
