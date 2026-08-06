import pytest

from song2midi.device import SEGMENT_COST_GB, DeviceBudget, choose_segment, resolve


def test_plenty_of_memory_picks_the_largest_segment():
    assert choose_segment(available_gb=8.0, device="cuda") == max(SEGMENT_COST_GB["cuda"])


def test_tight_memory_picks_a_smaller_segment():
    chosen = choose_segment(available_gb=2.2, device="cuda")
    assert chosen < max(SEGMENT_COST_GB["cuda"])
    assert SEGMENT_COST_GB["cuda"][chosen] <= 2.2 * 0.8


def test_insufficient_memory_falls_back_to_the_smallest_segment():
    assert choose_segment(available_gb=0.2, device="cuda") == min(SEGMENT_COST_GB["cuda"])


def test_cpu_budget_is_more_conservative_than_cuda():
    largest = max(SEGMENT_COST_GB["cpu"])
    assert SEGMENT_COST_GB["cpu"][largest] > SEGMENT_COST_GB["cuda"][largest]


def test_explicit_cpu_request_is_honoured():
    assert resolve("cpu").device == "cpu"


def test_resolve_returns_a_usable_budget():
    budget = resolve("auto")
    assert isinstance(budget, DeviceBudget)
    assert budget.device in ("cpu", "cuda")
    assert budget.segment_seconds > 0


def test_halved_budget_never_goes_below_the_floor():
    budget = DeviceBudget(device="cuda", segment_seconds=2.5, available_gb=4.0)
    assert budget.halved().segment_seconds == 2.0
