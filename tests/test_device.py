import pytest

from song2midi.device import SEGMENT_COST_GB, DeviceBudget, choose_segment, resolve


def test_plenty_of_memory_picks_the_largest_segment():
    assert choose_segment(available_gb=8.0, device="cuda") == max(SEGMENT_COST_GB["cuda"])


def test_tight_memory_picks_a_smaller_segment():
    # Genuinely tight against the measured table, not the old 4x-inflated one.
    tight = 1.3
    chosen = choose_segment(available_gb=tight, device="cuda")
    assert chosen < max(SEGMENT_COST_GB["cuda"])
    assert SEGMENT_COST_GB["cuda"][chosen] <= tight * 0.8


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


def test_the_cost_tables_are_monotone():
    """A longer segment can never cost less than a shorter one."""
    for device, costs in SEGMENT_COST_GB.items():
        seconds = sorted(costs)
        assert [costs[s] for s in seconds] == sorted(costs[s] for s in seconds), device


def test_a_typical_machine_is_not_pushed_to_the_smallest_segment():
    """The old table claimed 6.0 GB for segment 7.8 where 1.6 GB is measured,
    so an ordinary laptop got the smallest segment for no reason."""
    assert choose_segment(available_gb=4.5, device="cpu") == max(SEGMENT_COST_GB["cpu"])
    assert choose_segment(available_gb=3.9, device="cuda") == max(SEGMENT_COST_GB["cuda"])
