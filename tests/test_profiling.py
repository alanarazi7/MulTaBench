import time

from multabench.utils.profiling import PeakMemoryTracker


def test_wall_time_is_not_rounded_up_to_the_sampling_interval():
    with PeakMemoryTracker(phase="train", interval=0.5) as tracker:
        time.sleep(0.05)
    assert 0.05 <= tracker.wall_time_s < 0.3
