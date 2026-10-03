import time

from multabench.utils.timing import embedding_step, pop_embedding_seconds


@embedding_step
def _encode():
    time.sleep(0.05)


@embedding_step
def _fit_and_encode():
    _encode()
    time.sleep(0.05)


def test_nested_steps_are_counted_once():
    pop_embedding_seconds()
    _fit_and_encode()
    assert 0.1 <= pop_embedding_seconds() < 0.15
    assert pop_embedding_seconds() == 0.0
