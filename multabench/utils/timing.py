"""Wall-clock time spent embedding text and image features, reported apart from the model's own time."""
import time
from functools import wraps

_seconds = 0.0
_depth = 0


def embedding_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _seconds, _depth
        if _depth:
            return func(*args, **kwargs)
        _depth += 1
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            _seconds += time.perf_counter() - start
            _depth -= 1
    return wrapper


def pop_embedding_seconds() -> float:
    global _seconds
    seconds, _seconds = _seconds, 0.0
    return seconds
