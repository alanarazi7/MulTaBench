import builtins
import contextlib
import io
import warnings
from typing import Tuple


# Default noisy markers for baseline models
BASELINE_NOISY_MARKERS: Tuple[str, ...] = (
    # TabM using 1.6.1 pytabkit version
    "y_pred.flatten(0, 1).shape=",
    "Setting seed:",
    "Reducing num_emb_n_bins",
    # LightGBM
    "No further splits with positive gain, best gain",
    "Stopped training because there are no more leaves that meet the split",
    "Met negative value in categorical features",
    "Met categorical feature which contains sparse values",
    # TabSTAR-v1
    "UserWarning: Please use the new API settings to control TF32 behavior",
)

# Kaggle-specific noisy markers
KAGGLE_NOISY_MARKERS: Tuple[str, ...] = (
    "outdated `kagglehub` version",
    "please consider upgrading",
)


@contextlib.contextmanager
def silence_noisy_prints(noisy_markers: Tuple[str, ...] = BASELINE_NOISY_MARKERS):
    """
    Filter out noisy print statements containing any of the specified markers.
    
    Args:
        noisy_markers: Tuple of strings to filter out from print statements.
                      Defaults to BASELINE_NOISY_MARKERS.
    """
    orig_print = builtins.print
    
    def _filtered_print(*args, **kwargs):
        s = " ".join(str(a) for a in args)
        if any(m in s for m in noisy_markers):
            return
        return orig_print(*args, **kwargs)

    builtins.print = _filtered_print
    try:
        yield
    finally:
        builtins.print = orig_print


@contextlib.contextmanager
def silence_baselines_prints():
    """Filter noisy prints and stderr from baseline models (TabM, LightGBM, etc.)."""
    import sys
    with silence_noisy_prints(BASELINE_NOISY_MARKERS):
        old_stderr = sys.stderr
        sys.stderr = _FilteredStream(old_stderr, BASELINE_NOISY_MARKERS)
        try:
            yield
        finally:
            sys.stderr = old_stderr


@contextlib.contextmanager
def silence_kaggle_prints():
    """Filter noisy prints from kagglehub (version warnings, stderr, etc.)."""
    with silence_noisy_prints(KAGGLE_NOISY_MARKERS):
        import sys
        old_stderr = sys.stderr
        sys.stderr = _FilteredStream(old_stderr, KAGGLE_NOISY_MARKERS)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*kagglehub.*')
                yield
        finally:
            sys.stderr = old_stderr


class _FilteredStream(io.TextIOBase):
    """Stream wrapper that drops lines matching any of the given markers."""
    def __init__(self, stream, markers):
        self._stream = stream
        self._markers = markers

    def write(self, s):
        if any(m in s for m in self._markers):
            return len(s)
        return self._stream.write(s)

    def flush(self):
        self._stream.flush()


class _ChannelDimFilter:
    """Logging filter that drops 'channel dimension is ambiguous' messages."""
    def filter(self, record):
        return "channel dimension is ambiguous" not in record.getMessage()


@contextlib.contextmanager
def suppress_channel_dimension_warning():
    """Suppress HuggingFace warning about ambiguous channel dimension for small images."""
    import logging
    _filter = _ChannelDimFilter()
    # The warning comes from transformers loggers (logging.warning, not warnings.warn)
    logger_names = ["transformers.image_processing_utils", "transformers.image_utils"]
    loggers = [logging.getLogger(name) for name in logger_names]
    for lgr in loggers:
        lgr.addFilter(_filter)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*channel dimension is ambiguous.*')
            yield
    finally:
        for lgr in loggers:
            lgr.removeFilter(_filter)