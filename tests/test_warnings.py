import builtins
import importlib

from multabench.utils.warnings import silence_baselines_prints


def test_the_filtered_print_is_found_where_it_says_it_lives():
    with silence_baselines_prints():
        filtered = builtins.print
        assert getattr(importlib.import_module(filtered.__module__), filtered.__name__) is filtered
