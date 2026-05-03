import pandas as pd


from multabench.leaderboard.data.keys import MM, DINO_TUNE


def _filter_non_image_tuned(row: pd.Series) -> bool:
    if row[MM] != "non":
        return True
    if row[DINO_TUNE] not in {"yes", "true", True}:
        return True
    return False
