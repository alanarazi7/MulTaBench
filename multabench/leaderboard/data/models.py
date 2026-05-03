from pandas import DataFrame
import streamlit as st

from multabench.leaderboard.data.keys import DATASET, MODEL


def filter_models(df: DataFrame, key: str) -> DataFrame:
    only_text = st.checkbox(
        "Only text models",
        value=True,
        key=f"only_text_{key}",
        help="Show only models with 'text' in the name",
    )
    if only_text:
        df = df[df[DATASET].apply(lambda x: '_text_' in x.lower())]
    model_options = sorted(set(df[MODEL]))
    models = st.multiselect("Models", model_options, default=model_options,
                            key=f"model_filter_{key}")
    df = df[df[MODEL].isin(models)]
    return df