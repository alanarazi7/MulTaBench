from pandas import DataFrame, Series

from multabench.utils.nulls import get_invalid_indices, get_valid_values


# TODO: this logic was copy pasted from old tabular repo, make simpler
def get_dataset_description(name: str, x: DataFrame, y: Series | None = None,
                            url: str | None = None,
                            desc: str | None = None) -> str:
    x.columns = [c.strip() for c in x.columns]
    feat_types = {col: x[col].dtype for col in x.columns}
    if y is not None:
        feat_types[str(y.name)] = y.dtype
    features = _get_x_features_description_block(x=x, feat_types=feat_types)
    sep = "\n====\n"
    if y is not None:
        feat_type = feat_types[str(y.name)]
        target_var = f"Target Variable: {get_series_summary(s=y, feat_type=feat_type)}"
    else:
        target_var = ""

    strings = [f"Dataset Name: {name}",
               f"Examples: {len(x)}",
               f"URL: {url}" if url else None,
               f"Description: {desc}" if desc else None,
               target_var,
               f"Features:\n\n{features}"]
    description = sep.join([s for s in strings if s])
    print(description)
    return description


def _get_x_features_description_block(x: DataFrame, feat_types: dict[str, str]) -> str:
    features = []
    for feat_name, feat_type in feat_types.items():
        if feat_name not in x.columns:
            continue
        feat_rep = get_series_summary(s=x[feat_name], feat_type=feat_type)
        features.append(feat_rep)
    features = "\n".join(features)
    return features

def get_series_summary(s: Series, feat_type: str | None = None) -> str:
    if feat_type is None:
        feat_type = s.dtype
    try:
        n_unique = set(get_valid_values(s))
    except TypeError:
        n_unique = []
    common_values = []
    for v in s.value_counts().index[:10]:
        if isinstance(v, (int, float)):
            v = round(v, 4)
        common_values.append(str(v))
    desc_string = f"{s.name} ({feat_type}, {len(n_unique)} distinct"
    missing_values = len(get_invalid_indices(s))
    if missing_values:
        missing_ratio = 100 * (missing_values / len(s))
        desc_string += f", {missing_ratio:.1f}% missing"
    return f"{desc_string}): {common_values}"