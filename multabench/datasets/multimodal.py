from enum import StrEnum

from pandas import DataFrame

from multabench.datasets.curation_objects import CuratedDataset
from multabench.datasets.objects import FeatureType
from multabench.preprocessing.feat_types import detect_text_features


class MultimodalState(StrEnum):
    ALL = "all 🔥"
    IMAGE_ONLY = "image 🖼️"
    NON_IMAGE = "non_image 🧑‍🦯"
    TEXT_ONLY = "text_only 📝"
    NO_TEXT = "no_text 😶"


class MultimodalError(Exception):
    pass


def filter_by_multimodality(x: DataFrame, multimodal_state: MultimodalState | None, curation: CuratedDataset):
    if multimodal_state is None:
        return x
    features = [c for c in curation.features if c.new_name in x.columns]
    img_features = [c.new_name for c in features if c.feat_type == FeatureType.IMAGE]
    text_features = detect_text_features(x, exclude_columns=img_features)
    multimodal_features = list({*img_features, *text_features})
    if multimodal_state == MultimodalState.ALL:
        # Always return all features, even if all are images
        return x
    elif multimodal_state == MultimodalState.IMAGE_ONLY:
        img_features = [c.new_name for c in curation.features if c.feat_type == FeatureType.IMAGE]
        if len(img_features) == 0:
            raise MultimodalError("No image features found, irrelevant")
        return x[img_features]
    elif multimodal_state == MultimodalState.NON_IMAGE:
        if len(img_features) == 0:
            raise MultimodalError("No image features found, irrelevant")
        x = x.drop(columns=img_features)
        if len(x.columns) == 0:
            raise MultimodalError("No features left, irrelevant")
        return x
    elif multimodal_state == MultimodalState.TEXT_ONLY:
        if len(text_features) == 0:
            raise MultimodalError("No text features found, irrelevant")
        return x[text_features]
    elif multimodal_state == MultimodalState.NO_TEXT:
        if len(text_features) == 0:
            return x
        x = x.drop(columns=text_features)
        if len(x.columns) == 0:
            raise MultimodalError("No features left, irrelevant")
        return x
    else:
        raise ValueError(f"Invalid value for multimodal_state: {multimodal_state}")
