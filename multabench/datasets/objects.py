from enum import Enum


class SupervisedTask(Enum):
    REGRESSION = "📈 regression"
    BINARY = "⚖️ binary"
    MULTICLASS = "🎨 multiclass"


class FeatureType(Enum):
    CATEGORICAL = "🏷️ categorical"
    NUMERIC = "🔢 numeric"
    TEXT = "📝 text"
    DATE = "📅 date"
    BOOLEAN = "☑️ boolean"
    IMAGE = "🖼️ image"
