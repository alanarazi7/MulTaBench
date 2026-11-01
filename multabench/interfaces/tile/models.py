"""The models TILE wraps: the PFNs with the feature window TILE selected for them, the others with its default."""
from multabench.baselines.catboost import CatBoost
from multabench.baselines.causilo import Causilo
from multabench.baselines.exaone_tabular import EXAONETabular
from multabench.baselines.kumo_tabular import KumoTabular, KumoTabularMedium, KumoTabularSmall
from multabench.baselines.lgbm import LightGBM
from multabench.baselines.random_forest import RandomForest
from multabench.baselines.realmlp import RealMLP
from multabench.baselines.tabicl_v2 import TabICLv2
from multabench.baselines.tabm import TabM
from multabench.baselines.tabpfn import TabPFN3, TabPFN3p5, TabPFNv2p5
from multabench.baselines.xgboost import XGBoost
from multabench.interfaces.tile.tile_interface import TileInterface


class TileTabPFNv2p5(TileInterface):
    TABULAR_MODEL, TILE_LEARNER = TabPFNv2p5, "tabpfn-2.5"


class TileTabPFN3(TileInterface):
    TABULAR_MODEL, TILE_LEARNER = TabPFN3, "tabpfn-3"


class TileTabPFN3p5(TileInterface):
    TABULAR_MODEL, TILE_LEARNER = TabPFN3p5, "tabpfn-3.5"


class TileTabICLv2(TileInterface):
    TABULAR_MODEL, TILE_LEARNER = TabICLv2, "tabicl"


class TileEXAONETabular(TileInterface):
    TABULAR_MODEL, TILE_LEARNER = EXAONETabular, "exaone"


class TileCausilo(TileInterface):
    TABULAR_MODEL, TILE_LEARNER = Causilo, "causilo"


class TileKumoTabular(TileInterface):
    TABULAR_MODEL, TILE_LEARNER = KumoTabular, "kumo-large"


class TileKumoTabularMedium(TileInterface):
    TABULAR_MODEL, TILE_LEARNER = KumoTabularMedium, "kumo-medium"


class TileKumoTabularSmall(TileInterface):
    TABULAR_MODEL, TILE_LEARNER = KumoTabularSmall, "kumo-small"


class TileRealMLP(TileInterface):
    TABULAR_MODEL, TILE_LEARNER = RealMLP, "realmlp"


class TileXGBoost(TileInterface):
    TABULAR_MODEL, TILE_LEARNER = XGBoost, "xgboost"


class TileCatBoost(TileInterface):
    TABULAR_MODEL, TILE_LEARNER = CatBoost, "catboost"


class TileLightGBM(TileInterface):
    TABULAR_MODEL = LightGBM


class TileRandomForest(TileInterface):
    TABULAR_MODEL = RandomForest


class TileTabM(TileInterface):
    TABULAR_MODEL = TabM


TILE_MODELS = [TileTabPFNv2p5, TileTabPFN3, TileTabPFN3p5, TileTabICLv2, TileEXAONETabular, TileCausilo, TileKumoTabular,
               TileKumoTabularMedium, TileKumoTabularSmall, TileRealMLP, TileXGBoost, TileCatBoost, TileLightGBM,
               TileRandomForest, TileTabM]
