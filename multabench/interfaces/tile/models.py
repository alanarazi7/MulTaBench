"""The models TILE wraps, each with TILE's defaults (TODO: revisit them per model)."""
from multabench.baselines.catboost import CatBoost
from multabench.baselines.causilo import Causilo
from multabench.baselines.exaone_tabular import EXAONETabular
from multabench.baselines.kumo_tabular import KumoTabular, KumoTabularMedium, KumoTabularSmall
from multabench.baselines.lgbm import LightGBM
from multabench.baselines.limix2 import LimiX2
from multabench.baselines.mitra2 import MitraV2
from multabench.baselines.random_forest import RandomForest
from multabench.baselines.realmlp import RealMLP
from multabench.baselines.tabdpt import TabDPT
from multabench.baselines.tabdpt13 import TabDPT13
from multabench.baselines.tabfm import TabFM
from multabench.baselines.tabicl_v2 import TabICLv2
from multabench.baselines.tabldm import XiaomiTabLDM
from multabench.baselines.tabm import TabM
from multabench.baselines.tabpfn import TabPFN3, TabPFN3p5, TabPFN3p5Fast, TabPFNv2, TabPFNv2p5
from multabench.baselines.xgboost import XGBoost
from multabench.interfaces.tile.tile_interface import TileInterface


class TileTabPFNv2(TileInterface):
    TABULAR_MODEL = TabPFNv2


class TileTabPFNv2p5(TileInterface):
    TABULAR_MODEL = TabPFNv2p5


class TileTabPFN3(TileInterface):
    TABULAR_MODEL = TabPFN3


class TileTabPFN3p5(TileInterface):
    TABULAR_MODEL = TabPFN3p5


class TileTabPFN3p5Fast(TileInterface):
    TABULAR_MODEL = TabPFN3p5Fast


class TileTabICLv2(TileInterface):
    TABULAR_MODEL = TabICLv2


class TileTabDPT(TileInterface):
    TABULAR_MODEL = TabDPT


class TileTabDPT13(TileInterface):
    TABULAR_MODEL = TabDPT13


class TileEXAONETabular(TileInterface):
    TABULAR_MODEL = EXAONETabular


class TileCausilo(TileInterface):
    TABULAR_MODEL = Causilo


class TileTabFM(TileInterface):
    TABULAR_MODEL = TabFM


class TileKumoTabular(TileInterface):
    TABULAR_MODEL = KumoTabular


class TileKumoTabularMedium(TileInterface):
    TABULAR_MODEL = KumoTabularMedium


class TileKumoTabularSmall(TileInterface):
    TABULAR_MODEL = KumoTabularSmall


class TileMitraV2(TileInterface):
    TABULAR_MODEL = MitraV2


class TileLimiX2(TileInterface):
    TABULAR_MODEL = LimiX2


class TileXiaomiTabLDM(TileInterface):
    TABULAR_MODEL = XiaomiTabLDM


class TileRealMLP(TileInterface):
    TABULAR_MODEL = RealMLP


class TileTabM(TileInterface):
    TABULAR_MODEL = TabM


class TileXGBoost(TileInterface):
    TABULAR_MODEL = XGBoost


class TileCatBoost(TileInterface):
    TABULAR_MODEL = CatBoost


class TileLightGBM(TileInterface):
    TABULAR_MODEL = LightGBM


class TileRandomForest(TileInterface):
    TABULAR_MODEL = RandomForest

TILE_MODELS = [TileTabPFNv2, TileTabPFNv2p5, TileTabPFN3, TileTabPFN3p5, TileTabPFN3p5Fast, TileTabICLv2, TileTabDPT,
               TileTabDPT13, TileEXAONETabular, TileCausilo, TileTabFM, TileKumoTabular, TileKumoTabularMedium,
               TileKumoTabularSmall, TileMitraV2, TileLimiX2, TileXiaomiTabLDM, TileRealMLP, TileTabM, TileXGBoost,
               TileCatBoost, TileLightGBM, TileRandomForest]
