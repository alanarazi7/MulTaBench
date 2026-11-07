"""The models CASE wraps, each with CaseTransformer's defaults."""
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
from multabench.interfaces.case.case_interface import CaseInterface


class CaseTabPFNv2(CaseInterface):
    TABULAR_MODEL = TabPFNv2


class CaseTabPFNv2p5(CaseInterface):
    TABULAR_MODEL = TabPFNv2p5


class CaseTabPFN3(CaseInterface):
    TABULAR_MODEL = TabPFN3


class CaseTabPFN3p5(CaseInterface):
    TABULAR_MODEL = TabPFN3p5


class CaseTabPFN3p5Fast(CaseInterface):
    TABULAR_MODEL = TabPFN3p5Fast


class CaseTabICLv2(CaseInterface):
    TABULAR_MODEL = TabICLv2


class CaseTabDPT(CaseInterface):
    TABULAR_MODEL = TabDPT


class CaseTabDPT13(CaseInterface):
    TABULAR_MODEL = TabDPT13


class CaseEXAONETabular(CaseInterface):
    TABULAR_MODEL = EXAONETabular


class CaseCausilo(CaseInterface):
    TABULAR_MODEL = Causilo


class CaseTabFM(CaseInterface):
    TABULAR_MODEL = TabFM


class CaseKumoTabular(CaseInterface):
    TABULAR_MODEL = KumoTabular


class CaseKumoTabularMedium(CaseInterface):
    TABULAR_MODEL = KumoTabularMedium


class CaseKumoTabularSmall(CaseInterface):
    TABULAR_MODEL = KumoTabularSmall


class CaseMitraV2(CaseInterface):
    TABULAR_MODEL = MitraV2


class CaseLimiX2(CaseInterface):
    TABULAR_MODEL = LimiX2


class CaseXiaomiTabLDM(CaseInterface):
    TABULAR_MODEL = XiaomiTabLDM


class CaseRealMLP(CaseInterface):
    TABULAR_MODEL = RealMLP


class CaseTabM(CaseInterface):
    TABULAR_MODEL = TabM


class CaseXGBoost(CaseInterface):
    TABULAR_MODEL = XGBoost


class CaseCatBoost(CaseInterface):
    TABULAR_MODEL = CatBoost


class CaseLightGBM(CaseInterface):
    TABULAR_MODEL = LightGBM


class CaseRandomForest(CaseInterface):
    TABULAR_MODEL = RandomForest

CASE_MODELS = [CaseTabPFNv2, CaseTabPFNv2p5, CaseTabPFN3, CaseTabPFN3p5, CaseTabPFN3p5Fast, CaseTabICLv2, CaseTabDPT,
               CaseTabDPT13, CaseEXAONETabular, CaseCausilo, CaseTabFM, CaseKumoTabular, CaseKumoTabularMedium,
               CaseKumoTabularSmall, CaseMitraV2, CaseLimiX2, CaseXiaomiTabLDM, CaseRealMLP, CaseTabM, CaseXGBoost,
               CaseCatBoost, CaseLightGBM, CaseRandomForest]
