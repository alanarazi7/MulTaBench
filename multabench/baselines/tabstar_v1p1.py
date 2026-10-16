from multabench.baselines.tabstar_v1 import TabSTAR
from multabench.constants import SEED


class TabSTARv1p1(TabSTAR):
    """TabSTAR v1.1 at its package defaults; its checkpoints were decontaminated against MulTaBench, so no mapping."""

    MODEL_NAME = "TabSTAR-v1.1"
    SHORT_NAME = "tabstar11"
    IS_PAPER_VERSION = False

    def initialize_model(self):
        from tabstar.tabstarv1p1_model import TabSTARv1p1Classifier, TabSTARv1p1Regressor
        tabstar_cls = TabSTARv1p1Classifier if self.is_cls else TabSTARv1p1Regressor
        return tabstar_cls(device=self.device, verbose=True, random_state=SEED)
