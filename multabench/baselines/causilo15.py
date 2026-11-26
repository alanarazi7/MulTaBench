from multabench.baselines.causilo import Causilo
from multabench.constants import SEED


class Causilo15(Causilo):
    """Causilo 1.5 with the defaults and pinned weights of Causilo 1.5.1."""

    MODEL_NAME = "Causilo 1.5"
    SHORT_NAME = "causilo15"
    ENV = "causilo15"

    def initialize_model(self):
        from causilo import CausiloClassifier, CausiloRegressor

        cls = CausiloClassifier if self.is_cls else CausiloRegressor
        return cls(model_version="1.5", random_state=SEED, device=str(self.device))
