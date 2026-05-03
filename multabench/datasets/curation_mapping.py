from importlib import import_module
from pkgutil import iter_modules

from multabench.datasets.all_datasets import MultimodalDatasetID
from multabench.datasets import annotated
from multabench.datasets.curation_objects import CuratedDataset

CURATIONS = {}


def construct_curation_dict():
    modules = [m for m in iter_modules(annotated.__path__, "multabench.datasets.annotated.") if not m.ispkg]
    for m in modules:
        try:
            module = import_module(m.name)
        except Exception as e:
            # optionally log
            print(f"[WARN] Skipping {m.name}: {e}")
            raise e        
        module = import_module(m.name)
        sid = m.name.split('.')[-1]
        curated = CuratedDataset.from_module(module)
        CURATIONS[sid] = curated

def get_curated(dataset_id: MultimodalDatasetID) -> CuratedDataset:
    return CURATIONS[dataset_id.name]


construct_curation_dict()
