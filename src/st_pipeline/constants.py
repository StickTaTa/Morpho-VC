from pathlib import Path
from typing import NamedTuple


PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data"


class _Keys(NamedTuple):
    X: str = "X"
    Y_BAG: str = "Y_bag"
    PTR_BAG_INSTANCE: str = "ptr_idx_bag_inst"
    SIZE_FACTOR: str = "size_factor"
    GENE_IDS: str = "gene_ids"
    CELL_IDS: str = "cell_ids"
    SPOT_IDS: str = "spot_ids"


KEYS = _Keys()


DEFAULT_GENE_SET = "HVG:1000"
