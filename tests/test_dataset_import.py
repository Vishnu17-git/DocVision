import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_dataset_module_imports_without_torch():
    module = importlib.import_module("src.dataset.cord_dataset")
    assert hasattr(module, "CORDDataset")
