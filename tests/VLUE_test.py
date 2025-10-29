"""
Basic tests for the VLUEDataset in data_utils/VLUE.py
Run with: python tests/VLUE_test.py
"""

import os
import sys

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_utils.VLUE import VLUEDataset


def test_init():
    print("[TEST] init")
    ds = VLUEDataset(split="test", max_samples=5)
    assert ds.name == "VLUE"
    # assert ds.dataset is not None
    print("  - ok")
    return ds


def test_len(ds: VLUEDataset):
    print("[TEST] __len__")
    n = len(ds.dataset)
    assert n >= 0
    print(f"  - len={n}")


def test_getitem(ds: VLUEDataset):
    print("[TEST] __getitem__")
    if len(ds.dataset) == 0:
        print("  - skipped (empty)")
        return
    sample = ds.dataset[0]
    assert isinstance(sample, dict)
    print(f"  - keys={list(sample.keys())}")


def test_dataloader(ds: VLUEDataset):
    print("[TEST] dataloader")
    loader = ds.get_dataloader(batch_size=2, shuffle=False, num_workers=0)
    for batch in loader:
        assert isinstance(batch, dict)
        print(f"  - batch keys={list(batch.keys())}")
        break


def run_all():
    print("== VLUE dataset tests ==")
    ds = test_init()
    # test_len(ds)
    # test_getitem(ds)
    # test_dataloader(ds)
    print("== done ==")


if __name__ == "__main__":
    run_all()


