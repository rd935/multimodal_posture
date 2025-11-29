import sys
from pathlib import Path
from collections import Counter

# Add project root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]   # one level up from scripts/
sys.path.append(str(ROOT))

from datasets.utd_mhad_rgbd import UTDMHADRGBD

train_ds = UTDMHADRGBD("data/utd_mhad/splits/utd_mhad_train.csv", label_mode="stability3")
val_ds   = UTDMHADRGBD("data/utd_mhad/splits/utd_mhad_val.csv",   label_mode="stability3")
test_ds  = UTDMHADRGBD("data/utd_mhad/splits/utd_mhad_test.csv",  label_mode="stability3")

def count_labels(ds):
    return Counter(int(ds[i]["label"]) for i in range(len(ds)))

print("train:", count_labels(train_ds))  # should have 0,1,2 present
print("val:",   count_labels(val_ds))
print("test:",  count_labels(test_ds))
