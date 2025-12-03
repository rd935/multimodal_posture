# scripts/make_utd_mhad_splits.py

import csv
from pathlib import Path
import random

# Path to your full index.csv
INDEX_CSV = Path("data/utd_mhad/index.csv")   # <-- change if needed
OUT_DIR = Path("data/utd_mhad/splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    rows = []
    with INDEX_CSV.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Your columns: subject,action,trial,label,rgb_path,depth_path
            row["subject"] = int(row["subject"])
            row["action"]  = int(row["action"])
            row["trial"]   = int(row["trial"])
            row["label"]   = int(row["label"])
            rows.append(row)

    # ----------------------------------------
    # 1) Discover which subjects exist
    # ----------------------------------------
    subjects = sorted({row["subject"] for row in rows})
    print(f"[INFO] Found subjects: {subjects}")

    if len(subjects) < 3:
        raise ValueError("Need at least 3 subjects to do train/val/test split.")

    random.seed(42)  # or any fixed seed for reproducibility
    random.shuffle(subjects)
    print(f"[INFO] Shuffled subjects: {subjects}")

    # ----------------------------------------
    # 2) Split subjects into train/val/test
    #    For 8 subjects -> 4/2/2 split
    # ----------------------------------------
    n = len(subjects)
    # heuristic: 50% train, 25% val, 25% test (by subject)
    n_train = max(1, int(round(0.5 * n)))
    n_val = max(1, int(round(0.25 * n)))
    # ensure total doesn't exceed n
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1
    n_test = n - n_train - n_val

    train_subjects = set(subjects[:n_train])
    val_subjects   = set(subjects[n_train:n_train + n_val])
    test_subjects  = set(subjects[n_train + n_val:])

    print(f"[INFO] Train subjects: {sorted(train_subjects)}")
    print(f"[INFO] Val subjects:   {sorted(val_subjects)}")
    print(f"[INFO] Test subjects:  {sorted(test_subjects)}")

    splits = {"train": [], "val": [], "test": []}

    for row in rows:
        s = row["subject"]
        if s in train_subjects:
            splits["train"].append(row)
        elif s in val_subjects:
            splits["val"].append(row)
        elif s in test_subjects:
            splits["test"].append(row)
        else:
            print(f"[WARN] Subject {s} not in any split; skipping row with rgb_path={row['rgb_path']}")

    # ----------------------------------------
    # 3) Write split CSVs
    # ----------------------------------------
    fieldnames = ["subject", "action", "trial", "label", "rgb_path", "depth_path"]

    for split_name, split_rows in splits.items():
        out_csv = OUT_DIR / f"utd_mhad_{split_name}.csv"
        if not split_rows:
            print(f"[WARN] No rows for split={split_name}, skipping write.")
            continue

        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)

        print(f"[INFO] Wrote {len(split_rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
