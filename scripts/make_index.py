import os
import csv
import re
from pathlib import Path

# ---------- CONFIGURE THESE PATHS ----------
BASE_DIR = Path("data/utd_mhad")

DEPTH_DIR = BASE_DIR / "Depth"
RGB_DIRS = [
    BASE_DIR / "RGB-part1",
    BASE_DIR / "RGB-part2",
    BASE_DIR / "RGB-part3",
    BASE_DIR / "RGB-part4",
]

OUT_CSV = BASE_DIR / "index.csv"

# ---------- FILENAME PATTERNS ----------
# Depth files in your case: a9_s8_t4_depth.mat  (no leading zeros, .mat)
# Some mirrors: a01_s01_t01_depth.avi
DEPTH_PATTERN = re.compile(
    r"^a(?P<act>\d+)_s(?P<subj>\d+)_t(?P<trial>\d+)_depth\.(mat|avi)$",
    re.IGNORECASE,
)

# RGB is still .avi
RGB_PATTERN = "a{act}_s{subj}_t{trial}_color.avi"


def find_rgb_file(act: int, subj: int, trial: int) -> str | None:
    """
    Search across all RGB folders for a{act}_s{subj}_t{trial}_color.avi
    """
    target_name = RGB_PATTERN.format(act=act, subj=subj, trial=trial)
    for rgb_dir in RGB_DIRS:
        candidate = rgb_dir / target_name
        if candidate.exists():
            return str(candidate)
    return None


def main():
    if not DEPTH_DIR.exists():
        raise FileNotFoundError(f"Depth directory not found: {DEPTH_DIR}")

    rows = []
    depth_files = sorted(os.listdir(DEPTH_DIR))

    for fname in depth_files:
        # only consider files that look like depth.* (mat or avi)
        m = DEPTH_PATTERN.match(fname)
        if not m:
            # we can log and skip anything else
            print(f"[warn] skipping non-depth file: {fname}")
            continue

        act = int(m.group("act"))
        subj = int(m.group("subj"))
        trial = int(m.group("trial"))

        depth_path = str(DEPTH_DIR / fname)

        # try to find the matching RGB .avi
        rgb_path = find_rgb_file(act, subj, trial)
        if rgb_path is None:
            print(f"[warn] no RGB found for a{act}_s{subj}_t{trial}")
            continue

        # UTD-MHAD actions are 1..27, make them 0..26
        label = act - 1

        rows.append(
            {
                "subject": subj,
                "action": act,
                "trial": trial,
                "label": label,
                "rgb_path": rgb_path,
                "depth_path": depth_path,
            }
        )

    # write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject",
                "action",
                "trial",
                "label",
                "rgb_path",
                "depth_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] wrote {len(rows)} paired samples to {OUT_CSV}")


if __name__ == "__main__":
    main()
