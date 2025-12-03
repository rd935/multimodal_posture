# datasets/utd_mhad_rgbd.py

import csv
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from pathlib import Path
from torchvision import transforms

# ------------------------------------------------------------------
#  STABILITY MAPPING: ACTION (1..27) -> {0: stable, 1: unstable, 2: falling}
# ------------------------------------------------------------------
# !!! IMPORTANT !!!
# Replace these lists with the actual mapping your project/TA expects.
# Action IDs here are the original UTD-MHAD action numbers (1..27),
# NOT the zero-based "label" field.
#
# Example ONLY — you MUST EDIT these:
STABLE_ACTIONS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 16, 18, 19]

UNSTABLE_ACTIONS = [7, 12, 13, 14, 15, 17, 20, 21, 22]

FALLING_ACTIONS = [23, 24, 25, 26, 27]


STABILITY_MAP = {}
for a in STABLE_ACTIONS:
    STABILITY_MAP[a] = 0  # stable
for a in UNSTABLE_ACTIONS:
    STABILITY_MAP[a] = 1  # unstable
for a in FALLING_ACTIONS:
    STABILITY_MAP[a] = 2  # falling


def action_to_stability(action: int) -> int:
    """
    Map UTD-MHAD action ID (1..27) to stability class:
        0: stable
        1: unstable
        2: falling
    """
    if action not in STABILITY_MAP:
        raise ValueError(
            f"Action {action} not found in STABILITY_MAP. "
            "Add it to STABLE_ACTIONS / UNSTABLE_ACTIONS / FALLING_ACTIONS."
        )
    return STABILITY_MAP[action]


# ------------------------------------------------------------------
#  Video / depth loading helpers
# ------------------------------------------------------------------

def load_video_opencv(path, max_frames=None):
    """
    Load video using OpenCV and return a list of frames (H, W, 3) in RGB.
    If max_frames is set, we stop after that many frames.
    """
    cap = cv2.VideoCapture(str(path))
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        count += 1
        if max_frames is not None and count >= max_frames:
            break
    cap.release()
    return frames  # list of np arrays


def load_depth_mat(path):
    """
    Load depth from .mat file.
    UTD-MHAD depth .mat files often store the variable under a simple key.
    We'll try a few common keys.
    Returns a numpy array (T, H, W) or (H, W) depending on file.
    """
    data = loadmat(path)
    # try to guess the depth key
    for key in ["depth", "Depth", "d", "frame"]:
        if key in data:
            arr = data[key]
            break
    else:
        # fallback: take the first non-metadata entry
        arr = None
        for k, v in data.items():
            if not k.startswith("__"):
                arr = v
                break
    if arr is None:
        raise ValueError(f"Could not find depth array in {path}")
    # convert to float32
    arr = np.array(arr, dtype=np.float32)
    return arr


def resample_sequence(arr, target_len):
    """
    Resample a temporal sequence arr (T, H, W) to length target_len.
    Uses nearest-neighbor in time via np.linspace + indexing.
    """
    T = arr.shape[0]
    if T == target_len:
        return arr
    # indices in original time
    idxs = np.linspace(0, T - 1, target_len).astype(int)
    return arr[idxs]


# ------------------------------------------------------------------
#  Dataset
# ------------------------------------------------------------------

class UTDMHADRGBD(Dataset):
    def __init__(
        self,
        index_csv,
        rgb_frames=16,
        resize=(224, 224),
        label_mode="stability3",
    ):
        """
        index_csv: path to the CSV we created
        rgb_frames: how many frames to sample from the RGB video
        resize: (H, W) to resize frames
        label_mode:
            "stability3" -> 3-class stable/unstable/falling labels (0,1,2)
            "action27"   -> original 27-class action labels (0..26)
        """
        self.items = []
        with open(index_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.items.append(row)

        self.rgb_frames = rgb_frames
        self.resize = resize
        self.label_mode = label_mode

        # transforms for RGB frames
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),  # (H, W, C) -> (C, H, W), [0,255] -> [0,1]
            transforms.Resize(resize),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # depth: just to tensor + resize
        self.depth_transform = transforms.Compose([
            transforms.ToTensor(),   # (H, W) -> (1, H, W)
            transforms.Resize(resize),
        ])

    def __len__(self):
        return len(self.items)

    def _sample_indices(self, total, num_samples):
        """Uniformly sample num_samples indices from [0, total-1]."""
        if total <= num_samples:
            return list(range(total))
        # linspace then round
        return np.linspace(0, total - 1, num_samples).astype(int).tolist()

    def __getitem__(self, idx):
        row = self.items[idx]
        rgb_path = Path(row["rgb_path"].replace("\\", "/"))
        depth_path = Path(row["depth_path"].replace("\\", "/"))

        # Original 27-class label (0..26) and 1-based action ID from CSV
        raw_label_27 = int(row["label"])
        action_id = int(row["action"])  # should be 1..27

        # Decide which label to output
        if self.label_mode == "stability3":
            label = action_to_stability(action_id)  # -> 0/1/2
        elif self.label_mode == "action27":
            label = raw_label_27
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

        # ---------- RGB ----------
        rgb_frames_np = load_video_opencv(rgb_path)
        if len(rgb_frames_np) == 0:
            raise RuntimeError(f"Could not read RGB video: {rgb_path}")

        # sample frames
        frame_idxs = self._sample_indices(len(rgb_frames_np), self.rgb_frames)
        rgb_tensor_list = []
        for i in frame_idxs:
            frame = rgb_frames_np[i]  # (H, W, 3) RGB uint8
            # to PIL-like tensor pipeline: transforms.ToTensor expects PIL or ndarray
            frame_t = self.rgb_transform(frame)
            rgb_tensor_list.append(frame_t)  # (3, H, W)

        # stack to (T, 3, H, W)
        rgb_tensor = torch.stack(rgb_tensor_list, dim=0)

        # ---------- DEPTH ----------
        if depth_path.suffix.lower() == ".mat":
            depth_arr = load_depth_mat(depth_path)
            if depth_arr.ndim == 2:
                # (H, W) -> pretend T=1 then resample to len(rgb_tensor)
                depth_arr = np.repeat(depth_arr[None, ...], 1, axis=0)
            elif depth_arr.ndim == 3:
                # guess layout and convert to (T, H, W)
                if depth_arr.shape[0] < 10 and depth_arr.shape[-1] > 10:
                    # likely (H, W, T)
                    depth_arr = depth_arr.transpose(2, 0, 1)
                # else: assume (T, H, W)
            else:
                raise ValueError(f"Unexpected depth shape {depth_arr.shape} in {depth_path}")

            # simple temporal resampling to match rgb length
            target_T = rgb_tensor.shape[0]
            depth_arr = resample_sequence(depth_arr, target_T)  # (T, H, W)

            depth_tensor_list = []
            for i in range(target_T):
                dframe = depth_arr[i]  # (H, W)
                dframe_t = self.depth_transform(dframe)
                depth_tensor_list.append(dframe_t)  # (1, H, W)
            depth_tensor = torch.stack(depth_tensor_list, dim=0)  # (T, 1, H, W)

        else:
            # if in future you have depth .avi
            depth_tensor = None  # placeholder

        sample = {
            "rgb": rgb_tensor,           # (T, 3, H, W)
            "depth": depth_tensor,       # (T, 1, H, W)
            "label": torch.tensor(label, dtype=torch.long),
            "subject": int(row["subject"]),
            "action": action_id,
            "trial": int(row["trial"]),
        }
        return sample
