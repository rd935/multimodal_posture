# datasets/utd_mhad_rgbd.py

import csv
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from pathlib import Path
from torchvision import transforms


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
    # print(data.keys()) if unsure
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


class UTDMHADRGBD(Dataset):
    def __init__(
        self,
        index_csv,
        rgb_frames=16,
        resize=(224, 224),
    ):
        """
        index_csv: path to the CSV we created
        rgb_frames: how many frames to sample from the RGB video
        resize: (H, W) to resize frames
        """
        self.items = []
        with open(index_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.items.append(row)

        self.rgb_frames = rgb_frames
        self.resize = resize

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
        rgb_path = Path(row["rgb_path"])
        depth_path = Path(row["depth_path"])
        label = int(row["label"])

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
        # since yours are .mat files, load them differently
        if depth_path.suffix.lower() == ".mat":
            depth_arr = load_depth_mat(depth_path)
            # depth_arr could be (H, W, T) or (T, H, W) depending on file
            # let's try to make it (T, H, W)
            if depth_arr.ndim == 2:
                # single frame depth -> repeat to match rgb length
                depth_arr = np.repeat(depth_arr[None, ...], rgb_tensor.shape[0], axis=0)
            elif depth_arr.ndim == 3:
                # we need to decide if time is first or last
                # let's assume (T, H, W); if not, swap
                if depth_arr.shape[0] < 10 and depth_arr.shape[-1] > 10:
                    # likely (H, W, T)
                    depth_arr = depth_arr.transpose(2, 0, 1)
            else:
                raise ValueError(f"Unexpected depth shape {depth_arr.shape} in {depth_path}")

            # sample same number of frames as RGB
            d_idxs = self._sample_indices(depth_arr.shape[0], rgb_tensor.shape[0])
            depth_tensor_list = []
            for i in d_idxs:
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
            "action": int(row["action"]),
            "trial": int(row["trial"]),
        }
        return sample
