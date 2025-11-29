# datasets/utd_mhad_dataloaders.py

from torch.utils.data import DataLoader
from .utd_mhad_rgbd import UTDMHADRGBD


def make_utd_mhad_loaders(
    train_csv,
    val_csv,
    test_csv,
    batch_size=8,
    num_workers=4,
    rgb_frames=16,
    resize=(224, 224),
    label_mode="stability3",
):
    """
    Create train/val/test DataLoaders for UTD-MHAD RGB-D dataset.

    label_mode:
        "stability3" -> 3-class stable/unstable/falling labels (0,1,2)
        "action27"   -> 27-class action labels (0..26)
    """

    train_ds = UTDMHADRGBD(
        train_csv,
        rgb_frames=rgb_frames,
        resize=resize,
        label_mode=label_mode,
    )
    val_ds = UTDMHADRGBD(
        val_csv,
        rgb_frames=rgb_frames,
        resize=resize,
        label_mode=label_mode,
    )
    test_ds = UTDMHADRGBD(
        test_csv,
        rgb_frames=rgb_frames,
        resize=resize,
        label_mode=label_mode,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
