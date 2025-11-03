from torch.utils.data import DataLoader
from datasets.utd_mhad_rgbd import UTDMHADRGBD

ds = UTDMHADRGBD("data/utd_mhad/index.csv", rgb_frames=8)
dl = DataLoader(ds, batch_size=2, shuffle=True)

batch = next(iter(dl))
print(batch["rgb"].shape)    # expect (2, 8, 3, 224, 224)
print(batch["depth"].shape)  # expect (2, 8, 1, 224, 224)
print(batch["label"])
