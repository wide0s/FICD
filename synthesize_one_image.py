import os
os.environ["XFORMERS_MORE_DETAILS"] = "1"
from generative.networks.nets import DiffusionModelUNet

from pathlib import Path
import torch
#import torchio as tio

# display MRI
#MRI_path = Path('./20200000_t1_sag_3D_20251018113337_5.nii.gz')
#one_subject = tio.Subject(mri=tio.ScalarImage(MRI_path))
#print(one_subject.mri)
#one_subject.plot()

device = torch.device("cuda")

model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=2,
    out_channels=1,
    num_channels=[16, 32, 64],
    attention_levels=[False, False, True],
    num_head_channels=[0, 0, 64],
    num_res_blocks=2,
    norm_num_groups=8,
    use_flash_attention=True,
    with_conditioning=True,
    cross_attention_dim=64
)

model.to(device)
print(model)
