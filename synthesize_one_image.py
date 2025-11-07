import os
os.environ["XFORMERS_MORE_DETAILS"] = "1"

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

import multiprocessing
import random
import numpy as np
from pathlib import Path
import torch
import torchio as tio
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model, optimizer, epoch

# 1
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

# 2
validation_transform = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)),
    tio.transforms.Crop([11, 10, 20, 17, 0, 21]),
    tio.Resize((160, 180, 160))
]) # TODO: check this

MRI_path = Path('./20200000_t1_sag_3D_20251018113337_5.nii.gz')
subject = tio.Subject(
    mri=tio.ScalarImage(MRI_path)
    )

validation_set = tio.SubjectsDataset(
    [subject], transform=validation_transform)

val_loader = torch.utils.data.DataLoader(
    validation_set,
    batch_size=1,
    num_workers=multiprocessing.cpu_count(),
)

# 3
scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195)
inferer = DiffusionInferer(scheduler)
optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-5)

# 4
epoch_to_load = 49
load_path = f'checkpoints/epoch{epoch_to_load}_checkpoint.pt'
model, optimizer, epoch = load_checkpoint(model, optimizer, load_path)

for step, batch in enumerate(val_loader):
    for seed in range(10):
        if seed == 0 and step == 0: # generate one output to show an example
                SEED=seed
                torch.manual_seed(SEED)
                print('MRI subject ',batch["mri"]["path"])
                condition = batch["mri"]["data"].to(device)

                input_noise = torch.randn((1, 1, 160, 180, 160))
                input_noise = input_noise.to(device)
                scheduler.set_timesteps(num_inference_steps=1000)
                with autocast(enabled=True):
                    pred_PET, intermediates = inferer.sample(input_noise=input_noise,
                                diffusion_model=model,
                                scheduler=scheduler,
                                save_intermediates=True,
                                intermediate_steps=100,
                                conditioning=torch.unsqueeze(condition[0,:,:,:,:], 0))

                print("Model output")
                plt.style.use("default")
                plotting_image_0 = np.concatenate([pred_PET[0, 0, :, :, 80].cpu(), np.flipud(pred_PET[0, 0, :, 90, :].cpu().T)], axis=1)
                plotting_image_1 = np.concatenate([np.flipud(pred_PET[0, 0, 90, :, :].cpu().T), np.zeros((160, 160))], axis=1)
                plt.imshow(np.concatenate([plotting_image_0, plotting_image_1], axis=0), cmap="gray")
                plt.tight_layout()
                plt.axis("off")
                plt.show()

                print("Input MRI")
                plt.style.use("default")
                plotting_image_0 = np.concatenate([condition[0, 0, :, :, 80].cpu(), np.flipud(condition[0, 0, :, 90, :].cpu().T)], axis=1)
                plotting_image_1 = np.concatenate([np.flipud(batch["mri"]["data"][0, 0, 90, :, :].cpu().T), np.zeros((160, 160))], axis=1)
                plt.imshow(np.concatenate([plotting_image_0, plotting_image_1], axis=0), cmap="gray")
                plt.tight_layout()
                plt.axis("off")
                plt.show()

                plt.figure()
                f, axarr = plt.subplots(1, 10, figsize=(50, 50))
                for i in range(10):
                    axarr[i].imshow(intermediates[i][0, 0, :, :, 90].cpu(), cmap="gray")
                plt.show()

                torch.save(pred_PET, 'epoch' + str(epoch_to_load) + '_val_' + str(batch["mri"]["stem"][0]) + 'seed' + str(SEED) + '_output.pt')
