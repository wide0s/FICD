import os
os.environ["XFORMERS_MORE_DETAILS"] = "1"

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

import argparse
import multiprocessing
import random
import nibabel as nib
import numpy as np
from pathlib import Path
import torch
import torchio as tio
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt


def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model, optimizer, epoch


def check_natural_int(value):
    try:
        v = int(value)
        if v < 1:
            raise ValueError('value < 1')
        return v
    except ValueError:
         raise argparse.ArgumentTypeError(f"{value} is not a valid natural number")


parser = argparse.ArgumentParser(
     description="Run inference with PyTorch's FICD model on an MRI 3D brain image."
)

parser.add_argument(
     '-e', '--epoch',
     choices=['epoch49', 'epoch50'],
     default='epoch49',
     help='epoch to load (default: epoch49)'
)

parser.add_argument(
     '-i', '--input',
     type=Path,
     required=True,
     help='input MRI image',
)

parser.add_argument(
     '-v', '--verbose',
     help='increase output verbosity',
     action="store_true"
)

parser.add_argument(
     '--num-inference-steps',
     type=int,
     default=1000,
     help='number of diffusion steps used when generating samples with a pre-trained model (default: 1000)'
)

parser.add_argument(
     '--num-mc-samples',
     type=check_natural_int,
     default=5,
     help='number of Monte-Carlo (MC) samples (default: 5, paper recommends 10)'
)

args = parser.parse_args()
_VERBOSE = args.verbose

num_mc_samples = args.num_mc_samples
if num_mc_samples == 1:
    print('Sets all seeds to 0 to ensure reproducibility')
    random.seed(0)
    np.random.seed(0)

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

if _VERBOSE:
    print(model)


validation_transform = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)),
    tio.transforms.Crop([11, 10, 20, 17, 0, 21]),
    tio.Resize((160, 180, 160))
]) # TODO: check this

MRI_path = args.input
subject = tio.Subject(
     mri=tio.ScalarImage(MRI_path))

validation_set = tio.SubjectsDataset(
    [subject], transform=validation_transform)

val_loader = torch.utils.data.DataLoader(
    validation_set,
    batch_size=1,
    num_workers=multiprocessing.cpu_count(),
)

scheduler = DDPMScheduler(num_train_timesteps=1000,
                          schedule="scaled_linear_beta",
                          beta_start=0.0005,
                          beta_end=0.0195)
inferer = DiffusionInferer(scheduler)
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=5e-5)

epoch_to_load = args.epoch
load_path = f'checkpoints/{epoch_to_load}_checkpoint.pt'
model, optimizer, epoch = load_checkpoint(model, optimizer, load_path)
print('Checkpoint', load_path)

predicted_images_data = []
base_file_name = None

num_inference_steps = args.num_inference_steps
for step, batch in enumerate(val_loader):
    for seed in range(num_mc_samples):
        print(f'\nGenerating diffusion sample {seed + 1} of {num_mc_samples}')
        torch.manual_seed(seed)
        print('MRI subject ',batch["mri"]["path"])
        condition = batch["mri"]["data"].to(device)

        input_noise = torch.randn((1, 1, 160, 180, 160))
        input_noise = input_noise.to(device)
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        with autocast(enabled=True):
            pred_PET, intermediates = inferer.sample(input_noise=input_noise,
                        diffusion_model=model,
                        scheduler=scheduler,
                        save_intermediates=True,
                        intermediate_steps=100,
                        conditioning=torch.unsqueeze(condition[0,:,:,:,:], 0))

        if base_file_name is None:
            base_file_name = str(batch["mri"]["stem"][0]) + '_' + str(epoch_to_load) \
                + '_steps' + str(num_inference_steps)

        torch.save(pred_PET, base_file_name + '_seed' + str(seed) + '.pt')

        image_data = pred_PET.squeeze().cpu().numpy().astype("float32") # extract numpy
        affine = np.eye(4) # affine matrix
        pet_nii = nib.Nifti1Image(image_data, affine)
        nii_file_name = base_file_name + '_seed' + str(seed) + '.nii.gz'
        nib.save(pet_nii, nii_file_name)

        if _VERBOSE:
            print(f'Done with synthesis of sample {seed} -> {nii_file_name}')

        predicted_images_data.append(image_data)

# averaging MC samples
image_data = np.mean(np.stack(predicted_images_data, axis=0), axis=0)
affine = np.eye(4)
nii_file_name = base_file_name + '_averaged' + '.nii.gz'
#nib.save(nib.Nifti1Image(image_data.astype(np.float32), affine), averaged_nii)
nib.save(nib.Nifti1Image(image_data, affine), nii_file_name)

if _VERBOSE:
    print(f'Saved MC-averaged PET -> {nii_file_name}')