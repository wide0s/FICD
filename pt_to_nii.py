import argparse
from pathlib import Path
import torch
import nibabel as nib
import numpy as np

parser = argparse.ArgumentParser(
    'Converts image in PyTorch\'s pt file to NIfTI.'
)

parser.add_argument(
    'filename',
    type=Path,
    help='The path to the pt file to process.')

args = parser.parse_args()

tensor = torch.load(args.filename)

if isinstance(tensor, torch.Tensor):
    arr = tensor.squeeze().cpu().numpy().astype("float32")
elif isinstance(tensor, dict):
    # if saved as dict
    arr = tensor["pred"].squeeze().cpu().numpy().astype("float32")
else:
    raise ValueError("Unknown .pt format")

# Affine matrix
# Replace with real affine if you saved it earlier
affine = np.eye(4)

# Save as NIfTI
pet_nii = nib.Nifti1Image(arr, affine)
nib.save(pet_nii, "output.nii.gz")

print("Saved: output.nii.gz")


