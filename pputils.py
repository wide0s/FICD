import argparse
from typing import Any

import torch
import torch.nn as nn

import nibabel as nib
import numpy as np


def pt2nii(args: argparse.Namespace):
    if not args.filename.endswith("pt"):
        raise ValueError(f"The input filename must ends with pt")

    output = args.output if args.output else "output.nii.gz"
    if not output.endswith("nii.gz"):
        raise ValueError(f"The output filename must ends with nii.gz")

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
    nib.save(pet_nii, output)

    print(f"Saved: {output}")


def load_tensor(filename: str) -> torch.Tensor:
    if filename.endswith("pt"):
        data = torch.load(filename) # loads an arbitrary Python object saved
                                    # with torch.save()
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"The file {filename} does not contain a tensor.")
        return data
    if filename.endswith("nii.gz"):
        image_data = nib.load(filename)
        numpy_array = image_data.get_fdata().astype(np.float32) # ensure data is float32,
                                                                # which is standard for PyTorch
        return torch.from_numpy(numpy_array)
    raise ValueError(f"The file {filename} has unknown format. It must be .pt or .nii.gz.")


def MAE(args: argparse.Namespace):
    if args.verbose:
        print(f"device: {args.device}")
    device = torch.device(args.device)
    tensors = []
    for image in (args.image1, args.image2):
        if args.verbose:
            print(f"image: {image}")
        tensors.append(
            load_tensor(image).to(device)
        )
    criterion_mae = nn.L1Loss(reduction="mean")
    mae_loss = criterion_mae(*tensors)
    print(f"MAE loss: {mae_loss.item()}")


parser = argparse.ArgumentParser(
    description="The post-processing utilitites for pt and NIfTI files."
)

parser.add_argument(
    "-v", "--verbose",
    action="store_true",
    help="increase output verbosity"
)

subparsers = parser.add_subparsers(
    title="commands",
    dest="command",
    metavar="<command>",
    required=True
)

# --- pt2nii commands ----
pt2nii_parser = subparsers.add_parser(
    "pt2nii",
    help="Converts image in PyTorch\'s pt file to NIfTI."
)
pt2nii_parser.add_argument(
    "filename",
    type=str,
    help="The path to the pt file to process."
)
pt2nii_parser.add_argument(
    "-o", "--output",
    type=str,
    help="The output file name (default: output.nii.gz)."
)
pt2nii_parser.set_defaults(func=pt2nii)

# --- mae commands ---
mae_parser = subparsers.add_parser(
    "mae",
    help="Calculates the Mean Absolute Error (MAE)."
)
mae_parser.add_argument(
    "image1",
    type=str,
    help="The path to the first pt or NIfTI file to calculate MAE."
)
mae_parser.add_argument(
    "image2",
    type=str,
    help="The path to the second pt or NIfTI file to calculate MAE."
)
mae_parser.add_argument(
    "--device",
    choices=["cpu", "cuda"],
    default="cpu",
    help="device type (default: cpu)."
)
mae_parser.set_defaults(func=MAE)

args = parser.parse_args()
args.func(args)
