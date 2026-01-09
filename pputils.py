import argparse
from pathlib import Path
import os

import torch
import torch.nn as nn

import nibabel as nib
import numpy as np


class CommandException(Exception):
    pass


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

    print(f"Saved as {output}")


def dcm2nii(args: argparse.Namespace):
    if not args.directory.is_dir():
        raise ValueError(f"The path '{args.directory}' does not exist or is not a directory.")
    output_dir = input_dir = args.directory.resolve()
    if args.output:
        if args.output.exists() and not args.output.is_dir():
            raise CommandException(f"The '{args.output}' exists and is not a directory.")
        elif not args.output.exists():
            os.makedirs(args.output)
        output_dir = args.output.resolve()
    wrm = args.write_mode
    cmd = (
        f"dcm2niix -w {wrm} -t y -z y -f %f_%p_subj_%i_tm_%t_%s -o {output_dir} {input_dir}"
    )
    if args.verbose:
        print(cmd)
    return_code = os.system(cmd)
    if return_code != 0:
        raise CommandException(f"Error converting to NIfTI format: {return_code}.")
    print(f"Saved in {output_dir}")


def load_tensor(filename: str) -> torch.Tensor:
    if filename.endswith("pt"):
        data = torch.load(filename) # loads an arbitrary Python object saved
                                    # with torch.save()
        if not isinstance(data, torch.Tensor):
            raise ValueError(f"The file '{filename}' does not contain a tensor.")
        return data
    if filename.endswith("nii.gz"):
        image_data = nib.load(filename)
        numpy_array = image_data.get_fdata().astype(np.float32) # ensure data is float32,
                                                                # which is standard for PyTorch
        return torch.from_numpy(numpy_array)
    raise ValueError(f"The file '{filename}' has unknown format. It must be .pt or .nii.gz.")


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
    description="The post-processing utilitites for pt, DICOM and NIfTI files."
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

# --- dcm2nii commands ---
dcm2nii_parser = subparsers.add_parser(
    "dcm2nii",
    help="Converts image in DICOM format to NIfTI."
)
dcm2nii_parser.add_argument(
    "directory",
    type=Path,
    help="The path to the input directory with DCM files."
)
dcm2nii_parser.add_argument(
    "-o", "--output",
    type=Path,
    help="The output directory (omit to save to input directory)."
)
dcm2nii_parser.add_argument(
    "--write-mode",
    choices=[0, 1, 2],
    default=0,
    type=int,
    help="Write behavior for name conflicts (default 0: 0=skip duplicates, 1=overwrite, 2=add suffix)."
)
dcm2nii_parser.set_defaults(func=dcm2nii)

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
