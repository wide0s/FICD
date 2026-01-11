import argparse
from pathlib import Path
import os

import torch
import torch.nn as nn

import nibabel as nib
import numpy as np


class CommandException(Exception):
    pass


def load_tensor(filename: str | Path, filters=["pt", "nii.gz"]) -> torch.Tensor:
    if isinstance(filename, Path):
        filename = str(filename)
    if isinstance(filters, str):
        filters = [filters]
    elif not isinstance(filters, list):
        raise ValueError("The filters must be a list or a string.")
    if "pt" in filters and filename.endswith("pt"):
        tensor = torch.load(filename)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"The file '{filename}' does not contain a tensor.")
        if tensor.dtype != torch.float32:
            print(f"Warning: Tensor in {filename} has dtype {tensor.dtype}," \
                  " convert it to torch.float32.")
            tensor = tensor.to(dtype=torch.float32)
        return tensor
    if "nii.gz" in filters and filename.endswith("nii.gz"):
        image_data = nib.load(filename)
        numpy_array = image_data.get_fdata().astype(np.float32) # get_fdata() returns ndarray as np.float64;
                                                                # ensure data is float32, which is standard
                                                                # for PyTorch
        return torch.from_numpy(numpy_array)
    raise ValueError(f"The file '{filename}' has unknown format. The supported formats are " \
                     f"{filters}")


def pt2nii(args: argparse.Namespace) -> None:
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


def dcm2nii(args: argparse.Namespace) -> None:
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


def MAE(args: argparse.Namespace) -> None:
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


def mean_nii(args: argparse.Namespace) -> None:
    arrays, shape, affine, header = \
        [], None, None, None
    for path in args.input:
        if not path.is_file():
            raise ValueError(f"The file '{path}' does not exist or is not a file.")
        image = nib.load(path)
        if len(arrays) == 0:
            affine, header, shape = \
                image.affine, image.header, image.shape
            print(f"Using header and affine transform from {path}.")
        else:
            if shape != image.shape:
                raise ValueError(f"The {path} has different dimensions than the first image." \
                                 " Images must have the same dimensions to calculate the mean.")
            if not np.array_equal(affine, image.affine):
                print("Warning: Affine transforms differ. The output image will use the first" \
                      " image's transform.")
        arrays.append(
            image.get_fdata().astype(np.float32)
        )
    mean_array = sum(arrays) / len(arrays) # TODO: test me
    nib.save(
        nib.Nifti1Image(mean_array, affine, header),
        args.output
    )
    print(f"Saved in {args.output}")


parser = argparse.ArgumentParser(
    description="The processing utilitites for pt, DICOM and NIfTI files."
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
    description="Converts image in PyTorch\'s pt file to NIfTI.",
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
    description="Converts image in DICOM format to NIfTI.",
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
    description="Calculates the Mean Absolute Error (MAE).",
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
    type=str,
    help="device type (default: cpu)."
)
mae_parser.set_defaults(func=MAE)

# --- mean NIfTI commands ---
mean_nii_parser = subparsers.add_parser(
    "mean_nii",
    description="Creates a voxel-wise mean image from a list of NIfTI images.",
    help="Creates a voxel-wise mean image from a list of NIfTI images."
)
mean_nii_parser.add_argument(
    "-i", "--input",
    action="append",
    type=Path,
    required=True,
    help="(multiple) NIfTI images."
)
mean_nii_parser.add_argument(
    "-o", "--output",
    type=Path,
    default="mean.nii.gz",
    help="The path to the output NIfTI file (default: mean.nii.gz)."
)
mean_nii_parser.set_defaults(func=mean_nii)

args = parser.parse_args()
args.func(args)
