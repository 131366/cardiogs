# CardioGS

Deformation-Driven 4D Gaussian Splatting with Self-Supervised Cardiac Clock Learning for Rotational DSA Reconstruction.

This repository contains the public code skeleton for CardioGS, including the data interface, camera geometry, renderer wrapper, utility modules, and documented entry points for training and evaluation. The complete training logic, learned cardiac-clock implementation, deformation graph, appearance field, pretrained weights, and example data will be released with the final public code release.

## Overview

CardioGS is a deformation-driven 4D Gaussian Splatting framework for rotational DSA reconstruction. It separates quasi-periodic cardiac motion from monotonic contrast-agent transport.

Key components include:

- ClockNet: a self-supervised monotonic cardiac clock learned with Gauss-Legendre quadrature.
- Control-node deformation graph: sparse deformation control nodes with KNN-RBF interpolation and cycle-consistency regularization.
- Canonical-space appearance field: dual hash-grid opacity modeling conditioned on acquisition time.
- Conditioning curriculum: gradual transition from linear-time to clock-driven deformation conditioning.

## Repository Layout

```text
CardioGS/
|-- arguments/
|   |-- __init__.py          # Argument parsing
|   `-- nerf.conf            # Field and deformation configuration
|-- gaussian_renderer_voxelizer/
|   `-- __init__.py          # X-ray rasterizer and voxelizer wrapper
|-- scene/
|   |-- __init__.py          # Scene management
|   |-- cameras.py           # Cone-beam X-ray camera model
|   |-- dataset_readers.py   # Data loading and FDK initialization
|   |-- gaussian_model.py    # GaussianModel and cardiac-clock interface
|   |-- deformation.py       # Control-node deformation graph interface
|   `-- field.py             # Canonical-space appearance field interface
|-- utils/
|   |-- camera_utils.py      # Camera list construction
|   |-- general_utils.py     # Math utilities
|   |-- graphics_utils.py    # Projection and geometry helpers
|   |-- image_utils.py       # PSNR, SSIM, normalization
|   |-- loss_utils.py        # L1, SSIM, TV, correlation losses
|   |-- mesh_utils.py        # Marching cubes, ICP, CD/HD metrics
|   `-- system_utils.py      # Filesystem utilities
|-- train.py                 # Training entry point
`-- test.py                  # Evaluation entry point
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- tiny-cuda-nn
- SimpleITK
- Open3D
- PyMeshLab
- scikit-image
- PyMCubes
- OpenCV
- pyhocon
- tqdm
- plyfile

The project also expects CUDA extensions or local modules used by the original implementation:

- `diff_Xray_gaussian_rasterization_voxelization`
- `simple_knn`
- `ct.astra_ct`

These components are not vendored in this skeleton.

## Data Format

Input data should be organized as:

```text
data/
|-- transforms.json      # Camera parameters: angles, SAD, SID, spacing, etc.
|-- mask_run.nii.gz      # Mask-run projections
|-- fill_run.nii.gz      # Fill-run projections
`-- mesh_ref.obj         # Optional ground-truth vessel mesh for evaluation
```

## Usage

The current repository is a skeleton release. The commands below document the intended interface for the full release.

```bash
python train.py -s <data_path> -m <output_path> --Nviews 30 --field_conf ./arguments/nerf.conf \
    --use_phase_learning --phase_blend_max 1.0 --phase_cycle_weight 0.2 --phase_tv_weight 0.01

python test.py -m <output_path> --field_conf ./arguments/nerf.conf --render_2d --VQR
```

## Citation

```bibtex
@inproceedings{cardiogs2026,
  title={CardioGS: Deformation-Driven 4D Gaussian Splatting with Self-Supervised Cardiac Clock Learning for Rotational DSA Reconstruction},
  author={CardioGS Authors},
  booktitle={MICCAI},
  year={2026}
}
```

## Acknowledgements

This codebase builds upon 3D Gaussian Splatting and 4DRGS. We thank the authors for their open-source contributions.
