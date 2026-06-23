# ============================================================
# CardioGS - Evaluation Script
# ============================================================
# Evaluation includes:
#   - 2D rendering (full-set and fixed-viewpoint)
#   - 3D voxel query reconstruction
#   - Mesh extraction, ICP alignment, and CD/HD metric computation
#
# NOTE: This skeleton documents the public evaluation interface.
#       Full implementation will be added in the complete release.
# ============================================================

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer_voxelizer import render
from gaussian_renderer_voxelizer import query
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model import GaussianModel
from utils.image_utils import *
from utils.mesh_utils import *
import numpy as np
import SimpleITK as sitk
from pyhocon import ConfigFactory
import json


def render_set(scene, gaussians, pipe, dataset, phase='test'):
    """Render all views and compute PSNR/SSIM metrics."""
    # ... implementation pending full release ...
    raise NotImplementedError("Full implementation will be added in the complete release.")


def render_set_fixview(scene, gaussians, pipe, dataset, phase='test'):
    """Render temporal sequence from a fixed viewpoint."""
    # ... implementation pending full release ...
    raise NotImplementedError("Full implementation will be added in the complete release.")


def voxel_query_reconstruction(scene, gaussians, pipe, dataset, phase='test'):
    """3D reconstruction via voxel query, with mesh extraction and metric evaluation."""
    # ... implementation pending full release ...
    raise NotImplementedError("Full implementation will be added in the complete release.")


def render_sets(dataset, pipeline, field_conf, args):
    with torch.no_grad():
        scale_bound = None
        if dataset.use_scale_bound:
            if dataset.scale_min is not None and dataset.scale_max is not None:
                scale_bound = np.asarray([dataset.scale_min, dataset.scale_max])
        gaussians = GaussianModel(field_conf)
        scene = Scene(dataset, gaussians, scale_bound, load_iteration=args.iteration, shuffle=False)

        if args.render_2d:
            render_set(scene, gaussians, pipeline, dataset, "test")
        if args.render_fixview:
            render_set_fixview(scene, gaussians, pipeline, dataset, "test")
        if args.VQR:
            voxel_query_reconstruction(scene, gaussians, pipeline, dataset, "test")


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--Nviews", type=int, default=30)
    parser.add_argument("--field_conf", type=str, default="./arguments/nerf.conf", help='field config file')
    parser.add_argument("--render_2d", action="store_true")
    parser.add_argument("--render_fixview", action="store_true")
    parser.add_argument("--VQR", action="store_true")

    args = get_combined_args(parser)

    print("Rendering " + args.model_path)

    field_conf = ConfigFactory.parse_file(args.field_conf)

    safe_state(args.quiet)

    model = model.extract(args)

    render_sets(model, pipeline.extract(args), field_conf, args)
