#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_Xray_gaussian_rasterization_voxelization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    GaussianVoxelizationSettings,
    GaussianVoxelizer,
)
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
import numpy as np
import copy

def query(pc: GaussianModel,
          timestamp,
          recon_args,
          nVoxel,
          sVoxel,
          center,
          pipe,
          scaling_modifier=1.0,
          opacity_precomp=None,
          opacity_type='final_opacity'
          ):
    """
    Query a volume with voxelization.
    """

    voxel_settings = GaussianVoxelizationSettings(
        scale_modifier=scaling_modifier,
        nVoxel_x=int(nVoxel[0]),
        nVoxel_y=int(nVoxel[1]),
        nVoxel_z=int(nVoxel[2]),
        sVoxel_x=float(sVoxel[0]),
        sVoxel_y=float(sVoxel[1]),
        sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        prefiltered=False,
        debug=pipe.debug,
    )
    voxelizer = GaussianVoxelizer(voxel_settings=voxel_settings)

    means3D = pc.get_xyz

    if timestamp != -1:
        ret = pc._field(pc, timestamp, recon_args)
        opacity = ret[opacity_type]

    if opacity_precomp is not None:
        opacity = opacity_precomp

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    vol_pred, radii, vol_geo = voxelizer(
        means3D=means3D,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "vol": vol_pred,
        "radii": radii,
        "vol_geo": vol_geo,
    }


def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    recon_args,
    TP_std,
    pipe,
    scaling_modifier=1.0,
    opacity_precomp=None,
    opacity_type='final_opacity',
    deformation_mode=None,
    deformation_timestamp_override=None,
    opacity_timestamp_override=None,
):
    """
    Render the scene.
    """

    screenspace_points = (torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0 )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        mode=1,   # mode = 1, default cone beam mode
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points

    timestamp = viewpoint_camera.timestamp
    if TP_std > 0:
        timestamp = timestamp + np.random.normal(loc=0, scale=TP_std)
    timestamp = np.clip(timestamp, 0, 1)

    # Decouple deformation-time and opacity-time when needed.
    deform_t = deformation_timestamp_override if (deformation_timestamp_override is not None) else timestamp
    field_t = opacity_timestamp_override if (opacity_timestamp_override is not None) else timestamp

    if pc.use_deformation:
        pc.apply_deformation(deform_t, mode=deformation_mode)

    ret = pc._field(pc, field_t, recon_args)
    opacity = ret[opacity_type]

    if opacity_precomp is not None:
        opacity = opacity_precomp

    dummy_opacity = opacity.clone().detach().requires_grad_(True)
    try:
        dummy_opacity.retain_grad()
    except:
        pass

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    rendered_image, radii, render_others = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        dummy_opacities=dummy_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "render_geo": render_others[0:1],
        "dummy_opacity": dummy_opacity,
    }
