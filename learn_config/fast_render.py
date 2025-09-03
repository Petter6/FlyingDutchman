import bpy
import numpy as np 
import OpenEXR
import Imath
import sys
from PIL import Image
import os

from contextlib import contextmanager

def displacement_render(config):
    scene = bpy.context.scene
    tmp = config['render']['tmp_dump_path']
    N = config['scene']['num_scenes']

    # 1) Set up for multilayer EXR output & Vector pass
    scene.render.engine                = 'BLENDER_EEVEE_NEXT'
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
    scene.render.image_settings.color_depth    = '16'
    scene.render.image_settings.exr_codec      = 'ZIP'

    # enable Vector pass
    view_layer = bpy.context.view_layer
    view_layer.use_pass_vector = True
    scene.use_nodes      = False
    scene.render.use_compositing = False
    scene.render.use_sequencer   = False
    scene.eevee.taa_render_samples = 1

    # 2) Tell Blender to render frames 0 through (2*N-1)
    scene.frame_start = 0
    scene.frame_end   = 2 * N - 1
    # Filepath template: Blender will replace #### with frame number
    scene.render.filepath = f"{tmp}/frame_####"

    # 3) Fire off the animation render
    bpy.ops.render.render(animation=True)

    # scene.render.image_settings.file_format = 'PNG'
    # bpy.ops.render.render(animation=True)

    # 4) Now load only the odd frames’ EXRs and grab Z/W channels
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    h = config['render']['resolution']['y']
    w = config['render']['resolution']['x']

    if config['background']['use_3d']:
            layer_z = "RenderLayer.Vector.Z"
            layer_w = "RenderLayer.Vector.W"
    else:
        layer_z = "ViewLayer.Vector.Z"
        layer_w = "ViewLayer.Vector.W"

    all_flows = []
    for i in range(0, 2 * N, 2):   
        exr_path = f"{tmp}/frame_{i:04d}.exr"
        exr = OpenEXR.InputFile(exr_path)

        vx = np.frombuffer(exr.channel(layer_z, FLOAT), dtype=np.float32).reshape(h, w)
        vy = np.frombuffer(exr.channel(layer_w, FLOAT), dtype=np.float32).reshape(h, w)
        exr.close()

        flow = np.stack([vx, vy], axis=-1)  # shape (H, W, 2)
        all_flows.append(flow)

    # all_flows: (num_scenes, H, W, 2)
    magnitudes = np.linalg.norm(all_flows, axis=-1)     # (num_scenes, H, W)
    mask       = magnitudes > 0.01                     # boolean mask
    valid_mag  = magnitudes[mask]                 # 1D array of positive floats

    if valid_mag.size == 0:
        return 0.0

    avg_magnitude = valid_mag.mean()                   # scalar ≥ 0
    return float(avg_magnitude)

def luminance_render(config):
    # 1) Cycles-instellingen
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 1
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    scene.cycles.use_persistent_data = True
    scene.cycles.sample_clamp_indirect = 10.0

    N = config['scene']['num_scenes']
    tmp = config['render']['tmp_dump_path']

    # GPU
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'METAL'
    prefs.get_devices()
    for dev in prefs.devices:
        dev.use = True

    # 2) Door alle even frames renderen en helderheid middelen
    frame_means = []
    for frame in range(0, N * 2, 2):
        scene.render.image_settings.file_format = 'PNG'
        scene.frame_set(frame)

        png_path = f"{tmp}/frame_{frame:04d}.png"
        scene.render.filepath = png_path
        bpy.ops.render.render(write_still=True)

        # Lees PNG → luminantie
        rgb = np.asarray(Image.open(png_path).convert('RGB'), dtype=np.float32) / 255.0
        brightness = rgb.mean(axis=2)          # (H, W) → luminantie per pixel
        frame_means.append(float(brightness.mean()))

        print(f"🔹 Frame {frame:03d} | avg brightness: {frame_means[-1]:.3f}")

    # 3) Gemiddelde helderheid over alle frames retourneren
    if not frame_means:
        return 0.0
    avg_brightness = float(np.mean(frame_means))
    print(f"🔆 Average brightness across all frames: {avg_brightness:.4f}")
    return avg_brightness
