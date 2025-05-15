import bpy
import numpy as np 
import OpenEXR
import Imath
import sys
from PIL import Image

def displacement_render(config):
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'

    bpy.context.scene.render.resolution_x = config['x_resolution']
    bpy.context.scene.render.resolution_y = config['y_resolution']
    bpy.context.scene.render.resolution_percentage = 100

    view_layer = bpy.context.view_layer
    view_layer.use_pass_vector = True
    bpy.context.scene.use_nodes = False
    bpy.context.scene.render.use_compositing = False
    bpy.context.scene.render.use_sequencer = False

    eevee = bpy.context.scene.eevee
    eevee.taa_render_samples = 1

    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.context.scene.render.image_settings.exr_codec = 'ZIP'

    if config['use_3d_bg']:
        layer_z = "RenderLayer.Vector.Z"
        layer_w = "RenderLayer.Vector.W"
    else:
        layer_z = "ViewLayer.Vector.Z"
        layer_w = "ViewLayer.Vector.W"

    h, w = config['y_resolution'], config['x_resolution']
    all_flows = []

    for scene_idx in range(0, config['num_scenes'] * 2, 2):
        # Render the frame
        bpy.context.scene.frame_set(scene_idx)
        exr_path = f"{config['tmp_exr_path']}/_frame_{scene_idx:03d}.exr"
        bpy.context.scene.render.filepath = exr_path
        bpy.ops.render.render(write_still=True)

        # Load flow data directly
        file = OpenEXR.InputFile(exr_path)
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

        vec_x = np.frombuffer(file.channel(layer_z, FLOAT), dtype=np.float32).reshape(h, w)
        vec_y = np.frombuffer(file.channel(layer_w, FLOAT), dtype=np.float32).reshape(h, w)

        flow = np.stack([-vec_x, vec_y], axis=-1)  # (H, W, 2)
        all_flows.append(flow)

        file.close()

    # Stack and save all flows
    all_flows = np.stack(all_flows, axis=0)  # shape: (num_scenes, H, W, 2)
    
    np.save("./tmp/all_flows.npy", all_flows)


def luminance_render(config):
    # Set Cycles 
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.feature_set = 'SUPPORTED'
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    bpy.context.scene.cycles.use_persistent_data = True
    bpy.context.scene.cycles.sample_clamp_indirect = 10.0

    # GPU rendering
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'METAL'
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    for device in bpy.context.preferences.addons['cycles'].preferences.devices:
        device.use = True

    # Set resolution
    bpy.context.scene.render.resolution_x = config['x_resolution']
    bpy.context.scene.render.resolution_y = config['y_resolution']
    bpy.context.scene.render.resolution_percentage = 100

    h, w = config['y_resolution'], config['x_resolution']
    brightness_maps = []

    for scene_idx in range(0, config['num_scenes'] * 2, 2):
        # Set render settings to PNG
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.frame_set(scene_idx)

        png_path = f"{config['tmp_exr_path']}_frame_{scene_idx:03d}.png"
        bpy.context.scene.render.filepath = png_path
        bpy.ops.render.render(write_still=True)

        # Read PNG and convert to luminance
        img = Image.open(png_path).convert('RGB')
        rgb = np.asarray(img).astype(np.float32) / 255.0  # shape: (H, W, 3)

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        brightness = (r + g + b) / 3.0  
        brightness_maps.append(brightness)

        print(f"🔹 Frame {scene_idx:03d} | Max R: {r.max():.3f}, G: {g.max():.3f}, B: {b.max():.3f}, Brightness: {brightness.max():.3f}")

    brightness_maps = np.stack(brightness_maps, axis=0)  # shape: (num_frames, H, W)
    max_brightness = np.max(brightness_maps)
    print(f"🔆 Max brightness across all frames: {max_brightness:.4f}")
    np.save("./tmp/brightness_maps.npy", brightness_maps)
