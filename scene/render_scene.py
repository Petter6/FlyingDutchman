import bpy
import numpy as np 
from PIL import Image, ImageOps
import os 
import sys

import numpy as np
from skimage import color
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from utils.folder import get_img_path
import random

def add_background(config):
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    nodes.clear()

    bg = nodes.new(type="ShaderNodeBackground")
    env_texture = nodes.new(type="ShaderNodeTexEnvironment")
    output = nodes.new(type="ShaderNodeOutputWorld")

    hdri_path = os.path.join(config['background']['2d_path'], np.random.choice(os.listdir(config['background']['2d_path']))) 
   
    env_texture.image = bpy.data.images.load(hdri_path)

    links.new(env_texture.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    return bg  # So we can animate it in `create_scene`

def render_dataset(config):
    # Set the render engine to Cycles (required for GPU rendering)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.feature_set = 'SUPPORTED'
    bpy.context.scene.cycles.sample_clamp_indirect = 10.0

    bpy.context.scene.cycles.samples = config['render']['nr_of_samples']  # Adjust this value as needed
    bpy.context.scene.cycles.use_adaptive_sampling = True

    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'  # or 'OPTIX' if available

    bpy.context.scene.cycles.max_bounces = 8  # Adjust as needed
    bpy.context.scene.cycles.diffuse_bounces = 4
    bpy.context.scene.cycles.glossy_bounces = 4
    bpy.context.scene.cycles.transmission_bounces = 4

    bpy.context.scene.cycles.use_persistent_data = True

    # Access Cycles preferences
    prefs = bpy.context.preferences.addons['cycles'].preferences

    # Initialize devices
    prefs.get_devices(True)

    # Check available device types
    device_types = {device.type for device in prefs.devices}

    # Prefer Metal on macOS
    if 'METAL' in device_types:
        prefs.compute_device_type = 'METAL'
    elif 'CUDA' in device_types:
        prefs.compute_device_type = 'CUDA'
    elif 'OPTIX' in device_types:
        prefs.compute_device_type = 'OPTIX'
    elif 'OPENCL' in device_types:
        prefs.compute_device_type = 'OPENCL'
    else:
        print("⚠️ No compatible GPU backend found. Falling back to CPU.")
        bpy.context.scene.cycles.device = 'CPU'
        prefs.compute_device_type = 'NONE'

    # Re-initialize after setting the device type
    prefs.get_devices(True)

    # Enable GPU rendering if a valid type was set
    if prefs.compute_device_type != 'NONE':
        bpy.context.scene.cycles.device = 'GPU'

        # Enable all available devices
        for device in prefs.devices:
            device.use = True

    print("✅ Using:", prefs.compute_device_type)
    print("Enabled devices:", [(d.name, d.type, d.use) for d in prefs.devices if d.use])
    
    #Set the resolution
    bpy.context.scene.render.resolution_x = config['render']['resolution']['x']
    bpy.context.scene.render.resolution_y = config['render']['resolution']['y']
    bpy.context.scene.render.resolution_percentage = 100  # Ensure it's not scaled down

    exr_path = config['render']['tmp_dump_path']


    for idx_scene in range(config['scene']['num_scenes']):
        if not config['background']['use_3d']:
            add_background(config)

        img_path = get_img_path(idx_scene, config)

        start_frame = idx_scene * 2
        end_frame = start_frame + 1

        bpy.context.scene.frame_set(start_frame) 
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'

        # Ensure the active view layer exists
        view_layer = bpy.context.view_layer

        if view_layer is None:
            print("No active view layer found. Please ensure a valid view layer is selected.")
        else:
            # Access render pass properties safely
            try:
                view_layer.use_pass_combined = True
                print("Successfully set use_pass_combined.")
            except AttributeError:
                print("Failed to set use_pass_combined. Check render settings and engine.")

        # Enable necessary passes
        # view_layer.use_pass_combined = True
        view_layer.use_pass_vector = True
        view_layer.use_pass_z = True
        view_layer.use_pass_object_index = True
        view_layer.use_pass_position = True

        # Render exr-file of the first frame 
            
        bpy.context.scene.render.filepath = os.path.join(exr_path, f"scene_{idx_scene}.exr")
        
        bpy.ops.render.render(write_still=True)

        bpy.context.scene.render.image_settings.file_format = 'PNG'

        # Render the first frame (PNG)
        bpy.context.scene.frame_set(start_frame)  
        bpy.context.scene.render.filepath = f"{img_path}0.png"
        bpy.ops.render.render(write_still=True)

        if config['effects']['inverted_colors']:
            img = Image.open(f"{img_path}/frame_0.png")
            inverted_img = ImageOps.invert(img.convert("RGB"))
            inverted_img.save(f"{img_path}/frame_0.png")

        if config['camera']['shutter_speed'] != 0.0: 
            # Enable motion blur
            bpy.context.scene.render.use_motion_blur = True

            # Adjust motion blur settings
            bpy.context.scene.render.motion_blur_shutter = config['camera']['shutter_speed']  
            bpy.context.scene.render.motion_blur_position = 'END'

        # Render the second frame (PNG)
        bpy.context.scene.frame_set(end_frame)  
        bpy.context.scene.render.filepath = f"{img_path}1.png"
        bpy.ops.render.render(write_still=True)

        bpy.context.scene.render.use_motion_blur = False

        if config['effects']['inverted_colors']:
            img = Image.open(f"{img_path}/frame_1.png")
            inverted_img = ImageOps.invert(img.convert("RGB"))
            inverted_img.save(f"{img_path}/frame_1.png")

    print("Rendering complete.")

   

