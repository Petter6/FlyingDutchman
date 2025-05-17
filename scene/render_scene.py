import bpy
import numpy as np 
from PIL import Image, ImageOps
import os 

import numpy as np
from skimage import color
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
import random

import sys

def compute_ciede2000_score(image_path, sample_size=10000):
    # Load and convert to RGB
    img = Image.open(image_path).convert("RGB")
    img_np = np.asarray(img) / 255.0  # Normalize

    # Convert to Lab
    lab_img = color.rgb2lab(img_np)

    # Reshape for sampling
    h, w, _ = lab_img.shape
    pixels = lab_img.reshape(-1, 3)

    # Random sample of pixel pairs
    if len(pixels) < sample_size:
        sample_size = len(pixels)
    sample_indices = random.sample(range(len(pixels)), sample_size)

    total_diff = 0.0
    count = 0
    for i in range(0, sample_size - 1, 2):
        lab1 = LabColor(*pixels[sample_indices[i]])
        lab2 = LabColor(*pixels[sample_indices[i+1]])
        total_diff += delta_e_cie2000(lab1, lab2)
        count += 1

    avg_diff = total_diff / count if count > 0 else 0
    return avg_diff

def render_dataset(config):
    # Set the render engine to Cycles (required for GPU rendering)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.feature_set = 'SUPPORTED'
    bpy.context.scene.cycles.sample_clamp_indirect = 10.0

    bpy.context.scene.cycles.samples = 10  # Adjust this value as needed
    bpy.context.scene.cycles.use_adaptive_sampling = True

    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'  # or 'OPTIX' if available

    bpy.context.scene.cycles.max_bounces = 8  # Adjust as needed
    bpy.context.scene.cycles.diffuse_bounces = 4
    bpy.context.scene.cycles.glossy_bounces = 4
    bpy.context.scene.cycles.transmission_bounces = 4

    bpy.context.scene.cycles.use_persistent_data = True

    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons["cycles"].preferences

    available_devices = cycles_prefs.get_devices()
    available_types = [d.type for d in cycles_prefs.devices]

    if "CUDA" in available_types:
        cycles_prefs.compute_device_type = "CUDA"
    elif "OPTIX" in available_types:
        cycles_prefs.compute_device_type = "OPTIX"
    elif "HIP" in available_types:
        cycles_prefs.compute_device_type = "HIP"
    elif "ONEAPI" in available_types:
        cycles_prefs.compute_device_type = "ONEAPI"
    else:
        cycles_prefs.compute_device_type = "NONE"
        print("⚠️ No compatible GPU backend found — falling back to CPU.")

    bpy.context.scene.cycles.device = 'GPU' if cycles_prefs.compute_device_type != 'NONE' else 'CPU'
    
    #Set the resolution
    bpy.context.scene.render.resolution_x = config['render']['resolution']['x']
    bpy.context.scene.render.resolution_y = config['render']['resolution']['y']
    bpy.context.scene.render.resolution_percentage = 100  # Ensure it's not scaled down

    base_path = config['render']['output_folder']
    exr_path = config['render']['tmp_dump_path']


    for idx_scene in range(config['scene']['num_scenes']):
        flow_path = os.path.join(base_path, 'training/flow/scene_' + str(idx_scene))
        img_path = os.path.join(base_path, 'training/clean/scene_' + str(idx_scene))

        os.mkdir(flow_path)
        os.mkdir(img_path)

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
        bpy.context.scene.render.filepath = f"{img_path}/frame_0.png"
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

        # Render the second frame (PNG)
        bpy.context.scene.frame_set(end_frame)  
        bpy.context.scene.render.filepath = f"{img_path}/frame_1.png"
        bpy.ops.render.render(write_still=True)

        if config['effects']['inverted_colors']:
            img = Image.open(f"{img_path}/frame_1.png")
            inverted_img = ImageOps.invert(img.convert("RGB"))
            inverted_img.save(f"{img_path}/frame_1.png")

    print("Rendering complete.")

    if config['stats']['calc_ciede2000']:
        score = compute_ciede2000_score(f"{img_path}/frame_0.png")
        print(f"CIEDE2000 score: {score:.2f}")

        

def render_market(frame1, frame2):
    # Set the render engine to Cycles (required for GPU rendering)
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.feature_set = 'SUPPORTED'
        bpy.context.scene.cycles.sample_clamp_indirect = 10.0

        # Enable GPU rendering
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'METAL'
        bpy.context.preferences.addons['cycles'].preferences.get_devices()

        # Ensure the GPU device is selected for rendering
        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            device.use = True  # Enable all available devices (GPU)

        #Set the resolution
        bpy.context.scene.render.resolution_x = 960
        bpy.context.scene.render.resolution_y = 540
        bpy.context.scene.render.resolution_percentage = 100  # Ensure it's not scaled down

        bpy.context.scene.cycles.samples = 20  # Adjust this value as needed
        bpy.context.scene.cycles.use_adaptive_sampling = True

        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'  # or 'OPTIX' if available

        bpy.context.scene.cycles.max_bounces = 8  # Adjust as needed
        bpy.context.scene.cycles.diffuse_bounces = 4
        bpy.context.scene.cycles.glossy_bounces = 4
        bpy.context.scene.cycles.transmission_bounces = 4

        bpy.context.scene.render.image_settings.file_format = 'PNG'

        # Render the first frame
        bpy.context.scene.frame_set(frame1)  # Set to the first frame
        bpy.context.scene.render.filepath = f"./out/frame{frame1}.png"
        bpy.ops.render.render(write_still=True)

        # Render the second frame
        bpy.context.scene.frame_set(frame2)  # Set to the second frame
        bpy.context.scene.render.filepath = f"./out/frame{frame2}.png"
        bpy.ops.render.render(write_still=True)

        print("Rendering complete.")