import bpy
import numpy as np 
from PIL import Image, ImageOps
import os 

import numpy as np
from skimage import color
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
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

    base_path = config['render']['output_folder']
    exr_path = config['render']['tmp_dump_path']


    for idx_scene in range(config['scene']['num_scenes']):
        if not config['background']['use_3d']:
            add_background(config)

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
            bpy.context.scene.render.motion_blur_position = 'END'

        # Render the second frame (PNG)
        bpy.context.scene.frame_set(end_frame)  
        bpy.context.scene.render.filepath = f"{img_path}/frame_1.png"
        bpy.ops.render.render(write_still=True)

        bpy.context.scene.render.use_motion_blur = False

        if config['effects']['inverted_colors']:
            img = Image.open(f"{img_path}/frame_1.png")
            inverted_img = ImageOps.invert(img.convert("RGB"))
            inverted_img.save(f"{img_path}/frame_1.png")

    print("Rendering complete.")

    if config['stats']['calc_ciede2000']:
        score = compute_ciede2000_score(f"{img_path}/frame_0.png")
        print(f"CIEDE2000 score: {score:.2f}")

# def render_dataset(config):
#     # Set up Cycles
#     bpy.context.scene.render.engine = 'CYCLES'
#     bpy.context.scene.cycles.feature_set = 'SUPPORTED'
#     bpy.context.scene.cycles.sample_clamp_indirect = 10.0
#     bpy.context.scene.cycles.samples = config['render']['nr_of_samples']
#     bpy.context.scene.cycles.use_adaptive_sampling = True
#     bpy.context.scene.cycles.use_denoising = True
#     bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
#     bpy.context.scene.cycles.max_bounces = 8
#     bpy.context.scene.cycles.diffuse_bounces = 4
#     bpy.context.scene.cycles.glossy_bounces = 4
#     bpy.context.scene.cycles.transmission_bounces = 4
#     bpy.context.scene.cycles.use_persistent_data = True

#     # Setup devices
#     prefs = bpy.context.preferences.addons['cycles'].preferences
#     prefs.get_devices(True)
#     device_types = {device.type for device in prefs.devices}
#     if 'METAL' in device_types:
#         prefs.compute_device_type = 'METAL'
#     elif 'CUDA' in device_types:
#         prefs.compute_device_type = 'CUDA'
#     elif 'OPTIX' in device_types:
#         prefs.compute_device_type = 'OPTIX'
#     elif 'OPENCL' in device_types:
#         prefs.compute_device_type = 'OPENCL'
#     else:
#         prefs.compute_device_type = 'NONE'
#         bpy.context.scene.cycles.device = 'CPU'
#     prefs.get_devices(True)
#     if prefs.compute_device_type != 'NONE':
#         bpy.context.scene.cycles.device = 'GPU'
#         for d in prefs.devices:
#             d.use = True

#     print("✅ Using:", prefs.compute_device_type)
#     print("Enabled devices:", [(d.name, d.type, d.use) for d in prefs.devices if d.use])

#     # Set resolution
#     bpy.context.scene.render.resolution_x = config['render']['resolution']['x']
#     bpy.context.scene.render.resolution_y = config['render']['resolution']['y']
#     bpy.context.scene.render.resolution_percentage = 100

#     # Set frame range for all scenes
#     num_scenes = config['scene']['num_scenes']
#     bpy.context.scene.frame_start = 0
#     bpy.context.scene.frame_end = num_scenes * 2 - 1

#     # Motion blur
#     bpy.context.scene.render.use_motion_blur = config['camera']['shutter_speed'] != 0.0
#     bpy.context.scene.render.motion_blur_shutter = config['camera']['shutter_speed']

#     # Set view layer passes
#     view_layer = bpy.context.view_layer
#     view_layer.use_pass_vector = True
#     view_layer.use_pass_z = True
#     view_layer.use_pass_object_index = True
#     view_layer.use_pass_position = True

#     # Paths
#     base_path = config['render']['output_folder']
#     exr_temp_path = config['render']['tmp_dump_path']
#     png_temp_path = os.path.join(exr_temp_path, "png_dump")
#     os.makedirs(exr_temp_path, exist_ok=True)
#     os.makedirs(png_temp_path, exist_ok=True)

#     # ---------- RENDER EXRs ----------
#     bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
#     bpy.context.scene.render.filepath = os.path.join(exr_temp_path, "frame_")
#     bpy.ops.render.render(animation=True)

#     # ---------- RENDER PNGs ----------
#     bpy.context.scene.render.image_settings.file_format = 'PNG'
#     bpy.context.scene.render.filepath = os.path.join(png_temp_path, "frame_")
#     bpy.ops.render.render(animation=True)

#     # ---------- POSTPROCESS AND MOVE ----------
#     for idx_scene in range(num_scenes):
#         flow_path = os.path.join(base_path, f"training/flow/scene_{idx_scene}")
#         img_path = os.path.join(base_path, f"training/clean/scene_{idx_scene}")
#         os.makedirs(flow_path, exist_ok=True)
#         os.makedirs(img_path, exist_ok=True)

#         for i in range(2):
#             frame_num = idx_scene * 2 + i
#             padded = f"{frame_num:04d}"

#             # EXR
#             exr_src = os.path.join(exr_temp_path, f"frame_{padded}.exr")
#             exr_dst = os.path.join(flow_path, f"frame_{i}.exr")
#             if os.path.exists(exr_src):
#                 shutil.move(exr_src, exr_dst)

#             # PNG
#             png_src = os.path.join(png_temp_path, f"frame_{padded}.png")
#             png_dst = os.path.join(img_path, f"frame_{i}.png")
#             if os.path.exists(png_src):
#                 if config['effects']['inverted_colors']:
#                     img = Image.open(png_src)
#                     inverted = ImageOps.invert(img.convert("RGB"))
#                     inverted.save(png_dst)
#                     os.remove(png_src)
#                 else:
#                     shutil.move(png_src, png_dst)

#         # Optional CIEDE2000 on frame 0
#         if config['stats']['calc_ciede2000']:
#             ref_img = os.path.join(img_path, "frame_0.png")
#             if os.path.exists(ref_img):
#                 score = compute_ciede2000_score(ref_img)
#                 print(f"Scene {idx_scene} CIEDE2000: {score:.2f}")

#     print("✅ All scenes rendered and organized.")

        

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