import bpy 
from utils.scene_utils import link_blend_file, hide_all_objects, center_object_origin, rand_init, rand_displace, reset_camera
import numpy as np 
from fast_render import displacement_render, luminance_render
from datetime import datetime


scene = bpy.context.scene

def lum_scene(config):
    blender_path = config['objects_with_text']
    link_blend_file(blender_path)

    for obj in bpy.context.scene.objects:
        if obj.name == 'Point':
            obj.data.energy = config['light_intensity_p0']
        elif obj.name == 'Point.001':
            obj.data.energy = config['light_intensity_p1']
        elif obj.name == 'Point.002':
            obj.data.energy = config['light_intensity_p2']
        elif obj.name == 'Point.003':
            obj.data.energy = config['light_intensity_p3']
       

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            mat = obj.active_material
            if mat and mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'EMISSION':
                        node.inputs['Strength'].default_value = config['light_intensity_p3'] / 20


def disp_scene(config):
     # load lib
    if config['textures_enabled']:
        blender_path = config['objects_with_text']
    else:
        blender_path = config['objects_path']
    
    link_blend_file(blender_path)

    # all objects from the .blend file 
    objects = [obj for obj in bpy.context.scene.collection.objects if obj.name.startswith("_")]
    dataset_size = len(objects)
    subset_size = config['num_obj']

    # fix origin for all mesh objects
    for obj in objects:
        if obj.type == 'MESH':
            center_object_origin(obj)

    hide_all_objects(objects)

    for i in range(config['num_scenes']):
        start_frame = i * 2
        end_frame = start_frame + 1

        # Hide everything initially
        for obj in objects:
            obj.hide_render = True
            obj.keyframe_insert(data_path="hide_render", frame=start_frame)
            obj.keyframe_insert(data_path="hide_render", frame=end_frame)

        # Choose and animate a subset
        subset_indices = np.random.choice(dataset_size, subset_size, replace=False)

        for idx in subset_indices:
            obj = objects[idx]

            # Make visible for render
            obj.hide_render = False
            obj.keyframe_insert(data_path="hide_render", frame=start_frame)
            obj.keyframe_insert(data_path="hide_render", frame=end_frame)

            # Initial position
            rand_init(config, obj)
            obj.keyframe_insert(data_path="location", frame=start_frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=start_frame)

            # Move/displace to new position
            rand_displace(config, obj)
            obj.keyframe_insert(data_path="location", frame=end_frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=end_frame)

            # Optionally hide again after animation
            obj.hide_render = True
            obj.keyframe_insert(data_path="hide_render", frame=end_frame + 1)

            # Get the camera object
            camera = bpy.context.scene.camera

            # Insert initial camera position and rotation at frame 1
            camera.keyframe_insert(data_path="location", frame=start_frame)
            camera.keyframe_insert(data_path="rotation_euler", frame=start_frame)

            # Move camera slightly to the left (in local or world X-axis)
            cam_loc = camera.location
            camera.location = (cam_loc.x + config["camera_x_trans"], 
                               cam_loc.y + config["camera_y_trans"], 
                               cam_loc.z + config["camera_z_trans"])  # adjust -0.1 as needed
            cam_rot = camera.rotation_euler
            camera.rotation_euler = (cam_rot.x + config["camera_x_rot"], 
                                     cam_rot.y + config["camera_y_rot"], 
                                     cam_rot.z + config["camera_z_rot"])
            
            # Insert new camera position at frame 2
            camera.keyframe_insert(data_path="location", frame=end_frame)
            camera.keyframe_insert(data_path="rotation_euler", frame=end_frame)
            
            reset_camera(config)


def create_dataset(config):
    bpy.context.scene.render.resolution_x = config['x_resolution']
    bpy.context.scene.render.resolution_y = config['y_resolution']

    if config["use_3d_bg"]:
        bpy.ops.wm.open_mainfile(filepath="/Users/Petter/Downloads/mini-supermarket/MiniMarket/Shop.blend")

    if config['seed']:
        np.random.seed(config['seed'])
    else:
        np.random.seed(datetime.now().timestamp())


    
    reset_camera(config)

    if config['train_mode'] == 'disp': # learn a displacement distribution 
        disp_scene(config)
        displacement_render(config)
    elif config['train_mode'] == 'lum': # learn a luminance distribution
        lum_scene(config)
        luminance_render(config)
    else: # learn a fluctuation distribution 
        pass


    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
   