import bpy
from flow.calc_flow import exr2flow
import numpy as np
from scene.render_scene import render_dataset
from utils.coordinate import local_to_world
from scene.add_elements import add_fog, add_background
from utils.scene_utils import delete_all, set_camera, reset_camera, hide_all_objects, center_object_origin, link_blend_file, rand_init, rand_displace, create_random_material
import shutil 
import time


light_object = None
scene = bpy.context.scene

def kill_all_lights():
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0

    for obj in bpy.data.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'SUN':
            obj.data.energy = 0.0

def load_objects(config):
    if config['objects']['textures_enabled']:
        blender_path = config['objects']['with_text']
    else:
        blender_path = config['objects']['path']
   
    # Link objects from .blend
    with bpy.data.libraries.load(blender_path, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects

    return data_to.objects

def assign_materials(objects, config):
    updated_objects = []
    for idx, obj in enumerate(objects):
        if obj is not None:
            obj.name = f"_{obj.name}"
            bpy.context.scene.collection.objects.link(obj)

            if not config['objects']['textures_enabled']:
                mat = create_random_material(idx)
                if len(obj.data.materials) == 0:
                    obj.data.materials.append(mat)
                else:
                    obj.data.materials[0] = mat

            updated_objects.append(obj)
    return updated_objects

def prepare_scene(objects, config):
    # fix origin for all mesh objects
    for obj in objects:
        if obj.type == 'MESH':
            center_object_origin(obj)

    hide_all_objects(objects, 0)

def setup_environment(config, start_frame, end_frame):
    if config['effects']['fog']:
        add_fog(config)

    if config['background']['use_3d']:

        p0 = config['lighting']['3d_scene_light_intensities']['p0'][0]
        p1 = config['lighting']['3d_scene_light_intensities']['p1'][0]
        p2 = config['lighting']['3d_scene_light_intensities']['p2'][0]
        p3 = config['lighting']['3d_scene_light_intensities']['p3'][0]

        # 3D scene lighting keyframes
        lighting_color = config['lighting']['lighting_color']
        for obj in bpy.context.scene.objects:
            if obj.type == 'LIGHT':
                key = None
                if obj.name == 'Point':
                    key = 0
                    obj.data.energy = p0
                elif obj.name == 'Point.001':
                    key = 1
                    obj.data.energy = p1
                elif obj.name == 'Point.002':
                    key = 2
                    obj.data.energy = p2
                elif obj.name == 'Point.003':
                    key = 3
                    obj.data.energy = p3
                
                if key:
                    obj.data.color = (
                        lighting_color['red'][0],
                        lighting_color['green'][0],
                        lighting_color['blue'][0]
                    )
                    obj.data.keyframe_insert("color", frame=start_frame)
                    obj.data.color = (
                        lighting_color['red'][1],
                        lighting_color['green'][1],
                        lighting_color['blue'][1]
                    )
                    obj.data.keyframe_insert("color", frame=end_frame)
       

def animate_objects(objects, config, start_frame, end_frame):
    dataset_size = len(objects)
    subset_size = config['scene']['num_obj']
    
    # Choose and animate a subset
    subset_indices = np.random.choice(dataset_size, subset_size, replace=False)

    for idx in range(dataset_size):
        obj = objects[idx]

        if idx in subset_indices:
            print(f"start_frame: {start_frame}")
            print(f"end_frame: {end_frame}")

            # Make visible for render
            obj.hide_render = False
            obj.hide_viewport = False
            obj.keyframe_insert(data_path="hide_render", frame=start_frame)
            obj.keyframe_insert(data_path="hide_render", frame=end_frame)
            obj.keyframe_insert(data_path="hide_viewport", frame=start_frame)
            obj.keyframe_insert(data_path="hide_viewport", frame=end_frame)

            # Initial position
            rand_init(config, obj)
            obj.keyframe_insert(data_path="location", frame=start_frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=start_frame)

            # Move/displace to new position
            rand_displace(config, obj)
            obj.keyframe_insert(data_path="location", frame=end_frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=end_frame)
        else:
            obj.hide_render = True
            obj.hide_viewport = True
            obj.keyframe_insert(data_path="hide_render", frame=start_frame)
            obj.keyframe_insert(data_path="hide_render", frame=end_frame)
            obj.keyframe_insert(data_path="hide_viewport", frame=start_frame)
            obj.keyframe_insert(data_path="hide_viewport", frame=end_frame)

def coin_flip():
    return  ( 1 if np.random.uniform() < 0.5 else -1)

def animate_camera(config, start_frame, end_frame):
    # Get the camera object
    camera = bpy.context.scene.camera

    # Insert initial camera position and rotation at frame 1
    camera.keyframe_insert(data_path="location", frame=start_frame)
    camera.keyframe_insert(data_path="rotation_euler", frame=start_frame)

    # Move camera slightly to the left (in local or world X-axis)
    cam_loc = camera.location
    camera.location = (
        cam_loc.x + (coin_flip() * config['camera']['translation'][0]), 
        cam_loc.y + (coin_flip() * config['camera']['translation'][1]), 
        cam_loc.z + (coin_flip() * config['camera']['translation'][2])
    )
    cam_rot = camera.rotation_euler
    camera.rotation_euler = (
        cam_rot.x + (coin_flip() * config['camera']['rotation_offset'][0]), 
        cam_rot.y + (coin_flip() * config['camera']['rotation_offset'][1]), 
        cam_rot.z + (coin_flip() * config['camera']['rotation_offset'][2])
    )

    # Insert new camera position at frame 2
    camera.keyframe_insert(data_path="location", frame=end_frame)
    camera.keyframe_insert(data_path="rotation_euler", frame=end_frame)

    print("Active scene camera:", bpy.context.scene.camera.name)

    for obj in bpy.context.scene.objects:
        if obj.type == 'CAMERA':
            print("Available camera:", obj.name)

    reset_camera(config)

def create_scene(config):
    objects = load_objects(config)
    objects = assign_materials(objects, config)
    prepare_scene(objects, config)

    for i in range(config['scene']['num_scenes']):
        start_frame = i * 2
        end_frame = start_frame + 1

        setup_environment(config, start_frame, end_frame)
        animate_objects(objects, config, start_frame, end_frame)
        animate_camera(config, start_frame, end_frame)

def create_dataset(config):
    if config['scene']['seed'] >= 0:
        np.random.seed(config['scene']['seed'])
    else:
        seed = int(time.time())  
        np.random.seed(seed)

   # Then in create_dataset()
    if config['background']['use_3d']:
        bpy.ops.wm.open_mainfile(filepath=config['background']['3d_path'])

        kill_all_lights()

        p0 = config['lighting']['3d_scene_light_intensities']['p0'][0]
        p1 = config['lighting']['3d_scene_light_intensities']['p1'][0]
        p2 = config['lighting']['3d_scene_light_intensities']['p2'][0]
        p3 = config['lighting']['3d_scene_light_intensities']['p3'][0]          

        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                mat = obj.active_material
                if mat and mat.use_nodes:
                    for node in mat.node_tree.nodes:
                        if node.type == 'EMISSION':
                            node.inputs['Strength'].default_value = (p0 + p1 + p2 + p3) / 80

        reset_camera(config)
        create_scene(config)
        render_dataset(config)
        exr2flow(config)
        
    else:
        delete_all()
        set_camera(config)
        create_scene(config)
        render_dataset(config)
        exr2flow(config)
    
    shutil.rmtree(config['render']['tmp_dump_path'])