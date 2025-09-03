import bpy 
from utils.scene_utils import link_blend_file, hide_all_objects, center_object_origin, rand_init, rand_displace, reset_camera, set_camera, create_random_material
import numpy as np 
from fast_render import displacement_render, luminance_render
import sys

def kill_all_lights():
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0

    for obj in bpy.data.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'SUN':
            obj.data.energy = 0.0

def lum_scene(config):
    objects = load_objects(config)
    objects = assign_materials(objects, config)
    prepare_scene(objects, config)

    p0 = config['lighting']['3d_scene_light_intensities']['p0']
    p1 = config['lighting']['3d_scene_light_intensities']['p1']
    p2 = config['lighting']['3d_scene_light_intensities']['p2']
    p3 = config['lighting']['3d_scene_light_intensities']['p3']
   
    for obj in bpy.context.scene.objects:
        if obj.name == 'Point':
            obj.data.energy = p0
        elif obj.name == 'Point.001':
            obj.data.energy = p1
        elif obj.name == 'Point.002':
            obj.data.energy = p2
        elif obj.name == 'Point.003':
            obj.data.energy = p3
       

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            mat = obj.active_material
            if mat and mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'EMISSION':
                        node.inputs['Strength'].default_value = (p0 + p1 + p2 + p3) / 80

def delete_all():
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.scene.camera = None

    for obj in list(bpy.data.objects):
        for coll in obj.users_collection:
            coll.objects.unlink(obj)
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.outliner.orphans_purge(do_recursive=True)
    bpy.data.batch_remove(bpy.data.libraries)

def animate_objects(objects, config, start_frame, end_frame):
    dataset_size = len(objects)
    subset_size = config['scene']['num_obj']
    
    # Choose and animate a subset
    subset_indices = np.random.choice(dataset_size, subset_size, replace=False)

    for idx in range(dataset_size):
        obj = objects[idx]

        if idx in subset_indices:
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


def disp_scene(config):
    objects = load_objects(config)
    objects = assign_materials(objects, config)
    prepare_scene(objects, config)

    for i in range(config['scene']['num_scenes']):
        start_frame = i * 2
        end_frame = start_frame + 1

        animate_objects(objects, config, start_frame, end_frame)



def create_dataset(config):
    if config['background']['use_3d']:
        bpy.ops.wm.open_mainfile(filepath="/Users/Petter/Downloads/mini-supermarket/MiniMarket/Shop.blend")
        kill_all_lights()
        reset_camera(config)
    else:
        delete_all()
        set_camera(config)
   
    np.random.seed(config['scene']['seed'])
    bpy.context.scene.render.resolution_x = config['render']['resolution']['x']
    bpy.context.scene.render.resolution_y = config['render']['resolution']['y']

    if config['train_mode'] == 'disp': # learn a displacement distribution 
        disp_scene(config)
        res = displacement_render(config)
    elif config['train_mode'] == 'lum': # learn a luminance distribution
        lum_scene(config)
        res = luminance_render(config)
    else: # learn a fluctuation distribution 
        pass


    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()

    return res
   