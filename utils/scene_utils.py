import bpy 
from math import radians
import numpy as np
from coordinate import local_to_world, world_to_local, sample_in_frustum
import sys

scene = bpy.context.scene

# delete everything in the scene 
def delete_all():
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    scene.camera = None

    for obj in list(bpy.data.objects):
        for coll in obj.users_collection:
            coll.objects.unlink(obj)
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.outliner.orphans_purge(do_recursive=True)
    bpy.data.batch_remove(bpy.data.libraries)


def set_camera(config):
    # add a new camera
    bpy.ops.object.camera_add()
    camera = bpy.context.object  

    # set the camera as the active camera
    bpy.context.scene.camera = camera

    reset_camera(config)

def reset_camera(config):
    camera = bpy.context.scene.camera 

    if config['background']['use_3d']:
        camera.rotation_euler = (1.469, 0.0129, -9.625)
        camera.location = (0.4609, 7.083, 3.2841)
    else:
        camera.rotation_euler = (1.57, 0, 0)
        camera.location = (0, 0, 0)


def hide_all_objects(objects, frame=None):
    for obj in objects:
        obj.hide_render = True
        obj.hide_viewport = True
        if frame is not None:
            obj.keyframe_insert(data_path="hide_render", frame=frame)
            obj.keyframe_insert(data_path="hide_viewport", frame=frame)

def center_object_origin(obj):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.select_set(False)


def link_blend_file(blender_path):
    with bpy.data.libraries.load(blender_path, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects

    for obj in data_to.objects:
        if obj is not None:
            # Voeg een underscore toe aan de naam
            obj.name = f"_{obj.name}"
            bpy.context.scene.collection.objects.link(obj)

def coin_flip():
    return  ( 1 if np.random.uniform() < 0.5 else -1)

def rand_displace(config, obj):
    x_loc, y_loc, z_loc = world_to_local(obj.location[0], obj.location[1], obj.location[2])
    x_rot, y_rot, z_rot = obj.rotation_euler

    # Extract mean and sigma from config
    mean_translation = config['motion']["mean_translation"]
    sigma_translation = config['motion']["sigma_translation"]

    print(mean_translation)
       
    trans_x = coin_flip() * np.random.normal(mean_translation["x"], sigma_translation["x"])
    trans_y = coin_flip() * np.random.normal(mean_translation["y"], sigma_translation["y"])
    trans_z = coin_flip() * np.random.normal(mean_translation["z"], sigma_translation["z"])

    # print(trans_x, trans_y, trans_z)
    # sys.exit(1)

    # Apply translations
    x_loc += trans_x
    y_loc += trans_y
    z_loc += trans_z

    # Apply random rotation
    obj.rotation_euler = (
        x_rot + np.random.normal(coin_flip()*config['motion']['mean_rotation'], config['motion']['sigma_rotation']),
        y_rot + np.random.normal(coin_flip()*config['motion']['mean_rotation'], config['motion']['sigma_rotation']),
        z_rot + np.random.normal(coin_flip()*config['motion']['mean_rotation'], config['motion']['sigma_rotation']),
    )
    
    obj.location = local_to_world(x_loc, y_loc, z_loc)

def create_random_material(idx):
    """Creates a material with a random base color."""
    mat = bpy.data.materials.new(name=f"RandomColor_{idx}")
    mat.use_nodes = True  # Enable shader nodes

    # Get the Principled BSDF node
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        # Assign a random color to Base Color
        bsdf.inputs["Base Color"].default_value = (np.random.random(), np.random.random(), np.random.random(), 1.0)

    return mat


def rand_init(config, object):
    # rotate it randomly
    object.rotation_euler = (
        np.random.uniform(0, 2 * 3.14159),  # Random rotation around X (0 to 2π radians)
        np.random.uniform(0, 2 * 3.14159),  # Random rotation around Y (0 to 2π radians)
        np.random.uniform(0, 2 * 3.14159)   # Random rotation around Z (0 to 2π radians)
    )

    # assign a x, y, z
    object.location = sample_in_frustum(config)