import bpy
from math import radians
import os
import random
from flow import exr2flow, writeFLO
import matplotlib.pyplot as plt
import numpy as np
from render import render_two_frames
import config
from mathutils import Vector
from coordinate import sample_in_frustum, local_to_world, world_to_local
import sys

asset_path = '/Users/Petter/Documents/uni/thesis/SynthDet/SynthDet/Assets'
bb_list = []
light_object = None

# delete everything in the scene 
def delete_all():
    # Ensure the context is in OBJECT mode
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    
def set_camera():
    # add a new camera
    bpy.ops.object.camera_add(location=config.camera_pos)
    camera = bpy.context.object  

    # randomly rotate the camera along the z-axis
    camera.rotation_euler = (radians(90), radians(0), radians(np.random.uniform(0, 360)))

    # set the camera as the active camera
    bpy.context.scene.camera = camera

def add_light():
    global light_object  # Use the global variable to store light
    light_data = bpy.data.lights.new(name="Light", type=config.light_type)  
    light_data.energy = config.light_intensity  # Initial intensity

    light_object = bpy.data.objects.new(name="Light_Object", object_data=light_data)
    bpy.context.collection.objects.link(light_object)

    # Set the light's location
    match config.light_orientation:
        case 'left':
            light_object.location = local_to_world(-1, 0, 2)  
        case 'right':
            light_object.location = local_to_world(1, 0, 2) 
        case 'down':
            light_object.location = local_to_world(0, -1, 2) 
        case 'up':
            light_object.location = local_to_world(0, 1, 2) 
        case _:
            light_object.location = local_to_world(0, 0, 0) 


# Utility function to import FBX
def import_fbx(fbx_path):
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    imported_objects = bpy.context.selected_objects
    if not imported_objects:
        raise ValueError("No objects were imported. Check the FBX file path and contents.")
    return imported_objects[0]  # Assume the first imported object is the desired one

def apply_texture(obj, texture_path):
    # Get or create a material
    if not obj.data.materials:
        material = bpy.data.materials.new(name="TextureMaterial")
        obj.data.materials.append(material)
    else:
        material = obj.data.materials[0]

    # Enable nodes
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)

    # Add necessary nodes
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')

    texture_node = nodes.new(type='ShaderNodeTexImage')
    texture_node.image = bpy.data.images.load(filepath=texture_path)

    output_node = nodes.new(type='ShaderNodeOutputMaterial')

    # Connect nodes
    links.new(texture_node.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])

# return the name of a random model 
def random_object():
    # query for a random file in the models folder 
    path = random.choice(os.listdir(os.path.join(asset_path, "Foreground Objects", "Textures")))
    
    # drop the extension of the file ( error prone code needs to be changed )
    return path.split('.')[0]

def rand_displace(obj):
    x_loc, y_loc, z_loc = world_to_local(obj.location[0], obj.location[1], obj.location[2])
    x_rot, y_rot, z_rot = obj.rotation_euler

    if config.gaussian_trans:
        x_loc = np.random.normal(x_loc, config.sigma_trans)
        y_loc = np.random.normal(y_loc, config.sigma_trans)
        z_loc = np.random.normal(z_loc, config.sigma_trans)

        
    else:
        x_loc = x_loc + np.random.uniform(-config.x_trans, config.x_trans)
        y_loc = y_loc + np.random.uniform(-config.y_trans, config.y_trans)
        z_loc = z_loc + np.random.uniform(-config.z_trans, config.z_trans)
    

    obj.rotation_euler = (
            np.random.normal(x_rot, config.sigma_rot),
            np.random.normal(y_rot, config.sigma_rot),
            np.random.normal(z_rot, config.sigma_rot),
        )
    
    obj.location = local_to_world(x_loc, y_loc, z_loc)
  



def rand_init(object):
    # rotate it randomly
    object.rotation_euler = (
        random.uniform(0, 2 * 3.14159),  # Random rotation around X (0 to 2π radians)
        random.uniform(0, 2 * 3.14159),  # Random rotation around Y (0 to 2π radians)
        random.uniform(0, 2 * 3.14159)   # Random rotation around Z (0 to 2π radians)
    )

    # assign a x, y, z
    object.location = sample_in_frustum()

# Main function to create the object
def create_object(name, idx):
    # construct paths
    fbx_path = os.path.join(asset_path, "Foreground Objects", "Models", f"{name}.fbx")
    texture_path = os.path.join(asset_path, "Foreground Objects", "Textures", f"{name}.jpg")

    # import FBX
    imported_object = import_fbx(fbx_path)

    # rename the object 
    imported_object.name = str(idx)
    imported_object.pass_index = idx

    if imported_object and imported_object.type == 'MESH':
        mesh = imported_object.data  # Access mesh data
        
        print(f"Object: {imported_object.name} has {len(mesh.polygons)} faces.")
        
        # Loop through all faces (polygons)
        for i, face in enumerate(mesh.polygons):
            print(f"Face {i}: {face.vertices[:]}")  # Print the vertex indices of each face

    # randomly place, rotate and name object
    rand_init(imported_object)

    # assign material and texture
    if config.textures_enabled:
        apply_texture(imported_object, texture_path)

    print(f"Object '{name}' created from FBX and texture applied.")

def add_background():
    # set the world background to use an HDRI
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    # use nodes for the world shader
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # clear default nodes
    for node in nodes:
        nodes.remove(node)

    # add a Background node and an Environment Texture node
    bg = nodes.new(type="ShaderNodeBackground")
    env_texture = nodes.new(type="ShaderNodeTexEnvironment")
    output = nodes.new(type="ShaderNodeOutputWorld")

    # load an HDRI bg
    if not config.bg_path:
        hdri_path = os.path.join('background', random.choice(os.listdir('./background')))  # random bg
    else:
        hdri_path = config.bg_path # load specific bg

    env_texture.image = bpy.data.images.load(hdri_path)

    # link nodes
    links.new(env_texture.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], output.inputs["Surface"])

    # set the background strength
    bg.inputs["Strength"].default_value = 0.5


def create_scene(scene):
    global light_object
    add_background()
    
    for idx in range(config.num_obj):
        object_name = random_object()
        create_object(object_name, idx)

    for obj in bpy.data.objects:
        if obj.name == 'Light_Object' or obj.name == 'Camera':
            continue
        else:
            obj.keyframe_insert("location", frame=scene*2)
            obj.keyframe_insert("rotation_euler", frame=scene*2)
            rand_displace(obj)
            obj.keyframe_insert("location", frame=(scene*2)+1)
            obj.keyframe_insert("rotation_euler", frame=(scene*2)+1)

    # Change the light intensity in the second frame
    if light_object:
        light_object.data.energy = config.light_intensity_second_frame  
        light_object.data.keyframe_insert("energy", frame=scene * 2)

        light_object.data.energy = config.light_intensity  
        light_object.data.keyframe_insert("energy", frame=(scene * 2) + 1)

def render(scene):
    flow_path = config.folder_path + '/training/flow/scene_' + str(scene)
    img_path = config.folder_path + '/training/clean/scene_' + str(scene)

    os.mkdir(flow_path)
    os.mkdir(img_path)

    render_two_frames(scene * 2, (scene * 2) + 1, scene, img_path)

    x_out, y_out = exr2flow('./tmp/img_frame1.exr', config.x_resolution, config.y_resolution)
    x = np.array(x_out).reshape(config.y_resolution, config.x_resolution)
    y = np.array(y_out).reshape(config.y_resolution, config.x_resolution)
    x_y = np.stack((x, y), axis=2)
    
    file_path = os.path.join(flow_path, 'flow.flo')
    writeFLO(file_path, config.x_resolution, config.y_resolution, x, y)

def create_dataset():
    bpy.context.scene.render.resolution_x = config.x_resolution
    bpy.context.scene.render.resolution_y = config.y_resolution

    for scene in range(config.num_scenes):
        #clear entire scene
        delete_all()
        set_camera()
        add_light()
        create_scene(scene)
        render(scene)