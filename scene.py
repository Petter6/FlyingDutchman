import bpy
from math import radians
import os
import random
from mathutils import Vector
from flow import exr2flow, writeFLO
import matplotlib.pyplot as plt
import numpy as np
import shutil

asset_path = '/Users/Petter/Documents/uni/thesis/SynthDet/SynthDet/Assets'
bb_list = []

# delete everything in the scene 
def delete_all():
    # Ensure the context is in OBJECT mode
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
def set_camera(x_pos, y_pos, z_pos):
    # add a new camera
    bpy.ops.object.camera_add(location=(x_pos, y_pos, z_pos))
    camera = bpy.context.object  # The new camera becomes the active object

    # point the camera at the origin
    camera.rotation_euler = (radians(270), radians(180), radians(180))

    # set the camera as the active camera
    bpy.context.scene.camera = camera

def add_light(intensity, orientation):
    # Create a new light data block
    light_data = bpy.data.lights.new(name="Light", type='POINT')  # 'POINT', 'SUN', 'SPOT', 'AREA'
    light_data.energy = intensity  # Set light energy (intensity)

    # Create a new object with the light data block
    light_object = bpy.data.objects.new(name="Light_Object", object_data=light_data)

    # Link the light object to the scene collection
    bpy.context.collection.objects.link(light_object)

    # Set the light's location
    match orientation:
        case 'left':
            light_object.location = (-2, -3, 0)  
        case 'right':
            light_object.location = (2, -3, 0) 
        case 'down':
            light_object.location = (0, -3, -2) 
        case 'up':
            light_object.location = (0, -3, 2) 
        case _:
            light_object.location = (0, -3, 0) 

def get_BoundBox(ob):
    """
    Returns the corners of the bounding box of an object in world coordinates.
    """
    # Ensure the object's transformations are up-to-date
    bpy.context.view_layer.update()

    # Transform each local bounding box corner to world coordinates
    bbox_corners = [ob.matrix_world @ Vector(corner) for corner in ob.bound_box]
    return bbox_corners
 
def check_Collision(box1, box2):
    """
    Check Collision of 2 Bounding Boxes
    box1 & box2 muss Liste mit Vector sein,
    welche die Eckpunkte der Bounding Box
    enthält
    #  ________ 
    # |\       |\
    # |_\______|_\
    # \ |      \ |
    #  \|_______\|
    # 
    #
    """
 
    x_max = max([e[0] for e in box1])
    x_min = min([e[0] for e in box1])
    y_max = max([e[1] for e in box1])
    y_min = min([e[1] for e in box1])
    z_max = max([e[2] for e in box1])
    z_min = min([e[2] for e in box1])
     
    x_max2 = max([e[0] for e in box2])
    x_min2 = min([e[0] for e in box2])
    y_max2 = max([e[1] for e in box2])
    y_min2 = min([e[1] for e in box2])
    z_max2 = max([e[2] for e in box2])
    z_min2 = min([e[2] for e in box2])
     
    isColliding = ((x_max >= x_min2 and x_max <= x_max2) \
                    or (x_min <= x_max2 and x_min >= x_min2)) \
                    and ((y_max >= y_min2 and y_max <= y_max2) \
                    or (y_min <= y_max2 and y_min >= y_min2)) \
                    and ((z_max >= z_min2 and z_max <= z_max2) \
                    or (z_min <= z_max2 and z_min >= z_min2))
 
    if isColliding:
        return True
         
    return False


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
    
    # drop the extension of the file 
    return path.split('.')[0]

def rand_displace(obj, sigma_trans, sigma_rot):
    x_loc, y_loc, z_loc = obj.location
    x_rot, y_rot, z_rot = obj.rotation_euler

    obj.location = (
        np.random.normal(x_loc, sigma_trans),
        np.random.normal(y_loc, sigma_trans),
        np.random.normal(z_loc, sigma_trans),
    )

    obj.rotation_euler = (
        np.random.normal(x_rot, sigma_rot),
        np.random.normal(y_rot, sigma_rot),
        np.random.normal(z_rot, sigma_rot),
    )

def rand_init(object):
    # rotate it randomly
    object.rotation_euler = (
        random.uniform(0, 2 * 3.14159),  # Random rotation around X (0 to 2π radians)
        random.uniform(0, 2 * 3.14159),  # Random rotation around Y (0 to 2π radians)
        random.uniform(0, 2 * 3.14159)   # Random rotation around Z (0 to 2π radians)
    )

    # assign a x, y, z
    object.location = (
        random.uniform(-0.7, 0.7),  
        random.uniform(0, 2),  
        random.uniform(-0.5, 0.5)  
    )

# Main function to create the object
def create_object(name):
    # construct paths
    fbx_path = os.path.join(asset_path, "Foreground Objects", "Models", f"{name}.fbx")
    texture_path = os.path.join(asset_path, "Foreground Objects", "Textures", f"{name}.jpg")

    # import FBX
    imported_object = import_fbx(fbx_path)

    # rename the object 
    imported_object.name = name

    free_standing = False

    while not free_standing:
        # randomly place, rotate and name object
        rand_init(imported_object)
         
        prov_bb = get_BoundBox(imported_object)
        free_standing = True

        for bb in bb_list:
            if check_Collision(prov_bb, bb):
                free_standing = False
                break

    bb_list.append(prov_bb)

    # assign material and texture
    apply_texture(imported_object, texture_path)

    print(f"Object '{name}' created from FBX and texture applied.")


def render_two_frames(frame1, frame2):
    try:
        # Set render engine
        bpy.context.scene.render.engine = 'CYCLES'  # Use 'EEVEE' if preferred
        bpy.context.scene.render.resolution_percentage = 100  # Use full resolution

        # Setup for EXR file
        view_layer = bpy.context.scene.view_layers.get("ViewLayer")

        # Enable necessary passes
        view_layer.use_pass_combined = True
        view_layer.use_pass_vector = True
        view_layer.use_pass_z = True

        # Set output filepath
        base_filepath = '/Users/Petter/Documents/uni/thesis/RAFT/datasets/Sintel-kort/training/clean/scene_1'
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'

         # Render the second frame
        bpy.context.scene.frame_set(frame2)  # Set to the second frame
        bpy.context.scene.render.filepath = f"./out/img_frame1.exr"
        bpy.ops.render.render(write_still=True)

        bpy.context.scene.render.image_settings.file_format = 'PNG'

        # Render the first frame
        bpy.context.scene.frame_set(frame1)  # Set to the first frame
        bpy.context.scene.render.filepath = f"{base_filepath}/frame{frame2}.png"
        bpy.ops.render.render(write_still=True)

        # Render the second frame
        bpy.context.scene.frame_set(frame2)  # Set to the second frame
        bpy.context.scene.render.filepath = f"{base_filepath}/frame{frame1}.png"
        bpy.ops.render.render(write_still=True)

        print("Rendering complete.")

    except Exception as e:
        print(f"Error during rendering: {e}")



# clear the entire scene 
delete_all()
set_camera(0, -2, 0)

# watts per square meter per steradia
light_intensity = 500
# left / right / up / down / center
light_orientation = 'up'

# licht kleur
# meerdere lichtbronnen



add_light(light_intensity, light_orientation)

start_frame = 1
end_frame = 2

for i in range(3):
        object_name = random_object()
        create_object(object_name)

for obj in bpy.data.objects:
    if obj.name == 'Light_Object' or obj.name == 'Camera':
        continue
    else:
        obj.keyframe_insert("location", frame=start_frame)
        obj.keyframe_insert("rotation_euler", frame=start_frame)
        rand_displace(obj, 0.02, 0.1)
        obj.keyframe_insert("location", frame=end_frame)
        obj.keyframe_insert("rotation_euler", frame=end_frame)

filepath = "/Users/Petter/Downloads/3438571191.jpg"

# Access the world settings
world = bpy.context.scene.world
world.use_nodes = True
nodes = world.node_tree.nodes
links = world.node_tree.links

# Clear existing nodes
for node in nodes:
    nodes.remove(node)

# Add new nodes
bg_node = nodes.new(type='ShaderNodeBackground')
bg_node.location = (0, 0)

tex_image_node = nodes.new(type='ShaderNodeTexImage')
tex_image_node.location = (-300, 0)

mapping_node = nodes.new(type='ShaderNodeMapping')
mapping_node.location = (-600, 0)

tex_coord_node = nodes.new(type='ShaderNodeTexCoord')
tex_coord_node.location = (-900, 0)

output_node = nodes.new(type='ShaderNodeOutputWorld')
output_node.location = (300, 0)

# Load the image
img = bpy.data.images.load(filepath)
tex_image_node.image = img

# Connect nodes
links.new(tex_coord_node.outputs['Generated'], mapping_node.inputs['Vector'])
links.new(mapping_node.outputs['Vector'], tex_image_node.inputs['Vector'])
links.new(tex_image_node.outputs['Color'], bg_node.inputs['Color'])
links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

# Adjust mapping to control zoom
mapping_node.inputs['Scale'].default_value = (1.0, 1.0, 1.0)

folder_path = '/Users/Petter/Documents/uni/thesis/RAFT/datasets/Sintel-kort'

if os.path.isdir(folder_path):
    shutil.rmtree(folder_path) 

os.mkdir(folder_path)
os.mkdir(folder_path + '/test')
os.mkdir(folder_path + '/training')
os.mkdir(folder_path + '/training/clean')
os.mkdir(folder_path + '/training/flow')
os.mkdir(folder_path + '/training/clean/scene_1')
os.mkdir(folder_path + '/training/flow/scene_1')

# Set the render resolution
y_resolution = 540
x_resolution = 960
bpy.context.scene.render.resolution_x = x_resolution
bpy.context.scene.render.resolution_y = y_resolution

# # Render the frames
render_two_frames(start_frame, end_frame)

# Calculate optical flow with the new resolution
x_out, y_out = exr2flow('./out/img_frame1.exr', x_resolution, y_resolution)

# Reshape the flow data
x = np.array(x_out).reshape(y_resolution, x_resolution)
y = np.array(y_out).reshape(y_resolution, x_resolution)

# Combine the flow components
x_y = np.stack((x, y), axis=2)

# Full file path
file_path = os.path.join(folder_path+'/training/flow/scene_1/', 'flow.flo')

# Create the file
with open(file_path, 'w') as file:
    # Write initial content or leave empty
    file.write('')

# Write the flow to .flo file
writeFLO(file_path, x_resolution, y_resolution, x, y)