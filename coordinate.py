import bpy
import numpy as np
from mathutils import Vector

def local_to_world(x, y, z):
     # Get world transformation matrix of camera
    camera = bpy.context.scene.camera

    # Get the camera's transformation matrix
    cam_matrix = camera.matrix_world
    
    # Convert the local point to world space
    world_point = cam_matrix @ Vector((x, y, z))

    print(world_point)

    return world_point


def world_to_local(x, y, z):
    # Get world transformation matrix of camera
    camera = bpy.context.scene.camera
    
    # Get the inverse of the camera's world transformation matrix
    inv_cam_matrix = camera.matrix_world.inverted()
    
    # Convert the world point to local space
    local_point = inv_cam_matrix @ Vector((x, y, z))
    
    return local_point


def sample_in_frustum():
    # Get world transformation matrix of camera
    camera = bpy.context.scene.camera

    z = -np.random.uniform(1, 3)

    # Compute frustum bounds
    x_fov = camera.data.angle_x  # horizontal FOV
    aspect_ratio = bpy.context.scene.render.resolution_x / bpy.context.scene.render.resolution_y
    y_fov = 2 * np.arctan(np.tan(x_fov / 2) / aspect_ratio) # vertical FOV 

    x_bound = np.tan(x_fov / 2) * abs(z)  
    y_bound = np.tan(y_fov / 2) * abs(z)  

    x = np.random.uniform(-x_bound, x_bound)
    y = np.random.uniform(-y_bound, y_bound)    
   
    return local_to_world(x, y, z)