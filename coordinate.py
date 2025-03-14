import bpy
import numpy as np
import mathutils
import sys


def local_to_world(x, y, z):
     # Get world transformation matrix of camera
    camera = bpy.context.scene.camera

    # Get the camera's transformation matrix
    cam_matrix = camera.matrix_world
    
    # Convert the local point to world space
    world_point = cam_matrix @ mathutils.Vector((x, y, z))

    return world_point


def world_to_local(x, y, z):
    # Get world transformation matrix of camera
    camera = bpy.context.scene.camera
    
    # Get the inverse of the camera's world transformation matrix
    inv_cam_matrix = camera.matrix_world.inverted()
    
    # Convert the world point to local space
    local_point = inv_cam_matrix @ mathutils.Vector((x, y, z))
    
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


# compute ray origin and direction
def get_ray(pixel_x, pixel_y, frame_number):
    """Casts a ray from the camera through the given pixel at a specific frame."""
    curr_scene = bpy.context.scene
    curr_scene.frame_set(frame_number)
    
    cam = bpy.context.scene.camera
    frame = cam.data.view_frame(scene = bpy.context.scene)

    # Its in local space so transform to global
    frame = [cam.matrix_world @ corner for corner in frame]

    #rechtsboven = frame[0]
    rechtsonder = frame[1]
    linksboven = frame[2]
    linksonder = frame[3]

    x = linksonder[0] +  pixel_x * ((rechtsonder[0] - linksonder[0]) / bpy.context.scene.render.resolution_x)
    y = linksonder[1] +  pixel_x * ((rechtsonder[1] - linksonder[1]) / bpy.context.scene.render.resolution_x)
    z = linksonder[2] +  pixel_y * ((linksboven[2] - linksonder[2]) / bpy.context.scene.render.resolution_y)

    # Convert `pos` to a Blender mathutils.Vector
    pos = mathutils.Vector((x, y, z))

    # Compute ray direction
    ray_direction = (pos).normalized()

    # Perform the ray cast
    depsgraph = bpy.context.evaluated_depsgraph_get()
    hit, location, normal, face_index, hit_obj, matrix = curr_scene.ray_cast(depsgraph, (0, 0, 0), ray_direction)

    return hit, hit_obj, face_index