import numpy as np
import bpy
import config
import matplotlib.pyplot as plt
import mathutils
from utils.stats import global_stats
import time
import sys
from flow.calc_fluctuation import calculate_fluctuation

# def debug_object_visibility(ref_obj_name="TEST_LOCALIZED"):
#     ref_obj = bpy.data.objects.get(ref_obj_name)
#     if not ref_obj:
#         print(f"Object '{ref_obj_name}' niet gevonden.")
#         return

#     print(f"\n🔍 Referentieobject: {ref_obj_name}")
#     print(f"  - hide_viewport: {ref_obj.hide_viewport}")
#     print(f"  - hide_render: {ref_obj.hide_render}")
#     print(f"  - type: {ref_obj.type}")
#     print(f"  - data: {ref_obj.data.name}")
#     print(f"  - faces: {len(ref_obj.data.polygons)}")
#     print(f"  - linked: {ref_obj.library is not None}")
#     print(f"  - visible_get(): {ref_obj.visible_get()}")
#     print(f"  - in view_layer: {ref_obj.name in bpy.context.view_layer.objects}")

#     print("\n📋 Andere objecten in scene:")
#     for obj in bpy.context.scene.objects:
#         if obj.name == ref_obj_name:
#             continue
#         print(f"\n🔸 {obj.name}")
#         print(f"  - hide_viewport: {obj.hide_viewport}")
#         print(f"  - hide_render: {obj.hide_render}")
#         print(f"  - type: {obj.type}")
#         if obj.type == 'MESH' and obj.data:
#             print(f"  - faces: {len(obj.data.polygons)}")
#         print(f"  - linked: {obj.library is not None}")
#         print(f"  - visible_get(): {obj.visible_get()}")
#         print(f"  - in view_layer: {obj.name in bpy.context.view_layer.objects}")
#         print(f"  - users_collection: {[c.name for c in obj.users_collection]}")

# def print_visible_objects(frame_number):
#     scene = bpy.context.scene
#     scene.frame_set(frame_number)

#     print(f"\n🕵️‍♂️ Zichtbare objecten op frame {frame_number}:")
#     for obj in scene.objects:
#         if obj.type == 'MESH' and obj.visible_get():
#             print(f"  ✅ {obj.name}")

def get_ray(pixel_x, pixel_y, frame_number):
    """Accurate ray cast from camera through pixel coordinates (supports correct image orientation)."""
    import mathutils
    scene = bpy.context.scene
    scene.frame_set(frame_number)
    cam = scene.camera
    render = scene.render

    # Fix Y flip: image (0,0) is top-left, but NDC expects bottom-left
    ndc_x = (pixel_x / render.resolution_x) * 2 - 1
    ndc_y = 1 - (pixel_y / render.resolution_y) * 2  # <-- FIXED

    ndc = mathutils.Vector((ndc_x, ndc_y, -1.0, 1.0))

    # Projection inverse
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam_eval = cam.evaluated_get(depsgraph)
    proj_matrix = cam_eval.calc_matrix_camera(
        depsgraph, x=render.resolution_x, y=render.resolution_y, scale_x=1.0, scale_y=1.0
    )
    inv_proj = proj_matrix.inverted()

    # From NDC to camera space
    cam_space = inv_proj @ ndc
    cam_space = cam_space.xyz / cam_space.w

    # World-space ray
    ray_origin = cam.matrix_world.translation
    ray_direction = (cam.matrix_world.to_3x3() @ cam_space).normalized()

    hit, location, normal, face_index, hit_obj, matrix = scene.ray_cast(
        depsgraph, origin=ray_origin, direction=ray_direction
    )

    return hit, hit_obj, face_index

def calculate_occlusion(h, w, img, config, scene_idx, obj_ids):
    vis = np.ones((h, w), dtype=bool)

    count_oob, count_rot, count_obj = 0, 0, 0

    print(f'{time.time()}')

    start = time.time()

    # Iterate over all pixels
    for pixel_y in range(h):
        print(pixel_y)
        for pixel_x in range(w):
            new_x = img[pixel_y, pixel_x][0] + pixel_x
            new_y = img[pixel_y, pixel_x][1] + pixel_y

            # Check bounds
            if new_x < 0 or new_x >= w:
                # print(f"Horizontally out of bounds at ({pixel_x}, {pixel_y})")
                vis[pixel_y, pixel_x] = False
                count_oob += 1
            elif new_y < 0 or new_y >= h:
                # print(f"Vertically out of bounds at ({pixel_x}, {pixel_y})")
                vis[pixel_y, pixel_x] = False
                count_oob += 1
            else:
                # Perform ray casts at original and new locations
                hit_orig, obj_orig, face_orig = get_ray(pixel_x, pixel_y, frame_number=0)

                if not hit_orig:
                    continue

                if hit_orig and (obj_ids[pixel_y, pixel_x] != obj_ids[int(new_y), int(new_x)]):
                    vis[pixel_y, pixel_x] = False
                    count_obj += 1
                    continue
                            
                # print(obj_orig)
                hit_new, obj_new, face_new = get_ray(new_x, new_y, frame_number=1)

                # Check occlusion: If object or face ID changes, mark it as occluded
                if hit_new and (face_orig != face_new):
                    vis[pixel_y, pixel_x] = False
                    count_rot += 1
                else:
                    if config['stats']['calc_fluctuation']:
                        calculate_fluctuation(pixel_x, pixel_y, new_x, new_y, config, scene_idx)
                    continue

            
    # Convert boolean `vis` to an image-friendly format (0 for False, 1 for True)
    vis_image = vis.astype(np.uint8)  # Convert to 0 and 1

    # Show the visibility map
    plt.figure(figsize=(10, 5))
    plt.imshow(vis_image, cmap='gray', interpolation='nearest')
    plt.colorbar(label="Visibility (1=Visible, 0=Occluded)")
    plt.title("Visibility Map")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.show()

    tot_occ = count_rot + count_obj + count_oob
    global_stats.update('occ_px', tot_occ)
    global_stats.update('occ_self', count_rot)
    global_stats.update('occ_obj', count_obj)
    global_stats.update('occ_oob', count_oob)

    end = time.time() - start
    print(time.time())
    print(end)