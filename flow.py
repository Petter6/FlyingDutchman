import array
import OpenEXR
import Imath
import numpy as np
import cv2
import sys
from pathlib import Path
import numpy as np
import struct
from coordinate import get_ray
import bpy
import mathutils

import matplotlib.pyplot as plt

# Create a color wheel
def create_color_wheel(size):
    xx, yy = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    mag = np.sqrt(xx**2 + yy**2)
    ang = np.arctan2(yy, xx)
    ang[ang < 0] += 2 * np.pi # creates range [0, 2pi]

    hsv_wheel = np.zeros((size, size, 3), dtype=np.uint8)
    hsv_wheel[..., 0] = ang * 180 / (2 * np.pi)  # number between 0-180 
    hsv_wheel[..., 1] = 255  # saturation (full saturation)
    hsv_wheel[..., 2] = 255  # value (full brightness)

    color_wheel = cv2.cvtColor(hsv_wheel, cv2.COLOR_HSV2BGR)
    mask = (mag > 1).astype(np.uint8) * 255 # makes it a circle
    color_wheel = cv2.bitwise_and(color_wheel, color_wheel, mask=~mask)
    return color_wheel


def exr2flow(exr, w,h):
    file = OpenEXR.InputFile(exr)
    file2 = OpenEXR.InputFile('/Users/Petter/Documents/uni/thesis/Blender2flow/tmp/img_frame2.exr')
        
    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    channels = file.header()['channels'].keys()

    print("Available channels:")
    for channel in channels:
        print(channel)

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R,G,B, A) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("ViewLayer.Vector.X", "ViewLayer.Vector.Y", "ViewLayer.Vector.W", "ViewLayer.Vector.Z") ]
    # (R,G,B) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("View Layer.Vector.X", "View Layer.Vector.Y", "View Layer.Vector.Z") ]

    img = np.zeros((h,w,3), np.float64)
    img[:,:,0] = -np.array(A).reshape(img.shape[0],-1)
    img[:,:,1] = np.array(B).reshape(img.shape[0],-1)
    img[:,:,2] = np.array(B).reshape(img.shape[0],-1)
    vis = np.ones((h, w), dtype=bool)

    count_hor, count_ver, count_rot, count_obj = 0, 0, 0, 0

    # Iterate over all pixels
    for pixel_y in range(h):
        for pixel_x in range(w):
            new_x = img[pixel_y, pixel_x][0] + pixel_x
            new_y = img[pixel_y, pixel_x][1] + pixel_y

            # Check bounds
            if new_x < 0 or new_x >= w:
                # print(f"Horizontally out of bounds at ({pixel_x}, {pixel_y})")
                vis[pixel_y, pixel_x] = False
                count_hor += 1
            elif new_y < 0 or new_y >= h:
                # print(f"Vertically out of bounds at ({pixel_x}, {pixel_y})")
                vis[pixel_y, pixel_x] = False
                count_ver += 1
            else:
                # Perform ray casts at original and new locations
                hit_orig, obj_orig, face_orig = get_ray(pixel_x, pixel_y, frame_number=0)

                if not hit_orig:
                    continue

                hit_new, obj_new, face_new = get_ray(new_x, new_y, frame_number=1)

                # Check occlusion: If object or face ID changes, mark it as occluded
                if hit_new and (obj_orig != obj_new):
                    vis[pixel_y, pixel_x] = False  # Pixel is occluded
                    count_obj += 1
                elif hit_new and (face_orig != face_new):
                    vis[pixel_y, pixel_x] = False
                    count_rot += 1
                else:
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

    print("Rotation: " + str(count_rot))
    print("Object: " + str(count_obj))
    print("OOB_Hor: " + str(count_hor))
    print("OOB_Ver: " + str(count_ver))


    hsv = np.zeros((h,w,3), np.uint8)
    hsv[...,1] = 255

    mag, ang = cv2.cartToPolar(img[...,0], img[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Create a color wheel of desired size
    color_wheel_size = 100
    color_wheel = create_color_wheel(color_wheel_size)

    # Overlay the color wheel on the original image
    x_offset = rgb.shape[1] - color_wheel_size - 10
    y_offset = 10
    rgb[y_offset:y_offset+color_wheel_size, x_offset:x_offset+color_wheel_size] = color_wheel

    # cv2.imshow("RGB Image", rgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # save the image
    output_path = "img.png"  # Change to your desired path and filename
    cv2.imwrite(output_path, rgb)
    print(f"Image saved to {output_path}")

    return A, B

def calculate_occlusion():
    pass

def writeFLO(filename, width, height, u_data, v_data):
    """
    Create a binary flow file with the specified format.

    Args:
        filename (str): Output file name.
        width (int): Width of the flow data.
        height (int): Height of the flow data.
        u_data (np.ndarray): 2D array of horizontal flow components (HxW).
        v_data (np.ndarray): 2D array of vertical flow components (HxW).
    """
    # Check input validity
    if u_data.shape != (height, width) or v_data.shape != (height, width):
        raise ValueError("u_data and v_data must have the shape (height, width).")

    # Open the file in binary write mode
    with open(filename, 'wb+') as f:
        # Write the header: "PIEH" as float (202021.25 in little-endian)
        f.write(struct.pack('<f', 202021.25))  # '<' for little-endian, 'f' for float

        # Write the width and height as integers (little-endian)
        f.write(struct.pack('<i', width))  # '<i' for little-endian 32-bit integer
        f.write(struct.pack('<i', height))

        # Interleave u and v data row by row and write as floats
        for y in range(height):
            for x in range(width):
                f.write(struct.pack('<f', u_data[y, x]))  # Write u component
                f.write(struct.pack('<f', -1.0 * v_data[y, x]))  # Write v component