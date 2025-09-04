import array
import OpenEXR
import Imath
import numpy as np
import cv2
import numpy as np
import struct
import os
from flow.calc_occlusion import calculate_occlusion
from flow.calc_displacement import calculate_displacement
import sys
from utils.folder import get_flow_path
from numpngw import write_png

import matplotlib.pyplot as plt


def load_exr_channels(filepath, channels):
    file = OpenEXR.InputFile(filepath)
    header = file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    channel_data = {}
    for chan in channels:
        raw = file.channel(chan, pt)
        channel_data[chan] = np.frombuffer(raw, dtype=np.float32).reshape((height, width))

    return channel_data, width, height

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

def add_color_wheel_to_image(image, wheel, margin=10):
    h_img, w_img = image.shape[:2]
    h_wheel, w_wheel = wheel.shape[:2]

    x_offset = w_img - w_wheel - margin
    y_offset = h_img - h_wheel - margin

    overlay = image.copy()
    mask = cv2.cvtColor(wheel, cv2.COLOR_BGR2GRAY) != 0

    for c in range(3):
        overlay[y_offset:y_offset + h_wheel, x_offset:x_offset + w_wheel, c][mask] = \
            wheel[..., c][mask]

    return overlay

def exr2flow(config):
    w, h = config['render']['resolution']['x'], config['render']['resolution']['y']
    exr_path = config['render']['tmp_dump_path']

    if config['background']['use_3d']:
        layer_x = "RenderLayer.Vector.Z"
        layer_y = "RenderLayer.Vector.W"
        layer_id = "RenderLayer.IndexOB.X"
    else:
        layer_x = "ViewLayer.Vector.Z"
        layer_y = "ViewLayer.Vector.W"
        layer_id = "ViewLayer.IndexOB.X"

    for idx_scene in range(config['scene']['num_scenes']):
        flow_path = get_flow_path(idx_scene, config)
        tmp_path = os.path.join(exr_path, f"scene_{idx_scene}.exr")

        channels = [layer_x, layer_y, layer_id]
        data, _, _ = load_exr_channels(tmp_path, channels)

        x = data[layer_x]
        y = data[layer_y]
        obj_ids = data[layer_id].astype(np.int32)

        img = np.zeros((h, w, 2), np.float32)
        img[:, :, 0] = -x
        img[:, :, 1] = y

        if config['stats']['calc_occlusion']:
            calculate_occlusion(h, w, img, config, idx_scene, obj_ids)

        if config['stats']['calc_displacement']:
            calculate_displacement(h, w, img)

        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(img[..., 0], img[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        wheel = create_color_wheel(size=100)
        rgb_with_wheel = add_color_wheel_to_image(rgb, wheel)
        output_path = f"img.png"
        cv2.imwrite(output_path, rgb_with_wheel)
        print(f"Image with color wheel saved to {output_path}")

        if config['render']['format'] == 'sintel':
            writeSintelFlow(flow_path, config, x, y)
        else:
            writeKITTIFlow(flow_path, config, x, y)

def writeKITTIFlow(filename, config, u_data, v_data):
    u_data = -1.0 * u_data
    Scaled_Flow = np.stack((u_data, v_data), axis=-1)

    # first channel for height, second for width, third for validity
    sf16 = (64*Scaled_Flow + 2**15).astype(np.uint16)

    # all ones so the entire flow is valid 
    imgdata = np.concatenate((sf16, np.ones(sf16.shape[:2] + (1,), dtype=sf16.dtype)), axis=2)

    write_png(filename, imgdata)

def writeSintelFlow(filename, config, u_data, v_data):
    """
    Create a binary flow file with the specified format.

    Args:
        filename (str): Output file name.
        width (int): Width of the flow data.
        height (int): Height of the flow data.
        u_data (np.ndarray): 2D array of horizontal flow components (HxW).
        v_data (np.ndarray): 2D array of vertical flow components (HxW).
    """

    height = config['render']['resolution']['y']
    width = config['render']['resolution']['x']

    # Check input validity
    if u_data.shape != (height, width) or v_data.shape != (height, width):
        raise ValueError("u_data and v_data must have the shape (height, width).")
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)

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
                f.write(struct.pack('<f', -1.0 * u_data[y, x]))  # Write u component
                f.write(struct.pack('<f', v_data[y, x]))  # Write v component

import png
import numpy as np
import matplotlib.colors as cl
import matplotlib.pyplot as plt
from PIL import Image


UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


def show_flow(filename):
    """
    visualize optical flow map using matplotlib
    :param filename: optical flow file
    :return: None
    """
    flow = read_flow(filename)
    img = flow_to_image(flow)
    plt.imshow(img)
    plt.show()

def read_flow(filename):
    """
    Read optical flow from Middlebury .flo file
    :param filename: name of the flow file
    :return: optical flow data in matrix
    """
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)[0]
        
        if magic != 202021.25:
            raise ValueError("Invalid .flo file: Magic number incorrect")
        
        w = int(np.fromfile(f, np.int32, count=1)[0])  # ✅ Convert to Python int
        h = int(np.fromfile(f, np.int32, count=1)[0])  # ✅ Convert to Python int
        
        data2d = np.fromfile(f, np.float32, count=2 * w * h)  # ✅ Now works correctly
        
        # Reshape to (H, W, 2)
        return np.resize(data2d, (h, w, 2))



def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # print(np.max(u))
    # print(np.min(u))
    # print(np.max(v))
    # print(np.min(v))

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


# show_flow('/Users/Petter/Documents/uni/thesis/CLEAN-MODELS/datasets/mean_80/training/flow/scene_7/flow.flo')

# channels = file.header()['channels'].keys()

    # print("Available channels:")
    # for channel in channels:
    #     print(channel)