import os 
from PIL import Image
from utils.stats import global_stats  # Zorg dat deze import klopt

def calculate_fluctuation(pixel_x, pixel_y, new_x, new_y, config, scene_idx):
    base_path = config['render']['output_folder']
    scene_folder = os.path.join(base_path, f'training/clean/scene_{scene_idx}')
    
    img_0 = Image.open(os.path.join(scene_folder, "frame_0.png")).convert('RGB')
    img_1 = Image.open(os.path.join(scene_folder, "frame_1.png")).convert('RGB')

    rgb_0 = img_0.getpixel((int(pixel_x), int(pixel_y)))
    rgb_1 = img_1.getpixel((int(new_x), int(new_y)))

    r_diff = abs(rgb_0[0] - rgb_1[0])
    g_diff = abs(rgb_0[1] - rgb_1[1])
    b_diff = abs(rgb_0[2] - rgb_1[2])

    # Update stats
    global_stats.update("color_diff_r", r_diff)
    global_stats.update("color_diff_g", g_diff)
    global_stats.update("color_diff_b", b_diff)
    global_stats.update("tot_px", 1)

    return r_diff, g_diff, b_diff
