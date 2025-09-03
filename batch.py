from utils.parser import load_json_config
from scene.scene_generator import create_dataset
import os 
import shutil

import os
import shutil

def move_output(root, out_root, start_idx, config):
    ROOT = root
    OUT_ROOT = os.path.join(ROOT, out_root, "training")

    CLEAN_OUT = os.path.join(OUT_ROOT, "clean")
    FLOW_OUT = os.path.join(OUT_ROOT, "flow")

    # Ensure output directories exist
    os.makedirs(CLEAN_OUT, exist_ok=True)
    os.makedirs(FLOW_OUT, exist_ok=True)

    # Loop over all configs
    for i in range(config['scene']['num_scenes']):
        config_folder = f"./tmp"
        clean_src = os.path.join(ROOT, config_folder, "training", "clean", f"scene_{i}")
        flow_src = os.path.join(ROOT, config_folder, "training", "flow", f"scene_{i}")

        clean_dst = os.path.join(CLEAN_OUT, f"scene_{start_idx + i}")
        flow_dst = os.path.join(FLOW_OUT, f"scene_{start_idx + i}")

        # Copy clean images
        if os.path.exists(clean_src):
            shutil.copytree(clean_src, clean_dst, dirs_exist_ok=True)
        else:
            print(f"⚠️ Missing: {clean_src}")

        # Copy flow files
        if os.path.exists(flow_src):
            shutil.copytree(flow_src, flow_dst, dirs_exist_ok=True)
        else:
            print(f"⚠️ Missing: {flow_src}")

    print("✅ All scenes moved to combined_dataset/training in Sintel format.")

def create_output_folders(path: str):
    """Delete existing path and create fresh dataset folders."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(os.path.join(path, 'test'), exist_ok=True)
    os.makedirs(os.path.join(path, 'training', 'clean'), exist_ok=True)
    os.makedirs(os.path.join(path, 'training', 'flow'), exist_ok=True)

def concatenate(config_path, batch_size):
    # scene count
    scene_count = 0

    create_output_folders('./batch_output')

    for seed in range(1, batch_size):
        # lees config 
        config_name = config_path + '_' + str(seed) + '.json'
        config = load_json_config(config_name)

        config['render']['output_folder'] = './tmp'
        config['stats']['calc_displacement'] = True
        create_output_folders(config['render']['output_folder'])

        # creeer dataset 
        create_dataset(config)

        # verplaats scenes terug pas namen aan van scenes 
        move_output('./', './batch_output', scene_count, config)

        scene_count += int(config['scene']['num_scenes'])