from utils.parser import load_json_config
from scene.scene_generator import create_dataset
import os 
import shutil
import os
import shutil

def create_sintel_folder(path: str):
    # Create fresh dataset sintel-format folders
    os.makedirs(os.path.join(path, 'test'), exist_ok=True)
    os.makedirs(os.path.join(path, 'training', 'clean'), exist_ok=True)
    os.makedirs(os.path.join(path, 'training', 'flow'), exist_ok=True)

def create_kitti_folder(path: str):
     # Create fresh dataset kitti-format folders
    os.makedirs(os.path.join(path, 'testing'), exist_ok=True)
    os.makedirs(os.path.join(path, 'training', 'image_2'), exist_ok=True)
    os.makedirs(os.path.join(path, 'training', 'flow_occ'), exist_ok=True)

def create_output_folders(format, path):
    # Delete currently existing path
    if os.path.isdir(path):
        shutil.rmtree(path)

    match format:
        case 'sintel':
            create_sintel_folder(path)
        case 'kitti':
            create_kitti_folder(path)
        case _:
            print("This format does not exist, choose either 'sintel' or 'kitti'")
            return False
    return True

def move_kitti(root, out_root, start_idx, config):
    ROOT = root
    OUT_ROOT = os.path.join(ROOT, out_root, "training")

    CLEAN_OUT = os.path.join(OUT_ROOT, "image_2")
    FLOW_OUT = os.path.join(OUT_ROOT, "flow_occ")

    # Ensure output directories exist
    os.makedirs(CLEAN_OUT, exist_ok=True)
    os.makedirs(FLOW_OUT, exist_ok=True)

    # Loop over all configs
    for i in range(config['scene']['num_scenes']):
        config_folder = f"./tmp"
        clean_src_0 = os.path.join(ROOT, config_folder, 'training/image_2', str(f"{i:06d}") + '_10.png')
        clean_src_1 = os.path.join(ROOT, config_folder, 'training/image_2', str(f"{i:06d}") + '_11.png')
        flow_src = os.path.join(ROOT, config_folder,'training/flow_occ', str(f"{i:06d}") + '_10.png')

        clean_dst_0 = os.path.join(CLEAN_OUT, str(f"{start_idx+i:06d}") + '_10.png')
        clean_dst_1 = os.path.join(CLEAN_OUT, str(f"{start_idx+i:06d}") + '_11.png')
        flow_dst = os.path.join(FLOW_OUT, str(f"{start_idx+i:06d}") + '_10.png')

        # Copy clean images
        if os.path.exists(clean_src_0):
            shutil.copyfile(clean_src_0, clean_dst_0)
        else:
            print(f"⚠️ Missing: {clean_src_0}")

        if os.path.exists(clean_src_1):
            shutil.copyfile(clean_src_1, clean_dst_1)
        else:
            print(f"⚠️ Missing: {clean_src_1}")

        # Copy flow files
        if os.path.exists(flow_src):
            shutil.copyfile(flow_src, flow_dst)
        else:
            print(f"⚠️ Missing: {flow_src}")

    print("✅ All scenes moved to combined_dataset/training in KITTI format.")    

def move_sintel(root, out_root, start_idx, config):
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

def concatenate(config_path, batch_size, format):
    # scene count
    scene_count = 0

    create_output_folders(format, './batch_output')

    for seed in range(1, batch_size):
        # lees config 
        config_name = config_path + '_' + str(seed) + '.json'
        config = load_json_config(config_name)

        config['render']['output_folder'] = './tmp'
        config['render']['format'] = format
       
        if format == 'kitti':
            create_kitti_folder(config['render']['output_folder'])
        else:
            create_sintel_folder(config['render']['output_folder'])

        # creeer dataset 
        create_dataset(config)

        # verplaats scenes terug pas namen aan van scenes 
        if format == 'sintel':
            move_sintel('./', './batch_output', scene_count, config)
        else:
            move_kitti('./', './batch_output', scene_count, config)

        scene_count += int(config['scene']['num_scenes'])
        
        # verwijder de scenes 
        shutil.rmtree(config['render']['output_folder'])
        shutil.rmtree(config['render']['tmp_dump_path'])