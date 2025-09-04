import os
import shutil 
import sys

def create_output_folders(config, path):
    # Delete currently existing path
    if os.path.isdir(path):
        shutil.rmtree(path)

    match config['render']['format']:
        case 'sintel':
            create_sintel_folder(path)
        case 'kitti':
            create_kitti_folder(path)
        case _:
            print("This format does not exist, choose either 'sintel' or 'kitti'")
            return False
    return True

def get_img_path(idx, config):
    base_path = config['render']['output_folder']

    if config['render']['format'] == 'sintel':
        img_path = os.path.join(base_path, 'training/clean/scene_' + str(idx), 'frame_')
    else:
        zeroes = str(f"{idx:06d}") + '_1'
        img_path = os.path.join(base_path, 'training/image_2', zeroes)
    
    return img_path

def get_flow_path(idx, config):
    base_path = config['render']['output_folder']

    if config['render']['format'] == 'sintel':
        flow_path = os.path.join(base_path, 'training/flow/scene_' + str(idx), 'flow.flo')
    else:
        flow_path = os.path.join(base_path, 'training/flow_occ', str(f"{idx:06d}") + '_10.png')
    
    return flow_path


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

def move_folder(config, temp_path: str, result_path: str):
    create_output_folders(config, config['render']['final_folder'])

    print("Moving: ", temp_path)
    print("To: ", result_path)
    try:
        if os.path.isdir(temp_path):
            shutil.copytree(temp_path, result_path, dirs_exist_ok=True)  # Python 3.8+
        else:
            shutil.copy(temp_path, result_path)

        print("Moved to:", result_path)

    except PermissionError as e:
        print(f"PermissionError: {e}")
    except Exception as e:
        print(f"Unhandled error: {e}")