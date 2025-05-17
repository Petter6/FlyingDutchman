import os
import shutil
import json
import os
from flow.calc_flow import read_flow
import numpy as np
from PIL import Image
import copy

from scene.scene_generator import create_dataset
from utils.parser import parse_args
from utils.stats import global_stats
from learn_config.optimizer import run_optimization


# ----------------------------- #
#       UTILITY FUNCTIONS      #
# ----------------------------- #

def create_output_folders(path: str):
    """Delete existing path and create fresh dataset folders."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(os.path.join(path, 'test'), exist_ok=True)
    os.makedirs(os.path.join(path, 'training', 'clean'), exist_ok=True)
    os.makedirs(os.path.join(path, 'training', 'flow'), exist_ok=True)


def get_next_config_path(base_name="train", folder="./config/setting"):
    """Create a uniquely named empty JSON config file."""
    os.makedirs(folder, exist_ok=True)
    i = 1
    while True:
        file_path = os.path.join(folder, f"{base_name}_{i}.json")
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                pass
            print(f"Created empty config file: {file_path}")
            return file_path
        i += 1


def load_json_config(path: str) -> dict:
    """Load JSON config from given path."""
    with open(path, 'r') as f:
        return json.load(f)
    
def check_flow_max(flow, flow_ax, target, tol=1e-3):
    comp = 1 if flow_ax == 'y' else 0  # y = vertical flow = index 1

    # Extract the component of the flow field
    selected_flow = flow[:, :, comp]  # Shape: (H, W)

    max_val = np.max(selected_flow)
    
    return abs(max_val - target) <= tol

def check_flow_min(flow, flow_ax, target, tol=1e-3):
    comp = 1 if flow_ax == 'y' else 0  # y = vertical flow = index 1

    # Extract the component of the flow field
    selected_flow = flow[:, :, comp]  # Shape: (H, W)

    min_val = np.min(selected_flow)
    
    return abs(min_val - target) <= tol
    
    
def check_flow_range(flow, flow_ax, lowest, highest):
    # Select only the component (u or v) you care about: u is x (index 0), v is y (index 1)
    comp = 1 if flow_ax == 'y' else 0  # y = vertical flow = index 1

    # Extract the component of the flow field
    selected_flow = flow[:, :, comp]  # Shape: (H, W)

    max_val = np.max(selected_flow)
    min_val = np.min(selected_flow)

    if max_val > highest or min_val < lowest:
        return False
    return True


    
def check_res(flow, config):
    if not flow.shape[0] == config['render']['resolution']['y']:
        return False
    
    if not flow.shape[1] == config['render']['resolution']['x']:
        return False
    
    if not flow.shape[2] == 2:
        return False

    return True

def is_inverse_of_zero_reference(img_path, idx, tol=1e-4):
    """
    Compares whether the given image is the inverse of the reference image
    at config_zero/training/clean/scene_0/frame_{idx}.png

    :param img_path: Path to the image to test
    :param idx: Frame index to match against in config_zero
    :param tol: Tolerance for pixel-wise inverse check
    :return: True if inverted, False otherwise
    """
    ref_path = os.path.join('config_zero', 'training', 'clean', 'scene_0', f'frame_{idx}.png')
    path = os.path.join(img_path, f'frame_{idx}.png')
    # Load images and normalize
    img_test = np.asarray(Image.open(path).convert('RGB')).astype(np.float32) / 255.0
    img_ref  = np.asarray(Image.open(ref_path).convert('RGB')).astype(np.float32) / 255.0

    if img_test.shape != img_ref.shape:
        raise ValueError("Image shapes do not match")

    # Check if test ≈ 1 - ref
    diff = np.abs(img_test - (1.0 - img_ref))
    max_diff = np.max(diff)

    return max_diff <= tol

def check_luminance_mean(img_dir, idx, target, tol=1e-5):
    path = os.path.join(img_dir, f'frame_{idx}.png')
    img = Image.open(path).convert('RGB')
    rgb = np.asarray(img).astype(np.float32) / 255.0
    brightness = np.mean(rgb, axis=2)

    brightness = np.mean(brightness)

    return abs(brightness - target) <= tol

def check_luminance(img_dir, idx, target, tol=1e-5):
    path = os.path.join(img_dir, f'frame_{idx}.png')
    img = Image.open(path).convert('RGB')
    rgb = np.asarray(img).astype(np.float32) / 255.0
    brightness = np.mean(rgb, axis=2)

    return abs(np.std(brightness) - target) <= tol

def check_3d_lights(flo_path: str, img_path: str, config) -> bool:
    flow = read_flow(flo_path)

    if not check_res(flow, config):
        print('Wrong dimensions of the flo-file')
        return False
    
    if not check_flow_max(flow, 'x', 0): 
        print(f'The max x-flow is not correct')
        return False
    
    if not check_flow_min(flow, 'x', 0): 
        print(f'The min x-flow is not correct')
        return False
    
    if not check_flow_max(flow, 'y', 0): 
        print(f'The max y-flow is not correct')
        return False
    
    if not check_flow_min(flow, 'y', 0): 
        print(f'The min y-flow is not correct')
        return False
    
    if not check_flow_range(flow, 'y', -39, 60): 
        print(f'The y-flow range is not correct')
        return False
    
    if not check_luminance_mean(img_path, 0, 0.22883534):
        print(f'The avg. brightness of image {0} is not correct')
        return False
    
    if not check_luminance_mean(img_path, 1, 0.16531107):
        print(f'The avg. brightness of image {1} is not correct')
        return False
    
    return True

def check_3d(flo_path: str, img_path: str, config) -> bool:
    flow = read_flow(flo_path)

    if not check_res(flow, config):
        print('Wrong dimensions of the flo-file')
        return False
    
    if not check_flow_range(flow, 'x', 100, 250): 
        print(f'The x-flow range is not correct')
        return False
    
    if not check_flow_range(flow, 'y', -39, 60): 
        print(f'The y-flow range is not correct')
        return False
    
    if not check_luminance(img_path, 0, 0.20670196):
        print(f'The avg. brightness of image {0} is not correct')
        return False
    
    if not check_luminance(img_path, 1, 0.22147237):
        print(f'The avg. brightness of image {1} is not correct')
        return False
    
    return True
    
def check_inverted(flo_path: str, img_path: str, config) -> bool:
    flow = read_flow(flo_path)

    if not check_res(flow, config):
        print('Wrong dimensions of the flo-file')
        return False
    
    if not check_flow_min(flow, 'x', 0):
        print('The min-x-flow is not correct')
        return False
    
    if not check_flow_max(flow, 'x', 0):
        print('The max-x-flow is not correct')
        return False

    if not check_flow_min(flow, 'y', 0):
        print('The min-y-flow is not correct')
        return False
    
    if not check_flow_max(flow, 'y', 0):
        print('The max-y-flow is not correct')
        return False
    
    if not check_luminance(img_path, 0, 0.051505182):
        print(f'The avg. brightness of image {0} is not correct')
        return False
    
    if not check_luminance(img_path, 1, 0.051505182):
        print(f'The avg. brightness of image {1} is not correct')
        return False
    
    if not is_inverse_of_zero_reference(img_path, 0):
        print(f'The inverse of image {0} is not correct')
        return False
    
    if not is_inverse_of_zero_reference(img_path, 1):
        print(f'The inverse of image {1} is not correct')
        return False
    
    return True

def check_fog_0(flo_path: str, img_path: str, config) -> bool:
    flow = read_flow(flo_path)

    if not check_res(flow, config):
        print('Wrong dimensions of the flo-file')
        return False
    
    if not check_flow_min(flow, 'x', 0):
        print('The min-x-flow is not correct')
        return False
    
    if not check_flow_max(flow, 'x', 0):
        print('The max-x-flow is not correct')
        return False

    if not check_flow_min(flow, 'y', 0):
        print('The min-y-flow is not correct')
        return False
    
    if not check_flow_max(flow, 'y', 0):
        print('The max-y-flow is not correct')
        return False
    
    if not check_luminance(img_path, 0, 0.04519902):
        print(f'The avg. brightness of image {0} is not correct')
        return False
    
    if not check_luminance(img_path, 1, 0.04519902):
        print(f'The avg. brightness of image {1} is not correct')
        return False
    
    return True

def check_fog(flo_path: str, img_path: str, config) -> bool:
    flow = read_flow(flo_path)

    if not check_res(flow, config):
        print('Wrong dimensions of the flo-file')
        return False
    
    if not check_flow_min(flow, 'x', 0):
        print('The min-x-flow is not correct')
        return False
    
    if not check_flow_max(flow, 'x', 0):
        print('The max-x-flow is not correct')
        return False

    if not check_flow_min(flow, 'y', 0):
        print('The min-y-flow is not correct')
        return False
    
    if not check_flow_max(flow, 'y', 0):
        print('The max-y-flow is not correct')
        return False
    
    if not check_luminance(img_path, 0, 0.018280286):
        print(f'The avg. brightness of image {0} is not correct')
        return False
    
    if not check_luminance(img_path, 1, 0.018280286):
        print(f'The avg. brightness of image {1} is not correct')
        return False
    
    return True

def check_obj_with_rot(flo_path: str, img_path: str, config) -> bool:
    flow = read_flow(flo_path)

    if not check_res(flow, config):
        print('Wrong dimensions of the flo-file')
        return False
    
    if not check_flow_min(flow, 'x', -17.900452):
        print('The min-x-flow is not correct')
        return False
    
    if not check_flow_max(flow, 'x', 19.840363):
        print('The max-x-flow is not correct')
        return False
    
    if not check_luminance(img_path, 0, 0.16708724):
        print(f'The avg. brightness of image {0} is not correct')
        return False
    
    if not check_luminance(img_path, 1, 0.16637781):
        print(f'The avg. brightness of image {1} is not correct')
        return False
    
    return True

def check_obj_with_text(flo_path: str, img_path: str, config) -> bool:
    flow = read_flow(flo_path)

    if not check_res(flow, config):
        print('Wrong dimensions of the flo-file')
        return False
    
    if not check_flow_min(flow, 'x', -138.65114):
        print('The min-x-flow is not correct')
        return False
    
    if not check_flow_max(flow, 'x', 0):
        print('The max-x-flow is not correct')
        return False
    
    if not check_luminance(img_path, 0, 0.16708724):
        print(f'The avg. brightness of image {0} is not correct')
        return False
    
    if not check_luminance(img_path, 1, 0.16727266):
        print(f'The avg. brightness of image {1} is not correct')
        return False
    
    return True

def check_obj(flo_path: str, img_path: str, config) -> bool:
    flow = read_flow(flo_path)

    if not check_res(flow, config):
        print('Wrong dimensions of the flo-file')
        return False
    
    if not check_flow_min(flow, 'x', -157.99559):
        print('The min-x-flow is not correct')
        return False
    
    if not check_flow_max(flow, 'x', 0):
        print('The max-x-flow is not correct')
        return False
    
    if not check_luminance(img_path, 0, 0.14725654):
        print(f'The avg. brightness of image {0} is not correct')
        return False
    
    if not check_luminance(img_path, 1, 0.14738296):
        print(f'The avg. brightness of image {1} is not correct')
        return False
    
    return True

def check_camera_mot(flo_path: str, img_path: str, config) -> bool:
    flow = read_flow(flo_path)

    if not check_res(flow, config):
        print('Wrong dimensions of the flo-file')
        return False

    if not check_flow_range(flow, 'y', -0.0005, 0.0005):  # vertical flow
        print('The y-flow is not correct')
        return False
    
    if not check_flow_range(flow, 'x', -0.0005, 0.0005):  # horizontal flow
        print('The x-flow is not correct')
        return False
    
    if not check_luminance(img_path, 0, 0.14742371):
        print(f'The avg. brightness of image {0} is not correct')
        return False
    
    if not check_luminance(img_path, 1, 0.14742371):
        print(f'The avg. brightness of image {1} is not correct')
        return False
    
    return True

def check_camera_rot(flo_path: str, img_path: str, config) -> bool:
    flow = read_flow(flo_path)

    if not check_res(flow, config):
        print('Wrong dimensions of the flo-file')
        return False

    if not check_flow_range(flow, 'y', -0.0005, 0.0005):  # vertical flow
        print('The y-flow is not correct')
        return False
    
    if not check_flow_range(flow, 'x', -0.0005, 0.0005):  # horizontal flow
        print('The x-flow is not correct')
        return False
    
    if not check_luminance(img_path, 0, 0.051505186):
        print(f'The avg. brightness of image {0} is not correct')
        return False
    
    if not check_luminance(img_path, 1, 0.048577204):
        print(f'The avg. brightness of image {1} is not correct')
        return False
    
    return True
    
def check_zero(flo_path: str, img_path: str, config) -> bool:
    flow = read_flow(flo_path)

    if not check_res(flow, config):
        print('Wrong dimensions of the flo-file')
        return False

    if not check_flow_range(flow, 'y', -0.0005, 0.0005):  # vertical flow
        print('The y-flow is not correct')
        return False
    
    if not check_flow_range(flow, 'x', -0.0005, 0.0005):  # horizontal flow
        print('The x-flow is not correct')
        return False
    
    if not check_luminance(img_path, 0, 0.051505186):
        print(f'The avg. brightness of image {0} is not correct')
        return False
    
    if not check_luminance(img_path, 1, 0.051505186):
        print(f'The avg. brightness of image {1} is not correct')
        return False
    
    return True
  


# ----------------------------- #
#         MAIN ROUTER          #
# ----------------------------- #

def start(config, mode):
    if mode == 'learn':
        run_optimization(config)  # or however your config is structured
    elif mode == 'create':
        create_output_folders(config['render']['output_folder'])
        create_dataset(config=config)

    if config['stats']['print']:
        global_stats.report()

def status_line(label, passed):
    check = "✅" if passed else "❌"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    return f"{label:<40} {color}{check} {str(passed):<5}{reset}"


def test():
    # Define paths
    zero_config_path = './config/create/zero.json'
    output_flo_path = 'training/flow/scene_0/flow.flo'
    output_img_dir = 'training/clean/scene_0/'

    # Load base config where all objects are stationary
    config_zero = load_json_config(zero_config_path)
   
    # Create a modified config with camera rotation applied
    config_with_camera_rotation = copy.deepcopy(config_zero)
    config_with_camera_rotation['camera']['rotation_offset'] = [20, 50, 50]
    config_with_camera_rotation['render']['output_folder'] = './config_camera_rot'
   
    # Create a config with camera movement, this shouldnt change anything in 2D
    config_with_camera_movement = copy.deepcopy(config_zero)
    config_with_camera_movement['camera']['rotation_start'] = [1.57, 0, 0]
    config_with_camera_movement['camera']['translation'] = [50, 0, 0]
    config_with_camera_movement['render']['output_folder'] = './config_camera_mot'

    # Create a config with object
    config_with_object = copy.deepcopy(config_zero)
    config_with_object['camera']['rotation_start'] = [1.57, 0, 0]
    config_with_object['render']['output_folder'] = './config_obj'
    config_with_object['scene']['num_obj'] = 1
    config_with_object['motion']['mean_translation']['x'] = 0.5

    # Create a config with object with text
    config_with_object_with_text = copy.deepcopy(config_zero)
    config_with_object_with_text['camera']['rotation_start'] = [1.57, 0, 0]
    config_with_object_with_text['render']['output_folder'] = './config_obj_with_text'
    config_with_object_with_text['scene']['num_obj'] = 1
    config_with_object_with_text['motion']['mean_translation']['x'] = 0.5
    config_with_object_with_text['objects']['textures_enabled'] = True
    
    # Create a config with object with rot
    config_with_object_with_rot = copy.deepcopy(config_zero)
    config_with_object_with_rot['camera']['rotation_start'] = [1.57, 0, 0]
    config_with_object_with_rot['render']['output_folder'] = './config_obj_with_rot'
    config_with_object_with_rot['scene']['num_obj'] = 1
    config_with_object_with_rot['motion']['mean_rotation'] = 0.5
    config_with_object_with_rot['objects']['textures_enabled'] = True

    # Create a config with object with rot
    config_with_fog = copy.deepcopy(config_zero)
    config_with_fog['camera']['rotation_start'] = [1.57, 0, 0]
    config_with_fog['render']['output_folder'] = './config_fog'
    config_with_fog['effects']['fog'] = True
    config_with_fog['effects']['fog_percentage'] = 1.0
    config_with_fog['scene']['num_obj'] = 5

    # Fog with 0 percent 
    config_with_fog_0 = copy.deepcopy(config_zero)
    config_with_fog_0['render']['output_folder'] = './config_fog_0'
    config_with_fog_0['effects']['fog'] = True
    config_with_fog_0['effects']['fog_percentage'] = 0.0

    # Inverted colors 
    config_inverted = copy.deepcopy(config_zero)
    config_inverted['render']['output_folder'] = './config_inverted'
    config_inverted['effects']['inverted_colors'] = True

    # 3d-bg  
    config_3d = copy.deepcopy(config_zero)
    config_3d['render']['output_folder'] = './config_3d'
    config_3d['background']['use_3d'] = True
    config_3d['camera']['rotation_offset'] = [0.0, 0.0, 0.2]
    config_3d['camera']['rotation_start'] = [1.469, 0.0129, -9.625]
    config_3d['camera']['position_start'] = [0.4609, 7.083, 3.2841]

    # 3d-bg with lights
    config_3d_lights = copy.deepcopy(config_zero)
    config_3d_lights['render']['output_folder'] = './config_3d_lights'
    config_3d_lights['background']['use_3d'] = True
    config_3d_lights['camera']['rotation_start'] = [1.469, 0.0129, -9.625]
    config_3d_lights['camera']['position_start'] = [0.4609, 7.083, 3.2841]
    config_3d_lights['lighting']['lighting_color']['red'] = [1.0, 0.0]
    config_3d_lights['lighting']['lighting_color']['green'] = [0.0, 1.0]
    config_3d_lights['lighting']['3d_scene_light_intensities']['p0'] = [500, 250]
    config_3d_lights['lighting']['3d_scene_light_intensities']['p1'] = [500, 250]
    config_3d_lights['lighting']['3d_scene_light_intensities']['p2'] = [500, 250]
    config_3d_lights['lighting']['3d_scene_light_intensities']['p3'] = [500, 250]


    start(config_zero, 'create')
    start(config_with_camera_rotation, 'create')
    start(config_with_camera_movement, 'create')
    start(config_with_object, 'create')
    start(config_with_object_with_text, 'create')
    start(config_with_object_with_rot, 'create')
    start(config_with_fog, 'create')
    start(config_with_fog_0, 'create')
    start(config_inverted, 'create')
    start(config_3d, 'create')
    start(config_3d_lights, 'create')
   
    # Run checks
    zero_passed = check_zero(
        os.path.join('./config_zero', output_flo_path),
        os.path.join('./config_zero', output_img_dir),
        config_zero
    )

    camera_rot_passed = check_camera_rot(
        os.path.join('./config_camera_rot', output_flo_path),
        os.path.join('./config_camera_rot', output_img_dir),
        config_with_camera_rotation
    )

    camera_mot_passed = check_camera_mot(
        os.path.join('./config_camera_mot', output_flo_path),
        os.path.join('./config_camera_mot', output_img_dir),
        config_with_camera_movement
    )

    obj_passed = check_obj(
        os.path.join('./config_obj', output_flo_path),
        os.path.join('./config_obj', output_img_dir),
        config_with_object
    )

    obj_with_text_passed = check_obj_with_text(
        os.path.join('./config_obj_with_text', output_flo_path),
        os.path.join('./config_obj_with_text', output_img_dir),
        config_with_object_with_text
    )

    obj_with_text_rot = check_obj_with_rot(
        os.path.join('./config_obj_with_rot', output_flo_path),
        os.path.join('./config_obj_with_rot', output_img_dir),
        config_with_object_with_rot
    )

    fog_passed = check_fog(
        os.path.join('./config_fog', output_flo_path),
        os.path.join('./config_fog', output_img_dir),
        config_with_fog
    )
    
    fog_0_passed = check_fog_0(
        os.path.join('./config_fog_0', output_flo_path),
        os.path.join('./config_fog_0', output_img_dir),
        config_with_fog_0
    )

    inverted_passed = check_inverted(
        os.path.join('./config_inverted', output_flo_path),
        os.path.join('./config_inverted', output_img_dir),
        config_inverted
    )

    threed_passed = check_3d(
        os.path.join('./config_3d', output_flo_path),
        os.path.join('./config_3d', output_img_dir),
        config_3d
    )

    threed_lights_passed = check_3d_lights(
        os.path.join('./config_3d_lights', output_flo_path),
        os.path.join('./config_3d_lights', output_img_dir),
        config_3d_lights
    )

    print("\n" + "=" * 50)
    print("\033[1m\033[94m📊 Test Results Summary\033[0m")
    print("=" * 50)
    print(status_line("Zero-motion test", zero_passed))
    print(status_line("Camera rotation test", camera_rot_passed))
    print(status_line("Camera movement test", camera_mot_passed))
    print(status_line("Object (no-text) test", obj_passed))
    print(status_line("Object (with-text) test", obj_with_text_passed))
    print(status_line("Object (with-rotation) test", obj_with_text_rot))
    print(status_line("Fog test", fog_passed))
    print(status_line("Fog 0% test", fog_0_passed))
    print(status_line("Inverted test", inverted_passed))
    print(status_line("3D background test", threed_passed))
    print(status_line("3D bg (alt lighting) test", threed_lights_passed))
    print("=" * 50)
        

if __name__ == '__main__':
    test()

