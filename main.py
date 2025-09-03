import os
import shutil
import argparse
import json
import sys
import os

from scene.scene_generator import create_dataset
from utils.parser import load_json_config
from utils.stats import global_stats
from learn_config.optimizer import run_optimization
from batch import concatenate

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

def move_folder(config, temp_path: str, result_path: str):
    create_output_folders(config['render']['final_folder'])

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


# ----------------------------- #
#         MAIN ROUTER          #
# ----------------------------- #

def start():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['learn', 'create'], required=True, help='Mode: "learn" to optimize config, "create" to generate dataset')
    parser.add_argument('--config', required=True, type=str, help='Path to configuration file')
    parser.add_argument('--batch', required=False, type=int, help='Combine multiple seeds into 1 dataset')
    args = parser.parse_args()

    if args.batch:    
        concatenate(args.config, args.batch)
        global_stats.report()
        return
    
    config = load_json_config(args.config)

   
    if args.mode == 'learn':
        run_optimization(config)  # or however your config is structured
        global_stats.report()
    elif args.mode == 'create':
        create_output_folders(config['render']['output_folder'])
        create_dataset(config=config)
        move_folder(config, config['render']['output_folder'], config['render']['final_folder'])

    if config['stats']['print']:
        global_stats.report()


if __name__ == '__main__':
    start()