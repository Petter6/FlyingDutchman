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
from utils.folder import create_output_folders, move_folder

# ----------------------------- #
#       UTILITY FUNCTIONS      #
# ----------------------------- #


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
    parser.add_argument('--format', required=True, type=str, help='Define the output format used (kitti / sintel)')
    args = parser.parse_args()

    if args.batch:    
        concatenate(args.config, args.batch, args.format)
        global_stats.report()
        return
    
    config = load_json_config(args.config)

   
    if args.mode == 'learn':
        run_optimization(config)  # or however your config is structured
    elif args.mode == 'create':
        ret = create_output_folders(config, config['render']['output_folder'])

        if not ret:
            return
        
        config['render']['format'] = args.format
        create_dataset(config=config)
        move_folder(config, config['render']['output_folder'], config['render']['final_folder'])

        shutil.rmtree(config, config['render']['output_folder'])
        shutil.rmtree(config, config['render']['tmp_dump_path'])
        

    if config['stats']['print']:
        global_stats.report()


if __name__ == '__main__':
    start()