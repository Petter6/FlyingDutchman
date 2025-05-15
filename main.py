import os
import shutil
import argparse
import json
import sys
import os

# Zet het project root path (waar main.py zit) in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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


# ----------------------------- #
#         MAIN ROUTER          #
# ----------------------------- #

def start():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['learn', 'create'], required=True, help='Mode: "learn" to optimize config, "create" to generate dataset')
    parser.add_argument('--config', required=True, type=str, help='Path to configuration file')
    args = parser.parse_args()

    config = load_json_config(args.config)

    if args.mode == 'learn':
        run_optimization(config)  # or however your config is structured
    elif args.mode == 'create':
        create_output_folders(config['render']['output_folder'])
        create_dataset(config=config)

    if config['stats']['print']:
        global_stats.report()


if __name__ == '__main__':
    start()