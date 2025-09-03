import json
import argparse

def json_to_args(json_path):
    # return a argparse.Namespace object
    with open(json_path, 'r') as f:
        data = json.load(f)
    args = argparse.Namespace()
    args_dict = args.__dict__
    for key, value in data.items():
        args_dict[key] = value
    return args

def parse_args(parser):
    entry = parser.parse_args()
    json_path = entry.cfg
    args = json_to_args(json_path)
    args_dict = args.__dict__
    for index, (key, value) in enumerate(vars(entry).items()):
        args_dict[key] = value
    return args

def load_json_config(path: str) -> dict:
    """Load JSON config from given path."""
    with open(path, 'r') as f:
        return json.load(f)