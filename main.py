import os
import shutil 
from scene import create_dataset
import config
# from human import create_dataset


def create_folder():
    if os.path.isdir(config.folder_path):
        shutil.rmtree(config.folder_path) 
    os.mkdir(config.folder_path)
    os.mkdir(config.folder_path + '/test')
    os.mkdir(config.folder_path + '/training')
    os.mkdir(config.folder_path + '/training/clean')
    os.mkdir(config.folder_path + '/training/flow')

def start():
    create_folder()
    create_dataset()
    

if __name__ == '__main__':
    start()