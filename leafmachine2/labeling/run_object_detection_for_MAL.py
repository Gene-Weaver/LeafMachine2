# Run yolov5 on dir
import os, sys, inspect
from os import walk
import pandas as pd
import shutil
import subprocess

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from machine.general_utils import get_datetime, get_cfg_from_full_path, bcolors, make_file_names_valid
from component_detector.detect import run

# pip install cython matplotlib tqdm scipy ipython ninja yacs opencv-python ffmpeg opencv-contrib-python Pillow scikit-image scikit-learn lmfit imutils pyyaml jupyterlab==3
# pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

def run_MAL(cfg, dir_base):
    do_save_prediction_overlay_images = not cfg['do_save_prediction_overlay_images']
    dir_weights =  os.path.join(dir_base,'runs','train',cfg['detector_type'],cfg['detector_version'],cfg['detector_iteration'],'weights',cfg['detector_weights'])
    dir_project = os.path.join(dir_base,'runs','detect', cfg['name_project'], cfg['name_iteration'])
    
    if cfg['do_process_subdirectories']:
        f = []
        for (dirpath, dirnames, filenames) in walk(cfg['dir_containing_original_images']):
            f.extend(dirnames)
            break
        for subfolder in f:
            dir_source = os.path.abspath(os.path.join(cfg['dir_containing_original_images'],subfolder))
            print(f"{bcolors.BOLD}Running MAL for {dir_source}{bcolors.ENDC}")
            print(f"{bcolors.BOLD}      Project: {dir_project}{bcolors.ENDC}")
            print(f"{bcolors.BOLD}      Run: {subfolder}{bcolors.ENDC}")
            run(weights=dir_weights,
                source=dir_source,
                project=dir_project,
                name=subfolder,
                imgsz=(1280, 1280),
                nosave=do_save_prediction_overlay_images,
                anno_type=cfg['detector_type'],
                conf_thres= cfg['minimum_confidence_threshold'])

    else:
        print(f"{bcolors.BOLD}Running MAL for {cfg['dir_containing_original_images']}{bcolors.ENDC}")
        print(f"{bcolors.BOLD}      Project: {dir_project}{bcolors.ENDC}")
        print(f"{bcolors.BOLD}      Run: {cfg['name_run']}{bcolors.ENDC}")
        run(weights=dir_weights,
            source=cfg['dir_containing_original_images'],
            project=dir_project,
            name=cfg['name_run'],
            imgsz=(1280, 1280),
            nosave=do_save_prediction_overlay_images,
            anno_type=cfg['detector_type'],
            conf_thres= cfg['minimum_confidence_threshold'])

def process_dir_for_MAL():
    # Read configs
    dir_labeling = os.path.dirname(__file__)
    path_cfg = os.path.join(dir_labeling,'config_run_object_detection_for_MAL.yaml')
    cfg = get_cfg_from_full_path(path_cfg)

    # dir_base = os.path.dirname(__file__)
    dir_base = os.path.join(parentdir,'component_detector')
    # make_file_names_valid(cfg['dir_containing_original_images'], cfg)
    run_MAL(cfg, dir_base)

if __name__ == '__main__':
    process_dir_for_MAL()


