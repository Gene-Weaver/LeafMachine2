import os, yaml, datetime, argparse, re, cv2, random, shutil, time
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import concurrent.futures
from time import perf_counter
import glob
import imageio
from PIL import Image
from colour.temperature import xy_to_CCT, CCT_to_xy #pip install colour-science
import subprocess
import platform
from xml.etree.ElementTree import Element, SubElement, tostring, register_namespace
from xml.dom import minidom
import sqlite3
import ast
'''
TIFF --> DNG
Install
https://helpx.adobe.com/camera-raw/using/adobe-dng-converter.html
Read
https://helpx.adobe.com/content/dam/help/en/photoshop/pdf/dng_commandline.pdf

'''


# https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal

def validate_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

def get_cfg_from_full_path(path_cfg):
    with open(path_cfg, "r") as ymlfile:
        cfg = yaml.full_load(ymlfile)
    return cfg

# def load_cfg(pathToCfg):
#     try:
#         with open(os.path.join(pathToCfg,"LeafMachine2.yaml"), "r") as ymlfile:
#             cfg = yaml.full_load(ymlfile)
#     except:
#         with open(os.path.join(os.path.dirname(os.path.dirname(pathToCfg)),"LeafMachine2.yaml"), "r") as ymlfile:
#             cfg = yaml.full_load(ymlfile)
#     return cfg

# def load_cfg_VV(pathToCfg):
#     try:
#         with open(os.path.join(pathToCfg,"VoucherVision.yaml"), "r") as ymlfile:
#             cfg = yaml.full_load(ymlfile)
#     except:
#         with open(os.path.join(os.path.dirname(os.path.dirname(pathToCfg)),"VoucherVision.yaml"), "r") as ymlfile:
#             cfg = yaml.full_load(ymlfile)
#     return cfg

def load_cfg(pathToCfg, system='LeafMachine2'):
    if system not in ['LeafMachine2', 'VoucherVision', 'SpecimenCrop', 'DetectPhenology', 'CensorArchivalComponents']:
        raise ValueError("Invalid system. Expected 'LeafMachine2', 'VoucherVision' or 'SpecimenCrop' or 'DetectPhenology' or 'CensorArchivalComponents'.")

    try:
        with open(os.path.join(pathToCfg, f"{system}.yaml"), "r") as ymlfile:
            cfg = yaml.full_load(ymlfile)
    except:
        with open(os.path.join(os.path.dirname(os.path.dirname(pathToCfg)), f"{system}.yaml"), "r") as ymlfile:
            cfg = yaml.full_load(ymlfile)
    return cfg


def import_csv(full_path):
    csv_data = pd.read_csv(full_path,sep=',',header=0, low_memory=False, dtype=str)
    return csv_data

def import_tsv(full_path):
    csv_data = pd.read_csv(full_path,sep='\t',header=0, low_memory=False, dtype=str)
    return csv_data

def parse_cfg():
    parser = argparse.ArgumentParser(
            description='Parse inputs to read  config file',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    optional_args = parser._action_groups.pop()
    required_args = parser.add_argument_group('MANDATORY arguments')
    required_args.add_argument('--path-to-cfg',
                                type=str,
                                required=True,
                                help='Path to config file - LeafMachine.yaml. Do not include the file name, just the parent dir.')

    parser._action_groups.append(optional_args)
    args = parser.parse_args()
    return args

def check_for_subdirs(cfg):
    original_in = cfg['leafmachine']['project']['dir_images_local']
    dirs_list = []
    run_name = []
    has_subdirs = False
    if os.path.isdir(original_in):
        # list contents of the directory
        contents = os.listdir(original_in)
        
        # check if any of the contents is a directory
        subdirs = [f for f in contents if os.path.isdir(os.path.join(original_in, f))]
        
        if len(subdirs) > 0:
            print("The directory contains subdirectories:")
            for subdir in subdirs:
                has_subdirs = True
                print(os.path.join(original_in, subdir))
                dirs_list.append(os.path.join(original_in, subdir))
                run_name.append(subdir)
        else:
            print("The directory does not contain any subdirectories.")
            dirs_list.append(original_in)
            run_name.append(cfg['leafmachine']['project']['run_name'])

    else:
        print("The specified path is not a directory.")

    return run_name, dirs_list, has_subdirs

def check_for_subdirs_VV(cfg):
    original_in = cfg['leafmachine']['project']['dir_images_local']
    dirs_list = []
    run_name = []
    has_subdirs = False
    if os.path.isdir(original_in):
        dirs_list.append(original_in)
        run_name.append(os.path.basename(os.path.normpath(original_in)))
        # list contents of the directory
        contents = os.listdir(original_in)
        
        # check if any of the contents is a directory
        subdirs = [f for f in contents if os.path.isdir(os.path.join(original_in, f))]
        
        if len(subdirs) > 0:
            print("The directory contains subdirectories:")
            for subdir in subdirs:
                has_subdirs = True
                print(os.path.join(original_in, subdir))
                dirs_list.append(os.path.join(original_in, subdir))
                run_name.append(subdir)
        else:
            print("The directory does not contain any subdirectories.")
            dirs_list.append(original_in)
            run_name.append(cfg['leafmachine']['project']['run_name'])

    else:
        print("The specified path is not a directory.")

    return run_name, dirs_list, has_subdirs

def get_datetime():
    day = "_".join([str(datetime.datetime.now().strftime("%Y")),str(datetime.datetime.now().strftime("%m")),str(datetime.datetime.now().strftime("%d"))])
    time = "-".join([str(datetime.datetime.now().strftime("%H")),str(datetime.datetime.now().strftime("%M")),str(datetime.datetime.now().strftime("%S"))])
    new_time = "__".join([day,time])
    return new_time

def save_config_file(cfg, logger, Dirs):
    logger.info("Save config file")
    name_yaml = ''.join([Dirs.run_name,'.yaml'])
    write_yaml(cfg, os.path.join(Dirs.path_config_file, name_yaml))

def write_yaml(cfg, path_cfg):
    with open(path_cfg, 'w') as file:
        yaml.dump(cfg, file)

def split_into_batches(Project, logger, cfg):
    logger.name = 'Creating Batches'
    n_batches, n_images = Project.process_in_batches(cfg)
    m = f'Created {n_batches} Batches to Process {n_images} Images'
    logger.info(m)
    return Project, n_batches, m 

def split_into_batches_spoof(Project, logger, cfg):
    logger.name = 'Creating spoof Batches'
    n_batches, n_images = Project.process_in_batches_spoof(cfg)
    m = f'Created {n_batches} Batches to Process {n_images} Images'
    logger.info(m)
    return Project, 1, m 

# Define shared variables and their locks
import threading
import concurrent.futures
n_rotate = 0
n_rotate_lock = threading.Lock()
n_corrupt = 0
n_corrupt_lock = threading.Lock()

def process_image_vertical(image_path, dir_images_unprocessed, rotate_lock, corrupt_lock):
    global n_rotate, n_corrupt

    try:
        image = cv2.imread(os.path.join(dir_images_unprocessed, image_path))
        h, w, img_c = image.shape
        image, img_h, img_w, did_rotate = make_image_vertical(image, h, w, do_rotate_180=False)
        if did_rotate:
            with rotate_lock:
                n_rotate += 1
        os.remove(os.path.join(dir_images_unprocessed, image_path))
        cv2.imwrite(os.path.join(dir_images_unprocessed, image_path), image)
    except:
        with corrupt_lock:
            n_corrupt += 1
        os.remove(os.path.join(dir_images_unprocessed, image_path))

def make_images_in_dir_vertical(dir_images_unprocessed, cfg, N_THREADS=16):
    global n_rotate, n_corrupt

    if cfg['leafmachine']['do']['check_for_corrupt_images_make_vertical']:
        n_rotate = 0
        n_corrupt = 0
        n_total = len(os.listdir(dir_images_unprocessed))

        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            image_paths = [image_name for image_name in os.listdir(dir_images_unprocessed) if image_name.lower().endswith((".jpg", ".jpeg", ".tiff", ".tif", ".png", ".jp2", ".bmp", ".dib"))]
            futures = [executor.submit(process_image_vertical, image_path, dir_images_unprocessed, n_rotate_lock, n_corrupt_lock) for image_path in image_paths]
            for _ in tqdm(concurrent.futures.as_completed(futures), desc=f'{bcolors.BOLD}     Checking Image Dimensions{bcolors.ENDC}', colour="cyan", position=0, total=n_total):
                pass

        m = ''.join(['Number of Images Rotated: ', str(n_rotate)])
        Print_Verbose(cfg, 2, m).bold()
        m2 = ''.join(['Number of Images Corrupted: ', str(n_corrupt)])
        if n_corrupt > 0:
            Print_Verbose(cfg, 2, m2).warning()
        else:
            Print_Verbose(cfg, 2, m2).bold()

# def process_image_vertical(image_path, dir_images_unprocessed):
#     try:
#         image = cv2.imread(os.path.join(dir_images_unprocessed, image_path))
#         h, w, img_c = image.shape
#         image, img_h, img_w, did_rotate = make_image_vertical(image, h, w, do_rotate_180=False)
#         if did_rotate:
#             n_rotate += 1
#         cv2.imwrite(os.path.join(dir_images_unprocessed, image_path), image)
#     except:
#         n_corrupt += 1
#         os.remove(os.path.join(dir_images_unprocessed, image_path))

# def make_images_in_dir_vertical(dir_images_unprocessed, cfg, N_THREADS=16):
#     if cfg['leafmachine']['do']['check_for_corrupt_images_make_vertical']:
#         n_rotate = 0
#         n_corrupt = 0
#         n_total = len(os.listdir(dir_images_unprocessed))

#         # Create a ThreadPoolExecutor with N_THREADS
#         with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as executor:
#             image_paths = [image_name_jpg for image_name_jpg in os.listdir(dir_images_unprocessed) if
#                            image_name_jpg.endswith((".jpg", ".JPG", ".jpeg", ".JPEG", ".tiff", ".tif", ".png", ".PNG",
#                                                      ".TIFF", ".TIF", ".jp2", ".JP2", ".bmp", ".BMP", ".dib", ".DIB"))]
#             for _ in tqdm(executor.map(process_image_vertical, image_paths, [dir_images_unprocessed] * len(image_paths)),
#                           desc=f'{bcolors.BOLD}     Checking Image Dimensions{bcolors.ENDC}', colour="cyan",
#                           position=0, total=n_total):
#                 pass

#         m = ''.join(['Number of Images Rotated: ', str(n_rotate)])
#         Print_Verbose(cfg, 2, m).bold()
#         m2 = ''.join(['Number of Images Corrupted: ', str(n_corrupt)])
#         if n_corrupt > 0:
#             Print_Verbose(cfg, 2, m2).warning
#         else:
#             Print_Verbose(cfg, 2, m2).bold()


# def make_images_in_dir_vertical(dir_images_unprocessed, cfg):
#     if cfg['leafmachine']['do']['check_for_corrupt_images_make_vertical']:
#         n_rotate = 0
#         n_corrupt = 0
#         n_total = len(os.listdir(dir_images_unprocessed))
#         for image_name_jpg in tqdm(os.listdir(dir_images_unprocessed), desc=f'{bcolors.BOLD}     Checking Image Dimensions{bcolors.ENDC}',colour="cyan",position=0,total = n_total):
#             if image_name_jpg.endswith((".jpg",".JPG",".jpeg",".JPEG")):
#                 try:
#                     image = cv2.imread(os.path.join(dir_images_unprocessed, image_name_jpg))
#                     h, w, img_c = image.shape
#                     image, img_h, img_w, did_rotate = make_image_vertical(image, h, w, do_rotate_180=False)
#                     if did_rotate:
#                         n_rotate += 1
#                     cv2.imwrite(os.path.join(dir_images_unprocessed,image_name_jpg), image)
#                 except:
#                     n_corrupt +=1
#                     os.remove(os.path.join(dir_images_unprocessed, image_name_jpg))
#             # TODO check that below works as intended 
#             elif image_name_jpg.endswith((".tiff",".tif",".png",".PNG",".TIFF",".TIF",".jp2",".JP2",".bmp",".BMP",".dib",".DIB")):
#                 try:
#                     image = cv2.imread(os.path.join(dir_images_unprocessed, image_name_jpg))
#                     h, w, img_c = image.shape
#                     image, img_h, img_w, did_rotate = make_image_vertical(image, h, w, do_rotate_180=False)
#                     if did_rotate:
#                         n_rotate += 1
#                     image_name_jpg = '.'.join([image_name_jpg.split('.')[0], 'jpg'])
#                     cv2.imwrite(os.path.join(dir_images_unprocessed,image_name_jpg), image)
#                 except:
#                     n_corrupt +=1
#                     os.remove(os.path.join(dir_images_unprocessed, image_name_jpg))
#         m = ''.join(['Number of Images Rotated: ', str(n_rotate)])
#         Print_Verbose(cfg, 2, m).bold()
#         m2 = ''.join(['Number of Images Corrupted: ', str(n_corrupt)])
#         if n_corrupt > 0:
#             Print_Verbose(cfg, 2, m2).warning
#         else:
#             Print_Verbose(cfg, 2, m2).bold

def make_image_vertical(image, h, w, do_rotate_180):
    did_rotate = False
    if do_rotate_180:
        # try:
        image = cv2.rotate(image, cv2.ROTATE_180)
        img_h, img_w, img_c = image.shape
        did_rotate = True
        # print("      Rotated 180")
    else:
        if h < w:
            # try:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            img_h, img_w, img_c = image.shape
            did_rotate = True
            # print("      Rotated 90 CW")
        elif h >= w:
            image = image
            img_h = h
            img_w = w
            # print("      Not Rotated")
    return image, img_h, img_w, did_rotate
    

def make_image_horizontal(image, h, w, do_rotate_180):
    if h > w:
        if do_rotate_180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), w, h, True
    return image, w, h, False

def make_images_in_dir_horizontal(dir_images_unprocessed, cfg):
    # if cfg['leafmachine']['do']['check_for_corrupt_images_make_horizontal']:
    n_rotate = 0
    n_corrupt = 0
    n_total = len(os.listdir(dir_images_unprocessed))
    for image_name_jpg in tqdm(os.listdir(dir_images_unprocessed), desc=f'{bcolors.BOLD}     Checking Image Dimensions{bcolors.ENDC}', colour="cyan", position=0, total=n_total):
        if image_name_jpg.endswith((".jpg",".JPG",".jpeg",".JPEG")):
            try:
                image = cv2.imread(os.path.join(dir_images_unprocessed, image_name_jpg))
                h, w, img_c = image.shape
                image, img_h, img_w, did_rotate = make_image_horizontal(image, h, w, do_rotate_180=False)
                if did_rotate:
                    n_rotate += 1
                cv2.imwrite(os.path.join(dir_images_unprocessed,image_name_jpg), image)
            except:
                n_corrupt +=1
                os.remove(os.path.join(dir_images_unprocessed, image_name_jpg))
        # TODO check that below works as intended 
        elif image_name_jpg.endswith((".tiff",".tif",".png",".PNG",".TIFF",".TIF",".jp2",".JP2",".bmp",".BMP",".dib",".DIB")):
            try:
                image = cv2.imread(os.path.join(dir_images_unprocessed, image_name_jpg))
                h, w, img_c = image.shape
                image, img_h, img_w, did_rotate = make_image_horizontal(image, h, w, do_rotate_180=False)
                if did_rotate:
                    n_rotate += 1
                image_name_jpg = '.'.join([image_name_jpg.split('.')[0], 'jpg'])
                cv2.imwrite(os.path.join(dir_images_unprocessed,image_name_jpg), image)
            except:
                n_corrupt +=1
                os.remove(os.path.join(dir_images_unprocessed, image_name_jpg))
    m = ''.join(['Number of Images Rotated: ', str(n_rotate)])
    print(m)
    # Print_Verbose(cfg, 2, m).bold()
    m2 = ''.join(['Number of Images Corrupted: ', str(n_corrupt)])
    print(m2)


@dataclass
class Print_Verbose_Error():
    cfg: str = ''
    indent_level: int = 0
    message: str = ''
    error: str = ''

    def __init__(self, cfg,indent_level,message,error) -> None:
        self.cfg = cfg
        self.indent_level = indent_level
        self.message = message
        self.error = error

    def print_error_to_console(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['optional_warnings']:
            print(f"{bcolors.FAIL}{white_space}{self.message} ERROR: {self.error}{bcolors.ENDC}")

    def print_warning_to_console(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['optional_warnings']:
            print(f"{bcolors.WARNING}{white_space}{self.message} ERROR: {self.error}{bcolors.ENDC}")

@dataclass
class Print_Verbose():
    cfg: str = ''
    indent_level: int = 0
    message: str = ''

    def __init__(self, cfg, indent_level, message) -> None:
        self.cfg = cfg
        self.indent_level = indent_level
        self.message = message

    def bold(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['verbose']:
            print(f"{bcolors.BOLD}{white_space}{self.message}{bcolors.ENDC}")

    def green(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['verbose']:
            print(f"{bcolors.OKGREEN}{white_space}{self.message}{bcolors.ENDC}")

    def cyan(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['verbose']:
            print(f"{bcolors.OKCYAN}{white_space}{self.message}{bcolors.ENDC}")

    def blue(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['verbose']:
            print(f"{bcolors.OKBLUE}{white_space}{self.message}{bcolors.ENDC}")

    def warning(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['verbose']:
            print(f"{bcolors.WARNING}{white_space}{self.message}{bcolors.ENDC}")

    def plain(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['verbose']:
            print(f"{white_space}{self.message}")

def print_main_start(message):
    indent_level = 1
    white_space = " " * 5 * indent_level
    end = " " * int(80 - len(message) - len(white_space))
    # end_white_space = " " * end
    blank = " " * 80
    print(f"{bcolors.CBLUEBG2}{blank}{bcolors.ENDC}")
    print(f"{bcolors.CBLUEBG2}{white_space}{message}{end}{bcolors.ENDC}")
    print(f"{bcolors.CBLUEBG2}{blank}{bcolors.ENDC}")

def print_main_success(message):
    indent_level = 1
    white_space = " " * 5 * indent_level
    end = " " * int(80 - len(message) - len(white_space))
    blank = " " * 80
    # end_white_space = " " * end
    print(f"{bcolors.CGREENBG2}{blank}{bcolors.ENDC}")
    print(f"{bcolors.CGREENBG2}{white_space}{message}{end}{bcolors.ENDC}")
    print(f"{bcolors.CGREENBG2}{blank}{bcolors.ENDC}")

def print_main_warn(message):
    indent_level = 1
    white_space = " " * 5 * indent_level
    end = " " * int(80 - len(message) - len(white_space))
    # end_white_space = " " * end
    blank = " " * 80
    print(f"{bcolors.CYELLOWBG2}{blank}{bcolors.ENDC}")
    print(f"{bcolors.CYELLOWBG2}{white_space}{message}{end}{bcolors.ENDC}")
    print(f"{bcolors.CYELLOWBG2}{blank}{bcolors.ENDC}")

def print_main_fail(message):
    indent_level = 1
    white_space = " " * 5 * indent_level
    end = " " * int(80 - len(message) - len(white_space))
    # end_white_space = " " * end
    blank = " " * 80
    print(f"{bcolors.CREDBG2}{blank}{bcolors.ENDC}")
    print(f"{bcolors.CREDBG2}{white_space}{message}{end}{bcolors.ENDC}")
    print(f"{bcolors.CREDBG2}{blank}{bcolors.ENDC}")

def print_main_info(message):
    indent_level = 2
    white_space = " " * 5 * indent_level
    end = " " * int(80 - len(message) - len(white_space))
    # end_white_space = " " * end
    print(f"{bcolors.CGREYBG}{white_space}{message}{end}{bcolors.ENDC}")
    
# def report_config(dir_home, cfg_file_path):
#     print_main_start("Loading Configuration File")
#     if cfg_file_path == None:
#         print_main_info(''.join([os.path.join(dir_home, 'LeafMachine2.yaml')]))
#     elif cfg_file_path == 'test_installation':
#         print_main_info(''.join([os.path.join(dir_home, 'demo','LeafMachine2_demo.yaml')]))
#     else:
#         print_main_info(cfg_file_path)

# def report_config_VV(dir_home, cfg_file_path):
#     print_main_start("Loading Configuration File")
#     if cfg_file_path == None:
#         print_main_info(''.join([os.path.join(dir_home, 'VoucherVision.yaml')]))
#     elif cfg_file_path == 'test_installation':
#         print_main_info(''.join([os.path.join(dir_home, 'demo','VoucherVision_demo.yaml')]))
#     else:
#         print_main_info(cfg_file_path)

def report_config(dir_home, cfg_file_path, system='LeafMachine2'):
    print_main_start("Loading Configuration File")
    
    if system not in ['LeafMachine2', 'VoucherVision', 'SpecimenCrop', 'DetectPhenology', 'CensorArchivalComponents']:
        raise ValueError("Invalid system. Expected 'LeafMachine2' or 'VoucherVision' or 'SpecimenCrop', or 'DetectPhenology', 'CensorArchivalComponents'.")
    
    if cfg_file_path == None:
        print_main_info(''.join([os.path.join(dir_home, f'{system}.yaml')]))
    elif cfg_file_path == 'test_installation':
        print_main_info(''.join([os.path.join(dir_home, 'demo', f'{system}_demo.yaml')]))
    else:
        print_main_info(cfg_file_path)


def make_file_names_valid(dir, cfg):
    if cfg['leafmachine']['do']['check_for_illegal_filenames']:
        n_total = len(os.listdir(dir))
        for file in tqdm(os.listdir(dir), desc=f'{bcolors.HEADER}     Removing illegal characters from file names{bcolors.ENDC}',colour="cyan",position=0,total = n_total):
            name = Path(file).stem
            ext = Path(file).suffix
            name_cleaned = re.sub(r"[^a-zA-Z0-9_-]","-",name)
            if name_cleaned != name:
                name_new = ''.join([name_cleaned,ext])
                i = 0
                try:
                    os.rename(os.path.join(dir,file), os.path.join(dir,name_new))
                except:
                    time.sleep(0.1)
                    while os.path.exists(os.path.join(dir,name_new)):
                        i += 1
                        name_new = '_'.join([name_cleaned, str(i), ext])
                    os.rename(os.path.join(dir,file), os.path.join(dir,name_new))

# def load_config_file(dir_home, cfg_file_path):
#     if cfg_file_path == None: # Default path
#         return load_cfg(dir_home)
#     else:
#         if cfg_file_path == 'test_installation':
#             path_cfg = os.path.join(dir_home,'demo','LeafMachine2_demo.yaml')                     
#             return get_cfg_from_full_path(path_cfg)
#         else: # Custom path
#             return get_cfg_from_full_path(cfg_file_path)
        
# def load_config_file_VV(dir_home, cfg_file_path):
#     if cfg_file_path == None: # Default path
#         return load_cfg_VV(dir_home)
#     else:
#         if cfg_file_path == 'test_installation':
#             path_cfg = os.path.join(dir_home,'demo','VoucherVision_demo.yaml')                     
#             return get_cfg_from_full_path(path_cfg)
#         else: # Custom path
#             return get_cfg_from_full_path(cfg_file_path)

def load_config_file(dir_home, cfg_file_path, system='LeafMachine2'):
    if system not in ['LeafMachine2', 'VoucherVision', 'SpecimenCrop','DetectPhenology', 'CensorArchivalComponents']:
        raise ValueError("Invalid system. Expected 'LeafMachine2' or 'VoucherVision' or 'SpecimenCrop' or 'DetectPhenology', 'CensorArchivalComponents'.")

    if cfg_file_path is None:  # Default path
        if system == 'LeafMachine2':
            return load_cfg(dir_home, system='LeafMachine2')  # For LeafMachine2

        elif system == 'VoucherVision': # VoucherVision
            return load_cfg(dir_home, system='VoucherVision')  # For VoucherVision

        elif system == 'SpecimenCrop': # SpecimenCrop
            return load_cfg(dir_home, system='SpecimenCrop')  # For SpecimenCrop
        
        elif system == 'DetectPhenology': # DetectPhenology
            return load_cfg(dir_home, system='DetectPhenology')  # For DetectPhenology
        
        elif system == 'CensorArchivalComponents': # CensorArchivalComponents
            return load_cfg(dir_home, system='CensorArchivalComponents')  # For CensorArchivalComponents


    else:
        if cfg_file_path == 'test_installation':
            path_cfg = os.path.join(dir_home, 'demo', f'{system}_demo.yaml')                     
            return get_cfg_from_full_path(path_cfg)
        else:  # Custom path
            return get_cfg_from_full_path(cfg_file_path)

        
def load_config_file_testing(dir_home, cfg_file_path):
    if cfg_file_path == None: # Default path
        return load_cfg(dir_home)
    else:
        if cfg_file_path == 'test_installation':
            path_cfg = os.path.join(dir_home,'demo','demo.yaml')                     
            return get_cfg_from_full_path(path_cfg)
        else: # Custom path
            return get_cfg_from_full_path(cfg_file_path)

def subset_dir_images(cfg, Project, Dirs):
    if cfg['leafmachine']['project']['process_subset_of_images']:
        dir_images_subset = cfg['leafmachine']['project']['dir_images_subset']
        num_images_per_species = cfg['leafmachine']['project']['n_images_per_species']
        if cfg['leafmachine']['project']['species_list'] is not None:
            species_list = import_csv(cfg['leafmachine']['project']['species_list'])
            species_list = species_list.iloc[:, 0].tolist()
        else:
            species_list = None

        validate_dir(dir_images_subset)

        species_counts = {}
        filenames = os.listdir(Project.dir_images)
        random.shuffle(filenames)
        for filename in filenames:
            species_name = filename.split('.')[0]
            species_name = species_name.split('_')[2:]
            species_name = '_'.join([species_name[0], species_name[1], species_name[2]])

            if (species_list is None) or ((species_name in species_list) and (species_list is not None)):
            
                if species_name not in species_counts:
                    species_counts[species_name] = 0
                
                if species_counts[species_name] < num_images_per_species:
                    species_counts[species_name] += 1
                    src_path = os.path.join(Project.dir_images, filename)
                    dest_path = os.path.join(dir_images_subset, filename)
                    shutil.copy(src_path, dest_path)
        
        Project.dir_images = dir_images_subset
        
        subset_csv_name = os.path.join(Dirs.dir_images_subset, '.'.join([Dirs.run_name, 'csv']))
        df = pd.DataFrame({'species_name': list(species_counts.keys()), 'count': list(species_counts.values())})
        df.to_csv(subset_csv_name, index=False)
        return Project
    else:
        return Project

'''# Define function to be executed by each worker
def worker_crop(rank, cfg, dir_home, Project, Dirs):
    # Set worker seed based on rank
    np.random.seed(rank)
    # Call function for this worker
    crop_detections_from_images(cfg, dir_home, Project, Dirs)

def crop_detections_from_images(cfg, dir_home, Project, Dirs):
    num_workers = 6
    
    # Initialize and start worker processes
    processes = []
    for rank in range(num_workers):
        p = mp.Process(target=worker_crop, args=(rank, cfg, dir_home, Project, Dirs))
        p.start()
        processes.append(p)

    # Wait for all worker processes to finish
    for p in processes:
        p.join()'''
def crop_detections_from_images_worker_SpecimenCrop(filename, analysis, Project, Dirs, cfg, save_list, original_img_dir):
    try:
        full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpg'])))
    except:
        full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpeg'])))

    try:
        archival = analysis['Detections_Archival_Components']
        has_archival = True
    except: 
        archival = None
        has_archival = False

    try:
        plant = analysis['Detections_Plant_Components']
        has_plant = True
    except: 
        plant = None
        has_plant = False

    if has_archival or has_plant:
        crop_component_from_yolo_coords_SpecimenCrop(Dirs, cfg, analysis, has_archival, has_plant, archival, plant, full_image, filename, save_list, original_img_dir)

def crop_detections_from_images_worker_VV(filename, analysis, Project, Dirs, save_per_image, save_per_class, save_list, binarize_labels):
    try:
        full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpg'])))
    except:
        full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpeg'])))

    try:
        archival = analysis['Detections_Archival_Components']
        has_archival = True
    except: 
        has_archival = False

    try:
        plant = analysis['Detections_Plant_Components']
        has_plant = True
    except: 
        has_plant = False

    if has_archival and (save_per_image or save_per_class):
        crop_component_from_yolo_coords_VV('ARCHIVAL', Dirs, analysis, archival, full_image, filename, save_per_image, save_per_class, save_list)
 
"""
Works with Project, not SQL
def crop_detections_from_images_worker(filename, analysis, Project, Dirs, save_per_image, save_per_class, save_list, binarize_labels):
    try:
        full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpg'])))
    except:
        full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpeg'])))

    try:
        archival = analysis['Detections_Archival_Components']
        has_archival = True
    except: 
        has_archival = False

    try:
        plant = analysis['Detections_Plant_Components']
        has_plant = True
    except: 
        has_plant = False

    if has_archival and (save_per_image or save_per_class):
        crop_component_from_yolo_coords('ARCHIVAL', Dirs, analysis, archival, full_image, filename, save_per_image, save_per_class, save_list)
    if has_plant and (save_per_image or save_per_class):
        crop_component_from_yolo_coords('PLANT', Dirs, analysis, plant, full_image, filename, save_per_image, save_per_class, save_list)
"""
def crop_detections_from_images_worker(image_name, archival_components, plant_components, Dirs, save_per_image, save_per_class, save_list, binarize_labels):
    try:
        full_image = cv2.imread(os.path.join(Dirs.dir_images_local, f"{image_name}.jpg"))
        if full_image is None:
            full_image = cv2.imread(os.path.join(Dirs.dir_images_local, f"{image_name}.jpeg"))
    except Exception as e:
        print(e)
        full_image = cv2.imread(os.path.join(Dirs.dir_images_local, f"{image_name}.jpeg"))

    # Get height and width from the images table
    try:
        conn = sqlite3.connect(Dirs.database)
        cursor = conn.cursor()
        cursor.execute("SELECT width, height FROM images WHERE name = ?", (image_name,))
        row = cursor.fetchone()
        width, height = row if row else (None, None)
        conn.close()
    except Exception as e:
        width, height = None, None
        print(f"Error retrieving dimensions for {image_name}: {e}")

    if archival_components:
        archival_annotations = ast.literal_eval(archival_components[0][1]) 
        crop_component_from_yolo_coords('ARCHIVAL', Dirs, archival_annotations, full_image, image_name, save_per_image, save_per_class, save_list, width, height)
    
    if plant_components:
        plant_annotations = ast.literal_eval(plant_components[0][1]) 
        crop_component_from_yolo_coords('PLANT', Dirs, plant_annotations, full_image, image_name, save_per_image, save_per_class, save_list, width, height)




"""
works with Project, not with SQL
def crop_detections_from_images(cfg, time_report, logger, dir_home, Project, Dirs, batch_size=50):
    t2_start = perf_counter()
    logger.name = 'Crop Components'
    
    if cfg['leafmachine']['cropped_components']['do_save_cropped_annotations']:
        detections = cfg['leafmachine']['cropped_components']['save_cropped_annotations']
        logger.info(f"Cropping {detections} components from images")

        save_per_image = cfg['leafmachine']['cropped_components']['save_per_image']
        save_per_class = cfg['leafmachine']['cropped_components']['save_per_annotation_class']
        save_list = cfg['leafmachine']['cropped_components']['save_cropped_annotations']
        try:
            binarize_labels = cfg['leafmachine']['cropped_components']['binarize_labels']
        except:
            binarize_labels = False
        if cfg['leafmachine']['project']['batch_size'] is None:
            batch_size = 50
        else:
            batch_size = int(cfg['leafmachine']['project']['batch_size'])
        if cfg['leafmachine']['project']['num_workers'] is None:
            num_workers = 4 
        else:
            num_workers = int(cfg['leafmachine']['project']['num_workers'])

        if binarize_labels:
            save_per_class = True

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(0, len(Project.project_data), batch_size):
                batch = list(Project.project_data.items())[i:i+batch_size]
                # print(f'Cropping Detections from Images {i} to {i+batch_size}')
                logger.info(f'Cropping {detections} from images {i} to {i+batch_size} [{len(Project.project_data)}]')
                for filename, analysis in batch:
                    if len(analysis) != 0:
                        futures.append(executor.submit(crop_detections_from_images_worker, filename, analysis, Project, Dirs, save_per_image, save_per_class, save_list, binarize_labels))

                for future in concurrent.futures.as_completed(futures):
                    pass
                futures.clear()

    t2_stop = perf_counter()
    t_crops = f"[Cropping elapsed time] {round(t2_stop - t2_start)} seconds ({round((t2_stop - t2_start)/60)} minutes)"
    logger.info(t_crops)
    time_report['t_crops'] = t_crops
    return time_report
"""
def crop_detections_from_images(cfg, time_report, logger, dir_home, Project, Dirs, batch_size=50):
    t2_start = perf_counter()
    logger.name = 'Crop Components'
    
    if cfg['leafmachine']['cropped_components']['do_save_cropped_annotations']:
        detections = cfg['leafmachine']['cropped_components']['save_cropped_annotations']
        logger.info(f"Cropping {detections} components from images")

        save_per_image = cfg['leafmachine']['cropped_components']['save_per_image']
        save_per_class = cfg['leafmachine']['cropped_components']['save_per_annotation_class']
        save_list = cfg['leafmachine']['cropped_components']['save_cropped_annotations']
        try:
            binarize_labels = cfg['leafmachine']['cropped_components']['binarize_labels']
        except:
            binarize_labels = False
        if cfg['leafmachine']['project']['batch_size'] is None:
            batch_size = 50
        else:
            batch_size = int(cfg['leafmachine']['project']['batch_size'])
        if cfg['leafmachine']['project']['num_workers'] is None:
            num_workers = 4 
        else:
            num_workers = int(cfg['leafmachine']['project']['num_workers'])

        if binarize_labels:
            save_per_class = True

        conn = sqlite3.connect(Dirs.database)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM images")
        image_names = [row[0] for row in cursor.fetchall()]

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(0, len(image_names), batch_size):
                batch = image_names[i:i+batch_size]
                logger.info(f'Cropping {detections} from images {i} to {i+batch_size} [{len(image_names)}]')
                for image_name in batch:
                    cursor.execute("SELECT component, annotations FROM archival_components WHERE image_name = ?", (image_name,))
                    archival_components = cursor.fetchall()
                    
                    cursor.execute("SELECT component, annotations FROM plant_components WHERE image_name = ?", (image_name,))
                    plant_components = cursor.fetchall()
                    
                    if archival_components or plant_components:
                        futures.append(executor.submit(crop_detections_from_images_worker, image_name, archival_components, plant_components, Dirs, save_per_image, save_per_class, save_list, binarize_labels))

                for future in concurrent.futures.as_completed(futures):
                    pass
                futures.clear()

        conn.close()

    t2_stop = perf_counter()
    t_crops = f"[Cropping elapsed time] {round(t2_stop - t2_start)} seconds ({round((t2_stop - t2_start)/60)} minutes)"
    logger.info(t_crops)
    time_report['t_crops'] = t_crops
    return time_report





def crop_detections_from_images_VV(cfg, logger, dir_home, Project, Dirs, batch_size=50):
    t2_start = perf_counter()
    logger.name = 'Crop Components'
    
    if cfg['leafmachine']['cropped_components']['do_save_cropped_annotations']:
        detections = cfg['leafmachine']['cropped_components']['save_cropped_annotations']
        logger.info(f"Cropping {detections} components from images")

        save_per_image = cfg['leafmachine']['cropped_components']['save_per_image']
        save_per_class = cfg['leafmachine']['cropped_components']['save_per_annotation_class']
        save_list = cfg['leafmachine']['cropped_components']['save_cropped_annotations']
        binarize_labels = cfg['leafmachine']['cropped_components']['binarize_labels']
        if cfg['leafmachine']['project']['batch_size'] is None:
            batch_size = 50
        else:
            batch_size = int(cfg['leafmachine']['project']['batch_size'])
        if cfg['leafmachine']['project']['num_workers'] is None:
            num_workers = 4 
        else:
            num_workers = int(cfg['leafmachine']['project']['num_workers'])

        if binarize_labels:
            save_per_class = True

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(0, len(Project.project_data), batch_size):
                batch = list(Project.project_data.items())[i:i+batch_size]
                # print(f'Cropping Detections from Images {i} to {i+batch_size}')
                logger.info(f'Cropping {detections} from images {i} to {i+batch_size} [{len(Project.project_data)}]')
                for filename, analysis in batch:
                    if len(analysis) != 0:
                        futures.append(executor.submit(crop_detections_from_images_worker_VV, filename, analysis, Project, Dirs, save_per_image, save_per_class, save_list, binarize_labels))

                for future in concurrent.futures.as_completed(futures):
                    pass
                futures.clear()

    t2_stop = perf_counter()
    logger.info(f"Save cropped components --- elapsed time: {round(t2_stop - t2_start)} seconds")
# def crop_detections_from_images_VV(cfg, logger, dir_home, Project, Dirs, batch_size=50):
#     t2_start = perf_counter()
#     logger.name = 'Crop Components'
    
#     if cfg['leafmachine']['cropped_components']['do_save_cropped_annotations']:
#         detections = cfg['leafmachine']['cropped_components']['save_cropped_annotations']
#         logger.info(f"Cropping {detections} components from images")

#         save_per_image = cfg['leafmachine']['cropped_components']['save_per_image']
#         save_per_class = cfg['leafmachine']['cropped_components']['save_per_annotation_class']
#         save_list = cfg['leafmachine']['cropped_components']['save_cropped_annotations']
#         binarize_labels = cfg['leafmachine']['cropped_components']['binarize_labels']
#         if cfg['leafmachine']['project']['batch_size'] is None:
#             batch_size = 50
#         else:
#             batch_size = int(cfg['leafmachine']['project']['batch_size'])

#         if binarize_labels:
#             save_per_class = True

#         for i in range(0, len(Project.project_data), batch_size):
#             batch = list(Project.project_data.items())[i:i+batch_size]
#             logger.info(f"Cropping {detections} from images {i} to {i+batch_size} [{len(Project.project_data)}]")
#             for filename, analysis in batch:
#                 if len(analysis) != 0:
#                     crop_detections_from_images_worker_VV(filename, analysis, Project, Dirs, save_per_image, save_per_class, save_list, binarize_labels)

#     t2_stop = perf_counter()
#     logger.info(f"Save cropped components --- elapsed time: {round(t2_stop - t2_start)} seconds")



def crop_detections_from_images_SpecimenCrop(cfg, time_report, logger, dir_home, Project, Dirs, original_img_dir=None):
    t2_start = perf_counter()
    logger.name = 'Crop Components --- Specimen Crop'

    if cfg['leafmachine']['modules']['specimen_crop']:
        # save_list = ['ruler', 'barcode', 'colorcard', 'label', 'map', 'envelope', 'photo', 'attached_item', 'weights',
        #               'leaf_whole', 'leaf_partial', 'leaflet', 'seed_fruit_one', 'seed_fruit_many', 'flower_one', 'flower_many', 'bud', 'specimen', 'roots', 'wood']
        save_list = cfg['leafmachine']['cropped_components']['include_these_objects_in_specimen_crop']

        logger.info(f"Cropping to include {save_list} components from images")

        for i, (filename, analysis) in enumerate(Project.project_data.items()):
            if len(analysis) != 0:
                logger.info(f'Cropping {save_list} from image {i} of {len(Project.project_data)}: {filename}')
                crop_detections_from_images_worker_SpecimenCrop(filename, analysis, Project, Dirs, cfg, save_list, original_img_dir)

    t2_stop = perf_counter()
    t_speccrops = f"[SpecimenCrop elapsed time] {round(t2_stop - t2_start)} seconds ({round((t2_stop - t2_start)/60)} minutes)"
    logger.info(t_speccrops)
    time_report['t_speccrops'] = t_speccrops
    return time_report

# def crop_detections_from_images_SpecimenCrop(cfg, logger, dir_home, Project, Dirs, original_img_dir=None, batch_size=50):
#     t2_start = perf_counter()
#     logger.name = 'Crop Components --- Specimen Crop'
    
#     if cfg['leafmachine']['modules']['specimen_crop']:
#         # save_list = ['ruler', 'barcode', 'colorcard', 'label', 'map', 'envelope', 'photo', 'attached_item', 'weights',
#         #               'leaf_whole', 'leaf_partial', 'leaflet', 'seed_fruit_one', 'seed_fruit_many', 'flower_one', 'flower_many', 'bud', 'specimen', 'roots', 'wood']
#         save_list = cfg['leafmachine']['cropped_components']['include_these_objects_in_specimen_crop']

#         logger.info(f"Cropping to include {save_list} components from images")

#         if cfg['leafmachine']['project']['batch_size'] is None:
#             batch_size = 50
#         else:
#             batch_size = int(cfg['leafmachine']['project']['batch_size'])
#         if cfg['leafmachine']['project']['num_workers'] is None:
#             num_workers = 4 
#         else:
#             num_workers = int(cfg['leafmachine']['project']['num_workers'])

#         with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
#             futures = []
#             for i in range(0, len(Project.project_data), batch_size):
#                 batch = list(Project.project_data.items())[i:i+batch_size]
#                 # print(f'Cropping Detections from Images {i} to {i+batch_size}')
#                 logger.info(f'Cropping {save_list} from images {i} to {i+batch_size} [{len(Project.project_data)}]')
#                 for filename, analysis in batch:
#                     if len(analysis) != 0:
#                         futures.append(executor.submit(crop_detections_from_images_worker_SpecimenCrop, filename, analysis, Project, Dirs, save_list, original_img_dir))

#                 for future in concurrent.futures.as_completed(futures):
#                     pass
#                 futures.clear()

#     t2_stop = perf_counter()
#     logger.info(f"Save cropped components --- elapsed time: {round(t2_stop - t2_start)} seconds")

'''
# Single threaded
def crop_detections_from_images(cfg, dir_home, Project, Dirs):
    if cfg['leafmachine']['cropped_components']['do_save_cropped_annotations']:
        save_per_image = cfg['leafmachine']['cropped_components']['save_per_image']
        save_per_class = cfg['leafmachine']['cropped_components']['save_per_annotation_class']
        save_list = cfg['leafmachine']['cropped_components']['save_cropped_annotations']
        binarize_labels = cfg['leafmachine']['cropped_components']['binarize_labels']
        if binarize_labels:
            save_per_class = True

        for filename, analysis in  tqdm(Project.project_data.items(), desc=f'{bcolors.BOLD}     Cropping Detections from Images{bcolors.ENDC}',colour="cyan",position=0,total = len(Project.project_data.items())):
            if len(analysis) != 0:
                try:
                    full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpg'])))
                except:
                    full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpeg'])))

                try:
                    archival = analysis['Detections_Archival_Components']
                    has_archival = True
                except: 
                    has_archival = False

                try:
                    plant = analysis['Detections_Plant_Components']
                    has_plant = True
                except: 
                    has_plant = False

                if has_archival and (save_per_image or save_per_class):
                    crop_component_from_yolo_coords('ARCHIVAL', Dirs, analysis, archival, full_image, filename, save_per_image, save_per_class, save_list)
                if has_plant and (save_per_image or save_per_class):
                    crop_component_from_yolo_coords('PLANT', Dirs, analysis, plant, full_image, filename, save_per_image, save_per_class, save_list)
'''
# def extract_exif_data(cr2_file):
#     try:
#         with pyexiv2.Image(cr2_file) as img:
#             return img.read_exif()
#     except Exception as e:
#         raise ValueError(f"Error reading EXIF data from {cr2_file}: {str(e)}")
    
# def add_crop_to_description(description, cr2_file, min_x, min_y, max_x, max_y):
#     # Extracting just the filename
#     name_CR2 = cr2_file.split(os.path.sep)[-1]
    
#     try:
#         with rawpy.imread(cr2_file) as raw:
#             width, height = raw.sizes.raw_width, raw.sizes.raw_height
#     except Exception as e:
#         raise ValueError(f"Error reading image dimensions from {cr2_file}: {str(e)}")

#     top = min_y / height
#     left = min_x / width
#     bottom = max_y / height
#     right = max_x / width

#     # Set the cropping info as attributes of the description element
#     description.set('crs:CropTop', str(top))
#     description.set('crs:CropLeft', str(left))
#     description.set('crs:CropBottom', str(bottom))
#     description.set('crs:CropRight', str(right))
    
#     # Add other attributes
#     description.set('crs:Version', "15.1")
#     description.set('crs:ProcessVersion', "11.0")
#     description.set('crs:HasSettings', "True")
#     description.set('crs:CropAngle', "0")
#     description.set('crs:CropConstrainToWarp', "0")
#     description.set('crs:HasCrop', "True")
#     description.set('crs:AlreadyApplied', "False")
#     description.set('crs:RawFileName', name_CR2)

def create_XMP(cr2_file, xmp_file_path, min_x, min_y, max_x, max_y, orientation, padding):
    import rawpy
    # Extracting just the filename
    name_CR2 = cr2_file.split(os.path.sep)[-1]
    
    try:
        with rawpy.imread(cr2_file) as raw:
            width, height = raw.sizes.raw_width, raw.sizes.raw_height
    except Exception as e:
        raise ValueError(f"Error reading image dimensions from {cr2_file}: {str(e)}")

    # If image is in landscape orientation, adjust the coordinates
    # If top of image is on east/west side:
    if width > height:
        if (min_x - padding) < 0:
            top = 0
        else:
            top = (min_x - padding) / height # left

        if (max_x + padding) > height:
            bottom = 1
        else:
            bottom = (max_x + (2*padding)) / height # right


        if (min_y - padding) < 0:
            left = 0
        else:
            left = 1 - ((min_y - padding) / width) # top

        if ((max_y + padding) > width) or (max_y + (2*padding) > width):
            right = 0
        else:
            right = 1 - ((max_y + (2*padding)) / width) # bottom
        
        # Clamp values between 0 and 1
        top = max(0, min(1, top))
        left = max(0, min(1, left))
        bottom = max(0, min(1, bottom))
        right = max(0, min(1, right))
    else:
        top = (min_y - padding) / height
        left = (min_x - padding) / width
        bottom = (max_y + padding) / height
        right = (max_x + padding) / width

        # Clamp values between 0 and 1
        top = max(0, min(1, top))
        left = max(0, min(1, left))
        bottom = max(0, min(1, bottom))
        right = max(0, min(1, right))
    
    # Create the root element
    xmpmeta = Element('x:xmpmeta', {
        'xmlns:x': "adobe:ns:meta/",
        'x:xmptk': "Adobe XMP Core 7.0-c000 1.000000, 0000/00/00-00:00:00        "
    })
    
    # Create the RDF element
    rdf = SubElement(xmpmeta, 'rdf:RDF', {
        'xmlns:rdf': "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    })

    # Create the Description element with appropriate namespaces
    description_attribs = {
        'rdf:about': "",
        'xmlns:tiff': "http://ns.adobe.com/tiff/1.0/",
        'xmlns:exif': "http://ns.adobe.com/exif/1.0/",
        'xmlns:dc': "http://purl.org/dc/elements/1.1/",
        'xmlns:aux': "http://ns.adobe.com/exif/1.0/aux/",
        'xmlns:exifEX': "http://cipa.jp/exif/1.0/",
        'xmlns:xmp': "http://ns.adobe.com/xap/1.0/",
        'xmlns:photoshop': "http://ns.adobe.com/photoshop/1.0/",
        'xmlns:xmpMM': "http://ns.adobe.com/xap/1.0/mm/",
        'xmlns:stEvt': "http://ns.adobe.com/xap/1.0/sType/ResourceEvent#",
        'xmlns:crd': "http://ns.adobe.com/camera-raw-defaults/1.0/",
        'xmlns:xmpRights': "http://ns.adobe.com/xap/1.0/rights/",
        'xmlns:Iptc4xmpCore': "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/",
        'xmlns:crs': "http://ns.adobe.com/camera-raw-settings/1.0/",
        'tiff:Orientation': str(orientation),
        'crs:Version': "15.1",
        'crs:ProcessVersion': "11.0",
        'crs:HasSettings': "True",
        'crs:CropTop': str(top),
        'crs:CropLeft': str(left),
        'crs:CropBottom': str(bottom),
        'crs:CropRight': str(right),
        'crs:CropAngle': "0",
        'crs:CropConstrainToWarp': "0",
        'crs:HasCrop': "True",
        'crs:AlreadyApplied': "False",
        'crs:RawFileName': name_CR2
    }
    description = SubElement(rdf, 'rdf:Description', description_attribs)
    
    # Now we add the attributes that were inside the Description tag
    attributes_inside_description = {
        'crs:Version': "15.1",
        'crs:ProcessVersion': "11.0",
        'crs:HasSettings': "True",
        'crs:CropTop': str(top),
        'crs:CropLeft': str(left),
        'crs:CropBottom': str(bottom),
        'crs:CropRight': str(right),
        'crs:CropAngle': "0",
        'crs:CropConstrainToWarp': "0",
        'crs:HasCrop': "True",
        'crs:AlreadyApplied': "False",
        'crs:RawFileName': name_CR2
    }

    for attribute, value in attributes_inside_description.items():
        description.set(attribute, value)
    
    # Prettify and save the XML
    rough_string = tostring(xmpmeta, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    formatted_xml = reparsed.toprettyxml(indent="  ")

    # Removing the XML declaration as it's not present in the desired format
    formatted_xml = '\n'.join(formatted_xml.split('\n')[1:])

    with open(xmp_file_path, 'w') as xmp_file:
        xmp_file.write(formatted_xml)





def crop_image_raw(img, output_image_path, min_x, min_y, max_x, max_y):
    cropped = img.crop((min_x, min_y, max_x, max_y))
    cropped.save(output_image_path, quality='keep')

        
def create_temp_tiffs_dir(new_tiff_dir):
    temp_tiffs_dir = os.path.join(new_tiff_dir, 'temp_tiffs')

    # Create the new directory if it doesn't exist
    if not os.path.exists(temp_tiffs_dir):
        os.makedirs(temp_tiffs_dir)
        
    return temp_tiffs_dir

def copy_exif_data(input_image_path, output_image_path):
    import pyexiv2

    # Open the original image file
    with pyexiv2.Image(input_image_path) as img:
        exif_data = img.read_exif()
        # print(exif_data)

    # Open the output image file
    with pyexiv2.Image(output_image_path) as img:
        img.modify_exif(exif_data)
        img.modify_exif({'Exif.Image.Orientation': 1})


def convert_to_dng(tiff_path, dng_path):
    # Check the system
    system = platform.system()
    if system == "Windows":
        executable_path = "C:\Program Files\Adobe\Adobe DNG Converter\Adobe DNG Converter.exe"
    elif system == "Darwin":
        executable_path = "/Applications/Adobe DNG Converter.app/Contents/MacOS/Adobe DNG Converter"

    # prepare the command
    # command = f'"{executable_path}" -u -d "{os.path.dirname(tiff_path)}" -o "{dng_path}"'
    command = f'"{executable_path}" -c -d "{os.path.dirname(dng_path)}" "{tiff_path}"'
    print(command)
    # call the command
    subprocess.run(command, shell=False)

def crop_and_save_dng(input_path, output_path, x, y, width, height):
    import rawpy
    # Read the raw DNG file
    raw = rawpy.imread(input_path)

    # Crop the raw sensor data
    raw_cropped = raw.raw_image.copy()
    raw_cropped = raw_cropped[y:y+height, x:x+width]

    # Update the rawpy object with the cropped data
    raw.raw_image[:height, :width] = raw_cropped
    raw.raw_image_visible[:height, :width] = raw_cropped

    # Save the cropped image as a DNG file
    raw.save(output_path)


def get_colorspace(colorspace_choice):
    import rawpy
    # Match the string from YAML to the corresponding rawpy.ColorSpace attribute
    if colorspace_choice == 'raw':
        return rawpy.ColorSpace.raw
    elif colorspace_choice == 'sRGB':
        return rawpy.ColorSpace.sRGB
    elif colorspace_choice == 'Adobe':
        return rawpy.ColorSpace.Adobe
    elif colorspace_choice == 'Wide':
        return rawpy.ColorSpace.Wide
    elif colorspace_choice == 'ProPhoto':
        return rawpy.ColorSpace.ProPhoto
    elif colorspace_choice == 'XYZ':
        return rawpy.ColorSpace.XYZ
    else:
        raise ValueError("Invalid colorspace choice")

def process_detections(success, save_list, detections, detection_type, height, width, min_x, min_y, max_x, max_y):
    for detection in detections:
        detection_class = detection[0]
        detection_class = set_index_for_annotation(detection_class, detection_type)

        if (detection_class in save_list) or ('save_all' in save_list):
            location = yolo_to_position_ruler(detection, height, width)
            ruler_polygon = [
                (location[1], location[2]), 
                (location[3], location[2]), 
                (location[3], location[4]), 
                (location[1], location[4])
            ]

            x_coords = [x for x, y in ruler_polygon]
            y_coords = [y for x, y in ruler_polygon]

            min_x = min(min_x, *x_coords)
            min_y = min(min_y, *y_coords)
            max_x = max(max_x, *x_coords)
            max_y = max(max_y, *y_coords)
            success = True

    return min_x, min_y, max_x, max_y, success

def crop_component_from_yolo_coords_SpecimenCrop(Dirs, cfg, analysis, has_archival, has_plant, archival_detections, 
                                                 plant_detections, full_image, filename, save_list, original_img_dir):
    import rawpy
    import pyexiv2
    
    padding = int(cfg['leafmachine']['project']['padding_for_crop'])
    dir_images_local = cfg['leafmachine']['project']['dir_images_local']
    orientation = str(cfg['leafmachine']['project']['orientation'])

    save_tiff_to_original_dir = cfg['leafmachine']['project']['save_TIFF_to_original_dir']
    save_tiff_to_dir_output = cfg['leafmachine']['project']['save_TIFF_to_dir_output']

    # save_jpg_to_original_dir = cfg['leafmachine']['project']['save_JPG_to_original_dir']
    save_jpg_to_dir_output = cfg['leafmachine']['project']['save_JPG_to_dir_output']
        
    save_XMP = cfg['leafmachine']['project']['save_XMP_to_original_dir']

    colorspace_choice = cfg['leafmachine']['project']['colorspace']
    colorspace = get_colorspace(colorspace_choice)

    height = analysis['height']
    width = analysis['width']

    # Initialize variables for minimum and maximum coordinates
    min_x_init, min_y_init = float('inf'), float('inf')
    max_x_init, max_y_init = float('-inf'), float('-inf')

    success = False

    # If original_img_dir is provided, load the CR2 image instead of using the full_image
    if original_img_dir is not None:
        temp_tiffs_dir = create_temp_tiffs_dir(os.path.join(Dirs.dir_project,'Cropped_Images'))
        # Remove the extension from filename
        filename_stem = os.path.splitext(filename)[0]
        # Use glob to find the CR2 file in original_img_dir with the matching stem
        cr2_file = glob.glob(os.path.join(original_img_dir, filename_stem + '*.CR2'))[0]
        # Use rawpy to convert the raw CR2 file to TIFF
        temporary_tiff_path = os.path.join(temp_tiffs_dir, filename_stem + "_temp.TIFF")

        with rawpy.imread(cr2_file) as raw:
            # Get RGB image
            rgb = raw.postprocess(use_camera_wb=True, use_auto_wb=False, output_bps=16, output_color=colorspace, half_size=False)
            imageio.imsave(temporary_tiff_path, rgb)
        # Load the temporary TIFF as full_image
        full_image_tiff = Image.open(temporary_tiff_path)

    if has_archival:
        min_x_init, min_y_init, max_x_init, max_y_init, success = process_detections(success, save_list, archival_detections, "ARCHIVAL", height, width, min_x_init, min_y_init, max_x_init, max_y_init)

    if has_plant:
        min_x_init, min_y_init, max_x_init, max_y_init, success = process_detections(success, save_list, plant_detections, "PLANT", height, width, min_x_init, min_y_init, max_x_init, max_y_init)

    if success:
        ### Add padding, apply crop, save images ###
        # Calculate new min/max coordinates, ensuring they are within image bounds
        min_x = max(0, min_x_init - padding)
        min_y = max(0, min_y_init - padding)
        try:
            max_x = min(full_image.width, max_x_init + padding)
            max_y = min(full_image.height, max_y_init + padding)
        except:
            try:
                max_x = min(full_image.shape[1], max_x_init + padding)
                max_y = min(full_image.shape[0], max_y_init + padding)
            except:
                max_x = min(full_image_tiff.shape[1], max_x_init + padding)
                max_y = min(full_image_tiff.shape[0], max_y_init + padding)
        detection_cropped_name = '.'.join([filename, 'jpg'])

        # Save the cropped image
        # if original_img_dir is None, then the images are already jpgs or pngs
        # Save the image as a TIFF in the new_tiff_dir
        if save_tiff_to_original_dir and original_img_dir is not None: # Must have started with raw images
            cropped_tiff_path = os.path.join(original_img_dir, detection_cropped_name.replace('.jpg', '.TIFF'))
            crop_image_raw(full_image_tiff, cropped_tiff_path, min_x, min_y, max_x, max_y)
            copy_exif_data(cr2_file, cropped_tiff_path)
        
        if save_tiff_to_dir_output and original_img_dir is not None:
            cropped_tiff_path = os.path.join(Dirs.save_specimen_crop, detection_cropped_name.replace('.jpg', '.TIFF'))
            crop_image_raw(full_image_tiff, cropped_tiff_path, min_x, min_y, max_x, max_y)
            copy_exif_data(cr2_file, cropped_tiff_path)

        if save_XMP and original_img_dir is not None:
            path_XMP = os.path.join(original_img_dir, detection_cropped_name.replace('.jpg', '.XMP'))
            create_XMP(cr2_file, path_XMP, min_x_init, min_y_init, max_x_init, max_y_init, orientation, padding)

        # Convert the TIFF to DNG
        # https://helpx.adobe.com/camera-raw/using/adobe-dng-converter.html
        # Convert the TIFF to DNG
        # cropped_dng_path = cropped_tiff_path.replace('.TIFF', '.DNG')
        # convert_to_dng(cr2_file, cropped_dng_path)

        # JPG only
        # Since not raw, dir_images_local is the original folder location

        # if save_jpg_to_original_dir and original_img_dir is not None: # For saving jpgs to original dir
        #     cropped_jpg_path = os.path.join(dir_images_local, detection_cropped_name)
        #     detection_cropped = full_image[min_y:max_y, min_x:max_x]
        #     cv2.imwrite(cropped_jpg_path, detection_cropped)
        #     copy_exif_data(cr2_file, cropped_jpg_path)

        if save_jpg_to_dir_output: # cannot save jpgs to original dir if original images were jpgs
            detection_cropped = full_image[min_y:max_y, min_x:max_x]
            cropped_jpg_path = os.path.join(dir_images_local, detection_cropped_name)
            cv2.imwrite(os.path.join(Dirs.save_specimen_crop, detection_cropped_name), detection_cropped)
            copy_exif_data(cropped_jpg_path, os.path.join(Dirs.save_specimen_crop, detection_cropped_name))


        # if save_jpg_to_original_dir and original_img_dir is not None: # For saving jpgs to original dir
        #     params = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        #     cropped_jpg_path = os.path.join(dir_images_local, detection_cropped_name)
        #     # Perform a single crop using the minimum and maximum coordinates
        #     detection_cropped = full_image[min_y:max_y, min_x:max_x]
        #     cv2.imwrite(os.path.join(cropped_jpg_path, detection_cropped_name), detection_cropped, params)
        #     copy_exif_data(cr2_file, cropped_jpg_path)

        # if save_jpg_to_dir_output: # cannot save jpgs to original dir if original images were jpgs
        #     params = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        #     cropped_jpg_path = os.path.join(dir_images_local, detection_cropped_name)
        #     # Perform a single crop using the minimum and maximum coordinates
        #     detection_cropped = full_image[min_y:max_y, min_x:max_x]
        #     cv2.imwrite(os.path.join(Dirs.save_specimen_crop, detection_cropped_name), detection_cropped, [cv2.IMWRITE_JPEG_QUALITY, 100])
        #     copy_exif_data(cr2_file, cropped_jpg_path)

    else:
        print('Failed')



def crop_component_from_yolo_coords_VV(anno_type, Dirs, analysis, all_detections, full_image, filename, save_per_image, save_per_class, save_list):
    height = analysis['height']
    width = analysis['width']

    # Initialize a list to hold all the cropped images
    cropped_images = []

    if len(all_detections) < 1:
        print('     MAKE THIS HAVE AN EMPTY PLACEHOLDER') # TODO ###################################################################################
    else:
        for detection in all_detections:
            detection_class = detection[0]
            detection_class = set_index_for_annotation(detection_class, anno_type)

            if (detection_class in save_list) or ('save_all' in save_list):

                location = yolo_to_position_ruler(detection, height, width)
                ruler_polygon = [(location[1], location[2]), (location[3], location[2]), (location[3], location[4]), (location[1], location[4])]

                x_coords = [x for x, y in ruler_polygon]
                y_coords = [y for x, y in ruler_polygon]

                min_x, min_y = min(x_coords), min(y_coords)
                max_x, max_y = max(x_coords), max(y_coords)

                detection_cropped = full_image[min_y:max_y, min_x:max_x]
                cropped_images.append(detection_cropped)
                loc = '-'.join([str(min_x), str(min_y), str(max_x), str(max_y)])
                detection_cropped_name = '.'.join(['__'.join([filename, detection_class, loc]), 'jpg'])
                # detection_cropped_name = '.'.join([filename,'jpg'])

                # save_per_image
                if (detection_class in save_list) and save_per_image:
                    if detection_class == 'label':
                        detection_class2 = 'label_ind'
                    else:
                        detection_class2 = detection_class
                    dir_destination = os.path.join(Dirs.save_per_image, filename, detection_class2)
                    # print(os.path.join(dir_destination,detection_cropped_name))
                    validate_dir(dir_destination)
                    # cv2.imwrite(os.path.join(dir_destination,detection_cropped_name), detection_cropped)
                    
                # save_per_class
                if (detection_class in save_list) and save_per_class:
                    if detection_class == 'label':
                        detection_class2 = 'label_ind'
                    else:
                        detection_class2 = detection_class
                    dir_destination = os.path.join(Dirs.save_per_annotation_class, detection_class2)
                    # print(os.path.join(dir_destination,detection_cropped_name))
                    validate_dir(dir_destination)
                    # cv2.imwrite(os.path.join(dir_destination,detection_cropped_name), detection_cropped)
            else:
                # print(f'detection_class: {detection_class} not in save_list: {save_list}')
                pass

    # Initialize a list to hold all the acceptable cropped images
    acceptable_cropped_images = []

    for img in cropped_images:
        # Calculate the aspect ratio of the image
        aspect_ratio = min(img.shape[0], img.shape[1]) / max(img.shape[0], img.shape[1])
        # Only add the image to the acceptable list if the aspect ratio is more square than 1:8
        if aspect_ratio >= 1/8:
            acceptable_cropped_images.append(img)

    # Sort acceptable_cropped_images by area (largest first)
    acceptable_cropped_images.sort(key=lambda img: img.shape[0] * img.shape[1], reverse=True)


    # If there are no acceptable cropped images, set combined_image to None or to a placeholder image
    if not acceptable_cropped_images:
        combined_image = None  # Or a placeholder image here
    else:
    #     # Recalculate max_width and total_height for acceptable images
    #     max_width = max(img.shape[1] for img in acceptable_cropped_images)
    #     total_height = sum(img.shape[0] for img in acceptable_cropped_images)

    #     # Now, combine all the acceptable cropped images into a single image
    #     combined_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)

    #     y_offset = 0
    #     for img in acceptable_cropped_images:
    #         combined_image[y_offset:y_offset+img.shape[0], :img.shape[1]] = img
    #         y_offset += img.shape[0]
        # Start with the first image
        # Recalculate max_width and total_height for acceptable images
        max_width = max(img.shape[1] for img in acceptable_cropped_images)
        total_height = sum(img.shape[0] for img in acceptable_cropped_images)
        combined_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)

        y_offset = 0
        y_offset_next_row = 0
        x_offset = 0

        # Start with the first image
        combined_image[y_offset:y_offset+acceptable_cropped_images[0].shape[0], :acceptable_cropped_images[0].shape[1]] = acceptable_cropped_images[0]
        y_offset_next_row += acceptable_cropped_images[0].shape[0]

        # Add the second image below the first one
        y_offset = y_offset_next_row
        combined_image[y_offset:y_offset+acceptable_cropped_images[1].shape[0], :acceptable_cropped_images[1].shape[1]] = acceptable_cropped_images[1]
        y_offset_next_row += acceptable_cropped_images[1].shape[0]

        # Create a list to store the images that are too tall for the current row
        too_tall_images = []

        # Now try to fill in to the right with the remaining images
        current_width = acceptable_cropped_images[1].shape[1]

        for img in acceptable_cropped_images[2:]:
            if current_width + img.shape[1] > max_width:
                # If this image doesn't fit, start a new row
                y_offset = y_offset_next_row
                combined_image[y_offset:y_offset+img.shape[0], :img.shape[1]] = img
                current_width = img.shape[1]
                y_offset_next_row = y_offset + img.shape[0]
            else:
                # If this image fits, add it to the right
                max_height = y_offset_next_row - y_offset
                if img.shape[0] > max_height:
                    too_tall_images.append(img)
                else:
                    combined_image[y_offset:y_offset+img.shape[0], current_width:current_width+img.shape[1]] = img
                    current_width += img.shape[1]

        # Process the images that were too tall for their rows
        for img in too_tall_images:
            y_offset = y_offset_next_row
            combined_image[y_offset:y_offset+img.shape[0], :img.shape[1]] = img
            y_offset_next_row += img.shape[0]

        # Trim the combined_image to remove extra black space
        combined_image = combined_image[:y_offset_next_row]


        # save the combined image
        # if (detection_class in save_list) and save_per_class:
        dir_destination = os.path.join(Dirs.save_per_annotation_class, 'label')
        validate_dir(dir_destination)
        # combined_image_name = '__'.join([filename, detection_class]) + '.jpg'
        combined_image_name = '.'.join([filename,'jpg'])
        cv2.imwrite(os.path.join(dir_destination, combined_image_name), combined_image)

        original_image_name = '.'.join([filename,'jpg'])
        cv2.imwrite(os.path.join(Dirs.save_original, original_image_name), full_image)
        


def crop_component_from_yolo_coords(anno_type, Dirs, all_detections, full_image, filename, save_per_image, save_per_class, save_list, width, height):
    if len(all_detections) < 1:
        print('     MAKE THIS HAVE AN EMPTY PLACEHOLDER') # TODO ###################################################################################
    else:
        for detection in all_detections:
            detection_class = detection[0]
            detection_class = set_index_for_annotation(detection_class, anno_type)

            if (detection_class in save_list) or ('save_all' in save_list):

                location = yolo_to_position_ruler(detection, height, width)
                ruler_polygon = [(location[1], location[2]), (location[3], location[2]), (location[3], location[4]), (location[1], location[4])]

                x_coords = [x for x, y in ruler_polygon]
                y_coords = [y for x, y in ruler_polygon]

                min_x, min_y = min(x_coords), min(y_coords)
                max_x, max_y = max(x_coords), max(y_coords)

                detection_cropped = full_image[min_y:max_y, min_x:max_x]
                loc = '-'.join([str(min_x), str(min_y), str(max_x), str(max_y)])
                detection_cropped_name = '.'.join(['__'.join([filename, detection_class, loc]), 'jpg'])

                # save_per_image
                if (detection_class in save_list) and save_per_image:
                    dir_destination = os.path.join(Dirs.save_per_image, filename, detection_class)
                    # print(os.path.join(dir_destination,detection_cropped_name))
                    validate_dir(dir_destination)
                    cv2.imwrite(os.path.join(dir_destination,detection_cropped_name), detection_cropped)
                    
                # save_per_class
                if (detection_class in save_list) and save_per_class:
                    dir_destination = os.path.join(Dirs.save_per_annotation_class, detection_class)
                    # print(os.path.join(dir_destination,detection_cropped_name))
                    validate_dir(dir_destination)
                    cv2.imwrite(os.path.join(dir_destination,detection_cropped_name), detection_cropped)
            else:
                # print(f'detection_class: {detection_class} not in save_list: {save_list}')
                pass

def yolo_to_position_ruler(annotation, height, width):
    return ['ruler', 
        int((annotation[1] * width) - ((annotation[3] * width) / 2)), 
        int((annotation[2] * height) - ((annotation[4] * height) / 2)), 
        int(annotation[3] * width) + int((annotation[1] * width) - ((annotation[3] * width) / 2)), 
        int(annotation[4] * height) + int((annotation[2] * height) - ((annotation[4] * height) / 2))]


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    CEND      = '\33[0m'
    CBOLD     = '\33[1m'
    CITALIC   = '\33[3m'
    CURL      = '\33[4m'
    CBLINK    = '\33[5m'
    CBLINK2   = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK  = '\33[30m'
    CRED    = '\33[31m'
    CGREEN  = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE   = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE  = '\33[36m'
    CWHITE  = '\33[37m'

    CBLACKBG  = '\33[40m'
    CREDBG    = '\33[41m'
    CGREENBG  = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG   = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG  = '\33[46m'
    CWHITEBG  = '\33[47m'

    CGREY    = '\33[90m'
    CRED2    = '\33[91m'
    CGREEN2  = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2   = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2  = '\33[96m'
    CWHITE2  = '\33[97m'

    CGREYBG    = '\33[100m'
    CREDBG2    = '\33[101m'
    CGREENBG2  = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2   = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2  = '\33[106m'
    CWHITEBG2  = '\33[107m'
    CBLUEBG3   = '\33[112m'


def set_index_for_annotation(cls,annoType):
    if annoType == 'PLANT':
        if cls == 0:
            annoInd = 'Leaf_WHOLE'
        elif cls == 1:
            annoInd = 'Leaf_PARTIAL'
        elif cls == 2:
            annoInd = 'Leaflet'
        elif cls == 3:
            annoInd = 'Seed_Fruit_ONE'
        elif cls == 4:
            annoInd = 'Seed_Fruit_MANY'
        elif cls == 5:
            annoInd = 'Flower_ONE'
        elif cls == 6:
            annoInd = 'Flower_MANY'
        elif cls == 7:
            annoInd = 'Bud'
        elif cls == 8:
            annoInd = 'Specimen'
        elif cls == 9:
            annoInd = 'Roots'
        elif cls == 10:
            annoInd = 'Wood'
    elif annoType == 'ARCHIVAL':
        if cls == 0:
            annoInd = 'Ruler'
        elif cls == 1:
            annoInd = 'Barcode'
        elif cls == 2:
            annoInd = 'Colorcard'
        elif cls == 3:
            annoInd = 'Label'
        elif cls == 4:
            annoInd = 'Map'
        elif cls == 5:
            annoInd = 'Envelope'
        elif cls == 6:
            annoInd = 'Photo'
        elif cls == 7:
            annoInd = 'Attached_item'
        elif cls == 8:
            annoInd = 'Weights'
    return annoInd.lower()
# def set_yaml(path_to_yaml, value):
#     with open('file_to_edit.yaml') as f:
#         doc = yaml.load(f)

#     doc['state'] = state

#     with open('file_to_edit.yaml', 'w') as f:
#         yaml.dump(doc, f)