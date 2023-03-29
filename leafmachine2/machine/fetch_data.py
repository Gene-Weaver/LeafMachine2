from __future__ import annotations
import os, yaml, shutil
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import urllib.request
from tqdm import tqdm
import subprocess

VERSION = 'v-2-1'

def fetch_data(logger, dir_home, cfg_file_path):
    logger.name = 'Fetch Data'
    ready_to_use = False
    do_fetch = True
    current = ''.join(['release_', VERSION])

    # Make sure weights are present
    if os.path.isfile(os.path.join(dir_home,'bin','version.yml')):
        ver = load_version(dir_home)
        
        if ver['version'] == VERSION:
            if current in os.listdir(os.path.join(dir_home,'bin')): # The release  dir is present
                do_fetch = False
                ready_to_use = True
                logger.warning(f"Version file --- {os.path.join(dir_home,'bin','version.yml')}")
                logger.warning(f"Current version --- {ver['version']}")
                logger.warning(f"Last updated --- {ver['last_update']}")
            else:  # right version, no release dir yet
                do_fetch = True
                logger.warning(f"--------------------------------")
                logger.warning(f"   Downloading data files...    ")
                logger.warning(f"--------------------------------")
                logger.warning(f"Version file --- {os.path.join(dir_home,'bin','version.yml')}")
                logger.warning(f"Current version --- {ver['version']}")
                logger.warning(f"Last updated --- {ver['last_update']}")
        else:
            do_fetch = True
            logger.warning(f"--------------------------------")
            logger.warning(f"   Out of date...               ")
            logger.warning(f"   Downloading data files...    ")
            logger.warning(f"--------------------------------")
            logger.warning(f"Version file --- {os.path.join(dir_home,'bin','version.yml')}")
            logger.warning(f"Current version --- {ver['version']}")
            logger.warning(f"Last updated --- {ver['last_update']}")

        
    else:
        do_fetch = True
        logger.warning(f"--------------------------------")
        logger.warning(f"   Missing version.yml...       ")
        logger.warning(f"   Downloading data files...    ")
        logger.warning(f"--------------------------------")
        logger.warning(f"Version file --- {os.path.join(dir_home,'bin','version.yml')}")
        logger.warning(f"Current version --- {ver['version']}")
        logger.warning(f"Last updated --- {ver['last_update']}")


    if do_fetch:
        logger.warning(f"Fetching files for version --> {ver['version']}")
        path_release = get_weights(dir_home, current, logger)
        if path_release is not None:
            logger.warning(f"Data download successful. Unzipping...")
            move_data_to_home(path_release, dir_home, logger)
            ready_to_use = True
            logger.warning(f"--------------------------------")
            logger.warning(f"   LeafMachine2 is up to date   ")
            logger.warning(f"--------------------------------")

    else:
        logger.warning(f"--------------------------------")
        logger.warning(f"   LeafMachine2 is up to date   ")
        logger.warning(f"--------------------------------")

    return ready_to_use



def get_weights(dir_home, current, logger):
    
    try:
        path_zip = os.path.join(dir_home,'bin',current)
        zipurl = ''.join(['https://leafmachine.org/LM2/', current,'.zip'])
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}

        req = urllib.request.Request(url=zipurl, headers=headers)

        # Get the file size from the Content-Length header
        with urllib.request.urlopen(req) as url_response:
            file_size = int(url_response.headers['Content-Length'])

        # Download the ZIP file from the URL with progress bar
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, total=file_size) as pbar:
            with urllib.request.urlopen(req) as url_response:
                with open(current + '.zip', 'wb') as file:
                    while True:
                        chunk = url_response.read(4096)
                        if not chunk:
                            break
                        file.write(chunk)
                        pbar.update(len(chunk))

        # Extract the contents of the ZIP file to the current directory
        zipfilename = current + '.zip'
        with ZipFile(zipfilename, 'r') as zip_file:
            zip_file.extractall(os.path.join(dir_home,'bin'))

        print(f"{bcolors.CGREENBG2}Data extracted to {path_zip}{bcolors.ENDC}")
        logger.warning(f"Data extracted to {path_zip}")

        return path_zip
    except Exception as e:
        print(f"{bcolors.CREDBG2}ERROR --- Could not download or extract machine learning models\n{e}{bcolors.ENDC}")
        logger.warning(f"ERROR --- Could not download or extract machine learning models")
        logger.warning(f"ERROR --- {e}")
        return None
        

def load_version(dir_home):
    try:
        with open(os.path.join(dir_home,'bin',"version.yml"), "r") as ymlfile:
            ver = yaml.full_load(ymlfile)
    except:
        with open(os.path.join(os.path.dirname(os.path.dirname(dir_home)),'bin',"version.yml"), "r") as ymlfile:
            ver = yaml.full_load(ymlfile)
    return ver

def move_data_to_home(path_release, dir_home, logger):
    path_list_file = os.path.join(path_release, 'path_list.yml')

    with open(path_list_file, 'r') as file:
        path_list = yaml.safe_load(file)

    paths = {
        'path_ruler_classifier': os.path.join(dir_home, *path_list['path_ruler_classifier'].split('___')),
        'path_ruler_binary_classifier': os.path.join(dir_home, *path_list['path_ruler_binary_classifier'].split('___')),
        'path_ruler_classifier_binary_classes': os.path.join(dir_home, *path_list['path_ruler_classifier_binary_classes'].split('___')),
        'path_ruler_classifier_ruler_classes': os.path.join(dir_home, *path_list['path_ruler_classifier_ruler_classes'].split('___')),
        'path_DocEnTR': os.path.join(dir_home, *path_list['path_DocEnTR'].split('___')),
        'path_ACD': os.path.join(dir_home, *path_list['path_ACD'].split('___')),
        'path_PCD': os.path.join(dir_home, *path_list['path_PCD'].split('___')),
        'path_landmarks': os.path.join(dir_home, *path_list['path_landmarks'].split('___')),
        'path_YOLO': os.path.join(dir_home, *path_list['path_YOLO'].split('___')),
        'path_segment': os.path.join(dir_home, *path_list['path_segment'].split('___')),
    }


    ### Ruler classifier
    source_file = os.path.join(path_release, 'ruler_classifier', 'ruler_classifier_38classes_v-1.pt')
    destination_dir = paths['path_ruler_classifier']
    os.makedirs(destination_dir, exist_ok=True)
    try_move(logger, source_file, destination_dir )
    

    source_file = os.path.join(path_release, 'ruler_classifier', 'model_scripted_resnet_720_withCompression.pt')
    destination_dir = paths['path_ruler_binary_classifier']
    os.makedirs(destination_dir, exist_ok=True)
    try_move(logger, source_file, destination_dir )

    source_file = os.path.join(path_release, 'ruler_classifier', 'binary_classes.txt')
    destination_dir = paths['path_ruler_classifier_binary_classes']
    os.makedirs(destination_dir, exist_ok=True)
    try_move(logger, source_file, destination_dir )

    source_file = os.path.join(path_release, 'ruler_classifier', 'ruler_classes.txt')
    destination_dir = paths['path_ruler_classifier_ruler_classes']
    os.makedirs(destination_dir, exist_ok=True)
    try_move(logger, source_file, destination_dir )


    ### Ruler segmentation
    source_file = os.path.join(path_release, 'ruler_segment', 'small_256_8__epoch-10.pt')
    destination_dir = paths['path_DocEnTR']
    os.makedirs(destination_dir, exist_ok=True)
    try_move(logger, source_file, destination_dir )


    ### ACD
    source_file = os.path.join(path_release, 'acd', 'best.pt')
    destination_dir = paths['path_ACD']
    os.makedirs(destination_dir, exist_ok=True)
    try_move(logger, source_file, destination_dir )


    ### PCD
    source_file = os.path.join(path_release, 'pcd', 'best.pt')
    destination_dir = paths['path_PCD']
    os.makedirs(destination_dir, exist_ok=True)
    try_move(logger, source_file, destination_dir )


    ### Landmarks
    source_file = os.path.join(path_release, 'landmarks', 'best.pt')
    destination_dir = paths['path_landmarks']
    os.makedirs(destination_dir, exist_ok=True)
    try_move(logger, source_file, destination_dir )


    ### YOLO
    source_file = os.path.join(path_release, 'YOLO', 'yolov5x6.pt')
    destination_dir = paths['path_YOLO']
    os.makedirs(destination_dir, exist_ok=True)
    try_move(logger, source_file, destination_dir )


    ### Segmentation
    source_file = os.path.join(path_release, 'segmentation', 'model_final.pth')
    destination_dir = paths['path_segment']
    os.makedirs(destination_dir, exist_ok=True)
    try_move(logger, source_file, destination_dir )

def git_pull_no_clean():
    # Define the git command to run
    git_cmd = ['git', 'pull', '--no-clean']

    # Run the git command using subprocess
    result = subprocess.run(git_cmd, capture_output=True, text=True)

    # Check if the command was successful
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(result.stderr)

def try_move(logger, source_file, destination_dir ):
    try:
        # Try to move the file using shutil.move()
        shutil.move(source_file, destination_dir, copy_function=shutil.copy2)
        logger.warning(f"{source_file}\nmoved to\n{destination_dir}")
        print(f"{source_file}\nmoved to\n{destination_dir}")
    except FileExistsError:
        # If the file already exists in the destination directory, skip it
        logger.warning(f"Already exists in\n{destination_dir}.\nSkipping...")
        print(f"Already exists in\n{destination_dir}.\nSkipping...")
        pass
    except Exception as e:
        # Catch any other exceptions that might occur
        logger.warning(f"[ERROR] occurred while moving:\n{source_file}:\n{str(e)}")
        print(f"ERROR occurred while moving:\n{source_file}:\n{str(e)}")

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
    CGREENBG2  = '\33[102m'
    CREDBG2    = '\33[101m'
    CWHITEBG2  = '\33[107m'

if __name__ == '__main__':
    git_pull_no_clean()