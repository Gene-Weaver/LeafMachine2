import os, inspect, sys
currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)
try:
    from leafmachine2.machine.utils_GBIF import download_all_images_in_images_csv, get_cfg_from_full_path
except:
    from utils_GBIF import download_all_images_in_images_csv, get_cfg_from_full_path


'''
This script attempts to download all images (every row) that are in the provided images.csv file.
This means that you have either pruned an images.csv file or that you wish to download every possible image.

*** USAGE *** 
Use the config_download_from_GBIF_all_images_in_file.yml file to set all parameters, then run this script

images.csv files used here should be in a standard Darwin Core (DWC) format.

This script is meant for Darwin Core files retrieved from GBIF.org
    *** Note: not all DWC files have the same column names
    *** Note: This script should adjust to slightly different column names, but errors may occur with non-GBIF files

*** NOTES ***
There are different scripts in the LeafMachine2/leafmachine2/machine folder to download images from:
    1) non-GBIF sources i.e. SEINet, SERNEC
    2) provide a list of species/genera/families and retrive all available images from GBIF 
    3) provide a list of species/genera/families and retrive a custom set of images that are available on GBIF
'''
def download_all_images_from_GBIF_LM2(dir_LM2, mode):
    dir_current_config = os.path.join(dir_LM2,'configs')
    if mode in ['all','All','ALL','a','A']:
        path_cfg = os.path.join(dir_current_config,'config_download_from_GBIF_all_images_in_file.yml')
    elif mode in ['filter','Filter','FILTER','f','F']:
        path_cfg = os.path.join(dir_current_config,'config_download_from_GBIF_all_images_in_filter.yml')
    cfg = get_cfg_from_full_path(path_cfg)

    # Run Download
    download_all_images_in_images_csv(cfg)
    return cfg

if __name__ == '__main__':
    dir_LM2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dir_current_config = os.path.join(dir_LM2,'configs')
    path_cfg = os.path.join(dir_current_config,'config_download_from_GBIF_all_images_in_file.yml')
    cfg = get_cfg_from_full_path(path_cfg)

    # Run Download
    download_all_images_in_images_csv(cfg)