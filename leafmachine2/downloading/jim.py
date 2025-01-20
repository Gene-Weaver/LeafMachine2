import os, inspect, sys, asyncio, csv, json
from multiprocessing import Manager, Process
from timeit import default_timer as timer
import pandas as pd

currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)

from leafmachine2.downloading.utils_downloads_refactored import run_download_parallel, load_wishlist_to_tracker
from leafmachine2.machine.general_utils import bcolors
import platform
if platform.system() == "Windows":
    from leafmachine2.downloading.autoupdate_chrome_windows import is_new_update_available, download_and_install_chrome
else:
    from leafmachine2.downloading.autoupdate_chrome import is_new_update_available, download_and_install_chrome

'''
Despite the file name this is the main script used to download large GBIF datasets
'''

def save_download_tracker_to_csv(download_tracker, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['fullname', 'count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for key, value in download_tracker.items():
            writer.writerow({'fullname': key, 'count': value})

def save_banned_url_counts_tracker_to_csv(banned_url_counts_tracker, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['url', 'count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for key, value in banned_url_counts_tracker.items():
            writer.writerow({'url': key, 'count': value})

def count_total_downloaded_images(download_tracker):
    return sum(download_tracker.values())

if __name__ == '__main__':
        # Initialize variables for testing
    is_custom_download = False
    custom_download_url_column = ""
    custom_download_filename_column = ""
    filename_occ = "occurrence.txt"
    filename_img = "multimedia.txt"
    filename_csv_family_counts_stem = "images_per_species"
    n_to_download = 1000000 # This just means you want all the images
    DWC_min_res = 1   # This is the minimum megapixel resolution that will be saved
    DWC_max_res = 25  # This is the maximum megapixel resolution that will be saved, larger images will be resized to this
    do_resize = True
    n_threads = 24
    num_drivers = 12
    taxonomic_level = 'fullname' # family genus fullname


    '''List of the paths to your Hackelia folders'''
    project_download_list = [
        # "FULLPATHGOESHERE/Hackelia_GBIF/Boraginaceae_Hackelia_Opiz", 
        # "FULLPATHGOESHERE/Hackelia_GBIF/Boraginaceae_Hackelia_Opiz_ex_Berchtold", 
        # "D:/T_Downloads/Boraginaceae_Hackelia_Opiz", 
        "D:/T_Downloads/Boraginaceae_Hackelia_Opiz_ex_Berchtold", 
        ]
    failure_csv_path = "D:/T_Downloads/failed_downloads.csv"

    for path in project_download_list:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path {path} does not exist.")
        
        if not os.path.exists(failure_csv_path):
            with open(failure_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Filename", "URL", "Reason"])  # Headers for the CSV file

        ### Autoupdate chrome (comment out if you are going to do it manually. You might need admin access if on University computers)
        if is_new_update_available():
            print("A new stable Chrome update is available!")
            download_and_install_chrome()
        else:
            print("Chrome is up to date.")
        
        start_time = timer()  # Start timer

        dir_destination_images = os.path.join(path, 'img')
        dir_destination_csv = os.path.join(path, 'occ')
        cfg = {
            'dir_home': path,
            'min_occ_cutoff': 1,
            'dir_destination_images': dir_destination_images,
            'dir_destination_csv': dir_destination_csv,
            'failure_csv_path': failure_csv_path,
            'filename_occ': filename_occ,
            'filename_img': filename_img, 
            'project_multimedia_file': None,
            'filename_combined': 'combined_occ_img_downloaded.csv',
            'filename_csv_family_counts_stem': filename_csv_family_counts_stem,
            'n_to_download': n_to_download,
            'taxonomic_level': taxonomic_level,
            'use_large_file_size_methods': True,
            'MP_low': DWC_min_res,
            'MP_high': DWC_max_res,
            'do_resize': do_resize,
            'n_threads': n_threads,
            'num_drivers': num_drivers,
            'ignore_banned_herb': True,
            'banned_url_stems': ['www.herbariumhamburgense.de', 'imagens4.jbrj.gov.br', 'imagens1.jbrj.gov.br', 
              'arbmis.arcosnetwork.org', '128.171.206.220', 'ia801503.us.archive.org', 'procyon.acadiau.ca',
              'www.inaturalist.org'],
            'is_custom_file': is_custom_download,
            'custom_url_column_name': custom_download_url_column,
            'custom_download_filename_column': custom_download_filename_column,
            'col_url': custom_download_url_column,
            'col_name': custom_download_filename_column,
        }

        # Change the download folder if requried
        # cfg['dir_destination_images'] = os.path.join(path, 'img_subset')

        manager = Manager()
        wishlist_tracker = None
        download_tracker = manager.dict()  # Shared dictionary for tracking downloads
        completed_tracker = manager.list()  # Shared dictionary for tracking downloads
        banned_url_tracker = manager.list(cfg['banned_url_stems'])  # Initialize with banned_url_stems
        banned_url_counts_tracker = manager.dict()  # Shared dictionary for tracking banned URL counts


        process = Process(target=run_download_parallel, args=(cfg, wishlist_tracker, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker))
        process.start()
        process.join()

        # Save download_tracker to CSV
        save_download_tracker_to_csv(download_tracker, os.path.join(path, 'download_info.csv'))

        save_banned_url_counts_tracker_to_csv(banned_url_counts_tracker, os.path.join(path, 'banned_url_counts_tracker.csv'))

        # Count total number of downloaded images
        n_total = count_total_downloaded_images(download_tracker)

        # Measure the download time
        download_time = timer() - start_time

        # Append the loop's download time and n_total to the summary text file
        with open(os.path.join(path,'download_summary.txt'), 'a') as summary_file:
            summary_file.write(
                f"Path: {path}\n"
                f"Download Time: {download_time:.2f} seconds\n"
                f"Download Time: {download_time / 60:.2f} minutes\n"
                f"Download Time: {download_time / 3600:.2f} hours\n"
                f"Total Downloads: {n_total}\n"
                f"Banned URL Tracker: {list(banned_url_tracker)}\n"  
                f"Configuration:\n{json.dumps(cfg, indent=4)}\n\n"
            )
        print(f"{bcolors.CGREENBG}Downloaded [{n_total}] images for {os.path.basename(os.path.normpath(path))}{bcolors.ENDC}")

    print(f"{bcolors.CGREENBG}ALL DOWNLOADS COMPLETED{bcolors.ENDC}")

