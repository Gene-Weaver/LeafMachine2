import os, inspect, sys, asyncio, csv, json
from multiprocessing import Manager, Process
from timeit import default_timer as timer
import pandas as pd

currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)

# from leafmachine2.downloading.utils_downloads import run_download_parallel, load_wishlist_to_tracker
from leafmachine2.downloading.utils_downloads_refactored import run_download_parallel, load_wishlist_to_tracker
from leafmachine2.machine.general_utils import bcolors
from leafmachine2.machine.email_updates import send_update
# from leafmachine2.downloading.utils_downloads import download_all_images_in_images_csv
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
    # NOTE: the asyncio has to load all the "Working on image: ..." stuff before it downloads, can take a long time if lots of multimedia.txt
    # DWC_folder_containing_records = "/media/nas/GBIF_Downloads/Cornales"
    # DWC_folder_containing_records = 'F:/test_parallel/test_cornaceae'
    is_custom_download = False
    custom_download_url_column = ""
    custom_download_filename_column = ""
    # filename_occ = "occurrence.txt"
    filename_occ = "occurrence.csv"
    filename_img = "multimedia.txt"
    filename_csv_family_counts_stem = "images_per_species"
    # n_to_download = 1000000 # PER taxonomic_level

    project_multimedia_file = None # But set this to the file name with the other project info below. None allows it to dynamically generate the file path

    DWC_min_res = 1
    DWC_max_res = 25
    do_resize = True
    n_threads = 24#24
    num_drivers = 24#12#48

    '''Magnoliopsida 10/15/24          '''
    project_download_family_splits = "/media/nas/GBIF_Downloads/Magnoliopsida_By_Family"
    project_multimedia_file = "/media/nas/GBIF_Downloads/Magnoliopsida/multimedia.txt" # This is the huge version, used for all families

    taxonomic_level = 'fullname' # family genus fullname
    use_large_file_size_methods = True # False for < 1 GB csv files           
    wishlist_csv_path = "/media/nas/GBIF_Downloads/big_tree_names_USE.csv"
    failure_csv_path = "/media/nas/GBIF_Downloads/Magnoliopsida_By_Family/failed_downloads.csv"
    n_to_download = 50 # PER taxonomic_level
    min_occ_cutoff = 100 # min number of occ to consider the taxa


    # Initialize an empty list to store the full paths
    project_download_list = []

    # Iterate over all items in the specified directory
    for folder in os.listdir(project_download_family_splits):
        # Construct the full path for the current folder
        full_path = os.path.join(project_download_family_splits, folder)
        
        # Check if the current item is a directory
        if os.path.isdir(full_path):
            # Add the full path to the list
            project_download_list.append(full_path)


    for path in project_download_list:
        print(f"\n\n\nWorking on {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"The path {path} does not exist.")
        
        if not os.path.exists(failure_csv_path):
            with open(failure_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Filename", "URL", "Reason"])  # Headers for the CSV file

        
        # Define the path for the tracking JSON file
        tracking_file = os.path.join(path, 'download_is_complete.json')
        # Check if the tracking file exists and its status
        if os.path.exists(tracking_file):
            with open(tracking_file, 'r') as file:
                status = json.load(file)
            if status.get("finished", 0) >= 1:
                print(f"    Skipping {path} as it is already marked as completed.")
                continue
        else:
            # Create the tracking file with "finished": 0 if it doesn't exist
            with open(tracking_file, 'w') as file:
                json.dump({"finished": 0}, file)

        send_update(path, "Downloading ---", pc="quadro")

        if is_new_update_available():
            print("A new stable Chrome update is available!")
            send_update(path, "UPDATING CHROME ---", pc="quadro")
            download_and_install_chrome()
            send_update(path, "CHROME UPDATED! ---", pc="quadro")
        else:
            print("Chrome is up to date.")

        current_family = os.path.basename(path)
        filename_csv_family_counts_core = current_family
        
        start_time = timer()  # Start timer

        dir_destination_images = os.path.join(path, 'img')
        dir_destination_csv = os.path.join(path, 'occ')
        cfg = {
            'dir_home': path,
            'dir_destination_images': dir_destination_images,
            'dir_destination_csv': dir_destination_csv,
            'failure_csv_path': failure_csv_path,
            'filename_occ': filename_occ,
            'filename_img': filename_img, 
            'filename_csv_family_counts_stem': filename_csv_family_counts_stem,
            'min_occ_cutoff': min_occ_cutoff,
            'project_multimedia_file': project_multimedia_file,
            'filename_combined': 'combined_occ_img_downloaded.csv',
            'n_to_download': n_to_download,
            'taxonomic_level': taxonomic_level,
            'use_large_file_size_methods': use_large_file_size_methods,
            'MP_low': DWC_min_res,
            'MP_high': DWC_max_res,
            'do_resize': do_resize,
            'n_threads': n_threads,
            'num_drivers': num_drivers,
            'ignore_banned_herb': True,
            'banned_url_stems': ['www.herbariumhamburgense.de', 'imagens4.jbrj.gov.br', 'imagens1.jbrj.gov.br', 'plant.depo.msu.ru','depo.msu.ru','msu.ru',
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
        wishlist_tracker = load_wishlist_to_tracker(taxonomic_level, wishlist_csv_path) # Shared dictionary for tracking wishlist taxa [family, genus, species, genus_species, fullname], uses taxonomic_level
        download_tracker = manager.dict()  # Shared dictionary for tracking downloads
        completed_tracker = manager.list()  # Shared dictionary for tracking downloads
        banned_url_tracker = manager.list(cfg['banned_url_stems'])  # Initialize with banned_url_stems
        banned_url_counts_tracker = manager.dict()  # Shared dictionary for tracking banned URL counts


        process = Process(target=run_download_parallel, args=(cfg, wishlist_tracker, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker, filename_csv_family_counts_core))
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

        # Update the tracking JSON file to indicate completion
        with open(tracking_file, 'w') as file:
            json.dump({"finished": 1}, file)

        send_update(path, "Downloading Complete! ---", pc="quadro")

    print(f"{bcolors.CGREENBG}ALL DOWNLOADS COMPLETED{bcolors.ENDC}")

