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
# from leafmachine2.downloading.utils_downloads import download_all_images_in_images_csv


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
    filename_occ = "occurrence.txt"
    filename_img = "multimedia.txt"
    n_to_download = 1000000 # PER taxonomic_level



    DWC_min_res = 1
    DWC_max_res = 25
    do_resize = True
    n_threads = 24#24
    num_drivers = 24#12#48

    # project_download_list = ['/media/nas/GBIF_Downloads/Cornales_wCoords/Cornaceae']
    
    # project_download_list = [
    #     "/media/nas/GBIF_Downloads/Cornales/Cornaceae", # downloaded this, but 13,000 were still in queue, mostly harvard and nt2ark # UPDATE: fixed a bug that re-added last try urls back to the queue... oops
    #     "/media/nas/GBIF_Downloads/Cornales/Hydrangeaceae",
    #     "/media/nas/GBIF_Downloads/Cornales/Loasaceae",
    #     "/media/nas/GBIF_Downloads/Cornales/Nyssaceae",
    #     ]
    # project_download_list = ['F:/test_parallel/test1','F:/test_parallel/test_cornaceae', ]
    # taxonomic_level = 'fullname' # family genus fullname
    # use_large_file_size_methods = False # False for < 1 GB csv files

    '''Fagales --- all --- DONE June 19, 2024'''
    # project_download_list = [
    #     "/media/nas/GBIF_Downloads/Fagales/Casuarinaceae", # Not suitable for LM2?
    #     "/media/nas/GBIF_Downloads/Fagales/Ticodendraceae", # round 1
    #     "/media/nas/GBIF_Downloads/Fagales/Nothofagaceae", # round 1
    #     "/media/nas/GBIF_Downloads/Fagales/Myricaceae", # round 1
    #     "/media/nas/GBIF_Downloads/Fagales/Juglandaceae", # round 1
    #     "/media/nas/GBIF_Downloads/Fagales/Fagaceae", # round 2
    #     "/media/nas/GBIF_Downloads/Fagales/Betulaceae", # round 3
    #     ]
    # taxonomic_level = 'fullname' # family genus fullname
    # use_large_file_size_methods = False # False for < 1 GB csv files

    '''Magnoliales --- all --- DONE June 10, 2024''' 
    # project_download_list = [
    #     "/media/nas/GBIF_Downloads/Magnoliales/Myristicaceae", # round 4
    #     "/media/nas/GBIF_Downloads/Magnoliales/Magnoliaceae", # round 4
    #     "/media/nas/GBIF_Downloads/Magnoliales/Degeneriaceae", # round 4
    #     "/media/nas/GBIF_Downloads/Magnoliales/Eupomatiaceae", # round 4
    #     "/media/nas/GBIF_Downloads/Magnoliales/Himantandraceae", # round 4
    #     "/media/nas/GBIF_Downloads/Magnoliales/Annonaceae", # round 4
    # ]
    # taxonomic_level = 'fullname' # family genus fullname
    # use_large_file_size_methods = False # False for < 1 GB csv files

    '''Dipsacales --- all --- DONE June 20, 2024'''
    # project_download_list = [
    #     "/media/nas/GBIF_Downloads/Dipsacales/Caprifoliaceae", 
    #     "/media/nas/GBIF_Downloads/Dipsacales/Viburnaceae", 
    #     ]
    # taxonomic_level = 'fullname' # family genus fullname
    # use_large_file_size_methods = False # False for < 1 GB csv files
    
    '''Ericales --- all --- '''
    # project_download_list = [
    #     "/media/nas/GBIF_Downloads/Ericales/Actinidiaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Balsaminaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Clethraceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Cyrillaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Diapensiaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Ebenaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Ericaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Fouquieriaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Lecythidaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Marcgraviaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Mitrastemonaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Pentaphylacaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Polemoniaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Primulaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Roridulaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Sapotaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Sarraceniaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Sladeniaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Styracaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Symplocaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Tetrameristaceae", 
    #     "/media/nas/GBIF_Downloads/Ericales/Theaceae", 
    #     ]
    # taxonomic_level = 'fullname' # family genus fullname
    # use_large_file_size_methods = False # False for < 1 GB csv files

    '''Moraceae 10/11/24   DONE'''
    # project_download_list = [
    #     "/media/nas/GBIF_Downloads/Moraceae", 
    #     ]
    # taxonomic_level = 'fullname' # family genus fullname
    # use_large_file_size_methods = False # False for < 1 GB csv files

    '''Urticaceae 10/15/24   DONE'''
    # project_download_list = [
    #     "/media/nas/GBIF_Downloads/Urticaceae", 
    #     ]
    # taxonomic_level = 'fullname' # family genus fullname
    # use_large_file_size_methods = False # False for < 1 GB csv files


    '''Populus_Populus_tremuloides 10/29/24 for Dave Tank, with coordinates   DONE'''
    project_download_list = [
        "/media/nas/GBIF_Downloads/Tank/Salicaceae_Populus_tremuloides2", 
        ]
    taxonomic_level = 'fullname' # family genus fullname
    use_large_file_size_methods = True # False for < 1 GB csv files
    failure_csv_path = "/media/nas/GBIF_Downloads/Tank/Salicaceae_Populus_tremuloides2/failed_downloads.csv"
    wishlist_csv_path = None
    n_to_download = 50 # PER taxonomic_level
    

    '''Magnoliopsida 10/15/24          '''
    # project_download_list = [
    #     "/media/nas/GBIF_Downloads/Magnoliopsida", 
    #     ]
    # taxonomic_level = 'fullname' # family genus fullname
    # use_large_file_size_methods = True # False for < 1 GB csv files           
    # wishlist_csv_path = "/media/nas/GBIF_Downloads/big_tree_names_USE.csv"
    # failure_csv_path = "/media/nas/GBIF_Downloads/Magnoliopsida/failed_downloads.csv"
    # n_to_download = 10 # PER taxonomic_level

    for path in project_download_list:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path {path} does not exist.")
        
        if not os.path.exists(failure_csv_path):
            with open(failure_csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Filename", "URL", "Reason"])  # Headers for the CSV file

        
        start_time = timer()  # Start timer

        dir_destination_images = os.path.join(path, 'img2')
        dir_destination_csv = os.path.join(path, 'occ2')
        cfg = {
            'dir_home': path,
            'dir_destination_images': dir_destination_images,
            'dir_destination_csv': dir_destination_csv,
            'failure_csv_path': failure_csv_path,
            'filename_occ': filename_occ,
            'filename_img': filename_img, 
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
        wishlist_tracker = load_wishlist_to_tracker(taxonomic_level, wishlist_csv_path) # Shared dictionary for tracking wishlist taxa [family, genus, species, genus_species, fullname], uses taxonomic_level
        download_tracker = manager.dict()  # Shared dictionary for tracking downloads
        completed_tracker = manager.list()  # Shared dictionary for tracking downloads
        banned_url_tracker = manager.list(cfg['banned_url_stems'])  # Initialize with banned_url_stems
        banned_url_counts_tracker = manager.dict()  # Shared dictionary for tracking banned URL counts


        process = Process(target=run_download_parallel, args=(cfg, wishlist_tracker, download_tracker, completed_tracker, banned_url_tracker, banned_url_counts_tracker))
        process.start()
        process.join()

        # Save download_tracker to CSV
        save_download_tracker_to_csv(download_tracker, os.path.join(path, 'download_info2.csv'))

        save_banned_url_counts_tracker_to_csv(banned_url_counts_tracker, os.path.join(path, 'banned_url_counts_tracker2.csv'))

        # Count total number of downloaded images
        n_total = count_total_downloaded_images(download_tracker)

        # Measure the download time
        download_time = timer() - start_time

        # Append the loop's download time and n_total to the summary text file
        with open(os.path.join(path,'download_summary2.txt'), 'a') as summary_file:
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

