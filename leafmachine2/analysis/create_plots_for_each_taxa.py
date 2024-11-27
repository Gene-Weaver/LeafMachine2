import os, glob, shutil, sys
import sqlite3
import dask.dataframe as dd
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import h5py
from multiprocessing import Pool
import numpy as np
import phate
import matplotlib.pyplot as plt
import seaborn as sns
import scprep

currentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from leafmachine2.ect_methods.utils_ect import preprocessing, ingest_DWC_large_files, save_to_hdf5, load_from_hdf5
from leafmachine2.ect_methods.leaf_ect import LeafECT
from leafmachine2.ect_methods.utils_PHATE import run_phate_simple, run_phate_with_shapes, run_phate_grid_by_taxa, run_phate_heatmap, compute_phate
from leafmachine2.analysis.manage_LM2_data import LM2DataVault, collate_ect_data

'''
This is for local PC testing of plotting methods
All new functions should be incorporated into 
manage_LM2_data.py 
OR
one of the utils___.py

'''
def run_PHATE_on_everything_h5(h5_file_path, bin_by_class="family"):
    # Extract the parent directory of the h5 file path
    save_path = os.path.join(os.path.dirname(h5_file_path), 'PHATE')

    save_path_PHATE_scores_2D = os.path.join(save_path, 'PHATE_scores_2D.npz')
    save_path_PHATE_scores_3D = os.path.join(save_path, 'PHATE_scores_3D.npz')
    save_path_PHATE_3D_heatmaps = os.path.join(save_path, 'PHATE_3D_heatmaps')

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path_PHATE_3D_heatmaps, exist_ok=True)

    # Load ECT data and labels
    ect_data, group_labels, shapes, component_names = load_from_hdf5(h5_file_path)

    phate_scores_2D = compute_phate(ect_data, save_path_PHATE_scores_2D, n_components=2)
    phate_scores_3D = compute_phate(ect_data, save_path_PHATE_scores_3D, n_components=3)

    run_phate_grid_by_taxa(ect_data, group_labels, shapes, save_path, phate_scores_2D, bin_by_class)

    run_phate_with_shapes(ect_data, group_labels, shapes, component_names, save_path, phate_scores_2D, bin_by_class)

    run_phate_simple(ect_data, group_labels, shapes, component_names, save_path, phate_scores_2D, phate_scores_3D, bin_by_class)

    run_phate_heatmap(phate_scores_3D, group_labels, save_path_PHATE_3D_heatmaps, bin_by_class)









if __name__ == '__main__':
    run_CLEAN = False
    run_ingest = False

    run_ECT = False
    run_collate_ECTs = False 

    run_PHATE_on_family = True
    run_PHATE_on_everything = True

    # file_path = "C:/Users/Will/Downloads/GBIF_DetailedSample_50Spp/LM2_2024_09_18__07-52-47/Data/Measurements/LM2_2024_09_18__07-52-47_MEASUREMENTS.csv"
    # outline_path = "C:/Users/Will/Downloads/GBIF_DetailedSample_50Spp/LM2_2024_09_18__07-52-47/Keypoints/Simple_Labels"
    db_path = 'C:/Users/Will/Downloads/GBIF_DetailedSample_50Spp/LM2_2024_09_18__07-52-47/LM2_Data_ANALYSIS.db'

    # Directories that contain the desired LM2* _MEASUREMENTS.csv files
    input_dirs = [
        'C:/Users/Will/Downloads/GBIF_DetailedSample_50Spp/LM2_2024_09_18__07-52-47'
    ]  

    if run_CLEAN:
        clean_files = []

        # Preprocessing step for each directory
        for dir_path in input_dirs:
            # Look for any file with the pattern LM2* _MEASUREMENTS.csv or LM2_MEASUREMENTS.csv
            measurements_pattern = os.path.join(dir_path, 'Data', 'Measurements', 'LM2*_MEASUREMENTS.csv')
            matched_files = glob.glob(measurements_pattern)

            # Also check if a file is already named LM2_MEASUREMENTS.csv
            lm2_measurements_file = os.path.join(dir_path, 'Data', 'Measurements', 'LM2_MEASUREMENTS.csv')
            
            # If the file already exists with the correct name, no need to rename
            if os.path.exists(lm2_measurements_file):
                data_path = lm2_measurements_file
            elif matched_files:
                # Take the first match if there's a pattern match and rename it to LM2_MEASUREMENTS.csv
                original_file_path = matched_files[0]
                data_path = os.path.join(dir_path, 'Data', 'Measurements', 'LM2_MEASUREMENTS.csv')
                shutil.copy(original_file_path, data_path)  # This will copy the file, keeping the original intact
            else:
                print(f"No measurements file found in {dir_path}, skipping.")
                continue

            # Define other necessary paths for preprocessing
            outline_path = os.path.join(dir_path, 'Keypoints', 'Simple_Labels')
            path_figure = os.path.join(dir_path, 'Data', 'Measurements', 'CF_Plot_Disagreement.png')

            # Generate the clean file path
            clean_file_path = data_path.replace("LM2_MEASUREMENTS.csv", "LM2_MEASUREMENTS_CLEAN.csv")

            # Preprocess the file
            print(f"Working on {data_path}")
            cleaned_df = preprocessing(data_path, outline_path, 
                                    show_CF_plot=False, show_first_raw_contour=False, show_df_head=False, is_Thais=False,
                                    path_figure=path_figure)
            
            # Save the cleaned dataframe
            clean_files.append(clean_file_path)


    # Ingest the cleaned files into the vault
    vault = LM2DataVault(db_path)

    if run_ingest:
        vault.ingest_files(clean_files)

    if run_ECT:
        vault.store_ect_data(input_dirs, num_workers=6)
    
    
    # vault.update_unique_values_for_whole_db()

    if run_collate_ECTs:
        output_h5_path = 'C:/Users/Will/Downloads/GBIF_DetailedSample_50Spp/LM2_2024_09_18__07-52-47/collated_ect_data.h5'
        output_npz_path = 'C:/Users/Will/Downloads/GBIF_DetailedSample_50Spp/LM2_2024_09_18__07-52-47/collated_ect_data.npz'

        collate_ect_data(input_dirs, output_h5_path, output_npz_path)   


    if run_PHATE_on_family:
        for dir_path in input_dirs:
            
            
            family_ECT_path = os.path.join(dir_path, 'Data', 'Measurements', f'{os.path.basename(os.path.dirname(dir_path))}_combined_ECT.h5')
            
            run_PHATE_on_everything_h5(family_ECT_path,
                                bin_by_class="fullname")
        
    if run_PHATE_on_everything:
        run_PHATE_on_everything_h5('C:/Users/Will/Downloads/GBIF_DetailedSample_50Spp/LM2_2024_09_18__07-52-47/collated_ect_data.h5',
                                bin_by_class="fullname")
        