import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from tqdm import tqdm  # Standard tqdm for scripts
from sklearn.manifold import TSNE
from multiprocessing import Pool, Manager
import seaborn as sns
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import h5py

currentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parentdir = os.path.dirname(currentdir)
try:
    from leafmachine2.ect_methods.utils_ect import preprocessing, plot_confidence_ellipse_tsne, load_from_hdf5, save_to_hdf5
    from leafmachine2.ect_methods.utils_tsne import plot_tSNE, plot_tSNE_by_taxa, plot_tsne_hull_boundary
    from leafmachine2.ect_methods.utils_PHATE import run_phate_simple, run_phate_dimensionality_comparison, run_phate_with_shapes, run_phate_grid_by_taxa
    from leafmachine2.ect_methods.leaf_ect import LeafECT
except:
    from utils_tsne import plot_tSNE, plot_tSNE_by_taxa, plot_tsne_hull_boundary
    from utils_ect import preprocessing, plot_confidence_ellipse_tsne, load_from_hdf5, save_to_hdf5
    from utils_PHATE import run_phate_simple, run_phate_dimensionality_comparison, run_phate_with_shapes, run_phate_grid_by_taxa
    from leaf_ect import LeafECT



def merge_cleaned_files(cleaned_dfs, experiment_tag, output_dir):
    """
    Merge all cleaned DataFrames into one large DataFrame with an experiment tag.
    
    Args:
    - cleaned_dfs (list of tuples): List of tuples with (DataFrame, alias).
    - experiment_tag (str): A user-supplied tag for the merged output file.
    - output_dir (str): Directory where the merged file will be saved.
    
    Returns:
    - Merged DataFrame.
    """
    # Add an 'experiment' column to each DataFrame to track the origin
    merged_df = pd.concat(
        [df.assign(experiment=alias) for df, alias in cleaned_dfs], 
        ignore_index=True
    )
    
    # Save merged DataFrame to a file with the experiment tag
    output_file = os.path.join(output_dir, f"merged_CLEANED_EXP_{experiment_tag}.csv")
    merged_df.to_csv(output_file, index=False)
    
    print(f"Merged file saved to: {output_file}")
    
    return merged_df

# Function to process a chunk of component names for a given taxa
def process_chunk(task):
    cleaned_df, outline_path, num_dirs, num_thresh, max_pts, taxa, component_names = task
    leaf_ect = LeafECT(cleaned_df, outline_path, num_dirs=num_dirs, num_thresh=num_thresh, max_pts=max_pts)
    results = []
    
    for component_name in component_names:
        ECT_matrix, points = leaf_ect.compute_ect_for_contour(component_name, is_DP='DP')
        if ECT_matrix is not None:
            results.append((ECT_matrix, points, taxa, component_name))
    
    return results


# Worker function that processes a chunk of tasks and appends results to shared lists
def worker(chunk):
    return process_chunk(chunk)


# Encapsulate the multiprocessing logic with chunking using a fixed pool of workers
def main(file_path, outline_path, cleaned_df, taxa_list,
         num_dirs=8,
         num_thresh=200,
         max_pts='DP',
         num_samples=-1,
         chunk_size=50,  # Chunk size to process in each worker
         num_workers=8):  # Fixed number of workers

    # Initialize shared data storage using Manager
    manager = Manager()
    ect_data = manager.list()  # Shared list to store ECT matrices
    group_labels = manager.list()  # Shared list to store species labels
    shapes = manager.list()  # Shared list to store contour points (shapes)
    component_names_list = manager.list()  # Shared list to store component names

    chunks = []
    
    # Prepare chunks of component names
    for taxa in taxa_list:
        species_df = cleaned_df[cleaned_df['genus_species'] == taxa]  # Subset by genus_species
        component_names = (species_df['component_name'].sample(min(num_samples, len(species_df)), random_state=42).tolist()
                   if num_samples != -1 else species_df['component_name'].tolist())

        # Chunk component names
        for i in range(0, len(component_names), chunk_size):
            chunk = (cleaned_df, outline_path, num_dirs, num_thresh, max_pts, taxa, component_names[i:i + chunk_size])
            chunks.append(chunk)

    # Create a Pool of workers to handle chunks
    with Pool(processes=num_workers) as pool:
        # Monitor progress using tqdm based on the number of chunks
        for result in tqdm(pool.imap_unordered(worker, chunks), total=len(chunks), desc="Processing Contours"):
            # Process the results from each chunk
            for ECT_matrix, points, taxa, component_name in result:
                ect_data.append(ECT_matrix)
                group_labels.append(taxa)
                shapes.append(points)
                component_names_list.append(component_name)

    # Convert shared lists back to regular lists
    return list(ect_data), list(group_labels), list(shapes), list(component_names_list)



if __name__ == "__main__":
    file_path = "C:/Users/Will/Downloads/GBIF_DetailedSample_50Spp/LM2_2024_09_18__07-52-47/Data/Measurements/LM2_2024_09_18__07-52-47_MEASUREMENTS.csv"
    outline_path = "C:/Users/Will/Downloads/GBIF_DetailedSample_50Spp/LM2_2024_09_18__07-52-47/Keypoints/Simple_Labels"

    # output_path_hdf5 = "D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/tsne_50taxa_all_leaves_best_settings_wCNames.hdf5"
    # output_path_tsne = "D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/tsne_50taxa_all_leaves_best_settings_tsne.png"

    # output_path_tsne_taxa_specific = "D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/tsne_50taxa_all_leaves_best_settings_highlighted.pdf"

    # output_path_hull_boundary_names = "D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/convex_hull_boundary_component_names10x10.txt"
    # output_path_hull_boundary_figure = "D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/tsne_hull_boundary_grid_sampling10x10.png"
    # # output_path_hdf5 = "D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/tsne_2taxa_all_leaves_best_settings.hdf5"
    # # output_path_tsne = "D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/tsne_2taxa_all_leaves_best_settings_tsne.png"

    # output_dir_phate = "D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/PHATE_50taxa_all_leaves"
    output_path_hdf5 = "D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/tsne_50taxa_all_leaves_best_settings_wCNames_16x20.hdf5"
    output_path_tsne = "D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/tsne_50taxa_all_leaves_best_settings_tsne_16x20.png"

    output_path_tsne_taxa_specific = "D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/tsne_50taxa_all_leaves_best_settings_highlighted_16x20.pdf"

    output_path_hull_boundary_names = "D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/convex_hull_boundary_component_names10x10_16x20.txt"
    output_path_hull_boundary_figure = "D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/tsne_hull_boundary_grid_sampling10x10_16x20.png"
    # output_path_hdf5 = "D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/tsne_2taxa_all_leaves_best_settings.hdf5"
    # output_path_tsne = "D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/tsne_2taxa_all_leaves_best_settings_tsne.png"

    output_dir_phate = "D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/PHATE_50taxa_all_leaves_16x20"
    os.makedirs(output_dir_phate, exist_ok=True)

    # is_reload = True
    is_reload = False


    cleaned_df = preprocessing(file_path, outline_path, show_CF_plot=False, show_first_raw_contour=False, show_df_head=False)
    
    # Step 3: Initialize LeafECT class (as already defined in your script)
    # Step 4: Compute ECT for each genus subset
    num_dirs = 16#8  # Number of directions for ECT
    num_thresh = 20 #200  # Number of thresholds for ECT
    max_pts = 'DP'  
    num_samples = -1  # Number of samples per species for processing, -1 for ALL
    num_workers = 8
    # genus_list = cleaned_df['genus'].unique()  # 25 genera
    # included_taxa = [
    #     "Quercus_alba",
    #     "Quercus_macrocarpa",
    #     "Quercus_gambelii",
    #     "Quercus_velutina"
    # ]
    # included_taxa = [
    #     "Quercus_alba",
    #     "Annona_montana",
    # ]

    # Use selected taxa only
    # taxa_list = [genus for genus in cleaned_df['genus_species'].unique() if genus in included_taxa]
    # Use all taxa 
    taxa_list = cleaned_df['genus_species'].unique().tolist()
    
    if is_reload:
        # To reload the data later for plotting or experimentation:
        ect_data, group_labels, shapes, component_names = load_from_hdf5(output_path_hdf5)
    else:
        ect_data, group_labels, shapes, component_names = main(file_path, outline_path, cleaned_df, taxa_list,
                                          num_dirs=num_dirs, 
                                          num_thresh=num_thresh, 
                                          max_pts=max_pts, 
                                          num_samples=num_samples, 
                                          chunk_size=50,  # Chunk size to process in each worker
                                          num_workers=num_workers,)
    
        # Save the results to a hdf5 file
        save_to_hdf5(ect_data, group_labels, shapes, component_names, output_path_hdf5)


    # Define the output path for saving the plot
    run_phate_grid_by_taxa(ect_data, group_labels, shapes, output_dir_phate)

    run_phate_with_shapes(ect_data, group_labels, shapes, component_names, output_dir_phate)

    # Run PHATE and save plots
    run_phate_simple(ect_data, group_labels, shapes, component_names, output_dir_phate)

    # Run comparison with other methods
    run_phate_dimensionality_comparison(ect_data, group_labels, None, output_dir_phate) # VERY memory hungry with MDS

    plot_tSNE(ect_data, group_labels, output_path_tsne, shapes)
    
    plot_tsne_hull_boundary(ect_data, group_labels, output_path_hull_boundary_names, output_path_hull_boundary_figure, shapes, component_names)

    plot_tSNE_by_taxa(ect_data, group_labels, output_path_tsne_taxa_specific, shapes)
