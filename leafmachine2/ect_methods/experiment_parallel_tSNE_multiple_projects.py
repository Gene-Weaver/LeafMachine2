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
import dask.dataframe as dd

currentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parentdir = os.path.dirname(currentdir)
try:
    from leafmachine2.ect_methods.utils_tsne import plot_tSNE, plot_tSNE_by_taxa, plot_tsne_hull_boundary
    from leafmachine2.ect_methods.utils_ect import preprocessing, load_from_hdf5, save_to_hdf5
    from leafmachine2.ect_methods.leaf_ect import LeafECT
except:
    from utils_tsne import plot_tSNE, plot_tSNE_by_taxa, plot_tsne_hull_boundary
    from utils_ect import preprocessing, load_from_hdf5, save_to_hdf5
    from leaf_ect import LeafECT




def load_merged_file(merged_df_file):
    """
    Load an existing merged DataFrame using Dask for parallelization.
    
    Args:
    - merged_df_file (str): Path to the CSV file with the merged data.
    
    Returns:
    - Dask DataFrame (lazy-loaded DataFrame).
    """
    print(f"Loading existing merged file using Dask: {merged_df_file}")
    
    # Load the CSV file using Dask
    dask_df = dd.read_csv(merged_df_file)
    
    # If you want to convert it to a Pandas DataFrame
    merged_df = dask_df.compute()  # This triggers the actual computation and loads data
    
    return merged_df

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
def main(use_these_taxa, outline_path, cleaned_df, taxa_list,
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
        species_df = cleaned_df[cleaned_df[use_these_taxa] == taxa]  # Subset by use_these_taxa
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


    # is_reload = True # TODO
    is_reload = False  # TODO


    num_dirs = 8      # Number of directions for ECT                               # TODO
    num_thresh = 200  # Number of thresholds for ECT                               # TODO
    max_pts = 'DP'    # DP for auto                                                # TODO
    num_samples = 10 #-1  # Number of samples per species for processing, -1 for ALL   # TODO
    num_workers = 8   # num_workers                                                # TODO
    use_these_taxa = 'genus_species' # family genus genus_species fullname


    experiment_tag = "cornales_10LeavesPerSpp" # TODO

    
    output_path_hdf5 = f"D:/D_Desktop/LM2_Cornales/experiments/tsne_{experiment_tag}.hdf5"      # TODO
    output_path_tsne = f"D:/D_Desktop/LM2_Cornales/experiments/tsne_{experiment_tag}.png"  # TODO
    output_path_tsne_taxa_specific = f"D:/D_Desktop/LM2_Cornales/experiments/tsne_{experiment_tag}_taxa_highlighted.pdf" # TODO

    output_path_hull_boundary_names = f"D:/D_Desktop/LM2_Cornales/experiments/tsne_{experiment_tag}_convex_hull_boundary_component_names10x10.txt"
    output_path_hull_boundary_figure = f"D:/D_Desktop/LM2_Cornales/experiments/tsne_{experiment_tag}_hull_boundary_grid_sampling10x10.png"

    output_dir = "D:/D_Desktop/LM2_Cornales/experiments"                                                                             # TODO


    # Example of multiple file paths and outline paths
    file_paths = [
        "D:/D_Desktop/LM2_Cornales/Cornaceae/LM2_MEASUREMENTS.csv",
        "D:/D_Desktop/LM2_Cornales/Loasaceae/LM2_MEASUREMENTS.csv",
        "D:/D_Desktop/LM2_Cornales/Nyssaceae/LM2_MEASUREMENTS.csv",
        "D:/D_Desktop/LM2_Cornales/Hydrangeaceae/LM2_MEASUREMENTS.csv",
    ]
    
    outline_paths = [
        "D:/D_Desktop/LM2_Cornales/Cornaceae/Simple_Labels",
        "D:/D_Desktop/LM2_Cornales/Loasaceae/Simple_Labels",
        "D:/D_Desktop/LM2_Cornales/Nyssaceae/Simple_Labels",
        "D:/D_Desktop/LM2_Cornales/Hydrangeaceae/Simple_Labels",
    ]
    
    aliases = [f"Cornaceae_{experiment_tag}", 
               f"Loasaceae_{experiment_tag}", 
               f"Nyssaceae_{experiment_tag}", 
               f"Hydrangeaceae_{experiment_tag}"]  # Unique tags/aliases for each experiment
    
    
    
    cleaned_dfs = []
    
    # Merge file path
    merged_df_file = os.path.join(output_dir, f"merged_CLEANED_EXP_{experiment_tag}.csv")

    # # Check if the merged file already exists
    # if os.path.exists(merged_df_file):
    #     # Load existing merged file
    #     merged_df = load_merged_file(merged_df_file)
    # else:
    #     # Loop over all file paths and outline paths, preprocess each one
    #     for file_path, outline_path, alias in zip(file_paths, outline_paths, aliases):
    #         print(f"Processing: {alias}")

    #         if is_reload:
    #             # Reload previously processed data
    #             ect_data, group_labels, shapes = load_from_hdf5(f"{output_dir}/{alias}_data.hdf5")
    #         else:
    #             # Preprocess the data
    #             cleaned_df = preprocessing(file_path, outline_path)
    #             cleaned_dfs.append((cleaned_df, alias))  # Add the cleaned DataFrame and its alias

    #     # Merge all cleaned DataFrames and save them to a new file with the experiment tag
    #     if cleaned_dfs:
    #         merged_df = merge_cleaned_files(cleaned_dfs, experiment_tag, output_dir)

    # Check if the merged file already exists
    if os.path.exists(merged_df_file):
        # Load existing merged file
        merged_df = load_merged_file(merged_df_file)
    else:
        # Loop over all file paths and outline paths, preprocess each one
        for file_path, outline_path, alias in zip(file_paths, outline_paths, aliases):
            clean_file_path = file_path.replace("LM2_MEASUREMENTS.csv", "LM2_MEASUREMENTS_CLEAN.csv")
            
            if os.path.exists(clean_file_path):
                print(f"Skipping {alias} as {clean_file_path} already exists.")
                cleaned_df = pd.read_csv(clean_file_path)  # Load the cleaned CSV if it exists
                cleaned_dfs.append((cleaned_df, alias))  # Add the cleaned DataFrame and its alias
            else:
                print(f"Processing: {alias}")
                
                if is_reload:
                    # Reload previously processed data
                    ect_data, group_labels, shapes = load_from_hdf5(f"{output_dir}/{alias}_data.hdf5")
                else:
                    # Preprocess the data
                    cleaned_df = preprocessing(file_path, outline_path)
                    cleaned_dfs.append((cleaned_df, alias))  # Add the cleaned DataFrame and its alias

                    # Save the cleaned file to avoid re-processing next time
                    cleaned_df.to_csv(clean_file_path, index=False)
                    print(f"Saved cleaned file: {clean_file_path}")

        # Merge all cleaned DataFrames and save them to a new file with the experiment tag
        if cleaned_dfs:
            merged_df = merge_cleaned_files(cleaned_dfs, experiment_tag, output_dir)

    
    # Use selected taxa only
    # taxa_list = [genus for genus in cleaned_df['genus_species'].unique() if genus in included_taxa]
    # Use all taxa 
    taxa_list = merged_df[use_these_taxa].unique().tolist()
    
    if is_reload:
        # To reload the data later for plotting or experimentation:
        ect_data, group_labels, shapes, component_names = load_from_hdf5(output_path_hdf5)
    else:
        ect_data, group_labels, shapes, component_names = main(use_these_taxa, outline_path, merged_df, taxa_list,
                                                                num_dirs=num_dirs,
                                                                num_thresh=num_thresh,
                                                                max_pts=max_pts,
                                                                num_samples=num_samples,
                                                                chunk_size=50,  # Chunk size to process in each worker
                                                                num_workers=num_workers)
        # Save the results to a hdf5 file
        save_to_hdf5(ect_data, group_labels, shapes, component_names, output_path_hdf5)

    # Define the output path for saving the plot
    plot_tSNE(ect_data, group_labels, output_path_tsne, shapes)
    
    plot_tsne_hull_boundary(ect_data, group_labels, output_path_hull_boundary_names, output_path_hull_boundary_figure, shapes, component_names)

    plot_tSNE_by_taxa(ect_data, group_labels, output_path_tsne_taxa_specific, shapes)

