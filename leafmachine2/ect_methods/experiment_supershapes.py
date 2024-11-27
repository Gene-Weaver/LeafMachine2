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
    from leafmachine2.ect_methods.utils_PHATE import run_phate, run_phate_dimensionality_comparison, run_phate_with_shapes, run_phate_grid_by_taxa
    from leafmachine2.ect_methods.leaf_ect import LeafECT
except:
    from utils_tsne import plot_tSNE, plot_tSNE_by_taxa, plot_tsne_hull_boundary
    from utils_ect import preprocessing, plot_confidence_ellipse_tsne, load_from_hdf5, save_to_hdf5
    from utils_PHATE import run_phate, run_phate_dimensionality_comparison, run_phate_with_shapes, run_phate_grid_by_taxa
    from leaf_ect import LeafECT



# Function to process a chunk of component names for a given taxa
def process_chunk(task):
    outline_path, num_dirs, num_thresh, max_pts, component_names = task
    results = []
    
    leaf_ect = LeafECT(df=None, outline_path=outline_path, num_dirs=num_dirs, num_thresh=num_thresh, max_pts=max_pts)

    for component_name in component_names:
        ECT_matrix, points = leaf_ect.compute_ect_for_contour(component_name, is_DP='DP')
        if ECT_matrix is not None:
            results.append((ECT_matrix, points, component_name))
    
    return results


# Worker function that processes a chunk of tasks and appends results to shared lists
def worker(chunk):
    return process_chunk(chunk)


# Main function modified to process all the files in `outline_path`
def main(outline_path,
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

    # List all files in the outline_path directory
    component_names = [f for f in os.listdir(outline_path) if os.path.isfile(os.path.join(outline_path, f))]
    
    # Optionally, sample a subset of files if num_samples is specified
    if num_samples != -1:
        component_names = np.random.choice(component_names, size=min(num_samples, len(component_names)), replace=False).tolist()

    # Chunk component names
    for i in range(0, len(component_names), chunk_size):
        chunk = (outline_path, num_dirs, num_thresh, max_pts, component_names[i:i + chunk_size])
        chunks.append(chunk)

    # Create a Pool of workers to handle chunks
    with Pool(processes=num_workers) as pool:
        # Monitor progress using tqdm based on the number of chunks
        for result in tqdm(pool.imap_unordered(worker, chunks), total=len(chunks), desc="Processing Contours"):
            # Process the results from each chunk
            for ECT_matrix, points, component_name in result:
                ect_data.append(ECT_matrix)
                group_labels.append(component_name)  # Use component_name as the label
                shapes.append(points)
                component_names_list.append(component_name)

    # Convert shared lists back to regular lists
    return list(ect_data), list(group_labels), list(shapes), list(component_names_list)



if __name__ == "__main__":
    outline_path = "D:/D_Desktop/LM2_Cornales/experiments/supershapes_sweep"


    output_path_hdf5 = "D:/D_Desktop/LM2_Cornales/experiments/supershapes_sweep.hdf5"
    output_path_tsne = "D:/D_Desktop/LM2_Cornales/experiments/supershapes_tsne.png"

    output_path_tsne_taxa_specific = "D:/D_Desktop/LM2_Cornales/experiments/supershapes_tsne_highlighted.pdf"

    output_path_hull_boundary_names = "D:/D_Desktop/LM2_Cornales/experiments/supershapes_convex_hull_boundary_component_names10x10.txt"
    output_path_hull_boundary_figure = "D:/D_Desktop/LM2_Cornales/experiments/supershapes_tsne_hull_boundary_grid_sampling10x10.png"

    output_dir_phate = "D:/D_Desktop/LM2_Cornales/experiments/PHATE_supershapes"
    os.makedirs(output_dir_phate, exist_ok=True)

    # is_reload = True
    is_reload = False

    
    # Step 3: Initialize LeafECT class (as already defined in your script)
    # Step 4: Compute ECT for each genus subset
    num_dirs = 8  # Number of directions for ECT
    num_thresh = 200  # Number of thresholds for ECT
    max_pts = 'DP'  
    num_samples = 100000  # Number of samples per species for processing, -1 for ALL
    num_workers = 8

    
    if is_reload:
        # To reload the data later for plotting or experimentation:
        ect_data, group_labels, shapes, component_names = load_from_hdf5(output_path_hdf5)
    else:
        ect_data, group_labels, shapes, component_names = main(outline_path,
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
    run_phate(ect_data, group_labels, shapes, component_names, output_dir_phate)

    # Run comparison with other methods
    run_phate_dimensionality_comparison(ect_data, group_labels, None, output_dir_phate) # VERY memory hungry with MDS

    # not ready for single class yet, need to add the by_group support
    # plot_tSNE(ect_data, group_labels, output_path_tsne, shapes)
    
    # plot_tsne_hull_boundary(ect_data, group_labels, output_path_hull_boundary_names, output_path_hull_boundary_figure, shapes, component_names)

    # plot_tSNE_by_taxa(ect_data, group_labels, output_path_tsne_taxa_specific, shapes)
