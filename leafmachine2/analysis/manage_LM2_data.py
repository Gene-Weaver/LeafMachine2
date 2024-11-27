import os, glob, shutil, sys, ast
import sqlite3
import dask.dataframe as dd
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import h5py
import numpy as np
import phate
import matplotlib.pyplot as plt
import seaborn as sns
import dask
import dask.array as da
from dask import delayed, compute


currentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from leafmachine2.ect_methods.utils_ect import preprocessing, ingest_DWC_large_files, save_to_hdf5, store_ect_data
from leafmachine2.ect_methods.leaf_ect import LeafECT
from leafmachine2.ect_methods.utils_PHATE import run_phate_simple, run_phate_with_shapes, run_phate_grid_by_taxa, run_phate_heatmap, compute_phate
from leafmachine2.ect_methods.utils_UMAP import compute_umap_direct, load_direct, compute_umap, run_umap_with_shapes, run_umap_grid_by_taxa, run_umap_simple, run_umap_heatmap 
from leafmachine2.ect_methods.utils_UMAP_decay import sample_umap_sensitivity
from leafmachine2.ect_methods.utils_metrics import (plot_ddr_hsi_scatter_with_sd, 
                                                    plot_ddr_ect_mean_scatter_with_sd,
                                                    plot_ddr_Weighted_Avg_Density_mean_ect_3d_to_gif,
                                                    plot_ddr_Weighted_Avg_Density_mean_scatter_with_sd,
                                                    plot_ddr_hsi_ect_3d_to_gif,
                                                    compute_ect_summary_metrics, 
                                                    compute_metrics,
                                                    compute_ect_means,
                                                    plot_ddr_ect_mean_scatter_with_sd_SPECIES_KMEANS,
                                                    )

'''
https://docs.rapids.ai/install/

pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12==24.10.* dask-cudf-cu12==24.10.* cuml-cu12==24.10.* \
    cugraph-cu12==24.10.* nx-cugraph-cu12==24.10.* cuspatial-cu12==24.10.* \
    cuproj-cu12==24.10.* cuxfilter-cu12==24.10.* cucim-cu12==24.10.* \
    pylibraft-cu12==24.10.* raft-dask-cu12==24.10.* cuvs-cu12==24.10.* \
    nx-cugraph-cu12==24.10.*
pip install pandas==2.2.2
'''
# def run_PHATE_on_everything_h5(h5_file_path, bin_by_class="family"):
def run_PHATE_on_everything_h5(dir_path, h5_file_path, bin_by_class="family", ECT_dim=128):

    do_compute_summary_metrics = True
    do_compute_summary_metrics_plots = False

    do_compute_UMAP = False
    do_test_UMAP_decay = False
    do_compute_PHATE = False

    # Extract the parent directory of the h5 file path
    save_path_UMAP = os.path.join(os.path.dirname(h5_file_path), 'UMAP')
    save_path_UMAP_Decay = os.path.join(os.path.dirname(h5_file_path), 'UMAP_Decay')
    save_path_PHATE = os.path.join(os.path.dirname(h5_file_path), 'PHATE')
    save_path_SUMMARY = os.path.join(os.path.dirname(h5_file_path), 'Summary_Metrics')
    save_path_SUMMARY_SPP = os.path.join(os.path.dirname(h5_file_path), 'Summary_Metrics_Species')

    save_path_PHATE_scores_2D = os.path.join(save_path_PHATE, 'PHATE_scores_2D.npz')
    save_path_PHATE_scores_3D = os.path.join(save_path_PHATE, 'PHATE_scores_3D.npz')
    save_path_PHATE_3D_heatmaps = os.path.join(save_path_PHATE, 'PHATE_3D_heatmaps')

    save_path_UMAP_scores_2D = os.path.join(save_path_UMAP, "UMAP_2D.npz")
    save_path_UMAP_scores_3D = os.path.join(save_path_UMAP, "UMAP_3D.npz")
    save_path_UMAP_3D_heatmaps = os.path.join(save_path_UMAP, 'UMAP_3D_heatmaps')


    os.makedirs(save_path_UMAP, exist_ok=True)
    os.makedirs(save_path_UMAP_Decay, exist_ok=True)
    os.makedirs(save_path_PHATE, exist_ok=True)
    os.makedirs(save_path_SUMMARY, exist_ok=True)
    os.makedirs(save_path_SUMMARY_SPP, exist_ok=True)

    os.makedirs(save_path_PHATE_3D_heatmaps, exist_ok=True)
    os.makedirs(save_path_UMAP_3D_heatmaps, exist_ok=True)

    ect_data, group_labels, shapes, component_names = load_direct(dir_path)

    # Load only the relevant columns from LM2_MEASUREMENTS_CLEAN
    # component_name, area, perimeter, convex_hull, convex_hull.1, convexity, concavity, circularity, aspect_ratio, bbox_min_long_side, bbox_min_short_side
    # distance_lamina	distance_width	distance_petiole	distance_midvein_span	distance_petiole_span	trace_midvein_distance	trace_petiole_distance	apex_angle	apex_is_reflex	base_angle	base_is_reflex, megapixels
    # these are boolean: apex_is_reflex, base_is_reflex
    # These are str: component_name
    # All others are floats
    path_to_CLEAN_data = os.path.join(os.path.dirname(h5_file_path), 'LM2_MEASUREMENTS_CLEAN.csv')
    LM2_measurements = load_cleaned_data(path_to_CLEAN_data)

    
    labels_fullname = [ast.literal_eval(g)['fullname'] for g in group_labels]
    labels_genus = [ast.literal_eval(g)['genus'] for g in group_labels]
    labels_family = [ast.literal_eval(g)['family'] for g in group_labels]

    overall_family = ast.literal_eval(group_labels[0])['family']
    

    if do_compute_summary_metrics:
        print("Starting Summary Metrics")

        matrix_means = [np.sum(matrix) / (ECT_dim * ECT_dim) for matrix in ect_data]

        ect_matrix_means_fullname = compute_ect_means(save_path_SUMMARY, matrix_means, shapes, labels_fullname, taxa_type='fullname', do_print=False)
        ect_matrix_means_genus = compute_ect_means(save_path_SUMMARY, matrix_means, shapes, labels_genus, taxa_type='genus', do_print=False)
        ect_matrix_means_family = compute_ect_means(save_path_SUMMARY, matrix_means, shapes, labels_family, taxa_type='family', do_print=False)

        all_metrics = compute_ect_summary_metrics(save_path_SUMMARY, LM2_measurements, ect_data, component_names, matrix_means, ect_matrix_means_fullname, ect_matrix_means_genus, ect_matrix_means_family, labels_fullname, labels_genus, labels_family, do_print=False)
        metrics_summary_species, metrics_summary_genus, metrics_summary_family, class_stats_species, class_stats_genus, class_stats_family = all_metrics 

        def make_plots(save_path, class_stats, ect_matrix_means, matrix_means, labels, shapes, TAXA):
            # if TAXA == 'SPECIES':
            plot_ddr_ect_mean_scatter_with_sd_SPECIES_KMEANS(save_path_SUMMARY_SPP, ect_data, matrix_means, labels, shapes)
            plot_ddr_ect_mean_scatter_with_sd(save_path, class_stats, ect_matrix_means, matrix_means, labels, shapes, filename=f'plot_ddr_ectMean_{TAXA}.png') # KMeans
            plot_ddr_hsi_scatter_with_sd(save_path, class_stats, matrix_means, labels, shapes, filename=f"plot_ddr_hsi_{TAXA}.png") # KMedoids
            plot_ddr_Weighted_Avg_Density_mean_scatter_with_sd(save_path, class_stats, matrix_means, labels, shapes, filename=f"plot_ddr_Weighted_Avg_Density_mean_{TAXA}.png") # KMeans 
            plot_ddr_hsi_ect_3d_to_gif(save_path, class_stats, ect_matrix_means, matrix_means, labels, shapes, filename=f"plot_ddr_hsi_ectMean_3d_{TAXA}.gif")
            plot_ddr_Weighted_Avg_Density_mean_ect_3d_to_gif(save_path, class_stats, ect_matrix_means, matrix_means, labels, shapes, filename=f"plot_ddr_Weighted_Avg_Density_mean_{TAXA}.gif")
        
        if do_compute_summary_metrics_plots:
            make_plots(save_path_SUMMARY, class_stats_genus, ect_matrix_means_genus, matrix_means, labels_genus, shapes, TAXA='GENUS')
            make_plots(save_path_SUMMARY, class_stats_species, ect_matrix_means_fullname, matrix_means, labels_fullname, shapes, TAXA='SPECIES')
            make_plots(save_path_SUMMARY, class_stats_family, ect_matrix_means_family, matrix_means, labels_family, shapes, TAXA='FAMILY')
        

    if do_compute_UMAP:
        print("Computing UMAP")
        umap_scores2D, umap_scores3D = compute_umap_direct(ect_data)

        # umap_scores2D, umap_scores3D = compute_umap(save_path_UMAP_scores_2D, save_path_UMAP_scores_3D, ect_data)

        # umap_scores2D, umap_scores3D = compute_umap_rapids(save_path_UMAP_scores_2D, save_path_UMAP_scores_3D,
        #                                                    ect_data, group_labels, shapes, component_names)

        # Compute metrics for UMAP
        print("Computing UMAP Metrics")
        core_metrics_df, comparison_metrics_df = compute_metrics(save_path_UMAP, umap_scores2D, umap_scores3D, labels_fullname, labels_genus, overall_family, test_type="UMAP")
        print("UMAP core_metrics_df:", core_metrics_df)
        print("UMAP comparison_metrics_df:", comparison_metrics_df)

        run_umap_grid_by_taxa(ect_data, group_labels, shapes, save_path_UMAP, umap_scores2D, bin_by_class='genus')
        # run_umap_grid_by_taxa(ect_data, group_labels, shapes, save_path_UMAP, umap_scores2D, bin_by_class='fullname')
        
        run_umap_with_shapes(ect_data, group_labels, shapes, component_names, save_path_UMAP, umap_scores2D, bin_by_class='genus')
        run_umap_with_shapes(ect_data, group_labels, shapes, component_names, save_path_UMAP, umap_scores2D, bin_by_class='fullname')

        run_umap_simple(ect_data, group_labels, shapes, component_names, save_path_UMAP, umap_scores2D, umap_scores3D, bin_by_class='genus')
        # run_umap_simple(ect_data, group_labels, shapes, component_names, save_path_UMAP, umap_scores2D, umap_scores3D, bin_by_class='fullname')

        run_umap_heatmap(umap_scores3D, group_labels, save_path_UMAP_3D_heatmaps, bin_by_class='genus')
        # run_umap_heatmap(umap_scores3D, group_labels, save_path_UMAP_3D_heatmaps, bin_by_class='fullname')

    
    if do_test_UMAP_decay:
        print("Starting UMAP Sensitivity")
        # test_umap_decay_with_metrics(ect_data, group_labels, shapes, component_names, save_path_UMAP_Decay, labels_fullname, labels_genus, labels_family, overall_family)
        # test_umap_decay_with_metrics(ect_data, group_labels, shapes, component_names, save_path_UMAP_Decay, labels_fullname, labels_genus, labels_family, overall_family,
        #                         n_iterations=5, n_jobs=-1)
        sample_umap_sensitivity(ect_data, save_path_UMAP_Decay)

    if do_compute_PHATE:
        phate_scores_2D, phate_scores_3D = compute_phate(save_path_PHATE_scores_2D, save_path_PHATE_scores_3D,
                                                         ect_data, group_labels, shapes, component_names)

        # Generate the "fullname" and "genus" labels from `group_labels`
        labels_fullname = [ast.literal_eval(g)['fullname'] for g in group_labels]
        labels_genus = [ast.literal_eval(g)['genus'] for g in group_labels]
        overall_family = ast.literal_eval(group_labels[0])['family']

        # Compute metrics for UMAP
        metrics = compute_metrics(save_path_PHATE, ect_data, shapes, phate_scores_2D, phate_scores_3D, labels_fullname, labels_genus, overall_family, test_type="PHATE")
        print("PHATE Metrics:", metrics)

        run_phate_grid_by_taxa(ect_data, group_labels, shapes, save_path_PHATE, phate_scores_2D, bin_by_class='genus')
        run_phate_grid_by_taxa(ect_data, group_labels, shapes, save_path_PHATE, phate_scores_2D, bin_by_class='fullname')

        run_phate_with_shapes(ect_data, group_labels, shapes, component_names, save_path_PHATE, phate_scores_2D, bin_by_class='genus')
        run_phate_with_shapes(ect_data, group_labels, shapes, component_names, save_path_PHATE, phate_scores_2D, bin_by_class='fullname')

        run_phate_simple(ect_data, group_labels, shapes, component_names, save_path_PHATE, phate_scores_2D, phate_scores_3D, bin_by_class='genus')
        run_phate_simple(ect_data, group_labels, shapes, component_names, save_path_PHATE, phate_scores_2D, phate_scores_3D, bin_by_class='fullname')

        run_phate_heatmap(phate_scores_3D, group_labels, save_path_PHATE_3D_heatmaps, bin_by_class='genus')
        run_phate_heatmap(phate_scores_3D, group_labels, save_path_PHATE_3D_heatmaps, bin_by_class='fullname')
        

    print(f"DONE WITH >>> {h5_file_path} <<<")



# def run_PHATE_on_family_h5(h5_file_path, bin_by_class="fullname"):
#     # Extract the parent directory of the h5 file path
#     save_path = os.path.join(os.path.dirname(h5_file_path), 'PHATE')

#     save_path_PHATE_scores_2D = os.path.join(save_path, 'PHATE_scores_2D.npz')
#     save_path_PHATE_scores_3D = os.path.join(save_path, 'PHATE_scores_3D.npz')
#     save_path_PHATE_3D_heatmaps = os.path.join(save_path, 'PHATE_3D_heatmaps')

#     os.makedirs(save_path, exist_ok=True)
#     os.makedirs(save_path_PHATE_3D_heatmaps, exist_ok=True)

#     # Load ECT data and labels
#     ect_data, group_labels, shapes, component_names = load_from_hdf5(h5_file_path)

#     phate_scores_2D = compute_phate(ect_data, save_path_PHATE_scores_2D, n_components=2)
#     phate_scores_3D = compute_phate(ect_data, save_path_PHATE_scores_3D, n_components=3)

#     run_phate_grid_by_taxa(ect_data, group_labels, shapes, save_path, phate_scores_2D, bin_by_class)

#     run_phate_with_shapes(ect_data, group_labels, shapes, component_names, save_path, phate_scores_2D, bin_by_class)

#     run_phate_simple(ect_data, group_labels, shapes, component_names, save_path, phate_scores_2D, phate_scores_3D, bin_by_class)

#     run_phate_heatmap(phate_scores_3D, group_labels, save_path_PHATE_3D_heatmaps, bin_by_class)


# def test_PHATE_on_family_h5(h5_file_path):
#     # Extract the parent directory of the h5 file path
#     save_path = os.path.dirname(h5_file_path)

#     # Load ECT data and labels
#     ect_arr, labels = load_ect_data(h5_file_path)

#     # Flatten the ECT array
#     flat_arr = flatten_ect(ect_arr)

#     # Create a DataFrame from the labels
#     df = pd.DataFrame(labels)

#     # Run PHATE in 2D
#     phate_2d = run_phate(flat_arr, n_components=2)
#     visualize_phate_2d(phate_2d, df, save_path)  # Pass save path for 2D plot

#     # Run PHATE in 3D
#     phate_3d = run_phate(flat_arr, n_components=3)
#     visualize_phate_3d(phate_3d, df, save_path)  # Pass save path for 3D plot

#     # Optional: Create a rotating 3D plot
#     sample_labels = df["fullname"]
#     scprep.plot.rotate_scatter3d(phate_3d, c=sample_labels, 
#                                  figsize=(8, 6), 
#                                  ticks=True, 
#                                  label_prefix="PHATE")

# Load the family_combined_ECT.h5 file


def load_cleaned_data(path):
    # Define the columns to load and their respective data types
    columns_to_load = [
        'component_name', 'area', 'perimeter', 'convex_hull', 'convex_hull.1', 
        'convexity', 'concavity', 'circularity', 'aspect_ratio', 
        'bbox_min_long_side', 'bbox_min_short_side', 'distance_lamina', 
        'distance_width', 'distance_petiole', 'distance_midvein_span', 
        'distance_petiole_span', 'trace_midvein_distance', 
        'trace_petiole_distance', 'apex_angle', 'apex_is_reflex', 
        'base_angle', 'base_is_reflex', 'megapixels'
    ]

    # Specify data types for each column
    dtypes = {
        'component_name': 'str',
        'area': 'float',
        'perimeter': 'float',
        'convex_hull': 'float',
        'convex_hull.1': 'float',
        'convexity': 'float',
        'concavity': 'float',
        'circularity': 'float',
        'aspect_ratio': 'float',
        'bbox_min_long_side': 'float',
        'bbox_min_short_side': 'float',
        'distance_lamina': 'float',
        'distance_width': 'float',
        'distance_petiole': 'float',
        'distance_midvein_span': 'float',
        'distance_petiole_span': 'float',
        'trace_midvein_distance': 'float',
        'trace_petiole_distance': 'float',
        'apex_angle': 'float',
        'apex_is_reflex': 'bool',
        'base_angle': 'float',
        'base_is_reflex': 'bool',
        'megapixels': 'float'
    }

    # Load the data with selected columns and dtypes
    try:
        clean_data = pd.read_csv(path, usecols=columns_to_load, dtype=dtypes)
        print("Loaded LM2_MEASUREMENTS_CLEAN data successfully.")
    except FileNotFoundError:
        print(f"File not found: {path}")
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")

    # Example: Display the first few rows of the loaded data
    print(clean_data.head())
    return clean_data

@delayed
def load_dataset(hdf5_file, dataset_path):
    """Load a specific dataset from an HDF5 file."""
    with h5py.File(hdf5_file, 'r') as f:
        return f[dataset_path][:]
    
@delayed
def decode_label(label):
    """Decode a byte label to string."""
    return label.decode()

def load_from_hdf5_dask(hdf5_file):
    """Optimized function to load data from an HDF5 file using Dask delayed for each dataset individually."""
    with h5py.File(hdf5_file, 'r') as f:
        # Load each ECT matrix as a delayed object (treating as list of arrays to avoid stacking)
        ect_data = [load_dataset(hdf5_file, f'ECT_matrices/matrix_{i}') for i in range(len(f['ECT_matrices']))]
        
        # Load each shape as a delayed object
        shapes = [load_dataset(hdf5_file, f'shapes/shape_{i}') for i in range(len(f['shapes']))]

        # Load group labels and component names as delayed lists
        group_labels = [decode_label(label) for label in f['group_labels'][:]]
        component_names = [decode_label(name) for name in f['component_names'][:]]

    # Compute all delayed tasks at once
    ect_data, shapes, group_labels, component_names = compute(ect_data, shapes, group_labels, component_names)

    return ect_data, group_labels, shapes, component_names



def load_ect_data(h5_file_path):
    labels = {
        "family": [],
        "genus": [],
        "genus_species": [],
        "fullname": []
    }
    
    with h5py.File(h5_file_path, 'r') as f:
        base_names = list(f.keys())  # Get the list of base names in the HDF5 file
        
        # Assume all matrices have the same shape, get the shape from the first one
        first_matrix = f[f'{base_names[0]}/ECT_matrix'][:]
        matrix_shape = first_matrix.shape
        
        # Pre-allocate a NumPy array with the correct shape (n_samples, matrix_shape)
        ect_arr = np.zeros((len(base_names), *matrix_shape))  # Pre-allocate array
        
        # Now fill the NumPy array with the ECT matrices
        for idx, base_name in enumerate(tqdm(base_names, desc="Unpacking HDF5 files")):
            ect_arr[idx] = f[f'{base_name}/ECT_matrix'][:]
            
            # Extract family, genus, genus_species, and fullname from base_name
            parts = base_name.split('_')
            family = parts[2]
            genus = parts[3]
            genus_species = f"{parts[3]}_{parts[4]}"
            fullname = f"{parts[2]}_{parts[3]}_{parts[4]}"
            
            labels["family"].append(family)
            labels["genus"].append(genus)
            labels["genus_species"].append(genus_species)
            labels["fullname"].append(fullname)
    
    return ect_arr, labels

# Flatten the ECT array for PHATE
def flatten_ect(ect_arr):
    return np.reshape(ect_arr, (np.shape(ect_arr)[0], np.shape(ect_arr)[1] * np.shape(ect_arr)[2]))

# Perform PHATE embedding in 2D or 3D
def run_phate(flat_arr, n_components=2):
    phate_operator = phate.PHATE(n_components=n_components)
    phate_embedding = phate_operator.fit_transform(flat_arr)
    return phate_embedding

# Visualize PHATE results in 2D
def visualize_phate_2d(phate_ect, df, save_path):
    df["2DPHATE1"] = phate_ect[:, 0]
    df["2DPHATE2"] = phate_ect[:, 1]
    
    plt.figure(figsize=(7, 7))
    # sns.scatterplot(data=df, x="2DPHATE1", y="2DPHATE2", hue="fullname", legend=True)
    sns.scatterplot(data=df, x="2DPHATE1", y="2DPHATE2", hue="genus", legend=True)
    plt.title("PHATE1 and PHATE2\nby fullname")
    plt.gca().set_aspect("equal")

    # Save the figure to the specified path at 600 DPI
    plt.savefig(os.path.join(save_path, "PHATE_2D.png"), dpi=600)
    plt.close()  # Close the figure after saving

# Visualize PHATE results in 3D
def visualize_phate_3d(phate_ect, df, save_path):
    df["3DPHATE1"] = phate_ect[:, 0]
    df["3DPHATE2"] = phate_ect[:, 1]
    df["3DPHATE3"] = phate_ect[:, 2]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    # sns.scatterplot(data=df, x="3DPHATE1", y="3DPHATE2", hue="fullname", legend=False)
    sns.scatterplot(data=df, x="3DPHATE1", y="3DPHATE2", hue="genus", legend=False)
    plt.title("PHATE1 and PHATE2\nby fullname")
    plt.gca().set_aspect("equal")

    plt.subplot(1, 3, 2)
    # sns.scatterplot(data=df, x="3DPHATE1", y="3DPHATE3", hue="fullname", legend=False)
    sns.scatterplot(data=df, x="3DPHATE1", y="3DPHATE3", hue="genus", legend=False)
    plt.title("PHATE1 and PHATE3\nby fullname")
    plt.gca().set_aspect("equal")

    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df, x="3DPHATE2", y="3DPHATE3", hue="genus", legend=False)
    plt.title("PHATE2 and PHATE3\nby fullname")
    plt.gca().set_aspect("equal")

    plt.tight_layout()

    # Save the figure to the specified path at 600 DPI
    plt.savefig(os.path.join(save_path, "PHATE_3D.png"), dpi=600)
    plt.close()  # Close the figure after saving

# def collate_ect_data(input_dirs, output_h5_path, output_npz_path, include_simplified_contours=False):
#     """
#     Collate ECT matrices and simplified contours from each HDF5 file in input_dirs into a single HDF5 and NumPy file.
#     Additionally, save a separate HDF5 and NumPy file for each directory.

#     Parameters:
#         input_dirs (list): A list of directories to search for ECT HDF5 files.
#         output_h5_path (str): Path where the combined HDF5 file will be saved.
#         output_npz_path (str): Path where the combined NumPy file will be saved.
#         include_simplified_contours (bool): Whether to include simplified contours in the output files.
#     """
#     all_ect_matrices = []
#     all_group_labels = []
#     all_shapes = []
#     all_component_names = []

#     # Iterate over each directory
#     for dir_path in input_dirs:
#         append_mode = False
#         family_name = os.path.basename(os.path.dirname(dir_path))
#         hdf5_dir_path_save = os.path.join(dir_path, 'Data', 'Measurements')

#         if "LM2_" in family_name:
#             family_name = os.path.basename(os.path.dirname(os.path.dirname(dir_path)))
#             family_1 = os.path.join(os.path.dirname(os.path.dirname(dir_path)), 'LM2_1','LM2') # PUT ALL THE DATA IN LM2_1    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#             hdf5_dir_path_save = os.path.join(family_1, 'Data', 'Measurements')
#             append_mode = True

#         hdf5_dir_path = os.path.join(dir_path, 'Data', 'Measurements', 'ECT')

#         # Directory-specific paths for saving data
#         path_npz = os.path.join(hdf5_dir_path_save, f'{family_name}_combined_ECT.npz')
#         path_h5 = os.path.join(hdf5_dir_path_save, f'{family_name}_combined_ECT.h5')

#         family_ect_matrices = []
#         family_group_labels = []
#         family_shapes = []
#         family_component_names = []

#         # Check if the HDF5 directory exists
#         if not os.path.exists(hdf5_dir_path):
#             print(f"HDF5 directory not found: {hdf5_dir_path}. Skipping.")
#             continue

#         # List all .h5 files in the directory
#         h5_files = [f for f in os.listdir(hdf5_dir_path) if f.endswith('.h5')]

#         # Iterate over each HDF5 file in the directory
#         for h5_file in tqdm(h5_files, desc=f"Processing HDF5 files in {hdf5_dir_path}"):
#             hdf5_file_path = os.path.join(hdf5_dir_path, h5_file)

#             try:
#                 # Load data using load_from_hdf5 function
#                 ect_data, group_labels, shapes, component_names = load_from_hdf5(hdf5_file_path)

#                 # Append data to directory-specific lists
#                 family_ect_matrices.extend(ect_data)
#                 family_group_labels.extend(group_labels)
#                 family_shapes.extend(shapes)
#                 family_component_names.extend(component_names)

#                 # Append data to cumulative lists for overall collated file
#                 all_ect_matrices.extend(ect_data)
#                 all_group_labels.extend(group_labels)
#                 all_shapes.extend(shapes)
#                 all_component_names.extend(component_names)

#             except Exception as e:
#                 print(f"Error processing file {hdf5_file_path}: {e}")

#         # Save directory-specific data to an HDF5 file
#         save_to_hdf5(family_ect_matrices, family_group_labels, family_shapes, family_component_names, path_h5, append_mode=append_mode)

#         # Save directory-specific data to a NumPy (.npz) file
#         # np.savez(path_npz, ect_matrices=family_ect_matrices, group_labels=family_group_labels)

#         print(f"Directory-specific ECT data saved to {path_h5} and {path_npz}")

#     # Save cumulative combined data to an HDF5 file
#     save_to_hdf5(all_ect_matrices, all_group_labels, all_shapes, all_component_names, output_h5_path)

#     # Save cumulative combined data to a NumPy (.npz) file
#     # np.savez(output_npz_path, ect_matrices=all_ect_matrices, group_labels=all_group_labels)

#     print(f"Cumulative ECT data saved to {output_h5_path}")


def collate_ect_data_family(dir_path):
    """
    Collate ECT data for a single family directory and save as an HDF5 and NumPy file.
    Parameters:
        dir_path (str): The directory containing the ECT HDF5 files for a family.
    """
    append_mode = False
    family_name = os.path.basename(os.path.dirname(dir_path))
    hdf5_dir_path_save = os.path.join(dir_path, 'Data', 'Measurements')

    if "LM2_" in family_name:
        family_name = os.path.basename(os.path.dirname(os.path.dirname(dir_path)))
        family_1 = os.path.join(os.path.dirname(os.path.dirname(dir_path)), 'LM2_1', 'LM2')  # Store all data in LM2_1
        hdf5_dir_path_save = os.path.join(family_1, 'Data', 'Measurements')
        append_mode = True

    hdf5_dir_path = os.path.join(dir_path, 'Data', 'Measurements', 'ECT')

    # Directory-specific paths for saving data
    path_npz = os.path.join(hdf5_dir_path_save, f'{family_name}_combined_ECT.npz')
    path_h5 = os.path.join(hdf5_dir_path_save, f'{family_name}_combined_ECT.h5')

    family_ect_matrices = []
    family_group_labels = []
    family_shapes = []
    family_component_names = []

    # Check if the HDF5 directory exists
    if not os.path.exists(hdf5_dir_path):
        print(f"HDF5 directory not found: {hdf5_dir_path}. Skipping.")
        return

    # List all .h5 files in the directory
    h5_files = [f for f in os.listdir(hdf5_dir_path) if f.endswith('.h5')]

    # Iterate over each HDF5 file in the directory
    for h5_file in tqdm(h5_files, desc=f"Processing HDF5 files in {hdf5_dir_path}"):
        hdf5_file_path = os.path.join(hdf5_dir_path, h5_file)

        try:
            # Load data using load_from_hdf5 function
            ect_data, group_labels, shapes, component_names = load_from_hdf5_dask(hdf5_file_path)

            # Append data to directory-specific lists
            family_ect_matrices.extend(ect_data)
            family_group_labels.extend(group_labels)
            family_shapes.extend(shapes)
            family_component_names.extend(component_names)

        except Exception as e:
            print(f"Error processing file {hdf5_file_path}: {e}")

    # Save directory-specific data to an HDF5 file
    save_to_hdf5(family_ect_matrices, family_group_labels, family_shapes, family_component_names, path_h5, append_mode=append_mode)

    # Save directory-specific data to a NumPy (.npz) file
    # np.savez(path_npz, ect_matrices=family_ect_matrices, group_labels=family_group_labels)

    print(f"Directory-specific ECT data saved to {path_h5} and {path_npz}")


def collate_ect_data_family_to_total(input_dirs, output_h5_path, output_npz_path):
    """
    Collate data from all family-specific HDF5 files into a single combined HDF5 and NumPy file.

    Parameters:
        input_dirs (list): A list of directories containing family-specific combined ECT HDF5 files.
        output_h5_path (str): Path where the combined HDF5 file will be saved.
        output_npz_path (str): Path where the combined NumPy file will be saved.
    """
    all_ect_matrices = []
    all_group_labels = []
    all_shapes = []
    all_component_names = []

    # Iterate over each directory
    for dir_path in input_dirs:
        family_name = os.path.basename(os.path.dirname(dir_path))
        hdf5_dir_path_save = os.path.join(dir_path, 'Data', 'Measurements')

        if "LM2_" in family_name:
            family_name = os.path.basename(os.path.dirname(os.path.dirname(dir_path)))
            family_1 = os.path.join(os.path.dirname(os.path.dirname(dir_path)), 'LM2_1', 'LM2')  # Store all data in LM2_1
            hdf5_dir_path_save = os.path.join(family_1, 'Data', 'Measurements')
        
        # Path to the family-specific combined HDF5 file
        path_h5 = os.path.join(hdf5_dir_path_save, f'{family_name}_combined_ECT.h5')

        if not os.path.exists(path_h5):
            print(f"Family combined HDF5 file not found: {path_h5}. Skipping.")
            continue

        # Load the family-specific HDF5 data
        try:
            ect_data, group_labels, shapes, component_names = load_from_hdf5_dask(path_h5)

            # Append data to cumulative lists for overall collated file
            all_ect_matrices.extend(ect_data)
            all_group_labels.extend(group_labels)
            all_shapes.extend(shapes)
            all_component_names.extend(component_names)

        except Exception as e:
            print(f"Error processing file {path_h5}: {e}")

    # Save cumulative combined data to an HDF5 file
    save_to_hdf5(all_ect_matrices, all_group_labels, all_shapes, all_component_names, output_h5_path)

    # Save cumulative combined data to a NumPy (.npz) file
    # np.savez(output_npz_path, ect_matrices=all_ect_matrices, group_labels=all_group_labels)

    print(f"Cumulative ECT data saved to {output_h5_path}")


def collate_ect_data(input_dirs, output_h5_path, output_npz_path):
    """
    Collate ECT data by first processing each family directory and then combining all data.

    Parameters:
        input_dirs (list): A list of directories to search for ECT HDF5 files.
        output_h5_path (str): Path where the combined HDF5 file will be saved.
        output_npz_path (str): Path where the combined NumPy file will be saved.
    """
    # First, process each family directory
    for dir_path in input_dirs:
        collate_ect_data_family(dir_path)

    # Then, combine the family-specific files into a single total file
    collate_ect_data_family_to_total(input_dirs, output_h5_path, output_npz_path)


class LM2DataVault:
    def __init__(self, db_path):
        """
        Initialize the LM2DataVault class with a database path. Create the database if it does not exist.
        """
        self.db_path = db_path
        self._create_db_if_not_exists()
        self._create_tables()

    def _create_db_if_not_exists(self):
        """
        Create the SQLite database if it does not exist at db_path.
        """
        if not os.path.exists(self.db_path):
            # Establish a connection which creates the file if it doesn't exist
            print(f"Database not found at {self.db_path}. Creating a new database.")
            conn = sqlite3.connect(self.db_path)
            conn.close()
        else:
            print(f"Database found at {self.db_path}.")

    def _create_tables(self):
        """
        Create necessary tables for the vault and metadata tracking.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL, 
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT
                )
            """)

            # Vault table for storing the ingested data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vault (
                    filename TEXT,
                    component_name TEXT,
                    org_herbcode TEXT,
                    org_gbifid TEXT,
                    org_family TEXT,
                    org_genus TEXT,
                    org_species TEXT,
                    org_fullname TEXT,
                    org_sciname TEXT,
                    image_height INTEGER,
                    image_width INTEGER,
                    n_pts_in_polygon INTEGER,
                    conversion_mean FLOAT,
                    predicted_conversion_factor_cm FLOAT,
                    area FLOAT,
                    perimeter FLOAT,
                    convex_hull FLOAT,
                    rotate_angle FLOAT,
                    bbox_min_long_side FLOAT,
                    bbox_min_short_side FLOAT,
                    convexity FLOAT,
                    concavity FLOAT,
                    circularity FLOAT,
                    aspect_ratio FLOAT,
                    angle FLOAT,
                    distance_lamina FLOAT,
                    distance_width FLOAT,
                    distance_petiole FLOAT,
                    distance_midvein_span FLOAT,
                    distance_petiole_span FLOAT,
                    trace_midvein_distance FLOAT,
                    trace_petiole_distance FLOAT,
                    apex_angle FLOAT,
                    base_angle FLOAT,
                    base_is_reflex BOOLEAN,
                    apex_is_reflex BOOLEAN,
                    UNIQUE(filename, component_name) ON CONFLICT IGNORE
                )
            """)
            
            # Table for storing unique values of the org_ columns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS unique_values (
                    org_column TEXT,
                    unique_value TEXT,
                    PRIMARY KEY (org_column, unique_value)
                )
            """)
            conn.commit()

    def _log_ingestion(self, conn, file_path, start=True, success=False):
        """
        Log ingestion start or end in the ingestion_log table, using the full file path.
        """
        cursor = conn.cursor()
        if start:
            cursor.execute(
                "INSERT INTO ingestion_log (file_path, start_time, status) VALUES (?, ?, ?)",
                (file_path, datetime.now().isoformat(), 'started')
            )
        else:
            status = 'success' if success else 'failed'
            cursor.execute(
                "UPDATE ingestion_log SET end_time = ?, status = ? WHERE file_path = ? AND status = 'started'",
                (datetime.now().isoformat(), status, file_path)
            )
        conn.commit()

    def _is_already_ingested(self, conn, file_path):
        """
        Check if the file (using full file path) has already been ingested.
        """
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM ingestion_log WHERE file_path = ? AND status = 'success'", (file_path,))
        return cursor.fetchone() is not None
    
    def _update_unique_values(self, conn, df):
        """
        Update the unique_values table with unique values from the new data in org_ columns.
        """
        cursor = conn.cursor()
        org_columns = ['org_herbcode', 'org_family', 'org_genus', 'org_fullname', 'org_sciname']

        for col in tqdm(org_columns, desc="     Updating unique values"):
            # Get the new unique values from the new data
            new_unique_values = df[col].unique()

            # Get existing unique values from the database for this column
            cursor.execute(f"SELECT unique_value FROM unique_values WHERE org_column = ?", (col,))
            existing_unique_values = set([row[0] for row in cursor.fetchall()])

            # Find the unique values not yet in the database
            values_to_add = set(new_unique_values) - existing_unique_values

            # Insert the new unique values
            for value in values_to_add:
                cursor.execute("INSERT OR IGNORE INTO unique_values (org_column, unique_value) VALUES (?, ?)", (col, value))

        conn.commit()

    def update_unique_values_for_whole_db(self):
        """
        Backfill the unique_values table by processing the entire vault table and updating unique org_ values.
        """
        with sqlite3.connect(self.db_path) as conn:
            # Query the entire vault table to get all existing data
            df = pd.read_sql("SELECT org_herbcode, org_family, org_genus, org_fullname, org_sciname FROM vault", conn)
            
            # Run _update_unique_values to update the unique values
            self._update_unique_values(conn, df)

        print("Unique values updated for the whole database.")


    def _split_filename(self, filename):
        """
        Split the filename into its component parts.
        """
        
        parts = filename.split("_")
        # Ensure we have enough parts to avoid IndexError
        if len(parts) < 5:
            return None, None, None, None, None, None, None
        
        org_herbcode = parts[0]
        org_gbifid = parts[1]
        org_family = parts[2]
        org_genus = parts[3]
        org_species = parts[4]
        org_fullname = f"{org_family}_{org_genus}_{org_species}"
        org_sciname = f"{org_genus}_{org_species}"
        return org_herbcode, org_gbifid, org_family, org_genus, org_species, org_fullname, org_sciname
    
    def _batch_insert(self, conn, data, chunk_size=10000):
        """
        Perform batch inserts into the vault table.
        """
        cursor = conn.cursor()
        insert_query = """
            INSERT OR IGNORE INTO vault (
                filename, component_name, org_herbcode, org_gbifid, org_family, 
                org_genus, org_species, org_fullname, org_sciname, image_height, 
                image_width, n_pts_in_polygon, conversion_mean, predicted_conversion_factor_cm, area, perimeter,
                convex_hull, rotate_angle, bbox_min_long_side, bbox_min_short_side, convexity, 
                concavity, circularity, aspect_ratio, angle, distance_lamina, 
                distance_width, distance_petiole, distance_midvein_span, distance_petiole_span, trace_midvein_distance, 
                trace_petiole_distance, apex_angle, base_angle, base_is_reflex, apex_is_reflex
            ) VALUES (
            ?, ?, ?, ?, ?, 
            ?, ?, ?, ?, ?, 
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, 
            ?, ?, ?, ?, ?, 
            ?, ?, ?, ?, ?, 
            ?, ?, ?, ?, ?)
        """
        
        # Create tqdm progress bar
        total_batches = (len(data) + chunk_size - 1) // chunk_size  # Total number of batches
        for i in tqdm(range(0, len(data), chunk_size), total=total_batches, desc="     Inserting into database"):
            batch = data[i:i + chunk_size]
            cursor.executemany(insert_query, batch)
            conn.commit()

    def ingest_files(self, paths):
        """
        Ingest the CSV files into the vault database.
        """
        with sqlite3.connect(self.db_path) as conn:
            print("Current ingestion log:")
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ingestion_log")
            rows = cursor.fetchall()
            for row in rows:
                print(row)  # Print each row in the ingestion log table

            for file_path in paths:
                # Use full file path to check for ingestion
                if self._is_already_ingested(conn, file_path):
                    print(f"     File {file_path} has already been ingested. Skipping.")
                    continue

                try:
                    # Begin a transaction
                    conn.execute("BEGIN TRANSACTION")

                    # Log ingestion start
                    self._log_ingestion(conn, file_path, start=True)

                    # Load CSV using your custom function
                    df = ingest_DWC_large_files(file_path)

                    # Check if df is None
                    if df is None:
                        raise ValueError(f"Failed to load file: {file_path}")

                    # Define the meta for each of the new columns as a Series
                    meta = pd.Series({
                        'org_herbcode': 'object',
                        'org_gbifid': 'object',
                        'org_family': 'object',
                        'org_genus': 'object',
                        'org_species': 'object',
                        'org_fullname': 'object',
                        'org_sciname': 'object'
                    })

                    # Apply map function to 'filename' and extract the parts
                    split_cols = df['filename'].map(self._split_filename, meta=meta)

                    # Split result to individual columns
                    df = df.assign(
                        org_herbcode=split_cols.map(lambda x: x[0], meta=('org_herbcode', 'object')),
                        org_gbifid=split_cols.map(lambda x: x[1], meta=('org_gbifid', 'object')),
                        org_family=split_cols.map(lambda x: x[2], meta=('org_family', 'object')),
                        org_genus=split_cols.map(lambda x: x[3], meta=('org_genus', 'object')),
                        org_species=split_cols.map(lambda x: x[4], meta=('org_species', 'object')),
                        org_fullname=split_cols.map(lambda x: x[5], meta=('org_fullname', 'object')),
                        org_sciname=split_cols.map(lambda x: x[6], meta=('org_sciname', 'object'))
                    )

                    # Define columns to keep in the exact order as the INSERT statement
                    columns_to_keep = [
                        'filename', 'component_name', 'org_herbcode', 'org_gbifid', 'org_family', 
                        'org_genus', 'org_species', 'org_fullname', 'org_sciname', 'image_height', 
                        'image_width', 'n_pts_in_polygon', 'conversion_mean', 'predicted_conversion_factor_cm', 'area', 'perimeter',
                        'convex_hull', 'rotate_angle', 'bbox_min_long_side', 'bbox_min_short_side', 'convexity', 
                        'concavity', 'circularity', 'aspect_ratio', 'angle', 'distance_lamina', 
                        'distance_width', 'distance_petiole', 'distance_midvein_span', 'distance_petiole_span', 'trace_midvein_distance', 
                        'trace_petiole_distance', 'apex_angle', 'base_angle', 'base_is_reflex', 'apex_is_reflex'
                    ]

                    # Ensure there are exactly 36 columns
                    assert len(columns_to_keep) == 36, f"Expected 36 columns, got {len(columns_to_keep)}"

                    # Filter the dataframe to include only these columns
                    df = df[columns_to_keep]

                    # Ensure the dataframe has the expected number of columns
                    assert len(df.columns) == 36, f"Expected 36 columns in DataFrame, but got {len(df.columns)}"

                    # Convert dask dataframe to pandas in manageable chunks and insert into database
                    num_partitions = df.npartitions  # Get number of partitions
                    for chunk in tqdm(df.partitions, total=num_partitions, desc=f"     Ingesting {file_path}"):
                        pandas_chunk = chunk.compute()  # Convert the chunk to a Pandas DataFrame

                        # Convert dataframe rows to list of tuples
                        data = pandas_chunk.to_records(index=False).tolist()
                        self._batch_insert(conn, data)

                    # Update unique values with the new data
                    self._update_unique_values(conn, df)

                    # Commit the transaction after successful processing
                    conn.commit()

                    # Log ingestion success
                    self._log_ingestion(conn, file_path, start=False, success=True)

                except Exception as e:
                    print(f"Error ingesting file {file_path}: {e}")
                    # Rollback the transaction if an error occurs
                    conn.rollback()

                    # Log ingestion failure
                    self._log_ingestion(conn, file_path, start=False, success=False)



if __name__ == '__main__':
    run_CLEAN = False
    run_ingest = False

    run_ECT = False
    run_collate_ECTs = False 

    run_PHATE_test = False # DEP
    run_PHATE_on_family = True
    run_PHATE_on_everything = False

    db_path = '/media/nas/GBIF_Downloads/Combined_LM2_Data/lm2_vault_copy.db'

    # Directories that contain the desired LM2* _MEASUREMENTS.csv files
    input_dirs = [
        '/media/nas/GBIF_Downloads/Magnoliales/Eupomatiaceae/LM2',# DONE ECT
        '/media/nas/GBIF_Downloads/Cornales/Cornaceae/LM2_2024_09_25__13-47-42',  # DONE ECT
        '/media/nas/GBIF_Downloads/Cornales/Loasaceae/LM2', # DONE ECT
        # '/media/nas/GBIF_Downloads/Magnoliales/Annonaceae/LM2_1/LM2', # 1-7 # DONE ECT
        # '/media/nas/GBIF_Downloads/Magnoliales/Annonaceae/LM2_2/LM2', # 1-7 # DONE ECT
        # '/media/nas/GBIF_Downloads/Magnoliales/Annonaceae/LM2_3/LM2', # 1-7 # DONE ECT
        # '/media/nas/GBIF_Downloads/Magnoliales/Annonaceae/LM2_4/LM2', # 1-7 # DONE ECT
        # '/media/nas/GBIF_Downloads/Magnoliales/Annonaceae/LM2_5/LM2', # 1-7 # DONE ECT
        # '/media/nas/GBIF_Downloads/Magnoliales/Annonaceae/LM2_6/LM2', # 1-7 # DONE ECT
        # '/media/nas/GBIF_Downloads/Magnoliales/Annonaceae/LM2_7/LM2', # 1-7 # DONE ECT
        '/media/nas/GBIF_Downloads/Cornales/Hydrangeaceae/LM2',# DONE ECT
        '/media/nas/GBIF_Downloads/Cornales/Nyssaceae/LM2',# DONE ECT
        '/media/nas/GBIF_Downloads/Magnoliales/Degeneriaceae/LM2',# DONE ECT
        '/media/nas/GBIF_Downloads/Magnoliales/Himantandraceae/LM2',# DONE ECT
        '/media/nas/GBIF_Downloads/Magnoliales/Magnoliaceae/LM2',# DONE ECT
        '/media/nas/GBIF_Downloads/Magnoliales/Myristicaceae/LM2_2024_09_29__19-09-14',# DONE ECT
        '/media/nas/GBIF_Downloads/Moraceae/LM2', # have coordinates only # DONE ECT
        '/media/nas/GBIF_Downloads/Urticaceae/LM2', # have coordinates only # DONE ECT
    ]  
    input_dirs2 = [
        '/media/nas/GBIF_Downloads/Magnoliales/Annonaceae/LM2_1/LM2', # 1-7 # DONE ECT
        '/media/nas/GBIF_Downloads/Magnoliales/Annonaceae/LM2_2/LM2', # 1-7 # DONE ECT
        '/media/nas/GBIF_Downloads/Magnoliales/Annonaceae/LM2_3/LM2', # 1-7 # DONE ECT
        '/media/nas/GBIF_Downloads/Magnoliales/Annonaceae/LM2_4/LM2', # 1-7 # DONE ECT
        '/media/nas/GBIF_Downloads/Magnoliales/Annonaceae/LM2_5/LM2', # 1-7 # DONE ECT
        '/media/nas/GBIF_Downloads/Magnoliales/Annonaceae/LM2_6/LM2', # 1-7 # DONE ECT
        '/media/nas/GBIF_Downloads/Magnoliales/Annonaceae/LM2_7/LM2', # 1-7 # DONE ECT
        '/media/nas/GBIF_Downloads/Cornales/Loasaceae/LM2', # DONE ECT
        '/media/nas/GBIF_Downloads/Cornales/Cornaceae/LM2_2024_09_25__13-47-42',  # DONE ECT
        '/media/nas/GBIF_Downloads/Cornales/Hydrangeaceae/LM2',# DONE ECT
        '/media/nas/GBIF_Downloads/Cornales/Nyssaceae/LM2',# DONE ECT
        '/media/nas/GBIF_Downloads/Magnoliales/Degeneriaceae/LM2',# DONE ECT
        '/media/nas/GBIF_Downloads/Magnoliales/Eupomatiaceae/LM2',# DONE ECT
        '/media/nas/GBIF_Downloads/Magnoliales/Himantandraceae/LM2',# DONE ECT
        '/media/nas/GBIF_Downloads/Magnoliales/Magnoliaceae/LM2',# DONE ECT
        '/media/nas/GBIF_Downloads/Magnoliales/Myristicaceae/LM2_2024_09_29__19-09-14',# DONE ECT
        '/media/nas/GBIF_Downloads/Urticaceae/LM2', # have coordinates only # DONE ECT
        '/media/nas/GBIF_Downloads/Moraceae/LM2', # have coordinates only # DONE ECT
    ]  
    # input_dirs = [
        # '/media/nas/GBIF_Downloads/Magnoliales/Eupomatiaceae/LM2',
        # '/media/nas/GBIF_Downloads/Magnoliales/Degeneriaceae/LM2',
        # '/media/nas/GBIF_Downloads/Cornales/Loasaceae/LM2',
    # ]  

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
        store_ect_data(input_dirs, num_dirs=128, num_thresh=128)
    
    
    # vault.update_unique_values_for_whole_db()

    if run_collate_ECTs:
        output_h5_path = '/media/nas/GBIF_Downloads/Combined_LM2_Data/collated_ect_data_newphyt.h5'
        output_npz_path = '/media/nas/GBIF_Downloads/Combined_LM2_Data/collated_ect_data_newphyt.npz'

        ### collate_ect_data(input_dirs, output_h5_path, output_npz_path)  

        # First, process each family directory
        for dir_path in input_dirs2:
            collate_ect_data_family(dir_path)

        # Then, combine the family-specific files into a single total file
        collate_ect_data_family_to_total(input_dirs, output_h5_path, output_npz_path) 


    # if run_PHATE_test:
        # test_PHATE_on_family_h5('/media/nas/GBIF_Downloads/Cornales/Loasaceae/LM2/Data/Measurements/Loasaceae_combined_ECT.h5')

    if run_PHATE_on_family:
        for dir_path in input_dirs:
            family_name = os.path.basename(os.path.dirname(dir_path))
            family_ECT_path = os.path.join(dir_path, 'Data', 'Measurements', f'{family_name}_combined_ECT.h5')

            if "LM2_" in family_name:
                family_name = os.path.basename(os.path.dirname(os.path.dirname(dir_path)))
                family_1 = os.path.join(os.path.dirname(os.path.dirname(dir_path)), 'LM2_1', 'LM2')  # Store all data in LM2_1
                family_ECT_path = os.path.join(family_1, 'Data', 'Measurements', f'{family_name}_combined_ECT.h5')
            
            
            # run_PHATE_on_everything_h5(family_ECT_path,
            #                     bin_by_class="fullname")
            run_PHATE_on_everything_h5(dir_path,family_ECT_path,
                    bin_by_class="genus")
        
    if run_PHATE_on_everything:
        run_PHATE_on_everything_h5('/media/nas/GBIF_Downloads/Combined_LM2_Data/collated_ect_data_newphyt.h5',
                                bin_by_class="genus") # genus?
        