import os, ast, scprep, gc
import matplotlib.pyplot as plt
import sklearn.manifold  # MDS, t-SNE
import time
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import h5py
import umap
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import warnings
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings("ignore", category=UserWarning, module="umap")


currentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parentdir = os.path.dirname(currentdir)
try:
    from leafmachine2.ect_methods.utils_tsne import CustomLegendShape, HandlerShape
    from leafmachine2.ect_methods.utils_PHATE import load_direct, compute_distances_in_batches
    from leafmachine2.ect_methods.utils_metrics import compute_metrics
except:
    from utils_tsne import CustomLegendShape, HandlerShape
    from utils_PHATE import load_direct, compute_distances_in_batches
    from utils_metrics import compute_metrics
                                                   
def save_umap_results(umap_scores, output_path):
    """Save UMAP scores to an npz file."""
    np.savez(output_path, umap_scores=umap_scores)

def load_umap_results(npz_file_path):
    """Load UMAP scores from an npz file."""
    data = np.load(npz_file_path)
    return data['umap_scores']

def flatten_core_metrics(core_metrics, fraction):
    """Flatten the nested core_metrics dictionary and add fraction."""
    flat_metrics = {}
    for category, metrics in core_metrics.items():
        for metric_name, value in metrics.items():
            flat_metrics[f"{category}_{metric_name}"] = value
    flat_metrics["fraction"] = fraction
    return flat_metrics
'''
def test_umap_decay_with_metrics(ect_data, group_labels, shapes, component_names, save_dir, labels_fullname, labels_genus, labels_family, overall_family):
    output_npz_path2D = os.path.join(save_dir, "UMAP_2D.npz")
    output_npz_path3D = os.path.join(save_dir, "UMAP_3D.npz")

    fractions = [1.0, 0.5, 0.25, 0.2, 0.1, 0.05]
    all_core_metrics = []
    all_comparison_metrics = []

    for frac in fractions:
        subset_size = int(len(ect_data) * frac)
        subset_indices = np.random.RandomState(2024).choice(len(ect_data), subset_size, replace=False).astype(int)
        subset_indices.sort()

        subset_ect_data = [ect_data[i] for i in subset_indices]
        subset_labels_fullname = [labels_fullname[i] for i in subset_indices]
        subset_labels_genus = [labels_genus[i] for i in subset_indices]

        subset_output_npz_path2D = output_npz_path2D.replace('.npz', f'__{int(frac * 100):03}.npz')
        subset_output_npz_path3D = output_npz_path3D.replace('.npz', f'__{int(frac * 100):03}.npz')
        test_type = f"__{int(frac * 100):03}"

        print(f"Running UMAP and computing metrics with {int(frac * 100)}% of the data...")

        umap_scores2D, umap_scores3D = compute_umap(subset_output_npz_path2D, subset_output_npz_path3D, subset_ect_data, subset_indices)
        core_metrics, comparison_metrics = compute_metrics(
            save_dir, umap_scores2D, umap_scores3D, subset_labels_fullname, subset_labels_genus, overall_family, f"UMAP{test_type}")

        # Flatten core_metrics and add fraction
        flat_core_metrics = flatten_core_metrics(core_metrics, int(frac * 100))
        all_core_metrics.append(pd.DataFrame([flat_core_metrics]))

        # Convert comparison_metrics to DataFrame and add fraction
        comparison_metrics_df = pd.DataFrame(comparison_metrics)
        comparison_metrics_df['fraction'] = int(frac * 100)
        all_comparison_metrics.append(comparison_metrics_df)

    # Combine all metrics into single DataFrames
    combined_core_metrics_df = pd.concat(all_core_metrics, ignore_index=True)
    combined_comparison_metrics_df = pd.concat(all_comparison_metrics, ignore_index=True)

    # Save combined metrics to CSV
    combined_core_metrics_df.to_csv(os.path.join(save_dir, "UMAP_core_metrics_comparison.csv"), index=False)
    combined_comparison_metrics_df.to_csv(os.path.join(save_dir, "UMAP_comparison_metrics_comparison.csv"), index=False)

    # Plot each metric in its own subplot
    metrics = combined_core_metrics_df.columns.drop("fraction")  # Exclude 'fraction' from the metrics list
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 20), sharex=True)

    for i, metric in enumerate(metrics):
        axes[i].plot(combined_core_metrics_df['fraction'], combined_core_metrics_df[metric], marker='o')
        axes[i].set_title(metric)
        axes[i].set_ylabel("Metric Value")
        axes[i].grid(True)
        
    axes[-1].set_xlabel("Fraction of Data (%)")
    axes[-1].set_xticks(combined_core_metrics_df['fraction'].unique())
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "UMAP_metrics_comparison_plot.png"))
    plt.close()

    print("Core metrics and comparison metrics have been saved and plotted.")
    return combined_core_metrics_df, combined_comparison_metrics_df
'''

def compute_umap(output_npz_path2D, output_npz_path3D, ect_data, subset_indices=None):
    """Check if UMAP results exist; if not, compute and save them."""
    if os.path.exists(output_npz_path2D):
        print(f"Loading UMAP results from {output_npz_path2D}")
        umap_scores2D = load_umap_results(output_npz_path2D)
        print(f"Loading UMAP results from {output_npz_path3D}")
        umap_scores3D = load_umap_results(output_npz_path3D)

    else:
        print(f"Computing UMAP results and saving to {output_npz_path2D}")
        # Get the directory of the output path for saving intermediate results
        output_dir = os.path.dirname(output_npz_path2D)
        output_dir2 = os.path.dirname(os.path.dirname(output_npz_path2D))
        distance_matrix_path = os.path.join(output_dir2, "distance_matrix.h5")

        # Calculate pairwise distance matrix if not already calculated
        if os.path.exists(distance_matrix_path):
            print("Loading existing distance matrix.")
            # Load the distance matrix from file
            with h5py.File(distance_matrix_path, "r") as hf:
                distance_matrix = hf["distances"][:]
        else:
            # Compute the distance matrix in batches and save it
            print("Computing and saving new distance matrix.")
            distance_matrix = compute_distances_in_batches(
                ect_data, 
                distance_matrix_path=distance_matrix_path, 
                batch_size=10000, 
                save_intermediate_dir=output_dir
            )

        # If subset_indices is provided, use it to select the subset of the distance matrix
        if subset_indices is not None:
            distance_matrix = distance_matrix[np.ix_(subset_indices, subset_indices)]

        # 2D UMAP computation
        umap_operator2D = umap.UMAP(n_components=2,
                                    n_neighbors=5,
                                    metric="precomputed",
                                    random_state=2024,
                                    n_jobs=-1)

        umap_scores2D = umap_operator2D.fit_transform(distance_matrix)
        save_umap_results(umap_scores2D, output_npz_path2D)

        # 3D UMAP computation
        print(f"Computing UMAP results and saving to {output_npz_path3D}")
        umap_operator3D = umap.UMAP(n_components=3,
                                    n_neighbors=5,
                                    metric="precomputed",
                                    random_state=2024,
                                    n_jobs=-1)

        umap_scores3D = umap_operator3D.fit_transform(distance_matrix)
        save_umap_results(umap_scores3D, output_npz_path3D)

    return umap_scores2D, umap_scores3D

def compute_umap_direct(ect_data, subset_indices=None, seed=2024):
    if not isinstance(ect_data, np.ndarray):
        ect_data = np.array([matrix.flatten() for matrix in ect_data])

    # If subset_indices is provided, select the subset of the data
    data = ect_data[subset_indices] if subset_indices is not None else ect_data

    # 2D UMAP computation
    print("Fitting 2D UMAP")
    umap_operator2D = umap.UMAP(n_components=2, n_neighbors=5, metric="euclidean", random_state=seed, n_jobs=-1)
    umap_scores2D = umap_operator2D.fit_transform(data)

    # Delete umap_operator2D and run garbage collection to free memory
    del umap_operator2D
    gc.collect()

    # 3D UMAP computation
    print("Fitting 3D UMAP")
    umap_operator3D = umap.UMAP(n_components=3, n_neighbors=5, metric="euclidean", random_state=seed, n_jobs=-1)
    umap_scores3D = umap_operator3D.fit_transform(data)

    # Delete umap_operator2D and run garbage collection to free memory
    del umap_operator3D
    gc.collect()

    return umap_scores2D, umap_scores3D

# def compute_umap_rapids(output_npz_path2D, output_npz_path3D, ect_data, group_labels, shapes, component_names):
#     import os
#     import cuml
#     from cuml.manifold import UMAP as cumlUMAP
#     import numpy as np
#     """Compute UMAP using RAPIDS cuML and save 2D and 3D embeddings."""
    
#     print("Running UMAP with RAPIDS cuML...")
    
#     # GPU Memory	Suggested nnd_n_clusters
#     # 8–12 GB	    12–16
#     # 16–24 GB	    8–10
#     # 32–48 GB	    4–6
#     # 64+ GB	    2–4
#     # Parameters for UMAP using NN-descent with batching
#     umap_params = {
#         'n_neighbors': 16,
#         'n_components': 2,  # 2D embedding for first run
#         'random_state': 2024,
#         'build_algo': 'nn_descent',  # Use NN-descent for faster performance
#         'build_kwds': {
#             'nnd_graph_degree': 32,
#             'nnd_do_batch': True,          # Enable batching
#             'nnd_n_clusters': 8,           # Number of clusters (adjustable)
#             'nnd_return_distances': True,  # Necessary for NN-descent with UMAP
#         }
#     }

#     # 2D UMAP embedding
#     umap_2d = cumlUMAP(**umap_params)
#     umap_scores2D = umap_2d.fit_transform(ect_data, data_on_host=True)
#     save_umap_results(umap_scores2D, output_npz_path2D)
#     print(f"2D UMAP results saved to {output_npz_path2D}")

#     # Update for 3D embedding
#     umap_params['n_components'] = 3
#     umap_3d = cumlUMAP(**umap_params)
#     umap_scores3D = umap_3d.fit_transform(ect_data, data_on_host=True)
#     save_umap_results(umap_scores3D, output_npz_path3D)
#     print(f"3D UMAP results saved to {output_npz_path3D}")

#     return ect_data, group_labels, shapes, component_names, umap_scores2D, umap_scores3D


def run_umap_with_shapes(ect_data, group_labels_all, shapes, component_names, output_dir, umap_scores2D, bin_by_class='fullname', suffix=None):
    """
    Plot the precomputed 2D UMAP embedding with shape overlays at specific points.
    """
    SIZE_PT = 5
    ALPHA = 0.2
    # Extract class information based on the specified bin_by_class (e.g., family, genus, etc.)
    overall_family = ast.literal_eval(group_labels_all[0])['family']
    group_labels = []
    for g_str in group_labels_all:
        g = ast.literal_eval(g_str)
        group_labels.append(g[bin_by_class])

    # Ensure the number of group labels matches the number of ECT matrices
    if len(group_labels) != len(ect_data):
        raise ValueError("The number of group labels does not match the number of ECT matrices.")

    # Map each unique taxa to a numeric value for coloring
    unique_labels = np.unique(group_labels)

    # Determine grouping
    if unique_labels.size == len(group_labels):
        by_group = False
        color = (0.5, 0.5, 0.5)
        marker = 'o'
    else:
        by_group = True
        cmap = plt.get_cmap('tab20')
        label_to_color = {label: i for i, label in enumerate(unique_labels)}
        marker_types = ['o', '^', 's', '*']

    # Use precomputed UMAP scores
    tree_umap = umap_scores2D

    # Scaling and shape parameters
    scale_val = 0.05 * (np.max(tree_umap[:, 0]) - np.min(tree_umap[:, 0]))
    shape_scale_factor = 0.2

    # Create the figure
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot the 2D UMAP embedding as scatter points
    unique_labels = np.unique(group_labels)
    label_to_color = {label: i for i, label in enumerate(unique_labels)}
    
    legend_handles = []
    legend_labels = []

    if not by_group:
        # Scatter plot all points in medium gray
        ax.scatter(
            tree_umap[:, 0], tree_umap[:, 1],
            s=SIZE_PT,
            color=color,
            edgecolor='none',
            alpha=1,
            marker=marker,
        )

        legend_handles.append(plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, markersize=15))
        legend_labels.append("All points (no grouping)")

        # Define the grid across all UMAP points
        x_min, x_max = np.min(tree_umap[:, 0]), np.max(tree_umap[:, 0])
        y_min, y_max = np.min(tree_umap[:, 1]), np.max(tree_umap[:, 1])
        x_grid = np.linspace(x_min, x_max, 7)
        y_grid = np.linspace(y_min, y_max, 7)

        already_have = []
        for x in x_grid:
            for y in y_grid:
                distances = np.linalg.norm(tree_umap - np.array([x, y]), axis=1)
                closest_idx = np.argmin(distances)

                points = shapes[closest_idx]
                points = points - np.mean(points, axis=0)
                points = scale_val * points / max(np.linalg.norm(points, axis=1))
                points *= shape_scale_factor
                trans_sh = points + tree_umap[closest_idx]

                if closest_idx not in already_have:
                    already_have.append(closest_idx)
                    ax.fill(trans_sh[:, 0], 
                            trans_sh[:, 1], 
                            color='black', 
                            lw=0.1, 
                            edgecolor='black', 
                            alpha=0.5)

        ax.set_aspect("equal", adjustable='box')
        plt.title("UMAP of Leaf Contours - All Points")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
    else:
        for taxa in unique_labels:
            species_idx = [idx for idx, label in enumerate(group_labels) if label == taxa]
            taxa_umap_points = tree_umap[species_idx]

            marker = marker_types[(label_to_color[taxa] // 20) % len(marker_types)]
            
            ax.scatter(
                taxa_umap_points[:, 0], taxa_umap_points[:, 1],
                s=SIZE_PT,
                color=plt.cm.tab20(label_to_color[taxa] % 20),
                edgecolor='none',
                alpha=ALPHA,
                label=taxa,
                marker=marker,
            )
            
            legend_handles.append(plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=cmap(label_to_color[taxa] % 20), markersize=15))
            legend_labels.append(taxa)

        min_samples = 5
        for taxa in unique_labels:
            species_idx = [idx for idx, label in enumerate(group_labels) if label == taxa]
            taxa_umap_points = tree_umap[species_idx]

            if taxa_umap_points.shape[0] < min_samples:
                print(f"Skipping {taxa} KDE due to insufficient samples")
                continue
            else:
                kde = gaussian_kde(taxa_umap_points.T)
                density_values = kde(taxa_umap_points.T)
                density_threshold = np.percentile(density_values, 80)
                dense_cluster_indices = np.where(density_values >= density_threshold)[0]

                densest_point_idx = np.argmax(density_values)
                densest_point = taxa_umap_points[densest_point_idx]
                closest_idx_within_taxa = species_idx[densest_point_idx]

                points = shapes[closest_idx_within_taxa]
                points = points - np.mean(points, axis=0)
                points = scale_val * points / max(np.linalg.norm(points, axis=1))
                points *= shape_scale_factor
                trans_dense = points + densest_point

                ax.fill(trans_dense[:, 0], 
                        trans_dense[:, 1], 
                        color=cmap(label_to_color[taxa] % 20), 
                        lw=0.1, 
                        edgecolor='black', 
                        alpha=1)

        ax.set_aspect('equal', adjustable='box')
        plt.title(f"UMAP of Leaf Contours - All Points with Outlines for Taxa = {overall_family}")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")

    ax.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), title="Taxa", ncol=2)

    if suffix:
        output_2d_path = os.path.join(output_dir, f"umap_2d_with_shapes_by_{bin_by_class}{suffix}.png")
    else:
        output_2d_path = os.path.join(output_dir, f"umap_2d_with_shapes_by_{bin_by_class}.png")
    plt.savefig(output_2d_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved UMAP 2D plot with shapes to {output_2d_path}")



def run_umap_grid_by_taxa(ect_data, group_labels_all, shapes, output_dir, umap_scores, bin_by_class='fullname'):
    # "family" "genus" "genus_species" "fullname"
    SIZE_PT = 5
    ALPHA = 0.5

    group_labels = []
    for g_str in group_labels_all:
        g = ast.literal_eval(g_str)  # Convert string to dict
        group_labels.append(g[bin_by_class])

    output_pdf_path = os.path.join(output_dir, f"umap_2d_grid_by_{bin_by_class}.pdf")

    unique_labels = np.unique(group_labels)

    # Detect if every label is unique (no grouping is needed)
    if unique_labels.size == len(group_labels):
        by_group = False
        color = (0.5, 0.5, 0.5)  # Use medium gray for all points
        marker = 'o'  # Single marker type
    else:
        by_group = True
        cmap = plt.get_cmap('tab20')
        label_to_color = {label: i for i, label in enumerate(unique_labels)}
        marker_types = ['o', '^', 's', '*']

    scale_val = 0.05 * (np.max(umap_scores[:, 0]) - np.min(umap_scores[:, 0]))  # Scale relative to UMAP data range
    shape_scale_factor = 0.5  # Adjust shape size if necessary

    legend_handles = []
    legend_labels = []
    
    # Create a PDF object to store multiple plots
    with PdfPages(output_pdf_path) as pdf:
        if not by_group:
            # No grouping, plot all points in gray
            plt.figure(figsize=(24, 16))
            ax = plt.gca()

            ax.scatter(
                umap_scores[:, 0], 
                umap_scores[:, 1], 
                s=10, 
                color=color,
                edgecolor='none',
                alpha=0.8, 
                marker=marker
            )

            # Define grid for shapes
            x_min, x_max = np.min(umap_scores[:, 0]), np.max(umap_scores[:, 0])
            y_min, y_max = np.min(umap_scores[:, 1]), np.max(umap_scores[:, 1])
            x_grid = np.linspace(x_min, x_max, 7)
            y_grid = np.linspace(y_min, y_max, 7)

            already_have = []
            for x in x_grid:
                for y in y_grid:
                    distances = np.linalg.norm(umap_scores - np.array([x, y]), axis=1)
                    closest_idx = np.argmin(distances)

                    points = shapes[closest_idx]
                    points = points - np.mean(points, axis=0)
                    points = scale_val * points / max(np.linalg.norm(points, axis=1))
                    points *= shape_scale_factor
                    trans_sh = points + umap_scores[closest_idx]

                    if closest_idx not in already_have:
                        already_have.append(closest_idx)
                        plt.fill(trans_sh[:, 0], trans_sh[:, 1], color='black', lw=0, alpha=0.5)

            plt.gca().set_aspect("equal", adjustable='box')
            plt.title("UMAP of Leaf Contours - All Points")
            plt.xlabel("UMAP Dimension 1")
            plt.ylabel("UMAP Dimension 2")
            pdf.savefig()
            plt.close()

        else:
            min_samples = 5

            for i, taxa in tqdm(enumerate(unique_labels), total=len(unique_labels), desc="Processing Taxa"):
                species_idx = [idx for idx, label in enumerate(group_labels) if label == taxa]
                taxa_umap_points = umap_scores[species_idx]

                plt.figure(figsize=(24, 16))
                ax = plt.gca()

                ax.scatter(
                    umap_scores[:, 0], umap_scores[:, 1],
                    s=10,  
                    color='lightgray',
                    edgecolor='none',#'lightgray',
                    alpha=ALPHA
                )

                marker = marker_types[(label_to_color[taxa] // 20) % len(marker_types)]
                ax.scatter(
                    taxa_umap_points[:, 0], taxa_umap_points[:, 1],
                    s=SIZE_PT,
                    color=plt.cm.tab20(label_to_color[taxa] % 20),
                    edgecolor='none',#plt.cm.tab20(label_to_color[taxa] % 20),
                    alpha=ALPHA,
                    label=taxa,
                    marker=marker
                )

                if taxa_umap_points.shape[0] < min_samples:
                    print(f"Skipping {taxa} KDE due to insufficient samples")
                else:
                    kde = gaussian_kde(taxa_umap_points.T)
                    density_values = kde(taxa_umap_points.T)

                    density_threshold = np.percentile(density_values, 80)
                    dense_cluster_indices = np.where(density_values >= density_threshold)[0]

                    densest_point_idx = np.argmax(density_values)
                    densest_point = taxa_umap_points[densest_point_idx]
                    closest_idx_within_taxa = species_idx[densest_point_idx]

                    points = shapes[closest_idx_within_taxa]
                    points = points - np.mean(points, axis=0)
                    points = scale_val * points / max(np.linalg.norm(points, axis=1))
                    points *= shape_scale_factor
                    trans_dense = points + densest_point

                    plt.fill(trans_dense[:, 0], trans_dense[:, 1], color=cmap(label_to_color[taxa] % 20), lw=1, edgecolor='black', alpha=1)

                    x_min, x_max = np.min(taxa_umap_points[:, 0]), np.max(taxa_umap_points[:, 0])
                    y_min, y_max = np.min(taxa_umap_points[:, 1]), np.max(taxa_umap_points[:, 1])
                    x_grid = np.linspace(x_min, x_max, 7)
                    y_grid = np.linspace(y_min, y_max, 7)

                    already_have = []
                    for x in x_grid:
                        for y in y_grid:
                            distances = np.linalg.norm(taxa_umap_points - np.array([x, y]), axis=1)
                            closest_idx_within_taxa = np.argmin(distances)

                            points = shapes[species_idx[closest_idx_within_taxa]]
                            points = points - np.mean(points, axis=0)
                            points = scale_val * points / max(np.linalg.norm(points, axis=1))
                            points *= shape_scale_factor
                            trans_sh = points + taxa_umap_points[closest_idx_within_taxa]

                            if closest_idx_within_taxa not in already_have:
                                already_have.append(closest_idx_within_taxa)
                                plt.fill(trans_sh[:, 0], trans_sh[:, 1], color=cmap(label_to_color[taxa] % 20), lw=0, alpha=0.5)

                    plt.fill(trans_dense[:, 0], trans_dense[:, 1], color=cmap(label_to_color[taxa] % 20), lw=1, edgecolor='black', alpha=1)

                plt.gca().set_aspect("equal", adjustable='box')
                plt.title(f"UMAP of Leaf Contours - {taxa} Highlighted")
                plt.xlabel("UMAP Dimension 1")
                plt.ylabel("UMAP Dimension 2")
                pdf.savefig()
                plt.close()

    print(f"All UMAP plots saved to {output_pdf_path}")




def run_umap_simple(ect_data, group_labels_all, shapes, component_names, output_dir, umap_scores_2D, umap_scores_3D, bin_by_class='fullname'):
    SIZE_PT = 5
    ALPHA = 0.2

    overall_family = ast.literal_eval(group_labels_all[0])['family']

    # Extract labels based on the specified bin_by_class (e.g., family, genus, etc.)
    group_labels = []
    for g_str in group_labels_all:
        g = ast.literal_eval(g_str)
        group_labels.append(g[bin_by_class])

    # Check that the number of group labels matches the number of ECT matrices
    if len(group_labels) != len(ect_data):
        raise ValueError("The number of group labels does not match the number of ECT matrices.")

    # Map each unique taxa to a color
    unique_labels = np.unique(group_labels)

    # Create colors for the points in the UMAP plot
    colors = np.full((umap_scores_3D.shape[0], 4), (0.5, 0.5, 0.5, 0.2))  # Light gray RGBA for all points

    if unique_labels.size == len(group_labels):
        # No groups, so all points are the same color
        by_group = False
        color = (0.5, 0.5, 0.5)  # Medium gray
        marker = 'o'
    else:
        by_group = True
        cmap = plt.get_cmap('tab20')
        label_to_color = {label: cmap(i % 20) for i, label in enumerate(unique_labels)}
        marker_types = ['o', '^', 's', '*']
        
        for i, label in enumerate(group_labels):
            if label in label_to_color:
                colors[i] = label_to_color[label]  # Assign RGBA color for each taxa

    # Save the 2D UMAP plot
    output_2d_path = os.path.join(output_dir, f"umap_2d_by_{bin_by_class}.png")
    fig, ax = plt.subplots(figsize=(18, 12))

    if not by_group:
        ax.scatter(
            umap_scores_2D[:, 0],
            umap_scores_2D[:, 1],
            s=SIZE_PT,
            marker=marker,
            color=color,
            edgecolor='none',
            alpha=ALPHA
        )
        plt.savefig(output_2d_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved UMAP 2D plot to {output_2d_path}")
    else:
        for i, label in enumerate(unique_labels):
            species_idx = [idx for idx, label_ in enumerate(group_labels) if label_ == label]
            taxa_umap_points = umap_scores_2D[species_idx]
            marker = marker_types[i % len(marker_types)]
            ax.scatter(
                taxa_umap_points[:, 0],
                taxa_umap_points[:, 1],
                s=SIZE_PT,
                marker=marker,
                color=label_to_color[label],
                edgecolor='none',
                alpha=ALPHA,
                label=label
            )
        
        legend_handles = [
            plt.Line2D([0], [0], marker=marker_types[(i // 20) % len(marker_types)], color='w',
                       markerfacecolor=label_to_color[label], markersize=10, label=label)
            for i, label in enumerate(unique_labels)
        ]
        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), title='Legend', ncol=2)
        plt.savefig(output_2d_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved UMAP 2D plot to {output_2d_path}")

    # Save the 3D UMAP plot
    output_3d_path = os.path.join(output_dir, f"umap_3d_by_{bin_by_class}.png")
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    if not by_group:
        ax.scatter(
            umap_scores_3D[:, 0], umap_scores_3D[:, 1], umap_scores_3D[:, 2],
            s=SIZE_PT,
            marker=marker,
            color=color,
            alpha=ALPHA,
            edgecolors='none'
        )
    else:
        for i, label in enumerate(unique_labels):
            species_idx = [idx for idx, label_ in enumerate(group_labels) if label_ == label]
            taxa_umap_points = umap_scores_3D[species_idx]
            marker = marker_types[i % len(marker_types)]
            ax.scatter(
                taxa_umap_points[:, 0], taxa_umap_points[:, 1], taxa_umap_points[:, 2],
                s=5,
                marker=marker,
                color=label_to_color[label],
                alpha=0.5,
                edgecolors='none'
            )
        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), title='Legend')

    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_zlabel("UMAP Dimension 3")
    plt.savefig(output_3d_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved UMAP 3D plot to {output_3d_path}")

    # Generate rotating 3D plot
    plot_rotating_umap(umap_scores_3D, colors, label_to_color, marker_types, unique_labels, output_dir, overall_family, bin_by_class)
    plot_umap_panel(umap_scores_3D, colors, label_to_color, marker_types, unique_labels, output_dir, overall_family, bin_by_class)

# def plot_rotating_umap(umap_scores_3D, colors, label_to_color, marker_types, unique_labels, output_dir, overall_family, bin_by_class):
#     """
#     Generates a rotating 3D scatter plot for UMAP.
#     """
#     SIZE_PT = 5
#     ALPHA = 0.2

#     # Prepare legend
#     fig_legend, ax_legend = plt.subplots(figsize=(5, 8))
#     ax_legend.set_axis_off()
#     legend_handles = [
#         plt.Line2D([0], [0], marker=marker_types[(i // 20) % len(marker_types)], color='w',
#                    markerfacecolor=label_to_color[label], markersize=10, label=label)
#         for i, label in enumerate(unique_labels)
#     ]
#     ax_legend.legend(handles=legend_handles, loc='center', title="Legend", ncol=1, frameon=False)
#     legend_path = os.path.join(output_dir, f"legend_UMAP_by_{bin_by_class}.png")
#     fig_legend.savefig(legend_path, bbox_inches='tight', dpi=100)
#     plt.close(fig_legend)

#     output_gif_path = os.path.join(output_dir, f"umap_3d_with_legend_by_{bin_by_class}.gif")
#     scprep.plot.rotate_scatter3d(
#         data=umap_scores_3D,
#         c=colors,
#         s=SIZE_PT,
#         alpha=ALPHA,
#         figsize=(10, 10),
#         filename=output_gif_path,
#         rotation_speed=30,
#         fps=10,
#         edgecolors='none'
#     )
#     print(f"Saved rotating 3D UMAP plot with legend to {output_gif_path}")
def plot_rotating_umap(umap_scores_3D, colors, label_to_color, marker_types, unique_labels, output_dir, overall_family, bin_by_class):
    """
    Generates a rotating 3D scatter plot for UMAP.
    """
    SIZE_PT = 5
    ALPHA = 0.2

    # Prepare legend
    fig_legend, ax_legend = plt.subplots(figsize=(5, 8))
    ax_legend.set_axis_off()
    legend_handles = [
        plt.Line2D([0], [0], marker=marker_types[(i // 20) % len(marker_types)], color='w',
                   markerfacecolor=label_to_color[label], markersize=10, label=label)
        for i, label in enumerate(unique_labels)
    ]
    ax_legend.legend(handles=legend_handles, loc='center', title="Legend", ncol=1, frameon=False)
    legend_path = os.path.join(output_dir, f"legend_UMAP_by_{bin_by_class}.png")
    fig_legend.savefig(legend_path, bbox_inches='tight', dpi=100)
    plt.close(fig_legend)

    # Plot 3D scatter and create rotation animation
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(umap_scores_3D[:, 0], umap_scores_3D[:, 1], umap_scores_3D[:, 2],
                         c=colors, s=SIZE_PT, alpha=ALPHA, edgecolors='none')

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")

    # Rotation function
    def rotate(angle):
        ax.view_init(30, angle)

    anim = FuncAnimation(fig, rotate, frames=range(0, 360, 2), interval=50)

    # Save as GIF
    output_gif_path = os.path.join(output_dir, f"umap_3d_with_legend_by_{bin_by_class}.gif")
    anim.save(output_gif_path, writer='pillow', fps=10)
    print(f"Saved rotating 3D UMAP plot with legend to {output_gif_path}")
    plt.close(fig)

def plot_umap_panel(umap_scores_3D, colors, label_to_color, marker_types, unique_labels, output_dir, overall_family, bin_by_class):
    """
    Creates a 3D panel with different UMAP views and legend.
    """
    SIZE_PT = 5
    ALPHA = 0.2

    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(2, 3, width_ratios=[2, 2, 2])
    view_angles = [(30, 30), (30, 150), (60, 45), (60, 210)]
    titles = ["View 1", "View 2", "View 3", "View 4"]

    for i, (elev, azim) in enumerate(view_angles):
        ax_plot = fig.add_subplot(gs[i // 2, i % 2], projection='3d')
        ax_plot.scatter(umap_scores_3D[:, 0], umap_scores_3D[:, 1], umap_scores_3D[:, 2], c=colors, s=SIZE_PT, alpha=ALPHA, edgecolors='none')
        ax_plot.view_init(elev=elev, azim=azim)
        ax_plot.set_title(titles[i])
        ax_plot.set_xlabel("UMAP Dimension 1")
        ax_plot.set_ylabel("UMAP Dimension 2")
        ax_plot.set_zlabel("UMAP Dimension 3")

    ax_legend = fig.add_subplot(gs[:, 2])
    ax_legend.set_axis_off()
    legend_handles = [
        plt.Line2D([0], [0], marker=marker_types[(i // 20) % len(marker_types)], color='w',
                   markerfacecolor=label_to_color[label], markersize=10, label=label)
        for i, label in enumerate(unique_labels)
    ]
    ax_legend.legend(handles=legend_handles, loc='center', title="Legend", ncol=2, frameon=False)

    plt.suptitle(f"UMAP of Leaf Contours - Taxa = {overall_family}", fontsize=16)
    output_image_path = os.path.join(output_dir, f"umap_3d_panel_with_legend_{bin_by_class}.png")
    plt.savefig(output_image_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"Saved 3D UMAP panel with views and legend to {output_image_path}")




import os
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
import scprep

# def run_umap_heatmap(umap_scores_3D, group_labels_all, output_dir, bin_by_class='fullname'):
#     """
#     Generate a rotating 3D heatmap for each class using KDE on the UMAP embedding and save it as a GIF.
    
#     Parameters:
#         umap_scores_3D: The precomputed UMAP embedding (n_samples x 3).
#         group_labels_all: The group labels for each point in UMAP space (as string representations of dicts).
#         output_dir: Directory where the heatmaps will be saved.
#         bin_by_class: The level of classification to bin by (e.g., "family", "genus", "genus_species", "fullname").
#     """
#     # Extract group labels from the provided string representation
#     group_labels = []
#     for g_str in group_labels_all:
#         g = ast.literal_eval(g_str)  # Convert string to dict
#         group_labels.append(g[bin_by_class])

#     # Get unique labels for the classes
#     unique_labels = np.unique(group_labels)

#     # Create a rotating 3D heatmap for each class
#     for label in unique_labels:
#         # Select the points corresponding to this label
#         class_idx = [i for i, l in enumerate(group_labels) if l == label]
#         class_points = umap_scores_3D[class_idx]

#         if class_points.shape[0] < 5:
#             continue  # Skip if not enough points for KDE

#         # Perform KDE on the 3D UMAP points for the actual class points
#         kde = gaussian_kde(class_points.T)
#         densities = kde(class_points.T)

#         # Normalize the density for better visualization
#         densities /= np.max(densities)

#         # Get the BuGn colormap and map the density values to colors
#         cmap = plt.get_cmap('winter')#.reversed() # Blue is least dense, green is most dense
#         bu_gn_colors = cmap(densities)  # Get the colormap values for the class points

#         # Create a color array for all points: gray for other points, density-based colors for the current class
#         class_colors = np.full((umap_scores_3D.shape[0], 4), (0.5, 0.0, 0.0, 0.05))  # Initialize as light gray (RGBA)
        
#         # Assign BuGn colormap values to the current class points
#         for i, idx in enumerate(class_idx):
#             class_colors[idx] = bu_gn_colors[i]  # Assign the BuGn color based on KDE density

#         # Create a rotating 3D GIF for the current label
#         output_gif_path = os.path.join(output_dir, f"umap_3d_rotation_heatmap_{label}.gif")

#         # Generate the rotating 3D scatter plot using scprep's built-in function
#         scprep.plot.rotate_scatter3d(
#             data=umap_scores_3D, 
#             c=class_colors,  # Use the color array with BuGn for the current class and light gray for others
#             figsize=(12, 12), 
#             filename=output_gif_path, 
#             rotation_speed=30,
#             fps=10,
#             elev=30
#         )

#         plot_umap_panel_heatmap(umap_scores_3D, class_colors, label, output_dir)

'''
def run_umap_heatmap(umap_scores_3D, group_labels_all, output_dir, bin_by_class='fullname'):
    """
    Generate a rotating 3D heatmap for each class using KDE on the UMAP embedding and save it as a GIF.
    
    Parameters:
        umap_scores_3D: The precomputed UMAP embedding (n_samples x 3).
        group_labels_all: The group labels for each point in UMAP space (as string representations of dicts).
        output_dir: Directory where the heatmaps will be saved.
        bin_by_class: The level of classification to bin by (e.g., "family", "genus", "genus_species", "fullname").
    """
    # Extract group labels from the provided string representation
    group_labels = [ast.literal_eval(g_str)[bin_by_class] for g_str in group_labels_all]
    unique_labels = np.unique(group_labels)

    # Create a rotating 3D heatmap for each class
    for label in unique_labels:
        # Select the points corresponding to this label
        class_idx = [i for i, l in enumerate(group_labels) if l == label]
        class_points = umap_scores_3D[class_idx]

        if class_points.shape[0] < 5:
            continue  # Skip if not enough points for KDE

        # Perform KDE on the 3D UMAP points for the actual class points
        kde = gaussian_kde(class_points.T)
        densities = kde(class_points.T)
        densities /= np.max(densities)  # Normalize density for visualization

        # Map density values to colors using the "winter" colormap
        cmap = plt.get_cmap('winter')
        bu_gn_colors = cmap(densities)  # Colors for the class points based on density

        # Initialize color array for all points, setting other points to light gray
        class_colors = np.full((umap_scores_3D.shape[0], 4), (0.5, 0.0, 0.0, 0.05))  # Light gray in RGBA
        for i, idx in enumerate(class_idx):
            class_colors[idx] = bu_gn_colors[i]  # Assign color based on density for current class

        # Create 3D plot and rotating animation
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(umap_scores_3D[:, 0], umap_scores_3D[:, 1], umap_scores_3D[:, 2],
                             c=class_colors, s=5, edgecolors='none')
        
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")

        # Rotation function for the animation
        def rotate(angle):
            ax.view_init(30, angle)

        # Create the rotating animation
        anim = FuncAnimation(fig, rotate, frames=range(0, 360, 2), interval=50)
        
        # Save animation as GIF
        output_gif_path = os.path.join(output_dir, f"umap_3d_rotation_heatmap_{label}.gif")
        anim.save(output_gif_path, writer='pillow', fps=10)
        plt.close(fig)
        
        print(f"Saved rotating 3D UMAP heatmap for class {label} to {output_gif_path}")
        plot_umap_panel_heatmap(umap_scores_3D, class_colors, label, output_dir)
        
def plot_umap_panel_heatmap(umap_scores_3D, class_colors, label, output_dir):
    """
    Plot a 2x2 panel of UMAP 3D views from different angles for a given class.
    
    Parameters:
        umap_scores_3D: The precomputed UMAP embedding (n_samples x 3).
        class_colors: Colors assigned to each point, based on KDE density or class coloring.
        label: The label for the current class, used in the plot title and file name.
        output_dir: Directory where the panel plot will be saved.
    """
    SIZE_PT = 5
    ALPHA = 0.2

    # Set up GridSpec for 2x2 panel of 3D views
    fig = plt.figure(figsize=(18, 18))
    gs = GridSpec(2, 2)

    # Define different view angles for the 3D plot
    view_angles = [(30, 30), (30, 150), (60, 45), (60, 210)]
    titles = ["View 1", "View 2", "View 3", "View 4"]

    # Loop through each view and create a 3D scatter plot with different angles
    for i, (elev, azim) in enumerate(view_angles):
        ax = fig.add_subplot(gs[i // 2, i % 2], projection='3d')
        ax.scatter(umap_scores_3D[:, 0], 
                   umap_scores_3D[:, 1], 
                   umap_scores_3D[:, 2],
                   c=class_colors, 
                   s=SIZE_PT, 
                   alpha=ALPHA, 
                   edgecolor='none')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(titles[i])
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.set_zlabel("UMAP Dimension 3")

    # Save the panel plot
    output_image_path = os.path.join(output_dir, f"umap_3d_panel_{label}.png")
    plt.suptitle(f"UMAP of Leaf Contours - Class: {label}", fontsize=16)
    plt.savefig(output_image_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved 3D panel plot with different views for class '{label}' to {output_image_path}")
'''



def run_umap_heatmap(umap_scores_3D, group_labels_all, output_dir, bin_by_class='fullname'):
    """
    Generate a rotating 3D heatmap for each class using KDE on the UMAP embedding and save it as a GIF.
    """
    group_labels = [ast.literal_eval(g_str)[bin_by_class] for g_str in group_labels_all]
    unique_labels = np.unique(group_labels)

    for label in unique_labels:
        class_idx = [i for i, l in enumerate(group_labels) if l == label]
        class_points = umap_scores_3D[class_idx]

        if class_points.shape[0] < 5:
            continue

        kde = gaussian_kde(class_points.T)
        densities = kde(class_points.T)
        densities /= np.max(densities)

        cmap = plt.get_cmap('winter')
        bu_gn_colors = cmap(densities)

        class_colors = np.full((umap_scores_3D.shape[0], 4), (0.5, 0.0, 0.0, 0.05))
        for i, idx in enumerate(class_idx):
            class_colors[idx] = bu_gn_colors[i]

        plot_umap_panel_heatmap(umap_scores_3D, class_colors, class_idx, label, output_dir)


def plot_umap_panel_heatmap(umap_scores_3D, class_colors, class_idx, label, output_dir):
    """
    Plot a 2x4 panel of UMAP 3D views, with class points on the left and non-class points on the right for each view.
    """
    SIZE_PT = 5
    SIZE_PT2 = 1
    ALPHA = 0.2
    ALPHA2 = 0.05
    non_class_idx = np.setdiff1d(np.arange(len(umap_scores_3D)), class_idx)
    non_class_colors = np.full((len(non_class_idx), 4), (0.5, 0.0, 0.0, ALPHA2))  # 

    fig = plt.figure(figsize=(24, 12))  # Larger figure to accommodate 8 subplots
    gs = GridSpec(2, 4)  # 2 rows, 4 columns for 8 subplots in total

    # Define different view angles for the 3D plot
    view_angles = [(30, 30), (30, 150), (60, 45), (60, 210)]
    titles = ["View 1", "View 2", "View 3", "View 4"]

    for i, (elev, azim) in enumerate(view_angles):
        # Left subplot for each view: class points only
        ax_class = fig.add_subplot(gs[i // 2, (i % 2) * 2], projection='3d')
        ax_class.scatter(umap_scores_3D[class_idx, 0], umap_scores_3D[class_idx, 1], umap_scores_3D[class_idx, 2],
                         c=class_colors[class_idx], s=SIZE_PT, alpha=ALPHA, edgecolor='none')
        ax_class.view_init(elev=elev, azim=azim)
        ax_class.set_title(f"{titles[i]} (Class Points)")
        ax_class.set_xlabel("UMAP Dimension 1")
        ax_class.set_ylabel("UMAP Dimension 2")
        ax_class.set_zlabel("UMAP Dimension 3")

        # Right subplot for each view: non-class points only
        ax_non_class = fig.add_subplot(gs[i // 2, (i % 2) * 2 + 1], projection='3d')
        ax_non_class.scatter(umap_scores_3D[non_class_idx, 0], umap_scores_3D[non_class_idx, 1], umap_scores_3D[non_class_idx, 2],
                             c=non_class_colors, s=SIZE_PT, alpha=ALPHA, edgecolor='none')
        ax_non_class.view_init(elev=elev, azim=azim)
        ax_non_class.set_title(f"{titles[i]} (Non-Class Points)")
        ax_non_class.set_xlabel("UMAP Dimension 1")
        ax_non_class.set_ylabel("UMAP Dimension 2")
        ax_non_class.set_zlabel("UMAP Dimension 3")

    output_image_path = os.path.join(output_dir, f"umap_3d_panel_{label}.png")
    plt.suptitle(f"UMAP of Leaf Contours - Class: {label}", fontsize=16)
    plt.savefig(output_image_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved 3D panel plot with class and non-class views for '{label}' to {output_image_path}")