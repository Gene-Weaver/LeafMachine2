import os, ast
import phate
import matplotlib.pyplot as plt
import sklearn.decomposition  # PCA
import sklearn.manifold  # MDS, t-SNE
import time
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from scipy.stats import gaussian_kde
import scprep
from PIL import Image, ImageDraw, ImageSequence
import h5py
from sklearn.metrics import pairwise_distances
from scipy.sparse import lil_matrix
from joblib import Parallel, delayed

currentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parentdir = os.path.dirname(currentdir)
try:
    from leafmachine2.ect_methods.utils_tsne import CustomLegendShape, HandlerShape
except:
    from utils_tsne import CustomLegendShape, HandlerShape


def load_from_hdf5(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        # Load ECT matrices
        ect_data = [f[f'ECT_matrices/matrix_{i}'][:] for i in range(len(f['ECT_matrices']))]
        
        # Load group labels
        group_labels = [label.decode() for label in f['group_labels'][:]]
        
        # Load shapes
        shapes = [f[f'shapes/shape_{i}'][:] for i in range(len(f['shapes']))]

        # Load component names
        component_names = [name.decode() for name in f['component_names'][:]]
    
    return ect_data, group_labels, shapes, component_names

def load_direct(dir_path):
    """
    Collate ECT data for a single family directory and save as an HDF5 and NumPy file.
    Parameters:
        dir_path (str): The directory containing the ECT HDF5 files for a family.
    """
    family_name = os.path.basename(os.path.dirname(dir_path))

    if "LM2_" in family_name:
        family_name = os.path.basename(os.path.dirname(os.path.dirname(dir_path)))

    hdf5_dir_path = os.path.join(dir_path, 'Data', 'Measurements', 'ECT')

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
            ect_data, group_labels, shapes, component_names = load_from_hdf5(hdf5_file_path)

            # Append data to directory-specific lists
            family_ect_matrices.extend(ect_data)
            family_group_labels.extend(group_labels)
            family_shapes.extend(shapes)
            family_component_names.extend(component_names)

        except Exception as e:
            print(f"Error processing file {hdf5_file_path}: {e}")
    
    return family_ect_matrices, family_group_labels, family_shapes, family_component_names

def save_phate_results(phate_scores, output_path):
    """Save PHATE scores to an npz file."""
    np.savez(output_path, phate_scores=phate_scores)

def load_phate_results(npz_file_path):
    """Load PHATE scores from an npz file."""
    data = np.load(npz_file_path)
    return data['phate_scores']

def compute_phate(output_npz_path2D, output_npz_path3D, ect_data, group_labels, shapes, component_names):
    """Check if PHATE results exist; if not, compute and save them."""
    if os.path.exists(output_npz_path2D):
        print(f"Loading PHATE results from {output_npz_path2D}")
        phate_scores2D = load_phate_results(output_npz_path2D)
        print(f"Loading PHATE results from {output_npz_path3D}")
        phate_scores3D = load_phate_results(output_npz_path3D)

    else:
        # Get the directory of the output path for saving intermediate results
        output_dir = os.path.dirname(output_npz_path2D)
        output_dir2 = os.path.dirname(os.path.dirname(output_npz_path2D))
        distance_matrix_path = os.path.join(output_dir2, "distance_matrix.h5")

        print(f"Computing PHATE results and saving to {output_npz_path2D}")
        
        # ect_data_flat = [matrix.flatten() for matrix in ect_data]  # Flatten ECT matrices for PHATE
        # distance_matrix = pairwise_distances(ect_data_flat, 
        #                                      metric='euclidean',
        #                                      random_state=2024,
        #                                      n_jobs=-2,
        #                                      )

        # Calculate pairwise distance matrix if not already calculated
        if os.path.exists(distance_matrix_path):
            # Load the distance matrix from file
            print("Loading existing distance matrix.")
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
    

        # ect_data_sparse = [csr_matrix(matrix.flatten()) for matrix in ect_data]
        # ect_data_sparse_stacked = vstack(ect_data_sparse)

        # # Step 3: Compute the pairwise distances on the sparse matrix
        # distance_matrix = pairwise_distances(
        #     ect_data_sparse_stacked,
        #     metric='euclidean',
        #     n_jobs=-2,
        # )

        phate_operator2D = phate.PHATE(n_components=2,
                                     knn=5,
                                     knn_max=25, # For speed
                                    #  knn_dist="euclidean",
                                     knn_dist='precomputed',  # Use precomputed distance matrix
                                     mds_dist="euclidean",
                                     mds="metric",
                                     mds_solver="sgd",
                                     n_jobs=-2, # For speed
                                     n_pca=20, # For speed
                                     random_state=2024,
                                     )

        # ect_data_array = np.array(ect_data_flat)  # Convert list of flattened matrices to a 2D array
        # phate_scores2D = phate_operator2D.fit_transform(ect_data_array)
        phate_scores2D = phate_operator2D.fit_transform(distance_matrix)
        save_phate_results(phate_scores2D, output_npz_path2D)

        print(f"Computing PHATE results and saving to {output_npz_path3D}")
        phate_operator3D = phate.PHATE(n_components=3,
                                     knn=5,
                                     knn_max=25, # For speed
                                    #  knn_dist="euclidean",
                                     knn_dist='precomputed',  # Use precomputed distance matrix
                                     mds_dist="euclidean",
                                     mds="metric",
                                     mds_solver="sgd",
                                     n_jobs=-2, # For speed
                                     n_pca=20, # For speed
                                     random_state=2024,
                                     )
        # phate_scores3D = phate_operator3D.fit_transform(ect_data_array)
        phate_scores3D = phate_operator3D.fit_transform(distance_matrix)

        save_phate_results(phate_scores3D, output_npz_path3D)
    return ect_data, group_labels, shapes, component_names, phate_scores2D, phate_scores3D

def compute_distances_in_batches(ect_data, distance_matrix_path, batch_size=10000, metric='euclidean', save_intermediate_dir=None):
    """
    Compute only the upper triangular part of the pairwise distances in batches and save to disk.
    
    Args:
        ect_data: List of matrices
        batch_size: Number of samples to process at once
        metric: Distance metric to use
        save_intermediate_dir: Directory to save the intermediate distance matrix
    """
    n_samples = len(ect_data)
    
    # Prepare the flattened data for distance calculations
    flattened_data = [matrix.flatten() for matrix in ect_data]
    batch_size = min(batch_size, n_samples)
    
    # Initialize the HDF5 file for storing the distance matrix
    if save_intermediate_dir:
        with h5py.File(distance_matrix_path, "w") as hf:
            # Create a dataset for the distance matrix
            distance_dset = hf.create_dataset("distances", shape=(n_samples, n_samples), dtype="float64")
            
            # Compute distances in batches and fill in both upper and lower triangles
            for i in tqdm(range(0, n_samples, batch_size), desc="Processing row batches"):
                batch_end = min(i + batch_size, n_samples)
                batch = np.array(flattened_data[i:batch_end])
                
                for j in range(0, n_samples, batch_size):
                    next_batch_end = min(j + batch_size, n_samples)
                    next_batch = np.array(flattened_data[j:next_batch_end])
                    
                    # Compute the distance between the current batch and the next batch
                    distances = pairwise_distances(batch, next_batch, metric=metric, n_jobs=-2)
                    
                    # Write to both upper and lower triangles of the matrix
                    distance_dset[i:batch_end, j:next_batch_end] = distances
                    if i != j:  # Avoid double writing the diagonal
                        distance_dset[j:next_batch_end, i:batch_end] = distances.T

                    del distances  # Free memory

    print(f"Distance matrix saved to {distance_matrix_path}")
    return distance_dset

def save_distance_matrix(distance_matrix, filename):
    """Save distance matrix to an npz file."""
    np.savez(filename, distance_matrix=distance_matrix)

def run_phate_simple(ect_data, group_labels_all, shapes, component_names, output_dir,  phate_scores_2D, phate_scores_3D, bin_by_class='fullname'):
    overall_family = ast.literal_eval(group_labels_all[0])['family']

    # "family" "genus" "genus_species" "fullname"
    group_labels = []
    for g_str in group_labels_all:
        g = ast.literal_eval(g_str)  # Convert string to dict
        group_labels.append(g[bin_by_class])
    """
    Run PHATE on the dataset and save 2D, 3D plots and animations to disk, with a static legend above the rotating plot.
    """
    # Perform PHATE on the ECT matrices
    phate_operator_3D = phate.PHATE(n_jobs=-2, n_components=3)


    output_2d_path = os.path.join(output_dir, f"phate_2d_{bin_by_class}.png")

    if len(group_labels) != len(ect_data):
        raise ValueError("The number of group labels does not match the number of ECT matrices.")

    # Map each unique taxa to a numeric value for coloring
    unique_labels = np.unique(group_labels)

    # Create a color array to match the number of points in phate_scores_3D
    colors = np.full((phate_scores_3D.shape[0], 4), (0.5, 0.5, 0.5, 0.2))  # Initialize all colors as light gray RGBA


    # Detect if every label is unique (no grouping is needed)
    if unique_labels.size == len(group_labels):
        by_group = False
        color = (0.5, 0.5, 0.5)  # Use medium gray for all points
        marker = 'o'  # Single marker type
    else:
        by_group = True
        cmap = plt.get_cmap('tab20')
        label_to_color = {label: cmap(i % 20) for i, label in enumerate(unique_labels)}
        marker_types = ['o', '^', 's', '*']
        # colors = [cmap(label_to_color[label] % 20) for label in group_labels]
        # Assign colors based on taxa groups
        # Update the color array for grouped points based on taxa
        for i, label in enumerate(group_labels):
            if label in label_to_color:
                colors[i] = label_to_color[label]  # Assign RGBA color for the specific taxa


    


    if not by_group:
        # Save 2D scatter plot with all points in gray
        fig, ax = plt.subplots(figsize=(18, 12))  # Increase width to 12 inches

        # Scatter plot for all points in gray
        ax.scatter(
            phate_scores_2D[:, 0], 
            phate_scores_2D[:, 1],
            s=2, 
            marker=marker, 
            color=color,  # All points in medium gray
            edgecolor='none', 
            alpha=0.2
        )

        # No legend required since there are no groups
        plt.savefig(output_2d_path, dpi=300, bbox_inches='tight')  # Use bbox_inches='tight' to prevent clipping
        plt.close()
        print(f"Saved PHATE 2D plot to {output_2d_path}")

        # Generate and save 3D scatter plot with all points in gray
        output_3d_path = os.path.join(output_dir, f"phate_3d_by_{bin_by_class}.png")
        fig = plt.figure(figsize=(18, 12))  # Increase width to 12 inches

        ax = fig.add_subplot(111, projection='3d')

        # Create the PHATE operator with 3D embedding
        # phate_operator = phate.PHATE(n_jobs=-2, n_components=3)
        
        # Fit the PHATE model and generate embeddings in 3D
        # phate_scores_3D = phate_operator.fit_transform(ect_data_array)

        # Scatter plot for all points in gray
        ax.scatter(
            phate_scores_3D[:, 0], 
            phate_scores_3D[:, 1], 
            phate_scores_3D[:, 2],  # Use 3D coordinates
            s=2, 
            marker=marker, 
            color=color, 
            alpha=0.2  # All points in medium gray
        )

        plt.savefig(output_3d_path, dpi=300, bbox_inches='tight')  # Prevent clipping
        plt.close()
        print(f"Saved PHATE 3D plot to {output_3d_path}")

        # Generate the rotating 3D scatter plot
        output_gif_path = os.path.join(output_dir, f"phate_3d_rotation_by_{bin_by_class}.gif")
        output_mp4_path = os.path.join(output_dir, f"phate_3d_rotation_by_{bin_by_class}.mp4")

        phate.plot.rotate_scatter3d(phate_operator_3D, c=None, figsize=(12, 12), filename=output_gif_path, ax=ax, legend=False)
        # phate.plot.rotate_scatter3d(phate_operator, c=None, figsize=(6, 6), filename=output_mp4_path, ax=ax)

        print(f"Saved PHATE 3D rotation to {output_gif_path} and {output_mp4_path}")


    else:
        # Save 2D scatter plot with marker types
        fig, ax = plt.subplots(figsize=(18, 12))  # Increase width to 12 inches

        # Plot each taxa with a unique marker and color
        for i, label in enumerate(unique_labels):
            species_idx = [idx for idx, label_ in enumerate(group_labels) if label_ == label]
            taxa_phate_points = phate_scores_2D[species_idx]
            marker = marker_types[i % len(marker_types)]  # Cycle through marker types

            ax.scatter(taxa_phate_points[:, 0], taxa_phate_points[:, 1],
                    s=2, 
                    marker=marker, 
                    color=label_to_color[label],
                    edgecolor='none', 
                    alpha=0.2, 
                    label=label)

        # Create a custom legend with marker types
        legend_handles = [
        plt.Line2D([0], [0], marker=marker_types[(i // 20) % len(marker_types)], color='w',
                            markerfacecolor=label_to_color[label], markersize=10, label=label)
            for i, label in enumerate(unique_labels)
        ]
        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), title='Legend', ncol=2)  # Move legend to the right
        ax.set_aspect("equal", adjustable='box')
        plt.title(f"PHATE of Leaf Contours - Taxa = {overall_family}")
        plt.xlabel("PHATE Dimension 1")
        plt.ylabel("PHATE Dimension 2")
        
        plt.savefig(output_2d_path, dpi=300, bbox_inches='tight')  # Use bbox_inches='tight' to prevent clipping
        plt.close()
        print(f"Saved PHATE 2D plot to {output_2d_path}")

        # Generate and save 3D scatter plot with marker types
        # output_3d_path = os.path.join(output_dir, "phate_3d.png")
        # fig = plt.figure(figsize=(18, 12))  # Increase width to 12 inches

        # ax = fig.add_subplot(111, projection='3d')

        # for i, label in enumerate(unique_labels):
        #     species_idx = [idx for idx, label_ in enumerate(group_labels) if label_ == label]
        #     taxa_phate_points = phate_scores_3D[species_idx]
        #     marker = marker_types[i % len(marker_types)]  # Cycle through marker types

        #     ax.scatter(taxa_phate_points[:, 0], taxa_phate_points[:, 1], taxa_phate_points[:, 2],
        #             s=5, marker=marker, color=label_to_color[label], alpha=0.5)

        # # Create a custom legend with marker types
        # legend_handles = [
        #     plt.Line2D([0], [0], marker=marker_types[i % len(marker_types)], color='w',
        #             markerfacecolor=label_to_color[taxa], markersize=10, label=taxa)
        #     for i, taxa in enumerate(unique_labels)
        # ]

        # # Add the legend to the plot
        # ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), title='Legend')

        
        # plt.savefig(output_3d_path, dpi=300, bbox_inches='tight')  # Prevent clipping
        # plt.close()
        # print(f"Saved PHATE 3D plot to {output_3d_path}")

        # Set up GridSpec for rotating 3D plot with legend
        # fig = plt.figure(figsize=(18, 12))
        # gs = GridSpec(1, 2, width_ratios=[4, 1])  # Plot on left, legend on right

        # # First subplot for rotating 3D plot
        # ax_plot = fig.add_subplot(gs[0], projection='3d')

        # # Generate rotating 3D scatter plot
        # output_gif_path = os.path.join(output_dir, "phate_3d_with_legend.gif")
        # scprep.plot.rotate_scatter3d(
        #     data=phate_scores_3D,
        #     c=colors,  # Color mapping
        #     figsize=(12, 12),
        #     filename=output_gif_path,
        #     ax=ax_plot,
        #     rotation_speed=30,
        #     fps=10
        # )

        # # Second subplot for the legend
        # ax_legend = fig.add_subplot(gs[1])
        # ax_legend.set_axis_off()  # No axis for legend

        # # Create custom legend
        # legend_handles = [
        #     plt.Line2D([0], [0], marker=marker_types[(i // 20) % len(marker_types)], color='w',
        #             markerfacecolor=label_to_color[label], markersize=10, label=label)
        #     for i, label in enumerate(unique_labels)
        # ]
        # ax_legend.legend(handles=legend_handles, loc='center', title="Legend", ncol=1, frameon=False)
        # # Set labels on the 3D plot axis
        # ax_plot.set_xlabel("PHATE Dimension 1")
        # ax_plot.set_ylabel("PHATE Dimension 2")
        # ax_plot.set_zlabel("PHATE Dimension 3")
        # ax_plot.set_aspect("auto", adjustable='box')
        # ax_plot.set_title(f"PHATE of Leaf Contours - Taxa = {overall_family}")
        # # Save the rotating plot and static legend
        # plt.savefig(os.path.join(output_dir, "phate_3d_with_legend.png"), dpi=300, bbox_inches='tight')
        # plt.close()


        plot_rotating_phate(phate_scores_3D, colors, label_to_color, marker_types, unique_labels, output_dir, overall_family, bin_by_class)
        plot_phate_panel(phate_scores_3D, colors, label_to_color, marker_types, unique_labels, output_dir, overall_family, bin_by_class)


def plot_rotating_phate(phate_scores_3D, colors, label_to_color, marker_types, unique_labels, output_dir, overall_family, bin_by_class):
    """
    Generates a rotating 3D scatter plot and saves it as a GIF with an embedded legend.
    """
    output_gif_path = os.path.join(output_dir, f"phate_3d_with_legend_by_{bin_by_class}.gif")

    # Prepare legend
    fig_legend, ax_legend = plt.subplots(figsize=(5, 8))
    ax_legend.set_axis_off()
    legend_handles = [
        plt.Line2D([0], [0], marker=marker_types[(i // 20) % len(marker_types)], color='w',
                   markerfacecolor=label_to_color[label], markersize=10, label=label)
        for i, label in enumerate(unique_labels)
    ]
    ax_legend.legend(handles=legend_handles, loc='center', title="Legend", ncol=1, frameon=False)
    legend_path = os.path.join(output_dir, f"legend_PHATE_by_{bin_by_class}.png")
    fig_legend.savefig(legend_path, bbox_inches='tight', dpi=100)
    plt.close(fig_legend)

    # Generate rotating 3D plot with scprep, save as GIF
    scprep.plot.rotate_scatter3d(
        data=phate_scores_3D,
        c=colors,
        s=2,
        alpha=0.2,
        figsize=(10, 10),
        filename=output_gif_path,
        rotation_speed=30,
        fps=10
        # dpi=300
    )

    # # Combine the rotating GIF and the legend
    # with Image.open(output_gif_path) as gif:
    #     frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    # legend_img = Image.open(legend_path)

    # combined_frames = []
    # for frame in frames:
    #     combined_frame = Image.new("RGB", (frame.width + legend_img.width, max(frame.height, legend_img.height)))
    #     combined_frame.paste(frame, (0, 0))
    #     combined_frame.paste(legend_img, (frame.width, 0))
    #     combined_frames.append(combined_frame)

    # combined_frames[0].save(output_gif_path, save_all=True, append_images=combined_frames[1:], loop=0, duration=gif.info['duration'])

    print(f"Saved rotating 3D scatter plot with legend to {output_gif_path}")

def plot_phate_panel(phate_scores_3D, colors, label_to_color, marker_types, unique_labels, output_dir, overall_family, bin_by_class):
    
    # Set up GridSpec for 2x2 panel of 3D views and 1x1 for the legend
    fig = plt.figure(figsize=(24, 18), facecolor='white')
    gs = GridSpec(2, 3, width_ratios=[2, 2, 2])  # Last column for legend only

    # Define different view angles for the 3D plot
    view_angles = [(30, 30), (30, 150), (60, 45), (60, 210)]
    titles = ["View 1", "View 2", "View 3", "View 4"]

    # Loop through each view and create a 3D scatter plot with different angles
    for i, (elev, azim) in enumerate(view_angles):
        ax_plot = fig.add_subplot(gs[i // 2, i % 2], projection='3d')
        ax_plot.scatter(phate_scores_3D[:, 0], 
                        phate_scores_3D[:, 1], 
                        phate_scores_3D[:, 2],
                        c=colors, 
                        s=2, 
                        alpha=0.2,
                        edgecolor='none')
        ax_plot.view_init(elev=elev, azim=azim)
        ax_plot.set_title(titles[i])
        
        # Set axis labels
        ax_plot.set_xlabel("PHATE Dimension 1")
        ax_plot.set_ylabel("PHATE Dimension 2")
        ax_plot.set_zlabel("PHATE Dimension 3")

    # Second subplot for the legend
    ax_legend = fig.add_subplot(gs[:, 2])  # Takes up both rows in the last column
    ax_legend.set_axis_off()  # No axis for legend

    # Create custom legend
    legend_handles = [
        plt.Line2D([0], [0], marker=marker_types[(i // 20) % len(marker_types)], color='w',
                   markerfacecolor=label_to_color[label], markersize=10, label=label)
        for i, label in enumerate(unique_labels)
    ]
    ax_legend.legend(handles=legend_handles, loc='center', title="Legend", ncol=2, frameon=False)

    # Add an overall title for the figure
    plt.suptitle(f"PHATE of Leaf Contours - Taxa = {overall_family}", fontsize=16)

    # Save the panel with 4 different views and the static legend
    output_image_path = os.path.join(output_dir, f"phate_3d_panel_with_legend_by_{bin_by_class}.png")
    plt.savefig(output_image_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"Saved 3D panel with four different views and static legend to {output_image_path}")


def run_phate_heatmap(phate_scores_3D, group_labels_all, output_dir, bin_by_class='fullname'):
    """
    Generate a rotating 3D heatmap for each class using KDE on the PHATE embedding and save it as a GIF.
    
    Parameters:
        phate_scores_3D: The precomputed PHATE embedding (n_samples x 3).
        group_labels_all: The group labels for each point in PHATE space (as string representations of dicts).
        output_dir: Directory where the heatmaps will be saved.
        bin_by_class: The level of classification to bin by (e.g., "family", "genus", "genus_species", "fullname").
    """
    # Extract group labels from the provided string representation
    group_labels = []
    for g_str in group_labels_all:
        g = ast.literal_eval(g_str)  # Convert string to dict
        group_labels.append(g[bin_by_class])

    # Get unique labels for the classes
    unique_labels = np.unique(group_labels)

    # Create a rotating 3D heatmap for each class
    for label in unique_labels:
        # Select the points corresponding to this label
        class_idx = [i for i, l in enumerate(group_labels) if l == label]
        class_points = phate_scores_3D[class_idx]

        if class_points.shape[0] < 5:
            continue  # Skip if not enough points for KDE

        # Perform KDE on the 3D PHATE points for the actual class points
        kde = gaussian_kde(class_points.T)
        densities = kde(class_points.T)

        # Normalize the density for better visualization
        densities /= np.max(densities)

        # Get the BuGn colormap and map the density values to colors
        cmap = plt.get_cmap('winter')#.reversed() # Blue is least dense, green is most dense
        bu_gn_colors = cmap(densities)  # Get the colormap values for the class points

        # Create a color array for all points: gray for other points, density-based colors for the current class
        class_colors = np.full((phate_scores_3D.shape[0], 4), (0.5, 0.0, 0.0, 0.05))  # Initialize as light gray (RGBA)
        
        # Assign BuGn colormap values to the current class points
        for i, idx in enumerate(class_idx):
            class_colors[idx] = bu_gn_colors[i]  # Assign the BuGn color based on KDE density

        # Create a rotating 3D GIF for the current label
        output_gif_path = os.path.join(output_dir, f"phate_3d_rotation_heatmap_{label}.gif")

        # Generate the rotating 3D scatter plot using PHATE's built-in function
        scprep.plot.rotate_scatter3d(
            data=phate_scores_3D, 
            c=class_colors,  # Use the color array with BuGn for the current class and light gray for others
            figsize=(12, 12), 
            filename=output_gif_path, 
            rotation_speed=30,  # Customize rotation speed if desired
            fps=10,  # Customize frames per second for smoother rotation
            elev=30  # Optional elevation angle for better visualization
        )

        plot_phate_panel_heatmap(phate_scores_3D, class_colors, label, output_dir)

def plot_phate_panel_heatmap(phate_scores_3D, class_colors, label, output_dir):
    """
    Plot a 2x2 panel of PHATE 3D views from different angles for a given class.
    
    Parameters:
        phate_scores_3D: The precomputed PHATE embedding (n_samples x 3).
        class_colors: Colors assigned to each point, based on KDE density or class coloring.
        label: The label for the current class, used in the plot title and file name.
        output_dir: Directory where the panel plot will be saved.
    """
    # Set up GridSpec for 2x2 panel of 3D views
    fig = plt.figure(figsize=(18, 18))
    gs = GridSpec(2, 2)

    # Define different view angles for the 3D plot
    view_angles = [(30, 30), (30, 150), (60, 45), (60, 210)]
    titles = ["View 1", "View 2", "View 3", "View 4"]

    # Loop through each view and create a 3D scatter plot with different angles
    for i, (elev, azim) in enumerate(view_angles):
        ax = fig.add_subplot(gs[i // 2, i % 2], projection='3d')
        ax.scatter(phate_scores_3D[:, 0], 
                   phate_scores_3D[:, 1], 
                   phate_scores_3D[:, 2],
                   c=class_colors, 
                   s=2, 
                   alpha=0.2, 
                   edgecolor='none')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(titles[i])
        ax.set_xlabel("PHATE Dimension 1")
        ax.set_ylabel("PHATE Dimension 2")
        ax.set_zlabel("PHATE Dimension 3")

    # Save the panel plot
    output_image_path = os.path.join(output_dir, f"phate_3d_panel_{label}.png")
    plt.suptitle(f"PHATE of Leaf Contours - Class: {label}", fontsize=16)
    plt.savefig(output_image_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved 3D panel plot with different views for class '{label}' to {output_image_path}")

def run_phate_dimensionality_comparison(ect_data, group_labels, branches, output_dir):
    """
    Run PCA, MDS, t-SNE, and PHATE, and compare results. Save plots to disk.
    """
    if len(group_labels) != len(ect_data):
        raise ValueError("The number of group labels does not match the number of ECT matrices.")

    # Map each unique taxa to a numeric value for coloring
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

    # Perform PHATE on the ECT matrices
    phate_operator = phate.PHATE(n_jobs=-2)
    ect_data_flat = [matrix.flatten() for matrix in ect_data]  # Flatten ECT matrices for PHATE
    ect_data_array = np.array(ect_data_flat)  # Convert list of flattened matrices to a 2D array
    tree_phate = phate_operator.fit_transform(ect_data_array)
    scale_val = 0.05 * (np.max(tree_phate[:, 0]) - np.min(tree_phate[:, 0]))  # Scale relative to PHATE data range
    shape_scale_factor = 0.2  # Reduce shape size further if needed
    

    if not by_group:
        # PCA
        start = time.time()
        pca_operator = sklearn.decomposition.PCA(n_components=2)
        tree_pca = pca_operator.fit_transform(ect_data_array)
        end = time.time()
        print(f"Embedded PCA in {end - start:.2f} seconds.")

        # t-SNE
        start = time.time()
        tsne_operator = sklearn.manifold.TSNE(n_components=2)
        tree_tsne = tsne_operator.fit_transform(ect_data_array)
        end = time.time()
        print(f"Embedded t-SNE in {end - start:.2f} seconds.")

        # Plot everything
        f, axes = plt.subplots(2, 3, figsize=(24, 10))  # Increase width to 24 inches
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

        # Plot PCA
        ax1.scatter(
            tree_pca[:, 0], tree_pca[:, 1],
            s=2, marker=marker, color=color,  # Plot all points in gray with same marker
            edgecolor='none', alpha=0.2
        )
        ax1.set_title("PCA")

        # Plot t-SNE
        ax3.scatter(
            tree_tsne[:, 0], tree_tsne[:, 1],
            s=2, marker=marker, color=color,  # Plot all points in gray with same marker
            edgecolor='none', alpha=0.2
        )
        ax3.set_title("t-SNE")

        # Plot PHATE
        ax4.scatter(
            tree_phate[:, 0], tree_phate[:, 1],
            s=2, marker=marker, color=color,  # Plot all points in gray with same marker
            edgecolor='none', alpha=0.2
        )
        ax4.set_title("PHATE")

        # Plot PHATE with square root potential (Gamma=0)
        phate_operator.set_params(gamma=0, t=120)
        tree_phate_sqrt = phate_operator.fit_transform(ect_data_array)
        ax5.scatter(
            tree_phate_sqrt[:, 0], tree_phate_sqrt[:, 1],
            s=2, marker=marker, color=color,  # Plot all points in gray with same marker
            edgecolor='none', 
            alpha=0.2
        )
        ax5.set_title("PHATE (Gamma=0)")

        # Hide unused axis
        ax6.set_axis_off()

        # Save figure
        output_comparison_path = os.path.join(output_dir, "dimensionality_comparison.png")
        plt.tight_layout()
        plt.savefig(output_comparison_path, dpi=300)
        plt.close()
        print(f"Saved dimensionality comparison plot to {output_comparison_path}")
    else:


        # PCA
        start = time.time()
        pca_operator = sklearn.decomposition.PCA(n_components=2)
        tree_pca = pca_operator.fit_transform(ect_data_array)
        end = time.time()
        print(f"Embedded PCA in {end - start:.2f} seconds.")
        

        # t-SNE
        start = time.time()
        tsne_operator = sklearn.manifold.TSNE(n_components=2)
        tree_tsne = tsne_operator.fit_transform(ect_data_array)
        end = time.time()
        print(f"Embedded t-SNE in {end - start:.2f} seconds.")

        # Plot everything
        f, axes = plt.subplots(2, 3, figsize=(24, 10))  # Increase width to 24 inches
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

        # Plot PCA
        for taxa in unique_labels:
            species_idx = [idx for idx, label in enumerate(group_labels) if label == taxa]
            marker = marker_types[(label_to_color[taxa] // 20) % len(marker_types)]  # Cycle through markers

            ax1.scatter(
                tree_pca[species_idx, 0], tree_pca[species_idx, 1],
                s=2, marker=marker, color=plt.cm.tab20(label_to_color[taxa] % 20),
                label=taxa, edgecolor='none',
                alpha=0.2
            )
        ax1.set_title("PCA")

        # # Plot MDS
        # for taxa in unique_labels:
        #     species_idx = [idx for idx, label in enumerate(group_labels) if label == taxa]
        #     marker = marker_types[(label_to_color[taxa] // 20) % len(marker_types)]  # Cycle through markers

        #     ax2.scatter(
        #         tree_mds[species_idx, 0], tree_mds[species_idx, 1],
        #         s=20, marker=marker, color=plt.cm.tab20(label_to_color[taxa] % 20),
        #         label=taxa, edgecolor=color, alpha=1
        #     )
        # ax2.set_title("MDS")

        # Plot t-SNE
        for taxa in unique_labels:
            species_idx = [idx for idx, label in enumerate(group_labels) if label == taxa]
            marker = marker_types[(label_to_color[taxa] // 20) % len(marker_types)]  # Cycle through markers

            ax3.scatter(
                tree_tsne[species_idx, 0], tree_tsne[species_idx, 1],
                s=2, 
                marker=marker, 
                color=plt.cm.tab20(label_to_color[taxa] % 20),
                label=taxa,
                edgecolor='none', 
                alpha=0.2
            )
        ax3.set_title("t-SNE")

        # Plot PHATE
        phate_operator = phate.PHATE(n_jobs=-2)
        tree_phate = phate_operator.fit_transform(ect_data_array)
        for taxa in unique_labels:
            species_idx = [idx for idx, label in enumerate(group_labels) if label == taxa]
            marker = marker_types[(label_to_color[taxa] // 20) % len(marker_types)]  # Cycle through markers

            ax4.scatter(
                tree_phate[species_idx, 0], tree_phate[species_idx, 1],
                s=2,
                marker=marker, 
                color=plt.cm.tab20(label_to_color[taxa] % 20),
                label=taxa, 
                edgecolor='none', 
                alpha=0.2
            )
        ax4.set_title("PHATE")

        # Plot PHATE - square root potential
        phate_operator.set_params(gamma=0, t=120)
        tree_phate_sqrt = phate_operator.fit_transform(ect_data_array)
        for taxa in unique_labels:
            species_idx = [idx for idx, label in enumerate(group_labels) if label == taxa]
            marker = marker_types[(label_to_color[taxa] // 20) % len(marker_types)]  # Cycle through markers

            ax5.scatter(
                tree_phate_sqrt[species_idx, 0], tree_phate_sqrt[species_idx, 1],
                s=2, 
                marker=marker, 
                color=plt.cm.tab20(label_to_color[taxa] % 20),
                label=taxa, 
                edgecolor='none',
                alpha=0.2
            )
        ax5.set_title("PHATE (Gamma=0)")

        # Hide unused axis
        ax6.set_axis_off()

        # Save figure
        output_comparison_path = os.path.join(output_dir, "dimensionality_comparison.png")
        plt.tight_layout()
        plt.savefig(output_comparison_path, dpi=300)
        plt.close()
        print(f"Saved dimensionality comparison plot to {output_comparison_path}")

def run_phate_with_shapes(ect_data, group_labels_all, shapes, component_names, output_dir, tree_phate, bin_by_class='fullname'):
    # "family" "genus" "genus_species" "fullname"
    overall_family = ast.literal_eval(group_labels_all[0])['family']
    group_labels = []
    for g_str in group_labels_all:
        g = ast.literal_eval(g_str)  # Convert string to dict
        group_labels.append(g[bin_by_class])
    """
    Run PHATE on the dataset and save 2D, 3D plots and animations to disk, while overlaying contour shapes at the densest points.
    """
    if len(group_labels) != len(ect_data):
        raise ValueError("The number of group labels does not match the number of ECT matrices.")

    # Map each unique taxa to a numeric value for coloring
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

    # Perform PHATE on the ECT matrices
    phate_operator = phate.PHATE(n_jobs=-2)
    ect_data_flat = [matrix.flatten() for matrix in ect_data]  # Flatten ECT matrices for PHATE
    ect_data_array = np.array(ect_data_flat)  # Convert list of flattened matrices to a 2D array
    # tree_phate = phate_operator.fit_transform(ect_data_array)
    scale_val = 0.05 * (np.max(tree_phate[:, 0]) - np.min(tree_phate[:, 0]))  # Scale relative to PHATE data range
    shape_scale_factor = 0.2  # Reduce shape size further if needed
    
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(20, 10))  # Increase width to 20 inches for the legend

    # Plot the 2D PHATE embedding as scatter points
    unique_labels = np.unique(group_labels)
    label_to_color = {label: i for i, label in enumerate(unique_labels)}
    
    legend_handles = []
    legend_labels = []

    if not by_group:
        # Scatter plot all points in medium gray
        ax.scatter(
            tree_phate[:, 0], tree_phate[:, 1],
            s=2,  # Size of the dots
            color=color,  # All points in gray
            edgecolor='none',  # No edge color
            alpha=1,  # Full opacity
            marker=marker,  # Single marker type
        )

        # No need for different legends or markers, as all points are the same color and marker
        legend_handles.append(plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, markersize=15))
        legend_labels.append("All points (no grouping)")

        # Define the grid across all PHATE points
        x_min, x_max = np.min(tree_phate[:, 0]), np.max(tree_phate[:, 0])
        y_min, y_max = np.min(tree_phate[:, 1]), np.max(tree_phate[:, 1])
        x_grid = np.linspace(x_min, x_max, 7)  # 7 points along x-axis
        y_grid = np.linspace(y_min, y_max, 7)  # 7 points along y-axis

        already_have = []
        # For each grid intersection, find the closest point and plot its shape
        for x in x_grid:
            for y in y_grid:
                # Calculate distances from each PHATE point to the grid intersection
                distances = np.linalg.norm(tree_phate - np.array([x, y]), axis=1)
                closest_idx = np.argmin(distances)

                # Get the corresponding shape for the closest point
                points = shapes[closest_idx]
                points = points - np.mean(points, axis=0)  # Center the shape
                points = scale_val * points / max(np.linalg.norm(points, axis=1))  # Scale the shape
                points *= shape_scale_factor
                trans_sh = points + tree_phate[closest_idx]  # Translate to the closest point

                # Plot the shape only if it hasn't been plotted already
                if closest_idx not in already_have:
                    already_have.append(closest_idx)
                    ax.fill(trans_sh[:, 0], trans_sh[:, 1], color='black', lw=0.1, edgecolor='black', alpha=0.5)

        # Set axis labels and title
        ax.set_aspect("equal", adjustable='box')
        plt.title(f"PHATE of Leaf Contours - All Points")
        plt.xlabel("PHATE Dimension 1")
        plt.ylabel("PHATE Dimension 2")
    else:
        for taxa in unique_labels:
            species_idx = [idx for idx, label in enumerate(group_labels) if label == taxa]
            taxa_phate_points = tree_phate[species_idx]

            # Determine the marker to use based on the color cycle
            marker = marker_types[(label_to_color[taxa] // 20) % len(marker_types)]
            
            # Scatter plot of PHATE points for the current taxa
            scatter = ax.scatter(
                taxa_phate_points[:, 0], taxa_phate_points[:, 1],
                s=2,  # Size of the dots
                color=plt.cm.tab20(label_to_color[taxa] % 20),#cmap(label_to_color[taxa] % 20),
                edgecolor='none',  # No edge color
                alpha=0.2,  # Transparency for the points
                label=taxa,
                marker=marker,
            )
            
            # For the legend, create handles with larger size (e.g. 15)
            legend_handles.append(plt.Line2D([0], [0], marker=marker, color='w', markerfacecolor=cmap(label_to_color[taxa] % 20), markersize=15))
            legend_labels.append(taxa)


        # Plot the contour shapes
        min_samples = 5
        for taxa in unique_labels:
            species_idx = [idx for idx, label in enumerate(group_labels) if label == taxa]
            taxa_phate_points = tree_phate[species_idx]

            if taxa_phate_points.shape[0] < min_samples:
                print(f"Skipping {taxa} KDE due to insufficient samples")
                continue
            else:
                # Use kernel density estimation to find the point of highest density
                kde = gaussian_kde(taxa_phate_points.T)
                density_values = kde(taxa_phate_points.T)

                # Define a threshold to exclude the densest points (e.g., top 20% most dense)
                density_threshold = np.percentile(density_values, 80)
                dense_cluster_indices = np.where(density_values >= density_threshold)[0]

                # Find the shape corresponding to the densest point
                densest_point_idx = np.argmax(density_values)
                densest_point = taxa_phate_points[densest_point_idx]
                closest_idx_within_taxa = species_idx[densest_point_idx]

                # Get the contour points for the shape at the densest point
                points = shapes[closest_idx_within_taxa]
                points = points - np.mean(points, axis=0)  # Center the shape
                points = scale_val * points / max(np.linalg.norm(points, axis=1))  # Scale the shape
                points *= shape_scale_factor
                trans_dense = points + densest_point  # Translate the shape to the densest point

                # Plot the contour shape at the densest point
                ax.fill(trans_dense[:, 0], trans_dense[:, 1], color=cmap(label_to_color[taxa] % 20), lw=0.1, edgecolor='black', alpha=1)

        # Adjust axis settings
        ax.set_aspect('equal', adjustable='box')
        plt.title(f"PHATE of Leaf Contours - All Points with Outlines for Taxa = {overall_family}")
        plt.xlabel("PHATE Dimension 1")
        plt.ylabel("PHATE Dimension 2")

    # Add the legend on the right
    ax.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), title="Taxa", ncol=2)

    # Save the plot
    output_2d_path = os.path.join(output_dir, f"phate_2d_with_shapes_by_{bin_by_class}.png")
    plt.savefig(output_2d_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved PHATE 2D plot with shapes to {output_2d_path}")


def run_phate_grid_by_taxa(ect_data, group_labels_all, shapes, output_dir, phate_scores, bin_by_class='fullname'):
    # "family" "genus" "genus_species" "fullname"
    group_labels = []
    for g_str in group_labels_all:
        g = ast.literal_eval(g_str)  # Convert string to dict
        group_labels.append(g[bin_by_class])

    output_pdf_path = os.path.join(output_dir, f"phate_2d_grid_by_{bin_by_class}.pdf")

    # if len(group_labels) != len(ect_data):
        # raise ValueError("The number of group labels does not match the number of ECT matrices.")

    # Map each unique taxa to a numeric value for coloring
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

    # Perform PHATE on the ECT matrices
    # phate_operator = phate.PHATE(n_jobs=-2)
    # ect_data_flat = [matrix.flatten() for matrix in ect_data]  # Flatten ECT matrices for PHATE
    # ect_data_array = np.array(ect_data_flat)  # Convert list of flattened matrices to a 2D array
    # phate_scores = phate_operator.fit_transform(ect_data_array)
    scale_val = 0.05 * (np.max(phate_scores[:, 0]) - np.min(phate_scores[:, 0]))  # Scale relative to PHATE data range
    shape_scale_factor = 0.5  # Reduce shape size


    legend_handles = []
    legend_labels = []
    # Create a PDF object to store multiple plots
    with PdfPages(output_pdf_path) as pdf:

        if not by_group:
            # No grouping, plot all points in gray
            plt.figure(figsize=(24, 16))
            ax = plt.gca()

            # Plot all points with the same marker and color
            ax.scatter(phate_scores[:, 0], 
                       phate_scores[:, 1], 
                       s=10, 
                       color=color,
                       edgecolor='none',
                       alpha=0.8, 
                       marker=marker)

            # Define the grid across all phate_scores
            x_min, x_max = np.min(phate_scores[:, 0]), np.max(phate_scores[:, 0])
            y_min, y_max = np.min(phate_scores[:, 1]), np.max(phate_scores[:, 1])
            x_grid = np.linspace(x_min, x_max, 7)  # 7 points along x-axis
            y_grid = np.linspace(y_min, y_max, 7)  # 7 points along y-axis

            already_have = []
            # For each grid intersection, find the closest point **within all points** and plot its shape
            for x in x_grid:
                for y in y_grid:
                    distances = np.linalg.norm(phate_scores - np.array([x, y]), axis=1)
                    closest_idx = np.argmin(distances)

                    # Get the corresponding shape for the closest point
                    points = shapes[closest_idx]  # Use closest_idx to map back to the original shape
                    points = points - np.mean(points, axis=0)  # Center the shape
                    points = scale_val * points / max(np.linalg.norm(points, axis=1))  # Scale the shape
                    points *= shape_scale_factor
                    trans_sh = points + phate_scores[closest_idx]  # Translate to the closest point

                    # Plot the shape only if it hasn't been plotted already
                    if closest_idx not in already_have:
                        already_have.append(closest_idx)
                        plt.fill(trans_sh[:, 0], trans_sh[:, 1], color='black', lw=0, alpha=0.5)

            plt.gca().set_aspect("equal", adjustable='box')
            plt.title("PHATE of Leaf Contours - All Points")
            plt.xlabel("PHATE Dimension 1")
            plt.ylabel("PHATE Dimension 2")

            # Save the plot to the PDF
            pdf.savefig()
            plt.close()

        else:
            min_samples = 5

            for i, taxa in tqdm(enumerate(unique_labels), total=len(unique_labels), desc="Processing Taxa"):
                species_idx = [idx for idx, label in enumerate(group_labels) if label == taxa]
                taxa_phate_points = phate_scores[species_idx]

                

                # Create a new figure for each taxa
                plt.figure(figsize=(24, 16))
                ax = plt.gca()

                # First, plot all the points in gray for context
                ax.scatter(
                    phate_scores[:, 0], phate_scores[:, 1],
                    s=10,  # Tiny dots
                    color='lightgray',  # Gray color for all points
                    edgecolor='none',  # No edge color
                    alpha=0.2  # Transparency for all points
                )

                # Now, highlight the points of the current taxa
                marker = marker_types[(label_to_color[taxa] // 20) % len(marker_types)]  # Change marker based on color cycle
                ax.scatter(
                    taxa_phate_points[:, 0], taxa_phate_points[:, 1],
                    s=2,  # Tiny dots
                    color=plt.cm.tab20(label_to_color[taxa] % 20),#cmap(label_to_color[taxa] % 20),  # Unique color for this taxa
                    edgecolor='none',  # No edge color
                    alpha=0.2,  # More opacity for the highlighted taxa
                    label=taxa,
                    marker=marker
                )

                if taxa_phate_points.shape[0] < min_samples:
                    print(f"Skipping {taxa} KDE due to insufficient samples")
                else:
                    # Use kernel density estimation to find the point of highest density
                    kde = gaussian_kde(taxa_phate_points.T)
                    density_values = kde(taxa_phate_points.T)

                    # Define a threshold to exclude the densest points (e.g., top 20% most dense)
                    density_threshold = np.percentile(density_values, 80)
                    dense_cluster_indices = np.where(density_values >= density_threshold)[0]

                    # Exclude the points that belong to the densest cluster for k-means clustering
                    remaining_indices = np.where(density_values < density_threshold)[0]
                    remaining_points = taxa_phate_points[remaining_indices]

                    # Find the shape corresponding to the densest point
                    densest_point_idx = np.argmax(density_values)
                    densest_point = taxa_phate_points[densest_point_idx]
                    closest_idx_within_taxa = species_idx[densest_point_idx]

                    # Get the contour points for the shape at the densest point
                    points = shapes[closest_idx_within_taxa]
                    points = points - np.mean(points, axis=0)  # Center the shape
                    points = scale_val * points / max(np.linalg.norm(points, axis=1))  # Scale the shape
                    points *= shape_scale_factor
                    trans_dense = points + densest_point  # Translate the shape to the densest point

                    # Store the shape in CustomLegendShape
                    legend_shape = CustomLegendShape(trans_dense, 
                                                     facecolor=plt.cm.tab20(label_to_color[taxa] % 20),#cmap(label_to_color[taxa] % 20), 
                                                     edgecolor='none')
                    legend_handles.append(legend_shape)
                    legend_labels.append(taxa)

                    # Plot the contour shape at the densest point
                    plt.fill(trans_dense[:, 0], trans_dense[:, 1], color=cmap(label_to_color[taxa] % 20), lw=1, edgecolor='black', alpha=1)

                    # Find the grid intersection points for the current taxa points
                    x_min, x_max = np.min(taxa_phate_points[:, 0]), np.max(taxa_phate_points[:, 0])
                    y_min, y_max = np.min(taxa_phate_points[:, 1]), np.max(taxa_phate_points[:, 1])
                    x_grid = np.linspace(x_min, x_max, 7)  # Divide into 6 equal intervals
                    y_grid = np.linspace(y_min, y_max, 7)

                    already_have = []
                    # For each grid intersection, find the closest point **within the current taxa points** and plot its shape
                    for x in x_grid:
                        for y in y_grid:
                            distances = np.linalg.norm(taxa_phate_points - np.array([x, y]), axis=1)
                            closest_idx_within_taxa = np.argmin(distances)

                            points = shapes[species_idx[closest_idx_within_taxa]]  # Use species_idx to map back to the original shape
                            points = points - np.mean(points, axis=0)  # Center the shape
                            points = scale_val * points / max(np.linalg.norm(points, axis=1))  # Scale the shape
                            points *= shape_scale_factor
                            trans_sh = points + taxa_phate_points[closest_idx_within_taxa]  # Translate to the closest point

                            if closest_idx_within_taxa not in already_have:
                                already_have.append(closest_idx_within_taxa)
                                plt.fill(trans_sh[:, 0], trans_sh[:, 1], color=cmap(label_to_color[taxa] % 20), lw=0, alpha=0.5)

                    # Plot contour shape at densest point again
                    plt.fill(trans_dense[:, 0], trans_dense[:, 1], color=cmap(label_to_color[taxa] % 20), lw=1, edgecolor='black', alpha=1)

                # Set axis labels, title, and aspect ratio
                plt.gca().set_aspect("equal", adjustable='box')
                plt.title(f"PHATE of Leaf Contours - {taxa} Highlighted")
                plt.xlabel("PHATE Dimension 1")
                plt.ylabel("PHATE Dimension 2")

                # Add the legend with custom shapes (using the stored shape points)
                ax.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, title="Taxa", 
                        handler_map={CustomLegendShape: HandlerShape()})

                # Save the current figure to the PDF
                pdf.savefig()  # Save the current page
                plt.close()  # Close the figure to free up memory

    print(f"All PHATE plots saved to {output_pdf_path}")






