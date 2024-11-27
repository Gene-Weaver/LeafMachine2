import os, sys, inspect
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns
from matplotlib.patches import Ellipse
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.affinity import scale, rotate
from shapely.geometry.polygon import Polygon
from scipy.spatial import ConvexHull
import matplotlib.patches as patches
import h5py
import dask.dataframe as dd
from multiprocessing import Pool

currentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parentdir = os.path.dirname(currentdir)
try:
    from leafmachine2.ect_methods.leaf_ect import LeafECT
except:
    from leaf_ect import LeafECT


def preprocessing(file_path, outline_path, show_CF_plot=True, show_first_raw_contour=True, show_df_head=True, is_Thais=False, path_figure=None):
    def handle_duplicates(group):
        # Filter out rows where conversion_mean is NaN or 0
        valid_conversion_mean = group[(group['conversion_mean'].notna()) & (group['conversion_mean'] != 0)]
        
        # Case 1: If there's only one valid row with a 'conversion_mean', keep that row
        if len(valid_conversion_mean) == 1:
            return valid_conversion_mean
        
        # Case 2: If more than one valid row has 'conversion_mean', keep the first row
        if len(valid_conversion_mean) > 1:
            # avg_conversion_mean = valid_conversion_mean['conversion_mean'].mean()
            # first_row = group.iloc[0].copy()  # Get a copy of the first row to modify
            # first_row['conversion_mean'] = avg_conversion_mean  # Update with the averaged value
            # return pd.DataFrame([first_row])  # Return the updated first row as a DataFrame
            first_row = valid_conversion_mean.iloc[0].copy()  # Get a copy of the first valid row
            return pd.DataFrame([first_row])  # Return the first valid row as a DataFrame
        
        # Case 3: If no valid rows have 'conversion_mean', just keep the first row of the group
        return group.iloc[[0]]
    
    def show_predicted_vs_measured_CF_plot(filtered_df, path_figure):
        # Step 4: Set up the plot
        plt.figure(figsize=(10, 6))

        # Step 5: Create the scatter plot
        scatter = sns.scatterplot(
            data=filtered_df,
            x='megapixels',
            y='cf_error',
            hue='herb',  # Color by herb
            palette='Set1',  # Use a color palette (optional, but Set1 works well for categories)
            s=100,  # Size of the points
            alpha=0.8  # Transparency of points
        )

        # Step 6: Add labels and a title
        plt.title("Scatterplot of Image Megapixels (Height * Width) vs Error", fontsize=14)
        plt.xlabel("Image Area (Megapixels)", fontsize=12)
        plt.ylabel("Error", fontsize=12)

        # Step 7: Show legend and grid
        plt.legend(title='Herb', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)

        # Step 8: Save the plot as a 600 DPI PNG file
        plt.tight_layout()
        plt.savefig(path_figure, dpi=600, format='png')

        # Optional: Clear the plot after saving to prevent overlap in future plots
        plt.clf()
    

    # Construct the clean file path
    clean_file_path = file_path.replace(".csv", "_CLEAN.csv")

    # Check if the clean file exists
    if os.path.exists(clean_file_path):
        print(f"     --> Cleaned file found at {clean_file_path}, loading...")
        # df_clean = pd.read_csv(clean_file_path)
        df = ingest_DWC_large_files(clean_file_path)
        filtered_df = df
        df_clean = df

    else:
        print(f"     Cleaned file NOT found. Building _CLEAN file from {file_path}...")

        # Step 1: Load the CSV using Dask to handle large files
        df = ingest_DWC_large_files(file_path)
        
        # Convert Dask DataFrame to Pandas DataFrame to continue cleaning
        df_clean = df.compute()  # This materializes the Dask DataFrame into a Pandas DataFrame
        
        df_clean = df_clean.dropna(subset=['image_height'])
        df_clean = df_clean.dropna(subset=['image_width'])

        df_clean['image_height'] = df_clean['image_height'].astype('int64')
        df_clean['image_width'] = df_clean['image_width'].astype('int64')

        # Step 2: Create a copy of the original data for safe cleaning
        df_clean = df_clean.copy()

        # Step 3: Remove rows where 'predicted_conversion_factor_cm' is missing
        df_clean = df_clean.dropna(subset=['predicted_conversion_factor_cm'])

        # Step 4: Remove rows where 'image_height' is less than 3000
        df_clean = df_clean[df_clean['image_height'] >= 3000]

        # Step 5: Handle duplicates in 'component_name'
        # Apply the deduplication logic to each group of duplicates in 'component_name'
        df_clean = df_clean.groupby('component_name', group_keys=False).apply(handle_duplicates).reset_index(drop=True)

        # Add the helper columns
        if is_Thais:
            df_clean = add_helper_columns_Thais(df_clean)
        else:
            df_clean = add_helper_columns(df_clean)

        # Add cf_error column
        df_clean = add_cf_error_column(df_clean)

        # Display the rows that have non-empty cf_error values
        filtered_df = df_clean[df_clean['cf_error'] != ""]

        # Show the filtered rows with cf_error values
        # filtered_df.head()

        # Step 1: Create a new column for image_area (image_height * image_width)
        df_clean['megapixels'] = df_clean['image_height'] * df_clean['image_width'] / 1000000

        # Step 2: Convert cf_error to numeric, forcing errors to NaN (this will handle empty strings)
        df_clean['cf_error'] = pd.to_numeric(df_clean['cf_error'], errors='coerce')

        # Step 3: Filter out rows where cf_error is NaN (invalid for plotting)
        filtered_df = df_clean.dropna(subset=['cf_error'])

        # Ensure cf_error is numeric, as before
        df_clean['cf_error'] = pd.to_numeric(df_clean['cf_error'], errors='coerce')

        # Filter rows where cf_error is greater than 2
        filtered_cf_error_gt_2 = df_clean[df_clean['cf_error'] > 0.2]

        # Step 6: Save the cleaned dataframe as _CLEAN.csv
        df_clean.to_csv(clean_file_path, index=False)
        print(f"Cleaned data saved to {clean_file_path}")

    # Continue with any additional operations (e.g., plotting)
    filtered_df = df_clean.dropna(subset=['cf_error'])

    if show_CF_plot:
        if path_figure is not None:
            show_predicted_vs_measured_CF_plot(filtered_df, path_figure)

    if show_first_raw_contour:
        first_component_name = df_clean['component_name'].iloc[0]
        visualize_first_contour_raw(first_component_name, outline_path)

    if show_df_head:
        print("\ndf_clean\n")
        print(df_clean.head())
        print("\nfiltered_df\n")
        print(filtered_df.head())

    return df_clean



def store_ect_data(input_dirs, num_workers=16, chunk_size=50, num_dirs=64, num_thresh=64):
    """
    Parallel processing of ECT matrices and simplified contours, storing them in HDF5 files.
    
    Parameters:
        input_dirs (list): A list of directories containing the `Keypoints/Simple_Labels` files.
        num_workers (int): Number of worker processes.
        chunk_size (int): Number of files to process in each chunk.
    """
    # Iterate over each directory in input_dirs
    for dir_path in input_dirs:
        chunks = []
        outline_path = os.path.join(dir_path, 'Keypoints', 'Simple_Labels')
        hdf5_dir_path = os.path.join(dir_path, 'Data', 'Measurements', 'ECT')

        # Ensure the HDF5 directory exists
        os.makedirs(hdf5_dir_path, exist_ok=True)

        # Check if the outline path exists
        if not os.path.exists(outline_path):
            print(f"Outline path not found: {outline_path}. Skipping directory.")
            continue

        # List all .txt files in the outline_path directory
        component_names = [f for f in os.listdir(outline_path) if f.endswith('.txt')]

        # Chunk the files for parallel processing
        for i in range(0, len(component_names), chunk_size):
            chunk = (outline_path, hdf5_dir_path, component_names[i:i + chunk_size], num_dirs, num_thresh)
            chunks.append(chunk)

        # Create a Pool of workers to handle chunks
        results = []
        with Pool(processes=num_workers) as pool:
            # Monitor progress using tqdm based on the number of chunks
            for result in tqdm(pool.imap_unordered(worker_store_ect_data, chunks), total=len(chunks), desc="Processing Contours"):
                if result:
                    results.append(result)  # Collect the results (ect_data, group_labels, shapes)

        # Now save the collected results sequentially
        for ect_data, group_labels, shapes, names in results:
            for i, base_name in enumerate(names):
                # Save each result with its corresponding base name
                output_hdf5_path = os.path.join(hdf5_dir_path, f'{base_name}.h5')
                save_to_hdf5([ect_data[i]], [group_labels[i]], [shapes[i]], [base_name], output_hdf5_path)

def worker_store_ect_data(task):
    """
    Worker function that processes a chunk of component names and returns the computed ECT matrices.
    """
    outline_path, hdf5_dir_path, component_names, num_dirs, num_thresh = task

    # Initialize LeafECT object
    leaf_ect = LeafECT(df=None, outline_path=outline_path, num_dirs=num_dirs, num_thresh=num_thresh, max_pts=500)

    # Prepare lists to hold the computed data
    ect_data = []
    group_labels = []
    shapes = []
    names = []

    for file_name in component_names:
        try:
            base_name = os.path.splitext(os.path.basename(file_name))[0]
            parts = base_name.split('_')

            labels = {
                "family": parts[2],
                "genus": parts[3],
                "genus_species": f"{parts[3]}_{parts[4]}",
                "fullname": f"{parts[2]}_{parts[3]}_{parts[4]}",
            }

            # Compute ECT matrix and simplified contour
            ECT_matrix, simplified_contour = leaf_ect.compute_ect_for_contour(file_name, is_DP='DP')

            if ECT_matrix is None or simplified_contour is None:
                raise ValueError(f"Failed to compute ECT for {file_name}")

            # Append the computed data
            ect_data.append(ECT_matrix)
            group_labels.append(labels)
            shapes.append(simplified_contour)
            names.append(base_name)

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            continue

    return ect_data, group_labels, shapes, names




def add_helper_columns(df):
    # Step 1: Split 'component_name' by "__" first
    split_component = df['component_name'].str.split("__", expand=True)
    
    # The first part before "__" will be the variable "to_split"
    df['to_split'] = split_component[0]
    
    # 'component_type' and 'component_id' come from the second part of the split
    df['component_type'] = split_component[1]
    df['component_id'] = split_component[2]
    
    # Step 2: Split "to_split" further by "_"
    split_to_split = df['to_split'].str.split("_", expand=True)

    # Assigning new columns based on the parsed parts
    df['herb'] = split_to_split[0]
    df['gbif_id'] = split_to_split[1]
    df['fullname'] = split_to_split[2] + "_" + split_to_split[3] + "_" + split_to_split[4]
    df['genus_species'] = split_to_split[3] + "_" + split_to_split[4]
    df['family'] = split_to_split[2]
    df['genus'] = split_to_split[3]
    df['specific_epithet'] = split_to_split[4]

    # Return the DataFrame with the new columns added
    return df

def add_helper_columns_Thais(df):
    # Step 1: Split 'component_name' by "__" first
    split_component = df['component_name'].str.split("__", expand=True)
    
    # The first part before "__" will be the variable "to_split"
    df['to_split'] = split_component[0]
    
    # 'component_type' and 'component_id' come from the second part of the split
    df['component_type'] = split_component[1]
    df['component_id'] = split_component[2]
    
    # Step 2: Split "to_split" further by "_"
    split_to_split = df['to_split'].str.split("_", expand=True)

    # Assigning new columns based on the parsed parts
    df['herb'] = 'unknown'
    df['gbif_id'] = split_to_split[3]
    df['fullname'] = split_to_split[0] + "_" + split_to_split[1] 
    df['genus_species'] = split_to_split[0] + "_" + split_to_split[1] 
    df['family'] = 'unknown'
    df['genus'] = split_to_split[0]
    df['specific_epithet'] = split_to_split[1]

    # Return the DataFrame with the new columns added
    return df


def add_cf_error_column(df):
    # Define a small tolerance to account for floating-point precision issues
    tolerance = 1e-9

    # Step to compute cf_error with protection against division by zero and comparison with tolerance for equality
    df['cf_error'] = df.apply(lambda row: "" if row['predicted_conversion_factor_cm'] == 0 
                              else "" if abs(1 - (row['conversion_mean'] / row['predicted_conversion_factor_cm'])) < tolerance
                              else 1 - (row['conversion_mean'] / row['predicted_conversion_factor_cm']), axis=1)
    
    # Return the updated DataFrame
    return df


# Function to read and visualize the first contour without transformations
def visualize_first_contour_raw(component_name, outline_path):
    # Build the path to the .txt file for the given component
    txt_file_path = os.path.join(outline_path, f'{component_name}.txt')
    
    if not os.path.exists(txt_file_path):
        print(f"File not found: {txt_file_path}")
        return
    
    # Read the data from the file
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    # Starting from line 11, parse the points (x, y)
    points = []
    for line in lines[11:]:
        x_str, y_str = line.strip().split(',')
        x = float(x_str)
        y = float(y_str)
        points.append((x, y))

    # Convert to a NumPy array for plotting
    points_np = np.array(points, dtype=np.float32)

    # Plot the raw contour points
    plt.figure(figsize=(6, 6))
    plt.plot(points_np[:, 0], points_np[:, 1], 'r-', linewidth=1)
    plt.title(f'Raw Contour for {component_name}')
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure the aspect ratio is equal
    plt.show()

def parse_availability(df):
    # Create a dictionary to hold the unique values
    unique_values = {
        'family': set(),
        'genus': set(),
        'genus_species': set(),
        'fullname': set()
    }
    
    # Iterate over the rows of the DataFrame to collect unique values
    for _, row in df.iterrows():
        unique_values['family'].add(row['family'])
        unique_values['genus'].add(row['genus'])
        unique_values['genus_species'].add(row['genus_species'])
        unique_values['fullname'].add(row['fullname'])
    
    # Convert sets back to lists for easier consumption
    for key in unique_values:
        unique_values[key] = list(unique_values[key])
    
    return unique_values







# def save_to_hdf5(ect_data, group_labels, shapes, component_names, output_path_hdf5, append_mode=False):
#     with h5py.File(output_path_hdf5, 'w') as f:
#         # Store ECT matrices as a dataset
#         ect_group = f.create_group('ECT_matrices')
#         for i, ect_matrix in enumerate(ect_data):
#             ect_group.create_dataset(f'matrix_{i}', data=ect_matrix)
        
#         # Store group labels as a dataset
#         f.create_dataset('group_labels', data=np.array(group_labels, dtype='S'))  # Store as string array
        
#         # Store shapes as a dataset
#         shapes_group = f.create_group('shapes')
#         for i, shape in enumerate(shapes):
#             shapes_group.create_dataset(f'shape_{i}', data=shape)

#         # Store component names as a dataset
#         f.create_dataset('component_names', data=np.array(component_names, dtype='S'))  # Store as string array

#     print(f"Data saved to {output_path_hdf5}")
def save_to_hdf5(ect_data, group_labels, shapes, component_names, output_path_hdf5, append_mode=False):
    # Use 'a' mode for append or 'w' mode for overwrite
    file_mode = 'a' if append_mode else 'w'
    with h5py.File(output_path_hdf5, file_mode) as f:
        # Handle 'ECT_matrices' group
        ect_group = f.require_group('ECT_matrices')
        # Determine the starting index for new datasets in 'ECT_matrices'
        existing_matrices = len(ect_group)  # Count existing datasets
        for i, ect_matrix in enumerate(ect_data):
            dataset_name = f'matrix_{existing_matrices + i}'  # New name, sequential
            ect_group.create_dataset(dataset_name, data=ect_matrix)

        # Append or create 'group_labels'
        if 'group_labels' in f and append_mode:
            # Combine existing labels with new labels
            existing_labels = f['group_labels'][:].tolist()
            combined_labels = existing_labels + [label.encode('utf-8') for label in group_labels]
            del f['group_labels']  # Delete old dataset
            f.create_dataset('group_labels', data=np.array(combined_labels, dtype='S'))
        else:
            # Create new dataset for 'group_labels'
            f.create_dataset('group_labels', data=np.array(group_labels, dtype='S'))  # Store as string array

        # Handle 'shapes' group
        shapes_group = f.require_group('shapes')
        # Determine the starting index for new datasets in 'shapes'
        existing_shapes = len(shapes_group)  # Count existing datasets
        for i, shape in enumerate(shapes):
            dataset_name = f'shape_{existing_shapes + i}'  # New name, sequential
            shapes_group.create_dataset(dataset_name, data=shape)

        # Append or create 'component_names'
        if 'component_names' in f and append_mode:
            # Combine existing names with new names
            existing_names = f['component_names'][:].tolist()
            combined_names = existing_names + [name.encode('utf-8') for name in component_names]
            del f['component_names']  # Delete old dataset
            f.create_dataset('component_names', data=np.array(combined_names, dtype='S'))
        else:
            # Create new dataset for 'component_names'
            f.create_dataset('component_names', data=np.array(component_names, dtype='S'))  # Store as string array

    print(f"Data saved to {output_path_hdf5}")

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







def torch_distance_matrix(X):
    # """
    # Compute pairwise Euclidean distances on the GPU using PyTorch.
    # X is assumed to be a 2D tensor with shape (n_samples, n_features).
    # """
    # Step 2.1: Compute the Gram matrix (X @ X.T)
    G = torch.mm(X, X.T)
    
    # Step 2.2: Compute squared Euclidean distances
    H = torch.diag(G).unsqueeze(0)  # Get diagonal of Gram matrix as a row vector
    D = torch.sqrt(H.T + H - 2 * G)  # Apply the Euclidean distance formula

    return D




def plot_bounding_ellipse(mds_scores, genus_idx, ax, n_std=2.0, **kwargs):
    # """
    # Plots a bounding ellipse that marks the spread of points in mds_scores for the genus.
    # The ellipse is based on the covariance and mean of the MDS points.

    # Args:
    #     mds_scores (np.ndarray): MDS scores for all samples.
    #     genus_idx (list of int): Indices of the samples belonging to the genus.
    #     ax (matplotlib.axes.Axes): The axes on which to plot the bounding ellipse.
    #     n_std (float): The number of standard deviations to scale the ellipse.
    #     **kwargs: Additional keyword arguments for the plot (e.g., edgecolor, facecolor, lw).

    # Returns:
    #     tuple: (mean, (width, height), angle) of the bounding ellipse.
    # """
    # Extract the points for the genus
    points = mds_scores[genus_idx, :]

    # Compute the mean and covariance of the points
    cov = np.cov(points, rowvar=False)
    mean = np.mean(points, axis=0)

    # Eigenvalue decomposition of the covariance matrix
    eigenvals, eigenvecs = np.linalg.eigh(cov)

    # Compute the angle of the ellipse based on the eigenvectors
    angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))

    # The width and height of the ellipse correspond to the eigenvalues
    # scaled by the desired number of standard deviations (n_std)
    width, height = 2 * n_std * np.sqrt(eigenvals)

    # Create and add the ellipse to the plot using kwargs for customization
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

    # Return ellipse parameters
    return (mean, (width, height), angle)

# def plot_convex_hull_mean(mds_scores, genus_idx, ax, **kwargs):
#     """
#     Plots the convex hull of the MDS scores for the points corresponding to a specific genus
#     and returns the mean of the points.

#     Args:
#         mds_scores (np.ndarray): MDS scores for all samples.
#         genus_idx (list of int): Indices of the samples belonging to the genus.
#         ax (matplotlib.axes.Axes): The axes on which to plot the convex hull.
#         **kwargs: Additional keyword arguments for the plot (e.g., color, alpha).

#     Returns:
#         np.ndarray: The mean (center) of the points in the convex hull.
#     """
#     # Extract the points for the genus
#     points = mds_scores[genus_idx, :]

#     # Compute the convex hull for the points
#     hull = ConvexHull(points)

#     # Plot the convex hull by connecting the vertices
#     for simplex in hull.simplices:
#         ax.plot(points[simplex, 0], points[simplex, 1], **kwargs)

#     # Compute the mean of the points
#     mean = np.mean(points, axis=0)

#     return mean

def convex_hull_to_polygon(convex_hull):
    # """
    # Convert a ConvexHull object into a Shapely Polygon.
    
    # Args:
    #     convex_hull (ConvexHull): The convex hull object.
    
    # Returns:
    #     Polygon: A Shapely polygon representing the convex hull.
    # """
    hull_points = convex_hull.points[convex_hull.vertices]
    return Polygon(hull_points)

def calculate_convex_hull_overlap(hull1, hull2):
    # """
    # Calculate the exact overlap area of two convex hulls using Shapely.
    
    # Args:
    #     hull1 (ConvexHull): The first convex hull.
    #     hull2 (ConvexHull): The second convex hull.
    
    # Returns:
    #     float: The normalized overlap area with respect to the smaller convex hull.
    # """
    # Convert the convex hulls to Shapely polygons
    polygon1 = convex_hull_to_polygon(hull1)
    polygon2 = convex_hull_to_polygon(hull2)

    # Check for overlap using Shapely's intersection method
    intersection_area = polygon1.intersection(polygon2).area

    # Calculate areas of both convex hulls
    area_hull1 = polygon1.area
    area_hull2 = polygon2.area

    # Return the normalized overlap area with respect to the smaller hull
    smaller_area = min(area_hull1, area_hull2)
    if smaller_area == 0:
        return 0.0  # Avoid division by zero
    return intersection_area / smaller_area









def plot_confidence_ellipse_tsne(cov, pos, n_std=1, ax=None, **kwargs):
        # """
        # Plots an ellipse representing the covariance matrix `cov` centered at `pos`.
        # `n_std` is the number of standard deviations for the ellipse.
        # """
        if ax is None:
            ax = plt.gca()

        # Eigenvalues and eigenvectors of the covariance matrix
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        # The angle to rotate the ellipse by
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

        # Width and height of the ellipse (scaled by number of std deviations)
        width, height = 2 * n_std * np.sqrt(eigvals)

        # Create the ellipse patch
        ellipse = patches.Ellipse(xy=pos, width=width, height=height, angle=angle, **kwargs)

        # Add the ellipse to the plot
        ax.add_patch(ellipse)

def plot_confidence_ellipse_mean(mds_scores, genus_idx, ax, n_std=2.0, do_plot=True, **kwargs):
    # Plots an ellipse that marks the spread of points in mds_scores for the genus.
    # The ellipse is drawn based on the covariance and mean of the MDS points.
    # Returns the center of the ellipse.

    # Extract the points for the genus
    points = mds_scores[genus_idx, :]
    
    # Compute the mean and covariance of the points
    cov = np.cov(points, rowvar=False)
    mean = np.mean(points, axis=0)  # This is the center of the ellipse
    
    # Eigenvalues and eigenvectors for the covariance matrix
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Get the angle of the ellipse based on the eigenvectors
    angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
    
    # Width and height of the ellipse: 2 * sqrt(eigenvalue) * n_std
    width, height = 2 * n_std * np.sqrt(eigenvals)
    
    # Create and add the ellipse to the plot
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    if do_plot:
        ax.add_patch(ellipse)
    
    return mean  # Ensure the center of the ellipse is returned

# Function to calculate the ellipse parameters
def plot_confidence_ellipse(mds_scores, genus_idx, ax, n_std=2.0, **kwargs):
    # Plots an ellipse that marks the spread of points in mds_scores for the genus.
    # The ellipse is drawn based on the covariance and mean of the MDS points.

    # Extract the points for the genus
    points = mds_scores[genus_idx, :]
    
    # Compute the mean and covariance of the points
    cov = np.cov(points, rowvar=False)
    mean = np.mean(points, axis=0)
    
    # Eigenvalues and eigenvectors for the covariance matrix
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    
    # Get the angle of the ellipse based on the eigenvectors
    angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
    
    # Width and height of the ellipse: 2 * sqrt(eigenvalue) * n_std
    width, height = 2 * n_std * np.sqrt(eigenvals)
    
    # Create and add the ellipse to the plot
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    
    return (mean, (width, height), angle)  # Return ellipse parameters


def ellipse_to_polygon(ellipse, num_points=100):
    # """Convert an ellipse to a polygon approximation using shapely."""
    (center_x, center_y), (width, height), angle = ellipse
    # Create a circular polygon and scale/rotate to match the ellipse
    circle = Point(center_x, center_y).buffer(1, resolution=num_points)
    ellipse_polygon = scale(circle, width / 2, height / 2)
    ellipse_polygon = rotate(ellipse_polygon, angle)
    return ellipse_polygon

def calculate_ellipse_overlap(ellipse1, ellipse2):
    # """Calculate the exact or approximate overlap area of two ellipses."""
    # Convert both ellipses to polygonal approximations
    ellipse1_polygon = ellipse_to_polygon(ellipse1)
    ellipse2_polygon = ellipse_to_polygon(ellipse2)

    # Check for overlap using shapely's intersection method
    intersection_area = ellipse1_polygon.intersection(ellipse2_polygon).area

    # Calculate areas of both ellipses
    area_ellipse1 = ellipse_area(ellipse1)
    area_ellipse2 = ellipse_area(ellipse2)

    # Return the normalized overlap area with respect to the smaller ellipse
    smaller_area = min(area_ellipse1, area_ellipse2)
    if smaller_area == 0:
        return 0.0  # Avoid division by zero
    return intersection_area / smaller_area

# Reuse your ellipse_area function to calculate ellipse area
def ellipse_area(ellipse):
    """Calculate the area of an ellipse given its parameters."""
    _, (width, height), _ = ellipse
    return np.pi * (width / 2) * (height / 2)

def ingest_DWC_large_files(file_path, block_size=512):
    # Method to ingest large DWC (Darwin Core) CSV or TXT files using Dask to handle larger-than-memory files.
    # This function supports .txt or .csv files with varying delimiters.
    file_extension = file_path.split('.')[-1]  # Extract file extension

    # Define the dtype for each column
    column_dtypes = {
        'filename': 'object',  # TEXT
        'image_height': 'float64',  # INTEGER
        'image_width': 'float64',  # INTEGER
        'component_name': 'object',  # TEXT
        'n_pts_in_polygon': 'float64',  # INTEGER
        'conversion_mean': 'float64',  # FLOAT
        'predicted_conversion_factor_cm': 'float64',  # FLOAT
        'area': 'float64',  # FLOAT
        'perimeter': 'float64',  # FLOAT
        'convex_hull': 'float64',  # FLOAT
        'rotate_angle': 'float64',  # FLOAT
        'bbox_min_long_side': 'float64',  # FLOAT
        'bbox_min_short_side': 'float64',  # FLOAT
        'convexity': 'float64',  # FLOAT
        'concavity': 'float64',  # FLOAT
        'circularity': 'float64',  # FLOAT
        'aspect_ratio': 'float64',  # FLOAT
        'angle': 'float64',  # FLOAT
        'distance_lamina': 'float64',  # FLOAT
        'distance_width': 'float64',  # FLOAT
        'distance_petiole': 'float64',  # FLOAT
        'distance_midvein_span': 'float64',  # FLOAT
        'distance_petiole_span': 'float64',  # FLOAT
        'trace_midvein_distance': 'float64',  # FLOAT
        'trace_petiole_distance': 'float64',  # FLOAT
        'apex_angle': 'float64',  # FLOAT
        'base_angle': 'float64',  # FLOAT
        'base_is_reflex': 'bool',  # BOOLEAN
        'apex_is_reflex': 'bool',  # BOOLEAN

        'filename': 'object',  # TEXT
        'bbox': 'object',  # TEXT
        'bbox_min': 'object',  # TEXT
        'efd_coeffs_features': 'object',  # TEXT 
        'efd_a0': 'object',  # TEXT 
        'efd_c0': 'object',  # TEXT 
        'efd_scale': 'object',  # TEXT
        'efd_angle': 'object',  # TEXT
        'efd_phase': 'object',  # TEXT 
        'efd_area': 'object',  # TEXT 
        'efd_perimeter': 'object',  # TEXT 
        'centroid': 'object',  # TEXT 
        'convex_hull.1': 'object',  # TEXT
        'polygon_closed': 'object',  # TEXT
        'polygon_closed_rotated': 'object',  # TEXT 
        'keypoints': 'object',  # TEXT 
        'tip': 'object',  # TEXT 
        'base': 'object',  # TEXT

        'to_split': 'object',  # TEXT
        'component_type': 'object',  # TEXT
        'component_id': 'object',  # TEXT
        'herb': 'object',  # TEXT
        'gbif_id': 'object',  # TEXT
        'fullname': 'object',  # TEXT
        'genus_species': 'object',  # TEXT
        'family': 'object',  # TEXT
        'genus': 'object',  # TEXT
        'specific_epithet': 'object',  # TEXT
        'cf_error': 'float64',  # TEXT
        'megapixels': 'float64',  # TEXT

    }

    try:
        if file_extension == 'txt':
            # Reading a .txt file with tab-delimited data
            df = dd.read_csv(file_path, sep="\t", header=0, dtype=column_dtypes, assume_missing=True, blocksize=f"{block_size}MB", on_bad_lines="skip")
        elif file_extension == 'csv':
            try:
                # Try reading as comma-separated first
                df = dd.read_csv(file_path, sep=",", header=0, dtype=column_dtypes, assume_missing=True, blocksize=f"{block_size}MB", on_bad_lines="skip")
            except Exception:
                try:
                    # Try reading as tab-separated in case it's not comma-separated
                    df = dd.read_csv(file_path, sep="\t", header=0, dtype=column_dtypes, assume_missing=True, blocksize=f"{block_size}MB", on_bad_lines="skip")
                except Exception:
                    try:
                        # Try reading as pipe-separated
                        df = dd.read_csv(file_path, sep="|", header=0, dtype=column_dtypes, assume_missing=True, blocksize=f"{block_size}MB", on_bad_lines="skip")
                    except Exception:
                        # Finally, try reading as semicolon-separated
                        df = dd.read_csv(file_path, sep=";", header=0, dtype=column_dtypes, assume_missing=True, blocksize=f"{block_size}MB", on_bad_lines="skip")
        else:
            print(f"DWC file {file_path} is not '.txt' or '.csv' and was not opened")
            return None
    except Exception as e:
        print(f"Error while reading file: {e}")
        return None

    return df