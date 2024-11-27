from sklearn.metrics import (
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_score,
)
import numpy as np
import os, ast, itertools
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from scipy.stats import f_oneway
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance, iqr
from scipy.ndimage import label, find_objects
from scipy.spatial import ConvexHull, distance
from scipy.optimize import curve_fit, OptimizeWarning
from collections import defaultdict
import seaborn as sns
from tqdm import tqdm
# from tqdm.notebook import tqdm  # Use notebook-compatible tqdm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
import warnings

def plot_2d_distribution(total_data, bins, class_data, bin_width, class_label, save_dir, test_type, taxa):
    
    # Plot total data distribution
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist2d(total_data[:, 0], total_data[:, 1], bins=bins, cmap='Blues')
    plt.colorbar(label='Density')
    plt.title("Total Data Distribution")

    # Plot class-specific distribution
    plt.subplot(1, 2, 2)
    plt.hist2d(class_data[:, 0], class_data[:, 1], bins=bins, cmap='Reds')
    plt.colorbar(label='Density')
    plt.title(f"{class_label} Distribution")

    plt.tight_layout()
    output_dir = os.path.join(save_dir, 'Histogram2D')
    os.makedirs(output_dir, exist_ok=True)

    # Define the full path for the output image file
    output_image_path = os.path.join(output_dir, f"Histogram2D_{test_type}_{taxa}_{class_label}.png")
    plt.savefig(output_image_path, dpi=600, bbox_inches='tight')
    plt.close()

def plot_3d_distribution(total_hist, edges, bins, class_data, bin_width, class_label, save_dir, test_type, taxa):

    class_hist, _ = np.histogramdd(class_data, bins=bins, density=True)

    # Plot 3D histogram for total data
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    xpos, ypos, zpos = np.meshgrid(edges[0][:-1], edges[1][:-1], edges[2][:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = zpos.ravel()
    dx = dy = dz = bin_width
    dz_total = total_hist.ravel()

    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz_total, shade=True)
    ax1.set_title("Total Data Distribution")

    # Plot 3D histogram for class data
    ax2 = fig.add_subplot(122, projection='3d')
    dz_class = class_hist.ravel()
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz_class, shade=True)
    ax2.set_title(f"{class_label} Distribution")

    plt.tight_layout()
    output_dir = os.path.join(save_dir, 'Histogram3D')
    os.makedirs(output_dir, exist_ok=True)
    # Define the full path for the output image file
    output_image_path = os.path.join(output_dir, f"Histogram3D_{test_type}_{taxa}_{class_label}.png")
    plt.savefig(output_image_path, dpi=600, bbox_inches='tight')
    plt.close()


def calculate_bin_width(data, method="sqrt"):
    """Calculate bin width based on the chosen method."""
    n = len(data)
    if method == "sqrt":
        # Square Root Rule
        bin_width = (np.max(data) - np.min(data)) / np.sqrt(n)
    elif method == "sturges":
        # Sturges' Formula
        bin_width = (np.max(data) - np.min(data)) / (1 + np.log2(n))
    elif method == "freedman-diaconis":
        # Freedman-Diaconis Rule
        bin_width = 2 * iqr(data) * n ** (-1/3)
    else:
        raise ValueError("Invalid binning method. Choose from 'sqrt', 'sturges', or 'freedman-diaconis'.")
    return bin_width

def get_bins(data, bin_width):
    """Generate bin edges for each dimension of data using the specified bin width."""
    bins = [np.arange(np.min(data[:, dim]), np.max(data[:, dim]) + bin_width, bin_width) for dim in range(data.shape[1])]
    return bins

def compute_jsd(total_data, class_data, bin_width):
    """Compute Jensen-Shannon Divergence between two distributions."""
    if class_data.size == 0 or total_data.size == 0:
        return None
    bins = get_bins(total_data, bin_width)
    total_hist, _ = np.histogramdd(total_data, bins=bins, density=True)
    class_hist, _ = np.histogramdd(class_data, bins=bins, density=True)
    return jensenshannon(class_hist.ravel(), total_hist.ravel())

def compute_kl(total_data, class_data, bin_width):
    """Compute Kullback-Leibler Divergence between two distributions."""
    if class_data.size == 0 or total_data.size == 0:
        return None
    bins = get_bins(total_data, bin_width)
    total_hist, _ = np.histogramdd(total_data, bins=bins, density=True)
    class_hist, _ = np.histogramdd(class_data, bins=bins, density=True)
    return entropy(class_hist.ravel(), total_hist.ravel())

def compute_emd(total_data, class_data, bin_width):
    """Compute Earth Mover's Distance (Wasserstein Distance) between two distributions."""
    if class_data.size == 0 or total_data.size == 0:
        return None
    bins = get_bins(total_data, bin_width)
    total_hist, _ = np.histogramdd(total_data, bins=bins, density=True)
    class_hist, _ = np.histogramdd(class_data, bins=bins, density=True)
    return wasserstein_distance(class_hist.ravel(), total_hist.ravel())

def plot_ect_means_violin(save_dir, class_stats_df, class_means, matrix_means, labels, shapes, taxa_type, IQR_X=1.0, N=30):
    """
    Plot a split violin plot of matrix means for each taxa, with shapes overlaid based on IQR-filtered mean.

    Parameters:
    - save_dir: directory where the plot will be saved
    - class_stats_df: DataFrame with each class's statistics (mean, SD, max, min)
    - class_means: list of mean values for each matrix (used for shape matching)
    - matrix_means: list of mean values for each matrix
    - labels: list of labels corresponding to each matrix mean
    - shapes: list of shapes corresponding to each matrix mean
    - taxa_type: string used for labeling plot axes and filename
    """
    # Sort stats_df by mean and limit to at most N rows
    stats_df = class_stats_df.sort_values(by='ect_mean').reset_index(drop=True)
    if len(stats_df) > N:
        stats_df = stats_df.iloc[np.linspace(0, len(stats_df) - 1, N, dtype=int)]

    # Prepare data for the violin plot
    plot_data = pd.DataFrame({
        taxa_type: labels,
        'matrix_mean': matrix_means
    })
    
    # Subset plot_data to only include classes present in the filtered stats_df
    plot_data = plot_data[plot_data[taxa_type].isin(stats_df[taxa_type])]

    # Calculate the IQR and IQR_X * IQR range for each class
    iqr_filtered_data = []
    full_data = {}  # Store all data (IQR + outliers) for each label
    iqr_means = {}  # Store IQR-filtered means for each class
    for label in stats_df[taxa_type]:
        class_data = plot_data[plot_data[taxa_type] == label]
        q1 = class_data['matrix_mean'].quantile(0.25)
        q3 = class_data['matrix_mean'].quantile(0.75)
        iqr = q3 - q1
        filtered_data = class_data[(class_data['matrix_mean'] >= q1 - IQR_X * iqr) &
                                   (class_data['matrix_mean'] <= q3 + IQR_X * iqr)]
        iqr_filtered_data.append(filtered_data)
        iqr_means[label] = filtered_data['matrix_mean'].mean()  # Calculate mean of IQR-filtered data
        full_data[label] = class_data  # Store full data (IQR + outliers) for each label

    # Concatenate the filtered data into a new DataFrame for "IQR"
    iqr_filtered_data = pd.concat(iqr_filtered_data)
    iqr_filtered_data['dataset'] = 'IQR'  # Label as Class 2 for split violin

    # Label the original data as "All"
    plot_data['dataset'] = 'All'

    # Combine the original and IQR-filtered data
    combined_data = pd.concat([plot_data, iqr_filtered_data])

    # Create the split violin plot
    num_classes = len(stats_df[taxa_type].unique())

    # Dynamically adjust figure height based on the number of classes
    plt.figure(figsize=(12, max(4, int(num_classes*4))), constrained_layout=True)  

    ax = sns.violinplot(y=taxa_type, x='matrix_mean', hue='dataset', data=combined_data, orient='h', 
                    order=stats_df[taxa_type], split=True, gap=0.1, inner="quart", 
                    palette=[(0.4, 0.4, 0.4, 0.5), (0.17254902, 0.62745098, 0.17254902, 1)])  # RGBA for black and green with alpha 0.8
    plt.xlabel("Mean ECT Value")
    plt.ylabel(taxa_type)
    
    # Map y-axis tick labels to their positions
    y_positions = {label.get_text(): pos for label, pos in zip(ax.get_yticklabels(), ax.get_yticks())}

    # Calculate the x-position for min, mean, and max shapes on the left side (IQR-only)
    overall_min_x = min(class_stats_df['ect_mean']) - 0.7
    min_x = overall_min_x - 0.2
    mean_x = overall_min_x
    max_x = overall_min_x + 0.2

    # Helper function to scale shapes proportionally
    def scale_shape_proportionally(shape, scale_factor=0.25):
        max_dim = np.max(np.ptp(shape, axis=0))  # Find the maximum dimension span
        shape = (shape - shape.mean(axis=0)) * (scale_factor / max_dim)  # Scale proportionally
        return shape

    # Overlay shapes (min, mean, max) for each class on the left (IQR-only)
    for i, row in stats_df.iterrows():
        current_label = row[taxa_type]
        mean_value = iqr_means.get(current_label, row['ect_mean'])  # Use IQR-filtered mean if available

        # Subset only the IQR-filtered data for this label
        filtered_means = iqr_filtered_data[iqr_filtered_data[taxa_type] == current_label]['matrix_mean'].values
        filtered_shapes = [shape for mean, label, shape in zip(matrix_means, labels, shapes) 
                           if label == current_label and mean in filtered_means]

        # Check if there are any shapes to overlay
        if not filtered_shapes:
            print(f"No shapes found for {current_label} within IQR range. Skipping overlay.")
            continue

        # Find indices of min, mean, and max shapes within the IQR-filtered data
        min_idx = np.argmin(filtered_means)
        max_idx = np.argmax(filtered_means)
        closest_mean_idx = np.argmin([abs(mean - mean_value) for mean in filtered_means])

        # Get shapes for min, mean, and max (IQR-only)
        min_shape = scale_shape_proportionally(filtered_shapes[min_idx])
        mean_shape = scale_shape_proportionally(filtered_shapes[closest_mean_idx])
        max_shape = scale_shape_proportionally(filtered_shapes[max_idx])

        # Align shapes to the left side of the plot with respective x-offsets
        y_position = y_positions[current_label]  # Get exact y-axis position for current label

        # Plot min shape
        min_shape_scaled = min_shape + [min_x, y_position]
        plt.fill(min_shape_scaled[:, 0], min_shape_scaled[:, 1], color="green", alpha=0.4)

        # Plot mean shape
        mean_shape_scaled = mean_shape + [mean_x, y_position]
        plt.fill(mean_shape_scaled[:, 0], mean_shape_scaled[:, 1], color="green", alpha=0.8)

        # Plot max shape
        max_shape_scaled = max_shape + [max_x, y_position]
        plt.fill(max_shape_scaled[:, 0], max_shape_scaled[:, 1], color="green", alpha=0.4)

        # Overlay "All" min, mean, and max shapes for each row on the far right (IQR + outliers)
        overall_max_x = max(class_stats_df['ect_mean']) + 0.7
        min_all_x = overall_max_x - 0.2
        mean_all_x = overall_max_x
        max_all_x = overall_max_x + 0.2

        # Subset the full data (IQR + outliers) for this label
        all_means = full_data[current_label]['matrix_mean'].values
        all_shapes = [shape for mean, label, shape in zip(matrix_means, labels, shapes) 
                      if label == current_label]

        # Find indices of min, mean, and max shapes within the full data
        min_all_idx = np.argmin(all_means)
        max_all_idx = np.argmax(all_means)
        global_mean_value = np.mean(all_means)
        closest_all_mean_idx = np.argmin([abs(mean - global_mean_value) for mean in all_means])

        # Get shapes for min, mean, and max (IQR + outliers)
        min_all_shape = scale_shape_proportionally(all_shapes[min_all_idx])
        mean_all_shape = scale_shape_proportionally(all_shapes[closest_all_mean_idx])
        max_all_shape = scale_shape_proportionally(all_shapes[max_all_idx])

        # Plot min shape for all data on the far right
        min_all_shape_scaled = min_all_shape + [min_all_x, y_position]
        plt.fill(min_all_shape_scaled[:, 0], min_all_shape_scaled[:, 1], color="black", alpha=0.4)

        # Plot mean shape for all data on the far right
        mean_all_shape_scaled = mean_all_shape + [mean_all_x, y_position]
        plt.fill(mean_all_shape_scaled[:, 0], mean_all_shape_scaled[:, 1], color="black", alpha=0.8)

        # Plot max shape for all data on the far right
        max_all_shape_scaled = max_all_shape + [max_all_x, y_position]
        plt.fill(max_all_shape_scaled[:, 0], max_all_shape_scaled[:, 1], color="black", alpha=0.4)

    plt.savefig(os.path.join(save_dir, f"ECT_means_plot_by_{taxa_type}.png"), dpi=300)
    plt.close()

def compute_ect_means(save_dir, matrix_means, shapes, labels, taxa_type, do_print=True):
    """
    Compute the mean, standard deviation, max, and min of matrix means grouped by labels,
    and save the results to a CSV file.

    Parameters:
    - save_dir: directory where the CSV file will be saved
    - matrix_means: list of mean values for each matrix
    - labels: list of labels corresponding to each matrix mean
    - taxa_type: string to include in the filename for specificity
    - do_print: boolean, if True, prints out each class mean, standard deviation, max, and min

    Returns:
    - class_stats: dictionary with labels as keys and a dictionary of statistics (mean, SD, max, min) as values
    """
    # Group means by labels
    class_means = defaultdict(list)
    for label, mean in zip(labels, matrix_means):
        class_means[label].append(mean)

    # Calculate mean, SD, max, and min for each class and store in class_stats dictionary
    class_stats = {
        label: {
            'mean': np.mean(means),
            'sd': np.std(means),
            'max': np.max(means),
            'min': np.min(means)
        }
        for label, means in class_means.items()
    }

    # Optionally print each class statistics
    if do_print:
        for label, stats in class_stats.items():
            print(f"{label}: Mean = {stats['mean']}, SD = {stats['sd']}, Max = {stats['max']}, Min = {stats['min']}")
            
    # Convert to DataFrame and save to CSV
    class_stats_df = pd.DataFrame(
        [(label, stats['mean'], stats['sd'], stats['max'], stats['min']) for label, stats in class_stats.items()],
        columns=[taxa_type, 'ect_mean', 'ect_sd', 'ect_max', 'ect_min']
    )
    os.makedirs(save_dir, exist_ok=True)
    class_stats_df.to_csv(os.path.join(save_dir, f"ECT_means_by_{taxa_type}.csv"), index=False)

    plot_ect_means_violin(save_dir, class_stats_df, class_means, matrix_means, labels, shapes, taxa_type)

    return class_stats

def compute_metrics(save_dir, umap_scores2D, umap_scores3D, labels_fullname, labels_genus, taxa, test_type, binning_method="sqrt"):

    core_metrics = {}

    # Calculate bin width based on the chosen method
    bin_width_2D = calculate_bin_width(umap_scores2D, method=binning_method)
    bin_width_3D = bin_width_2D  # Use the same bin width for 3D as for 2D


    # Check if labels_fullname has more than one unique value
    if len(set(labels_fullname)) > 1:
        core_metrics['fullname'] = {
            'davies_bouldin_score_2D': davies_bouldin_score(umap_scores2D, labels_fullname),
            'davies_bouldin_score_3D': davies_bouldin_score(umap_scores3D, labels_fullname),
            'silhouette_score_2D': silhouette_score(umap_scores2D, labels_fullname),
            'silhouette_score_3D': silhouette_score(umap_scores3D, labels_fullname),
            'calinski_harabasz_score_2D': calinski_harabasz_score(umap_scores2D, labels_fullname),
            'calinski_harabasz_score_3D': calinski_harabasz_score(umap_scores3D, labels_fullname),
        }
    else:
        core_metrics['fullname'] = {key: None for key in [
            'davies_bouldin_score_2D', 'davies_bouldin_score_3D', 'silhouette_score_2D', 'silhouette_score_3D',
            'calinski_harabasz_score_2D', 'calinski_harabasz_score_3D',
        ]}

    # Check if labels_genus has more than one unique value
    if len(set(labels_genus)) > 1:
        core_metrics['genus'] = {
            'davies_bouldin_score_2D': davies_bouldin_score(umap_scores2D, labels_genus),
            'davies_bouldin_score_3D': davies_bouldin_score(umap_scores3D, labels_genus),
            'silhouette_score_2D': silhouette_score(umap_scores2D, labels_genus),
            'silhouette_score_3D': silhouette_score(umap_scores3D, labels_genus),
            'calinski_harabasz_score_2D': calinski_harabasz_score(umap_scores2D, labels_genus),
            'calinski_harabasz_score_3D': calinski_harabasz_score(umap_scores3D, labels_genus),
        }
    else:
        core_metrics['genus'] = {key: None for key in [
            'davies_bouldin_score_2D', 'davies_bouldin_score_3D', 'silhouette_score_2D', 'silhouette_score_3D',
            'calinski_harabasz_score_2D', 'calinski_harabasz_score_3D',
        ]}

    # Save the main metrics to a CSV file
    core_metrics_df = pd.DataFrame(core_metrics).T
    core_metrics_df.to_csv(os.path.join(save_dir, f"{test_type}_{taxa}_metrics.csv"))


    comparison_metrics = []
    total_data_2D = umap_scores2D
    total_data_3D = umap_scores3D

    # Calculate comparison metrics for each unique class in labels_fullname
    if len(set(labels_fullname)) > 1:
        bins2D = [np.arange(np.min(total_data_2D[:, dim]), np.max(total_data_2D[:, dim]) + bin_width_2D, bin_width_2D) for dim in range(total_data_2D.shape[1])]
        for class_label in tqdm(np.unique(labels_fullname), desc="Processing Fullnames"):
            mask = np.array([label == class_label for label in labels_fullname])
            class_data_2D = umap_scores2D[mask]
            class_data_3D = umap_scores3D[mask]

            if class_data_2D.size > 0 and class_data_3D.size > 0:
                comparison_metrics.append({
                    'class': class_label,
                    'type': 'fullname',
                    'bin_width': bin_width_2D,
                    'jsd_2D': compute_jsd(total_data_2D, class_data_2D, bin_width_2D),
                    'jsd_3D': compute_jsd(total_data_3D, class_data_3D, bin_width_3D),
                    'kl_divergence_2D': compute_kl(total_data_2D, class_data_2D, bin_width_2D),
                    'kl_divergence_3D': compute_kl(total_data_3D, class_data_3D, bin_width_3D),
                    'emd_2D': compute_emd(total_data_2D, class_data_2D, bin_width_2D),
                    'emd_3D': compute_emd(total_data_3D, class_data_3D, bin_width_3D),
                })

                # plot_2d_distribution(total_data_2D, bins2D, class_data_2D, bin_width_2D, class_label, save_dir, test_type, taxa)

    # Calculate comparison metrics for each unique class in labels_genus
    if len(set(labels_genus)) > 1:
        bins2D = [np.arange(np.min(total_data_2D[:, dim]), np.max(total_data_2D[:, dim]) + bin_width_2D, bin_width_2D) for dim in range(total_data_2D.shape[1])]
        # bins3D = [np.arange(np.min(total_data_3D[:, dim]), np.max(total_data_3D[:, dim]) + bin_width_3D, bin_width_3D) for dim in range(total_data_3D.shape[1])]
        
        # total_hist3, edges3 = np.histogramdd(total_data_3D, bins=bins3D, density=True)
        
        for class_label in tqdm(np.unique(labels_genus), desc="Processing Genus Labels"):
            mask = np.array([label == class_label for label in labels_genus])
            class_data_2D = umap_scores2D[mask]
            class_data_3D = umap_scores3D[mask]

            if class_data_2D.size > 0 and class_data_3D.size > 0:
                comparison_metrics.append({
                    'class': class_label,
                    'type': 'genus',
                    'bin_width': bin_width_2D,
                    'jsd_2D': compute_jsd(total_data_2D, class_data_2D, bin_width_2D),
                    'jsd_3D': compute_jsd(total_data_3D, class_data_3D, bin_width_3D),
                    'kl_divergence_2D': compute_kl(total_data_2D, class_data_2D, bin_width_2D),
                    'kl_divergence_3D': compute_kl(total_data_3D, class_data_3D, bin_width_3D),
                    'emd_2D': compute_emd(total_data_2D, class_data_2D, bin_width_2D),
                    'emd_3D': compute_emd(total_data_3D, class_data_3D, bin_width_3D),
                })

                # Call plotting functions for visual comparison
                plot_2d_distribution(total_data_2D, bins2D, class_data_2D, bin_width_2D, class_label, save_dir, test_type, taxa)


    # Save comparison metrics to a CSV file
    comparison_metrics_df = pd.DataFrame(comparison_metrics)
    comparison_metrics_df.to_csv(os.path.join(save_dir, f"{test_type}_{taxa}_metrics_comparison.csv"), index=False)

    return core_metrics, comparison_metrics





















# 1. Density Dropoff Rate (DDR)
def exponential_decay(x, a, b):
    return a * np.exp(-b * x)

def calculate_ddr(matrix):
    unique_values = np.unique(matrix[matrix > 0])[::-1]  # Descending order
    densities = []
    
    # Calculate densities
    for value in unique_values:
        density = np.sum(matrix == value) / matrix.size
        densities.append(density)
    
    levels = np.arange(len(densities))
    
    # Initial guesses for p0
    p0_values = [(max(densities), 0.01), (max(densities), 0.001), (max(densities), 0.0001), (max(densities), 0.00001), (max(densities), 0.1), (max(densities), 0.25), (max(densities), 0.5), (max(densities), 1)]
    # print(densities)
    maxfev = 5000  # Increase max function evaluations

    for p0 in p0_values:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)  # Suppress warnings if fitting fails
            try:
                # Fit the exponential decay curve
                popt, _ = curve_fit(exponential_decay, levels, densities, p0=p0, maxfev=maxfev)
                
                # Check if decay rate (b) is positive
                if popt[1] > 0:
                    print(f"Positive decay rate ({popt[1]}) with p0={p0}. Retrying with smaller p0...")
                else:
                    # Return the decay rate if it's not positive
                    return popt[1]
            except RuntimeError:
                print(f"Curve fitting failed with p0={p0}: continuing to next p0...")
    
    # If all retries failed, return NaN
    print("All retries failed to produce a valid negative decay rate.")
    return np.nan

# 2. Cumulative Density Index (CDI)
def calculate_cdi(matrix):
    unique_values = np.unique(matrix[matrix > 0])
    cdi = 0
    for value in unique_values:
        density = np.sum(matrix == value) / matrix.size
        cdi += value * density
    return cdi

# 3. Hierarchical Spread Index (HSI)
def calculate_hsi(matrix):
    unique_values = np.unique(matrix[matrix > 0])
    region_areas = []
    for value in unique_values:
        labeled_matrix, num_features = label(matrix == value)
        for region in range(1, num_features + 1):
            region_area = np.sum(labeled_matrix == region)
            region_areas.append(region_area)
    
    return np.std(region_areas) / len(unique_values) if region_areas else 0


# 4. Weighted Average Density
def calculate_weighted_average_density(matrix):
    unique_values = np.unique(matrix[matrix > 0])
    total_density = 0
    total_weight = 0
    for value in unique_values:
        density = np.sum(matrix == value) / matrix.size
        total_density += value * density
        total_weight += value
    return total_density / total_weight if total_weight > 0 else 0

def compute_ect_summary_metrics(save_dir, LM2_measurements, ect_data, component_names, matrix_means, ect_matrix_means_fullname, ect_matrix_means_genus, ect_matrix_means_family, labels_fullname, labels_genus, labels_family, do_print=True):
    """
    Wrapper function to compute shape metrics (DDR, CDI, HSI, Weighted Average Density)
    for each matrix in the directory and save results.

    Parameters:
    - dir_path: Directory containing ECT data for a single family.
    - save_dir: Directory to save output CSV.
    - taxa_type: Specific identifier for the taxa being analyzed.
    - do_print: If True, prints out metrics for each sample.

    Returns:
    - metrics_summary: Dictionary summarizing metrics for each matrix and class.
    """
    metrics_summary_fullname = defaultdict(lambda: defaultdict(list))
    metrics_summary_genus = defaultdict(lambda: defaultdict(list))
    metrics_summary_family = defaultdict(lambda: defaultdict(list))
    results = []  # List to store row data for the CSV

    zipped_data = list(zip(ect_data, component_names, matrix_means, labels_fullname, labels_genus, labels_family))
    progress_bar = tqdm(total=len(zipped_data), desc="Computing summary metrics")

    for matrix, filename, ECT_Density, fullname, genus, family in zipped_data:
        ddr = calculate_ddr(matrix)
        cdi = calculate_cdi(matrix)
        hsi = calculate_hsi(matrix)
        weighted_avg_density = calculate_weighted_average_density(matrix)

        # Store metrics in the summary dictionary under the label of each class
        metrics_summary_fullname[fullname]['DDR'].append(ddr)
        metrics_summary_fullname[fullname]['CDI'].append(cdi)
        metrics_summary_fullname[fullname]['HSI'].append(hsi)
        metrics_summary_fullname[fullname]['Weighted_Avg_Density'].append(weighted_avg_density)

        metrics_summary_genus[genus]['DDR'].append(ddr)
        metrics_summary_genus[genus]['CDI'].append(cdi)
        metrics_summary_genus[genus]['HSI'].append(hsi)
        metrics_summary_genus[genus]['Weighted_Avg_Density'].append(weighted_avg_density)

        metrics_summary_family[family]['DDR'].append(ddr)
        metrics_summary_family[family]['CDI'].append(cdi)
        metrics_summary_family[family]['HSI'].append(hsi)
        metrics_summary_family[family]['Weighted_Avg_Density'].append(weighted_avg_density)
        
        if do_print:
            print(f"{fullname} ({genus} - {family}): DDR = {ddr}, CDI = {cdi}, HSI = {hsi}, Weighted Average Density = {weighted_avg_density}")

        columns_to_log = [
            'area', 'perimeter', 'convex_hull', 'bbox_min_long_side', 'bbox_min_short_side', 
            'distance_lamina', 'distance_width', 'distance_petiole', 'distance_midvein_span', 
            'distance_petiole_span', 'trace_midvein_distance', 'trace_petiole_distance'
        ]
        
        # Create a dictionary to hold the current row's data
        row = {
            'filename': filename,
            'fullname': fullname,
            'genus': genus,
            'family': family,
            'DDR': ddr,
            'CDI': cdi,
            'HSI': hsi,
            'WAD': weighted_avg_density,
            'ECT_Density': ECT_Density,
            'AVG_ECT_Density_Family': ect_matrix_means_family[family]['mean'],
            'AVG_ECT_Density_Genus': ect_matrix_means_genus[genus]['mean'],
            'AVG_ECT_Density_Species': ect_matrix_means_fullname[fullname]['mean'],
        }  

        # Match LM2_measurements to the current filename
        matching_rows = LM2_measurements[LM2_measurements['component_name'] == filename]

        # Add matching row data to the current row, excluding 'component_name'
        if not matching_rows.empty:
            for col in matching_rows.columns:
                if col != 'component_name':
                    row[col] = matching_rows.iloc[0][col]

            # Calculate solidity and add it to the row
            area = matching_rows.iloc[0].get('area', np.nan)
            convex_hull = matching_rows.iloc[0].get('convex_hull', np.nan)
            if convex_hull > 0:  # Avoid division by zero
                row['solidity'] = area / convex_hull
            else:
                row['solidity'] = 0

            # Calculate natural log for specified columns
            for col in columns_to_log:
                value = matching_rows.iloc[0].get(col, np.nan)
                row[f"log_{col}"] = np.log(value) if value > 0 else np.nan

        # Append the row to the results list
        results.append(row)
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    # Convert results to a DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'summary_metrics.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")


    run_and_save_anova(results_df, 'fullname', save_dir, output_filename='ANOVA.csv')



    def create_class_stats(metrics_summary, taxa_type):
        # Compute overall statistics by class, including count of entries per label
        class_stats = {
            label: {
                'n': len(metrics['DDR']),
                'DDR_mean': np.mean(metrics['DDR']),
                'DDR_sd': np.std(metrics['DDR']),
                'CDI_mean': np.mean(metrics['CDI']),
                'CDI_sd': np.std(metrics['CDI']),
                'HSI_mean': np.mean(metrics['HSI']),
                'HSI_sd': np.std(metrics['HSI']),
                'Weighted_Avg_Density_mean': np.mean(metrics['Weighted_Avg_Density']),
                'Weighted_Avg_Density_sd': np.std(metrics['Weighted_Avg_Density']),
            }
            for label, metrics in metrics_summary.items()
        }

        # Save class_stats to CSV
        os.makedirs(save_dir, exist_ok=True)
        class_stats_df = pd.DataFrame.from_dict(class_stats, orient='index')
        class_stats_df.to_csv(os.path.join(save_dir, f"ECT_summary_metrics_by_{taxa_type}.csv"))
        
        return class_stats
    
    class_stats_fullname = create_class_stats(metrics_summary_fullname, 'fullname')
    class_stats_genus = create_class_stats(metrics_summary_genus, 'genus')
    class_stats_family = create_class_stats(metrics_summary_family, 'family')

    return (metrics_summary_fullname, metrics_summary_genus, metrics_summary_family, class_stats_fullname, class_stats_genus, class_stats_family)

def run_and_save_anova(df, group_col, save_dir, output_filename='ANOVA.csv'):

    def calculate_anova(data, group_col, variables):
        results = {}
        groups = data[group_col].unique()
        for var in variables:
            # Collect data for each group
            group_data = [data[data[group_col] == group][var].dropna() for group in groups]
            # Perform ANOVA
            f_stat, p_val = f_oneway(*group_data)
            results[var] = (f_stat, p_val)  # Store both F-statistic and P-value
        return results
    # Define the list of all metrics in the dataframe
    metrics = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
    
    # Calculate ANOVA for each metric
    anova_results = calculate_anova(df, group_col, metrics)
    
    # Create a DataFrame from the ANOVA results
    anova_df = pd.DataFrame([
        {'metric': metric, 'F_stat': result[0], 'P_value': result[1]}
        for metric, result in anova_results.items()
    ])
    
    # Save the results to a CSV file
    output_path = os.path.join(save_dir, output_filename)
    anova_df.to_csv(output_path, index=False)
    print(f"ANOVA results saved to {output_path}")

def metric_evaluation(metrics_summary, labels_fullname):
    """
    Evaluate clustering effectiveness of each metric (DDR, CDI, HSI, Weighted Average Density)
    by measuring intra-class compactness and inter-class separability.
    
    Parameters:
    - metrics_summary: Dictionary containing metrics for each matrix, grouped by class.
    - labels_fullname: List of class labels corresponding to each matrix.

    Returns:
    - best_metric: The metric that best clusters data by class.
    - clustering_scores: Dictionary of clustering scores for each metric.
    """
    metrics_data = { 
        "DDR": [], 
        "CDI": [], 
        "HSI": [], 
        "Weighted_Avg_Density": [] 
    }
    labels = []
    
    # Flatten metrics_summary data and prepare it for clustering evaluation
    for label, metrics in metrics_summary.items():
        num_samples = len(metrics['DDR'])  # Assuming all metrics have the same length
        metrics_data["DDR"].extend(metrics['DDR'])
        metrics_data["CDI"].extend(metrics['CDI'])
        metrics_data["HSI"].extend(metrics['HSI'])
        metrics_data["Weighted_Avg_Density"].extend(metrics['Weighted_Avg_Density'])
        labels.extend([label] * num_samples)
    
    clustering_scores = {}

    # Evaluate each metric separately
    for metric_name, values in metrics_data.items():
        # Standardize values for silhouette score
        values_array = np.array(values).reshape(-1, 1)
        scaler = StandardScaler()
        standardized_values = scaler.fit_transform(values_array)
        
        # Calculate silhouette score
        try:
            silhouette = silhouette_score(standardized_values, labels)
        except ValueError:
            silhouette = np.nan  # Silhouette score requires at least 2 distinct classes
            
        # Perform one-way ANOVA for separability between classes
        grouped_values = [np.array(values)[np.array(labels) == label] for label in set(labels)]
        f_stat, p_value = f_oneway(*grouped_values)
        
        clustering_scores[metric_name] = {
            'silhouette_score': silhouette,
            'f_statistic': f_stat,
            'p_value': p_value
        }

    # Select the best metric based on highest silhouette score and significant ANOVA (low p-value)
    best_metric = max(clustering_scores, key=lambda k: (
        clustering_scores[k]['silhouette_score'], -clustering_scores[k]['p_value']
    ))

    return best_metric, clustering_scores

def plot_ddr_hsi_scatter_with_sd(save_dir, class_stats, matrix_means, labels_fullname, shapes, filename="plot_ddr_hsi.png"):
    """
    Plot DDR_mean on the X-axis and HSI_mean on the Y-axis, with each point represented by a scaled version 
    of the shape closest to the median of the matrix means. Saves the plot as a static image.
    
    Parameters:
    - save_dir: Directory to save the image file.
    - class_stats: Dictionary with class labels as keys and a dictionary of statistics (including DDR_mean, HSI_mean, and DDR_sd) as values.
    - matrix_means: List of mean values for each matrix.
    - labels_fullname: List of labels corresponding to each matrix mean.
    - shapes: List of shapes (as arrays) corresponding to each matrix mean.
    - filename: The name of the image file to save.
    """
    # Convert class_stats dictionary to DataFrame for easier plotting
    df = pd.DataFrame.from_dict(class_stats, orient='index')
    df['Label'] = df.index  # Add the label as a column
    df['Category'] = df['Label'].apply(lambda x: x.split('_')[0])  # Extract the category for color coding
    df['Size'] = df['DDR_sd'] * 100  # Scale SD for point diameter

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define color palette with exactly 20 colors (tab20 has 20 colors)
    palette = sns.color_palette("tab20", n_colors=20)

    # Set axis limits for consistent aspect ratio scaling
    x_limits = (df['DDR_mean'].min() - 0.5, df['DDR_mean'].max() + 0.5)
    y_limits = (df['HSI_mean'].min() - 20, df['HSI_mean'].max() + 20)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    try:
        N = 3
        # Perform k-means clustering on the DDR_mean and Z_Mean columns
        kmeans = KMedoids(n_clusters=N, random_state=0)
        cluster_data = df[['DDR_mean', 'HSI_mean']]
        df['Cluster'] = kmeans.fit_predict(cluster_data)
        cluster_centers = kmeans.cluster_centers_

        # Create background shading for each cluster
        cluster_colors = sns.color_palette("pastel", n_colors=N)  # Light colors for shading
        for cluster_idx in range(N):
            cluster_points = df[df['Cluster'] == cluster_idx][['DDR_mean', 'HSI_mean']].values
            
            # Calculate the convex hull for each cluster
            if len(cluster_points) > 2:  # ConvexHull requires at least 3 points
                hull = ConvexHull(cluster_points)
                hull_vertices = cluster_points[hull.vertices]

                # Plot the convex hull as a shaded area
                ax.fill(hull_vertices[:, 0], hull_vertices[:, 1], color=cluster_colors[cluster_idx], alpha=0.2)
    except:
        pass

    # Calculate the aspect ratio of the plot
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    aspect_ratio = x_range / y_range
    legend_elements = []


    # Loop through each category and plot with different colors
    unique_categories = df['Category'].unique()
    for i, category in enumerate(unique_categories):
        subset = df[df['Category'] == category]
        color = palette[i % 20]  # Cycle through colors every 20 categories

        for _, row in subset.iterrows():
            # Get shapes at specific quartiles for the current label
            mean_shape, q25_shape, closest_median_shape = get_mean_shape_for_class(
                class_name=row['Label'],
                matrix_means=matrix_means,
                labels=labels_fullname,
                shapes=shapes
            )
            
            # Skip if no median shape is available
            if closest_median_shape is None:
                continue

            # Scale the shape proportionally based on the aspect ratio and SD-derived size
            def scale_shape_proportionally(shape, scale_factor_x, scale_factor_y):
                shape = shape - shape.mean(axis=0)  # Center the shape
                shape[:, 0] *= scale_factor_x  # Scale x-dimension
                shape[:, 1] *= scale_factor_y  # Scale y-dimension
                return shape

            # Define scaling factors based on the plot aspect ratio and row's SD-derived size
            scale_factor_x = 0.25  # Adjust x-scale based on SD-derived size
            scale_factor_y = scale_factor_x / aspect_ratio  # Adjust y-scale to maintain aspect ratio

            # Apply scaling to the median shape
            scaled_shape = scale_shape_proportionally(closest_median_shape, scale_factor_x, scale_factor_y)

            # Translate the shape to the plot coordinates
            x_offset, y_offset = row['DDR_mean'], row['HSI_mean']
            translated_shape = scaled_shape + [x_offset, y_offset]

            # Plot the median shape at the specified location
            ax.fill(translated_shape[:, 0], translated_shape[:, 1], color=color, alpha=0.6, edgecolor='k', label=category if i == 0 else "")

            # Add shape for legend with the current category and color
            # Prepare label with n for legend
            n = int(row.get('n', 0))  # Assuming 'n' is included in class_stats
            label_with_n = f"{row['Label']} (n={n})"
            legend_elements.append((Polygon(closest_median_shape, closed=True), color, label_with_n))

    # Set plot title and labels
    ax.set_title("DDR Mean vs HSI Mean")
    ax.set_xlabel("DDR Mean")
    ax.set_ylabel("HSI Mean")
    # Prepare legend handlers with proper scaling for visibility
    custom_handlers = {
        patch: ShapeLegendHandler(shape=patch, m1=60, m2=0.75, color=color, scale_factor_x=scale_factor_x, scale_factor_z=scale_factor_y, aspect_ratio_xz=aspect_ratio)
        for patch, color, _ in legend_elements
    }

    # Create a separate legend figure
    legend_fig, legend_ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    legend_ax.legend(
        handles=[patch for patch, _, _ in legend_elements],
        labels=[label for _, _, label in legend_elements],
        handler_map=custom_handlers,
        title="Taxa",
        loc='center',
        ncol=2,
        frameon=False,
    )
    legend_ax.axis('off')  # Turn off the axis

    legend_path = os.path.join(save_dir, filename.replace(".png", "_legend.png"))
    legend_fig.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close(legend_fig)
    
    # Save the plot as a static image
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, filename)
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved as {image_path}")
    plt.close(fig)  # Close the plot to prevent display in notebooks






def plot_ddr_ect_mean_scatter_with_sd(save_dir, class_stats, metric_stats, matrix_means, labels_fullname, shapes, filename="plot_ddr_ectMean.png"):
    # Convert class_stats and metric_stats dictionaries to DataFrames
    class_df = pd.DataFrame.from_dict(class_stats, orient='index')
    metric_df = pd.DataFrame.from_dict(metric_stats, orient='index')
    df = class_df.copy()

    df['Z_Mean'] = metric_df['mean']  # Add mean from metric_stats as Z-axis
    df['Label'] = df.index  # Add the label as a column
    df['Category'] = df['Label'].apply(lambda x: x.split('_')[0])  # Extract the category for color coding
    df['Size'] = df['DDR_sd'] * 100  # Scale SD for point diameter

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define color palette with exactly 20 colors (tab20 has 20 colors)
    palette = sns.color_palette("tab20", n_colors=20)

    # Set axis limits for consistent aspect ratio scaling
    x_limits = (df['DDR_mean'].min() - 0.5, df['DDR_mean'].max() + 0.5)
    y_limits = (df['Z_Mean'].min() - 0.1, df['Z_Mean'].max() + 0.1)

    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    try:
        # Perform k-means clustering on the DDR_mean and Z_Mean columns
        kmeans = KMeans(n_clusters=3, random_state=0)
        cluster_data = df[['DDR_mean', 'Z_Mean']]
        df['Cluster'] = kmeans.fit_predict(cluster_data)
        cluster_centers = kmeans.cluster_centers_

        # Create background shading for each cluster
        cluster_colors = sns.color_palette("pastel", n_colors=3)  # Light colors for shading
        for cluster_idx in range(3):
            cluster_points = df[df['Cluster'] == cluster_idx][['DDR_mean', 'Z_Mean']].values
            
            # Calculate the convex hull for each cluster
            if len(cluster_points) > 2:  # ConvexHull requires at least 3 points
                hull = ConvexHull(cluster_points)
                hull_vertices = cluster_points[hull.vertices]

                # Plot the convex hull as a shaded area
                ax.fill(hull_vertices[:, 0], hull_vertices[:, 1], color=cluster_colors[cluster_idx], alpha=0.2)
    except:
        pass

    # Calculate the aspect ratio of the plot
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    aspect_ratio = x_range / y_range
    legend_elements = []


    # Loop through each category and plot with different colors
    unique_categories = df['Category'].unique()
    for i, category in enumerate(unique_categories):
        subset = df[df['Category'] == category]
        color = palette[i % 20]  # Cycle through colors every 20 categories

        for _, row in subset.iterrows():
            # Get shapes at specific quartiles for the current label
            mean_shape, q25_shape, closest_median_shape = get_mean_shape_for_class(
                class_name=row['Label'],
                matrix_means=matrix_means,
                labels=labels_fullname,
                shapes=shapes
            )
            
            # Skip if no median shape is available
            if closest_median_shape is None:
                continue

            # Scale the shape proportionally based on the aspect ratio and SD-derived size
            def scale_shape_proportionally(shape, scale_factor_x, scale_factor_y):
                shape = shape - shape.mean(axis=0)  # Center the shape
                shape[:, 0] *= scale_factor_x  # Scale x-dimension
                shape[:, 1] *= scale_factor_y  # Scale y-dimension
                return shape

            # Define scaling factors based on the plot aspect ratio and row's SD-derived size
            scale_factor_x = 0.25  # Adjust x-scale based on SD-derived size
            scale_factor_y = scale_factor_x / aspect_ratio  # Adjust y-scale to maintain aspect ratio

            # Apply scaling to the median shape
            scaled_shape = scale_shape_proportionally(closest_median_shape, scale_factor_x, scale_factor_y)

            # Translate the shape to the plot coordinates
            x_offset, y_offset = row['DDR_mean'], row['Z_Mean']
            translated_shape = scaled_shape + [x_offset, y_offset]

            # Plot the median shape at the specified location
            ax.fill(translated_shape[:, 0], translated_shape[:, 1], color=color, alpha=0.6, edgecolor='k', label=category if i == 0 else "")

            # Add shape for legend with the current category and color
            # Prepare label with n for legend
            n = int(row.get('n', 0))  # Assuming 'n' is included in class_stats
            label_with_n = f"{row['Label']} (n={n})"
            legend_elements.append((Polygon(closest_median_shape, closed=True), color, label_with_n))

    # Set plot title and labels
    ax.set_title("DDR Mean vs ECT Density")
    ax.set_xlabel("DDR Mean")
    ax.set_ylabel("ECT Density")
    # Prepare legend handlers with proper scaling for visibility
    custom_handlers = {
        patch: ShapeLegendHandler(shape=patch, m1=50, m2=200, color=color, scale_factor_x=scale_factor_x, scale_factor_z=scale_factor_y, aspect_ratio_xz=aspect_ratio)
        for patch, color, _ in legend_elements
    }

    # Create a separate legend figure
    legend_fig, legend_ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    legend_ax.legend(
        handles=[patch for patch, _, _ in legend_elements],
        labels=[label for _, _, label in legend_elements],
        handler_map=custom_handlers,
        title="Taxa",
        loc='center',
        ncol=2,
        frameon=False,
    )
    legend_ax.axis('off')  # Turn off the axis

    legend_path = os.path.join(save_dir, filename.replace(".png", "_legend.png"))
    legend_fig.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close(legend_fig)
    
    # Save the plot as a static image
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, filename)
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved as {image_path}")
    plt.close(fig)  # Close the plot to prevent display in notebooks















def plot_ddr_Weighted_Avg_Density_mean_scatter_with_sd(save_dir, class_stats, matrix_means, labels_fullname, shapes, filename="plot_ddr_Weighted_Avg_Density_mean.png"):
    """
    Plot DDR_mean on the X-axis and HSI_mean on the Y-axis, with point size determined by SD,
    alpha set to 0.8, and different marker types for categories exceeding color options.
    Saves the plot as a static image.
    
    Parameters:
    - save_dir: Directory to save the image file.
    - class_stats: Dictionary with class labels as keys and a dictionary of statistics (including DDR_mean, HSI_mean, and DDR_sd) as values.
    - filename: The name of the image file to save.
    """
    # Convert class_stats dictionary to DataFrame for easier plotting
    df = pd.DataFrame.from_dict(class_stats, orient='index')
    df['Label'] = df.index  # Add the label as a column
    df['Category'] = df['Label'].apply(lambda x: x.split('_')[0])  # Extract the category for color coding
    df['Size'] = df['DDR_sd'] * 100  # Scale SD for point diameter

    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define color palette with exactly 20 colors (tab20 has 20 colors)
    palette = sns.color_palette("tab20", n_colors=20)

    # Set axis limits for consistent aspect ratio scaling
    x_limits = (df['DDR_mean'].min() - 0.5, df['DDR_mean'].max() + 0.5)
    y_limits = (df['Weighted_Avg_Density_mean'].min() - 0.01, df['Weighted_Avg_Density_mean'].max() + 0.01)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    try:
        # Perform k-means clustering on the DDR_mean and Z_Mean columns
        kmeans = KMeans(n_clusters=3, random_state=0)
        cluster_data = df[['DDR_mean', 'Weighted_Avg_Density_mean']]
        df['Cluster'] = kmeans.fit_predict(cluster_data)
        cluster_centers = kmeans.cluster_centers_

        # Create background shading for each cluster
        cluster_colors = sns.color_palette("pastel", n_colors=3)  # Light colors for shading
        for cluster_idx in range(3):
            cluster_points = df[df['Cluster'] == cluster_idx][['DDR_mean', 'Weighted_Avg_Density_mean']].values
            
            # Calculate the convex hull for each cluster
            if len(cluster_points) > 2:  # ConvexHull requires at least 3 points
                hull = ConvexHull(cluster_points)
                hull_vertices = cluster_points[hull.vertices]

                # Plot the convex hull as a shaded area
                ax.fill(hull_vertices[:, 0], hull_vertices[:, 1], color=cluster_colors[cluster_idx], alpha=0.2)
    except:
        pass    
    # Calculate the aspect ratio of the plot
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    aspect_ratio = x_range / y_range

    legend_elements = []


    # # Loop through each category and plot with different colors and shapes
    unique_categories = df['Category'].unique()
    for i, category in enumerate(unique_categories):

        subset = df[df['Category'] == category]
        color = palette[i % 20]  # Cycle through colors every 20 categories

        for _, row in subset.iterrows():
            # Get the mean shape for the current label
            mean_shape, q25_shape, closest_median_shape = get_mean_shape_for_class(
                class_name=row['Label'],
                matrix_means=matrix_means,
                labels=labels_fullname,
                shapes=shapes
            )
            
            # Skip if no mean shape is available
            if mean_shape is None:
                continue

            # Scale the shape proportionally based on the aspect ratio and SD-derived size
            def scale_shape_proportionally(shape, scale_factor_x, scale_factor_y):
                shape = shape - shape.mean(axis=0)  # Center the shape
                shape[:, 0] *= scale_factor_x  # Scale x-dimension
                shape[:, 1] *= scale_factor_y  # Scale y-dimension
                return shape

            # Define scaling factors based on the plot aspect ratio and row's SD-derived size
            scale_factor_x = 0.25   # Adjust x-scale based on SD-derived size
            scale_factor_y = scale_factor_x / aspect_ratio  # Adjust y-scale to maintain aspect ratio

            # Apply scaling to the mean shape
            scaled_shape = scale_shape_proportionally(closest_median_shape, scale_factor_x, scale_factor_y)

            # Translate the shape to the plot coordinates
            x_offset, y_offset = row['DDR_mean'], row['Weighted_Avg_Density_mean']
            translated_shape = scaled_shape + [x_offset, y_offset]

            # Plot the mean shape at the specified location
            ax.fill(translated_shape[:, 0], translated_shape[:, 1], color=color, alpha=0.6, edgecolor='k', label=category if i == 0 else "")

            # Add shape for legend with the current category and color
            # Prepare label with n for legend
            n = int(row.get('n', 0))  # Assuming 'n' is included in class_stats
            label_with_n = f"{row['Label']} (n={n})"
            legend_elements.append((Polygon(closest_median_shape, closed=True), color, label_with_n))

    # Set plot title and labels
    ax.set_title("DDR Mean vs Weighted Avg Density Mean")
    ax.set_xlabel("DDR Mean")
    ax.set_ylabel("Weighted Avg Density Mean")
    # Prepare legend handlers with proper scaling for visibility
    custom_handlers = {
        patch: ShapeLegendHandler(shape=patch, m1=50, m2=2000, color=color, scale_factor_x=scale_factor_x, scale_factor_z=scale_factor_y, aspect_ratio_xz=aspect_ratio)
        for patch, color, _ in legend_elements
    }

    # Create a separate legend figure
    legend_fig, legend_ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    legend_ax.legend(
        handles=[patch for patch, _, _ in legend_elements],
        labels=[label for _, _, label in legend_elements],
        handler_map=custom_handlers,
        title="Taxa",
        loc='center',
        ncol=2,
        frameon=False,
    )
    legend_ax.axis('off')  # Turn off the axis

    legend_path = os.path.join(save_dir, filename.replace(".png", "_legend.png"))
    legend_fig.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close(legend_fig)
    
    # Save the plot as a static image
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, filename)
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved as {image_path}")
    plt.close(fig)  # Close the plot to prevent display in notebooks




    
# Custom handler to render the shape in the legend with preserved aspect ratio and uniform scaling
class ShapeLegendHandler(HandlerPatch):
    """ Custom legend handler to draw each shape in the legend with uniform scaling, preserving aspect ratio. """
    def __init__(self, shape, color, m1, m2, scale_factor_x, scale_factor_z, aspect_ratio_xz=1.0, **kwargs):
        self.m1 = m1
        self.m2 = m2
        self.shape = shape
        self.color = color
        self.scale_x = scale_factor_x
        self.scale_y = scale_factor_z  # Derived scale for y to preserve aspect ratio
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        # Get the original shape vertices
        verts = self.shape.get_xy()
        
        # Center the shape for consistent placement
        centered_shape = verts - verts.mean(axis=0)
        
        # Apply uniform scaling factors to maintain aspect ratio
        scaled_shape = np.copy(centered_shape)
        scaled_shape[:, 0] *= self.scale_x * self.m1  # Scale x dimension
        scaled_shape[:, 1] *= self.scale_y * self.m2 # Scale y dimension to preserve aspect ratio

        # Create and return the polygon with the transformed vertices
        p = mpatches.Polygon(scaled_shape, closed=True, transform=trans, facecolor=self.color, edgecolor='none', alpha=0.6)
        return [p]
    
def plot_ddr_Weighted_Avg_Density_mean_ect_3d_to_gif(save_dir, class_stats, metric_stats, matrix_means, labels_fullname, shapes, filename="plot_ddr_Weighted_Avg_Density_mean_ectMean_3d.gif"):
    """
    Plot DDR_mean on the X-axis, Weighted_Avg_Density_mean on the Y-axis, and mean from metric_stats on the Z-axis,
    with shapes representing each class, oriented in the x-z plane, and counter-rotating to face the viewer.
    """
    # Convert class_stats and metric_stats dictionaries to DataFrames
    class_df = pd.DataFrame.from_dict(class_stats, orient='index')
    metric_df = pd.DataFrame.from_dict(metric_stats, orient='index')
    
    # Combine the data for plotting
    df = class_df.copy()
    df['Z_Mean'] = metric_df['mean']  # Add mean from metric_stats as Z-axis
    df['Size'] = metric_df['sd'] * 200  # Use sd from metric_stats for point size
    df['Label'] = df.index  # Add the label as a column
    df['Category'] = df['Label'].apply(lambda x: x.split('_')[0])  # Extract the category for color coding

    # Initialize 3D plot
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')

    # Define color palette with exactly 20 colors (tab20 has 20 colors)
    palette = sns.color_palette("tab20", n_colors=20)

    # Set axis limits for consistent aspect ratio scaling
    x_limits = (df['DDR_mean'].min() - 0.5, df['DDR_mean'].max() + 0.5)
    y_limits = (df['Weighted_Avg_Density_mean'].min() - 0.02, df['Weighted_Avg_Density_mean'].max() + 0.02)
    z_limits = (df['Z_Mean'].min() - 0.01, df['Z_Mean'].max() + 0.01)
    print(x_limits, y_limits, z_limits)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_zlim(z_limits)

    # Calculate aspect ratio for consistent scaling in 3D
    x_range = x_limits[1] - x_limits[0]
    z_range = z_limits[1] - z_limits[0]
    aspect_ratio_xz = x_range / z_range

    # Prepare lists to store all shapes for animation and legend elements
    polygons = []
    legend_elements = []

    # Loop through each category and plot with different colors
    # for i, (category, color) in enumerate(zip(df['Category'], palette[:len(df['Category'])])):
    unique_categories = df['Category'].unique()
    for i, category in enumerate(unique_categories):
        subset = df[df['Category'] == category]
        color = palette[i % 20]  # Cycle through colors every 20 categories

        for _, row in subset.iterrows():
            mean_shape, q25_shape, closest_median_shape = get_mean_shape_for_class(
                class_name=row['Label'],
                matrix_means=matrix_means,
                labels=labels_fullname,
                shapes=shapes
            )
            
            if closest_median_shape is None:
                continue

            # Scale the shape proportionally in the x-z plane for both plot and legend
            def scale_shape_proportionally(shape, scale_factor_x, scale_factor_z):
                shape = shape - shape.mean(axis=0)  # Center the shape
                shape[:, 0] *= scale_factor_x  # Scale x-dimension
                shape[:, 1] *= scale_factor_z  # Scale z-dimension
                return shape

            scale_factor_x = 0.25
            scale_factor_z = scale_factor_x / aspect_ratio_xz

            # Apply scaling to the median shape and align in x-z plane
            scaled_shape = scale_shape_proportionally(closest_median_shape, scale_factor_x, scale_factor_z)

            # Translate the shape to the plot coordinates in 3D
            x_offset, y_offset, z_offset = row['DDR_mean'], row['Weighted_Avg_Density_mean'], row['Z_Mean']
            flat_shape_xz = np.hstack((scaled_shape[:, 0:1], np.full((scaled_shape.shape[0], 1), 0), scaled_shape[:, 1:2]))

            translated_shape = flat_shape_xz + [x_offset, y_offset, z_offset]

            # Create a Poly3DCollection for the shape and store it for animation updates
            polygon = Poly3DCollection([list(zip(translated_shape[:, 0], translated_shape[:, 1], translated_shape[:, 2]))],
                                       color=color, alpha=0.6, edgecolor='k')
            polygons.append((polygon, flat_shape_xz, x_offset, y_offset, z_offset))
            ax.add_collection3d(polygon)

            # Add shape for legend with the current category and color
            # Prepare label with n for legend
            n = int(row.get('n', 0))  # Assuming 'n' is included in class_stats
            label_with_n = f"{row['Label']} (n={n})"
            legend_elements.append((Polygon(closest_median_shape, closed=True), color, label_with_n))


    # Set plot title and labels
    ax.set_title("DDR Mean vs Weighted Avg Density Mean vs ECT Density")
    ax.set_xlabel("DDR Mean")
    ax.set_ylabel("Weighted Avg Density Mean")
    ax.set_zlabel("ECT Density")

    # Prepare legend handlers with proper scaling for visibility
    custom_handlers = {
        patch: ShapeLegendHandler(shape=patch, m1=75, m2=300, color=color, scale_factor_x=scale_factor_x, scale_factor_z=scale_factor_z, aspect_ratio_xz=aspect_ratio_xz)
        for patch, color, _ in legend_elements
    }

    # Create a separate legend figure
    legend_fig, legend_ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    legend_ax.legend(
        handles=[patch for patch, _, _ in legend_elements],
        labels=[label for _, _, label in legend_elements],
        handler_map=custom_handlers,
        title="Taxa",
        loc='center',
        ncol=2,
        frameon=False,
    )
    legend_ax.axis('off')  # Turn off the axis

    legend_path = os.path.join(save_dir, filename.replace(".gif", "_legend.png"))
    legend_fig.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close(legend_fig)

    # Animation function to rotate the plot and counter-rotate shapes to face the viewer
    def rotate(angle):
        ax.view_init(elev=10, azim=angle)
        
        # Update each shape to counter-rotate based on the current view angle
        for polygon, base_shape, x_offset, y_offset, z_offset in polygons:
            # Apply counter-rotation to exactly match the main rotation
            rotation_angle = 0#np.radians(angle)
            rotation_matrix = np.array([[np.cos(rotation_angle), 0, np.sin(rotation_angle)],
                                    [0, 1, 0],
                                    [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]])
        
            # Apply the counter-rotation around y-axis
            rotated_shape = base_shape @ rotation_matrix.T
            translated_shape = rotated_shape + [x_offset, y_offset, z_offset]
            
            # Update the shape's vertices
            polygon.set_verts([list(zip(translated_shape[:, 0], translated_shape[:, 1], translated_shape[:, 2]))])


    # Create the animation with the custom angle sequence
    anim = FuncAnimation(fig, rotate, frames=range(0, 360, 2), interval=20)

    os.makedirs(save_dir, exist_ok=True)
    gif_path = os.path.join(save_dir, filename)
    anim.save(gif_path, writer='pillow', fps=10, dpi=300)

    print(f"Animation saved as {gif_path}")
    print(f"Legend saved as {legend_path}")
    
    plt.close(fig)

def plot_ddr_hsi_ect_3d_to_gif(save_dir, class_stats, metric_stats, matrix_means, labels_fullname, shapes, filename="plot_ddr_hsi_ectMean_3d.gif"):
    """
    Plot DDR_mean on the X-axis, Weighted_Avg_Density_mean on the Y-axis, and mean from metric_stats on the Z-axis,
    with shapes representing each class, oriented in the x-z plane, and counter-rotating to face the viewer.
    """
    common_keys = set(class_stats.keys()).intersection(set(metric_stats.keys()))
    # print(len(common_keys))

    # Convert class_stats and metric_stats dictionaries to DataFrames
    class_df = pd.DataFrame.from_dict(class_stats, orient='index')
    metric_df = pd.DataFrame.from_dict(metric_stats, orient='index')
    
    # Combine the data for plotting
    df = class_df.copy()
    df['Z_Mean'] = metric_df['mean']  # Add mean from metric_stats as Z-axis
    df['Size'] = metric_df['sd'] * 200  # Use sd from metric_stats for point size
    df['Label'] = df.index  # Add the label as a column
    df['Category'] = df['Label'].apply(lambda x: x.split('_')[0])  # Extract the category for color coding

    # Initialize 3D plot
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')

    # Define color palette with exactly 20 colors (tab20 has 20 colors)
    palette = sns.color_palette("tab20", n_colors=20)

    # Set axis limits for consistent aspect ratio scaling
    x_limits = (df['DDR_mean'].min() - 0.5, df['DDR_mean'].max() + 0.5)
    y_limits = (df['HSI_mean'].min() - 20, df['HSI_mean'].max() + 20)
    z_limits = (df['Z_Mean'].min() - 0.01, df['Z_Mean'].max() + 0.01)
    # print(x_limits, y_limits, z_limits)
    # print(df['DDR_mean'].min(), df['DDR_mean'].max())
    # print(df['HSI_mean'].min(), df['HSI_mean'].max())
    # print(df['Z_Mean'].min(), df['Z_Mean'].max())
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_zlim(z_limits)

    # Calculate aspect ratio for consistent scaling in 3D
    x_range = x_limits[1] - x_limits[0]
    z_range = z_limits[1] - z_limits[0]
    aspect_ratio_xz = x_range / z_range

    # Prepare lists to store all shapes for animation and legend elements
    polygons = []
    legend_elements = []

    # Loop through each category and plot with different colors
    # ii=0
    # for i, (category, color) in enumerate(zip(df['Category'], palette[:len(df['Category'])])):
    unique_categories = df['Category'].unique()
    for i, category in enumerate(unique_categories):
        subset = df[df['Category'] == category]
        color = palette[i % 20]  # Cycle through colors every 20 categories

        for _, row in subset.iterrows():
            # ii+=1
            # print(ii)

            mean_shape, q25_shape, closest_median_shape = get_mean_shape_for_class(
                class_name=row['Label'],
                matrix_means=matrix_means,
                labels=labels_fullname,
                shapes=shapes
            )
            
            if closest_median_shape is None:
                print("NOOOO")
                print(category)
                print(row['Label'])
                continue

            # Scale the shape proportionally in the x-z plane for both plot and legend
            def scale_shape_proportionally(shape, scale_factor_x, scale_factor_z):
                shape = shape - shape.mean(axis=0)  # Center the shape
                shape[:, 0] *= scale_factor_x  # Scale x-dimension
                shape[:, 1] *= scale_factor_z  # Scale z-dimension
                return shape

            scale_factor_x = 0.25
            scale_factor_z = scale_factor_x / aspect_ratio_xz

            # Apply scaling to the median shape and align in x-z plane
            scaled_shape = scale_shape_proportionally(closest_median_shape, scale_factor_x, scale_factor_z)

            # Translate the shape to the plot coordinates in 3D
            x_offset, y_offset, z_offset = row['DDR_mean'], row['HSI_mean'], row['Z_Mean']
            flat_shape_xz = np.hstack((scaled_shape[:, 0:1], np.full((scaled_shape.shape[0], 1), 0), scaled_shape[:, 1:2]))

            translated_shape = flat_shape_xz# + [x_offset, y_offset, z_offset]

            # Create a Poly3DCollection for the shape and store it for animation updates
            polygon = Poly3DCollection([list(zip(translated_shape[:, 0], translated_shape[:, 1], translated_shape[:, 2]))],
                                       color=color, alpha=0.6, edgecolor='k')
            polygons.append((polygon, flat_shape_xz, x_offset, y_offset, z_offset))
            ax.add_collection3d(polygon)

            # Add shape for legend with the current category and color
            # Prepare label with n for legend
            n = int(row.get('n', 0))  # Assuming 'n' is included in class_stats
            label_with_n = f"{row['Label']} (n={n})"
            legend_elements.append((Polygon(closest_median_shape, closed=True), color, label_with_n))


    # Set plot title and labels
    ax.set_title("DDR Mean vs HSI Mean vs ECT Density")
    ax.set_xlabel("DDR Mean")
    ax.set_ylabel("HSI Mean")
    ax.set_zlabel("ECT Density")

    # Prepare legend handlers with proper scaling for visibility
    custom_handlers = {
        patch: ShapeLegendHandler(shape=patch, m1=75, m2=300, color=color, scale_factor_x=scale_factor_x, scale_factor_z=scale_factor_z, aspect_ratio_xz=aspect_ratio_xz)
        for patch, color, _ in legend_elements
    }

    # Create a separate legend figure
    legend_fig, legend_ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    legend_ax.legend(
        handles=[patch for patch, _, _ in legend_elements],
        labels=[label for _, _, label in legend_elements],
        handler_map=custom_handlers,
        title="Taxa",
        loc='center',
        ncol=2,
        frameon=False,
    )
    legend_ax.axis('off')  # Turn off the axis

    legend_path = os.path.join(save_dir, filename.replace(".gif", "_legend.png"))
    legend_fig.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close(legend_fig)

    # Animation function to rotate the plot and counter-rotate shapes to face the viewer
    def rotate(angle):
        ax.view_init(elev=10, azim=angle)
        
        # Update each shape to counter-rotate based on the current view angle
        for polygon, base_shape, x_offset, y_offset, z_offset in polygons:
            # Apply counter-rotation to exactly match the main rotation
            rotation_angle = 0#np.radians(angle)
            rotation_matrix = np.array([[np.cos(rotation_angle), 0, np.sin(rotation_angle)],
                                    [0, 1, 0],
                                    [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]])
        
            # Apply the counter-rotation around y-axis
            rotated_shape = base_shape @ rotation_matrix.T
            translated_shape = rotated_shape + [x_offset, y_offset, z_offset]
            
            # Update the shape's vertices
            polygon.set_verts([list(zip(translated_shape[:, 0], translated_shape[:, 1], translated_shape[:, 2]))])


    # Create the animation with the custom angle sequence
    anim = FuncAnimation(fig, rotate, frames=range(0, 360, 2), interval=20)

    os.makedirs(save_dir, exist_ok=True)
    gif_path = os.path.join(save_dir, filename)
    anim.save(gif_path, writer='pillow', fps=10, dpi=300)

    print(f"Animation saved as {gif_path}")
    print(f"Legend saved as {legend_path}")

    plt.close(fig)


def get_mean_shape_for_class(class_name, matrix_means, labels, shapes, IQR_X=1.0):
    """
    Get the shape that corresponds to the mean of the ECT means closest to the overall mean for a given class,
    using IQR filtering to reduce the influence of outliers.

    Parameters:
    - class_name: The name of the class for which the mean shape is calculated.
    - matrix_means: List of mean values for each matrix.
    - labels: List of labels corresponding to each matrix mean.
    - shapes: List of shapes (as arrays) corresponding to each matrix mean.
    - IQR_X: Multiplier for the IQR to filter out outliers. Default is 1.0 (within 1 IQR).

    Returns:
    - closest_mean_shape: The shape with a matrix mean closest to the overall IQR-filtered mean.
    """
    # Filter data for the specified class
    class_data = pd.DataFrame({
        'matrix_mean': matrix_means,
        'label': labels,
        'shape': shapes
    })
    class_data = class_data[class_data['label'] == class_name]
    
    if class_data.empty:
        print(f"No data found for class '{class_name}'")
        return None

    # Calculate IQR range for the class
    q1 = class_data['matrix_mean'].quantile(0.25)
    q3 = class_data['matrix_mean'].quantile(0.75)
    iqr = q3 - q1

    # Filter matrix means within the IQR range
    filtered_data = class_data[(class_data['matrix_mean'] >= q1 - IQR_X * iqr) & 
                               (class_data['matrix_mean'] <= q3 + IQR_X * iqr)]

    if filtered_data.empty:
        print(f"No data within IQR range for class '{class_name}'")
        return None

    # Calculate the mean of the IQR-filtered matrix means
    iqr_mean = filtered_data['matrix_mean'].mean()
    iqr_q1 = filtered_data['matrix_mean'].quantile(0.25)
    iqr_median = filtered_data['matrix_mean'].median()

    # Find the shape closest to the IQR mean
    closest_mean_idx = np.argmin(np.abs(filtered_data['matrix_mean'] - iqr_mean))
    closest_mean_shape = filtered_data.iloc[closest_mean_idx]['shape']

    # Find the shape closest to the IQR mean
    closest_Q25_idx = np.argmin(np.abs(filtered_data['matrix_mean'] - iqr_q1))
    closest_Q25_shape = filtered_data.iloc[closest_Q25_idx]['shape']

    # Find the shape closest to the IQR mean
    closest_median_idx = np.argmin(np.abs(filtered_data['matrix_mean'] - iqr_median))
    closest_median_shape = filtered_data.iloc[closest_median_idx]['shape']
    
    return closest_mean_shape, closest_Q25_shape, closest_median_shape










################################################################################################
################################################################################################
################################################################################################
# Plotting within each species
'''
TODO 
* matrix_means zipped with labels_fullname
* subset by unique labels_fullname to get ect_density
* zip labels_fullname with ect_data, then compute these to get ddr for each matrix
    ddr = calculate_ddr(matrix)

    cdi = calculate_cdi(matrix)
    hsi = calculate_hsi(matrix)
    weighted_avg_density = calculate_weighted_average_density(matrix)
* subset ddr by unique labels_fullname
* need to compute the 20 kmeans clusters on the subset, then plot the leaf shape centroid per cluster, plot the hull bounds

'''
def plot_ddr_ect_mean_scatter_with_sd_SPECIES_KMEANS(save_dir, ect_data, matrix_means, labels_fullname, shapes, filename="plot_ddr_ectMean_SPECIES_Kmeans.png"):
    N_K = 20
    # Scale the shape proportionally in the x-z plane for both plot and legend
    def scale_shape(shape, x_offset, y_offset, aspect_ratio, scale_factor=0.25):
        shape = shape - shape.mean(axis=0)
        shape[:, 0] *= scale_factor
        shape[:, 1] *= scale_factor / aspect_ratio
        return shape + [x_offset, y_offset]
    
     # Generate all combinations of X and Y features, with unique pairs only
    # metrics = ['ddr_mean', 'cdi_mean', 'hsi_mean', 'wad_mean', 'ECT_density']
    metrics = ['ddr_mean', 'hsi_mean', 'wad_mean', 'ECT_density']
    metric_pairs = list(itertools.combinations(metrics, 2))
    
    def one_plot(X, Y, species, species_df, ax):
        kmeans = KMeans(n_clusters=N_K, random_state=0)
        legend_elements = []
        palette = sns.color_palette("tab20", n_colors=20)

        if X == 'ddr_mean':
            fudgeX = 0.5
            t_x = 'DDR Mean'
        elif X == 'cdi_mean':
            fudgeX = 0.5
            t_x = 'CDI Mean'
        elif X == 'hsi_mean':
            fudgeX = 20
            t_x = 'HSI Mean'
        elif X == 'wad_mean':
            fudgeX = 0.01
            t_x = 'WAD Mean'
        elif X == 'ECT_density':
            fudgeX = 0.1
            t_x = 'ECT Density'

        if Y == 'ECT_density':
            fudgeY = 0.1
            t_y = 'ECT Density'
        elif Y == 'wad_mean':
            fudgeY = 0.01
            t_y = 'WAD Mean'
        elif Y == 'hsi_mean':
            fudgeY = 20
            t_y = 'HSI Mean'
        elif Y == 'cdi_mean':
            fudgeY = 0.1
            t_y = 'CDI MEAN'
        elif Y == 'ddr_mean':
            fudgeY = 0.5
            t_y = 'DDR Mean'
        
        shape_scale_factor = 0.25  # default scale factor
        if (X == 'hsi_mean' and Y == 'ECT_density') or (X == 'ECT_density' and Y == 'hsi_mean'):
            shape_scale_factor = 25  # 100 times larger
        elif (X == 'hsi_mean' and Y == 'wad_mean') or (X == 'wad_mean' and Y == 'hsi_mean'):
            shape_scale_factor = 25  # 10 times larger
        elif (X == 'wad_mean' and Y == 'ECT_density') or (X == 'ECT_density' and Y == 'wad_mean'):
            shape_scale_factor = 0.0075  # 10 times smaller

        species_df['Cluster'] = kmeans.fit_predict(species_df[[X, Y]])
        centroids = kmeans.cluster_centers_

        # Calculate x and y limits based on current species data
        x_limits = (species_df[X].min() - fudgeX, species_df[X].max() + fudgeX)
        y_limits = (species_df[Y].min() - fudgeY, species_df[Y].max() + fudgeY)

        # Calculate the aspect ratio of the plot
        x_range = x_limits[1] - x_limits[0]
        y_range = y_limits[1] - y_limits[0]
        aspect_ratio = x_range / y_range

        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)

        
        for cluster in species_df['Cluster'].unique():
            cluster_points = species_df[species_df['Cluster'] == cluster][[X, Y]].values
            # Plot each point in the cluster as a tiny black dot
            ax.plot(cluster_points[:, 0], cluster_points[:, 1], 'k.', markersize=1, alpha=0.2, zorder=1)  # 'k.' for black dots, markersize for tiny size
            
            if len(cluster_points) > 2:  # Ensure there are at least 3 points for a hull
                try:
                    hull = ConvexHull(cluster_points)
                    # Plot the convex hull as a polygon around the points in the cluster
                    ax.fill(cluster_points[hull.vertices, 0], cluster_points[hull.vertices, 1], color=palette[cluster % 20], alpha=0.2, edgecolor='none')
                except Exception as e:
                    print(f"ConvexHull error for species {species}, cluster {cluster}: {e}")

        # Overall bounding polygon for all points
        all_points = species_df[[X, Y]].values
        if len(all_points) > 2:
            try:
                overall_hull = ConvexHull(all_points)
                ax.fill(all_points[overall_hull.vertices, 0], all_points[overall_hull.vertices, 1], color='lightgray', alpha=0.1, edgecolor='none')
            except:
                # Fallback for the overall points if ConvexHull fails
                center = np.mean(all_points, axis=0)
                radius = np.max(distance.cdist([center], all_points))
                circle = mpatches.Circle(center, radius, color='lightgray', alpha=0.1, edgecolor='none')
                ax.add_patch(circle)

        # Plot each centroid's closest leaf shape
        for i, centroid in enumerate(centroids):
            cluster_data = species_df[species_df['Cluster'] == i]
            distances = np.linalg.norm(cluster_data[[X, Y]].values - centroid, axis=1)
            closest_idx_within_cluster = distances.argmin()
            closest_shape = cluster_data.iloc[closest_idx_within_cluster]['shapes']
            
            scaled_shape = scale_shape(closest_shape, centroid[0], centroid[1], aspect_ratio, shape_scale_factor)
            ax.fill(scaled_shape[:, 0], scaled_shape[:, 1], color=palette[i % 20], alpha=0.6, edgecolor='k', label=f"{species}_Cluster_{i}")
            
            label_with_n = f"{species} (Cluster={i})"
            color = palette[i % 20]
            legend_elements.append((Polygon(closest_shape, closed=True), color, label_with_n))
        
        # Prepare legend handlers with proper scaling for visibility
        custom_handlers = {
            patch: ShapeLegendHandler(shape=patch, m1=10, m2=10, color=color, scale_factor_x=1, scale_factor_z=1, aspect_ratio_xz=aspect_ratio)
            for patch, color, _ in legend_elements
        }

        # Create a separate legend figure
        ax.set_title(f"{t_x} vs {t_y} - {species} (n={len(species_shapes)})")
        ax.set_xlabel(t_x)
        ax.set_ylabel(t_y)
        # Set unique file paths for each species
        legend_fig, legend_ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
        legend_ax.legend(
            handles=[patch for patch, _, _ in legend_elements],
            labels=[label for _, _, label in legend_elements],
            handler_map=custom_handlers,
            title=f"Taxa Clusters - {species}",
            loc='center',
            ncol=2,
            frameon=False,
        )
        legend_ax.axis('off')
        legend_path = os.path.join(save_dir, filename.replace(".png", f"{species}_legend_{t_x}_vs_{t_y}.png"))
        # legend_fig.savefig(legend_path, dpi=600, bbox_inches='tight')
        plt.close(legend_fig)



    # Create a DataFrame from the input data
    data_df = pd.DataFrame({
        'label_fullname': labels_fullname,
        'ect_data': ect_data,
        'matrix_means': matrix_means,
        'shapes': shapes,
    })

    # Group by 'label_fullname' to organize all associated data for each label
    grouped_data = data_df.groupby('label_fullname')

    unique_species = pd.unique(labels_fullname)  # Unique species labels

    for species in tqdm(unique_species, desc="Processing Species", total=len(unique_species)):
        # fig, ax = plt.subplots(figsize=(12, 8))
        # palette = sns.color_palette("tab20", n_colors=20)

        # Access ect_data and matrix_means for the current species
        species_data = grouped_data.get_group(species)
        species_matrices = species_data['ect_data'].tolist()
        species_means = species_data['matrix_means'].tolist()
        species_shapes = species_data['shapes'].tolist()

        ddr_values = [calculate_ddr(matrix) for matrix in species_matrices]
        # cdi_values = [calculate_cdi(matrix) for matrix in species_matrices]
        hsi_values = [calculate_hsi(matrix) for matrix in species_matrices]
        wad_values = [calculate_weighted_average_density(matrix) for matrix in species_matrices]

        unique_means = pd.unique(species_means)

        if len(unique_means) >= N_K:
            species_df = pd.DataFrame({
                'ddr_mean': ddr_values,
                # 'cdi_mean': cdi_values,
                'hsi_mean': hsi_values,
                'wad_mean': wad_values,
                'ECT_density': species_means,
                'Label': [species] * len(species_means),
                'shapes': species_shapes
            })
            n_rows, n_cols = len(metric_pairs) // 3 + 1, min(2, len(metric_pairs))  # Adjust grid size if needed
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
            axes = axes.flatten() if len(metric_pairs) > 1 else [axes]  # Flatten for consistent indexing

            for i, (X, Y) in enumerate(metric_pairs):
                one_plot(X, Y, species, species_df, axes[i])
            
            # Plot convex hull for each cluster
            # for cluster in species_df['Cluster'].unique():
            #     cluster_points = species_df[species_df['Cluster'] == cluster][['DDR_mean', 'ECT_density']].values
            #     if len(cluster_points) > 2:  # Ensure there are at least 3 points for a hull
            #         try:
            #             hull = ConvexHull(cluster_points)
            #             # Plot the convex hull as a polygon around the points in the cluster
            #             ax.fill(cluster_points[hull.vertices, 0], cluster_points[hull.vertices, 1], color=palette[cluster % 20], alpha=0.2, edgecolor='k')
            #         except Exception as e:
            #             print(f"ConvexHull error for species {species}, cluster {cluster}: {e}")
                
            
            # Save the species plot
            fig.tight_layout()

            image_path = os.path.join(save_dir, filename.replace("SPECIES", species))
            plt.savefig(image_path, dpi=600, bbox_inches='tight')
            print(f"Plot saved as {image_path}")
            plt.close(fig)




