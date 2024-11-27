import os
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend for PDFs and images
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import PathPatch
from matplotlib.path import Path


# Custom handler to replicate shapes in the legend with a square aspect ratio
class HandlerShape(HandlerPatch):
    def __init__(self, scale_factor=1, **kw):
        HandlerPatch.__init__(self, **kw)
        self.scale_factor = scale_factor

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Retrieve the correct shape points from orig_handle
        shape = np.copy(orig_handle.shape_points)  # Access the shape points stored in CustomLegendShape

        # Scale the shape uniformly (preserving aspect ratio)
        shape -= shape.mean(axis=0)  # Center the shape
        shape_range = np.ptp(shape, axis=0)  # Get the range of x and y (max - min)
        max_range = max(shape_range)  # Get the largest dimension
        shape /= max_range  # Normalize by the largest dimension (preserve aspect ratio)
        shape *= self.scale_factor  # Apply the scale factor

        # To ensure a square ratio, use the minimum of the width and height provided by Matplotlib
        size = min(width, height)

        # Translate the shape to fit inside the square legend marker area
        shape[:, 0] = shape[:, 0] * size + xdescent + 0.5 * size
        shape[:, 1] = shape[:, 1] * size + ydescent + 0.5 * size

        # Create a Path object to represent the shape in the legend
        path = Path(shape)

        # Create a patch for the legend
        p = PathPatch(path, lw=1, edgecolor=orig_handle.edgecolor, facecolor=orig_handle.facecolor, transform=trans)
        return [p]

# Custom class to store shape points and colors for the legend
class CustomLegendShape:
    def __init__(self, shape_points, facecolor, edgecolor='none'):
        self.shape_points = shape_points
        self.facecolor = facecolor
        self.edgecolor = edgecolor


def plot_tSNE_by_taxa(ect_data, group_labels, output_pdf_path, shapes):
    if len(group_labels) != len(ect_data):
        raise ValueError("The number of group labels does not match the number of ECT matrices.")

    # Map each unique taxa to a numeric value for coloring
    unique_labels = np.unique(group_labels)
    label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_num[label] for label in group_labels])

    # Perform t-SNE on the ECT matrices
    tsne = TSNE(n_components=2, random_state=42)
    ect_data_flat = np.array(ect_data).reshape(len(ect_data), -1)  # Flatten ECT matrices for t-SNE
    tsne_scores = tsne.fit_transform(ect_data_flat)

    legend_handles = []
    legend_labels = []
    # Create a PDF object to store multiple plots
    with PdfPages(output_pdf_path) as pdf:
        # Map each unique taxa to a numeric value for coloring
        unique_labels = np.unique(group_labels)
        label_to_color = {label: i for i, label in enumerate(unique_labels)}

        shape_scale_factor = 0.5  # Reduce shape size
        scale_val = 6  # Set scale value to translate shapes to t-SNE positions
        cmap = plt.get_cmap('tab20')

        for i, taxa in tqdm(enumerate(unique_labels), total=len(unique_labels), desc="Processing Taxa"):
            species_idx = [idx for idx, label in enumerate(group_labels) if label == taxa]
            taxa_tsne_points = tsne_scores[species_idx]

            # Create a new figure for each taxa
            plt.figure(figsize=(24, 16))
            ax = plt.gca()

            # First, plot all the points in gray for context
            ax.scatter(
                tsne_scores[:, 0], tsne_scores[:, 1],
                s=10,  # Tiny dots
                color='lightgray',  # Gray color for all points
                edgecolor='none',  # No edge color
                alpha=0.2  # Transparency for all points
            )

            # Now, highlight the points of the current taxa
            ax.scatter(
                taxa_tsne_points[:, 0], taxa_tsne_points[:, 1],
                s=10,  # Tiny dots
                color=cmap(label_to_color[taxa] % 20),  # Unique color for this taxa
                edgecolor='none',  # No edge color
                alpha=0.8,  # More opacity for the highlighted taxa
                label=taxa
            )

            # Use kernel density estimation to find the point of highest density
            kde = gaussian_kde(taxa_tsne_points.T)
            density_values = kde(taxa_tsne_points.T)  # Evaluate density at each point
            
            # Define a threshold to exclude the densest points (e.g., top 20% most dense)
            density_threshold = np.percentile(density_values, 80)
            dense_cluster_indices = np.where(density_values >= density_threshold)[0]
            
            # Exclude the points that belong to the densest cluster for k-means clustering
            remaining_indices = np.where(density_values < density_threshold)[0]
            remaining_points = taxa_tsne_points[remaining_indices]

            # Find the shape corresponding to the densest point (this will be the center of the KDE-defined dense cluster)
            densest_point_idx = np.argmax(density_values)
            densest_point = taxa_tsne_points[densest_point_idx]
            closest_idx_within_taxa = species_idx[densest_point_idx]  # Correctly use species_idx

            # Get the contour points for the shape at the densest point
            points = shapes[closest_idx_within_taxa]
            points = points - np.mean(points, axis=0)  # Center the shape
            points = scale_val * points / max(np.linalg.norm(points, axis=1))  # Scale the shape
            points *= shape_scale_factor
            trans_dense = points + densest_point  # Translate the shape to the densest point

            # Store the shape in CustomLegendShape
            legend_shape = CustomLegendShape(trans_dense, facecolor=cmap(label_to_color[taxa] % 20), edgecolor='none')
            legend_handles.append(legend_shape)

            # Add the genus label for the legend
            legend_labels.append(taxa)

            # Find the grid intersection points for the current taxa points
            x_min, x_max = np.min(taxa_tsne_points[:, 0]), np.max(taxa_tsne_points[:, 0])
            y_min, y_max = np.min(taxa_tsne_points[:, 1]), np.max(taxa_tsne_points[:, 1])
            x_grid = np.linspace(x_min, x_max, 11)  # Divide into 10 equal intervals
            y_grid = np.linspace(y_min, y_max, 11)

            already_have = []
            # For each grid intersection, find the closest point **within the current taxa points** and plot its shape
            for x in x_grid:
                for y in y_grid:
                    # Find the closest point to the grid intersection (x, y) within the current taxa points
                    distances = np.linalg.norm(taxa_tsne_points - np.array([x, y]), axis=1)
                    closest_idx_within_taxa = np.argmin(distances)

                    # Get the contour points for the closest shape within the taxa
                    points = shapes[species_idx[closest_idx_within_taxa]]  # Use species_idx to map back to the original shape
                    points = points - np.mean(points, axis=0)  # Center the shape
                    points = scale_val * points / max(np.linalg.norm(points, axis=1))  # Scale the shape
                    points *= shape_scale_factor
                    trans_sh = points + taxa_tsne_points[closest_idx_within_taxa]  # Translate to the closest point within the taxa

                    if closest_idx_within_taxa not in already_have:
                        already_have.append(closest_idx_within_taxa)
                        # Plot the shape at the grid intersection within the taxa
                        plt.fill(trans_sh[:, 0], trans_sh[:, 1], color=cmap(label_to_color[taxa] % 20), lw=0, alpha=0.5)
            
            
            # Use k-means clustering to find 2 additional clusters from the remaining points
            if len(remaining_points) > 2:  # Ensure there are enough points for k-means
                kmeans = KMeans(n_clusters=2, random_state=42)
                kmeans.fit(remaining_points)
                cluster_centers = kmeans.cluster_centers_

                # # For each cluster, find the closest point to the cluster center
                # for cluster_center in cluster_centers:
                #     # Compute distances of all points to the cluster center
                #     distances_to_center = np.linalg.norm(remaining_points - cluster_center, axis=1)
                #     closest_point_idx = np.argmin(distances_to_center)  # Index of the point closest to the cluster center
                    
                #     # Correctly retrieve the index in the full dataset (species_idx) using the remaining_indices
                #     closest_idx_within_taxa = species_idx[remaining_indices[closest_point_idx]]

                #     # Get the contour points for this shape
                #     points = shapes[closest_idx_within_taxa]
                #     points = points - np.mean(points, axis=0)  # Center the shape
                #     points = scale_val * points / max(np.linalg.norm(points, axis=1))  # Scale the shape
                #     points *= shape_scale_factor
                #     trans_sh = points + taxa_tsne_points[closest_point_idx]  # Translate the shape to the closest point

                #     # Draw a line connecting the densest shape to the k-means shape
                #     plt.plot(
                #         [densest_point[0], taxa_tsne_points[closest_point_idx][0]],  # x-coordinates
                #         [densest_point[1], taxa_tsne_points[closest_point_idx][1]],  # y-coordinates
                #         color=cmap(label_to_color[taxa] % 20), lw=1, linestyle='--', alpha=0.4
                #     )
                    
                #     # Plot the contour shape
                #     plt.fill(trans_sh[:, 0], trans_sh[:, 1], color=cmap(label_to_color[taxa] % 20), lw=0, alpha=1)
                
                # Plot the contour shape at the densest point
                plt.fill(trans_dense[:, 0], trans_dense[:, 1], color=cmap(label_to_color[taxa] % 20), lw=1, edgecolor='black', alpha=1)
            
            # Set axis labels, title, and aspect ratio
            plt.gca().set_aspect("equal", adjustable='box')
            plt.title(f"t-SNE of Leaf Contours - {taxa} Highlighted")
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")

            # Add the legend with custom shapes (using the stored shape points)
            ax.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, title="Taxa", 
                    handler_map={CustomLegendShape: HandlerShape()})

            # Save the current figure to the PDF
            pdf.savefig()  # Save the current page
            plt.close()  # Close the figure to free up memory

    print(f"All t-SNE plots saved to {output_pdf_path}")

def plot_tSNE(ect_data, group_labels, output_path, shapes):
    if len(group_labels) != len(ect_data):
        raise ValueError("The number of group labels does not match the number of ECT matrices.")

    # Map each unique taxa to a numeric value for coloring
    unique_labels = np.unique(group_labels)
    label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_num[label] for label in group_labels])

    # Perform t-SNE on the ECT matrices
    tsne = TSNE(n_components=2, random_state=42)
    ect_data_flat = np.array(ect_data).reshape(len(ect_data), -1)  # Flatten ECT matrices for t-SNE
    tsne_scores = tsne.fit_transform(ect_data_flat)

    # Map each unique taxa to a numeric value for coloring
    unique_labels = np.unique(group_labels)
    label_to_color = {label: i for i, label in enumerate(unique_labels)}

    # Plot the contour shapes on the t-SNE map
    shape_scale_factor = 0.5  # Reduce shape size
    scale_val = 6  # Set scale value to translate shapes to t-SNE positions
    cmap = plt.get_cmap('tab20')

    plt.figure(figsize=(24, 16))
    ax = plt.gca()

    # Plot each t-SNE point as a tiny dot with transparency
    for i, taxa in tqdm(enumerate(unique_labels), total=len(unique_labels), desc="Processing Taxa (Scatter)"):
        species_idx = [idx for idx, label in enumerate(group_labels) if label == taxa]
        taxa_tsne_points = tsne_scores[species_idx]
        
        # Scatter plot for points
        ax.scatter(
            taxa_tsne_points[:, 0], taxa_tsne_points[:, 1], 
            s=10,  # Tiny dots
            color=cmap(label_to_color[taxa] % 20), 
            edgecolor='none',  # No edge color
            alpha=0.2,  # Transparency
            label=taxa
        )
    
    legend_handles = []
    legend_labels = []
    # Plot each t-SNE point as a tiny dot with transparency
    for i, taxa in tqdm(enumerate(unique_labels), total=len(unique_labels), desc="Processing Taxa (kernel density)"):
        species_idx = [idx for idx, label in enumerate(group_labels) if label == taxa]
        taxa_tsne_points = tsne_scores[species_idx]
        
        # Use kernel density estimation to find the point of highest density
        kde = gaussian_kde(taxa_tsne_points.T)
        density_values = kde(taxa_tsne_points.T)  # Evaluate density at each point
        
        # Define a threshold to exclude the densest points (e.g., top 20% most dense)
        density_threshold = np.percentile(density_values, 80)
        dense_cluster_indices = np.where(density_values >= density_threshold)[0]
        
        # Exclude the points that belong to the densest cluster for k-means clustering
        remaining_indices = np.where(density_values < density_threshold)[0]
        remaining_points = taxa_tsne_points[remaining_indices]

        # Find the shape corresponding to the densest point (this will be the center of the KDE-defined dense cluster)
        densest_point_idx = np.argmax(density_values)
        densest_point = taxa_tsne_points[densest_point_idx]
        closest_idx_within_taxa = species_idx[densest_point_idx]  # Correctly use species_idx

        # Get the contour points for the shape at the densest point
        points = shapes[closest_idx_within_taxa]
        points = points - np.mean(points, axis=0)  # Center the shape
        points = scale_val * points / max(np.linalg.norm(points, axis=1))  # Scale the shape
        points *= shape_scale_factor
        trans_dense = points + densest_point  # Translate the shape to the densest point

        # Store the shape in CustomLegendShape
        legend_shape = CustomLegendShape(trans_dense, facecolor=cmap(label_to_color[taxa] % 20), edgecolor='none')
        legend_handles.append(legend_shape)

        # Add the genus label for the legend
        legend_labels.append(taxa)

        # Use k-means clustering to find 2 additional clusters from the remaining points
        if len(remaining_points) > 2:  # Ensure there are enough points for k-means
            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans.fit(remaining_points)
            cluster_centers = kmeans.cluster_centers_

            # For each cluster, find the closest point to the cluster center
            for cluster_center in cluster_centers:
                # Compute distances of all points to the cluster center
                distances_to_center = np.linalg.norm(remaining_points - cluster_center, axis=1)
                closest_point_idx = np.argmin(distances_to_center)  # Index of the point closest to the cluster center
                
                # Correctly retrieve the index in the full dataset (species_idx) using the remaining_indices
                closest_idx_within_taxa = species_idx[remaining_indices[closest_point_idx]]

                # Get the contour points for this shape
                points = shapes[closest_idx_within_taxa]
                points = points - np.mean(points, axis=0)  # Center the shape
                points = scale_val * points / max(np.linalg.norm(points, axis=1))  # Scale the shape
                points *= shape_scale_factor
                trans_sh = points + taxa_tsne_points[closest_point_idx]  # Translate the shape to the closest point

                # Draw a line connecting the densest shape to the k-means shape
                plt.plot(
                    [densest_point[0], taxa_tsne_points[closest_point_idx][0]],  # x-coordinates
                    [densest_point[1], taxa_tsne_points[closest_point_idx][1]],  # y-coordinates
                    color=cmap(label_to_color[taxa] % 20), lw=1, linestyle='--', alpha=0.2
                )
                
                # Plot the contour shape
                plt.fill(trans_sh[:, 0], trans_sh[:, 1], color=cmap(label_to_color[taxa] % 20), lw=0, alpha=1)

        # Plot the contour shape at the densest point
        plt.fill(trans_dense[:, 0], trans_dense[:, 1], color=cmap(label_to_color[taxa] % 20), lw=1, edgecolor='black', alpha=1)
    
    plt.gca().set_aspect("equal", adjustable='box')
    plt.title(f"t-SNE of Leaf Contours by Taxa")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    # Add the legend with custom shapes (using the stored shape points)
    ax.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, title="Taxa", 
            handler_map={CustomLegendShape: HandlerShape()})
    
    # Save the plot to the specified path with 600 DPI
    plt.savefig(output_path, format='png', dpi=300)
    plt.close()  # Close the figure to free up memory


def plot_tsne_hull_boundary(ect_data, group_labels, output_txt_path, output_img_path, shapes, component_names):
    if len(group_labels) != len(ect_data):
        raise ValueError("The number of group labels does not match the number of ECT matrices.")

    # Perform t-SNE on the ECT matrices
    tsne = TSNE(n_components=2, random_state=42)
    ect_data_flat = np.array(ect_data).reshape(len(ect_data), -1)  # Flatten ECT matrices for t-SNE
    tsne_scores = tsne.fit_transform(ect_data_flat)

    # Create a convex hull around the t-SNE points
    hull = ConvexHull(tsne_scores)
    hull_indices = hull.vertices  # Indices of points that form the convex hull
    hull_points = tsne_scores[hull_indices]

    # Save the component names at the hull boundary to a .txt file
    boundary_component_names = [component_names[i] for i in hull_indices]
    with open(output_txt_path, 'w') as f:
        for name in boundary_component_names:
            f.write(f"{name}\n")

    # Create the plot
    plt.figure(figsize=(24, 16))
    ax = plt.gca()

    # Plot all points as tiny gray dots
    ax.scatter(tsne_scores[:, 0], tsne_scores[:, 1], s=10, color='lightgray', alpha=0.2, edgecolor='none')

    # Plot sampled points along the convex hull (every 10th point)
    shape_scale_factor = 0.5
    scale_val = 6

    # Map each unique taxa to a numeric value for coloring
    unique_labels = np.unique(group_labels)
    label_to_color = {label: i for i, label in enumerate(unique_labels)}

    cmap = plt.get_cmap('tab20')

    for i, hull_idx in enumerate(hull_indices[::5]):  # Sample every 10th point along the hull
        points = shapes[hull_idx]
        points = points - np.mean(points, axis=0)  # Center the shape
        points = scale_val * points / max(np.linalg.norm(points, axis=1))  # Scale the shape
        points *= shape_scale_factor
        trans_sh = points + tsne_scores[hull_idx]  # Translate to the t-SNE point

        # Plot the shape
        plt.fill(trans_sh[:, 0], trans_sh[:, 1], color=cmap(label_to_color[group_labels[hull_idx]] % 20), lw=0, alpha=1)


    # Find the grid intersection points
    x_min, x_max = np.min(tsne_scores[:, 0]), np.max(tsne_scores[:, 0])
    y_min, y_max = np.min(tsne_scores[:, 1]), np.max(tsne_scores[:, 1])
    x_grid = np.linspace(x_min, x_max, 11)  # Divide into 10 equal intervals
    y_grid = np.linspace(y_min, y_max, 11)

    # For each grid intersection, find the closest point and plot its shape
    for x in x_grid:
        for y in y_grid:
            # Find the closest point to the grid intersection (x, y)
            distances = np.linalg.norm(tsne_scores - np.array([x, y]), axis=1)
            closest_idx = np.argmin(distances)

            # Get the contour points for the closest shape
            points = shapes[closest_idx]
            points = points - np.mean(points, axis=0)  # Center the shape
            points = scale_val * points / max(np.linalg.norm(points, axis=1))  # Scale the shape
            points *= shape_scale_factor
            trans_sh = points + tsne_scores[closest_idx]  # Translate to the closest point

            # Plot the shape at the grid intersection
            plt.fill(trans_sh[:, 0], trans_sh[:, 1], color=cmap(label_to_color[group_labels[closest_idx]] % 20), lw=0, alpha=1)

    # List to store legend patches
    legend_handles = []
    legend_labels = []
    # Plot each t-SNE point as a tiny dot with transparency
    for i, taxa in tqdm(enumerate(unique_labels), total=len(unique_labels), desc="Processing Taxa (kernel density)"):
        species_idx = [idx for idx, label in enumerate(group_labels) if label == taxa]
        taxa_tsne_points = tsne_scores[species_idx]
        
        # Use kernel density estimation to find the point of highest density
        kde = gaussian_kde(taxa_tsne_points.T)
        density_values = kde(taxa_tsne_points.T)  # Evaluate density at each point
        
        # Define a threshold to exclude the densest points (e.g., top 20% most dense)
        density_threshold = np.percentile(density_values, 80)
        dense_cluster_indices = np.where(density_values >= density_threshold)[0]
        
        # Exclude the points that belong to the densest cluster for k-means clustering
        remaining_indices = np.where(density_values < density_threshold)[0]
        remaining_points = taxa_tsne_points[remaining_indices]

        # Find the shape corresponding to the densest point (this will be the center of the KDE-defined dense cluster)
        densest_point_idx = np.argmax(density_values)
        densest_point = taxa_tsne_points[densest_point_idx]
        closest_idx_within_taxa = species_idx[densest_point_idx]  # Correctly use species_idx

        # Get the contour points for the shape at the densest point
        points = shapes[closest_idx_within_taxa]
        points = points - np.mean(points, axis=0)  # Center the shape
        points = scale_val * points / max(np.linalg.norm(points, axis=1))  # Scale the shape
        points *= shape_scale_factor
        trans_dense = points + densest_point  # Translate the shape to the densest point

        # Store the shape in CustomLegendShape
        legend_shape = CustomLegendShape(trans_dense, facecolor=cmap(label_to_color[taxa] % 20), edgecolor='none')
        legend_handles.append(legend_shape)

        # Add the genus label for the legend
        legend_labels.append(taxa)

    # Set plot labels and title
    plt.gca().set_aspect("equal", adjustable='box')
    plt.title("t-SNE with Convex Hull Boundary and Grid Sampled Shapes")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")

    # Add the legend with custom shapes (using the stored shape points)
    ax.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, title="Taxa", 
            handler_map={CustomLegendShape: HandlerShape()})
    # Save the plot to a file
    plt.savefig(output_img_path, format='png', dpi=300)
    plt.close()  # Close the figure to free up memory

    print(f"Convex hull boundary saved to {output_txt_path}")
    print(f"t-SNE plot with hull boundary and grid sampling saved to {output_img_path}")