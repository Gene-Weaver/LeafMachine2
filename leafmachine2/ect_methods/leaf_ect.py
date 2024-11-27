import os
import numpy as np
import matplotlib.pyplot as plt
from ect import ECT, EmbeddedGraph
from shapely.geometry import LineString
from math import pi, sin, cos 
from sklearn.manifold import MDS # for MDS
import pandas as pd
import seaborn as sns

import phate # for using PHATE # pip install --user phate
import scprep # for using PHATE

# from leafmachine2.ect_methods.utils_ect import preprocessing, ingest_DWC_large_files


class LeafECT:
    def __init__(self, df, outline_path, num_dirs=128, num_thresh=128, max_pts=500):
        """
        Initialize the class with a cleaned DataFrame and outline path for contour data.
        """
        self.df = df
        self.outline_path = outline_path
        self.num_dirs = num_dirs
        self.num_thresh = num_thresh
        self.max_pts = max_pts

        self.DEGREES_TO_RADIANS = pi / 180

    def load_contour(self, component_name):
        """
        Load the contour from a text file for a given component name.
        """
        if not component_name.endswith('.txt'):
            component_name += '.txt'
        
        # Create the full file path
        txt_file_path = os.path.join(self.outline_path, component_name)
            
        if not os.path.exists(txt_file_path):
            print(f"File not found: {txt_file_path}")
            self.points = None
            return None

        # Read the contour points from the text file
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()

        # Starting from line 11, parse the points (x, y)
        points = []
        for line in lines[11:]:  # Skip the first 11 lines (header)
            x_str, y_str = line.strip().split(',')
            x = float(x_str)
            y = float(y_str)
            points.append((x, y))

        points = np.array(points, dtype=np.float32)  # Convert list to numpy array
        total_points = len(points)  # Total number of points

        # If the number of points exceeds max_pts, subsample them
        # if total_points > self.max_pts:
        #     step = total_points // self.max_pts  # Step size for subsampling
        #     points = points[::step]  # Keep every n-th point

        self.points = points
        return self.points

    def plot_contour(self, points, title="Raw Contour"):
        """
        Plot the raw contour points.
        """
        plt.figure(figsize=(6, 6))
        plt.plot(points[:, 0], points[:, 1], 'r-', linewidth=1)
        plt.title(title)
        plt.gca().set_aspect('equal', adjustable='box')  # Ensure the aspect ratio is equal
        plt.show()

    def create_embedded_graph(self, points):
        """
        Convert the contour points to an EmbeddedGraph.
        """
        G = EmbeddedGraph()

        # Add nodes to the graph using the contour points
        for i, (x, y) in enumerate(points):
            G.add_node(i, x, y)

        # Add edges between consecutive points (as in a contour)
        for i in range(len(points) - 1):
            G.add_edge(i, i + 1)
        
        # Connect the last point to the first to close the contour
        G.add_edge(len(points) - 1, 0)
        
        return G

    def compute_ECT(self, G):
        """
        Compute the Euler Characteristic Transform (ECT) for a given embedded graph.
        """
        myect = ECT(num_dirs=self.num_dirs, num_thresh=self.num_thresh)

        # Set bounding radius
        myect.set_bounding_radius(1.0 * G.get_bounding_radius()) #1.2

        # Compute the ECT
        myect.calculateECT(G)

        # Plot the ECT matrix
        myect.plot('ECT')
        
        return myect.get_ECT()

    def process_contour(self, component_name, plot=False):
        """
        Process the contour by loading, plotting, creating an embedded graph, and computing the ECT.
        """
        points = self.load_contour(component_name)

        if points is None:
            return None

        if plot:
            self.plot_contour(points, title=f"Raw Contour for {component_name}")

        G = self.create_embedded_graph(points)

        # Compute ECT for the graph
        ECT_matrix = self.compute_ECT(G)

        return ECT_matrix

    def process_all_contours(self):
        """
        Process all contours in the DataFrame.
        """
        results = {}
        for component_name in self.df['component_name']:
            print(f"Processing {component_name}...")
            ECT_matrix = self.process_contour(component_name, plot=False)
            results[component_name] = ECT_matrix

        return results
    
    def process_one_contour(self, i):
        """
        Process a single contour from the DataFrame based on the index.
        """
        # Check if the index is valid
        if i < 0 or i >= len(self.df):
            print(f"Index {i} is out of bounds.")
            return None
        
        # Get the component name for the given index
        component_name = self.df['component_name'].iloc[i]
        
        # Print the component name being processed
        print(f"Processing {component_name}...")

        # Process the contour and compute the ECT matrix
        ECT_matrix = self.process_contour(component_name, plot=False)
        
        # Return the result as a dictionary with the component name as the key
        return {component_name: ECT_matrix}
    
    def process_one_direction(self, component_name, direction):
        """
        Process the contour and compute the Euler Characteristic Curve (ECC)
        from the specified direction, and color-code the contour based on that direction.
        """
        # Load the contour points for the given component name
        points = self.load_contour(component_name)
        
        if points is None:
            return None
        
        # Create the embedded graph from the contour points
        G = self.create_embedded_graph(points)

        # Initialize the ECT with default parameters (we only need the ECC here)
        myect = ECT(num_dirs=self.num_dirs, num_thresh=self.num_thresh)
        
        # Set the bounding radius (optional but recommended for ECC computation)
        myect.set_bounding_radius(1.0 * G.get_bounding_radius()) #1.2

        # Compute the ECC in the given direction (e.g., -pi/2 for top-down)
        ecc_curve = myect.calculateECC(G, direction, bound_radius=True)
        
        # Set up the plot with two subplots: ECC on the left, contour on the right
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Manually plot the ECC curve on the left subplot
        axes[0].plot(myect.threshes, ecc_curve, marker='o', linestyle='-')
        axes[0].set_title(f'ECC Curve (Direction: {direction} radians)')
        axes[0].set_xlabel('Thresholds')
        axes[0].set_ylabel('Euler Characteristic')
        axes[0].grid(True)

        # Plot the contour on the right subplot, color-coded by direction
        function_values = G.g_omega(direction)
        
        # Extract the coordinates and corresponding function values for the nodes
        x_coords = [G.coordinates[node][0] for node in G.nodes]
        y_coords = [G.coordinates[node][1] for node in G.nodes]
        colors = [function_values[node] for node in G.nodes]
        
        scatter = axes[1].scatter(x_coords, y_coords, c=colors, cmap='viridis', s=50)
        axes[1].plot(points[:, 0], points[:, 1], 'r-', linewidth=1)
        axes[1].set_title('Contour Color-Coded by Direction')
        axes[1].set_aspect('equal', adjustable='box')

        # Add a color bar to indicate the function value scaling
        fig.colorbar(scatter, ax=axes[1], label='Function Value (Dot Product)')

        # Show the plots
        plt.tight_layout()
        plt.show()
        
        # Return the computed ECC values for further analysis if needed
        return ecc_curve
    
    def compute_ect_for_contour(self, component_name, is_DP='DP'):
        if is_DP == 'DP':
            self.points = self.simplify_contour(component_name, tolerances=[0.001], cutoff=500, plot=False, report_stats=False)
        else:
            self.load_contour(component_name)

        if self.points is None:
            return None, None
        # Validate the points: Ensure no NaN or inf values, and that the shape is valid
        if not np.isfinite(self.points).all():
            print(f"Warning: Non-finite points found in contour for component: {component_name}")
            return None, None
        # Optionally, check if the contour forms a valid polygon (at least 3 points)
        if len(self.points) < 3:
            print(f"Warning: Insufficient points in contour for component: {component_name}")
            return None, None
        # Step 3: Create the embedded graph and compute the ECT
        try:
            # Create the graph from the contour
            G = EmbeddedGraph()
            G.add_cycle(self.points)  # Create a cycle from the points (i.e., the contour)

            # Normalize the shape using PCA coordinates
            G.set_PCA_coordinates(center_type='min_max', scale_radius=1)

            # Initialize the ECT object and set bounding radius
            leafect = ECT(num_dirs=self.num_dirs, num_thresh=self.num_thresh)
            leafect.set_bounding_radius(1)  # Set the bounding radius to 1

            # Step 4: Compute the ECT
            leafect.calculateECT(G)

            M = leafect.get_ECT()

            # Return both the ECT matrix and the contour points
            return M, self.points

        except Exception as e:
            print(f"Error processing contour {component_name}: {str(e)}")
            return None, None
    

    def simplify_contour(self, component_name, tolerances=[0.0025], cutoff=500, plot=False, report_stats=False):
        """
        Simplify the contour of a given component using the Douglas-Peucker algorithm, 
        with the option to plot the simplified contours for multiple tolerance levels and report statistics.
        If `component_name` is a list, the function enters "demo_mode" and plots contours for multiple components
        in a stacked figure. In demo_mode, the function will return None.
        
        Args:
            component_name (str or list): Name of the component to simplify, or a list of component names for demo_mode.
            tolerances (list or float): A list of tolerance values for the Douglas-Peucker algorithm.
                                        If a single float is provided, it will be converted to a list.
            cutoff (int): If the number of points in the original contour is below this value, the contour is not simplified.
            plot (bool): Whether to plot the original and simplified contours for each tolerance.
            report_stats (bool): Whether to report statistics on the number of points before and after simplification.
        
        Returns:
            np.ndarray: Simplified contour points for the first tolerance in the list, or None in demo_mode.
        """
        # Convert a single tolerance to a list if necessary
        if isinstance(tolerances, float) or isinstance(tolerances, int):
            tolerances = [tolerances]

        # Check if we're in demo_mode (component_name is a list)
        if isinstance(component_name, list):
            print("Entering demo_mode for multiple component names.")
            
            # Initialize a stacked figure for demo mode
            num_components = len(component_name)
            fig_width = 5 + len(tolerances) * 5
            fig_height = 6 * num_components  # Stack vertically, 6 height per component
            fig, axes = plt.subplots(num_components, len(tolerances) + 1, figsize=(fig_width, fig_height))
            
            # Handle single-row case where only one component is passed
            if num_components == 1:
                axes = [axes]

            # Loop over each component in the list and plot the original and simplified contours
            for comp_idx, comp_name in enumerate(component_name):
                original_points = self.load_contour(comp_name)
                
                if original_points is None:
                    print(f"Contour for {comp_name} could not be loaded.")
                    continue
                
                original_point_count = len(original_points)

                try:
                    short_name = comp_name.split('_')[2:5]
                    short_name = '_'.join(short_name)
                except:
                    short_name = comp_name
                    
                # Plot the original contour in the first column of the row
                axes[comp_idx][0].plot(original_points[:, 0], original_points[:, 1], 'r-', linewidth=1)
                axes[comp_idx][0].set_title(f"Original {short_name}")
                axes[comp_idx][0].set_aspect('equal', adjustable='box')

                if original_point_count < cutoff:
                    if report_stats:
                        print(f"Component {short_name}: {original_point_count} points (below cutoff {cutoff}). Not simplifying.")
                else:
                    # Apply the Douglas-Peucker algorithm for each tolerance
                    for i, tolerance in enumerate(tolerances):
                        line = LineString(original_points)
                        simplified_line = line.simplify(tolerance, preserve_topology=True)
                        simplified_points = np.array(simplified_line.coords)

                        if report_stats:
                            simplified_point_count = len(simplified_points)
                            print(f"{short_name} - Tolerance {tolerance}: {simplified_point_count} points "
                                  f"(reduced by {original_point_count - simplified_point_count})")

                        # Plot the simplified contour in the next columns
                        axes[comp_idx][i + 1].plot(simplified_points[:, 0], simplified_points[:, 1], 'g-', linewidth=1)
                        axes[comp_idx][i + 1].set_title(f"Simplified {short_name} (tol={tolerance})")
                        axes[comp_idx][i + 1].set_aspect('equal', adjustable='box')

            plt.tight_layout()
            plt.show()

            return None  # Return None in demo_mode

        # If not in demo_mode, proceed with single component simplification
        original_points = self.load_contour(component_name)

        if original_points is None:
            return None

        first_simplified_points = None
        original_point_count = len(original_points)

        if report_stats:
            print(f"Original contour for {component_name}: {original_point_count} points")

        if original_point_count < cutoff:
            if report_stats:
                print(f"Original contour has less than cutoff={cutoff} points. Using original {original_point_count} contour points.")
            if plot:
                self.plot_contour(original_points, title=f"Original Contour for {component_name}")

            self.max_pts = original_point_count
            return original_points

        else:
            if plot:
                # Create a figure with subplots for the single component
                num_plots = len(tolerances) + 1
                fig_width = 5 + len(tolerances) * 5
                fig, axes = plt.subplots(1, num_plots, figsize=(fig_width, 6))

                # Plot the original contour in the first subplot
                axes[0].plot(original_points[:, 0], original_points[:, 1], 'r-', linewidth=1)
                axes[0].set_title(f"Original Contour")
                axes[0].set_aspect('equal', adjustable='box')

            # Apply the Douglas-Peucker algorithm for each tolerance
            for i, tolerance in enumerate(tolerances):
                line = LineString(original_points)
                simplified_line = line.simplify(tolerance, preserve_topology=True)
                simplified_points = np.array(simplified_line.coords)

                if i == 0:
                    first_simplified_points = simplified_points

                simplified_point_count = len(simplified_points)
                if report_stats:
                    print(f"Tolerance {tolerance}: {simplified_point_count} points "
                          f"(reduced by {original_point_count - simplified_point_count})")
    
                if plot:
                    # Plot the simplified contour
                    axes[i + 1].plot(simplified_points[:, 0], simplified_points[:, 1], 'g-', linewidth=1)
                    axes[i + 1].set_title(f"Simplified (tol={tolerance})")
                    axes[i + 1].set_aspect('equal', adjustable='box')
            
            if plot:
                plt.tight_layout()
                plt.show()

            self.max_pts = simplified_point_count

            return first_simplified_points