import os, sys
import numpy as np
from scipy.spatial import procrustes
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import math
import h5py

currentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(currentdir)
from leafmachine2.ect_methods.utils_ect import preprocessing

'''
Based off of Dan Chitwood's work
https://github.com/DanChitwood/coca_leaf_shape/blob/main/coca_leaves.ipynb

'''

class ShapeVariance:
    def __init__(self, aligned_shapes, overall_mean_shape):
        self.aligned_shapes = aligned_shapes
        self.overall_mean_shape = overall_mean_shape
    
    def procrustes_variance(self):
        # Compute the Procrustes distances and variance
        procrustes_distances = np.zeros(len(self.aligned_shapes))
        for i, shape in enumerate(self.aligned_shapes):
            _, _, procrustes_distances[i] = procrustes(self.overall_mean_shape, shape)
        variance = np.var(procrustes_distances)  # Variance of Procrustes distances
        return variance

    def pca_spread(self):
        # Perform PCA on the aligned shapes
        shapes_flattened = self.aligned_shapes.reshape(self.aligned_shapes.shape[0], -1)  # Flatten to 2D for PCA
        pca = PCA(n_components=5)
        pca.fit(shapes_flattened)
        eigenvalues = pca.explained_variance_  # Get the variance (eigenvalues) for each component
        return eigenvalues

    def compute_metrics(self):
        # Compute all metrics and return them in a dictionary
        return {
            'procrustes_variance': self.procrustes_variance(),
            'pca_spread': self.pca_spread(),
        }
    
def pca_align(points_np, component_name, vertical_tolerance=15):
    # Check if there are enough points to perform PCA
    if len(points_np) < 2:
        print(f"Skipping contour {component_name} due to insufficient points: {len(points_np)} points")
        return None, None

    # Prepare the matrix X = [c, r].T for PCA
    X = points_np[:, [0, 1]]  # X coordinates (columns), Y coordinates (rows)

    # Apply PCA to find the principal components
    pca = PCA(n_components=2)
    pca.fit(X)

    # Compute the angle between the first principal component and the x-axis
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    angle_degrees = np.degrees(angle)

    # Ensure angle is between -180 and 180 degrees
    angle_degrees = (angle_degrees + 360) % 360
    if angle_degrees > 180:
        angle_degrees -= 360

    # Check if the alignment is near vertical (90 degrees or -90 degrees)
    if not (abs(angle_degrees - 90) <= vertical_tolerance or abs(angle_degrees + 90) <= vertical_tolerance):
        print(f"Skipping contour {component_name} due to misalignment: {angle_degrees:.2f} degrees")
        return None, angle_degrees

    # Rotation matrix to rotate points
    theta = np.radians(-angle_degrees)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    rotated_points = X @ rotation_matrix.T

    return rotated_points, angle_degrees

class LeafPro:
    def __init__(self, df, outline_path, output_path, num_dirs=50, num_thresh=50, max_pts=100):
        self.df = df
        self.outline_path = outline_path
        self.output_path = output_path
        self.num_dirs = num_dirs
        self.num_thresh = num_thresh
        self.max_pts = max_pts

    def load_contour(self, component_name):
        # Borrowed from LeafECT
        if not component_name.endswith('.txt'):
            component_name += '.txt'
        txt_file_path = os.path.join(self.outline_path, component_name)
        # Handle case where file does not exist
        if not os.path.exists(txt_file_path):
            print(f"File not found: {txt_file_path}")
            return None, None, None
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()
        points = [(float(x), float(y)) for line in lines[11:] for x, y in [line.strip().split(',')]]
        base = [(float(x), float(y)) for x, y in [lines[8].strip().split(',')]]
        tip = [(float(x), float(y)) for x, y in [lines[7].strip().split(',')]]
        return np.array(points, dtype=np.float32), np.array(tip, dtype=np.float32)[0], np.array(base, dtype=np.float32)[0]
    
    def angle_between(self, p1, p2, p3):
        """
        define a function to find the angle between 3 points anti-clockwise in degrees, p2 being the vertex
        inputs: three angle points, as tuples
        output: angle in degrees
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        deg1 = (360 + math.degrees(math.atan2(x1 - x2, y1 - y2))) % 360
        deg2 = (360 + math.degrees(math.atan2(x3 - x2, y3 - y2))) % 360
        return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)

    def rotate_points(self, xvals, yvals, degrees):
        """"
        define a function to rotate 2D x and y coordinate points around the origin
        inputs: x and y vals (can take pandas dataframe columns) and the degrees (positive, anticlockwise) to rotate
        outputs: rotated and y vals
        """
        angle_to_move = 90-degrees
        rads = np.deg2rad(angle_to_move)
        
        new_xvals = xvals*np.cos(rads)-yvals*np.sin(rads)
        new_yvals = xvals*np.sin(rads)+yvals*np.cos(rads)
        
        return new_xvals, new_yvals

    def interpolation(self, x, y, number): 
        """
        define a function to return equally spaced, interpolated points for a given polyline
        inputs: arrays of x and y values for a polyline, number of points to interpolate
        ouputs: interpolated points along the polyline, inclusive of start and end points
        """
        distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
        distance = distance/distance[-1]

        fx, fy = interp1d( distance, x ), interp1d( distance, y )

        alpha = np.linspace(0, 1, number)
        x_regular, y_regular = fx(alpha), fy(alpha)
        
        return x_regular, y_regular

    def euclid_dist(self, x1, y1, x2, y2):
        """
        define a function to return the euclidean distance between two points
        inputs: x and y values of the two points
        output: the eulidean distance
        """
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)

    def poly_area(self, x,y):
        """
        define a function to calculate the area of a polygon using the shoelace algorithm
        inputs: separate numpy arrays of x and y coordinate values
        outputs: the area of the polygon
        """
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    def gpa_mean(self, leaf_arr, landmark_num, dim_num):
        
        """
        define a function that given an array of landmark data returns the Generalized Procrustes Analysis mean
        inputs: a 3 dimensional array of samples by landmarks by coordinate values, number of landmarks, number of dimensions
        output: an array of the Generalized Procrustes Analysis mean shape
        
        """

        ref_ind = 0 # select arbitrary reference index to calculate procrustes distances to
        ref_shape = leaf_arr[ref_ind, :, :] # select the reference shape

        mean_diff = 10**(-30) # set a distance between means to stop the algorithm

        old_mean = ref_shape # for the first comparison between means, set old_mean to an arbitrary reference shape

        d = 1000000 # set d initially arbitraily high

        while d > mean_diff: # set boolean criterion for Procrustes distance between mean to stop calculations

            arr = np.zeros( ((len(leaf_arr)),landmark_num,dim_num) ) # empty 3D array: # samples, landmarks, coord vals

            for i in range(len(leaf_arr)): # for each leaf shape 

                s1, s2, distance = procrustes(old_mean, leaf_arr[i]) # calculate procrustes adjusted shape to ref for current leaf
                arr[i] = s2 # store procrustes adjusted shape to array

            new_mean = np.mean(arr, axis=(0)) # calculate mean of all shapes adjusted to reference

            s1, s2, d = procrustes(old_mean, new_mean) # calculate procrustes distance of new mean to old mean

            old_mean = new_mean # set the old_mean to the new_mea before beginning another iteration

        return new_mean

    def filter_data(self, family=None, genus=None, genus_species=None, fullname=None):
        df_list = []  # List to store multiple DataFrames
    
        if family and isinstance(family, list):
            # Loop through each family and append the corresponding filtered DataFrame
            for fam in family:
                df_list.append(self.df[self.df['family'] == fam])
        elif family:
            # Filter by a single family
            df_list.append(self.df[self.df['family'] == family])

        if genus and isinstance(genus, list):
            # Loop through each genus and append the corresponding filtered DataFrame
            for gen in genus:
                df_list.append(self.df[self.df['genus'] == gen])
        elif genus:
            # Filter by a single genus
            df_list.append(self.df[self.df['genus'] == genus])

        if genus_species and isinstance(genus_species, list):
            # Loop through each genus_species and append the corresponding filtered DataFrame
            for species in genus_species:
                df_list.append(self.df[self.df['genus_species'] == species])
        elif genus_species:
            # Filter by a single genus_species
            df_list.append(self.df[self.df['genus_species'] == genus_species])

        if fullname and isinstance(fullname, list):
            # Loop through each fullname and append the corresponding filtered DataFrame
            for fname in fullname:
                df_list.append(self.df[self.df['fullname'] == fname])
        elif fullname:
            # Filter by a single fullname
            df_list.append(self.df[self.df['fullname'] == fullname])

        # If none of the filters are lists, return a single DataFrame
        if not df_list:
            df_list.append(self.df.copy())  # If no filters, return the full DataFrame

        return df_list
    
    def generate_output_name(self, default, family=None, genus=None, genus_species=None, fullname=None):
        output_name = None
        if genus_species:
            output_name = genus_species
        elif fullname:
            output_name = fullname
        elif genus:
            output_name = genus
        elif family:
            output_name = family
        else:
            output_name = default
        return output_name

    def plot_random_leaves_by_type(self, included_taxa, family=None, genus=None, genus_species=None, fullname=None, num_samples=18, max_subplots_per_fig=126):
        """
        Plot random leaves from each type in the included_taxa list.

        Parameters:
        - included_taxa: list of taxa types to plot.
        - num_samples: number of random samples per type.
        - max_subplots_per_fig: maximum number of subplots allowed per figure (default is 126).
        """
        total_plots = len(included_taxa) * num_samples
        num_figures = math.ceil(total_plots / max_subplots_per_fig)  # Calculate how many figures we need
        plot_count = 0  # Track total plots
        high_res_pts = 10000  # High resolution for initial interpolation
        res = 50  # Final number of points for each side

        figure_counter = 1  # To track figure number
        counter = 1  # Subplot counter for each figure

        for fig_num in range(num_figures):
            plt.figure(figsize=(20, 40))  # Create a new figure

            # Iterate over taxa to plot random samples
            for current_taxa in included_taxa:
                # Filter data based on family, genus, genus_species, or fullname
                df_subsets = self.filter_data(family=family, genus=genus, genus_species=[current_taxa], fullname=fullname)

                # Ensure the DataFrame list is not empty
                if not df_subsets:
                    print(f"No data available for type: {current_taxa}")
                    continue

                # Process each filtered DataFrame
                for subset in df_subsets:
                    if len(subset) == 0:
                        continue  # Skip empty subsets

                    # Randomly select `num_samples` rows
                    sampled_indices = subset.sample(min(len(subset), num_samples), random_state=42).index

                    # Plot each sampled contour
                    for idx in sampled_indices:
                        if plot_count >= total_plots:
                            break  # Exit if we have plotted all the necessary samples

                        # Load contour, tip, and base points
                        component_name = subset.loc[idx, 'component_name']
                        contour, tip, base = self.load_contour(component_name)

                        if contour is not None:
                            # Interpolation to high resolution
                            x, y = contour[:, 0], contour[:, 1]
                            high_res_x, high_res_y = self.interpolation(x, y, high_res_pts)

                            # Create the high-resolution contour array
                            lf_contour = np.column_stack((high_res_x, high_res_y))

                            # Interpolate the left and right sides
                            left_inter_x, left_inter_y = self.interpolation(lf_contour[0:res, 0], lf_contour[0:res, 1], res)
                            right_inter_x, right_inter_y = self.interpolation(lf_contour[res:, 0], lf_contour[res:, 1], res)

                            # Remove duplicate tip point from left side
                            left_inter_x = np.delete(left_inter_x, -1)
                            left_inter_y = np.delete(left_inter_y, -1)

                            # Combine left and right sides into one array
                            lf_pts_left = np.column_stack((left_inter_x, left_inter_y))
                            lf_pts_right = np.column_stack((right_inter_x, right_inter_y))
                            lf_pts = np.row_stack((lf_pts_left, lf_pts_right))

                            # Plot the contour
                            plt.subplot(math.ceil(total_plots / num_samples), num_samples, counter)
                            plt.fill(lf_pts[:, 0], lf_pts[:, 1], c="lightgray", lw=0.1)
                            
                            # Plot the base in red and tip in green
                            plt.scatter(base[0], base[1], s=10, c="r")  # Base point
                            plt.scatter(tip[0], tip[1], s=10, c="g")  # Tip point
                            
                            plt.gca().set_aspect("equal")
                            plt.gca().set_axis_off()
                            counter += 1
                            plot_count += 1

                            # Check if we have reached the maximum number of subplots per figure
                            if plot_count % max_subplots_per_fig == 0:
                                break  # Stop adding subplots to this figure

            plt.tight_layout()
            plt.axis("off")

            # Save the current figure and prepare for the next one
            output_file = os.path.join(self.output_path, f'random_leaves_by_type_fig_{figure_counter}.png')
            plt.savefig(output_file, dpi=600)
            plt.close()
            print(f"Figure {figure_counter} saved to {output_file}")
            figure_counter += 1
            counter = 1  # Reset subplot counter for the next figure


    def procrustes_analysis(self, family=None, genus=None, genus_species=None, fullname=None, res=50):
        landmark_num = (res*2)-1 # select number of landmarks
        dim_num = 2 # select number of coordinate value dimensions

        df_subset = self.filter_data(family, genus, genus_species, fullname)[0]
        leaf_contours = []
        for idx, row in df_subset.iterrows():
            contour, tip, base = self.load_contour(row['component_name'])
            if contour is not None:
                leaf_contours.append(contour)
        if not leaf_contours:
            print("No contours available for Procrustes analysis.")
            return
        
        landmark_coords = np.zeros((len(leaf_contours), landmark_num, 2))

        for i, contour in enumerate(leaf_contours):
            x, y = contour[:, 0], contour[:, 1]
            high_res_x, high_res_y = self.interpolation(x, y, landmark_num)
            landmark_coords[i] = np.column_stack((high_res_x, high_res_y))

        
        mean_shape = self.gpa_mean(landmark_coords, landmark_num, dim_num)

        # Align shapes to the GPA mean
        aligned_shapes = np.zeros_like(landmark_coords)
        for i in range(len(landmark_coords)):
            _, aligned_shapes[i], _ = procrustes(mean_shape, landmark_coords[i])

        # Visualization and saving the plot to output_path
        plt.figure(figsize=(8, 8))
        
        # # Plot each aligned shape in gray
        for shape in aligned_shapes:
            plt.plot(shape[:, 0], shape[:, 1], c="gray", lw=0.5, alpha=0.5)

        # Plot each original (interpolated) shape in gray
        # for shape in landmark_coords:
        #     plt.plot(shape[:, 0], shape[:, 1], c="gray", lw=0.5, alpha=0.5)
        
        # Plot the mean shape in green
        plt.plot(mean_shape[:, 0], mean_shape[:, 1], c="green", lw=2)

        # Add * markers on the mean shape at each landmark
        plt.scatter(mean_shape[:, 0], mean_shape[:, 1], c="red", marker='.', s=10, label='Landmarks')

        plt.gca().set_aspect('equal')
        plt.title("Mean Procrustes Shape with Aligned Contours")
        plt.axis("off")
        # Save plot to the output path
        output_name = self.generate_output_name(default = "procrustes_output", family=family, genus=genus, genus_species=genus_species, fullname=fullname)

        output_file = os.path.join(self.output_path, f'procrustes_{output_name}.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        print(f"Procrustes analysis plot saved to {output_file}")

    def procrustes_analysis_by_quartile_base_align(self, family=None, genus=None, genus_species=None, fullname=None, res=50):
        landmark_num = (res * 2) - 1  # Select number of landmarks
        dim_num = 2  # Select number of coordinate value dimensions

        # Step 1: Filter the data
        df_subset = self.filter_data(family, genus, genus_species, fullname)[0]
        leaf_contours = []
        base_points = []
        
        for idx, row in df_subset.iterrows():
            contour, tip, base = self.load_contour(row['component_name'])
            if contour is not None:
                leaf_contours.append(contour)
                base_points.append(base)

        if not leaf_contours:
            print("No contours available for analysis.")
            return
        
        # Convert base_points to numpy arrays
        base_points = np.array(base_points)

        # Step 2: Create landmark coordinates
        landmark_coords = np.zeros((len(leaf_contours), landmark_num, 2))
        for i, contour in enumerate(leaf_contours):
            x, y = contour[:, 0], contour[:, 1]
            high_res_x, high_res_y = self.interpolation(x, y, landmark_num)
            reordered_contour = np.column_stack((high_res_x, high_res_y))

            ##############################
            ### Reorder Contour to Start at the Base ###
            ##############################
            # Calculate distances of each point to the base point
            base = base_points[i]
            distances_to_base = np.sqrt((reordered_contour[:, 0] - base[0])**2 + (reordered_contour[:, 1] - base[1])**2)
            base_index = np.argmin(distances_to_base)  # Index of the closest point to the base

            # Reorder the contour to start at the base index
            reordered_contour = np.concatenate((reordered_contour[base_index:], reordered_contour[:base_index]), axis=0)

            # Ensure the contour is closed by making the last point the same as the first
            if not np.array_equal(reordered_contour[0], reordered_contour[-1]):
                reordered_contour = np.vstack((reordered_contour, reordered_contour[0]))  # Close the contour

            # Store the reordered contour
            landmark_coords[i] = reordered_contour

        # Step 3: Align shapes by base point without rotation
        aligned_shapes = np.zeros_like(landmark_coords)
        for i, contour in enumerate(landmark_coords):
            # Shift the contour so that the base is at (0, 0)
            shifted_contour = contour - base_points[i]
            aligned_shapes[i] = shifted_contour

        # Step 4: Compute the GPA mean shape
        overall_mean_shape = self.gpa_mean(aligned_shapes, landmark_num, dim_num)

        # Step 5: Align all shapes to the GPA mean using Procrustes
        proc_arr = np.zeros_like(aligned_shapes)
        for i in range(len(aligned_shapes)):
            _, proc_arr[i], _ = procrustes(overall_mean_shape, aligned_shapes[i])

        # Step 6: Compute Procrustes distance for each contour to the overall mean shape
        procrustes_distances = np.zeros(len(proc_arr))
        for i in range(len(proc_arr)):
            _, _, procrustes_distances[i] = procrustes(overall_mean_shape, proc_arr[i])

        # Step 7: Sort shapes based on Procrustes distance and compute quartiles
        sorted_indices = np.argsort(procrustes_distances)  # Sort by Procrustes distance to the mean
        quartiles = np.array_split(sorted_indices, 4)  # Split sorted indices into quartiles

        # Step 8: Assign quartile colors
        quartile_colors = ['blue', 'green', 'orange', 'red']  # Colors for each quartile
        shape_colors = np.empty(len(proc_arr), dtype=object)  # Array to store colors for each shape
        for q, quartile in enumerate(quartiles):
            shape_colors[quartile] = quartile_colors[q]  # Assign the corresponding color to each shape in the quartile

        # Step 9: Compute quartile-specific mean shapes using GPA
        mean_shapes_quartiles = []
        for quartile in quartiles:
            quartile_shapes = proc_arr[quartile]
            quartile_mean_shape = self.gpa_mean(quartile_shapes, landmark_num, dim_num)
            mean_shapes_quartiles.append(quartile_mean_shape)

        # Step 10: Visualization: Plot quartiles, overall mean, and gray individual contours with quartile colors
        plt.figure(figsize=(8, 8))

        # Plot each aligned shape with the color of its quartile
        for i, shape in enumerate(proc_arr):
            plt.plot(shape[:, 0], shape[:, 1], c=shape_colors[i], lw=0.5, alpha=0.3)  # Thinner lines with transparency

        # Plot overall mean shape
        plt.plot(overall_mean_shape[:, 0], overall_mean_shape[:, 1], c="black", lw=2, label='Overall Mean')

        # Plot quartile mean shapes
        for q, shape in enumerate(mean_shapes_quartiles):
            plt.plot(shape[:, 0], shape[:, 1], c=quartile_colors[q], lw=2, label=f'Quartile {q+1}')

        # Add red markers on the overall mean shape at each landmark
        plt.scatter(overall_mean_shape[:, 0], overall_mean_shape[:, 1], c="red", marker='.', s=10, label='Overall Mean Landmarks')

        plt.gca().set_aspect('equal')
        plt.title("Mean Shapes: Quartiles, SD and Individual Shapes by Quartile")
        plt.legend()
        plt.axis("off")

        # Save plot to the output path
        output_name = self.generate_output_name(default="aligned_quartile_output", family=family, genus=genus, genus_species=genus_species, fullname=fullname)
        output_file = os.path.join(self.output_path, f'aligned_quartiles_{output_name}.png')
        plt.savefig(output_file, dpi=300)
        plt.close()

        print(f"Quartile alignment plot with individual contours saved to {output_file}")

    def procrustes_analysis_n_divisions(self, family=None, genus=None, genus_species=None, fullname=None, res=50, n_divisions=4):
        """
        Perform Procrustes analysis, align shapes, and divide them into n divisions based on Procrustes distances.
        """
        USE_LM2_ALIGNMENT = False

        output_name = self.generate_output_name(default=f"aligned_divisions_output_{n_divisions}", family=family, genus=genus, genus_species=genus_species, fullname=fullname)
        landmark_num = (res * 2) - 1  # Select number of landmarks
        dim_num = 2  # Select number of coordinate value dimensions

        # Step 1: Filter the data
        df_subset = self.filter_data(family, genus, genus_species, fullname)[0]
        leaf_contours = []
        base_points = []
        
        for idx, row in df_subset.iterrows():
            contour, tip, base = self.load_contour(row['component_name'])
            if contour is not None:
                leaf_contours.append(contour)
                base_points.append(base)

        if not leaf_contours:
            print("No contours available for analysis.")
            return
        
        # Convert base_points to numpy arrays
        base_points = np.array(base_points)

        # Step 2: Create landmark coordinates
        landmark_coords = np.zeros((len(leaf_contours), landmark_num, 2))  # No need to close contour here
        for i, contour in enumerate(leaf_contours):
            x, y = contour[:, 0], contour[:, 1]
            high_res_x, high_res_y = self.interpolation(x, y, landmark_num)
            reordered_contour = np.column_stack((high_res_x, high_res_y))

            # Reorder the contour to start at the base index
            base = base_points[i]
            distances_to_base = np.sqrt((reordered_contour[:, 0] - base[0])**2 + (reordered_contour[:, 1] - base[1])**2)
            base_index = np.argmin(distances_to_base)
            reordered_contour = np.concatenate((reordered_contour[base_index:], reordered_contour[:base_index]), axis=0)
            landmark_coords[i] = reordered_contour

        # Step 3: Align shapes by base point without rotation
        if USE_LM2_ALIGNMENT:
            aligned_shapes = landmark_coords
        else:
            aligned_shapes = np.zeros_like(landmark_coords)
            for i, contour in enumerate(landmark_coords):
                shifted_contour = contour - base_points[i]
                aligned_shapes[i] = shifted_contour

        # Step 4: Compute the GPA mean shape
        overall_mean_shape = self.gpa_mean(aligned_shapes, landmark_num, dim_num)

        # Step 5: Align all shapes to the GPA mean using Procrustes
        if USE_LM2_ALIGNMENT:
            proc_arr = aligned_shapes
        else:
            proc_arr = np.zeros_like(aligned_shapes)
            for i in range(len(aligned_shapes)):
                _, proc_arr[i], _ = procrustes(overall_mean_shape, aligned_shapes[i])

        # Step 6: Compute Procrustes distance for each contour to the overall mean shape
        procrustes_distances = np.zeros(len(proc_arr))
        for i in range(len(proc_arr)):
            _, _, procrustes_distances[i] = procrustes(overall_mean_shape, proc_arr[i])

        # Step 7: Sort shapes based on Procrustes distance and split into n divisions
        sorted_indices = np.argsort(procrustes_distances)  # Sort by Procrustes distance to the mean
        divisions = np.array_split(sorted_indices, n_divisions)  # Split sorted indices into n divisions

        # Step 8: Assign colors for each division
        division_colors = plt.cm.viridis(np.linspace(0, 1, n_divisions))  # Use a colormap for the divisions
        shape_colors = np.empty(len(proc_arr), dtype=object)
        for d, division in enumerate(divisions):
            for idx in division:
                shape_colors[idx] = division_colors[d]  # Assign the corresponding color to each shape

        # Step 9: Compute division-specific mean shapes using GPA
        mean_shapes_divisions = []
        for division in divisions:
            division_shapes = proc_arr[division]
            division_mean_shape = self.gpa_mean(division_shapes, landmark_num, dim_num)
            mean_shapes_divisions.append(division_mean_shape)

        # **Step 10: Rescale all shapes to match real shapes size (for left plot)**
        if USE_LM2_ALIGNMENT:
            # Get real shape areas
            real_shape_areas = [np.ptp(shape[:, 0]) * np.ptp(shape[:, 1]) for shape in aligned_shapes]
            mean_real_area = np.mean(real_shape_areas)

            # Get areas of overall mean and division mean shapes
            overall_mean_area = np.ptp(overall_mean_shape[:, 0]) * np.ptp(overall_mean_shape[:, 1])
            mean_shapes_areas = [np.ptp(mean_shape[:, 0]) * np.ptp(mean_shape[:, 1]) for mean_shape in mean_shapes_divisions]

            # Compute scale factor to match sizes between real shapes and mean shapes
            scale_factor = np.sqrt(mean_real_area / overall_mean_area)

            # Apply scaling to all shapes and mean shapes
            # proc_arr *= scale_factor
            overall_mean_shape *= scale_factor
            mean_shapes_divisions = [mean_shape * scale_factor for mean_shape in mean_shapes_divisions]

        # Step 11: Visualization: Plot aligned shapes and division mean shapes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))  # Create two side-by-side plots

        # Set the title for the entire figure
        fig.suptitle(f"{output_name} - {len(leaf_contours)} leaves", fontsize=16)

        # Plot Left: Aligned shapes and division mean shapes
        for i, shape in enumerate(proc_arr):
            ax1.plot(shape[:, 0], shape[:, 1], c=shape_colors[i], lw=0.5, alpha=0.3)
        for d, shape in enumerate(mean_shapes_divisions):
            ax1.plot(shape[:, 0], shape[:, 1], c=division_colors[d], lw=2, label=f'Division {d+1}')
        ax1.plot(overall_mean_shape[:, 0], overall_mean_shape[:, 1], c="red", lw=4)
        ax1.scatter(overall_mean_shape[:, 0], overall_mean_shape[:, 1], c="black", marker='.', s=10, zorder=3, label='Pseudolandmarks')

        ax1.set_aspect('equal')
        ax1.set_title(f"Aligned Shapes and Division Mean Shapes")
        ax1.axis("off")

        # Step 12: Plot real contours and adjust centroid alignment and scaling (same as before)
        real_shapes = []
        for d, division_mean_shape in enumerate(mean_shapes_divisions):
            closest_index = None
            min_distance = float('inf')
            for i in range(len(aligned_shapes)):
                _, _, distance = procrustes(division_mean_shape, aligned_shapes[i])
                if distance < min_distance:
                    min_distance = distance
                    closest_index = i

            # Calculate the centroid of the closest real contour
            closest_real_shape = aligned_shapes[closest_index]
            centroid_real = np.mean(closest_real_shape, axis=0)

            # Shift the closest real contour so its centroid is at the origin
            closest_real_shape_shifted = closest_real_shape - centroid_real

            # Plot the shifted real contour in the same color
            real_shapes.append(closest_real_shape_shifted)
            ax2.plot(closest_real_shape_shifted[:, 0], closest_real_shape_shifted[:, 1], c=division_colors[d], lw=1)

        # **Step 13: Find and plot the real contour closest to the overall mean shape in cyan**
        closest_mean_index = None
        min_mean_distance = float('inf')
        for i in range(len(aligned_shapes)):
            _, _, distance = procrustes(overall_mean_shape, aligned_shapes[i])
            if distance < min_mean_distance:
                min_mean_distance = distance
                closest_mean_index = i
        # Calculate the centroid of the closest real contour to the overall mean
        closest_real_shape_to_mean = aligned_shapes[closest_mean_index]
        centroid_real_to_mean = np.mean(closest_real_shape_to_mean, axis=0)

        # Shift the closest real contour so its centroid is at the origin
        closest_real_shape_to_mean_shifted = closest_real_shape_to_mean - centroid_real_to_mean

        # Plot the shifted real contour closest to the overall mean in cyan
        ax2.plot(closest_real_shape_to_mean_shifted[:, 0], closest_real_shape_to_mean_shifted[:, 1], c="cyan", lw=2, label='Closest to Overall Mean', zorder=3)

        # Step 13: Rescale the overall mean shape to match the size of real contours for the right plot
        real_shape_areas = [np.ptp(shape[:, 0]) * np.ptp(shape[:, 1]) for shape in real_shapes]
        mean_real_area = np.mean(real_shape_areas)
        overall_mean_area = np.ptp(overall_mean_shape[:, 0]) * np.ptp(overall_mean_shape[:, 1])
        scale_factor = np.sqrt(mean_real_area / overall_mean_area)

        # Apply scaling to the overall mean shape for the right plot
        rescaled_overall_mean_shape = overall_mean_shape * scale_factor

        # Align the centroid of the rescaled overall mean shape with (0, 0)
        centroid_overall_mean = np.mean(rescaled_overall_mean_shape, axis=0)
        rescaled_overall_mean_shape_shifted = rescaled_overall_mean_shape - centroid_overall_mean

        # Plot the shifted and rescaled overall mean shape in cyan on the right plot
        ax2.plot(rescaled_overall_mean_shape_shifted[:, 0], rescaled_overall_mean_shape_shifted[:, 1], c="red", lw=2, label='Overall Mean', zorder=3)

        ax2.set_aspect('equal')
        ax2.set_title(f"Closest Real Contours to Division Means")
        ax2.axis("off")

        # Move the legend outside the second plot
        handles_ax1, labels_ax1 = ax1.get_legend_handles_labels()
        handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()
        handles = handles_ax1 + handles_ax2
        labels = labels_ax1 + labels_ax2
        ax2.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

        if USE_LM2_ALIGNMENT:
            output_file = os.path.join(self.output_path, f'aligned_divisions_{output_name}_LM2aligned.png')
        else:
            output_file = os.path.join(self.output_path, f'aligned_divisions_{output_name}_PROaligned.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Division alignment plot with individual contours saved to {output_file}")

        # Step 14: Calculate variance metrics using ShapeVariance
        shape_variance_calculator = ShapeVariance(aligned_shapes, overall_mean_shape)
        variance_metrics = shape_variance_calculator.compute_metrics()
        print(f"     Computed variance for {output_file}")

        # Return the closest real contour to the mean and the variance metrics
        return {
            'closest_real_shape': closest_real_shape_to_mean,
            'variance_metrics': variance_metrics
        }

if __name__ == '__main__':
    included_taxa = [
        "Trema_cannabina",
        "Ginkgo_biloba",
        "Rhododendron_maximum",
        "Quercus_alba",
        "Prunus_serotina",
        "Vaccinium_angustifolium",
        "Parthenocissus_quinquefolia",
        "Morus_alba",
        "Alnus_incana",
        "Annona_montana",
        "Quercus_macrocarpa",
        "Gymnocladus_dioicus",
        "Gleditsia_triacanthos",
        "Prunus_virginiana",
        "Ulmus_americana",
        "Vitis_riparia",
        "Byrsonima_intermedia",
        "Lonicera_maackii",
        "Viburnum_acerifolium",
        "Lecythis_pisonis",
        "Alnus_rubra",
        "Rhamnus_cathartica",
        "Acer_saccharum",
        "Acer_rubrum",
        "Notholithocarpus_densiflorus",
        "Quercus_velutina",
        "Couroupita_guianensis",
        "Tetrapterys_mucronata",
        "Celtis_occidentalis",
        "Dianthus_armeria",
        "Robinia_pseudoacacia",
        "Raphanus_raphanistrum",
        "Cephalanthus_occidentalis",
        "Nicotiana_tabacum",
        "Liquidambar_styraciflua",
        "Asimina_triloba",
        "Quercus_gambelii",
        "Lonicera_periclymenum",
        "Vaccinium_oxycoccos",
        "Rosa_multiflora",
        "Rubus_occidentalis",
        "Tiarella_cordifolia",
        "Brassica_rapa",
        "Lindera_benzoin",
        "Viburnum_opulus",
        "Populus_deltoides",
        "Hieracium_umbellatum",
        "Frangula_alnus",
        "Populus_tremuloides",
        "Gaultheria_procumbens",
        "Cannabis_sativa",
    ]
    # included_taxa = [
    #     "Viburnum_opulus",
        # "Populus_deltoides",
    #     "Hieracium_umbellatum",
    #     "Frangula_alnus",
        # "Populus_tremuloides",
        # "Quercus_alba",
    #     "Cannabis_sativa",
    # ]
    
    file_path = "C:/Users/Will/Downloads/GBIF_DetailedSample_50Spp/LM2_2024_09_18__07-52-47/Data/Measurements/LM2_2024_09_18__07-52-47_MEASUREMENTS.csv"
    outline_path = "C:/Users/Will/Downloads/GBIF_DetailedSample_50Spp/LM2_2024_09_18__07-52-47/Keypoints/Simple_Labels"
    output_path = "D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/procrustes"
    output_path_summary = "D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/procrustes"
    hdf5_file = 'D:/Dropbox/LeafMachine2/leafmachine2/ect_methods/experiments/procrustes/shape_variance_data.h5'

    os.makedirs(output_path, exist_ok=True)

    cleaned_df = preprocessing(file_path, outline_path, show_CF_plot=False, show_first_raw_contour=False, show_df_head=False)
    cleaned_df = cleaned_df.compute()

    # Initialize the class with your DataFrame and outline path
    leaf_pro = LeafPro(cleaned_df, outline_path, output_path)
    # Perform Procrustes analysis for a specific genus_species
    # leaf_pro.plot_random_leaves_by_type(included_taxa, genus_species=included_taxa, num_samples=18)
    # leaf_pro.procrustes_analysis(genus_species='Liquidambar_styraciflua', res=100)

    
    # leaf_pro.procrustes_analysis_n_divisions(genus_species='Liquidambar_styraciflua', res=150, n_divisions=12)
    # leaf_pro.procrustes_analysis_n_divisions(genus_species='Quercus_velutina', res=150, n_divisions=12)
    
  
    # for taxa in included_taxa:
    #     # leaf_pro.procrustes_analysis(genus_species=taxa, res=150)
    #     result = leaf_pro.procrustes_analysis_n_divisions(genus_species=taxa, res=150, n_divisions=12)

    # Open the HDF5 file in write mode
    run_analysis = False
    run_variance_plot = True

    if run_analysis:
        with h5py.File(hdf5_file, 'w') as h5f:

            # Create groups for variance metrics and contours
            variance_group = h5f.create_group('variance_metrics')
            contours_group = h5f.create_group('contours')

            # Iterate over the taxa
            for taxa in included_taxa:
                # Run the procrustes analysis and get the results
                result = leaf_pro.procrustes_analysis_n_divisions(genus_species=taxa, res=150, n_divisions=12)

                if result is not None:
                    # Extract closest real shape and variance metrics
                    closest_real_shape = result['closest_real_shape']
                    variance_metrics = result['variance_metrics']

                    # Save variance metrics for the taxa
                    taxa_group = variance_group.create_group(taxa)
                    for metric, value in variance_metrics.items():
                        taxa_group.create_dataset(metric, data=value)

                    # Save the closest real shape (contour)
                    contours_group.create_dataset(taxa, data=closest_real_shape)

        print(f"Variance metrics and contours saved to {hdf5_file}")
    
    if run_variance_plot:
        # Ensure the output path exists
        os.makedirs(output_path_summary, exist_ok=True)

        # Load the saved variance metrics and contours from the HDF5 file
        with h5py.File(hdf5_file, 'r') as h5f:
            variance_group = h5f['variance_metrics']
            contours_group = h5f['contours']

            # Initialize lists to store data for plotting
            taxa_list = []
            procrustes_variance_list = []
            pca_spread_list = []
            contours_list = []

            # Load data for each taxa
            for taxa in variance_group:
                taxa_list.append(taxa)

                # Load variance metrics
                metrics = variance_group[taxa]
                procrustes_variance_list.append(metrics['procrustes_variance'][()])
                pca_spread_list.append(np.sum(metrics['pca_spread'][()]))  # Use sum of PCA spread for plotting

                # Load the corresponding contour
                contours_list.append(contours_group[taxa][()])

        # Sort the taxa and corresponding data by Procrustes Variance (descending order)
        sorted_procrustes_indices = np.argsort(procrustes_variance_list)[::-1]  # Descending order
        taxa_sorted_procrustes = [taxa_list[i] for i in sorted_procrustes_indices]
        procrustes_variance_sorted = [procrustes_variance_list[i] for i in sorted_procrustes_indices]
        contours_sorted_procrustes = [contours_list[i] for i in sorted_procrustes_indices]

        # Sort the taxa and corresponding data by PCA Spread (ascending order)
        # sorted_pca_indices = np.argsort(pca_spread_list)  # Ascending order
        sorted_pca_indices = np.argsort(pca_spread_list)[::-1]  # Descending order
        taxa_sorted_pca = [taxa_list[i] for i in sorted_pca_indices]
        pca_spread_sorted = [pca_spread_list[i] for i in sorted_pca_indices]
        contours_sorted_pca = [contours_list[i] for i in sorted_pca_indices]

        # Create 2-panel plot for Procrustes Variance and PCA Spread
        fig, axs = plt.subplots(2, 1, figsize=(24, 16))

        # Taxa names for x-axis
        x_ticks_procrustes = np.arange(len(taxa_sorted_procrustes))
        x_ticks_pca = np.arange(len(taxa_sorted_pca))

        # Plot Procrustes Variance with Contours Above the Bars (Left)
        axs[0].bar(x_ticks_procrustes, procrustes_variance_sorted, color='skyblue')
        axs[0].set_title('Procrustes Variance')
        axs[0].set_xticks(x_ticks_procrustes)
        axs[0].set_xticklabels(taxa_sorted_procrustes, rotation=90)
        axs[0].set_ylabel('Variance')

        # Plot PCA Spread with Contours Above the Bars (Right)
        axs[1].bar(x_ticks_pca, pca_spread_sorted, color='salmon')
        axs[1].set_title('PCA Spread (Sum of Eigenvalues)')
        axs[1].set_xticks(x_ticks_pca)
        axs[1].set_xticklabels(taxa_sorted_pca, rotation=90)
        axs[1].set_ylabel('Sum of Eigenvalues')

        # Function to center, scale x and y independently, and translate contours to plot above bars
        def plot_scaled_contour(ax, contour, x_pos, y_pos, bar_width, y_scale_factor, color='black'):
            # Center the contour
            contour_centered = contour - np.mean(contour, axis=0)

            # Calculate the peak-to-peak range of the contour in x and y directions
            x_range = np.ptp(contour_centered[:, 0])
            y_range = np.ptp(contour_centered[:, 1])

            # Compute the unique x_scale_factor to fit the contour's width within the available bar width
            x_scale_factor = bar_width / x_range if x_range != 0 else 1  # Avoid division by zero

            # Apply separate scaling for x and y to maintain aspect ratio
            contour_scaled = contour_centered * np.array([x_scale_factor, y_scale_factor])

            # Translate the contour to the desired (x_pos, y_pos) position
            contour_translated = contour_scaled + np.array([x_pos, y_pos])

            # Plot the contour
            ax.plot(contour_translated[:, 0], contour_translated[:, 1], color=color, lw=1)

        # Dynamic scaling based on variance
        max_y_procrustes = max(procrustes_variance_sorted)  # Get the max height of bars for proper contour placement
        scale_factor_y_procrustes = max_y_procrustes * 0.1  # Adjust y-scale factor relative to the highest variance
        bar_width_procrustes = 0.5  # Set a reasonable bar width to avoid squashing

        # Plot contours above the bars in the Procrustes Variance plot
        for i, contour in enumerate(contours_sorted_procrustes):
            y_pos = procrustes_variance_sorted[i] + max_y_procrustes * 0.1  # Offset by 10% above the bar
            plot_scaled_contour(axs[0], contour, x_ticks_procrustes[i], y_pos, bar_width_procrustes, scale_factor_y_procrustes, color='blue')

        # Dynamic scaling based on PCA Spread
        max_y_pca = max(pca_spread_sorted)  # Get the max height of bars for proper contour placement
        scale_factor_y_pca = max_y_pca * 0.1  # Adjust y-scale factor relative to the highest PCA spread
        bar_width_pca = 0.5  # Set a reasonable bar width for PCA bars

        # Plot contours above the bars in the PCA Spread plot
        for i, contour in enumerate(contours_sorted_pca):
            y_pos = pca_spread_sorted[i] + max_y_pca * 0.1  # Offset by 10% above the bar
            plot_scaled_contour(axs[1], contour, x_ticks_pca[i], y_pos, bar_width_pca, scale_factor_y_pca, color='green')

        # Adjust layout for better visibility
        plt.tight_layout()

        # Save the plot to the specified output path
        output_file = os.path.join(output_path_summary, 'variance_summary_with_scaled_contours.png')
        plt.savefig(output_file, dpi=300)

        print(f"Summary plot saved to {output_file}")