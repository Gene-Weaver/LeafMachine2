from PIL import Image, ImageOps
import os, math, cv2, csv, gc
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import invert
from scipy.spatial import distance
from collections import deque
from tqdm import tqdm
from scipy.spatial import KDTree

class ConditionalPrinter:
    def __init__(self, verbose=False):
        """Initialize with a verbosity flag to control printing."""
        self.verbose = verbose

    def __call__(self, *args, **kwargs):
        """Mimic the behavior of print, but only print if verbose is True."""
        if self.verbose:
            print(*args, **kwargs)


class SegmentationMaskProcessor:
    # Color-to-tag mapping
    color_to_tag = {
        (46, 255, 0): 1,   # leaf
        (0, 173, 255): 2,  # petiole
        (209, 0, 255): 3,  # hole
    }

    def __init__(self, dir_to_process, dir_overlay, dir_data, save_visualization=False, show_visualization=False, verbose=False):
        self.dir_to_process = dir_to_process
        self.dir_overlay = dir_overlay  
        self.dir_data = dir_data  

        self.save_visualization = save_visualization  
        self.show_visualization = show_visualization  
        self.verbose = verbose  

        os.makedirs(self.dir_overlay, exist_ok=True)
        os.makedirs(self.dir_data, exist_ok=True)

    def save_visualization_panel(self, panel_image, filename, verbose=False):
        cPrint = ConditionalPrinter(verbose=verbose)  # Set to False to suppress printing
        """Save the visualized panel as a .jpg file in the overlay directory."""
        output_path = os.path.join(self.dir_overlay, f"{os.path.splitext(filename)[0]}.jpg")
        cv2.imwrite(output_path, panel_image)
        cPrint(f"Saved visualization as {output_path}")

    def export_csv(self, csv_path, data_rows, verbose=False):
        cPrint = ConditionalPrinter(verbose=verbose)  # Set to False to suppress printing
        """Save the width data into a CSV file."""
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Filename', 'Component_1', 'Component_2', 'Width', 'Location'])
            for row in data_rows:
                writer.writerow(row)
        cPrint(f"Width data saved to {csv_path}")

    def process_image(self, image_path):
        """Process a single image, converting colors to tags and checking for petiole presence."""
        img = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB mode
        img_data = np.array(img)  # Convert image to NumPy array for pixel manipulation

        # Compute a unique integer for each pixel color
        img_data_int = (
            img_data[:, :, 0].astype(np.uint32) * 256 * 256 +
            img_data[:, :, 1].astype(np.uint32) * 256 +
            img_data[:, :, 2].astype(np.uint32)
        )

        # Create mapping from color integers to tags
        color_to_tag_int = {
            color[0] * 256 * 256 + color[1] * 256 + color[2]: tag
            for color, tag in self.color_to_tag.items()
        }

        # Initialize the tag_image with zeros
        tag_image = np.zeros(img_data_int.shape, dtype=np.uint8)

        # Assign tags to pixels based on color mapping
        for color_int, tag in color_to_tag_int.items():
            mask = img_data_int == color_int
            tag_image[mask] = tag

        # Find coordinates of leaf and petiole pixels
        leaf_pixels = [tuple(coord) for coord in np.argwhere(tag_image == 1)]
        petiole_pixels = [tuple(coord) for coord in np.argwhere(tag_image == 2)]

        # Check for petiole presence
        petiole_present = len(petiole_pixels) > 0

        # Return img only if visualization is needed
        if self.save_visualization:
            return img, tag_image, petiole_present, leaf_pixels, petiole_pixels
        else:
            return None, tag_image, petiole_present, leaf_pixels, petiole_pixels


    # def process_image(self, image_path):
    #     """Process a single image, converting colors to tags and checking for petiole presence."""
    #     img = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB mode
    #     img_data = np.array(img)  # Convert image to NumPy array for pixel manipulation

    #     # Create an empty array for storing the processed tags (same dimensions as image, but grayscale)
    #     tag_image = np.zeros((img_data.shape[0], img_data.shape[1]), dtype=np.uint8)

    #     # Lists to store coordinates of leaf and petiole pixels
    #     leaf_pixels = []
    #     petiole_pixels = []

    #     # Boolean flag to check for petiole presence
    #     petiole_present = False

    #     # Iterate through each pixel and assign the correct tag based on color
    #     for i in range(img_data.shape[0]):
    #         for j in range(img_data.shape[1]):
    #             pixel = tuple(img_data[i, j])
                
    #             # Assign tag based on pixel color, default to 0 for background
    #             if pixel in self.color_to_tag:
    #                 tag = self.color_to_tag[pixel]
    #                 tag_image[i, j] = tag
    #                 if tag == 1:  # Leaf tag
    #                     leaf_pixels.append((i, j))
    #                 elif tag == 2:  # Petiole tag
    #                     petiole_pixels.append((i, j))
    #                     petiole_present = True
    #             else:
    #                 tag_image[i, j] = 0  # Background

    #     # Return img only if visualization is needed
    #     if self.save_visualization:
    #         return img, tag_image, petiole_present, leaf_pixels, petiole_pixels
    #     else:
    #         return None, tag_image, petiole_present, leaf_pixels, petiole_pixels

    def find_top_of_petiole(self, leaf_pixels, petiole_pixels):
        """Find the top of the petiole as the petiole pixel closest to the topmost leaf pixel."""
        if not leaf_pixels or not petiole_pixels:
            return None  # Return None if we don't have leaf or petiole pixels

        # Convert lists to numpy arrays
        leaf_pixels = np.array(leaf_pixels)
        petiole_pixels = np.array(petiole_pixels)

        # Find the topmost leaf pixel (the one with the smallest y-coordinate)
        topmost_leaf_pixel = leaf_pixels[np.argmin(leaf_pixels[:, 0])]

        # Calculate the Euclidean distance between the topmost leaf pixel and all petiole pixels
        distances = np.linalg.norm(petiole_pixels - topmost_leaf_pixel, axis=1)

        # Find the index of the closest petiole pixel
        closest_petiole_idx = np.argmin(distances)

        # Return the (x, y) coordinates of the closest petiole pixel
        return tuple(petiole_pixels[closest_petiole_idx])

    
    def skeletonize_petiole(self, petiole_pixels, image_shape):
        """Skeletonize the petiole using skimage's skeletonize function on petiole pixels only."""
        # Create a binary mask for the petiole, where 1 represents petiole pixels
        petiole_mask = np.zeros(image_shape, dtype=np.uint8)
        for px in petiole_pixels:
            petiole_mask[px] = 1

        # Perform skeletonization directly on the petiole mask
        skeleton = skeletonize(petiole_mask)

        return skeleton

    def find_furthest_point(self, skeleton, top_skeleton):
        """Find the furthest point from top_skeleton by tracing the skeleton path."""
        h, w = skeleton.shape
        visited = np.zeros((h, w), dtype=bool)
        queue = deque([(top_skeleton, 0)])  # Queue stores (point, distance)
        visited[top_skeleton] = True
        furthest_point = top_skeleton
        max_distance = 0

        # BFS to explore the skeleton points
        while queue:
            (x, y), dist = queue.popleft()

            # Check all 8-connected neighbors
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and skeleton[nx, ny] and not visited[nx, ny]:
                    visited[nx, ny] = True
                    queue.append(((nx, ny), dist + 1))
                    if dist + 1 > max_distance:
                        max_distance = dist + 1
                        furthest_point = (nx, ny)

        return furthest_point, max_distance

    def trace_perpendicular_line(self, petiole_mask, midpoint, perp_normalized, direction=1, max_range=100):
        """Trace the perpendicular line from the midpoint in one direction (left or right) to find the edge."""
        cPrint = ConditionalPrinter(verbose=False)  # Set to False to suppress printing

        h, w = petiole_mask.shape

        for t in range(max_range):
            x = int(midpoint[0] + t * perp_normalized[0] * direction)
            y = int(midpoint[1] + t * perp_normalized[1] * direction)

            cPrint(f"X = {x}")
            cPrint(f"Y = {y}")
            cPrint(petiole_mask[x, y])

            # Check if the point is within bounds and if the pixel value is 0 (indicating the edge)
            # Need to check to see if the pixel is NOT the petiole. petiole = 2, so 0 and 1 are no good
            if x < 0 or x >= h or y < 0 or y >= w or petiole_mask[x, y] != 2:
                # Stop when we find a boundary (transition from 1 to 0)
                return (x, y)

        return None


    def find_skeleton_midpoint(self, skeleton, top_skeleton, bottom_skeleton):
        """Find the pixel-wise midpoint along the skeleton path from top_skeleton to bottom_skeleton."""
        h, w = skeleton.shape
        visited = np.zeros((h, w), dtype=bool)
        queue = deque([(top_skeleton, 0)])  # Queue stores (point, distance)
        visited[top_skeleton] = True
        path = []  # To store the points along the path

        # Perform BFS to find the path from top_skeleton to bottom_skeleton
        while queue:
            (x, y), dist = queue.popleft()
            path.append((x, y))

            if (x, y) == bottom_skeleton:  # Stop when we reach the bottom
                break

            # Check all 8-connected neighbors
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and skeleton[nx, ny] and not visited[nx, ny]:
                    visited[nx, ny] = True
                    queue.append(((nx, ny), dist + 1))

        if not path:
            return None

        # Return the midpoint on the path
        return path[len(path) // 2]  # Pixel-wise midpoint
    
    def find_skeleton_path(self, skeleton, top_skeleton, bottom_skeleton):
        """Find the path between top_skeleton and bottom_skeleton along the skeleton using BFS."""
        h, w = skeleton.shape
        visited = np.zeros((h, w), dtype=bool)
        queue = deque([(top_skeleton, [])])  # Queue stores (point, path_to_point)
        visited[top_skeleton] = True

        while queue:
            (x, y), path = queue.popleft()
            path = path + [(x, y)]  # Append current point to the path

            if (x, y) == bottom_skeleton:
                return path  # Found the path to bottom_skeleton

            # Check all 8-connected neighbors
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and skeleton[nx, ny] and not visited[nx, ny]:
                    visited[nx, ny] = True
                    queue.append(((nx, ny), path))

        return []  # Return an empty list if no path is found

    
    def measure_widths(self, skeleton, petiole_mask, top_skeleton, bottom_skeleton, max_distance, verbose=False):
        cPrint = ConditionalPrinter(verbose=verbose)  # Set to False to suppress printing

        """Measure the width of the petiole at the midpoint along the skeleton, expanding perpendicularly to the left and right."""
        location = None

        # Find the skeleton path from top_skeleton to bottom_skeleton
        path = self.find_skeleton_path(skeleton, top_skeleton, bottom_skeleton)
        if not path:
            cPrint("No valid path found between top and bottom skeleton.")
            return None

        # Determine the location to measure the width based on max_distance
        if max_distance <= 10:
            # Use the midpoint (default behavior)
            path_idx = len(path) // 2
            perpendicular_point = bottom_skeleton
            location = 'midpoint'
        elif max_distance <= 50:
            # Measure at 20% of the distance from top_skeleton to bottom_skeleton
            path_idx = int(0.2 * len(path))
            perpendicular_point = path[path_idx]
            location = '0.20'
        else:
            # Measure at 10% of the distance from top_skeleton to bottom_skeleton
            path_idx = int(0.1 * len(path))
            perpendicular_point = path[path_idx]
            location = '0.10'

        midpoint = path[path_idx]

        # Calculate the vector between top_skeleton and bottom_skeleton
        dx = perpendicular_point[0] - top_skeleton[0]
        dy = perpendicular_point[1] - top_skeleton[1]

        # Calculate the perpendicular direction (-dy, dx)
        perpendicular = (-dy, dx)

        # Normalize the perpendicular vector
        length = math.hypot(*perpendicular)
        if length == 0:
            cPrint("Perpendicular vector has zero length.")
            return None
        perp_normalized = (perpendicular[0] / length, perpendicular[1] / length)

        # Measure width by tracing along the perpendicular line in both directions from the midpoint
        left_point = self.trace_perpendicular_line(petiole_mask, midpoint, perp_normalized, direction=-1)  # Left direction
        right_point = self.trace_perpendicular_line(petiole_mask, midpoint, perp_normalized, direction=1)  # Right direction

        if left_point and right_point:
            # Calculate the Euclidean distance between the two points (width)
            width = math.hypot(right_point[0] - left_point[0], right_point[1] - left_point[1])
            cPrint(f"Width at midpoint: {width:.2f} pixels.")
            return midpoint, left_point, right_point, width, location
        else:
            cPrint("Could not find both left and right edge points.")
            return None

    def petiole_touches_leaf(self, leaf_pixels, petiole_pixels, fudge_factor=20):
        """Check if any petiole pixel is within `fudge_factor` distance of any leaf pixel."""
        # Build a KDTree for leaf pixels
        leaf_tree = KDTree(leaf_pixels)

        # For each petiole pixel, check if there is any leaf pixel within `fudge_factor`
        for petiole_pixel in petiole_pixels:
            if leaf_tree.query(petiole_pixel, distance_upper_bound=fudge_factor)[0] != float('inf'):
                # Found a leaf pixel within `fudge_factor` distance
                return True

        # No petiole pixel is within `fudge_factor` distance of any leaf pixel
        return False


    def visualize_alignment(self, original_image, petiole_mask, skeleton_petiole, top_pixel, top_skeleton, bottom_skeleton):
        """Create a panel of the original image, petiole, and skeletonized petiole with the top and bottom points."""
        
        # Convert the petiole mask and skeleton to RGB images for display
        petiole_mask_img = Image.fromarray(np.uint8(petiole_mask * 255)).convert('RGB')
        skeleton_petiole_img = Image.fromarray(np.uint8(skeleton_petiole * 255)).convert('RGB')

        # Mark the closest point (top_pixel), top_skeleton, and bottom_skeleton on both the petiole mask and skeletonized image
        top_pixel_x, top_pixel_y = top_pixel
        top_skeleton_x, top_skeleton_y = top_skeleton
        bottom_skeleton_x, bottom_skeleton_y = bottom_skeleton

        # Convert images to OpenCV (BGR) format for drawing
        original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        petiole_cv = cv2.cvtColor(np.array(petiole_mask_img), cv2.COLOR_RGB2BGR)
        skeleton_cv = cv2.cvtColor(np.array(skeleton_petiole_img), cv2.COLOR_RGB2BGR)

        # Mark the top pixel, top skeleton, and bottom skeleton with circles
        cv2.circle(petiole_cv, (top_pixel_y, top_pixel_x), 2, (0, 0, 255), -1)  # Red point on petiole
        cv2.circle(skeleton_cv, (top_skeleton_y, top_skeleton_x), 2, (0, 0, 255), -1)  # Red point on skeleton
        cv2.circle(skeleton_cv, (bottom_skeleton_y, bottom_skeleton_x), 2, (255, 255, 0), -1)  # Blue point on skeleton

        # Create a panel of three images (Original, Petiole, Skeleton)
        panel_image = np.hstack([original_cv, petiole_cv, skeleton_cv])

        # Display the image using OpenCV
        cv2.imshow('Panel Visualization', panel_image)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()  # Close the OpenCV window

    def visualize_widths(self, original_image, petiole_mask, skeleton_petiole, top_pixel, top_skeleton, bottom_skeleton, midpoint, left_point, right_point, do_touch, pt_size=2):
        """Visualize the petiole object, skeleton, and key points with width measurements as part of a four-panel image, optimized for resizing."""
        
        # Convert petiole mask to BGR format for visualization
        petiole_cv = cv2.cvtColor(np.array(Image.fromarray(np.uint8(petiole_mask * 255)).convert('RGB')), cv2.COLOR_RGB2BGR)

        # Mark the points (top, bottom, midpoint, left, right) on the petiole mask
        top_skeleton_x, top_skeleton_y = top_skeleton
        bottom_skeleton_x, bottom_skeleton_y = bottom_skeleton
        midpoint_x, midpoint_y = midpoint
        left_point_x, left_point_y = left_point
        right_point_x, right_point_y = right_point

        # Draw circles on the petiole mask to mark the points
        cv2.circle(petiole_cv, (top_skeleton_y, top_skeleton_x), pt_size, (0, 0, 255), -1)  # Red for top_skeleton
        cv2.circle(petiole_cv, (bottom_skeleton_y, bottom_skeleton_x), pt_size, (255, 255, 0), -1)  # Yellow for bottom_skeleton
        cv2.circle(petiole_cv, (midpoint_y, midpoint_x), pt_size, (0, 255, 0), -1)  # Green for midpoint
        cv2.circle(petiole_cv, (left_point_y, left_point_x), pt_size, (255, 0, 0), -1)  # Blue for left edge
        cv2.circle(petiole_cv, (right_point_y, right_point_x), pt_size, (255, 0, 0), -1)  # Blue for right edge

        # Draw a line between left_point and right_point (the width)
        if do_touch:
            cv2.line(petiole_cv, (left_point_y, left_point_x), (right_point_y, right_point_x), (0, 255, 0), 1)  # Green line for width
        else:
            cv2.line(petiole_cv, (left_point_y, left_point_x), (right_point_y, right_point_x), (0, 0, 255), 1)  # Red line for width when don't touch


        # Crop the petiole region for zoomed-in visualization
        petiole_indices = np.where(petiole_mask == 1)
        min_x, min_y = np.min(petiole_indices[0]), np.min(petiole_indices[1])
        max_x, max_y = np.max(petiole_indices[0]), np.max(petiole_indices[1])

        # Crop the petiole image to the bounding box
        petiole_cropped = petiole_cv[min_x:max_x, min_y:max_y]

        # Get the height of the original image to match the size of the other panels
        original_height = original_image.height

        # Calculate the target width of the cropped petiole image to maintain the aspect ratio
        cropped_height = petiole_cropped.shape[0]
        cropped_width = petiole_cropped.shape[1]
        scaling_factor = original_height / cropped_height
        target_width = int(cropped_width * scaling_factor)

        # Resize the zoomed-in view to the target size using high-quality Lanczos interpolation
        zoomed_in_view = cv2.resize(petiole_cropped, (target_width, original_height), interpolation=cv2.INTER_NEAREST)

        # Convert original image to OpenCV BGR format
        original_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

        # Resize petiole mask and skeleton to match the original image size for display
        petiole_cv_resized = cv2.resize(petiole_cv, (original_cv.shape[1], original_cv.shape[0]), interpolation=cv2.INTER_NEAREST)
        skeleton_cv = cv2.cvtColor(np.array(Image.fromarray(np.uint8(skeleton_petiole * 255)).convert('RGB')), cv2.COLOR_RGB2BGR)
        skeleton_cv_resized = cv2.resize(skeleton_cv, (original_cv.shape[1], original_cv.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Create a panel with four images: original, petiole, skeleton, and zoomed-in view
        panel_image = np.hstack([original_cv, petiole_cv_resized, skeleton_cv_resized, zoomed_in_view])

        if self.show_visualization:
            # Display the panel using OpenCV
            cv2.imshow('Four-Panel Visualization', panel_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            return panel_image
        else:
            return panel_image

    

    def process_directory(self):
        cPrint = ConditionalPrinter(verbose=self.verbose)  # Set to False to suppress printing

        """Process all images in the given directory and save the results immediately to the CSV."""
        csv_path = os.path.join(self.dir_data, "width_data.csv")

        # Open the CSV file and write the header before starting the loop
        with open(csv_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write header
            csv_writer.writerow(["filename", "filepart1", "filepart2", "width_pixels", "location_along_petiole", "leaf_and_petiole_touch_within20pixels"])

            # Process images
            for filename in tqdm(os.listdir(self.dir_to_process), desc="Processing Images"):
                if filename.endswith('.png'):
                    image_path = os.path.join(self.dir_to_process, filename)
                    original_image, tag_image, petiole_present, leaf_pixels, petiole_pixels = self.process_image(image_path)

                    if petiole_present:
                        # Find the top of the petiole (closest point to the leaf)
                        top_pixel = self.find_top_of_petiole(leaf_pixels, petiole_pixels)
                        if top_pixel:
                            cPrint(f"Top of petiole located at {top_pixel} for {filename}")

                            do_touch = self.petiole_touches_leaf(leaf_pixels, petiole_pixels, fudge_factor=20)
                            cPrint(f"     >>> Leaf and petiole touch [{do_touch}]")


                            # Skeletonize the petiole using only the petiole pixels
                            skeleton_petiole = self.skeletonize_petiole(petiole_pixels, tag_image.shape)

                            # Find the skeleton point closest to the top pixel
                            skeleton_points = np.column_stack(np.where(skeleton_petiole > 0))
                            top_skeleton = skeleton_points[distance.cdist([top_pixel], skeleton_points).argmin()]
                            top_skeleton = tuple(top_skeleton)

                            # Trace the skeleton and find the furthest point (bottom_skeleton)
                            bottom_skeleton, max_distance = self.find_furthest_point(skeleton_petiole, top_skeleton)
                            cPrint(f"Bottom skeleton located at {bottom_skeleton} with path distance {max_distance} pixels.")

                            # Measure the width at the midpoint if distance <= 100
                            width_info = self.measure_widths(skeleton_petiole, tag_image, top_skeleton, bottom_skeleton, max_distance, self.verbose)

                            if width_info:
                                midpoint, left_point, right_point, width, location = width_info

                                # Write the data directly to the CSV file for each processed image
                                csv_writer.writerow([
                                    filename.split('.')[0],  
                                    '_'.join(filename.split("_")[0:2]),  # First two components
                                    '_'.join(filename.split("_")[0:3]),  # First three components
                                    width,  # Direct width value
                                    location,  # Direct location value
                                    do_touch,
                                ])
                                csvfile.flush()  # Explicitly flush the file
                                
                                if self.save_visualization:
                                    # If visualization flag is set, visualize and save as jpg
                                    petiole_mask = np.zeros_like(tag_image)
                                    for px in petiole_pixels:
                                        petiole_mask[px] = 1

                                    # Create the visualization and save it
                                    panel_image = self.visualize_widths(original_image, petiole_mask, skeleton_petiole, top_pixel, top_skeleton, bottom_skeleton, midpoint, left_point, right_point, do_touch)
                                    self.save_visualization_panel(panel_image, filename, self.verbose)

                        else:
                            cPrint(f"No top of petiole found for {filename}")
                    else:
                        cPrint(f"Skipping {filename}, no petiole present.")
                    del tag_image, leaf_pixels, petiole_pixels
                    gc.collect()  # Force garbage collection






if __name__ == "__main__":
    # Directory containing the PNG images to process

    # For running the LM2 images
    dir_to_process = "G:/Thais/LM2/Keypoints/Oriented_Masks"
    dir_overlay = "G:/Thais/LM2/Keypoints/testPetiole_Visualizations"
    dir_data = "G:/Thais/LM2/Keypoints/Petiole_Data_Fixed"
    save_visualization = False

    # For comparing GT with LM2
    # dir_to_process = "D:/Dropbox/LM2_Env/Image_Datasets/Thais_Petiole_Width/POINTS_Petiole_Width/cropped_leaves/oriented_masks"
    # # dir_to_process = "D:/Dropbox/LM2_Env/Image_Datasets/Thais_Petiole_Width/POINTS_Petiole_Width/cropped_leaves/to_fix"
    # dir_overlay = "D:/Dropbox/LM2_Env/Image_Datasets/Thais_Petiole_Width/POINTS_Petiole_Width/Petiole_Visualizations"
    # dir_data = "D:/Dropbox/LM2_Env/Image_Datasets/Thais_Petiole_Width/POINTS_Petiole_Width/Petiole_Data"
    # save_visualization = True

    
    # Create an instance of SegmentationMaskProcessor with visualization enabled
    processor = SegmentationMaskProcessor(dir_to_process, dir_overlay, dir_data, save_visualization=save_visualization, show_visualization=False, verbose=True)
    
    # Process the directory of images
    processor.process_directory()
