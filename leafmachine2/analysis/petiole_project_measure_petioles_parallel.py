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
from concurrent.futures import ProcessPoolExecutor


'''

THIS VERSION DOES NOT WORK CORRECTLY

USE THE non-PARALLELIZED VERSION

'''



class ConditionalPrinter:
    def __init__(self, verbose=False):
        """Initialize with a verbosity flag to control printing."""
        self.verbose = verbose

    def __call__(self, *args, **kwargs):
        """Mimic the behavior of print, but only print if verbose is True."""
        if self.verbose:
            print(*args, **kwargs)

def visualize_widths(original_image, petiole_mask, skeleton_petiole, top_pixel, top_skeleton, bottom_skeleton, midpoint, left_point, right_point, pt_size=2, show_visualization=False):
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
    cv2.line(petiole_cv, (left_point_y, left_point_x), (right_point_y, right_point_x), (0, 255, 0), 1)  # Green line for width

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

    if show_visualization:
        # Display the panel using OpenCV
        cv2.imshow('Four-Panel Visualization', panel_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return panel_image

def save_visualization_panel(panel_image, filename, dir_overlay, verbose=False):
        cPrint = ConditionalPrinter(verbose=verbose)  # Set to False to suppress printing
        """Save the visualized panel as a .jpg file in the overlay directory."""
        output_path = os.path.join(dir_overlay, f"{os.path.splitext(filename)[0]}.jpg")
        cv2.imwrite(output_path, panel_image)
        cPrint(f"Saved visualization as {output_path}")

def find_skeleton_path(skeleton, top_skeleton, bottom_skeleton):
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

def trace_perpendicular_line(petiole_mask, midpoint, perp_normalized, direction=1, max_range=1000):
    """Trace the perpendicular line from the midpoint in one direction (left or right) to find the edge."""
    h, w = petiole_mask.shape

    # Check if the initial midpoint is within bounds; if not, return None
    if not (0 <= midpoint[0] < h and 0 <= midpoint[1] < w):
        print(f"Midpoint {midpoint} is out of bounds (h={h}, w={w}), returning None.")
        return None

    for t in range(max_range):
        x = int(midpoint[0] + t * perp_normalized[0] * direction)
        y = int(midpoint[1] + t * perp_normalized[1] * direction)

        # Ensure the point is within bounds before accessing the array
        if not (0 <= x < h and 0 <= y < w):
            print(f"Point out of bounds at t={t}, x={x}, y={y}.")
            return None  # Return None when the point is out of bounds

        # Stop when we find a boundary (transition from 1 to 0)
        if petiole_mask[x, y] == 0:
            return (x, y)

    # Return None if no boundary is found within max_range
    return None



def measure_widths(skeleton, petiole_mask, top_skeleton, bottom_skeleton, max_distance, verbose=False):
    cPrint = ConditionalPrinter(verbose=verbose)  # Set to False to suppress printing

    """Measure the width of the petiole at the midpoint along the skeleton, expanding perpendicularly to the left and right."""
    location = None

    # Find the skeleton path from top_skeleton to bottom_skeleton
    path = find_skeleton_path(skeleton, top_skeleton, bottom_skeleton)
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
    left_point = trace_perpendicular_line(petiole_mask, midpoint, perp_normalized, direction=-1)  # Left direction
    right_point = trace_perpendicular_line(petiole_mask, midpoint, perp_normalized, direction=1)  # Right direction

    if left_point and right_point:
        # Calculate the Euclidean distance between the two points (width)
        width = math.hypot(right_point[0] - left_point[0], right_point[1] - left_point[1])
        cPrint(f"Width at midpoint: {width:.2f} pixels.")
        return midpoint, left_point, right_point, width, location
    else:
        cPrint("Could not find both left and right edge points.")
        return None



def find_furthest_point(skeleton, top_skeleton):
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

def process_single_image(args):
    """Process a single image and return the results needed."""
    (
        filename,
        image_path,
        color_to_tag,
        save_visualization,
        dir_overlay,
        verbose
    ) = args
    try:
        cPrint = ConditionalPrinter(verbose=verbose)

        # Process the image
        img = Image.open(image_path).convert('RGB')
        img_data = np.array(img)

        # Compute a unique integer for each pixel color
        img_data_int = (
            img_data[:, :, 0].astype(np.uint32) * 256 * 256 +
            img_data[:, :, 1].astype(np.uint32) * 256 +
            img_data[:, :, 2].astype(np.uint32)
        )

        # Create mapping from color integers to tags
        color_to_tag_int = {
            color[0] * 256 * 256 + color[1] * 256 + color[2]: tag
            for color, tag in color_to_tag.items()
        }

        # Initialize the tag_image with zeros
        tag_image = np.zeros(img_data_int.shape, dtype=np.uint8)

        # Assign tags to pixels based on color mapping
        try:
            for color_int, tag in color_to_tag_int.items():
                mask = img_data_int == color_int
                tag_image[mask] = tag
        except Exception as e:
            print(f"IndexError occurred while Assigning tags to pixels based on color mapping: {e}")



        # Find coordinates of leaf and petiole pixels
        leaf_pixels = [tuple(coord) for coord in np.argwhere(tag_image == 1)]
        petiole_pixels = [tuple(coord) for coord in np.argwhere(tag_image == 2)]

        # Check for petiole presence
        petiole_present = len(petiole_pixels) > 0
        
        if not petiole_present:
            cPrint(f"Skipping {filename}, no petiole present.")
            return None

        # Find the top of the petiole
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
        top_pixel = tuple(petiole_pixels[closest_petiole_idx])

        cPrint(f"Top of petiole located at {top_pixel} for {filename}")

        # Skeletonize the petiole
        petiole_mask = np.zeros(tag_image.shape, dtype=np.uint8)
        for px in petiole_pixels:
            petiole_mask[px] = 1

        skeleton_petiole = skeletonize(petiole_mask)

        # Find the skeleton point closest to the top pixel
        skeleton_points = np.column_stack(np.where(skeleton_petiole > 0))
        if skeleton_points.size == 0:
            cPrint(f"No skeleton points found for {filename}")
            return None

        try:
            top_skeleton_idx = distance.cdist([top_pixel], skeleton_points).argmin()
            top_skeleton = tuple(skeleton_points[top_skeleton_idx])
        except IndexError as e:
            print(f"IndexError occurred while finding top skeleton: {e}")
            print(f"Top pixel: {top_pixel}, skeleton points: {skeleton_points.shape}")
            return None

        # Find the furthest point on the skeleton
        bottom_skeleton, max_distance = find_furthest_point(
            skeleton_petiole, top_skeleton
        )
        cPrint(f"Bottom skeleton at {bottom_skeleton}, distance {max_distance}")

        # Measure the width
        width_info = measure_widths(
            skeleton_petiole, petiole_mask, top_skeleton, bottom_skeleton, max_distance, verbose
        )

        if width_info is None:
            cPrint(f"Width measurement failed for {filename}")
            return None

        midpoint, left_point, right_point, width, location = width_info

        if save_visualization:
            # If visualization flag is set, visualize and save as jpg
            petiole_mask = np.zeros_like(tag_image)
            for px in petiole_pixels:
                petiole_mask[px] = 1

            # Create the visualization and save it
            panel_image = visualize_widths(img, petiole_mask, skeleton_petiole, top_pixel, top_skeleton, bottom_skeleton, midpoint, left_point, right_point)
            save_visualization_panel(panel_image, filename, dir_overlay, False)

        # Return the data to write to CSV
        img.close()

        return [
            filename.split('.')[0],
            '_'.join(filename.split("_")[0:2]),
            '_'.join(filename.split("_")[0:3]),
            width,
            location
        ]
    except Exception as e:
        print(e)
        return None

def merge_csvs(dir_data, num_workers):
    """Merge CSV files from each worker into a final CSV."""
    final_csv_path = os.path.join(dir_data, "width_data.csv")
    
    with open(final_csv_path, mode='w', newline='') as final_csv:
        writer = csv.writer(final_csv)
        writer.writerow(["filename", "filepart1", "filepart2", "width_pixels", "location_along_petiole"])

        for worker_id in range(num_workers):
            worker_csv_path = os.path.join(dir_data, f"width_data__{worker_id}.csv")
            with open(worker_csv_path, mode='r') as worker_csv:
                reader = csv.reader(worker_csv)
                next(reader)  # Skip header
                for row in reader:
                    writer.writerow(row)




def process_images_in_parallel(
    dir_to_process,
    dir_overlay,
    dir_data,
    color_to_tag,
    save_visualization=True,
    verbose=False,
    num_workers=4,
):
    """Process all images in the directory using ProcessPoolExecutor."""
    cPrint = ConditionalPrinter(verbose=verbose)

    # Get list of all files to process
    all_files = [f for f in os.listdir(dir_to_process) if f.endswith('.png')]
    total_files = len(all_files)

    # Prepare arguments for each process
    args_list = [
        (
            filename,
            os.path.join(dir_to_process, filename),
            color_to_tag,
            save_visualization,
            dir_overlay,
            verbose
        )
        for filename in all_files
    ]

    csv_path = os.path.join(dir_data, "width_data.csv")
    with open(csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(["filename", "filepart1", "filepart2", "width_pixels", "location_along_petiole"])

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Use tqdm to show progress bar
            for result in tqdm(executor.map(process_single_image, args_list), total=total_files, desc="Processing Images"):
                if result is not None:
                    # Write the result to CSV
                    csv_writer.writerow(result)
                    csvfile.flush()

if __name__ == "__main__":
    '''All the images'''
    # dir_to_process = "G:/Thais/LM2/Keypoints/Oriented_Masks"
    # dir_overlay = "G:/Thais/LM2/Keypoints/Petiole_Visualizations"
    # dir_data = "G:/Thais/LM2/Keypoints/Petiole_Data"

    '''Process the manually annotated images for validation'''
    dir_to_process = "D:/Dropbox/LM2_Env/Image_Datasets/Thais_Petiole_Width/POINTS_Petiole_Width/cropped_leaves/oriented_masks"
    dir_overlay = "D:/Dropbox/LM2_Env/Image_Datasets/Thais_Petiole_Width/POINTS_Petiole_Width/Petiole_Visualizations"
    dir_data = "D:/Dropbox/LM2_Env/Image_Datasets/Thais_Petiole_Width/POINTS_Petiole_Width/Petiole_Data"

    color_to_tag = {
        (46, 255, 0): 1,   # leaf
        (0, 173, 255): 2,  # petiole
        (209, 0, 255): 3,  # hole
    }

    # Ensure directories exist
    os.makedirs(dir_overlay, exist_ok=True)
    os.makedirs(dir_data, exist_ok=True)

    process_images_in_parallel(
        dir_to_process,
        dir_overlay,
        dir_data,
        color_to_tag,
        save_visualization=True,
        verbose=False,
        num_workers=4,
    )