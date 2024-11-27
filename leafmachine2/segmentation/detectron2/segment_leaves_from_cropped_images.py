import os, json, random, glob, inspect, sys, cv2, itertools, torch, sqlite3, logging, warnings
from timeit import default_timer as timer
from PIL import Image, ImageDraw
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from dataclasses import dataclass, field
import matplotlib.patches as mpl_patches
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
import math
from time import perf_counter
from shapely.geometry import Polygon, Point, MultiPoint
from io import BytesIO
import concurrent.futures
import threading
from scipy.ndimage import binary_erosion, rotate
from tqdm import tqdm
import multiprocessing
from queue import Empty
from shapely.geometry import LineString

currentdir = os.path.dirname(inspect.getfile(inspect.currentframe()))
parentdir1 = os.path.dirname(os.path.dirname(currentdir))
parentdir2 = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
sys.path.append(currentdir)
sys.path.append(parentdir1)
sys.path.append(parentdir2)
from measure_leaf_segmentation import polygon_properties
from detector import Detector_LM2
from leafmachine2.keypoint_detector.ultralytics.models.yolo.pose.predict_direct import PosePredictor
from leafmachine2.segmentation.detectron2.segment_utils import get_largest_polygon, keep_rows, get_string_indices
from leafmachine2.machine.LM2_logger import initialize_logger_for_parallel_processes
from leafmachine2.machine.data_project_sql import get_database_path, test_sql
from leafmachine2.segmentation.detectron2.segment_leaves import create_overlay_and_calculate_props, create_insert, save_full_overlay_images, save_rgb_cropped, save_individual_segmentations, save_masks_color, save_full_masks

class Dirs:
    pass

def clean_file_extensions(dir_path):
    """Rename files in `dir_path` by removing '.jpg' from filenames ending with '.jpg.png'."""
    
    # Iterate through all files in the specified directory
    for filename in os.listdir(dir_path):
        # Check if filename ends with ".jpg.png"
        if filename.endswith('.jpg.png'):
            # Construct full file paths for the old and new filenames
            old_file_path = os.path.join(dir_path, filename)
            new_filename = filename.replace('.jpg.png', '.png')
            new_file_path = os.path.join(dir_path, new_filename)
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: '{filename}' to '{new_filename}'")

def segment_whole_leaves(dir_home, 
                         dir_images, 
                         Dirs, 
                         device_list,
                         num_workers=4):
    
    # Get a list of all filenames in the directory
    batch_filenames = os.listdir(dir_images)
    total_files = len(batch_filenames)

    # Calculate chunk size for even distribution
    chunk_size = (total_files + num_workers - 1) // num_workers  # Ceiling division

    # Split filenames into chunks, handling remainder
    chunks = [batch_filenames[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers - 1)]
    chunks.append(batch_filenames[(num_workers - 1) * chunk_size:])  # Last chunk handles the remainder

    # Create a progress queue to handle progress updates
    progress_queue = multiprocessing.Queue()

    # Create a tqdm progress bar in the main process
    with tqdm(total=total_files, desc="Segmenting Whole Leaves", unit="image") as pbar:
        processes = []
        for i, chunk in enumerate(chunks):
            # Assign device from device_list (distribute workload across devices)
            device = device_list[i % len(device_list)]  # Cycle through available devices

            p = multiprocessing.Process(
                target=segment_images_batch,
                args=(dir_home, dir_images, Dirs, device, chunk, progress_queue)
            )
            processes.append(p)
            p.start()

        # Continuously update the progress bar
        completed = 0
        while completed < total_files:
            progress_update = progress_queue.get()
            pbar.update(progress_update)
            completed += progress_update

        # Ensure all processes have finished
        for p in processes:
            p.join()

    torch.cuda.empty_cache()


def segment_images_batch(dir_home, 
                         dir_images, 
                         Dirs, 
                         device, 
                         filenames, 
                         progress_queue):
    
    torch.cuda.empty_cache()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    leaf_type = 0

    # Initialize the instance detector (model is loaded once per process)
    dir_seg_model = os.path.join(dir_home, 'leafmachine2', 'segmentation', 'models', 'Group3_Dataset_100000_Iter_1176PTS_512Batch_smooth_l1_LR00025_BGR')
    Instance_Detector = Detector_LM2(logger, dir_seg_model, 0.7, leaf_type, device)

    save_oriented_images = True
    save_keypoint_overlay = True
    generate_overlay = True
    overlay_dpi = 300
    bg_color = 'black'
    dict_name_seg = 'Segmentation_Whole_Leaf'
    keep_best = True
    save_each_segmentation_overlay_image = True
    save_individual_overlay_images = True
    save_rgb_cropped_images = True

    save_ind_masks_color = True
    save_full_image_masks_color = True
    use_efds_for_masks = False
    
    # Weights folder base
    dir_weights = os.path.join(dir_home, 'leafmachine2', 'keypoint_detector','keypoint_models')
    detector_version = 'uniform_spaced_oriented_traces_mid15_pet5_clean_640_flipidx_pt2'
    weights = os.path.join(dir_weights, detector_version, 'weights', 'best.pt')

    # Create dictionary for overrides
    overrides = {
        'model': weights,
        'name': detector_version,
        'boxes': False,
        'max_det': 1,
    }
    cfg = {}
    cfg['leafmachine'] = {}
    cfg['leafmachine']['leaf_segmentation'] = {}
    cfg['leafmachine']['leaf_segmentation']['overlay_line_width'] = 1
    cfg['leafmachine']['leaf_segmentation']['calculate_elliptic_fourier_descriptors'] = True
    cfg['leafmachine']['leaf_segmentation']['elliptic_fourier_descriptor_order'] = 40
    cfg['leafmachine']['leaf_segmentation']['find_minimum_bounding_box'] = True

    # Initialize PosePredictor (also loaded once per process)
    Pose_Predictor = PosePredictor(weights, Dirs.dir_oriented_images, Dirs.dir_keypoint_overlay, 
                                   save_oriented_images=save_oriented_images, save_keypoint_overlay=save_keypoint_overlay, 
                                   overrides=overrides)

    

    # Process each filename in the batch
    for filename in filenames:
        # Load image
        try:
            # img_path = glob.glob(os.path.join(dir_images, f"{filename}.*"))[0]
            img_path = os.path.join(dir_images, filename)
            img_cropped = cv2.imread(img_path)
            # img_cropped = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Failed to load image {filename}: {e}")
            continue
    
        # img_cropped = cv2.imdecode(np.frombuffer(full_image, np.uint8), cv2.IMREAD_COLOR)

        # Segment the image
        out_polygons, out_bboxes, out_labels, out_color = Instance_Detector.segment(img_cropped, 
                                                                                    generate_overlay,
                                                                                    overlay_dpi,
                                                                                    bg_color)

        # Further processing (e.g., keypoint detection, cropping, saving results)
        keypoint_data = Pose_Predictor.process_images(img_cropped, filename=filename)

        # if inddd == 2:
        #     print(keypoint_data)

        if len(out_polygons) > 0:
            if keep_best:
                out_polygons, out_bboxes, out_labels, out_color = keep_rows(out_polygons, out_bboxes, out_labels, out_color, get_string_indices(out_labels))

            detected_components, cropped_overlay, overlay_data = create_overlay_and_calculate_props(keypoint_data, filename, img_cropped, out_polygons, out_labels, out_color, cfg)
            try:
                img_cropped = create_insert(img_cropped, overlay_data, filename.split("__")[2], cfg)
            except:
                img_cropped = create_insert(img_cropped, overlay_data, filename.split("__")[1], cfg)

            cropped_overlay_size = cropped_overlay.shape
        else:
            detected_components = []
            cropped_overlay = []
            overlay_data = []
            cropped_overlay_size = []
        
        # Save results to SQL
        # save_segmentation_results_to_sql(cur, filename, filename, detected_components, out_polygons, out_bboxes, out_labels, out_color, cropped_overlay, overlay_data, dict_name_seg)
        # save_keypoints_results_to_sql(cur, filename, filename, keypoint_data[filename])


        # Save RGB cropped images
        # save_rgb_cropped(save_rgb_cropped_images, filename, img_cropped, leaf_type, Dirs)

        # Save individual segmentations
        # save_individual_segmentations(save_individual_overlay_images, dict_name_seg, filename, cropped_overlay, Dirs)

        # Handle full image masks
        try:
            full_size = img_cropped.shape
            full_mask = Image.new('RGB', (full_size[1], full_size[0]), color=(0, 0, 0))
        except:
            full_size = img_cropped.size
            full_mask = Image.new('RGB', (full_size[0], full_size[1]), color=(0, 0, 0))

        try:
            full_mask = save_masks_color(keypoint_data, save_oriented_images, save_ind_masks_color, save_full_image_masks_color, 
                                    use_efds_for_masks, full_mask, overlay_data, cropped_overlay_size, full_size, filename.split(".")[0], 
                                    filename.split("__")[2], leaf_type, Dirs, 
                                    0)
        except:
            full_mask = save_masks_color(keypoint_data, save_oriented_images, save_ind_masks_color, save_full_image_masks_color, 
                                    use_efds_for_masks, full_mask, overlay_data, cropped_overlay_size, full_size, filename.split(".")[0], 
                                    filename.split("__")[1], leaf_type, Dirs, 
                                    0)

        # Save full masks
        # if save_full_image_masks_color:
            # save_full_masks(save_full_image_masks_color, full_mask, filename, leaf_type, Dirs)

        # Save full overlay images
        # save_full_overlay_images(save_each_segmentation_overlay_image, full_image, filename, leaf_type, Dirs)
    progress_queue.put(len(filenames))
    torch.cuda.empty_cache()


    
if __name__ == '__main__':
    do_segment = True
    model_path = "D:/Dropbox/LeafMachine2/KP_2024/uniform_spaced_oriented_traces_mid15_pet5_clean_640_flipidx_pt2/weights/best.pt"
    dir_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))))
    
    Dir = Dirs()

    # Define the paths
    img_path = "D:/Dropbox/LM2_Env/Image_Datasets/Thais_Petiole_Width/POINTS_Petiole_Width/cropped_leaves/leaf"

    dir_oriented_images = "D:/Dropbox/LM2_Env/Image_Datasets/Thais_Petiole_Width/POINTS_Petiole_Width/cropped_leaves/oriented_leaves"
    dir_keypoint_overlay = "D:/Dropbox/LM2_Env/Image_Datasets/Thais_Petiole_Width/POINTS_Petiole_Width/cropped_leaves/keypoint_overlay"
    dir_segment_oriented = "D:/Dropbox/LM2_Env/Image_Datasets/Thais_Petiole_Width/POINTS_Petiole_Width/cropped_leaves/oriented_masks"
    dir_whole_leaves = "D:/Dropbox/LM2_Env/Image_Datasets/Thais_Petiole_Width/POINTS_Petiole_Width/cropped_leaves/whole_leaves"

    dir_simple_txt = "D:/Dropbox/LM2_Env/Image_Datasets/Thais_Petiole_Width/POINTS_Petiole_Width/cropped_leaves/simple_txt"
    dir_simple_raw_txt = "D:/Dropbox/LM2_Env/Image_Datasets/Thais_Petiole_Width/POINTS_Petiole_Width/cropped_leaves/simple_raw_txt"
    dir_simple_txt_DP = "D:/Dropbox/LM2_Env/Image_Datasets/Thais_Petiole_Width/POINTS_Petiole_Width/cropped_leaves/simple_txt_DP"

    pths = [dir_oriented_images,
            dir_keypoint_overlay,
            dir_segment_oriented,
            dir_simple_txt,
            dir_simple_raw_txt,
            dir_simple_txt_DP,
            dir_whole_leaves,
            ]


    if not os.path.exists(img_path):
        print(f"Image path {img_path} does not exist.")
        exit()
        
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist.")
        exit()

    for pth in pths:
        if not os.path.exists(pth):
            print(f"Creating {pth}")
            os.makedirs(pth, exist_ok=True)
    
    

    Dir.dir_oriented_masks = dir_segment_oriented
    Dir.dir_oriented_images = dir_oriented_images
    Dir.dir_keypoint_overlay = dir_keypoint_overlay
    Dir.segmentation_masks_color_whole_leaves = dir_whole_leaves
    Dir.dir_simple_txt = dir_simple_txt
    Dir.dir_simple_raw_txt = dir_simple_raw_txt
    Dir.dir_simple_txt_DP = dir_simple_txt_DP

    # segment_whole_leaves(dir_home, 
    #                      img_path, 
    #                      Dir, 
    #                      ['cuda'])


    # If the jpg remains in the file name of the png
    clean_file_extensions(dir_segment_oriented)