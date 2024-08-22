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

def segment_leaves(cfg, time_report, logger, dir_home, ProjectSQL, batch, n_batches, batch_filenames, Dirs): 

    start_t = perf_counter()
    logger.name = f'[BATCH {batch+1} Segment Leaves]'
    logger.info(f'Segmenting leaves for batch {batch+1} of {n_batches}')

    # batch_size = cfg['leafmachine']['project']['batch_size']
    if cfg['leafmachine']['project']['num_workers_seg'] is None:
        num_workers = 1
    else:
        num_workers = int(cfg['leafmachine']['project']['num_workers_seg'])

    for filename in batch_filenames:
        # Unpack classes from components and store them in the SQL database
        unpack_class_from_components(ProjectSQL, filename, 0, 'Whole_Leaf_BBoxes_YOLO', 'Whole_Leaf_BBoxes')
        unpack_class_from_components(ProjectSQL, filename, 1, 'Partial_Leaf_BBoxes_YOLO', 'Partial_Leaf_BBoxes')

        # Crop the images to bounding boxes and store the results in the SQL database
        crop_images_to_bbox(ProjectSQL, filename, 0, 'Whole_Leaf_Cropped', 'Whole_Leaf_BBoxes')
        crop_images_to_bbox(ProjectSQL, filename, 1, 'Partial_Leaf_Cropped', 'Partial_Leaf_BBoxes')






    # Run the leaf instance segmentation operations
    # dir_seg_model = os.path.join(dir_home, 'leafmachine2', 'segmentation', 'models',cfg['leafmachine']['leaf_segmentation']['segmentation_model'])
    # Instance_Detector_Whole = Detector_LM2(logger, dir_seg_model, cfg['leafmachine']['leaf_segmentation']['minimum_confidence_threshold'], 0)
    # Instance_Detector_Partial = Detector_LM2(logger, dir_seg_model, cfg['leafmachine']['leaf_segmentation']['minimum_confidence_threshold'], 1)


    save_oriented_images = cfg['leafmachine']['leaf_segmentation']['save_oriented_images']
    save_keypoint_overlay = cfg['leafmachine']['leaf_segmentation']['save_keypoint_overlay']
    save_oriented_mask = cfg['leafmachine']['leaf_segmentation']['save_oriented_mask']
    save_simple_txt = cfg['leafmachine']['leaf_segmentation']['save_simple_txt']
    # Weights folder base
    dir_weights = os.path.join(dir_home, 'leafmachine2', 'keypoint_detector','keypoint_models')
    detector_version = cfg['leafmachine']['leaf_segmentation']['detector_version']
    weights =  os.path.join(dir_weights,detector_version,'weights','best.pt')

    # Create dictionary for overrides
    overrides = {
        'model': weights,
        # 'source': img_path,
        'name': detector_version,
        'boxes':False,
        'max_det':1,
        # 'visualize': True,
        # 'save_txt': True,
        # 'show':True
    }


    if cfg['leafmachine']['leaf_segmentation']['segment_whole_leaves']:
        logger.info(f'Segmenting whole leaves')
        segment_whole_leaves(cfg, logger, dir_home, ProjectSQL, batch_filenames, Dirs, num_workers)

    if cfg['leafmachine']['leaf_segmentation']['segment_partial_leaves']:
        logger.info(f'Segmenting partial leaves')
        segment_partial_leaves(cfg, logger, dir_home, ProjectSQL, batch_filenames, Dirs, num_workers)



    
    end_t = perf_counter()
    # print(f'Batch {batch+1}/{n_batches}: Leaf Segmentation Duration --> {round((end_t - start_t)/60)} minutes')
    t_seg = f"[Batch {batch+1}/{n_batches}: Leaf Segmentation elapsed time] {round(end_t - start_t)} seconds ({round((end_t - start_t)/60)} minutes)"
    logger.info(t_seg)
    time_report['t_seg'] = t_seg
    return time_report



def segment_whole_leaves(cfg, logger, dir_home, ProjectSQL, batch_filenames, Dirs, num_workers):
    logger.info(f'Segmenting whole leaves')

    # Split filenames into chunks for each worker
    chunks = [batch_filenames[i::num_workers] for i in range(num_workers)]
    
    total_files = len(batch_filenames)
    
    # Create a progress queue to handle progress updates
    progress_queue = multiprocessing.Queue()

    # Create a tqdm progress bar in the main process
    with tqdm(total=total_files, desc="Segmenting Whole Leaves", unit="image") as pbar:
        processes = []
        for chunk in chunks:
            p = multiprocessing.Process(
                target=segment_images_batch,
                args=(cfg, dir_home, ProjectSQL.database, ProjectSQL.dir_images, chunk, 0,
                      'Whole_Leaf_Cropped', 'Whole_Leaf_BBoxes', 'Segmentation_Whole_Leaf', Dirs, progress_queue)
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

def segment_partial_leaves(cfg, logger, dir_home, ProjectSQL, batch_filenames, Dirs, num_workers):
    logger.info(f'Segmenting partial leaves')

    # Split filenames into chunks for each worker
    chunks = [batch_filenames[i::num_workers] for i in range(num_workers)]
    
    total_files = len(batch_filenames)
    
    # Create a progress queue to handle progress updates
    progress_queue = multiprocessing.Queue()

    # Create a tqdm progress bar in the main process
    with tqdm(total=total_files, desc="Segmenting Partial Leaves", unit="image") as pbar:
        processes = []
        for chunk in chunks:
            p = multiprocessing.Process(
                target=segment_images_batch,
                args=(cfg, dir_home, ProjectSQL.database, ProjectSQL.dir_images, chunk, 1,
                      'Partial_Leaf_Cropped', 'Partial_Leaf_BBoxes', 'Segmentation_Partial_Leaf', Dirs, progress_queue)
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
    

# def segment_images(cfg, logger, dir_home, ProjectSQL, filename, leaf_type, dict_name_cropped, dict_from, dict_name_seg, Dirs):
#     conn = sqlite3.connect(ProjectSQL.database)
#     cur = conn.cursor()

#     # Initialize the instance detector
#     dir_seg_model = os.path.join(dir_home, 'leafmachine2', 'segmentation', 'models', cfg['leafmachine']['leaf_segmentation']['segmentation_model'])
#     Instance_Detector = Detector_LM2(logger, dir_seg_model, cfg['leafmachine']['leaf_segmentation']['minimum_confidence_threshold'], leaf_type)

#     save_oriented_images = cfg['leafmachine']['leaf_segmentation']['save_oriented_images']
#     save_keypoint_overlay = cfg['leafmachine']['leaf_segmentation']['save_keypoint_overlay']
#     save_oriented_mask = cfg['leafmachine']['leaf_segmentation']['save_oriented_mask']
#     save_simple_txt = cfg['leafmachine']['leaf_segmentation']['save_simple_txt']
    
#     # Weights folder base
#     dir_weights = os.path.join(dir_home, 'leafmachine2', 'keypoint_detector','keypoint_models')
#     detector_version = cfg['leafmachine']['leaf_segmentation']['detector_version']
#     weights = os.path.join(dir_weights, detector_version, 'weights', 'best.pt')

#     # Create dictionary for overrides
#     overrides = {
#         'model': weights,
#         'name': detector_version,
#         'boxes': False,
#         'max_det': 1,
#     }

#     # Initialize PosePredictor
#     Pose_Predictor = PosePredictor(weights, Dirs.dir_oriented_images, Dirs.dir_keypoint_overlay, 
#                                    save_oriented_images=save_oriented_images, save_keypoint_overlay=save_keypoint_overlay, 
#                                    overrides=overrides)
   
def initialize_logger(logger_name, log_to_file=None, suppress_warnings=None, suppress_loggers=None):
    """
    Initialize and return a logger with the given name.

    Parameters:
        logger_name (str): Name of the logger.
        log_to_file (str or None): Path to a file to log to. If None, logs will only appear in the console.
        suppress_warnings (str or list of str or None): Messages or warning types to suppress.
        suppress_loggers (list of str or None): Specific loggers to suppress by name.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Initialize the logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.ERROR)
    
    # Set up log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure stream handler (console output)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # If log_to_file is provided, add a file handler
    if log_to_file:
        file_handler = logging.FileHandler(log_to_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Suppress specific warnings if provided
    if suppress_warnings:
        if isinstance(suppress_warnings, str):
            warnings.filterwarnings("ignore", message=suppress_warnings)
        elif isinstance(suppress_warnings, list):
            for warning in suppress_warnings:
                warnings.filterwarnings("ignore", message=warning)
    
    # Suppress specific loggers if provided
    if suppress_loggers:
        for suppressed_logger_name in suppress_loggers:
            suppressed_logger = logging.getLogger(suppressed_logger_name)
            suppressed_logger.setLevel(logging.ERROR)
            # Optionally, you can also disable propagation if needed
            suppressed_logger.propagate = False

    return logger

def segment_images_batch(cfg, dir_home, ProjectSQL, dir_images, filenames, leaf_type, dict_name_cropped, dict_from, dict_name_seg, Dirs, progress_queue):
    conn = sqlite3.connect(ProjectSQL)
    cur = conn.cursor()

    # Reinitialize the logger inside the process if necessary
    logger = initialize_logger('SEGMENT', 
                           suppress_warnings=["torch.meshgrid"],
                           suppress_loggers=['Checkpointer', 'ultralytics', 'detectron2', 'yolo'])


    # Initialize the instance detector (model is loaded once per process)
    dir_seg_model = os.path.join(dir_home, 'leafmachine2', 'segmentation', 'models', cfg['leafmachine']['leaf_segmentation']['segmentation_model'])
    Instance_Detector = Detector_LM2(logger, dir_seg_model, cfg['leafmachine']['leaf_segmentation']['minimum_confidence_threshold'], leaf_type)

    save_oriented_images = cfg['leafmachine']['leaf_segmentation']['save_oriented_images']
    save_keypoint_overlay = cfg['leafmachine']['leaf_segmentation']['save_keypoint_overlay']
    save_oriented_mask = cfg['leafmachine']['leaf_segmentation']['save_oriented_mask']
    save_simple_txt = cfg['leafmachine']['leaf_segmentation']['save_simple_txt']
    
    # Weights folder base
    dir_weights = os.path.join(dir_home, 'leafmachine2', 'keypoint_detector','keypoint_models')
    detector_version = cfg['leafmachine']['leaf_segmentation']['detector_version']
    weights = os.path.join(dir_weights, detector_version, 'weights', 'best.pt')

    # Create dictionary for overrides
    overrides = {
        'model': weights,
        'name': detector_version,
        'boxes': False,
        'max_det': 1,
    }

    # Initialize PosePredictor (also loaded once per process)
    Pose_Predictor = PosePredictor(weights, Dirs.dir_oriented_images, Dirs.dir_keypoint_overlay, 
                                   save_oriented_images=save_oriented_images, save_keypoint_overlay=save_keypoint_overlay, 
                                   overrides=overrides)

    generate_overlay = cfg['leafmachine']['leaf_segmentation']['generate_overlay']
    overlay_dpi = cfg['leafmachine']['leaf_segmentation']['overlay_dpi']
    bg_color = cfg['leafmachine']['leaf_segmentation']['overlay_background_color']
    keep_best = cfg['leafmachine']['leaf_segmentation']['keep_only_best_one_leaf_one_petiole']
    save_overlay_pdf = cfg['leafmachine']['leaf_segmentation']['save_segmentation_overlay_images_to_pdf']
    save_each_segmentation_overlay_image = cfg['leafmachine']['leaf_segmentation']['save_segmentation_overlay_images_to_pdf']
    save_individual_overlay_images = cfg['leafmachine']['leaf_segmentation']['save_individual_overlay_images']
    save_rgb_cropped_images = cfg['leafmachine']['leaf_segmentation']['save_rgb_cropped_images']

    save_ind_masks_color = cfg['leafmachine']['leaf_segmentation']['save_masks_color']
    save_full_image_masks_color = cfg['leafmachine']['leaf_segmentation']['save_full_image_masks_color']
    use_efds_for_masks = cfg['leafmachine']['leaf_segmentation']['use_efds_for_png_masks']

    # Process each filename in the batch
    for filename in filenames:
        # Safely increment the shared counter
        progress_queue.put(1)
        # Load image
        try:
            img_path = glob.glob(os.path.join(dir_images, f"{filename}.*"))[0]
            full_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        except:
            logger.error(f"Could not load image for {filename}")
            return

        # Determine conversion factor (CF)
        CF = determine_conversion_factor(filename, cur)

        # Retrieve the cropped images from SQL
        cur.execute(f"SELECT crop_name, cropped_image FROM {dict_name_cropped} WHERE file_name = ?", (filename,))
        crops = cur.fetchall()

        for crop_name, img_cropped_blob in crops:
            img_cropped = cv2.imdecode(np.frombuffer(img_cropped_blob, np.uint8), cv2.IMREAD_COLOR)

            # Segment the image
            out_polygons, out_bboxes, out_labels, out_color = Instance_Detector.segment(img_cropped, 
                                                                                        generate_overlay,
                                                                                        overlay_dpi,
                                                                                        bg_color)

            # Further processing (e.g., keypoint detection, cropping, saving results)
            keypoint_data = Pose_Predictor.process_images(img_cropped, filename=crop_name)

            if len(out_polygons) > 0:
                if keep_best:
                    out_polygons, out_bboxes, out_labels, out_color = keep_rows(out_polygons, out_bboxes, out_labels, out_color, get_string_indices(out_labels))

                detected_components, cropped_overlay, overlay_data = create_overlay_and_calculate_props(keypoint_data, crop_name, img_cropped, out_polygons, out_labels, out_color, cfg)
                full_image = create_insert(full_image, overlay_data, crop_name.split("__")[2], cfg)

                cropped_overlay_size = cropped_overlay.shape
            else:
                detected_components = []
                cropped_overlay = []
                overlay_data = []
                cropped_overlay_size = []

            # Save results to SQL
            save_segmentation_results_to_sql(cur, filename, crop_name, detected_components, out_polygons, out_bboxes, out_labels, out_color, cropped_overlay, overlay_data, dict_name_seg)


            conn.commit()

            # Save RGB cropped images
            save_rgb_cropped(save_rgb_cropped_images, crop_name, img_cropped, leaf_type, Dirs)

            # Save individual segmentations
            save_individual_segmentations(save_individual_overlay_images, dict_name_seg, crop_name, cropped_overlay, Dirs)

            # Handle full image masks
            try:
                full_size = full_image.shape
                if save_full_image_masks_color:
                    full_mask = Image.new('RGB', (full_size[1], full_size[0]), color=(0, 0, 0))
            except:
                full_size = full_image.size
                if save_full_image_masks_color:
                    full_mask = Image.new('RGB', (full_size[0], full_size[1]), color=(0, 0, 0))

            if save_full_image_masks_color:
                full_mask = save_masks_color(keypoint_data, save_oriented_images, save_ind_masks_color, save_full_image_masks_color, 
                                            use_efds_for_masks, full_mask, overlay_data, cropped_overlay_size, full_size, crop_name, 
                                            crop_name.split("__")[2], leaf_type, Dirs, CF)

            # Save full masks
            save_full_masks(save_full_image_masks_color, full_mask, filename, leaf_type, Dirs)

            # Save full overlay images
            save_full_overlay_images(save_each_segmentation_overlay_image, full_image, filename, leaf_type, Dirs)

    # Save full image segmentations to PDF if configured
    # if save_overlay_pdf:
        # save_full_image_segmentations(save_overlay_pdf, dict_name_seg, [full_image], [filename], Dirs, cfg)
        #  save_full_image_segmentations(save_overlay_pdf, dict_name_seg, full_images, filenames, Dirs, cfg, batch, n_batches, lock)#, start, end)

    
    conn.close()
    torch.cuda.empty_cache()

def convert_ndarray_to_list(d):
    """Recursively convert numpy data types to Python native types in a nested dictionary or list."""
    if isinstance(d, dict):
        return {k: convert_ndarray_to_list(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_ndarray_to_list(v) for v in d]
    elif isinstance(d, np.ndarray):
        return d.tolist()
    elif isinstance(d, (np.integer, np.int32, np.int64)):
        return int(d)
    elif isinstance(d, (np.floating, np.float32, np.float64)):
        return float(d)
    elif isinstance(d, np.bool_):
        return bool(d)
    else:
        return d
    
def save_segmentation_results_to_sql(cur, filename, crop_name, detected_components, out_polygons, out_bboxes, out_labels, out_color, cropped_overlay, overlay_data, dict_name_seg):
    # Convert detected components to a JSON serializable format
    detected_components_serializable = convert_ndarray_to_list(detected_components)

    # Convert the other data structures if needed
    out_polygons_serializable = convert_ndarray_to_list(out_polygons)
    out_bboxes_serializable = convert_ndarray_to_list(out_bboxes)
    out_labels_serializable = convert_ndarray_to_list(out_labels)
    out_color_serializable = convert_ndarray_to_list(out_color)
    overlay_data_serializable = convert_ndarray_to_list(overlay_data)

    # Serialize the data to JSON
    detected_components_json = json.dumps(detected_components_serializable)
    out_polygons_json = json.dumps(out_polygons_serializable)
    out_bboxes_json = json.dumps(out_bboxes_serializable)
    out_labels_json = json.dumps(out_labels_serializable)
    out_color_json = json.dumps(out_color_serializable)

    # Convert overlay data to JSON
    overlay_data_serializable = []
    for overlay in overlay_data:
        if isinstance(overlay, np.ndarray):
            overlay_data_serializable.append(overlay.tolist())
        else:
            overlay_data_serializable.append(overlay)
    overlay_data_json = json.dumps(overlay_data_serializable, default=lambda o: o.item() if isinstance(o, np.generic) else o)

    # Convert cropped overlay to binary (Blob) to store in SQLite if needed
    # _, cropped_overlay_blob = cv2.imencode('.png', cropped_overlay)

    # Save the results back to the SQL database
    cur.execute(f"""
        INSERT INTO {dict_name_seg} 
        (file_name, crop_name, segmentation_data, polygons, bboxes, labels, colors, overlay_data) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (filename, crop_name, detected_components_json, out_polygons_json, out_bboxes_json, out_labels_json, out_color_json, overlay_data_json)
    )   


''' #SEGMENT PARALLEL
def segment_images_parallel(logger, dir_home, dict_objects, leaf_type, dict_name_seg, dict_from, cfg, Project, Dirs, batch, n_batches, num_workers):
    
    
    seg_overlay = {}
    # seg_overlay_data = {}

    # Define a lock object
    lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            dict_objects_chunk = dict(list(dict_objects.items())[i::num_workers])
            futures.append(executor.submit(segment_images, logger, dir_home, dict_objects_chunk, leaf_type, dict_name_seg, dict_from, cfg, Project, Dirs, batch, n_batches, lock))

        for future in concurrent.futures.as_completed(futures):
            dict_objects, seg_overlay_chunk = future.result()
            # for filename, value in dict_objects.items(): # 7-12-24 memory saving test
                # seg_overlay[filename] = seg_overlay_chunk[filename] # 7-12-24 memory saving test

                # seg_overlay_data[filename] = seg_overlay_chunk[filename]

                # filenames.append(filename)

                # if save_overlay_pdf:
                #     full_images.append(Image.open(os.path.join(Dirs.path_segmentation_images, seg_overlay_name, filename)))

                # if save_full_image_masks_color:
                #     full_masks.append(Image.open(os.path.join(Dirs.path_segmentation_images, seg_overlay_name, filename.split(".")[0] + "_mask.png")))

    # save_full_image_segmentations(save_overlay_pdf, dict_name_seg, full_images, filenames, Dirs, cfg, batch, n_batches)#, start, end)
    torch.cuda.empty_cache()
    return dict_objects, seg_overlay#, seg_overlay_data
#SEGMENT PARALLEL'''


def determine_conversion_factor(filename, cur):
    """
    Determine the conversion factor (CF) for the given filename based on the conversion_mean and predicted_conversion_factor_cm.
    
    Parameters:
        filename (str): The filename key to look up in the database.
        cur (sqlite3.Cursor): The SQLite cursor object to execute queries.
        
    Returns:
        float: The calculated conversion factor (CF).
    """
    # Query to get the ruler information from the SQL database
    cur.execute("SELECT conversion_mean, predicted_conversion_factor_cm FROM ruler_data WHERE file_name = ?", (filename,))
    ruler_info_list = cur.fetchall()

    conversion_means = []
    predicted_conversion_factor_cm = None

    # Iterate over the fetched data
    for ruler_info in ruler_info_list:
        conversion_mean = ruler_info[0] if ruler_info[0] is not None else 0
        predicted_conversion_factor_cm = ruler_info[1] if ruler_info[1] is not None else 0
        conversion_means.append(conversion_mean)

    # Check if all conversion_mean values are 0
    if all(cm == 0 for cm in conversion_means):
        if predicted_conversion_factor_cm:
            CF = predicted_conversion_factor_cm
        else:
            CF = 0
    else:
        # Calculate the average of non-zero conversion_mean values
        non_zero_conversion_means = [cm for cm in conversion_means if cm > 0]
        CF = sum(non_zero_conversion_means) / len(non_zero_conversion_means) if non_zero_conversion_means else 0

    CF = float("{:.2f}".format(CF))

    return CF




'''
def segment_images(logger, dir_home, dict_objects, leaf_type, dict_name_seg, dict_from, cfg, Project, Dirs, batch, n_batches, lock):#, start, end):
    # Run the leaf instance segmentation operations
    dir_seg_model = os.path.join(dir_home, 'leafmachine2', 'segmentation', 'models',cfg['leafmachine']['leaf_segmentation']['segmentation_model'])
    Instance_Detector = Detector_LM2(logger, dir_seg_model, cfg['leafmachine']['leaf_segmentation']['minimum_confidence_threshold'], leaf_type)
    # Instance_Detector_Partial = Detector_LM2(logger, dir_seg_model, cfg['leafmachine']['leaf_segmentation']['minimum_confidence_threshold'], 1)


    save_oriented_images = cfg['leafmachine']['leaf_segmentation']['save_oriented_images']
    save_keypoint_overlay = cfg['leafmachine']['leaf_segmentation']['save_keypoint_overlay']
    save_oriented_mask = cfg['leafmachine']['leaf_segmentation']['save_oriented_mask']
    save_simple_txt = cfg['leafmachine']['leaf_segmentation']['save_simple_txt']
    # Weights folder base
    dir_weights = os.path.join(dir_home, 'leafmachine2', 'keypoint_detector','keypoint_models')
    detector_version = cfg['leafmachine']['leaf_segmentation']['detector_version']
    weights =  os.path.join(dir_weights,detector_version,'weights','best.pt')

    # Create dictionary for overrides
    overrides = {
        'model': weights,
        # 'source': img_path,
        'name': detector_version,
        'boxes':False,
        'max_det':1,
        # 'visualize': True,
        # 'save_txt': True,
        # 'show':True
    }

    # Initialize PosePredictor
    Pose_Predictor = PosePredictor(weights, Dirs.dir_oriented_images, Dirs.dir_keypoint_overlay, 
                                   save_oriented_images=save_oriented_images, save_keypoint_overlay=save_keypoint_overlay, 
                                   overrides=overrides)
    
    generate_overlay = cfg['leafmachine']['leaf_segmentation']['generate_overlay']
    overlay_dpi = cfg['leafmachine']['leaf_segmentation']['overlay_dpi']
    bg_color = cfg['leafmachine']['leaf_segmentation']['overlay_background_color']
    keep_best = cfg['leafmachine']['leaf_segmentation']['keep_only_best_one_leaf_one_petiole']
    save_overlay_pdf = cfg['leafmachine']['leaf_segmentation']['save_segmentation_overlay_images_to_pdf']
    save_each_segmentation_overlay_image = cfg['leafmachine']['leaf_segmentation']['save_segmentation_overlay_images_to_pdf']
    save_individual_overlay_images = cfg['leafmachine']['leaf_segmentation']['save_individual_overlay_images']
    save_oriented_images = cfg['leafmachine']['leaf_segmentation']['save_oriented_images']

    save_ind_masks_color = cfg['leafmachine']['leaf_segmentation']['save_masks_color']
    # save_ind_masks_index = cfg['leafmachine']['leaf_segmentation']['save_masks_index']
    save_full_image_masks_color = cfg['leafmachine']['leaf_segmentation']['save_full_image_masks_color']
    # save_full_image_masks_index = cfg['leafmachine']['leaf_segmentation']['save_full_image_masks_index']
    use_efds_for_masks = cfg['leafmachine']['leaf_segmentation']['use_efds_for_png_masks']
    save_rgb_cropped_images = cfg['leafmachine']['leaf_segmentation']['save_rgb_cropped_images']

    filenames = []
    full_images = []
    full_masks = []

    seg_overlay_name = '_'.join([dict_name_seg,'Overlay'])
    seg_overlay = {}
    # seg_overlay_data = {}

    for filename, value in dict_objects.items(): # Whole image
        value[dict_name_seg] = []
        seg_overlay[filename] = []
        # seg_overlay_data[filename] = []

        try:
            # full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpg'])))
            full_image = cv2.cvtColor(cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpg']))), cv2.COLOR_BGR2RGB)
        except:
            # full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpeg'])))
            full_image = cv2.cvtColor(cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpeg']))), cv2.COLOR_BGR2RGB)

        CF = determine_conversion_factor(dict_objects, filename)

        full_size = full_image.shape
        if save_full_image_masks_color:
            # Create a black image
            full_mask = Image.new('RGB', (full_size[1], full_size[0]), color=(0, 0, 0))
        else: 
            full_mask = []

        if value[dict_from] is not []:
            for cropped in value[dict_from]: # Individual leaf
                for seg_name, img_cropped in cropped.items():
                    with lock:
                    
                        keypoint_data = {}
                        # print(seg_name)
                        logger.debug(f'segmenting - {seg_name}')

                        seg_name_short = seg_name.split("__")[2]
                        # cropped_overlay = []


                        # Segment!
                        # fig, out_polygons, out_bboxes, out_labels, out_color = Instance_Detector.segment(img_cropped, generate_overlay, overlay_dpi, bg_color)
                        # try:
                        out_polygons, out_bboxes, out_labels, out_color = Instance_Detector.segment(img_cropped, generate_overlay, overlay_dpi, bg_color)
                        keypoint_data = Pose_Predictor.process_images(img_cropped, filename=seg_name)
                        # print(keypoint_data)

                        # except:
                        #     detected_components = []
                        #     cropped_overlay = []
                        #     overlay_data = []
                        #     cropped_overlay_size = []
                        #     out_polygons = []
                        #     keypoint_data = []
                        
                        if len(out_polygons) > 0: # Success
                            if keep_best:
                                out_polygons, out_bboxes, out_labels, out_color = keep_rows(out_polygons, out_bboxes, out_labels, out_color, get_string_indices(out_labels))
                            
                            if (out_polygons is None) and (out_bboxes is None) and (out_labels is None) and (out_color is None):
                                detected_components = []
                                cropped_overlay = []
                                overlay_data = []
                                cropped_overlay_size = []
                            else:
                                # detected_components, cropped_overlay, cropped_overlay_oriented, overlay_data, new_width, new_height = create_overlay_and_calculate_props(keypoint_data, seg_name, img_cropped, out_polygons, out_labels, out_color, cfg)
                                detected_components, cropped_overlay, overlay_data = create_overlay_and_calculate_props(keypoint_data, seg_name, img_cropped, out_polygons, out_labels, out_color, cfg)
                                # full_image = create_insert_legacy(full_image, cropped_overlay, seg_name_short)
                                full_image = create_insert(full_image, overlay_data, seg_name_short, cfg)

                                cropped_overlay_size = cropped_overlay.shape
                                # cropped_overlay_oriented_size = cropped_overlay_oriented.shape
                                # cropped_overlay_oriented_size = (new_width, new_height)

                        else: # Fail
                            detected_components = []
                            cropped_overlay = []
                            overlay_data = []
                            cropped_overlay_size = []
                            keypoint_data = []
                            # cropped_overlay_oriented_size = []

                        # with lock:
                        value[dict_name_seg].append({seg_name: detected_components})#*************************** TODO see how to save some RAM
                        # seg_overlay[filename].append({seg_name: cropped_overlay}) #*************************** TODO
                        # seg_overlay_data[filename].append({seg_name: overlay_data})#*************************** TODO

                        save_rgb_cropped(save_rgb_cropped_images, seg_name, img_cropped, leaf_type, Dirs)

                        save_individual_segmentations(save_individual_overlay_images, dict_name_seg, seg_name, cropped_overlay, Dirs)

                        full_mask = save_masks_color(keypoint_data, save_oriented_images, save_ind_masks_color, save_full_image_masks_color, 
                                                     use_efds_for_masks, full_mask, overlay_data, cropped_overlay_size, full_size, seg_name, 
                                                     seg_name_short, leaf_type, Dirs, CF)

        save_full_masks(save_full_image_masks_color, full_mask, filename, leaf_type, Dirs)
        save_full_overlay_images(save_each_segmentation_overlay_image, full_image, filename, leaf_type, Dirs)

        filenames.append(filename)
        
        if save_overlay_pdf:
            full_images.append(full_image)

        if save_full_image_masks_color:
            full_masks.append(full_mask)
        
    save_full_image_segmentations(save_overlay_pdf, dict_name_seg, full_images, filenames, Dirs, cfg, batch, n_batches, lock)#, start, end)
    return dict_objects, None #seg_overlay
'''

def save_rgb_cropped(save_rgb_cropped_images, seg_name, img_cropped, leaf_type, Dirs):
    if save_rgb_cropped_images:
        if leaf_type == 0:
            cv2.imwrite(os.path.join(Dirs.whole_leaves, '.'.join([seg_name, 'jpg'])), img_cropped)
        elif leaf_type == 1:
            cv2.imwrite(os.path.join(Dirs.partial_leaves, '.'.join([seg_name, 'jpg'])), img_cropped)

##### For mask saving
def rotate_mask_using_keypoint_data(dir_out, seg_name, save_oriented_images, keypoint_data, img):
    # Handle rotation 
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    angle = keypoint_data[seg_name]['angle']
    oriented_mask = rotate_image(-angle, img_cv2, save_oriented_images)
    ### oriented_mask = Image.fromarray(cv2.cvtColor(oriented_mask, cv2.COLOR_BGR2RGB))
    ### oriented_mask.save(os.path.join(dir_out, '.'.join([seg_name, 'png'])))
    
    # cv2.imwrite(os.path.join(dir_out, '.'.join([seg_name, 'png'])), oriented_mask)

    # return oriented_mask, -angle

    # Find centroids of the petiole and leaf
    color_leaf = (0, 255, 46)
    color_petiole = (255, 173, 0)
    
    def find_centroid(color, img):
        mask = cv2.inRange(img, np.array(color), np.array(color))
        moments = cv2.moments(mask)
        if moments["m00"] != 0:
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
            return (centroid_x, centroid_y)
        return None

    leaf_centroid = find_centroid(color_leaf, oriented_mask)
    petiole_centroid = find_centroid(color_petiole, oriented_mask)

    if leaf_centroid and petiole_centroid and petiole_centroid[1] < leaf_centroid[1]:
        # Rotate the image 180 degrees if petiole is above leaf
        oriented_mask = cv2.rotate(oriented_mask, cv2.ROTATE_180)
        angle = 180 - angle

    # Save the oriented mask
    cv2.imwrite(os.path.join(dir_out, '.'.join([seg_name, 'png'])), oriented_mask)

    return oriented_mask, -angle

def find_unique_colors(image_np):
    # Find unique colors in the image (ignoring the alpha channel if it exists)
    if image_np.shape[2] == 4: # Check if image has an alpha channel
        unique_colors = {tuple(color[:3]) for color in np.unique(image_np.reshape(-1, 4), axis=0)}
    else:
        unique_colors = {tuple(color) for color in np.unique(image_np.reshape(-1, 3), axis=0)}
    return unique_colors

def segment_masks(unique_colors, image_np):
    # color_leaf = (46,255,0)
    # color_stem = (0,173,255)
    color_leaf = (0,255,46)
    color_petiole = (255,173,0)
    masks = {} # Dictionary to hold masks for each unique color
    mask_leaf = None
    has_leaf_color = False  # Initially assume no leaf color is found
    # Segment the image into masks based on unique colors
    for color in unique_colors:
        # Create a mask for the current color
        mask = np.all(image_np[:, :, :3] == color, axis=-1)
        masks[color] = Image.fromarray(mask.astype(np.uint8) * 255)
        # Additionally, if the color matches the predefined leaf color, set self.mask_leaf
        if color == color_leaf:
            mask_leaf = Image.fromarray(mask.astype(np.uint8) * 255)
            # Check if there are any pixels of color_leaf
            if np.any(mask):
                has_leaf_color = True
    return mask_leaf, masks, has_leaf_color
def is_clockwise(contour):
    s = 0
    coords = contour
    for i in range(len(coords)):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % len(coords)]
        s += (x2 - x1) * (y2 + y1)
    return s > 0
def find_color_centroid(rotated_img, color):
    # Create a mask for the specified color
    mask = cv2.inRange(rotated_img, color, color)
    # Find coordinates of all non-zero points (where the color is found)
    points = cv2.findNonZero(mask)
    if points is not None:
        # Compute the centroid of these points
        centroid = np.mean(points, axis=0)[0]  # Average x and y coordinates
        return np.array([centroid[0], centroid[1]])  # Ensure output is numpy array
    return None
# def apply_color_markers(tip_base_img, tip_raw, base_raw):
#     height, width, _ = tip_base_img.shape  # Get dimensions of the image
#     # size = 1  # This will create a 3x3 square around the point
#     size = 2  # This will create a 5x5 square around the point

#     # Apply the red color to the tip area
#     for i in range(-size, size + 1):
#         for j in range(-size, size + 1):
#             # Calculate the indices for the tip
#             x_tip, y_tip = tip_raw[0] + j, tip_raw[1] + i
#             # Check if the indices are within the image boundaries
#             if 0 <= x_tip < width and 0 <= y_tip < height:
#                 tip_base_img[y_tip, x_tip] = [0, 0, 255]  # BGR for red

#     # Apply the blue color to the base area
#     for i in range(-size, size + 1):
#         for j in range(-size, size + 1):
#             # Calculate the indices for the base
#             x_base, y_base = base_raw[0] + j, base_raw[1] + i
#             # Check if the indices are within the image boundaries
#             if 0 <= x_base < width and 0 <= y_base < height:
#                 tip_base_img[y_base, x_base] = [255, 0, 0]  # BGR for blue

#     return tip_base_img
def apply_color_markers(tip_base_img, tip_raw, base_raw):
    def is_within_bounds(point, size, max_width, max_height):
        # Check if a point with the given size extends out of image bounds
        x, y = point
        return not (x - size < 0 or x + size >= max_width or y - size < 0 or y + size >= max_height)

    def pad_image(img, pad_width, pad_height):
        # Pad the image on all sides with the specified width and height
        return cv2.copyMakeBorder(img, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    size = 2  # This will create a 5x5 square around the point
    height, width, _ = tip_base_img.shape  # Get dimensions of the image
    
    # Check if padding is needed for either point
    if not is_within_bounds(tip_raw, size, width, height) or not is_within_bounds(base_raw, size, width, height):
        # Calculate padding size as 10% of the maximum dimension
        pad_size = int(max(width, height) * 2)
        # Pad image
        tip_base_img = pad_image(tip_base_img, pad_size, pad_size)
        # Adjust points positions due to padding
        tip_raw = (tip_raw[0] + pad_size, tip_raw[1] + pad_size)
        base_raw = (base_raw[0] + pad_size, base_raw[1] + pad_size)
        # Update dimensions after padding
        height, width, _ = tip_base_img.shape

    # Apply the red color to the tip area
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            x_tip, y_tip = tip_raw[0] + j, tip_raw[1] + i
            if 0 <= x_tip < width and 0 <= y_tip < height:
                tip_base_img[y_tip, x_tip] = [0, 0, 255]  # BGR for red

    # Apply the blue color to the base area
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            x_base, y_base = base_raw[0] + j, base_raw[1] + i
            if 0 <= x_base < width and 0 <= y_base < height:
                tip_base_img[y_base, x_base] = [255, 0, 0]  # BGR for blue

    return tip_base_img

def create_perimeter_normalize(mask_leaf, keypoint_data, seg_name): 
    '''Returns a contour with all points, oriented vertically, first point is leaf tip, ordered clockwise'''
    # Convert the PIL Image to a NumPy array for processing
    mask_leaf_np = np.array(mask_leaf) / 255  # Normalize to 0 and 1
    # Perform binary erosion
    eroded_mask = binary_erosion(mask_leaf_np, structure=np.ones((3, 3)))
    # Subtract the eroded mask from the original mask to get the perimeter
    perimeter = mask_leaf_np - eroded_mask
    # Convert back to a PIL Image for display
    perimeter = Image.fromarray((perimeter * 255).astype(np.uint8))

    perimeter_image = mask_leaf_np
        # Convert to grayscale if it's not already (assumes the image could be in a different format)
    if len(perimeter_image.shape) == 3:  # Check if the image is colored
        perimeter_image = cv2.cvtColor(perimeter_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to smooth edges
    perimeter_image = cv2.GaussianBlur(perimeter_image, (11, 11), 3)
    # # Threshold after blurring to maintain the binary nature
    # _, perimeter_image = cv2.threshold(perimeter_image, 127, 255, cv2.THRESH_BINARY)
    # perimeter_image = cv2.bitwise_not(perimeter_image)
    
    # Convert image to uint8 if it's not already
    if perimeter_image.dtype != np.uint8:
        perimeter_image = (perimeter_image * 255).astype(np.uint8) if perimeter_image.max() <= 1 else perimeter_image.astype(np.uint8)


    # Find contours of the perimeter
    # contours, _ = cv2.findContours(perimeter_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(perimeter_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print("No contours found.")
        return None

    # Assuming the largest contour is the one we're interested in
    main_contour_raw = max(contours, key=cv2.contourArea)

    # Squeeze to remove extra dimension
    main_contour_raw = main_contour_raw.squeeze()

    # Select every nth point based on the initial value and adjust if necessary
    # select_every_n_pts = initial_select_every_n_pts

    # Unpack the x and y coordinates from reduced_contour
    x_coords_raw, y_coords_raw = main_contour_raw[:, 0], main_contour_raw[:, 1]

    # Calculate the maximum extent in either direction
    max_extent = max(x_coords_raw.max() - x_coords_raw.min(), y_coords_raw.max() - y_coords_raw.min())

    # Normalize the coordinates
    x_min = x_coords_raw.min()
    y_min =  y_coords_raw.min()
    x_coords_raw_normalized = (x_coords_raw - x_min) / max_extent
    y_coords_raw_normalized = (y_coords_raw - y_min) / max_extent

    # Calculate the centroid of the normalized shape
    centroid_x = np.mean(x_coords_raw_normalized)
    centroid_y = np.mean(y_coords_raw_normalized)

    # Center the shape about the origin by adjusting coordinates
    x_coords_centered = x_coords_raw_normalized - centroid_x
    y_coords_centered = y_coords_raw_normalized - centroid_y

    # Store the centered and normalized coordinates for further processing
    # normalized_coordinates = (x_coords_centered, y_coords_centered)

    # main_contour = np.column_stack((x_coords_raw_normalized, y_coords_raw_normalized))
    main_contour = np.column_stack((x_coords_centered, y_coords_centered))

    # Find the index of the minimum y-coordinate, which corresponds to the topmost point
    min_y_index = np.argmin(y_coords_centered)

    # bottom-most point
    max_y_index = np.argmax(y_coords_centered)

    ### rotate the tip/base
    # Calculate the closest points on the contour to the normalized and centered tip and base
    def find_closest_point(contour, point):
        distances = np.sqrt((contour[:, 0] - point[0])**2 + (contour[:, 1] - point[1])**2)
        min_index = np.argmin(distances)
        return contour[min_index], min_index
    # Create an image to mark the tip and base
    height, width = mask_leaf_np.shape[:2]
    tip_base_img = np.zeros((height, width, 3), dtype=np.uint8)
    # Assign colors to tip (red) and base (blue)
    tip_raw = np.array(keypoint_data[seg_name]['tip'], dtype=np.int32)
    base_raw = np.array(keypoint_data[seg_name]['base'], dtype=np.int32)
    angle = keypoint_data[seg_name]['angle']
    # Apply color markers
    tip_base_img = apply_color_markers(tip_base_img, tip_raw, base_raw)
    # Rotate the image with markers
    rotated_tip_base_img = rotate_image(-angle, tip_base_img, True)
    # Find centroids of the colored areas
    tip_centroid = find_color_centroid(rotated_tip_base_img, (0, 0, 255))  # BGR for red
    base_centroid = find_color_centroid(rotated_tip_base_img, (255, 0, 0))  # BGR for blue

    if tip_centroid is None or base_centroid is None:
        print("Tip or base centroid not found. Check rotation and color marking.")
        return [(999, 999)], [(999, 999)], 999, 999, 999, np.array([999,999]), np.array([999,999]), np.array([999,999]), np.array([999,999])  # Or handle the error in another appropriate way for your application

    # Convert to NumPy arrays if they are not already
    tip_centroid = np.array(tip_centroid)
    base_centroid = np.array(base_centroid)
    min_coords = np.array([x_coords_raw.min(), y_coords_raw.min()])
    if tip_centroid is not None and base_centroid is not None:
        # Normalize and center the tip and base keypoints
        tip_normalized = (tip_centroid - min_coords) / max_extent
        tip_centered = tip_normalized - np.array([centroid_x, centroid_y])
        base_normalized = (base_centroid - min_coords) / max_extent
        base_centered = base_normalized - np.array([centroid_x, centroid_y])
        closest_tip_point, tip_index = find_closest_point(main_contour, tip_centered)
        closest_base_point, base_index = find_closest_point(main_contour, base_centered)
    else:
        closest_tip_point = np.array([999,999])
        closest_base_point = np.array([999,999])

    # Retrieve the x and y coordinates of the topmost point using the index
    topmost_x = x_coords_centered[min_y_index]
    topmost_y = y_coords_centered[min_y_index]
    top = (topmost_x, topmost_y)

    bottommost_x = x_coords_centered[max_y_index]
    bottommost_y = y_coords_centered[max_y_index]
    bottom = (bottommost_x, bottommost_y)

    # Rotate the contour so that the topmost point is first
    # if not is_clockwise(main_contour):
    #     main_contour = np.flip(main_contour, axis=0)
    rotated_contour = np.roll(main_contour, -min_y_index, axis=0)

    ### Invert the y's to mirror the image about the x axis
    rotated_contour[:, 1] = -rotated_contour[:, 1]
    closest_tip_point[1] = -closest_tip_point[1]
    closest_base_point[1] = -closest_base_point[1]

    top = (top[0], -top[1])
    bottom = (bottom[0], -bottom[1])

    show_order = False
    show_colors = False # the black/blue indexing is broken but not worh massing around with 
    if show_order:
        # Plotting and saving the image for each point
        plt.figure(figsize=(8, 6))
        for i, (x, y) in enumerate(rotated_contour):
            plt.scatter(x, y, color='blue')
            plt.xlim([rotated_contour[:, 0].min(), rotated_contour[:, 0].max()])
            plt.ylim([rotated_contour[:, 1].min(), rotated_contour[:, 1].max()])
            plt.gca().set_aspect('equal', adjustable='box')
            # Only save the plot every 50 points
            if i % 200 == 0:
                plt.savefig('contour_plot_step_{}.png'.format(i))  # Save the figure
    if show_colors:
        plt.figure(figsize=(8, 6))
        # Iterate over each point in the contour
        for i, (x, y) in enumerate(rotated_contour):
            if i < tip_index or (i > base_index):
                color = 'blue'  # Before tip and after base
            elif tip_index <= i <= base_index:
                color = 'black'  # Between tip and base inclusive
            plt.scatter(x, y, color=color)
        # Highlight the closest tip point in green
        # plt.scatter(*closest_tip_point, color='green', s=100, label='Closest Tip')
        plt.scatter(*np.array(top), color='green', s=100, label='Closest Tip')
        # Highlight the closest base point in red
        plt.scatter(*closest_base_point, color='red', s=100, label='Closest Base')
        plt.xlim([rotated_contour[:, 0].min() - 0.1, rotated_contour[:, 0].max() + 0.1])
        plt.ylim([rotated_contour[:, 1].min() - 0.1, rotated_contour[:, 1].max() + 0.1])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.savefig('contour_plot.png')  # Save the figure
    return main_contour_raw, rotated_contour, max_extent, x_min, y_min, top, bottom, closest_tip_point, closest_base_point

def save_simple_txt(dir_simple_txt, rotated_contour, top, bottom, closest_tip_point, closest_base_point, angle, filename, 
                    full_size, CF, max_extent, x_min, y_min):
    # Construct the full path for the txt file
    file_path = os.path.join(dir_simple_txt, '.'.join([filename, 'txt']))
    
    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write the height and width of the original image
        file.write(f"{full_size[0]}\n") # height
        file.write(f"{full_size[1]}\n") # width

        # Write the conversion factor
        file.write(f"{CF}\n") # CF

        # Scaling factor
        file.write(f"{max_extent}\n") # scaling factor
        file.write(f"{x_min}\n") # scaling factor
        file.write(f"{y_min}\n") # scaling factor

        # Write the angle to the first line
        file.write(f"{angle}\n")
        # Write the topmost point to the second line
        file.write(f"{top[0]},{top[1]}\n")
        # Write the bottommost point to the third line
        file.write(f"{bottom[0]},{bottom[1]}\n")
        # Write the closest_tip_point point to the 4th line
        file.write(f"{closest_tip_point[0]},{closest_tip_point[1]}\n")
        # Write the closest_base_point point to the 5th line
        file.write(f"{closest_base_point[0]},{closest_base_point[1]}\n")
        # Write each coordinate pair from the rotated contour
        for x, y in rotated_contour:
            file.write(f"{x},{y}\n")

def save_raw_contour_txt(dir_raw_txt, raw_contour, seg_name, full_size, CF, angle):
    # Construct the full path for the raw txt file
    file_path = os.path.join(dir_raw_txt, f'{seg_name}.txt')
    bbox_str = seg_name.split('__L__')[-1].split('.')[0]
    bbox = tuple(map(int, bbox_str.split('-')))
    
    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write the height and width of the original image
        file.write(f"{full_size[0]}\n") # height
        file.write(f"{full_size[1]}\n") # width

        # Write the conversion factor
        file.write(f"{CF}\n") # CF

        # Write the box location from the original image
        file.write(f"{bbox[0]}\n")
        file.write(f"{bbox[1]}\n")
        file.write(f"{bbox[2]}\n")
        file.write(f"{bbox[3]}\n")

        # Write the angle to the first line
        file.write(f"{angle}\n")

        # Write each coordinate pair from the raw contour
        for x, y in raw_contour:
            file.write(f"{x},{y}\n")
#######

def save_masks_color(keypoint_data, save_oriented_images, save_individual_masks_color, save_full_image_masks_color, 
                     use_efds_for_masks, full_mask, overlay_data, cropped_overlay_size, full_size, seg_name, seg_name_short, 
                     leaf_type, Dirs, CF):
    if len(overlay_data) > 0:
        # unpack
        overlay_poly, overlay_poly_oriented, overlay_efd, overlay_rect, overlay_color = overlay_data

        if use_efds_for_masks:
            use_polys = overlay_efd
        else:
            use_polys = overlay_poly
        
        if save_individual_masks_color:

            ### Normal image
            # Create a black image
            img = Image.new('RGB', (cropped_overlay_size[1], cropped_overlay_size[0]), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)

            if use_polys != []:
                for i, poly in enumerate(use_polys):
                    this_color = overlay_color[i]
                    cls, this_color = next(iter(this_color.items()))
                    # Set the color for the polygon based on its class    
                    if leaf_type == 0:
                        if 'leaf' in cls:
                            color = [46, 255, 0]
                        elif 'petiole' in cls:
                            color = [0, 173, 255]
                        elif 'hole' in cls:
                            color = [209, 0, 255]
                        else:
                            color = [255, 255, 255]
                    elif leaf_type == 1:
                        if 'leaf' in cls:
                            color = [0, 200, 255]
                        elif 'petiole' in cls:
                            color = [255, 140, 0]
                        elif 'hole' in cls:
                            color = [200, 0, 255]
                        else:
                            color = [255, 255, 255]
                    # Draw the filled polygon on the image
                    draw.polygon(poly, fill=tuple(color))
            if leaf_type == 0:
                img.save(os.path.join(Dirs.segmentation_masks_color_whole_leaves, '.'.join([seg_name, 'png'])))

                # Handle rotation 
                if keypoint_data:
                    oriented_mask, angle = rotate_mask_using_keypoint_data(Dirs.dir_oriented_masks, seg_name, save_oriented_images, keypoint_data, img)

                    # Simple txt file
                    unique_colors = find_unique_colors(oriented_mask)
                    mask_leaf, masks, has_leaf_color = segment_masks(unique_colors, oriented_mask)
                    if has_leaf_color:
                        # TODO ********* add original image's height/width & conversion factor to both txts as the first 2 lines
                        raw_contour, rotated_contour, max_extent, x_min, y_min, top, bottom, closest_tip_point, closest_base_point = create_perimeter_normalize(mask_leaf, keypoint_data, seg_name) 
                        save_simple_txt(Dirs.dir_simple_txt, rotated_contour, top, bottom, closest_tip_point, closest_base_point, angle, seg_name, full_size, CF, max_extent, x_min, y_min)
                        save_raw_contour_txt(Dirs.dir_simple_raw_txt, raw_contour, seg_name, full_size, CF, angle)


            elif leaf_type == 1:
                img.save(os.path.join(Dirs.segmentation_masks_color_partial_leaves, '.'.join([seg_name, 'png'])))

                # Handle rotation 
                if keypoint_data:
                    oriented_mask, angle = rotate_mask_using_keypoint_data(Dirs.dir_oriented_masks, seg_name, save_oriented_images, keypoint_data, img)

                    # Simple txt file
                    unique_colors = find_unique_colors(oriented_mask)
                    mask_leaf, masks, has_leaf_color = segment_masks(unique_colors, oriented_mask)
                    if has_leaf_color:
                        raw_contour, rotated_contour, max_extent, x_min, y_min, top, bottom = create_perimeter_normalize(mask_leaf, keypoint_data, seg_name)
                        save_simple_txt(Dirs.dir_simple_txt, rotated_contour, top, bottom, closest_tip_point, closest_base_point, angle, seg_name, full_size, CF, max_extent, x_min, y_min)
                        save_raw_contour_txt(Dirs.dir_simple_raw_txt, raw_contour, seg_name, full_size, CF, angle)

        if save_full_image_masks_color:
            if '-' in seg_name_short:
                origin_x = int(seg_name_short.split('-')[0])
                origin_y = int(seg_name_short.split('-')[1])

                # Create a black image
                draw = ImageDraw.Draw(full_mask)

                if use_polys != []:
                    for i, poly in enumerate(use_polys):
                        this_color = overlay_color[i]
                        poly = [(x+origin_x, y+origin_y) for x, y in poly]
                        cls, this_color = next(iter(this_color.items()))
                        # Set the color for the polygon based on its class
                        if leaf_type == 0:
                            if 'leaf' in cls:
                                color = [46, 255, 0]
                            elif 'petiole' in cls:
                                color = [0, 173, 255]
                            elif 'hole' in cls:
                                color = [209, 0, 255]
                            else:
                                color = [255,255,255]
                        elif leaf_type == 1:
                            if 'leaf' in cls:
                                color = [0, 200, 255]
                            elif 'petiole' in cls:
                                color = [255, 140, 0]
                            elif 'hole' in cls:
                                color = [200, 0, 255]
                            else:
                                color = [255, 255, 255]
                        # Draw the filled polygon on the image
                        draw.polygon(poly, fill=tuple(color))
    return full_mask

def save_full_masks(save_full_image_masks_color, full_mask, filename, leaf_type, Dirs):
    if save_full_image_masks_color:
        if leaf_type == 0:
            full_mask.save(os.path.join(Dirs.segmentation_masks_full_image_color_whole_leaves, '.'.join([filename, 'png'])))
        elif leaf_type == 1:
            full_mask.save(os.path.join(Dirs.segmentation_masks_full_image_color_partial_leaves, '.'.join([filename, 'png'])))

def save_full_overlay_images(save_each_segmentation_overlay_image, full_image, filename, leaf_type, Dirs):
    if save_each_segmentation_overlay_image:
        if leaf_type == 0:
            try:
                full_image.save(os.path.join(Dirs.segmentation_overlay_whole_leaves, '.'.join([filename, 'jpg'])))
            except:
                full_image = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(Dirs.segmentation_overlay_whole_leaves, '.'.join([filename, 'jpg'])) , full_image)
        elif leaf_type == 1:
            try:
                full_image.save(os.path.join(Dirs.segmentation_overlay_partial_leaves, '.'.join([filename, 'jpg'])))
            except:
                full_image = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(Dirs.segmentation_overlay_partial_leaves, '.'.join([filename, 'jpg'])) , full_image)

    

def create_insert(full_image, overlay_data, seg_name_short, cfg):
    if '-' in seg_name_short:
        width = cfg['leafmachine']['leaf_segmentation']['overlay_line_width']
        origin_x = int(seg_name_short.split('-')[0])
        origin_y = int(seg_name_short.split('-')[1])
        # unpack
        overlay_poly, overlay_poly_oriented, overlay_efd, overlay_rect, overlay_color = overlay_data
        # fill_color = overlay_color[0][0]
        # outline_color =  overlay_color[0][1]

        # initialize
        full_image = np.asarray(full_image)
        full_image = Image.fromarray(full_image)
        draw = ImageDraw.Draw(full_image, "RGBA")

        if overlay_poly != []:
            for i, poly in enumerate(overlay_poly):
                this_color = overlay_color[i]
                key, this_color = next(iter(this_color.items()))
                # outline_color =(this_color[1][2],this_color[1][1],this_color[1][0])
                # fill_color = (this_color[0][2],this_color[0][1],this_color[0][0],this_color[0][3])
                outline_color = (this_color[1][2],this_color[1][1],this_color[1][0])
                fill_color = (this_color[0][2],this_color[0][1],this_color[0][0],this_color[0][3])
                poly = [(x+origin_x, y+origin_y) for x, y in poly]
                # poly = np.asarray(poly)
                # first_point = poly[0]
                # poly_closed = np.vstack((poly, first_point))
                draw.polygon(poly, fill=fill_color, outline=outline_color, width=width)

        if overlay_rect != []:
            for i, rect in enumerate(overlay_rect):
                this_color = overlay_color[i]
                key, this_color = next(iter(this_color.items()))
                outline_color = (this_color[1][2],this_color[1][1],this_color[1][0])

                rect = [(x+origin_x, y+origin_y) for x, y in rect]
                draw.polygon(rect, fill=None, outline=outline_color, width=width)
        
        if overlay_efd != []:
            for efd in overlay_efd:
                efd = [(x+origin_x, y+origin_y) for x, y in efd]
                draw.polygon(efd, fill=None, outline=(135,30,210), width=width)

    return full_image

def create_insert_legacy(full_image, cropped_overlay, seg_name_short):
    # Get the coordinates from the seg_name_short string
    x1, y1, x2, y2 = seg_name_short.split("-")
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    full_image[y1:y2, x1:x2] = cv2.cvtColor(cropped_overlay, cv2.COLOR_BGR2RGB)
    # cv2.imshow('full_image', full_image)
    # cv2.waitKey(0)
    return full_image

def save_individual_segmentations(save_individual_overlay_images, dict_name_seg, seg_name, cropped_overlay, Dirs):
    if save_individual_overlay_images:
        if dict_name_seg == "Segmentation_Whole_Leaf":
            cv2.imwrite(os.path.join(Dirs.segmentation_whole_leaves, '.'.join([seg_name, 'jpg'])), cropped_overlay)
        elif dict_name_seg == "Segmentation_Partial_Leaf":
            cv2.imwrite(os.path.join(Dirs.segmentation_partial_leaves, '.'.join([seg_name, 'jpg'])), cropped_overlay)

def save_full_image_segmentations(save_overlay_pdf, dict_name_seg, full_images, filenames, Dirs, cfg, batch, n_batches, lock):#, start, end):
    color_bg = cfg['leafmachine']['leaf_segmentation']['overlay_background_color']
    overlay_dpi = cfg['leafmachine']['leaf_segmentation']['overlay_dpi']

    if color_bg == 'black':
        color_text = 'white'
    else:
        color_text = 'black'

    '''if save_overlay_pdf:
        pdf_name = os.path.join(Dirs.segmentation_overlay_pdfs, ''.join([dict_name_seg, '_',str(batch+1), 'of',str(n_batches),'.pdf']))
        with PdfPages(pdf_name) as pdf:
            for idx, img in enumerate(full_images):
                # Create a new figure
                fig = plt.figure()'''
    if save_overlay_pdf:
        pdf_name = os.path.join(Dirs.segmentation_overlay_pdfs, ''.join([dict_name_seg, '_',str(batch+1), 'of',str(n_batches),'.pdf']))
        with PdfPages(pdf_name) as pdf:
            for idx, img in enumerate(full_images):
                # Acquire the lock before accessing the list
                with lock:
                    # Create a new figure
                    try:
                        fig = plt.figure()
                        fig.set_facecolor(color_bg)
                        plt.tight_layout(pad=0)
                        # plt.subplots_adjust(left=1, right=1, bottom=1, top=1)
                        # Add the image to the figure
                        plt.imshow(img)
                        # plt.annotate(xy=(0, 0), xycoords='axes fraction', fontsize=6,
                        #             xytext=(1, 1), textcoords='offset points',
                        #             ha='left', va='bottom')
                        plt.suptitle(filenames[idx], fontsize=10, y=0.95, color=color_text)
                        # Save the current figure to the PDF
                        pdf.savefig(fig, dpi=overlay_dpi)
                        plt.close()
                    except:
                        pass




def rotate_point(cx, cy, angle, px, py):
    """Rotate a point around a given center by an angle in degrees."""
    # Convert angle from degrees to radians
    radians = math.radians(angle)  # Positive angle for counterclockwise rotation
    # Translate point to origin
    ox, oy = px - cx, py - cy
    # Apply rotation
    qx = ox * math.cos(radians) - oy * math.sin(radians)  # Adjusted for image coordinate system
    qy = ox * math.sin(radians) + oy * math.cos(radians)
    # Translate back
    return qx + cx, qy + cy

def rotate_polygon_around_image_center(points, angle_degrees, img_width, img_height):
    """Rotate a list of (x, y) points around the center of an image."""
    cx, cy = img_width / 2, img_height / 2
    rotated_points = [rotate_point(cx, cy, angle_degrees, x, y) for x, y in points]
    
    # Determine the bounding box of the rotated points
    min_x = min(point[0] for point in rotated_points)
    max_x = max(point[0] for point in rotated_points)
    min_y = min(point[1] for point in rotated_points)
    max_y = max(point[1] for point in rotated_points)
    
    # Calculate new image dimensions to ensure all points fit
    new_width = max_x - min_x
    new_height = max_y - min_y

    # Translate points to fit in the new image dimensions
    translated_points = [(x - min_x, y - min_y) for x, y in rotated_points]
    
    return translated_points, new_width, new_height



def rotate_image(angle, orig_img, save_oriented_images):
    if save_oriented_images:
        # Calculate the center of the image and the image size
        image_center = tuple(np.array(orig_img.shape[1::-1]) / 2)
        height, width = orig_img.shape[:2]

        # Calculate the rotation matrix for the given angle
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

        # Calculate the sine and cosine of the rotation angle
        abs_cos = abs(rot_mat[0, 0])
        abs_sin = abs(rot_mat[0, 1])

        # Calculate the new bounding dimensions of the image
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # Adjust the rotation matrix to take into account translation
        rot_mat[0, 2] += bound_w / 2 - image_center[0]
        rot_mat[1, 2] += bound_h / 2 - image_center[1]

        # Perform the rotation, adjusting the canvas size
        rotated_img = cv2.warpAffine(orig_img, rot_mat, (bound_w, bound_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        return rotated_img


def create_image_from_coords(coords, filename='output_image.png'):
    # Use the coordinates to define the perimeter in an image
    min_x, min_y = np.min(coords, axis=0)
    max_x, max_y = np.max(coords, axis=0)
    size = (max_x - min_x + 1, max_y - min_y + 1)
    
    # Create a new image filled with white
    image = Image.new('RGB', size, color=(255, 255, 255))
    
    # Create a draw object
    draw = ImageDraw.Draw(image)
    
    # Adjust coords for the image based on the min_x, min_y
    adjusted_coords = [(x - min_x, y - min_y) for x, y in coords]
    
    # Draw a polygon filled with a specific color using the adjusted coordinates
    draw.polygon(adjusted_coords, fill=(46, 255, 0))
    
    # Save the figure
    plt.figure()
    plt.imshow(image)
    plt.axis('off')  # Turn off the axis
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the plt object to free up resources
    print('hi')

def create_overlay_and_calculate_props(keypoint_data, seg_name, img_cropped, out_polygons, out_labels, out_color, cfg):
    width = cfg['leafmachine']['leaf_segmentation']['overlay_line_width']
    do_get_efds = cfg['leafmachine']['leaf_segmentation']['calculate_elliptic_fourier_descriptors']
    
    cropped_overlay = img_cropped
    # cropped_overlay = Image.fromarray(cv2.cvtColor(cropped_overlay, cv2.COLOR_BGR2RGB))
    cropped_overlay = Image.fromarray(cropped_overlay)
    draw = ImageDraw.Draw(cropped_overlay, "RGBA")

    # parse seg_name
    seg_name_short = seg_name.split("__")[2]

    # List of instances
    detected_components = []
    overlay_poly = []
    overlay_poly_oriented = []
    overlay_efd = []
    overlay_rect = []
    overlay_color = []
    for i, polys in enumerate(out_polygons):
        # PIL
        # fill_color = (color_rgb[2], color_rgb[1], color_rgb[0], 127)
        # outline_color = (color_rgb[2], color_rgb[1], color_rgb[0])

        # cv2
        color_rgb = tuple(map(lambda x: int(x*255), out_color[i]))
        fill_color = (color_rgb[0], color_rgb[1], color_rgb[2], 127)
        outline_color = (color_rgb[0], color_rgb[1], color_rgb[2])

        max_poly = get_largest_polygon(polys)#, value['height'], value['width'])
        # create_image_from_coords(max_poly)
        # max_poly_oriented, new_width, new_height = rotate_polygon_around_image_center(max_poly, angle, img_cropped.shape[1], img_cropped.shape[0])
        # cropped_overlay = plot_polygons_on_image(max_poly, cropped_overlay, color_rgb)#, 0.4) # cv2 implementation

        # rotated_img = Image.fromarray(cv2.cvtColor(rotate_image(angle, img_cropped, save_oriented_images), cv2.COLOR_BGR2RGB))
        # draw_oriented = ImageDraw.Draw(rotated_img, "RGBA")

        # calculate props
        if max_poly is None:
            component = None
        else:
            component, bbox = polygon_properties(max_poly, out_labels[i], seg_name_short, cfg, img_cropped)
            detected_components.append(component)

            # draw poly
            draw.polygon(max_poly, fill=fill_color, outline=outline_color, width=width)
            draw.polygon(bbox, outline=outline_color, width=width)
            overlay_rect.append(bbox)
            overlay_poly.append(max_poly)
            overlay_color.append({out_labels[i]: [fill_color, outline_color]})

            # draw_oriented
            # draw_oriented.polygon(max_poly_oriented, fill=fill_color, outline=outline_color, width=width)
            # draw_oriented.polygon(bbox, outline=outline_color, width=width)
            # overlay_rect.append(bbox)
            # overlay_poly_oriented.append(max_poly_oriented)
            # overlay_color.append({out_labels[i]: [fill_color, outline_color]})

            if '__L__' in seg_name:
                if do_get_efds:
                    _, value = next(iter(component.items()))
                    efd = value['efds']['efd_pts_PIL']
                    # efd = efd['pts_efd']
                    draw.polygon(efd, fill=None, outline=(210,30,135), width=width)
                    overlay_efd.append(efd)
            elif '__PL__' in seg_name:
                if do_get_efds:
                    _, value = next(iter(component.items()))
                    efd = value['efds']['efd_pts_PIL']
                    # efd = efd['pts_efd']
                    draw.polygon(efd, fill=None, outline=(255,120,0), width=width)
                    overlay_efd.append(efd)
    # PIL               
    # cropped_overlay.show() # wrong colors without changing to RGB
    # cv2
    cropped_overlay = np.array(cropped_overlay)
    # cropped_overlay_oriented = np.array(rotated_img)
    overlay_data = [overlay_poly, overlay_poly_oriented, overlay_efd, overlay_rect, overlay_color]
    # cv2.imshow('img_crop', cropped_overlay)
    # cv2.waitKey(0)
    return detected_components, cropped_overlay, overlay_data#, cropped_overlay_oriented, overlay_data



# def crop_images_to_bbox(dict, cls, dict_name_cropped, dict_from, Project):
#     # For each image, iterate through the whole leaves, segment, report data back to dict_plant_components
#     for filename, value in dict.items():
#         value[dict_name_cropped] = []
#         if dict_from in value:
#             bboxes_whole_leaves = [val for val in value[dict_from] if val[0] == convert_index_to_class(cls)]
#             if len(bboxes_whole_leaves) == 0:
#                 m = str(''.join(['No objects for class ', convert_index_to_class(0), ' were found']))
#                 # Print_Verbose(cfg, 3, m).plain()
#             else:
#                 try:
#                     img = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpg'])))
#                     # img = cv2.imread(os.path.join(Project, '.'.join([filename,'jpg']))) # Testing
#                 except:
#                     img = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpeg'])))
#                     # img = cv2.imread(os.path.join(Project, '.'.join([filename,'jpeg']))) # Testing
                
#                 for d in bboxes_whole_leaves:
#                     # img_crop = img.crop((d[1], d[2], d[3], d[4])) # PIL
#                     img_crop = img[d[2]:d[4], d[1]:d[3]]
#                     loc = '-'.join([str(d[1]), str(d[2]), str(d[3]), str(d[4])])
#                     if cls == 0:
#                         crop_name = '__'.join([filename,'L',loc])
#                     elif cls == 1:
#                         crop_name = '__'.join([filename,'PL',loc])
#                     value[dict_name_cropped].append({crop_name: img_crop})
#                     # cv2.imshow('img_crop', img_crop)
#                     # cv2.waitKey(0)
#                     # img_crop.show() # PIL
#     return dict
def crop_images_to_bbox(ProjectSQL, filename, cls, dict_name_cropped, dict_from):
    conn = ProjectSQL.conn
    cur = conn.cursor()

    # Retrieve bounding boxes from the SQL database
    cur.execute(f"SELECT x_min, y_min, x_max, y_max FROM {dict_from} WHERE file_name = ? AND class = ?", (filename, cls))
    bboxes = cur.fetchall()

    # Try to load the image
    try:
        img_path = glob.glob(os.path.join(ProjectSQL.dir_images, f"{filename}.*"))[0]
        img = cv2.imread(img_path)
    except:
        img = None

    if img is None:
        return

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        img_crop = img[y_min:y_max, x_min:x_max]
        loc = '-'.join(map(str, [x_min, y_min, x_max, y_max]))
        crop_name = f"{filename}__{'L' if cls == 0 else 'PL'}__{loc}"

        # Store the cropped image in the SQL database (as a BLOB)
        _, img_encoded = cv2.imencode('.jpg', img_crop)
        cur.execute(f"INSERT INTO {dict_name_cropped} (file_name, crop_name, cropped_image) VALUES (?, ?, ?)", 
                    (filename, crop_name, img_encoded.tobytes()))
    conn.commit()


# def unpack_class_from_components(dict, cls, dict_name_yolo, dict_name_location, Project):
#     # Get the dict that contains plant parts, find the whole leaves
#     for filename, value in dict.items():
#         if "Detections_Plant_Components" in value:
#             filtered_components = [val for val in value["Detections_Plant_Components"] if val[0] == cls]
#             value[dict_name_yolo] = filtered_components

#     for filename, value in dict.items():
#         if "Detections_Plant_Components" in value:
#             filtered_components = [val for val in value["Detections_Plant_Components"] if val[0] == cls]
#             height = value['height']
#             width = value['width']
#             converted_list = [[convert_index_to_class(val[0]), int((val[1] * width) - ((val[3] * width) / 2)), 
#                                                                 int((val[2] * height) - ((val[4] * height) / 2)), 
#                                                                 int(val[3] * width) + int((val[1] * width) - ((val[3] * width) / 2)), 
#                                                                 int(val[4] * height) + int((val[2] * height) - ((val[4] * height) / 2))] for val in filtered_components]
#             # Verify that the crops are correct
#             # img = Image.open(os.path.join(Project., '.'.join([filename,'jpg'])))
#             # for d in converted_list:
#             #     img_crop = img.crop((d[1], d[2], d[3], d[4]))
#             #     img_crop.show() 
#             value[dict_name_location] = converted_list
#     # print(dict)
#     return dict
def unpack_class_from_components(ProjectSQL, filename, cls, dict_name_yolo, dict_name_location):
    conn = ProjectSQL.conn
    cur = conn.cursor()

    # Get the width and height from the images table
    cur.execute("SELECT width, height FROM images WHERE name = ?", (filename,))
    width, height = cur.fetchone()

    # Retrieve plant annotations for the given filename and class
    cur.execute("SELECT annotation FROM annotations_plant WHERE file_name = ? AND component = 'Detections_Plant_Components'", (filename,))
    plant_annotations = cur.fetchall()

    for annotation in plant_annotations:
        # Process the annotation data to extract bounding box coordinates
        class_index, x_center, y_center, bbox_width, bbox_height = map(float, annotation[0].split(','))

        if int(class_index) == cls:
            x_min = int(x_center * width - (bbox_width * width / 2))
            y_min = int(y_center * height - (bbox_height * height / 2))
            x_max = int(x_center * width + (bbox_width * width / 2))
            y_max = int(y_center * height + (bbox_height * height / 2))

            # Insert the processed bounding box into the correct table (Whole_Leaf_BBoxes or Partial_Leaf_BBoxes)
            cur.execute(f"INSERT INTO {dict_name_location} (file_name, class, x_min, y_min, x_max, y_max) VALUES (?, ?, ?, ?, ?, ?)",
                        (filename, cls, x_min, y_min, x_max, y_max))

    conn.commit()




def plot_polygons_on_image(polygons, img, color):
    for polygon in polygons:
        # convert the list of points to a numpy array of shape Nx1x2
        polygon = np.array(polygon, dtype=np.int32).reshape((-1,1,2))
        # draw the polygon on the image
        cv2.polylines(img, [polygon], True, color, 2)
    # show the image with the polygons
    # cv2.imshow("Polygons", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img


def convert_index_to_class(ind):
    mapping = {
        0: 'Leaf_WHOLE',
        1: 'Leaf_PARTIAL',
        2: 'Leaflet',
        3: 'Seed_Fruit_ONE',
        4: 'Seed_Fruit_MANY',
        5: 'Flower_ONE',
        6: 'Flower_MANY',
        7: 'Bud',
        8: 'Specimen',
        9: 'Roots',
        10: 'Wood'
    }
    return mapping.get(ind, 'Invalid class').lower()



''''''
if __name__ == '__main__':
    with open('D:/Dropbox/LM2_Env/Image_Datasets/TEST_LM2/TEST_2023_01_24__16-03-18/Plant_Components/json/Plant_Components.json') as json_file:
        dict_plant_components = json.load(json_file)
    segment_leaves([], 'D:/Dropbox/LeafMachine2', 'D:/Dropbox/LM2_Env/Image_Datasets/SET_Acacia/Images_GBIF_Acacia_Prickles', [], dict_plant_components)
''''''