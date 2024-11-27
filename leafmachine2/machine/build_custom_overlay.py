import os, json, random, inspect, sys, cv2, itertools, sqlite3, ast, math
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass, field
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from time import perf_counter
from threading import Lock
from multiprocessing import Queue, Process, Lock

currentdir = os.path.dirname(inspect.getfile(inspect.currentframe()))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(currentdir)
sys.path.append(parentdir)
# from segmentation.detectron2.segment_leaves import create_insert

import concurrent.futures
from time import perf_counter
import sqlite3
from PIL import Image
import os
from tqdm import tqdm

def build_custom_overlay_parallel(cfg, time_report, logger, dir_home, ProjectSQL, batch, Dirs):
    start_t = perf_counter()
    logger.name = f'[BATCH {batch+1} Build Custom Overlay]'
    logger.info(f'Creating overlay for batch {batch+1}')

    # Configuration settings
    line_w_archival = cfg['leafmachine']['overlay']['line_width_archival']
    line_w_plant = cfg['leafmachine']['overlay']['line_width_plant']
    show_archival = cfg['leafmachine']['overlay']['show_archival_detections']
    ignore_archival = cfg['leafmachine']['overlay']['ignore_archival_detections_classes']
    show_plant = cfg['leafmachine']['overlay']['show_plant_detections']
    ignore_plant = cfg['leafmachine']['overlay']['ignore_plant_detections_classes']
    show_segmentations = cfg['leafmachine']['overlay']['show_segmentations']
    show_landmarks = cfg['leafmachine']['overlay']['show_landmarks']
    ignore_landmarks = cfg['leafmachine']['overlay']['ignore_landmark_classes']

    # Set the number of workers (processes)
    num_workers = int(cfg['leafmachine']['project']['num_workers_overlay']) if cfg['leafmachine']['project']['num_workers_overlay'] is not None else 12

    logger.info(f'Process ID: {os.getpid()} - Starting overlay creation for batch {batch+1}')

    # Fetch file names from the images table for the current batch
    conn = sqlite3.connect(ProjectSQL.database)
    cur = conn.cursor()

    cur.execute("""
        SELECT name
        FROM images
        WHERE id IN (
            SELECT id
            FROM images
            WHERE valid = 1
            LIMIT ? OFFSET ?
        );
    """, (ProjectSQL.batch_size, batch * ProjectSQL.batch_size))

    project_data = cur.fetchall()
    conn.close()

    # Calculate dynamic batch size based on the number of workers
    num_files = len(project_data)

    # Calculate batch size and ensure all files are processed
    batch_size = math.ceil(num_files / num_workers)

    logger.info(f"Total number of images: {num_files}, Number of workers: {num_workers}, Dynamic batch size: {batch_size}")

    # Split the files into smaller groups (file_batches) based on the calculated batch size
    file_batches = [project_data[i:i + batch_size] for i in range(0, num_files, batch_size)]

    # # Ensure all images are processed, even if batch size is not divisible by num_workers
    # if len(project_data) % num_workers != 0:
    #     file_batches[-1].extend(project_data[num_workers * batch_size:])

    dir_images = ProjectSQL.dir_images  # Image directory for workers

    # Use ProcessPoolExecutor for parallelism
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_file_batch, batch, ProjectSQL.database, dir_images, line_w_archival, show_archival, ignore_archival, line_w_plant, show_plant, ignore_plant, show_segmentations, show_landmarks, cfg, Dirs)
            for batch in file_batches
        ]

        # Use tqdm to track progress of the futures
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing File Batches", colour='green'):
            pass  # We do not need to collect results, just track the completion

    end_t = perf_counter()
    t_overlay = f"[Batch {batch+1}: Build Custom Overlay elapsed time] {round(end_t - start_t)} seconds ({round((end_t - start_t)/60)} minutes)"
    logger.info(t_overlay)
    time_report['t_overlay'] = t_overlay

    return time_report


def process_file_batch(filenames, database, dir_images, line_w_archival, show_archival, ignore_archival, line_w_plant, show_plant, ignore_plant, show_segmentations, show_landmarks, cfg, Dirs):
    """Process a batch of files in parallel and return results."""
    batch_results = []

    for (filename,) in filenames:
        # Process the file and add to batch results
        result = process_file(database, dir_images, filename, line_w_archival, show_archival, ignore_archival, line_w_plant, show_plant, ignore_plant, show_segmentations, show_landmarks, cfg, Dirs)
        if result:
            batch_results.append(result)
        # else:
            # print(f"NO RESULTS FOR {filename}")

    return batch_results


def process_file(database, dir_images, filename, line_w_archival, show_archival, ignore_archival, line_w_plant, show_plant, ignore_plant, show_segmentations, show_landmarks, cfg, Dirs):
    # Open a new SQLite connection for each process
    conn = sqlite3.connect(database)
    cur = conn.cursor()

    # Initialize the dictionary to store the analysis data
    analysis = {}

    try:
        # Fetch image dimensions from the `images` table
        cur.execute("""
            SELECT width, height
            FROM images
            WHERE name = ?;
        """, (filename,))
        image_data = cur.fetchone()
        if image_data:
            analysis['width'], analysis['height'] = image_data
        else:
            analysis['width'], analysis['height'] = 0, 0

        # Fetch all archival component detections from the `annotations_archival` table
        cur.execute("""
            SELECT annotation
            FROM annotations_archival
            WHERE file_name = ?;
        """, (filename,))
        archival_data = cur.fetchall()
        analysis['Detections_Archival_Components'] = [row[0] for row in archival_data] if archival_data else []

        # Fetch all plant component detections from the `annotations_plant` table
        cur.execute("""
            SELECT annotation
            FROM annotations_plant
            WHERE file_name = ?;
        """, (filename,))
        plant_data = cur.fetchall()
        analysis['Detections_Plant_Components'] = [row[0] for row in plant_data] if plant_data else []

        # Fetch all segmentation data for whole leaves from the `Segmentation_Whole_Leaf` table
        if cfg['leafmachine']['leaf_segmentation']['segment_whole_leaves']:
            cur.execute("""
                SELECT *
                FROM Segmentation_Whole_Leaf
                WHERE file_name = ?;
            """, (filename,))
            seg_whole_leaf_data = cur.fetchall()
            # Store all columns in the analysis dictionary
            analysis['Segmentation_Whole_Leaf'] = seg_whole_leaf_data if seg_whole_leaf_data else []
        else:
            analysis['Segmentation_Whole_Leaf'] = []

        # Fetch all segmentation data for partial leaves from the `Segmentation_Partial_Leaf` table
        if cfg['leafmachine']['leaf_segmentation']['segment_partial_leaves']:
            cur.execute("""
                SELECT *
                FROM Segmentation_Partial_Leaf
                WHERE file_name = ?;
            """, (filename,))
            seg_partial_leaf_data = cur.fetchall()
            # Store all columns in the analysis dictionary
            analysis['Segmentation_Partial_Leaf'] = seg_partial_leaf_data if seg_partial_leaf_data else []
        else:
            analysis['Segmentation_Partial_Leaf'] = []


        # Fetch all landmark data for whole leaves from the `Landmarks_Whole_Leaves` table
        if cfg['leafmachine']['landmark_detector']['landmark_whole_leaves']:
            cur.execute("""
                SELECT *
                FROM Landmarks_Whole_Leaves
                WHERE file_name = ?;
            """, (filename,))
            landmarks_whole_data = cur.fetchall()
            columns = [desc[0] for desc in cur.description]  # Get column names
            analysis['Landmarks_Whole_Leaves'] = [dict(zip(columns, row)) for row in landmarks_whole_data] if landmarks_whole_data else []
        else:
            analysis['Landmarks_Whole_Leaves'] = []

        # Fetch all landmark data for partial leaves from the `Landmarks_Partial_Leaves` table
        if cfg['leafmachine']['landmark_detector']['landmark_partial_leaves']:
            cur.execute("""
                SELECT *
                FROM Landmarks_Partial_Leaves
                WHERE file_name = ?;
            """, (filename,))
            landmarks_partial_data = cur.fetchall()
            columns = [desc[0] for desc in cur.description]  # Get column names
            analysis['Landmarks_Partial_Leaves'] = [dict(zip(columns, row)) for row in landmarks_partial_data] if landmarks_partial_data else []
        else:
            analysis['Landmarks_Partial_Leaves'] = []


        # Fetch all segmentation data for partial leaves from the `Segmentation_Partial_Leaf` table
        if cfg['leafmachine']['leaf_segmentation']['save_oriented_images'] or cfg['leafmachine']['leaf_segmentation']['save_oriented_mask']:
            cur.execute("""
                SELECT *
                FROM Keypoints_Data
                WHERE file_name = ?;
            """, (filename,))
            keypoints_data = cur.fetchall()
            # Store all columns in the analysis dictionary
            analysis['Keypoints_Data'] = keypoints_data if keypoints_data else []
        else:
            analysis['Keypoints_Data'] = []



        # Fetch all ruler information from the `ruler_data` table
        cur.execute("""
            SELECT file_name, ruler_image_name, conversion_mean, predicted_conversion_factor_cm
            FROM ruler_data
            WHERE file_name = ?;
        """, (filename,))
        ruler_data = cur.fetchall()

        # Store the fetched data in the analysis dictionary
        analysis['Ruler_Info'] = [{
                                'file_name': row[0], 
                                'ruler_image_name': row[1], 
                                'conversion_mean': row[2],
                                'predicted_conversion_factor_cm': row[3]} 
                                for row in ruler_data] if ruler_data else []
        print(f"{filename} has {analysis['Ruler_Info']}")
    except sqlite3.DatabaseError as db_err:
        print(f"Database error: {db_err}")
        # Handle database-related errors (e.g., missing records, connectivity issues)

    except FileNotFoundError as fnf_err:
        print(f"File not found: {fnf_err}")
        # Handle issues related to file paths

    except Exception as e:
        print(f"Unexpected error: {e}")
        # Catch any other unexpected errors

    finally:
        # Ensure the database connection is closed
        conn.close()

    # Process the image (this includes archival and plant detections, segmentations, etc.)
    image_overlay = None
    try:
        image_overlay = Image.open(os.path.join(dir_images, f"{filename}.jpg"))
    except FileNotFoundError:
        image_overlay = Image.open(os.path.join(dir_images, f"{filename}.jpeg"))
    except Exception as e:
        print(f"Error opening image file {filename}: {e}")
        return

    if image_overlay.mode != 'RGB':
        image_overlay = image_overlay.convert('RGB')

    # Process the image with various overlay functions
    image_overlay = add_archival_detections(image_overlay, analysis['Detections_Archival_Components'], analysis['height'], analysis['width'], analysis['Ruler_Info'], line_w_archival, show_archival, ignore_archival, cfg)
    image_overlay = add_plant_detections(image_overlay, analysis['Detections_Plant_Components'], analysis['height'], analysis['width'], line_w_plant, show_plant, ignore_plant, cfg)
    image_overlay = add_segmentations(image_overlay, analysis['Segmentation_Whole_Leaf'], analysis['Segmentation_Partial_Leaf'], show_segmentations, cfg)
    # image_overlay = add_landmarks(image_overlay, analysis['Landmarks_Whole_Leaves'], analysis['Landmarks_Partial_Leaves'], show_landmarks, cfg)
    image_overlay = add_keypoints(image_overlay, keypoints_data, True, cfg)

    # Process ruler images if available
    # ruler_img = get_ruler_images(analysis['Ruler_Info'], cfg)

    # Save the overlay image
    save_overlay_images_to_jpg(image_overlay, filename, Dirs, cfg)

    # return analysis['Ruler_Info']








# def build_custom_overlay_parallel(cfg, time_report, logger, dir_home, ProjectSQL, batch, Dirs):
#     start_t = perf_counter()
#     logger.name = f'[BATCH {batch+1} Build Custom Overlay]'
#     logger.info(f'Creating overlay for batch {batch+1}')

#     line_w_archival = cfg['leafmachine']['overlay']['line_width_archival']
#     line_w_plant = cfg['leafmachine']['overlay']['line_width_plant']
#     show_archival = cfg['leafmachine']['overlay']['show_archival_detections']
#     ignore_archival = cfg['leafmachine']['overlay']['ignore_archival_detections_classes']
#     show_plant = cfg['leafmachine']['overlay']['show_plant_detections']
#     ignore_plant = cfg['leafmachine']['overlay']['ignore_plant_detections_classes']
#     show_segmentations = cfg['leafmachine']['overlay']['show_segmentations']
#     show_landmarks = cfg['leafmachine']['overlay']['show_landmarks']
#     ignore_landmarks = cfg['leafmachine']['overlay']['ignore_landmark_classes']

#     lock = Lock()  # Create a lock obj

#     if cfg['leafmachine']['project']['num_workers'] is None:
#         num_workers = 1
#     else:
#         num_workers = int(cfg['leafmachine']['project']['num_workers'])

#     filenames = []
#     overlay_images = []
#     ruler_images = []
#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
#         futures = []
#         for filename, analysis in Project.project_data_list[batch].items():
#             logger.info(f'Creating overlay for {filename}')
#             futures.append(executor.submit(process_file, Project, filename, analysis, line_w_archival, show_archival, ignore_archival, line_w_plant, show_plant, ignore_plant, show_segmentations, show_landmarks, cfg, Dirs, lock))  # Pass the lock obj to the process_file function

#         logger.info(f'Merging results from {num_workers} workers')
#         for future in concurrent.futures.as_completed(futures):
#             filename, image_overlay, ruler_img = future.result()
#             # save_overlay_images_to_jpg(image_overlay, filename, Dirs, cfg)  # Use the lock obj when writing to the file
#             filenames.append(filename)
#             overlay_images.append(image_overlay)
#             ruler_images.append(ruler_img)

#     logger.info(f'Saving batch {batch+1} overlay images to PDF')
#     save_custom_overlay_to_PDF(filenames, overlay_images, ruler_images, batch, Dirs, Project, cfg)
#     end_t = perf_counter()

#     t_overlay = f"[Batch {batch+1}: Build Custom Overlay elapsed time] {round(end_t - start_t)} seconds ({round((end_t - start_t)/60)} minutes)"
#     logger.info(t_overlay)
#     time_report['t_overlay'] = t_overlay
#     return time_report

########## replaced with process/queue
# def build_custom_overlay_parallel(cfg, time_report, logger, dir_home, ProjectSQL, batch, Dirs):
#     start_t = perf_counter()
#     logger.name = f'[BATCH {batch+1} Build Custom Overlay]'
#     logger.info(f'Creating overlay for batch {batch+1}')

#     line_w_archival = cfg['leafmachine']['overlay']['line_width_archival']
#     line_w_plant = cfg['leafmachine']['overlay']['line_width_plant']
#     show_archival = cfg['leafmachine']['overlay']['show_archival_detections']
#     ignore_archival = cfg['leafmachine']['overlay']['ignore_archival_detections_classes']
#     show_plant = cfg['leafmachine']['overlay']['show_plant_detections']
#     ignore_plant = cfg['leafmachine']['overlay']['ignore_plant_detections_classes']
#     show_segmentations = cfg['leafmachine']['overlay']['show_segmentations']
#     show_landmarks = cfg['leafmachine']['overlay']['show_landmarks']
#     ignore_landmarks = cfg['leafmachine']['overlay']['ignore_landmark_classes']

#     lock = Lock()

#     num_workers = int(cfg['leafmachine']['project']['num_workers']) if cfg['leafmachine']['project']['num_workers'] is not None else 1

#     logger.info(f'Thread ID: {threading.get_ident()} - Starting overlay creation for batch {batch+1}')

#     # Fetch file names from the images table for the current batch
#     conn = sqlite3.connect(ProjectSQL.database)
#     cur = conn.cursor()
    
#     cur.execute("""
#         SELECT name
#         FROM images
#         WHERE id IN (
#             SELECT id
#             FROM images
#             WHERE valid = 1
#             LIMIT ? OFFSET ?
#         );
#     """, (ProjectSQL.batch_size, batch * ProjectSQL.batch_size))
    
#     project_data = cur.fetchall()
#     conn.close()

#     filenames = []
#     overlay_images = []
#     ruler_images = []

#     # for (filename,) in project_data:  # Fetch the filename from the tuple
#     #     logger.info(f'Creating overlay for {filename}')
#     #     result = process_file(ProjectSQL.database, ProjectSQL.dir_images, filename, line_w_archival, show_archival, ignore_archival, line_w_plant, show_plant, ignore_plant, show_segmentations, show_landmarks, cfg, Dirs)
            
#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor: # num_workers
#         futures = []
#         for (filename,) in project_data:  # Fetch the filename from the tuple
#             logger.info(f'Creating overlay for {filename}')
#             futures.append(executor.submit(process_file, ProjectSQL.database, ProjectSQL.dir_images, filename, line_w_archival, show_archival, ignore_archival, line_w_plant, show_plant, ignore_plant, show_segmentations, show_landmarks, cfg, Dirs, lock))  # Pass the lock obj to the process_file function

#         logger.info(f'Merging results from {num_workers} workers')



#         # for future in concurrent.futures.as_completed(futures):
#             # filename, image_overlay = future.result()
#             # filenames.append(filename)
#             # overlay_images.append(image_overlay)
#             # ruler_images.append(ruler_img)

#     # logger.info(f'Saving batch {batch+1} overlay images to PDF')
#     # save_custom_overlay_to_PDF(filenames, overlay_images, ruler_images, batch, Dirs, ProjectSQL, cfg)
    
#     end_t = perf_counter()
#     t_overlay = f"[Batch {batch+1}: Build Custom Overlay elapsed time] {round(end_t - start_t)} seconds ({round((end_t - start_t)/60)} minutes)"
#     logger.info(t_overlay)
#     time_report['t_overlay'] = t_overlay
    
#     return time_report

# def process_file(Project, filename, analysis, line_w_archival, show_archival, ignore_archival, line_w_plant, show_plant, ignore_plant, show_segmentations, show_landmarks, cfg, Dirs, lock):

#     if 'height' in analysis:
#         height = analysis['height']
#     else:
#         height = 0

#     if 'width' in analysis:
#         width = analysis['width']
#     else:
#         width = 0

#     if 'Detections_Archival_Components' in analysis:
#         archival = analysis['Detections_Archival_Components']
#     else:
#         archival = []

#     if 'Detections_Plant_Components' in analysis:
#         plant = analysis['Detections_Plant_Components']
#     else:
#         plant = []

#     if cfg['leafmachine']['leaf_segmentation']['segment_whole_leaves']:
#         if 'Segmentation_Whole_Leaf' in analysis:
#             Segmentation_Whole_Leaf = analysis['Segmentation_Whole_Leaf']
#         else:
#             Segmentation_Whole_Leaf = []
#     else:
#         Segmentation_Whole_Leaf = []

#     if cfg['leafmachine']['leaf_segmentation']['segment_partial_leaves']:
#         if 'Segmentation_Partial_Leaf' in analysis:
#             Segmentation_Partial_Leaf = analysis['Segmentation_Partial_Leaf']
#         else:
#             Segmentation_Partial_Leaf = []
#     else:
#         Segmentation_Partial_Leaf = []

#     if cfg['leafmachine']['landmark_detector']['landmark_whole_leaves']:
#         if 'Landmarks_Whole_Leaves' in analysis:
#             Landmarks_Whole_Leaves = analysis['Landmarks_Whole_Leaves']
#         else:
#             Landmarks_Whole_Leaves = []
#     else:
#         Landmarks_Whole_Leaves = []

#     if cfg['leafmachine']['landmark_detector']['landmark_partial_leaves']:
#         if 'Landmarks_Partial_Leaves' in analysis:
#             Landmarks_Partial_Leaves = analysis['Segmentation_Partial_Leaf']
#         else:
#             Landmarks_Partial_Leaves = []
#     else:
#         Landmarks_Partial_Leaves = []


#     if 'Ruler_Info' in analysis:
#         Ruler_Images = analysis['Ruler_Info']
#     else:
#         Ruler_Images = []


#     try:
#         image = Image.open(os.path.join(Project.dir_images, '.'.join([filename, 'jpg'])))
#     except:
#         image = Image.open(os.path.join(Project.dir_images, '.'.join([filename, 'jpeg'])))

#     image_overlay = image
#     if image_overlay.mode != 'RGB':
#         image_overlay = image_overlay.convert('RGB')

#     with lock:

#         image_overlay = add_archival_detections(image_overlay, archival, height, width, line_w_archival, show_archival, ignore_archival, cfg)

#         image_overlay = add_plant_detections(image_overlay, plant, height, width, line_w_plant, show_plant, ignore_plant, cfg)

#         image_overlay = add_segmentations(image_overlay, Segmentation_Whole_Leaf, Segmentation_Partial_Leaf, show_segmentations, cfg)

#         image_overlay = add_landmarks(image_overlay, Landmarks_Whole_Leaves, Landmarks_Partial_Leaves, show_landmarks, cfg)

#         ruler_img = get_ruler_images(Ruler_Images, cfg)

#         save_overlay_images_to_jpg(image_overlay, filename, Dirs, cfg)

#     return filename, image_overlay, ruler_img


############ replaced with process/queue
# def process_file(ProjectSQL, dir_images, filename, line_w_archival, show_archival, ignore_archival, line_w_plant, show_plant, ignore_plant, show_segmentations, show_landmarks, cfg, Dirs, lock):
#     # ProjectSQL.database was passed in as ProjectSQL to enable parallelization
#     conn = sqlite3.connect(ProjectSQL)
#     cur = conn.cursor()

#     # Initialize the dictionary to store the analysis data
#     analysis = {}

#     # Fetch image dimensions from the `images` table
#     cur.execute("""
#         SELECT width, height
#         FROM images
#         WHERE name = ?;
#     """, (filename,))
#     image_data = cur.fetchone()
#     if image_data:
#         analysis['width'], analysis['height'] = image_data
#     else:
#         analysis['width'], analysis['height'] = 0, 0

#     # Fetch all archival component detections from the `annotations_archival` table
#     cur.execute("""
#         SELECT annotation
#         FROM annotations_archival
#         WHERE file_name = ?;
#     """, (filename,))
#     archival_data = cur.fetchall()
#     analysis['Detections_Archival_Components'] = [row[0] for row in archival_data] if archival_data else []

#     # Fetch all plant component detections from the `annotations_plant` table
#     cur.execute("""
#         SELECT annotation
#         FROM annotations_plant
#         WHERE file_name = ?;
#     """, (filename,))
#     plant_data = cur.fetchall()
#     analysis['Detections_Plant_Components'] = [row[0] for row in plant_data] if plant_data else []

#     # Fetch all segmentation data for whole leaves from the `Segmentation_Whole_Leaf` table
#     if cfg['leafmachine']['leaf_segmentation']['segment_whole_leaves']:
#         cur.execute("""
#             SELECT *
#             FROM Segmentation_Whole_Leaf
#             WHERE file_name = ?;
#         """, (filename,))
#         seg_whole_leaf_data = cur.fetchall()
#         # Store all columns in the analysis dictionary
#         analysis['Segmentation_Whole_Leaf'] = seg_whole_leaf_data if seg_whole_leaf_data else []
#     else:
#         analysis['Segmentation_Whole_Leaf'] = []

#     # Fetch all segmentation data for partial leaves from the `Segmentation_Partial_Leaf` table
#     if cfg['leafmachine']['leaf_segmentation']['segment_partial_leaves']:
#         cur.execute("""
#             SELECT *
#             FROM Segmentation_Partial_Leaf
#             WHERE file_name = ?;
#         """, (filename,))
#         seg_partial_leaf_data = cur.fetchall()
#         # Store all columns in the analysis dictionary
#         analysis['Segmentation_Partial_Leaf'] = seg_partial_leaf_data if seg_partial_leaf_data else []
#     else:
#         analysis['Segmentation_Partial_Leaf'] = []


#    # Fetch all landmark data for whole leaves from the `Landmarks_Whole_Leaves` table
#     if cfg['leafmachine']['landmark_detector']['landmark_whole_leaves']:
#         cur.execute("""
#             SELECT *
#             FROM Landmarks_Whole_Leaves
#             WHERE file_name = ?;
#         """, (filename,))
#         landmarks_whole_data = cur.fetchall()
#         columns = [desc[0] for desc in cur.description]  # Get column names
#         analysis['Landmarks_Whole_Leaves'] = [dict(zip(columns, row)) for row in landmarks_whole_data] if landmarks_whole_data else []
#     else:
#         analysis['Landmarks_Whole_Leaves'] = []

#     # Fetch all landmark data for partial leaves from the `Landmarks_Partial_Leaves` table
#     if cfg['leafmachine']['landmark_detector']['landmark_partial_leaves']:
#         cur.execute("""
#             SELECT *
#             FROM Landmarks_Partial_Leaves
#             WHERE file_name = ?;
#         """, (filename,))
#         landmarks_partial_data = cur.fetchall()
#         columns = [desc[0] for desc in cur.description]  # Get column names
#         analysis['Landmarks_Partial_Leaves'] = [dict(zip(columns, row)) for row in landmarks_partial_data] if landmarks_partial_data else []
#     else:
#         analysis['Landmarks_Partial_Leaves'] = []

    
#     # Fetch all segmentation data for partial leaves from the `Segmentation_Partial_Leaf` table
#     if cfg['leafmachine']['leaf_segmentation']['save_oriented_images'] or cfg['leafmachine']['leaf_segmentation']['save_oriented_mask']:
#         cur.execute("""
#             SELECT *
#             FROM Keypoints_Data
#             WHERE file_name = ?;
#         """, (filename,))
#         keypoints_data = cur.fetchall()
#         # Store all columns in the analysis dictionary
#         analysis['Keypoints_Data'] = keypoints_data if keypoints_data else []
#     else:
#         analysis['Keypoints_Data'] = []



#     # Fetch all ruler information from the `ruler_data` table
#     cur.execute("""
#         SELECT ruler_image_name, conversion_mean, predicted_conversion_factor_cm
#         FROM ruler_data
#         WHERE file_name = ?;
#     """, (filename,))
#     ruler_data = cur.fetchall()

#     # Store the fetched data in the analysis dictionary
#     analysis['Ruler_Info'] = [{'ruler_image_name': row[0], 
#                             'conversion_mean': row[1],
#                             'predicted_conversion_factor_cm': row[2]} 
#                             for row in ruler_data] if ruler_data else []


#     conn.close()

#     # Open the image using the filename
#     try:
#         image = Image.open(os.path.join(dir_images, f"{filename}.jpg"))
#     except FileNotFoundError:
#         image = Image.open(os.path.join(dir_images, f"{filename}.jpeg"))
#     except Exception as e:
#         # Handle other potential exceptions, such as the file not existing
#         print(f"Error opening image file for {os.path.join(dir_images, filename)}: {e}")
#         return filename, None, None

#     image_overlay = image
#     if image_overlay.mode != 'RGB':
#         image_overlay = image_overlay.convert('RGB')

#     with lock:
#         # Process the image with various overlay functions
#         image_overlay = add_archival_detections(image_overlay, analysis['Detections_Archival_Components'], analysis['height'], analysis['width'], analysis['Ruler_Info'], line_w_archival, show_archival, ignore_archival, cfg)
#         image_overlay = add_plant_detections(image_overlay, analysis['Detections_Plant_Components'], analysis['height'], analysis['width'], line_w_plant, show_plant, ignore_plant, cfg)
#         image_overlay = add_segmentations(image_overlay, analysis['Segmentation_Whole_Leaf'], analysis['Segmentation_Partial_Leaf'], show_segmentations, cfg)
#         # image_overlay = add_landmarks(image_overlay, analysis['Landmarks_Whole_Leaves'], analysis['Landmarks_Partial_Leaves'], show_landmarks, cfg)
#         image_overlay = add_keypoints(image_overlay, keypoints_data, True, cfg)

#         # Process ruler images if available
#         # ruler_img = get_ruler_images(analysis['Ruler_Info'], cfg)

#         # Save the overlay image
#         save_overlay_images_to_jpg(image_overlay, filename, Dirs, cfg)

#     # return filename, image_overlay#, ruler_img


def build_custom_overlay(cfg, logger, dir_home, Project, batch, Dirs):
    start_t = perf_counter()
    logger.name = f'[BATCH {batch+1} Build Custom Overlay]'
    logger.info(f'Creating overlay for {batch+1}')

    line_w_archival = cfg['leafmachine']['overlay']['line_width_archival']
    line_w_plant = cfg['leafmachine']['overlay']['line_width_plant']
    show_archival = cfg['leafmachine']['overlay']['show_archival_detections']
    ignore_archival = cfg['leafmachine']['overlay']['ignore_archival_detections_classes']
    show_plant = cfg['leafmachine']['overlay']['show_plant_detections']
    ignore_plant = cfg['leafmachine']['overlay']['ignore_plant_detections_classes']
    show_segmentations = cfg['leafmachine']['overlay']['show_segmentations']
    show_landmarks = cfg['leafmachine']['overlay']['show_landmarks']
    ignore_landmarks = cfg['leafmachine']['overlay']['ignore_landmark_classes']

    filenames = []
    overlay_images = []
    for filename, analysis in Project.project_data_list[batch].items():
        logger.info(f'Creating overlay for {filename}')
        # print(filename)
        # print(analysis)
        if 'height' in analysis:
            height = analysis['height']
        else:
            height = 0

        if 'width' in analysis:
            width = analysis['width']
        else:
            width = 0

        if 'Detections_Archival_Components' in analysis:
            archival = analysis['Detections_Archival_Components']
        else:
            archival = []

        if 'Detections_Plant_Components' in analysis:
            plant = analysis['Detections_Plant_Components']
        else:
            plant = []

        # Whole_Leaf_BBoxes = analysis['Whole_Leaf_BBoxes']
        # Whole_Leaf_BBoxes_YOLO = analysis['Whole_Leaf_BBoxes_YOLO']
        # Whole_Leaf_Cropped = analysis['Whole_Leaf_Cropped']
        
        # Partial_Leaf_BBoxes_YOLO = analysis['Partial_Leaf_BBoxes_YOLO']
        # Partial_Leaf_BBoxes = analysis['Partial_Leaf_BBoxes']
        # Partial_Leaf_Cropped = analysis['Partial_Leaf_Cropped']

        if cfg['leafmachine']['leaf_segmentation']['segment_whole_leaves']:
            if 'Segmentation_Whole_Leaf' in analysis:
                Segmentation_Whole_Leaf = analysis['Segmentation_Whole_Leaf']
            else:
                Segmentation_Whole_Leaf = []
        else:
            Segmentation_Whole_Leaf = []

        if cfg['leafmachine']['leaf_segmentation']['segment_partial_leaves']:
            if 'Segmentation_Partial_Leaf' in analysis:
                Segmentation_Partial_Leaf = analysis['Segmentation_Partial_Leaf']
            else:
                Segmentation_Partial_Leaf = []
        else:
            Segmentation_Partial_Leaf = []

        if cfg['leafmachine']['landmark_detector']['landmark_whole_leaves']:
            if 'Landmarks_Whole_Leaves' in analysis:
                Landmarks_Whole_Leaves = analysis['Landmarks_Whole_Leaves']
            else:
                Landmarks_Whole_Leaves = []
        else:
            Landmarks_Whole_Leaves = []

        if cfg['leafmachine']['landmark_detector']['landmark_partial_leaves']:
            if 'Landmarks_Partial_Leaves' in analysis:
                Landmarks_Partial_Leaves = analysis['Segmentation_Partial_Leaf']
            else:
                Landmarks_Partial_Leaves = []
        else:
            Landmarks_Partial_Leaves = []
        
        try:
            image = Image.open(os.path.join(Project.dir_images, '.'.join([filename, 'jpg'])))
        except:
            image = Image.open(os.path.join(Project.dir_images, '.'.join([filename, 'jpeg'])))

        image_overlay = image
        if image_overlay.mode != 'RGB':
            image_overlay = image_overlay.convert('RGB')

        image_overlay = add_archival_detections(image_overlay, archival, height, width, line_w_archival, show_archival, ignore_archival, cfg)

        image_overlay = add_plant_detections(image_overlay, plant, height, width, line_w_plant, show_plant, ignore_plant, cfg)

        image_overlay = add_segmentations(image_overlay, Segmentation_Whole_Leaf, Segmentation_Partial_Leaf, show_segmentations, cfg)

        image_overlay = add_landmarks(image_overlay, Landmarks_Whole_Leaves, Landmarks_Partial_Leaves, show_landmarks, cfg)
        
        # add_efds()

        # add_landmarks()

        # create_panel() # with individual leaves inside a table to the right of the full image
        #images in panel have more info printed with them 

        save_overlay_images_to_jpg(image_overlay, filename, Dirs, cfg)

        filenames.append(filename)
        overlay_images.append(image_overlay)

    # save_custom_overlay_to_PDF(filenames, overlay_images, batch, Dirs, Project, cfg)
    end_t = perf_counter()
    logger.info(f'Batch {batch+1}: Build Custom Overlay Duration --> {round((end_t - start_t)/60)} minutes')
    

def add_landmarks(image_overlay, Landmarks_Whole_Leaves, Landmarks_Partial_Leaves, show_landmarks, cfg):
    if show_landmarks:
        if cfg['leafmachine']['landmark_detector']['landmark_whole_leaves']:
            leaf_type = 0
            for data in Landmarks_Whole_Leaves:
                image_overlay = insert_landmark(image_overlay, data, cfg)

            # im_show = cv2.cvtColor(np.array(image_overlay), cv2.COLOR_RGB2BGR)
            # cv2.imshow('overlay', im_show)
            # cv2.waitKey(0)
        if cfg['leafmachine']['landmark_detector']['landmark_partial_leaves']: # TODO finish this
            leaf_type = 1
            for data in Landmarks_Partial_Leaves:
                image_overlay = insert_landmark(image_overlay, data, cfg)
            # im_show = cv2.cvtColor(np.array(image_overlay), cv2.COLOR_RGB2BGR)
            # cv2.imshow('overlay', im_show)
            # cv2.waitKey(0)
    return image_overlay



def add_segmentations(image_overlay, Segmentation_Whole_Leaf, Segmentation_Partial_Leaf, show_segmentations, cfg):
    if show_segmentations:
        if cfg['leafmachine']['leaf_segmentation']['segment_whole_leaves']:
            leaf_type = 0
            for all_data in Segmentation_Whole_Leaf: # Each iteration is a whole_leaf
                # [0] is id
                # [1] is full image name
                # [2] is the cropped name e.g. LM_Validate_LowRes_10__L__1045-1351-1305-1619
                # [3] is the object_name  e.g. petiole_349-4-370-56
                # [4+] is the rest of the data
                # [] is the polygon_closed
                # [] is the bbox_min
                
                seg_name = all_data[2]
                seg_name_short = seg_name.split("__")[-1]
                overlay_data_all = ast.literal_eval(all_data[4])

                try:
                    overlay_poly_all, overlay_poly_oriented_all, overlay_efd_all, overlay_rect_all, overlay_color = overlay_data_all
                    for i, overlay_poly in enumerate(overlay_poly_all):
                        object_name = next(iter(overlay_color[i].keys()))
                        object_name_parts = object_name.split(' ')
                        if cfg['leafmachine']['leaf_segmentation']['calculate_elliptic_fourier_descriptors']:
                            overlay_efd = overlay_efd_all[i]
                        else:
                            overlay_efd = None

                        if 'leaf' in object_name_parts:
                            c_outline, c_fill = get_color('seg_leaf_whole', 'SEG_WHOLE', cfg)
                        elif 'petiole' in object_name_parts:
                            c_outline, c_fill = get_color('seg_leaf_whole_petiole', 'SEG_WHOLE', cfg)
                        elif 'hole' in object_name_parts:
                            c_outline, c_fill = get_color('seg_hole', 'SEG_WHOLE', cfg)

                        overlay_data_insert = [overlay_poly_all[i], overlay_efd, overlay_rect_all[i], c_outline, c_fill]
                        image_overlay = insert_seg(image_overlay, overlay_data_insert, seg_name_short, cfg)
                except:
                    continue
            # im_show = cv2.cvtColor(np.array(image_overlay), cv2.COLOR_RGB2BGR)
            # cv2.imshow('overlay', im_show)
            # cv2.waitKey(0)
        if cfg['leafmachine']['leaf_segmentation']['segment_partial_leaves']:
            ##################### TODO THIS IS NOT DONE
            leaf_type = 1
            for all_data in Segmentation_Whole_Leaf:
                # [0] is id
                # [1] is full image name
                # [2] is the cropped name e.g. LM_Validate_LowRes_10__L__1045-1351-1305-1619
                # [3] is the data for overlay
                # [4+] is the rest of the data
                seg_name = all_data[2]
                seg_name_short = seg_name.split("__")[-1]
                obj = json.loads(all_data[3])[0]
                for seg_specific_name, overlay_data in obj.items():
                    parts = seg_specific_name.split("_")

                    overlay_poly = overlay_data['polygon_closed']
                    overlay_rect = overlay_data['bbox_min']
                    overlay_efd = overlay_data['efds']['efd_pts_PIL']

                    if 'leaf' in parts:
                        c_outline, c_fill = get_color('seg_leaf_partial', 'SEG_PARTIAL', cfg)
                    elif 'petiole' in parts:
                        c_outline, c_fill = get_color('seg_leaf_partial_petiole', 'SEG_PARTIAL', cfg)
                    elif 'hole' in parts:
                        c_outline, c_fill = get_color('seg_hole', 'SEG_PARTIAL', cfg)

                    overlay_data_insert = [overlay_poly, overlay_efd, overlay_rect, c_outline, c_fill]
                    image_overlay = insert_seg(image_overlay, overlay_data_insert, seg_name_short, cfg)
            # im_show = cv2.cvtColor(np.array(image_overlay), cv2.COLOR_RGB2BGR)
            # cv2.imshow('overlay', im_show)
            # cv2.waitKey(0)
    return image_overlay

def add_keypoints(image_overlay, keypoints_data, show_keypoints, cfg):
    if show_keypoints:
        for data in keypoints_data:
            image_overlay = insert_keypoint(image_overlay, data, cfg)

    return image_overlay

# Keypoints groups
# lamina_tip = keypoints_list[0]
# apex_left = keypoints_list[1]
# apex_center = keypoints_list[2]
# apex_right = keypoints_list[3]
# midvein_points = [(x+origin_x, y+origin_y) for x, y in keypoints_list[4:19]]  # Translate midvein points
# base_left = keypoints_list[19]
# base_center = keypoints_list[20]
# base_right = keypoints_list[21]
# lamina_base = keypoints_list[22]
# petiole_points = [(x+origin_x, y+origin_y) for x, y in keypoints_list[23:28]]  # Translate petiole points
# width_left = keypoints_list[29]
# width_right = keypoints_list[30]
def convert_to_int_tuples(list_of_floats):
    """Converts a list of float tuples to a list of int tuples."""
    return [(int(x), int(y)) for x, y in list_of_floats]

def insert_keypoint(full_image, all_data, cfg):
    # Helper function to check if the point is valid (i.e., not hidden)
    def is_valid_point(x, origin_x, y, origin_y):
        return abs(x - origin_x) >= 2 and abs(y - origin_y) >= 2

    # Convert the strings from the database into Python objects
    seg_name = all_data[2]
    keypoints_list_str = all_data[3]
    tip_loc_str = all_data[5]
    base_loc_str = all_data[6]

    # Use ast.literal_eval to safely parse the string to lists of tuples, then convert to int tuples
    keypoints_list = convert_to_int_tuples(ast.literal_eval(keypoints_list_str))
    tip_loc = tuple(map(int, ast.literal_eval(tip_loc_str)))
    base_loc = tuple(map(int, ast.literal_eval(base_loc_str)))

    # print(f'IND[{ind}] seg_name {seg_name}')

    seg_name_short = seg_name.split("__")[-1]
    origin_x = int(seg_name_short.split('-')[0])
    origin_y = int(seg_name_short.split('-')[1])

    # Configuration for visual attributes
    width = cfg['leafmachine']['overlay']['line_width_seg'] + 2
    large_width = width * 2  # Larger ellipses for the tip and base
    midvein_color = (0, 255, 255, 255)  # Cyan for midvein points
    petiole_color = (255, 255, 0, 255)  # Yellow for petiole points
    tip_color = (0, 255, 0, 100)  # Green for tip
    base_color = (255, 0, 0, 100)  # Red for base
    width_color = (255, 0, 0, 255)  # Red for other keypoints
    base_angle_color = (0, 0, 0, 255) # black
    apex_angle_color = (137, 137, 137, 255) # gray
    outline_color = (0, 0, 0, 255)  # Black outline

    # Initialize the drawing object
    full_image = np.asarray(full_image)
    full_image = Image.fromarray(full_image)
    draw = ImageDraw.Draw(full_image, "RGBA")

    # Translate all points first
    translated_keypoints = [(x+origin_x, y+origin_y) for x, y in keypoints_list]
    translated_tip = (tip_loc[0] + origin_x, tip_loc[1] + origin_y)
    translated_base = (base_loc[0] + origin_x, base_loc[1] + origin_y)

    # Draw translated midvein points
    for x, y in translated_keypoints[4:19]:
        if is_valid_point(x, origin_x, y, origin_y):
            draw.ellipse((x - width, y - width, x + width, y + width), fill=midvein_color, outline=outline_color)

    # Draw translated petiole points
    for x, y in translated_keypoints[23:28]:
        if is_valid_point(x, origin_x, y, origin_y):
            draw.ellipse((x - width, y - width, x + width, y + width), fill=petiole_color, outline=outline_color)

    # Draw specific keypoint groups
    # Apex points (left, center, right)
    apex_points = [translated_keypoints[1], translated_keypoints[2], translated_keypoints[3]]
    for x, y in apex_points:
        if is_valid_point(x, origin_x, y, origin_y):
            draw.ellipse((x - width, y - width, x + width, y + width), fill=apex_angle_color, outline=outline_color)

    # Base angle points (left, center, right)
    base_angle_points = [translated_keypoints[19], translated_keypoints[20], translated_keypoints[21]]
    for x, y in base_angle_points:
        if is_valid_point(x, origin_x, y, origin_y):
            draw.ellipse((x - width, y - width, x + width, y + width), fill=base_angle_color, outline=outline_color)

    # Width points (left, right)
    width_points = [translated_keypoints[29], translated_keypoints[30]]
    for x, y in width_points:
        if is_valid_point(x, origin_x, y, origin_y):
            draw.ellipse((x - width, y - width, x + width, y + width), fill=width_color, outline=outline_color)

    # Draw translated tip with larger green ellipse
    if is_valid_point(translated_tip[0], origin_x, translated_tip[1], origin_y):
        draw.ellipse((translated_tip[0] - large_width, translated_tip[1] - large_width, translated_tip[0] + large_width, translated_tip[1] + large_width), fill=tip_color, outline=outline_color)

    # Draw translated base with larger red ellipse
    if is_valid_point(translated_base[0], origin_x, translated_base[1], origin_y):
        draw.ellipse((translated_base[0] - large_width, translated_base[1] - large_width, translated_base[0] + large_width, translated_base[1] + large_width), fill=base_color, outline=outline_color)

    return full_image


def str_to_list(data):
    """
    Converts all string representations of lists in a dictionary to actual Python lists.
    
    Args:
        data (dict): The dictionary containing potential string representations of lists.
        
    Returns:
        dict: The updated dictionary with actual lists instead of string representations.
    """
    def convert(item):
        # If the item is a string that looks like a list, convert it to a list
        if isinstance(item, str) and (item.startswith('[') and item.endswith(']')):
            try:
                return ast.literal_eval(item)
            except (ValueError, SyntaxError):
                return item
        # If the item is a list, recursively convert its elements
        elif isinstance(item, list):
            return [convert(i) for i in item]
        return item
    
    return {key: convert(value) for key, value in data.items()}

def insert_landmark(full_image, data_with_str, cfg):
    data = str_to_list(data_with_str)

    width = cfg['leafmachine']['overlay']['line_width_seg'] + 2
    # origin_x = int(seg_name_short.split('-')[0])
    # origin_y = int(seg_name_short.split('-')[1])
    # initialize
    full_image = np.asarray(full_image)
    full_image = Image.fromarray(full_image)
    draw = ImageDraw.Draw(full_image, "RGBA")

    A, B = check_lamina_length(data)
    if (A is not None) and (B is not None):
        order_points_plot([A, B], 'lamina_length', full_image, draw, width)

    if 't_midvein_fit_points' in data:
        if (data['t_midvein_fit_points'] != []):
            order_points_plot(data['t_midvein_fit_points'], 'midvein_fit', full_image, draw, width)

    if 't_apex_center' in data:
        if (data['t_apex_center'] != []) and (data['has_apex']):
            order_points_plot([data['t_apex_left'], data['t_apex_center'], data['t_apex_right']], data['apex_angle_type'], full_image, draw, width)
    
    if 't_base_center' in data:
        if (data['t_base_center'] != []) and (data['has_base']):
            order_points_plot([data['t_base_left'], data['t_base_center'], data['t_base_right']], data['base_angle_type'], full_image, draw, width)

    if 'has_width' in data:
        if 't_width_left' in data and 't_width_right' in data:
            if (data['t_width_left'] != []) and (data['t_width_right'] != []):
                order_points_plot([data['t_width_left'], data['t_width_right']], 'lamina_width', full_image, draw, width)
    else:
        if 't_width_infer' in data:
            if data['t_width_infer'] != []:
                order_points_plot(data['t_width_infer'], 'infer_width', full_image, draw, width)

    if 't_midvein' in data:
        if (data['t_midvein'] != []) and (data['has_midvein'] != []):
            order_points_plot(data['t_midvein'], 'midvein_trace', full_image, draw, width)
    
    if 't_petiole' in data:
        if (data['t_petiole'] != []) and (data['has_ordered_petiole'] != []):
            order_points_plot(data['t_petiole'], 'petiole_trace', full_image, draw, width)

    if 't_lobes' in data:
        if (data['t_lobes'] != []) and (data['has_lobes'] != []):
            order_points_plot(data['t_lobes'], 'lobes', full_image, draw, width)

    # Lamina tip and base
    if 't_lamina_tip' in data:
        if (data['t_lamina_tip'] != []) and (data['has_lamina_tip'] != []):
            draw.ellipse((data['t_lamina_tip'][0]-width, data['t_lamina_tip'][1]-width, data['t_lamina_tip'][0]+width, data['t_lamina_tip'][1]+width), fill=(0, 255, 0, 255), outline=(0, 0, 0, 255))
    if 't_lamina_base' in data:
        if (data['t_lamina_base'] != []) and (data['has_lamina_base'] != []):
            draw.ellipse((data['t_lamina_base'][0]-width, data['t_lamina_base'][1]-width, data['t_lamina_base'][0]+width, data['t_lamina_base'][1]+width), fill=(255, 0, 0, 255), outline=(0, 0, 0, 255))

        # Apex angle
    if 't_apex_left' in data:
        if (data['t_apex_left'] != []) and (data['has_apex'] != []):
            draw.ellipse((data['t_apex_left'][0]-width, data['t_apex_left'][1]-width, data['t_apex_left'][0]+width, data['t_apex_left'][1]+width), fill=(255, 0, 0, 255))
    if 't_apex_right' in data:
        if (data['t_apex_right'] != []) and (data['has_apex'] != []):
            draw.ellipse((data['t_apex_right'][0]-width, data['t_apex_right'][1]-width, data['t_apex_right'][0]+width, data['t_apex_right'][1]+width), fill=(0, 0, 255, 255))

        # Base angle
    if 't_base_left' in data:
        if (data['t_base_left'] != []) and (data['has_base'] != []):
            draw.ellipse((data['t_base_left'][0]-width, data['t_base_left'][1]-width, data['t_base_left'][0]+width, data['t_base_left'][1]+width), fill=(255, 0, 0, 255))
    if 't_base_right' in data:
        if (data['t_base_right'] != []) and (data['has_base'] != []):
            draw.ellipse((data['t_base_right'][0]-width, data['t_base_right'][1]-width, data['t_base_right'][0]+width, data['t_base_right'][1]+width), fill=(0, 0, 255, 255))

    return full_image


def order_points_plot(points, version, full_image, draw, thick):
    if version == 'midvein_trace':
        # color = (0, 255, 0)
        color = (0, 0, 0)
        # thick = 2
    elif version == 'petiole_trace':
        color = (0, 255, 255)
        # thick = 2
    elif version == 'lamina_width':
        color = (255, 0, 0)
    elif version == 'lamina_length':
        color = (255, 255, 255)
        # thick = 2
    elif version == 'lamina_width_alt':
        color = (255, 100, 100)
    elif version == 'infer_width':
        color = (255, 100, 100)
    elif version == 'midvein_fit':
        color = (255, 255, 255)
        thick = 2
    elif version == 'not_reflex':
        color = (255, 0, 200)
        # thick = 3
    elif version == 'reflex':
        color = (255, 120, 0)
        # thick = 3
    elif version == 'petiole_tip_alt':
        color = (100, 55, 255)
        # thick = 1
    elif version == 'petiole_tip':
        color = (55, 255, 100)
        # thick = 1
    elif version == 'failed_angle':
        color = (0, 0, 0)
        # thick = 3
    elif version == 'lobes':
        color = (0, 30, 255)
        # thick = 3

    # Convert the points to a list of tuples and round to integer values
    points_list = [tuple(np.round(p).astype(int)) for p in points]

    if version == 'infer_width':
        for p in points_list:
            draw.ellipse((p[0]-thick, p[1]-thick, p[0]+thick, p[1]+thick), fill=color+(255,))
    elif version == 'midvein_fit':
        for p in points_list:
            draw.ellipse((p[0]-thick, p[1]-thick, p[0]+thick, p[1]+thick), fill=color+(255,))
    elif version == 'lamina_length':
        for i in range(len(points_list) - 1):
            draw.line([points_list[i], points_list[i+1]], fill=color, width=thick)
    elif version == 'lobes':
        for i in range(len(points_list) - 1):
            lobe = points_list[i]
            draw.ellipse((lobe[0]-thick*3, lobe[1]-thick*3, lobe[0]+thick*3, lobe[1]+thick*3), outline=color, width=int(thick/2))
            draw.ellipse((lobe[0]-thick, lobe[1]-thick, lobe[0]+thick, lobe[1]+thick), outline=color, width=thick)
    else:
        for i in range(len(points_list) - 1):
            draw.line([points_list[i], points_list[i+1]], fill=color+(255,), width=thick)

    return full_image

def check_lamina_length(data):
    if data['has_lamina_base'] and data['has_lamina_tip']:
        return data['t_lamina_base'], data['t_lamina_tip']
    else:
        if data['has_lamina_base'] and (not data['has_lamina_tip']) and data['has_apex']: # lamina base and apex center
            return data['t_lamina_base'], data['t_apex_center']
        elif data['has_lamina_tip'] and (not data['has_lamina_base']) and data['has_base']: # lamina tip and base center
            return data['t_lamina_tip'], data['t_apex_center']
        elif (not data['has_lamina_tip']) and (not data['has_lamina_base']) and data['has_apex'] and data['has_base']: # apex center and base center
            return data['t_base_center'], data['t_apex_center']
        else:
            return None, None

def insert_seg(full_image, overlay_data, seg_name_short, cfg):
    width = cfg['leafmachine']['overlay']['line_width_seg']
    width_efd = cfg['leafmachine']['overlay']['line_width_efd']
    origin_x = int(seg_name_short.split('-')[0])
    origin_y = int(seg_name_short.split('-')[1])
    # unpack
    try:
        overlay_poly, overlay_efd, overlay_rect, c_outline, c_fill = overlay_data
        # fill_color = overlay_color[0][0]
        # outline_color =  overlay_color[0][1]

        # initialize
        full_image = np.asarray(full_image)
        full_image = Image.fromarray(full_image)
        draw = ImageDraw.Draw(full_image, "RGBA")

        if len(overlay_poly) != 0:
            t_overlay_poly = [(x+origin_x, y+origin_y) for x, y in overlay_poly] # x,y
            draw.polygon(t_overlay_poly, fill=c_fill, outline=c_outline, width=width)
            
        if len(overlay_rect) != 0:
            t_overlay_rect = [(x+origin_x, y+origin_y) for x, y in overlay_rect]
            draw.polygon(t_overlay_rect, fill=None, outline=c_outline, width=width)
            
        if overlay_efd:
            if len(overlay_efd) != 0:
                t_overlay_efd = [(x+origin_x, y+origin_y) for x, y in overlay_efd]
                draw.polygon(t_overlay_efd, fill=None, outline=(135,30,210), width=width_efd)
        
        # After calling insert_seg, use matplotlib to show the image
        # plt.imshow(full_image)
        # plt.axis('off')  # Hide axes
        # plt.show()
        return full_image
    except Exception as e:
        print(f'overlay_data is empty in insert_seg() ... [overlay_poly, overlay_efd, overlay_rect, c_outline, c_fill = overlay_data] {e}')

        return full_image

def add_plant_detections(image_overlay, plant, height, width, line_w, show_plant, ignore_plant, cfg):
    if show_plant:
        draw = ImageDraw.Draw(image_overlay, "RGBA")

        for annotation in plant:
            anno = yolo_to_position(annotation, height, width, 'PLANT')
            if anno[0] not in ignore_plant:
                polygon = [(anno[1], anno[2]), (anno[3], anno[2]), (anno[3], anno[4]), (anno[1], anno[4])]
                c_outline, c_fill = get_color(anno[0], 'PLANT', cfg)
                draw.polygon(polygon, fill=c_fill, outline=c_outline, width=line_w)
        # im_show = cv2.cvtColor(np.array(image_overlay), cv2.COLOR_RGB2BGR)
        # cv2.imshow('overlay', im_show)
        # cv2.waitKey(0)
    return image_overlay
def add_archival_detections(image_overlay, archival, height, width, ruler_info, line_w, show_archival, ignore_archival, cfg):
    if show_archival:
        draw = ImageDraw.Draw(image_overlay, "RGBA")
        
        # List to store all text elements to be drawn later using OpenCV
        cv2_text_elements = []

        for annotation in archival:
            anno = yolo_to_position(annotation, height, width, 'ARCHIVAL')
            if anno[0] not in ignore_archival:
                polygon = [(anno[1], anno[2]), (anno[3], anno[2]), (anno[3], anno[4]), (anno[1], anno[4])]
                c_outline, c_fill = get_color(anno[0], 'ARCHIVAL', cfg)
                draw.polygon(polygon, fill=c_fill, outline=c_outline, width=line_w)
        
        # Draw scale bars for conversion_mean and predicted_conversion_factor_cm
        spacing = 10  
        spacing_s = 60  # Spacing between squares

        if ruler_info:
            start_x, start_y = 30, 50  # Initial coordinates

            for i, rul in enumerate(ruler_info):
                current_y = start_y  # Start at the current Y position for this ruler
                square_positions = []  # To store the positions of squares for enclosing

                if rul['predicted_conversion_factor_cm'] is None:
                    print("rul['predicted_conversion_factor_cm'] is None")
                # Draw green square for conversion_mean
                if 'conversion_mean' in rul and rul['conversion_mean'] is not None and rul['conversion_mean'] != 0:
                    conv_mean_size = int(rul['conversion_mean'])
                    conv_mean_square = [(start_x, current_y), (start_x + conv_mean_size, current_y + conv_mean_size)]
                    draw.rectangle(conv_mean_square, fill=(0, 255, 0, 128), outline=(0, 255, 0, 255), width=1)
                    square_positions.append(conv_mean_square)

                    # Add OpenCV text inside the green square
                    text = f'1cm={rul["conversion_mean"]:.1f}'
                    text_x = start_x + 2 #conv_mean_size // 2
                    text_y = current_y - 10 #+ conv_mean_size // 2
                    cv2_text_elements.append((text, (text_x, text_y)))

                    # Update the X position for the next square (red square)
                    current_x = start_x + conv_mean_size + spacing
                else:
                    current_x = start_x  # If no green square, red square starts at the initial X position

                # Draw red square for predicted_conversion_factor_cm
                if 'predicted_conversion_factor_cm' in rul and rul['predicted_conversion_factor_cm'] is not None and rul['predicted_conversion_factor_cm'] != 0:
                    pred_conv_size = int(rul['predicted_conversion_factor_cm'])
                    pred_conv_square = [(current_x, current_y), (current_x + pred_conv_size, current_y + pred_conv_size)]
                    draw.rectangle(pred_conv_square, fill=(255, 0, 0, 128), outline=(255, 0, 0, 255), width=1)
                    square_positions.append(pred_conv_square)

                    # Add OpenCV text inside the red square
                    text = f'1cm={rul["predicted_conversion_factor_cm"]:.1f}'
                    text_x = current_x + 2 # pred_conv_size // 2
                    text_y = current_y - 10 # pred_conv_size // 2
                    cv2_text_elements.append((text, (text_x, text_y)))

                # Draw a white rectangle around the squares
                if square_positions:
                    min_x = min([pos[0][0] for pos in square_positions])
                    min_y = min([pos[0][1] for pos in square_positions])
                    max_x = max([pos[1][0] for pos in square_positions])
                    max_y = max([pos[1][1] for pos in square_positions])
                    outline_rect = [(min_x - spacing // 2, min_y - spacing // 2), (max_x + spacing // 2, max_y + spacing // 2)]
                    draw.rectangle(outline_rect, outline=(255, 255, 255, 255), width=3)

                    # Add label above the white rectangle
                    label_text = f'Ruler{i+1} {rul["ruler_image_name"]}'
                    label_x = min_x #- spacing // 2
                    label_y = min_y - spacing -20 #- 10  # Slightly above the rectangle
                    cv2_text_elements.append((label_text, (label_x, label_y)))

                # Update the Y position for the next row of squares
                start_y += (max_y - min_y) + spacing_s

        # Convert the PIL image to an OpenCV format (numpy array)
        image_cv = np.array(image_overlay)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGBA2BGR)  # Convert RGBA to BGR for OpenCV

        # Now use OpenCV to draw all the collected text elements
        for text, (x, y) in cv2_text_elements:
            cv2.putText(image_cv, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Convert back to PIL (from BGR to RGB)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        image_overlay = Image.fromarray(image_cv)

    return image_overlay


def get_ruler_images(ruler_info, cfg):
    ruler_composite = None

    for row in ruler_info:
        ruler_img = row['summary_img']
        ruler_composite = stack_images(ruler_composite, ruler_img)
        
    return ruler_composite
        
def stack_images(img1, img2):
    # If img1 is empty, use the dimensions of img2
    if img1 is None:
        return img2

    else:
        # Get the dimensions of the images
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape

        # Calculate the dimensions of the stacked image
        h_stacked = h1 + h2
        w_stacked = max(w1, w2)

        # Create a blank black image for the stacked image
        stacked_img = np.zeros((h_stacked, w_stacked, 3), dtype=np.uint8)

        # Insert the first image at the top
        stacked_img[:h1, :w1, :] = img1

        # Insert the second image at the bottom
        stacked_img[h_stacked-h2:h_stacked, :w2, :] = img2

        return stacked_img

def save_overlay_images_to_jpg(full_image, filename, Dirs, cfg):
    save_each_segmentation_overlay_image = cfg['leafmachine']['overlay']['save_overlay_to_jpgs']
    if save_each_segmentation_overlay_image:
        try:
            full_image.save(os.path.join(Dirs.custom_overlay_images, '.'.join([filename, 'jpg'])))
        except:
            full_image = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(Dirs.custom_overlay_images, '.'.join([filename, 'jpg'])) , full_image)
        
def save_custom_overlay_to_PDF(filenames, full_images, ruler_images, batch, Dirs, Project, cfg):
    color_bg = cfg['leafmachine']['overlay']['overlay_background_color']
    overlay_dpi = cfg['leafmachine']['overlay']['overlay_dpi']
    save_overlay_pdf = cfg['leafmachine']['overlay']['save_overlay_to_pdf']
    batch_size = cfg['leafmachine']['project']['batch_size']

    if save_overlay_pdf:
        if color_bg == 'black':
            color_text = 'white'
        else:
            color_text = 'black'
            
        if batch_size is None:
            batch_size = len(filenames)
        for i in range(0, len(filenames), batch_size):
            start = batch*batch_size
            end = batch*batch_size + batch_size
            # if end > len(os.listdir(Project.dir_images)):
            #     end = int(len(os.listdir(Project.dir_images)))

            filenames_batch = list(itertools.islice(filenames, i, i+batch_size))
            full_images_batch = list(itertools.islice(full_images, i, i+batch_size))
            ruler_images_batch = list(itertools.islice(ruler_images, i, i+batch_size))
            if len(filenames_batch) != batch_size:
                end = batch*batch_size + len(filenames)

            pdf_name = os.path.join(Dirs.custom_overlay_pdfs, ''.join(['Custom_Overlay', '_',str(start+1), 'to',str(end),'.pdf']))
            # with PdfPages(pdf_name) as pdf:
            #     for idx, img in enumerate(full_images_batch):
            #         # Create a new figure
            #         fig = plt.figure(dpi=overlay_dpi)
            #         fig.set_size_inches(8.5, 11)
            #         fig.set_facecolor(color_bg)
            #         plt.tight_layout(pad=0)
            #         # Add the image to the figure
            #         plt.imshow(img)
            #         plt.suptitle(filenames[idx], fontsize=10, y=0.95, color=color_text)
            #         # Save the current figure to the PDF
            #         pdf.savefig(fig)
            #         plt.close()
            # Define the size of the left and right columns
            fig_width = 11
            fig_height = 11
            left_column_size = 0.6
            right_column_size = 0.4
            # Create the PDF
            with PdfPages(pdf_name) as pdf:
                for idx, img in enumerate(full_images_batch):
                    # Create a new figure
                    fig = plt.figure(dpi=overlay_dpi)
                    fig.set_size_inches(fig_width, fig_height)
                    fig.set_facecolor(color_bg)
                    
                    # Create the left subplot for the full-sized image
                    left_ax = fig.add_axes([0, 0, left_column_size, 0.95])
                    left_ax.imshow(img)
                    left_ax.set_xticks([])
                    left_ax.set_yticks([])
                    left_ax.set_anchor('NW')  # right-justify the image
                    
                    # Create the right subplot for the ruler image
                    right_ax = fig.add_axes([left_column_size, 0, right_column_size, 0.95])
                    if ruler_images[idx] is not None:
                        right_ax.imshow(cv2.cvtColor(ruler_images[idx], cv2.COLOR_BGR2RGB))
                    right_ax.set_xticks([])
                    right_ax.set_yticks([])
                    right_ax.set_anchor('NW')  # left-justify the image
                    # right_ax.set_title(filenames[idx], fontsize=10, color=color_text, y=0.95)  # move the title to the top
                    # Add the image name to the title
                    plt.suptitle(filenames[idx], fontsize=10, y=0.97, color=color_text)
                    
                    # Save the figure to the PDF
                    pdf.savefig(fig)
                    
                    # Close the figure
                    plt.close()
        
def yolo_to_position(annotation, height, width, anno_type):
    annotation = list(map(float, annotation.split(',')))
    annotation_class = int(annotation[0])
    return [set_index_for_annotation(annotation_class, anno_type), 
        int((annotation[1] * width) - ((annotation[3] * width) / 2)), 
        int((annotation[2] * height) - ((annotation[4] * height) / 2)), 
        int(annotation[3] * width) + int((annotation[1] * width) - ((annotation[3] * width) / 2)), 
        int(annotation[4] * height) + int((annotation[2] * height) - ((annotation[4] * height) / 2))]
     
@dataclass
class Colors:
    alpha: int = 127

    seg_leaf_whole: tuple = (46,255,0)
    seg_leaf_partial: tuple = (0,200,255)
    seg_leaf_whole_petiole: tuple = (255, 0, 150) #(0,173,255)
    seg_leaf_partial_petiole: tuple = (90, 0, 75) #(255,140,0)
    seg_hole: tuple = (200,0,255)

    seg_leaf_whole_fill: tuple = (46,255,0, 127)
    seg_leaf_partial_fill: tuple = (0,200,255, 127)
    seg_leaf_whole_petiole_fill: tuple = (255, 0, 150, 127)#(0,173,255, 127)
    seg_leaf_partial_petiole_fill: tuple = (90, 0, 75, 127) #(255,140,0, 127)
    seg_hole_fill: tuple = (200,0,255, 127)

    ruler: tuple = (255, 0, 70)
    barcode: tuple = (0, 137, 65)
    colorcard: tuple = (242, 255, 0)
    label: tuple = (0, 0, 255)
    map: tuple = (0, 251, 255)
    envelope: tuple = (163, 0, 89)
    photo: tuple = (255, 205, 220)
    attached: tuple = (255, 172, 40)
    weights: tuple = (140, 140, 140)

    ruler_fill: tuple = (255, 0, 70, 127)
    barcode_fill: tuple = (0, 137, 65, 127)
    colorcard_fill: tuple = (242, 255, 0, 127)
    label_fill: tuple = (0, 0, 255, 127)
    map_fill: tuple = (0, 251, 255, 127)
    envelope_fill: tuple = (163, 0, 89, 127)
    photo_fill: tuple = (255, 205, 220, 127)
    attached_fill: tuple = (255, 172, 40, 127)
    weights_fill: tuple = (140, 140, 140, 127)

    leaf_whole: tuple = (0, 255, 55)
    leaf_partial: tuple = (0, 255, 250)
    leaflet: tuple = (255, 203, 0)
    seed_fruit_one: tuple = (252, 255, 0)
    seed_fruit_many: tuple = (0, 0, 80)
    flower_one: tuple = (255, 52, 255)
    flower_many: tuple = (154, 0, 255)
    bud: tuple = (255, 0, 9)
    specimen: tuple = (0, 0, 0)
    roots: tuple = (255, 134, 0)
    wood: tuple = (144, 22, 22)

    leaf_whole_fill: tuple = (0, 255, 55, 127)
    leaf_partial_fill: tuple = (0, 255, 250, 127)
    leaflet_fill: tuple = (255, 203, 0, 127)
    seed_fruit_one_fill: tuple = (252, 255, 0, 127)
    seed_fruit_many_fill: tuple = (0, 0, 80, 127)
    flower_one_fill: tuple = (255, 52, 255, 127)
    flower_many_fill: tuple = (154, 0, 255, 127)
    bud_fill: tuple = (255, 0, 9, 127)
    specimen_fill: tuple = (0, 0, 0, 127)
    roots_fill: tuple = (255, 134, 0, 127)
    wood_fill: tuple = (144, 22, 22, 127)

    def __init__(self, alpha):
        alpha = int(np.multiply(alpha, 255))
        self.ruler_fill = (255, 0, 70, alpha)
        self.barcode_fill = (0, 137, 65, alpha)
        self.colorcard_fill = (242, 255, 0, alpha)
        self.label_fill = (0, 0, 255, alpha)
        self.map_fill = (0, 251, 255, alpha)
        self.envelope_fill = (163, 0, 89, alpha)
        self.photo_fill = (255, 205, 220, alpha)
        self.attached_fill = (255, 172, 40, alpha)
        self.weights_fill = (140, 140, 140, alpha)
        self.leaf_whole_fill = (0, 255, 55, alpha)
        self.leaf_partial_fill = (0, 255, 250, alpha)
        self.leaflet_fill = (255, 203, 0, alpha)
        self.seed_fruit_one_fill = (252, 255, 0, alpha)
        self.seed_fruit_many_fill = (0, 0, 80, alpha)
        self.flower_one_fill = (255, 52, 255, alpha)
        self.flower_many_fill = (154, 0, 255, alpha)
        self.bud_fill = (255, 0, 9, alpha)
        self.specimen_fill = (0, 0, 0, alpha)
        self.roots_fill = (255, 134, 0, alpha)
        self.wood_fill = (144, 22, 22, alpha)
        self.seg_leaf_whole_fill = (46,255,0, alpha)
        self.seg_leaf_partial_fill = (0,200,255, alpha)
        self.seg_leaf_whole_petiole_fill = (255, 0, 150, alpha)
        self.seg_leaf_partial_petiole_fill = (90, 0, 75, alpha)
        self.seg_hole_fill = (200,0,255, alpha)

def get_color(anno, a_type, cfg):
    if a_type == 'ARCHIVAL':
        alpha = cfg['leafmachine']['overlay']['alpha_transparency_archival']
    elif a_type == 'PLANT':
        alpha = cfg['leafmachine']['overlay']['alpha_transparency_plant']
    elif a_type == 'SEG_WHOLE':
        alpha = cfg['leafmachine']['overlay']['alpha_transparency_seg_whole_leaf']
    elif a_type == 'SEG_PARTIAL':
        alpha = cfg['leafmachine']['overlay']['alpha_transparency_seg_partial_leaf']

    if alpha is None:
        alpha = 0.5
    Color = Colors(alpha)
    color_dict = {
        "ruler": (Color.ruler, Color.ruler_fill),
        "barcode": (Color.barcode, Color.barcode_fill),
        "colorcard": (Color.colorcard, Color.colorcard_fill),
        "label": (Color.label, Color.label_fill),
        "map": (Color.map, Color.map_fill),
        "envelope": (Color.envelope, Color.envelope_fill),
        "photo": (Color.photo, Color.photo_fill),
        "attached_item": (Color.attached, Color.attached_fill),
        "weights": (Color.weights, Color.weights_fill),
        "leaf_whole": (Color.leaf_whole, Color.leaf_whole_fill),
        "leaf_partial": (Color.leaf_partial, Color.leaf_partial_fill),
        "leaflet": (Color.leaflet, Color.leaflet_fill),
        "seed_fruit_one": (Color.seed_fruit_one, Color.seed_fruit_one_fill),
        "seed_fruit_many": (Color.seed_fruit_many, Color.seed_fruit_many_fill),
        "flower_one": (Color.flower_one, Color.flower_one_fill),
        "flower_many": (Color.flower_many, Color.flower_many_fill),
        "bud": (Color.bud, Color.bud_fill),
        "specimen": (Color.specimen, Color.specimen_fill),
        "roots": (Color.roots, Color.roots_fill),
        "wood": (Color.wood, Color.wood_fill),
        "seg_leaf_whole": (Color.seg_leaf_whole , Color.seg_leaf_whole_fill),
        "seg_leaf_partial": (Color.seg_leaf_partial , Color.seg_leaf_partial_fill),
        "seg_leaf_whole_petiole": (Color.seg_leaf_whole_petiole , Color.seg_leaf_whole_petiole_fill),
        "seg_leaf_partial_petiole": (Color.seg_leaf_partial_petiole , Color.seg_leaf_partial_petiole_fill),
        "seg_hole": (Color.seg_hole , Color.seg_hole_fill)
    }
    return color_dict[anno.lower()][0], color_dict[anno.lower()][1]

def set_index_for_annotation(cls,annoType):
    if annoType == 'PLANT':
        if cls == 0:
            annoInd = 'Leaf_WHOLE'
        elif cls == 1:
            annoInd = 'Leaf_PARTIAL'
        elif cls == 2:
            annoInd = 'Leaflet'
        elif cls == 3:
            annoInd = 'Seed_Fruit_ONE'
        elif cls == 4:
            annoInd = 'Seed_Fruit_MANY'
        elif cls == 5:
            annoInd = 'Flower_ONE'
        elif cls == 6:
            annoInd = 'Flower_MANY'
        elif cls == 7:
            annoInd = 'Bud'
        elif cls == 8:
            annoInd = 'Specimen'
        elif cls == 9:
            annoInd = 'Roots'
        elif cls == 10:
            annoInd = 'Wood'
    elif annoType == 'ARCHIVAL':
        if cls == 0:
            annoInd = 'Ruler'
        elif cls == 1:
            annoInd = 'Barcode'
        elif cls == 2:
            annoInd = 'Colorcard'
        elif cls == 3:
            annoInd = 'Label'
        elif cls == 4:
            annoInd = 'Map'
        elif cls == 5:
            annoInd = 'Envelope'
        elif cls == 6:
            annoInd = 'Photo'
        elif cls == 7:
            annoInd = 'Attached_item'
        elif cls == 8:
            annoInd = 'Weights'
    return annoInd.lower()