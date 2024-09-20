import os, cv2, yaml, math, sys, inspect, imutils, random, copy, sqlite3, glob, shutil, multiprocessing
import numpy as np
from numpy import NAN, ndarray
import pandas as pd
from dataclasses import dataclass,field
from scipy import ndimage,stats
from scipy.signal import find_peaks
from scipy.stats.mstats import gmean
from scipy.spatial.distance import pdist, squareform
from skimage.measure import label, regionprops_table
from skimage.util import crop
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Manager
import concurrent.futures
from threading import Lock
import torch
import os, argparse, time, copy, cv2, wandb
import torch
from torchvision import *
from sklearn.cluster import KMeans
import statistics
import csv, logging
from threading import Thread
from queue import Queue
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from time import perf_counter
from binarize_image_ML import DocEnTR
from multiprocessing import Process, Queue
from queue import Empty  # Correct import from queue
from PIL import Image
currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)
sys.path.append(parentdir2)
sys.path.append(currentdir)
# from machine.general_utils import print_plain_to_console, print_blue_to_console, print_green_to_console, print_warning_to_console, print_cyan_to_console
# from machine.general_utils import bcolors
from leafmachine2.analysis.predict_pixel_to_metric_conversion_factor import PolynomialModel
from leafmachine2.machine.LM2_logger import start_worker_logging, merge_worker_logs



def convert_rulers_testing(dir_rulers, cfg, time_report, logger, dir_home, Project, batch, Dirs):
    RulerCFG = RulerConfig(logger, dir_home, Dirs, cfg, device)
    Labels = DocEnTR()
    model, device = Labels.load_DocEnTR_model()

    acc_total = 0
    acc_error = 0
    acc_correct = 0
    incorrect_pair = []

    # For reference images
    dir_out = 'F:/Ruler_Reference_Processed_400px'
    csv_out = 'D:/Dropbox/LM2_Env/LeafMachine2_Manuscript/ruler_QC/ruler_acc.csv'
    acc_out_good = {}
    acc_out_bad = {}
    acc_out_total = {}

    for subdir, _, files in os.walk(dir_rulers):
        for img_name in files:
            true_class = os.path.basename(subdir)
            # print(true_class)
            if true_class != 'fail':
                path_img = os.path.join(subdir, img_name)
                # print(path_img)

                ruler_cropped = cv2.imread(path_img)
                ruler_crop_name = img_name.split('.')[0]

                Ruler = setup_ruler(Labels, model, device, cfg, Dirs, logger, RulerCFG, ruler_cropped, ruler_crop_name)
                message = ''.join(["True Class: ", str(true_class), "    Pred Class: ",Ruler.ruler_class_pred])
                
                # For reference images
                if Ruler.ruler_class not in acc_out_total:
                    if Ruler.ruler_class != true_class:
                        if 'fail' in Ruler.ruler_class.split('_'):
                            try:
                                RC = Ruler.ruler_class.split('__')[1]
                            except:
                                RC = 'fail'
                        else:
                            RC =  true_class
                    else:
                        RC =  Ruler.ruler_class
                if RC not in acc_out_total:
                    acc_out_good[RC] = 0
                    acc_out_bad[RC] = 0
                    acc_out_total[RC] = 0

                if Ruler.ruler_class != true_class:
                    acc_total += 1
                    acc_error += 1
                    incorrect_pair.append([true_class, Ruler.ruler_class])

                    acc_out_bad[RC] += 1
                    acc_out_total[RC] +=1


                    Print_Verbose(RulerCFG.cfg,1,message).warning()
                else:
                    acc_total += 1
                    acc_correct += 1

                    acc_out_good[RC] += 1
                    acc_out_total[RC] +=1

                    Print_Verbose(RulerCFG.cfg,1,message).green()


                Ruler_Info = convert_pixels_to_metric(logger, RulerCFG,Ruler,ruler_crop_name, Dirs)
                print('hi')

                # For reference images
                try: 
                    os.makedirs(os.path.join(dir_out, true_class), exist_ok=True)
                    rs = Ruler_Info.summary_image
                    h, w, c = rs.shape
                    new_h = 1000
                    new_w = int(w * (new_h / h))
                    # resize the image and save it
                    rs = cv2.resize(rs, (new_w, new_h))
                    cv2.imwrite(os.path.join(dir_out, true_class, '.'.join([ruler_crop_name,'jpg'])), rs)
                except:
                    pass # failed ruler

            # Project = add_ruler_to_Project(Project, batch, Ruler, BlockCandidate, filename, ruler_crop_name)
    keys = acc_out_total.keys()
    with open(csv_out, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Key', 'acc_out_good', 'acc_out_bad', 'acc_out_total'])
        for key in keys:
            row = [key, acc_out_good[key], acc_out_bad[key], acc_out_total[key]]
            writer.writerow(row)

    print(f"Total = {acc_total} Error = {acc_error} Correct = {acc_correct}")
    print(f"Accuracy = {acc_correct/acc_total}")
    print(f"True / Incorrect: {incorrect_pair}")
    return Project, time_report



# def parallel_convert_rulers(cfg, logger, dir_home, Project, batch, Dirs):
#     t1_start = perf_counter()
#     logger.info(f"Converting Rulers in batch {batch+1}")
    
#     num_workers = int(cfg['leafmachine']['project'].get('num_workers', 4))

#     # Split the keys of Project.project_data_list[batch] among workers
#     keys = list(Project.project_data_list[batch].keys())
#     chunks = [keys[i:i+num_workers] for i in range(0, len(keys), num_workers)]

#     # Use a shared dictionary to store the results
#     manager = Manager()
#     results_dict = manager.dict()

#     # Define a partial function to process a single chunk of files
#     process_chunk = partial(process_ruler_chunk, cfg, logger, dir_home, Project, batch, Dirs, results_dict)

#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         # Submit all the chunks for processing
#         futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

#         # Wait for all the futures to complete
#         for future in futures:
#             future.result()

#     # Update the Project object with the results
#     for filename, results in results_dict.items():
#         Project.project_data_list[batch][filename]['Ruler_Info'] = results
#     t1_stop = perf_counter()
#     logger.info(f"Converting Rulers in batch {batch+1} --- elapsed time: {round(t1_stop - t1_start)} seconds")
#     return Project


def process_ruler_chunk(cfg, logger, dir_home, Project, batch, Dirs, results_dict, keys):
    RulerCFG = RulerConfig(logger, dir_home, Dirs, cfg)
    Labels = DocEnTR()
    model, device = Labels.load_DocEnTR_model()
    
    for filename in keys:
        analysis = Project.project_data_list[batch][filename]
        if len(analysis) != 0:
            Project.project_data_list[batch][filename]['Ruler_Info'] = []
            Project.project_data_list[batch][filename]['Ruler_Data'] = []
            logger.debug(filename)
            try:
                full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpg'])))
            except:
                full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpeg'])))

            try:
                archival = analysis['Detections_Archival_Components']
                has_rulers = True
            except: 
                has_rulers = False

            if has_rulers:
                height = analysis['height']
                width = analysis['width']
                ruler_list = [row for row in archival if row[0] == 0]
                # print(ruler_list)
                if len(ruler_list) < 1:
                    logger.debug('no rulers detected')
                else:
                    for ruler in ruler_list:
                        ruler_location = yolo_to_position_ruler(ruler, height, width)
                        ruler_polygon = [(ruler_location[1], ruler_location[2]), (ruler_location[3], ruler_location[2]), (ruler_location[3], ruler_location[4]), (ruler_location[1], ruler_location[4])]
                        # print(ruler_polygon)
                        x_coords = [x for x, y in ruler_polygon]
                        y_coords = [y for x, y in ruler_polygon]

                        min_x, min_y = min(x_coords), min(y_coords)
                        max_x, max_y = max(x_coords), max(y_coords)

                        ruler_cropped = full_image[min_y:max_y, min_x:max_x]
                        # img_crop = img[min_y:max_y, min_x:max_x]
                        loc = '-'.join([str(min_x), str(min_y), str(max_x), str(max_y)])
                        ruler_crop_name = '__'.join([filename,'R',loc])
                        
                        # Process ruler in a separate function
                        ruler_info = process_ruler_single(cfg, logger, dir_home, Project, batch, Dirs, Labels, model, device, RulerCFG, ruler_cropped, ruler_crop_name)

                        # Append the ruler info to the Project object
                        Project.project_data_list[batch][filename]['Ruler_Info'].append(ruler_info)
    return Project

def process_ruler_single(cfg, logger, dir_home, Project, batch, Dirs, Labels, model, device, RulerCFG, ruler_cropped, ruler_crop_name):
    Ruler = setup_ruler(Labels, model, device, cfg, Dirs, logger, RulerCFG, ruler_cropped, ruler_crop_name)
    Ruler_Info = convert_pixels_to_metric(logger, RulerCFG, Ruler, ruler_crop_name, Dirs)

    if any(unit in Ruler_Info.conversion_data_all for unit in ['smallCM', 'halfCM', 'mm']):
        units_save = Ruler_Info.conversion_data_all
        if units_save == []:
            units_save = 0
        plot_points = Ruler_Info.data_list
    else:
        units_save = Ruler_Info.unit_list 
        plot_points = 0 

    try:
        ruler_info = {
            'ruler_image_name' : ruler_crop_name,
            'success' : Ruler_Info.conversion_successful ,
            'conversion_mean' :  Ruler_Info.conversion_mean ,
            'pooled_sd' :  Ruler_Info.pooled_sd ,
            'ruler_class' :  Ruler_Info.ruler_class ,
            'ruler_class_confidence': Ruler_Info.ruler_class_percentage,
            'units' :  units_save ,
            'cross_validation_count' :  Ruler_Info.cross_validation_count,
            'n_scanlines' :  Ruler_Info.conversion_mean_n,
            'n_data_points_in_avg' :  Ruler_Info.conversion_mean_n_vals ,
            'avg_tick_width' :  Ruler_Info.avg_width ,
            'plot_points' : plot_points,
            'summary_img' : Ruler_Info.summary_image
        }
    except:
        ruler_info = {
            'ruler_image_name' : ruler_crop_name,
            'success' : False ,
            'conversion_mean' :  0 ,
            'pooled_sd' :  0 ,
            'ruler_class' :  'fail' ,
            'ruler_class_confidence': 0,
            'units' :  0,
            'cross_validation_count' :  0,
            'n_scanlines' :  0,
            'n_data_points_in_avg' :  0 ,
            'avg_tick_width' :  0 ,
            'plot_points' : 0,
            'summary_img' : ruler_cropped
        }

    return ruler_info


def calc_MP(full_image):
    # Get image dimensions
    height, width = full_image.shape[:2]
    
    # Calculate megapixels
    megapixels = (height * width) / 1e6
    return megapixels

''' 5/9/2024 Works, but trying parallele'''
# def convert_rulers(cfg, logger, dir_home, Project, batch, Dirs):
#     t1_start = perf_counter()
#     logger.info(f"Converting Rulers in batch {batch+1}")
#     RulerCFG = RulerConfig(logger, dir_home, Dirs, cfg)
#     Labels = DocEnTR()
#     model, device = Labels.load_DocEnTR_model(logger)


#     poly_model = PolynomialModel()
#     poly_model.load_polynomial_model()
#     use_CF_predictor = cfg['leafmachine']['project']['use_CF_predictor']
#     logger.info(f"use_CF_predictor: cfg['leafmachine']['project']['use_CF_predictor'] = {use_CF_predictor}")
    

#     for filename, analysis in Project.project_data_list[batch].items():
#         if len(analysis) != 0:
#             Project.project_data_list[batch][filename]['Ruler_Info'] = []
#             Project.project_data_list[batch][filename]['Ruler_Data'] = []
#             logger.debug(filename)
#             try:
#                 full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpg'])))
#             except:
#                 full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpeg'])))

            
#             MP_value = calc_MP(full_image)
#             predicted_conversion_factor_cm = poly_model.predict_with_polynomial_single(MP_value)
#             logger.info(f"      Predicted conversion mean for MP={MP_value}: {predicted_conversion_factor_cm}")


#             try:
#                 archival = analysis['Detections_Archival_Components']
#                 has_rulers = True
#             except: 
#                 has_rulers = False

#             if has_rulers:
#                 height = analysis['height']
#                 width = analysis['width']
#                 ruler_list = [row for row in archival if row[0] == 0]
#                 # print(ruler_list)
#                 if len(ruler_list) < 1:
#                     logger.debug('no rulers detected')
#                 else:
#                     for ruler in ruler_list:
#                         ruler_location = yolo_to_position_ruler(ruler, height, width)
#                         ruler_polygon = [(ruler_location[1], ruler_location[2]), (ruler_location[3], ruler_location[2]), (ruler_location[3], ruler_location[4]), (ruler_location[1], ruler_location[4])]
#                         # print(ruler_polygon)
#                         x_coords = [x for x, y in ruler_polygon]
#                         y_coords = [y for x, y in ruler_polygon]

#                         min_x, min_y = min(x_coords), min(y_coords)
#                         max_x, max_y = max(x_coords), max(y_coords)

#                         ruler_cropped = full_image[min_y:max_y, min_x:max_x]
#                         # img_crop = img[min_y:max_y, min_x:max_x]
#                         loc = '-'.join([str(min_x), str(min_y), str(max_x), str(max_y)])
#                         ruler_crop_name = '__'.join([filename,'R',loc])

#                         # Get the cropped image using cv2.getRectSubPix
#                         # ruler_cropped = cv2.getRectSubPix(full_image, (int(ruler_location[3] - ruler_location[1]), int(ruler_location[4] - ruler_location[2])), (points[0][0][0], points[0][0][1]))

#                         Ruler = setup_ruler(Labels, model, device, cfg, Dirs, logger, RulerCFG, ruler_cropped, ruler_crop_name)

#                         Ruler_Info = convert_pixels_to_metric(logger, RulerCFG, Ruler, ruler_crop_name, predicted_conversion_factor_cm, use_CF_predictor, Dirs)

#                         '''
#                         **************************************
#                         **************************************
#                         **************************************
#                         **************************************
#                         FINISH THIS. NEED TO EXPORT THE DATA
#                         **************************************
#                         **************************************
#                         **************************************
#                         **************************************
#                         '''
#                         if any(unit in Ruler_Info.conversion_data_all for unit in ['smallCM', 'halfCM', 'mm']):
#                             units_save = Ruler_Info.conversion_data_all
#                             if units_save == []:
#                                 units_save = 0
#                             plot_points = Ruler_Info.data_list
#                         else:
#                             units_save = Ruler_Info.unit_list ##################
#                             plot_points = 0 ######################

#                         try: # TODO there are some un accounted for variable that are not initialized empty i.e. Ruler_Info.conversion_successful
#                             Project.project_data_list[batch][filename]['Ruler_Info'].append({
#                                 'ruler_image_name' : ruler_crop_name,
#                                 'success' : Ruler_Info.conversion_successful ,
#                                 'conversion_mean' :  Ruler_Info.conversion_mean ,
#                                 'predicted_conversion_factor_cm': predicted_conversion_factor_cm,
#                                 'pooled_sd' :  Ruler_Info.pooled_sd ,
#                                 'ruler_class' :  Ruler_Info.ruler_class ,
#                                 'ruler_class_confidence': Ruler_Info.ruler_class_percentage,
#                                 'units' :  units_save ,
#                                 'cross_validation_count' :  Ruler_Info.cross_validation_count,
#                                 'n_scanlines' :  Ruler_Info.conversion_mean_n,
#                                 'n_data_points_in_avg' :  Ruler_Info.conversion_mean_n_vals ,
#                                 'avg_tick_width' :  Ruler_Info.avg_width ,
#                                 'plot_points' : plot_points,
#                                 'summary_img' : Ruler_Info.summary_image
#                             })
#                         except:
#                             Project.project_data_list[batch][filename]['Ruler_Info'].append({
#                                 'ruler_image_name' : ruler_crop_name,
#                                 'success' : False ,
#                                 'conversion_mean' :  0 ,
#                                 'predicted_conversion_factor_cm': predicted_conversion_factor_cm,
#                                 'pooled_sd' :  0 ,
#                                 'ruler_class' :  'fail' ,
#                                 'ruler_class_confidence': 0,
#                                 'units' :  0,
#                                 'cross_validation_count' :  0,
#                                 'n_scanlines' :  0,
#                                 'n_data_points_in_avg' :  0 ,
#                                 'avg_tick_width' :  0 ,
#                                 'plot_points' : 0,
#                                 'summary_img' : ruler_cropped
#                             })
#                         # Project = add_ruler_to_Project(Project, batch, Ruler, BlockCandidate, filename, ruler_crop_name) 
       
#     t1_stop = perf_counter()
#     logger.info(f"Converting Rulers in batch {batch+1} --- elapsed time: {round(t1_stop - t1_start)} seconds")
#     return Project

def convert_rulers(cfg, time_report, logger, dir_home, ProjectSQL, batch, batch_filenames, Dirs, batch_size=100):
    num_workers = cfg['leafmachine']['project']['num_workers_ruler']

    if cfg['leafmachine']['project']['device'] == 'cuda':
        num_gpus = int(cfg['leafmachine']['project'].get('num_gpus', 1))  # Default to 1 GPU if not specified
        device_list = [i for i in range(num_gpus)]  # List of GPU indices like [0, 1, 2, ...]
    else:
        device_list = ['cpu']  # Default to CPU if CUDA is not available
    
    t1_start = perf_counter()
    logger.info(f"Converting Rulers in batch {batch+1}")

    filenames = batch_filenames
    num_files = len(filenames)
    chunk_size = max((num_files + num_workers - 1) // num_workers, 4)

    queue = Queue()
    result_queue = Queue()
    status_queue = Queue()  # New status queue for heartbeat messages
    workers = []

    worker_id = 0  # Unique worker ID

    # Assign workers in a round-robin fashion across devices
    for worker_id in range(num_workers):
        device = device_list[worker_id % len(device_list)]  # Cycle through available devices
        p = Process(target=worker, args=(queue, result_queue, status_queue, worker_id, cfg, dir_home, ProjectSQL.database, ProjectSQL.dir_images, Dirs, device))
        p.start()
        workers.append(p)
        logger.info(f"Starting worker {worker_id} on device {device}")

    # Start monitoring worker status in a separate process
    monitor_process = Process(target=monitor_worker_status, args=(status_queue, num_workers))
    monitor_process.start()

    # Enqueue work
    for i in range(0, num_files, chunk_size):
        queue.put(filenames[i:i + chunk_size])

    for _ in range(num_workers):
        queue.put(None)  # send stop signal to all workers

    all_results = []
    completed_workers = 0
    while completed_workers < num_workers:
        try:
            result = result_queue.get()
            if result is None:
                completed_workers += 1
            else:
                all_results.extend(result)

        except Empty:  # Correct usage for handling empty queue timeout
            logger.error("Timeout: One or more workers took too long to respond.")
            break

    # Check if any workers have hung
    timed_out_worker = monitor_process.join()
    if timed_out_worker is not None:
        logger.warning(f"Worker {timed_out_worker} has timed out and will be terminated.")
        for p in workers:
            if p.pid == timed_out_worker:
                p.terminate()  # Terminate the timed-out worker

    # Wait for all workers to finish
    for p in workers:
        p.join()


    monitor_process.terminate()  # Stop the monitor process

    # After all workers are done, insert data into the database in chunks
    conn = sqlite3.connect(ProjectSQL.database)
    cur = conn.cursor()

    # Chunk insert to avoid write conflicts
    for i in range(0, len(all_results), batch_size):
        batch_data = all_results[i:i + batch_size]
        cur.executemany("""
            INSERT INTO ruler_data (file_name, ruler_image_name, success, conversion_mean, predicted_conversion_factor_cm, pooled_sd, ruler_class, 
                                    ruler_class_confidence, units, cross_validation_count, n_scanlines, n_data_points_in_avg, avg_tick_width, plot_points)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", batch_data)
        conn.commit()

    conn.close()

    merge_worker_logs(Dirs, num_workers, main_log_name='ruler_logs', log_name_stem='worker_log_ruler')

    t1_stop = perf_counter()
    t_rulers = f"[Converting Rulers elapsed time] {round(t1_stop - t1_start)} seconds ({round((t1_stop - t1_start)/60)} minutes)"
    logger.info(t_rulers)
    time_report['t_rulers'] = t_rulers
    return ProjectSQL, time_report

# Move the worker function to the top level
def worker(queue, result_queue, status_queue, worker_id, cfg, dir_home, ProjectSQL, dir_images, Dirs, device):
    try:
        RulerCFG = RulerConfig(dir_home, Dirs, cfg, device)
        Labels = DocEnTR()
        model, __device = Labels.load_DocEnTR_model(device)  # Load model on each process
        poly_model = PolynomialModel()
        poly_model.load_polynomial_model()
        wlogger, wlogger_path = start_worker_logging(worker_id, Dirs, log_name='worker_log_ruler')

        use_CF_predictor = cfg['leafmachine']['project']['use_CF_predictor']

        while True:
            filenames_chunk = queue.get()
            if filenames_chunk is None:  # Stop signal
                break

            results = []
            for filename in filenames_chunk:
                # Process the filename (this is the long task)
                results.extend(process_filename(wlogger, filename, cfg, dir_home, ProjectSQL, dir_images, Dirs, RulerCFG, Labels, model, device, poly_model, use_CF_predictor))

                # Send a heartbeat to the status queue after each filename
                status_queue.put((worker_id, time.time()))  # Send worker id and current timestamp

            result_queue.put(results)  # Send the results back to the main process

    except Exception as e:
        wlogger.error(f"Error in worker {worker_id}: {str(e)}")

    finally:
        result_queue.put(None)  # Signal that this worker has finished
        wlogger.info("Clearing GPU memory in worker...")
        del model, poly_model
        torch.cuda.empty_cache()

def monitor_worker_status(status_queue, worker_count, timeout=600):
    """
    Monitor the workers' heartbeats and terminate if any worker is unresponsive.
    """
    worker_last_seen = {i: time.time() for i in range(worker_count)}

    while True:
        try:
            worker_id, heartbeat = status_queue.get(timeout=1)  # Check the status queue for updates
            worker_last_seen[worker_id] = heartbeat  # Update the last seen time for this worker
        except multiprocessing.queues.Empty:
            # No heartbeats received in the last second, check if any workers have timed out
            current_time = time.time()
            for worker_id, last_seen in worker_last_seen.items():
                if current_time - last_seen > timeout:
                    return worker_id  # This worker has timed out

def process_filename(wlogger, filename, cfg, dir_home, ProjectSQL, dir_images, Dirs, RulerCFG, Labels, model, device, poly_model, use_CF_predictor):
    # This function no longer inserts data directly into the database.
    # Instead, it returns the data to be inserted later by the main process.
    
    results = []  # Collect results here
    
    try:
        # Create a new connection in this worker process
        conn = sqlite3.connect(ProjectSQL)
        cur = conn.cursor()

        # Retrieve image dimensions from the images table
        cur.execute("SELECT width, height FROM images WHERE name = ?", (filename,))
        dimensions = cur.fetchone()
        if not dimensions:
            print(f"No dimensions found for {filename}")
            conn.close()
            return []
        width, height = dimensions

        # Retrieve archival components from annotations_archival table
        cur.execute("SELECT annotation FROM annotations_archival WHERE file_name = ?", (filename,))
        archival_components = cur.fetchall()

        # Process and structure the archival components
        archival_data = []
        for component in archival_components:
            component_data = list(map(float, component[0].split(',')))
            archival_data.append(component_data)

        analysis = {
            'Detections_Archival_Components': archival_data,
            'height': height,
            'width': width
        }

        if len(analysis) != 0:
            print(f"Processing file: {filename}")

            try:
                image_path = glob.glob(os.path.join(dir_images, filename + '.*'))[0]
                full_image = cv2.imread(image_path)
            except:
                raise FileNotFoundError(f"Could not load image for {filename}")

            # Predict conversion factor using the polynomial model
            MP_value = calc_MP(full_image)
            predicted_conversion_factor_cm = poly_model.predict_with_polynomial_single(MP_value)
            wlogger.info(f"Predicted conversion mean for MP={MP_value}: {predicted_conversion_factor_cm}")

            archival = analysis.get('Detections_Archival_Components', [])
            has_rulers = len(archival) > 0

            if has_rulers:
                height, width = analysis['height'], analysis['width']
                ruler_list = [row for row in archival if row[0] == 0]
                if len(ruler_list) < 1:
                    wlogger.info('No rulers detected')
                else:
                    for ruler in ruler_list:
                        ruler_location = yolo_to_position_ruler(ruler, height, width)
                        ruler_polygon = [
                            (ruler_location[1], ruler_location[2]),
                            (ruler_location[3], ruler_location[2]),
                            (ruler_location[3], ruler_location[4]),
                            (ruler_location[1], ruler_location[4])
                        ]
                        x_coords, y_coords = zip(*ruler_polygon)
                        min_x, min_y = min(x_coords), min(y_coords)
                        max_x, max_y = max(x_coords), max(y_coords)
                        ruler_cropped = full_image[min_y:max_y, min_x:max_x]
                        loc = '-'.join(map(str, [min_x, min_y, max_x, max_y]))
                        ruler_crop_name = '__'.join([filename, 'R', loc])

                        Ruler = setup_ruler(Labels, model, device, cfg, Dirs, wlogger, False, RulerCFG, ruler_cropped, ruler_crop_name)
                        Ruler_Info = convert_pixels_to_metric(wlogger, False, RulerCFG, Ruler, ruler_crop_name, predicted_conversion_factor_cm, use_CF_predictor, Dirs)

                        units_save = Ruler_Info.conversion_data_all if Ruler_Info.conversion_data_all else 0
                        plot_points = Ruler_Info.data_list if any(unit in Ruler_Info.conversion_data_all for unit in ['smallCM', 'halfCM', 'mm']) else 0
                        plot_points_blob = np.array(plot_points).tobytes() if plot_points else None

                        # Collect the result as a tuple
                        result = (
                            filename, ruler_crop_name, Ruler_Info.conversion_successful, Ruler_Info.conversion_mean, 
                            predicted_conversion_factor_cm, Ruler_Info.pooled_sd, Ruler_Info.ruler_class, 
                            Ruler_Info.ruler_class_percentage, str(units_save), Ruler_Info.cross_validation_count,
                            Ruler_Info.conversion_mean_n, Ruler_Info.conversion_mean_n_vals, Ruler_Info.avg_width, plot_points_blob
                        )
                        results.append(result)

        conn.close()

    except Exception as e:
        print(f"Failed to process {filename}: {e}")

    return results  # Return the results to the worker
############Replaced with version that has own models per process
# def convert_rulers(cfg, time_report, logger, dir_home, ProjectSQL, batch, batch_filenames, Dirs, num_workers=16):

#     t1_start = perf_counter()
#     logger.info(f"Converting Rulers in batch {batch+1}")

#     # create_ruler_data_table(ProjectSQL.conn)

#     show_all_logs = False

#     # Load shared resources outside the loop
#     RulerCFG = RulerConfig(logger, dir_home, Dirs, cfg)
#     Labels = DocEnTR()
#     model, device = Labels.load_DocEnTR_model(logger)
#     poly_model = PolynomialModel()
#     poly_model.load_polynomial_model()

#     use_CF_predictor = cfg['leafmachine']['project']['use_CF_predictor']
#     num_workers = cfg['leafmachine']['project']['num_workers_ruler']
    
#     logger.info(f"use_CF_predictor: {use_CF_predictor}")

#     filenames = batch_filenames  # Using the list of filenames passed as argument
#     num_files = len(filenames)
#     chunk_size = max((num_files + num_workers - 1) // num_workers, 4)

#     def worker(queue):
#         while True:
#             filenames_chunk = queue.get()
#             if filenames_chunk is None:  # Stop signal
#                 break

#             for filename in filenames_chunk:
#                 process_filename(filename, cfg, logger, show_all_logs, dir_home, ProjectSQL, Dirs, RulerCFG, Labels, model, device, poly_model, use_CF_predictor)


#     # Setup queue and start workers
#     queue = Queue()
#     workers = []
#     for _ in range(num_workers):
#         t = Thread(target=worker, args=(queue,))
#         t.start()
#         workers.append(t)

#     # Enqueue work
#     for i in range(0, num_files, chunk_size):
#         queue.put(filenames[i:i + chunk_size])

#     # Wait for all work to be done
#     queue.join()

#     # Stop workers
#     for _ in range(num_workers):
#         queue.put(None)
#     for t in workers:
#         t.join()

#     t1_stop = perf_counter()
#     t_rulers = f"[Converting Rulers elapsed time] {round(t1_stop - t1_start)} seconds ({round((t1_stop - t1_start)/60)} minutes)"
#     logger.info(t_rulers)
#     time_report['t_rulers'] = t_rulers
#     return ProjectSQL, time_report


# def create_ruler_data_table(conn):
#     try:
#         cur = conn.cursor()
#         sql_create_ruler_data_table = """
#         CREATE TABLE IF NOT EXISTS ruler_data (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             file_name TEXT NOT NULL,
#             ruler_image_name TEXT,
#             success BOOLEAN,
#             conversion_mean REAL,
#             predicted_conversion_factor_cm REAL,
#             pooled_sd REAL,
#             ruler_class TEXT,
#             ruler_class_confidence REAL,
#             units TEXT,
#             cross_validation_count INTEGER,
#             n_scanlines INTEGER,
#             n_data_points_in_avg INTEGER,
#             avg_tick_width REAL,
#             plot_points BLOB,
#             summary_img BLOB
#         );"""
#         cur.execute(sql_create_ruler_data_table)
#         conn.commit()
#     except sqlite3.Error as e:
#         print(f"Error creating ruler_data table: {e}")


##############Replaced with versio nthat has one moidel per process
# def process_filename(filename, cfg, logger, show_all_logs, dir_home, ProjectSQL, Dirs, RulerCFG, Labels, model, device, poly_model, use_CF_predictor):
#     # Create a new connection in this thread
#     conn = sqlite3.connect(ProjectSQL.database)
#     cur = conn.cursor()

#     # Retrieve image dimensions from the images table
#     cur.execute("SELECT width, height FROM images WHERE name = ?", (filename,))
#     dimensions = cur.fetchone()
#     if not dimensions:
#         logger.error(f"No dimensions found for {filename}")
#         conn.close()
#         return
#     width, height = dimensions

#     # Retrieve archival components from annotations_archival table
#     cur.execute("SELECT annotation FROM annotations_archival WHERE file_name = ?", (filename,))
#     archival_components = cur.fetchall()

#     # Process and structure the archival components
#     archival_data = []
#     for component in archival_components:
#         component_data = list(map(float, component[0].split(',')))
#         archival_data.append(component_data)

#     # Mimic the structure of the previous `analysis` dictionary
#     analysis = {
#         'Detections_Archival_Components': archival_data,
#         'height': height,
#         'width': width
#     }

#     if len(analysis) != 0:
#         logger.debug(filename)

#         # Try to load the image
#         try:
#             image_path = glob.glob(os.path.join(ProjectSQL.dir_images, filename + '.*'))[0]
#             full_image = cv2.imread(image_path)
#         except:
#             raise FileNotFoundError(f"Could not load image for {filename}")

#         # Calculate MP value and predict conversion factor
#         MP_value = calc_MP(full_image)
#         predicted_conversion_factor_cm = poly_model.predict_with_polynomial_single(MP_value)
#         logger.info(f"Predicted conversion mean for MP={MP_value}: {predicted_conversion_factor_cm}")

#         # Detect rulers
#         archival = analysis.get('Detections_Archival_Components', [])
#         has_rulers = len(archival) > 0

#         if has_rulers:
#             height, width = analysis['height'], analysis['width']
#             ruler_list = [row for row in archival if row[0] == 0]
#             if len(ruler_list) < 1:
#                 logger.debug('No rulers detected')
#             else:
#                 for ruler in ruler_list:
#                     # Process each ruler found
#                     ruler_location = yolo_to_position_ruler(ruler, height, width)
#                     ruler_polygon = [
#                         (ruler_location[1], ruler_location[2]), 
#                         (ruler_location[3], ruler_location[2]), 
#                         (ruler_location[3], ruler_location[4]), 
#                         (ruler_location[1], ruler_location[4])
#                     ]
#                     x_coords, y_coords = zip(*ruler_polygon)
#                     min_x, min_y = min(x_coords), min(y_coords)
#                     max_x, max_y = max(x_coords), max(y_coords)
#                     ruler_cropped = full_image[min_y:max_y, min_x:max_x]
#                     loc = '-'.join(map(str, [min_x, min_y, max_x, max_y]))
#                     ruler_crop_name = '__'.join([filename, 'R', loc])

#                     Ruler = setup_ruler(Labels, model, device, cfg, Dirs, logger, show_all_logs, RulerCFG, ruler_cropped, ruler_crop_name)
#                     Ruler_Info = convert_pixels_to_metric(logger, show_all_logs, RulerCFG, Ruler, ruler_crop_name, predicted_conversion_factor_cm, use_CF_predictor, Dirs)

#                     # Collect ruler data
#                     try:
#                         units_save = Ruler_Info.conversion_data_all if Ruler_Info.conversion_data_all else 0
#                         plot_points = Ruler_Info.data_list if any(unit in Ruler_Info.conversion_data_all for unit in ['smallCM', 'halfCM', 'mm']) else 0

#                         # summary_img = cv2.imencode('.jpg', Ruler_Info.summary_image)[1].tobytes() if Ruler_Info.summary_image is not None else None
#                         plot_points_blob = np.array(plot_points).tobytes() if plot_points else None

#                         # Insert into SQL table
#                         # cur.execute("""
#                         # INSERT INTO ruler_data (file_name, ruler_image_name, success, conversion_mean, predicted_conversion_factor_cm, pooled_sd, ruler_class, 
#                         #                         ruler_class_confidence, units, cross_validation_count, n_scanlines, n_data_points_in_avg, avg_tick_width, plot_points, summary_img)
#                         # VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
#                         #             (filename, ruler_crop_name, Ruler_Info.conversion_successful, Ruler_Info.conversion_mean, predicted_conversion_factor_cm,
#                         #              Ruler_Info.pooled_sd, Ruler_Info.ruler_class, Ruler_Info.ruler_class_percentage, str(units_save), Ruler_Info.cross_validation_count,
#                         #              Ruler_Info.conversion_mean_n, Ruler_Info.conversion_mean_n_vals, Ruler_Info.avg_width, plot_points_blob, summary_img))
#                         cur.execute("""
#                         INSERT INTO ruler_data (file_name, ruler_image_name, success, conversion_mean, predicted_conversion_factor_cm, pooled_sd, ruler_class, 
#                                                 ruler_class_confidence, units, cross_validation_count, n_scanlines, n_data_points_in_avg, avg_tick_width, plot_points)
#                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
#                                     (filename, ruler_crop_name, Ruler_Info.conversion_successful, Ruler_Info.conversion_mean, predicted_conversion_factor_cm,
#                                      Ruler_Info.pooled_sd, Ruler_Info.ruler_class, Ruler_Info.ruler_class_percentage, str(units_save), Ruler_Info.cross_validation_count,
#                                      Ruler_Info.conversion_mean_n, Ruler_Info.conversion_mean_n_vals, Ruler_Info.avg_width, plot_points_blob))

#                         conn.commit()

#                     except Exception as e:
#                         logger.error(f"Failed to process ruler data for {filename}: {str(e)}")




def convert_pixels_to_metric(logger, show_all_logs, RulerCFG, Ruler, img_fname, predicted_conversion_factor_cm, use_CF_predictor, Dirs):#cfg,Ruler,imgPath,fName,dirSave,dir_ruler_correction,pathToModel,labelNames):
    Ruler_Redo = Ruler

    Ruler_Info = RulerInfo(Ruler, predicted_conversion_factor_cm, use_CF_predictor, logger, show_all_logs)

    
    
    if Ruler_Info.is_ticks_only:
        Ruler = straighten_img(logger, show_all_logs, RulerCFG, Ruler, True, False, Dirs, Ruler_Info.is_inline, Ruler_Info.is_block_tick, Ruler_Info.do_skip_morph_cleaning)
        Ruler_Info.summary_message = Ruler.summary_message
        Ruler_Info = convert_ticks(logger, show_all_logs, Ruler_Info, RulerCFG, Ruler, img_fname, is_redo=False)

    elif Ruler_Info.is_block_tick:
        Ruler = straighten_img(logger, show_all_logs, RulerCFG, Ruler, True, False, Dirs, Ruler_Info.is_inline, Ruler_Info.is_block_tick, Ruler_Info.do_skip_morph_cleaning)
        Ruler_Info.summary_message = Ruler.summary_message
        Ruler_Info = convert_ticks(logger, show_all_logs, Ruler_Info, RulerCFG, Ruler, img_fname, is_redo=False)

    elif Ruler_Info.is_block_only:
        if Ruler_Info.is_block_regular:
            Ruler = straighten_img(logger, show_all_logs, RulerCFG, Ruler, True, False, Dirs, Ruler_Info.is_inline, Ruler_Info.is_block_tick, True)
            Ruler_Info.summary_message = Ruler.summary_message
            if 'white' in Ruler_Info.ruler_class_parts:
                colorOption = 'noinvert'
            elif 'black' in Ruler_Info.ruler_class_parts:
                colorOption = 'invert'
            Ruler_Out, BlockCandidate, summary_image = convert_blocks(logger, show_all_logs, RulerCFG, Ruler, predicted_conversion_factor_cm, use_CF_predictor, colorOption, img_fname, Dirs, is_redo=False)
            # TODO REFACTOR EVERYTHING TO CREATE ONE UNIFIED CLASS
            Ruler_Info = put_BlockCandidate_into_Ruler_Info(Ruler_Info, BlockCandidate, summary_image, logger, show_all_logs)
        else: # handle alternate and stagger
            pass

    elif Ruler_Info.is_grid:
        pass
        # Ruler = straighten_img(logger, show_all_logs, RulerCFG, Ruler, True, False, Dirs, Ruler_Info.is_inline, Ruler_Info.is_block_tick, Ruler_Info.do_skip_morph_cleaning)
        # Ruler_Info = convert_ticks(logger, Ruler_Info, RulerCFG, Ruler, img_fname, is_redo=False)




    '''if check_ruler_type(Ruler.ruler_class,'tick_black'):
        
        colorOption = 'black'
        # colorOption = 'white'
        Ruler = straighten_img(logger, RulerCFG, Ruler, True, False, Dirs)
        Ruler_Out, BlockCandidate = convert_ticks(logger, RulerCFG, Ruler, colorOption, img_fname, is_redo=False)
        if not BlockCandidate['gmean']:
            Ruler_Redo = straighten_img(logger, RulerCFG, Ruler_Redo, True, False, Dirs)
            Ruler_Out, BlockCandidate = convert_ticks(logger, RulerCFG, Ruler_Redo, colorOption, img_fname, is_redo=True)

    elif check_ruler_type(Ruler.ruler_class,'tick_white'):
        colorOption = 'white'
        # colorOption = 'black'
        Ruler = straighten_img(logger, RulerCFG, Ruler, True, False, Dirs)
        Ruler_Out, BlockCandidate = convert_ticks(logger, RulerCFG, Ruler,colorOption, img_fname, is_redo=False)
        if not BlockCandidate['gmean']:
            Ruler_Redo = straighten_img(logger, RulerCFG, Ruler_Redo, True, False, Dirs)
            Ruler_Out, BlockCandidate = convert_ticks(logger, RulerCFG, Ruler_Redo, colorOption, img_fname, is_redo=True)

    elif check_ruler_type(Ruler.ruler_class,'block_regular_cm'):
        colorOption = 'invert'
        Ruler = straighten_img(logger, RulerCFG, Ruler, True, False, Dirs)
        Ruler_Out, BlockCandidate = convert_blocks(logger, RulerCFG, Ruler, colorOption, img_fname, Dirs, is_redo=False)
        if BlockCandidate.conversion_factor <= 0:
            Ruler_Redo = straighten_img(logger, RulerCFG, Ruler_Redo, True, False, Dirs)
            Ruler_Out, BlockCandidate = convert_blocks(logger, RulerCFG, Ruler_Redo, colorOption, img_fname, Dirs, is_redo=True)
    elif check_ruler_type(Ruler.ruler_class,'block_invert_cm'):
        colorOption = 'noinvert'
        Ruler = straighten_img(logger, RulerCFG, Ruler, True, False, Dirs)
        Ruler_Out, BlockCandidate = convert_blocks(logger, RulerCFG, Ruler, colorOption, img_fname, Dirs, is_redo=False)
        if BlockCandidate.conversion_factor <= 0:
            Ruler_Redo = straighten_img(logger, RulerCFG, Ruler_Redo, True, False, Dirs)
            Ruler_Out, BlockCandidate = convert_blocks(logger, RulerCFG, Ruler_Redo, colorOption, img_fname, Dirs, is_redo=True)


    else: # currently unsupported rulers
        Ruler_Out = []
        BlockCandidate = []'''
    
    return Ruler_Info

def put_BlockCandidate_into_Ruler_Info(Ruler_Info, BlockCandidate, summary_image, logger, show_all_logs): # This is temporary. REmove after refactor
    if BlockCandidate.conversion_factor_pass:
        Ruler_Info.summary_image = summary_image
        Ruler_Info.at_least_one_correct_conversion = True
        Ruler_Info.avg_width = 0
        Ruler_Info.best_value = BlockCandidate.conversion_factor #conversion mean

         # diff from scanlines
        
        Ruler_Info.conversion_mean = BlockCandidate.conversion_factor #conversion mean
        
        Ruler_Info.conversion_mean_n = 0
        Ruler_Info.conversion_mean_n_vals = 0
        for pt in BlockCandidate.x_points:
            Ruler_Info.conversion_mean_n_vals += 1

        Ruler_Info.pooled_sd = 0 # TODO CAN ADD THIS LATER
        Ruler_Info.conversion_successful = True 
        Ruler_Info.cross_validation_count = 0 
        Ruler_Info.conversion_data_all = []
        for pt in BlockCandidate.point_types:
            if pt != 'NA':
                Ruler_Info.cross_validation_count += 1
                Ruler_Info.conversion_data_all.append(pt)

        for i, pt in enumerate(BlockCandidate.use_points):
            if pt:
                Ruler_Info.conversion_mean_n += 1

        Ruler_Info.data_list = [] # # scanline rows..... # diff from scanlines /->/ for block it will only be the plot points
        x = []
        y = []
        for i, pt in enumerate(BlockCandidate.x_points):
            x.append(pt[0])
        for i, pt in enumerate(BlockCandidate.y_points):
            y.append(pt[0])
        Ruler_Info.data_list = list(zip(x, y))

        logger.debug(f"[Cross Validate Conversion] - Conversion data")
        logger.debug(f"[Cross Validate Conversion] - !!! Conversion Ratio !!! - 1cm = {np.round(Ruler_Info.conversion_mean, 2)} pixels")
        logger.debug(f"[Cross Validate Conversion] - !!! Conversion Ratio !!! - prediction from image's MP - 1cm = {np.round(Ruler_Info.predicted_conversion_factor_cm, 2)} pixels")
        logger.debug(f"[Cross Validate Conversion] - !!! Conversion Ratio !!! - used {Ruler_Info.cross_validation_count} units - {Ruler_Info.conversion_data_all}")
        logger.debug(f"[Cross Validate Conversion] - !!! Conversion Ratio !!! - ratio is average of {Ruler_Info.conversion_mean_n} scanlines")
        logger.debug(f"[Cross Validate Conversion] - !!! Conversion Ratio !!! - pooled SD - {Ruler_Info.pooled_sd} ")
    else:
        Ruler_Info.summary_image = summary_image
        Ruler_Info.at_least_one_correct_conversion = False
        Ruler_Info.avg_width = 0
        Ruler_Info.best_value = 0 #conversion mean
        Ruler_Info.conversion_data_all = [] # diff from scanlines
        Ruler_Info.conversion_mean = 0 #conversion mean
        Ruler_Info.conversion_mean_n = 0
        Ruler_Info.conversion_mean_n_vals = 0 
        Ruler_Info.pooled_sd = 0 # TODO CAN ADD THIS LATER
        Ruler_Info.conversion_successful = False 
        Ruler_Info.cross_validation_count = 0 
        Ruler_Info.data_list = [] # # scanline rows..... # diff from scanlines /->/ for block it will only be the plot points
        logger.debug(f"[Cross Validate Conversion] - Conversion data")
        logger.debug(f"[Cross Validate Conversion] - !!! Conversion Ratio !!! - 1cm = {np.round(Ruler_Info.conversion_mean, 2)} pixels")
        logger.debug(f"[Cross Validate Conversion] - !!! Conversion Ratio !!! - prediction from image's MP - 1cm = {np.round(Ruler_Info.predicted_conversion_factor_cm, 2)} pixels")
        logger.debug(f"[Cross Validate Conversion] - !!! Conversion Ratio !!! - used {Ruler_Info.cross_validation_count} units - []")
        logger.debug(f"[Cross Validate Conversion] - !!! Conversion Ratio !!! - ratio is average of {Ruler_Info.conversion_mean_n} scanlines")
        logger.debug(f"[Cross Validate Conversion] - !!! Conversion Ratio !!! - pooled SD - {Ruler_Info.pooled_sd} ")
    return Ruler_Info


def convert_ticks(logger, show_all_logs, Ruler_Info, RulerCFG,Ruler,img_fname, is_redo):

    Ruler_Info.process_scanline_chunk()

    if Ruler_Info.union_means_list is not None:
        Ruler_Info.cross_validate_conversion()
            
        Ruler_Info.insert_scanline()

        bi_bg = copy.deepcopy(Ruler_Info.Ruler.img_bi_pad)
        bi_bg[bi_bg == 1] = 255
        validation = stack_2_imgs(cv2.cvtColor(bi_bg, cv2.COLOR_GRAY2RGB), Ruler_Info.Ruler.img_ruler_overlay)
        validation = create_overlay_bg_3(logger, show_all_logs, RulerCFG, validation)
    else:
        bi_bg = copy.deepcopy(Ruler_Info.Ruler.img_bi_pad)
        bi_bg[bi_bg == 1] = 255
        validation = stack_2_imgs(cv2.cvtColor(bi_bg, cv2.COLOR_GRAY2RGB), Ruler_Info.Ruler.img_copy)
        validation = create_overlay_bg_3(logger, show_all_logs, RulerCFG, validation)

    for i, t in enumerate(Ruler_Info.summary_message):
        validation = cv2.putText(img=validation, text=t[0], org=(10, (20 * i) + 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(155, 155, 155),thickness=1)

    if Ruler_Info.conversion_mean != 0:
        validation = cv2.putText(img=validation, text=Ruler_Info.message_validation[0], org=(10, 95), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(155, 155, 155),thickness=1)
        validation = cv2.putText(img=validation, text=Ruler_Info.message_validation[1], org=(10, 115), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(155, 155, 155),thickness=1)
        validation = cv2.putText(img=validation, text=Ruler_Info.message_validation[2], org=(10, 135), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(155, 155, 155),thickness=1)
    else:
        validation = cv2.putText(img=validation, text='Could not convert: No points found', org=(10, 115), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(155, 155, 155),thickness=1)
        # validation = add_text_to_img('Could not convert: No points found', validation)

    # cv2.imshow('img_total_overlay', newImg)
    # cv2.waitKey(0)
    newImg = stack_2_imgs(Ruler_Info.Ruler.img_total_overlay, validation)
    # cv2.imshow('img_total_overlay', Ruler_Info.Ruler.img_total_overlay)
    # cv2.waitKey(0)
    if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_validation']:
        cv2.imwrite(os.path.join(RulerCFG.dir_ruler_validation,'.'.join([img_fname, 'jpg'])), newImg)
    if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_validation_summary']:
        cv2.imwrite(os.path.join(RulerCFG.dir_ruler_validation_summary,'.'.join([img_fname, 'jpg'])), validation)

    Ruler_Info.summary_image = validation

    return Ruler_Info
    

'''
####################################
####################################
           Main Functions
####################################
####################################
'''

class RulerInfo:
    # Ensure that ximgproc is accessible
    if hasattr(cv2, 'ximgproc'):
        # Safe to use cv2.ximgproc.thinning
        pass
    else:
        # Handle the case where ximgproc is still not available
        raise ImportError("ximgproc module is not available in cv2.")


    fail = ['fail']

    grid = ['grid_white_cm']

    ticks_only = ['tick_black_4thcm', 'tick_black_cm_halfcm_4thcm', 'tick_black_cm_halfcm', 'tick_black_cm_halfcm_mm', 'tick_black_cm_halfcm_mm_halfmm', 'tick_black_cm_mm',
        'tick_black_dual_cm_4th_8th', 'tick_black_dual_cm_inch_halfinch_4th_8th', 'tick_black_dual_cm_mm_inch_8th', 'tick_black_halfcm_mm', 
        'tick_black_inch_4th_16th', 'tick_black_inch_8th', 'tick_black_inch_halfinch_4th_8th_16th', 'tick_black_inch_halfinch_4th_8th_16th_32nd', 
        'tick_nybg_white_cm_mm', 'tick_white_cm', 'tick_white_cm_halfcm_mm', 'tick_white_cm_mm', 'tick_white_inch_halfin_4th_8th_16th',]

    block_only = ['block_alternate_black_cm', 'block_alternate_white_cm', 'block_black_cm_halfcm_mm', 'block_fieldprism_black_cm', 
        'block_mini_black_cm_halfcm_mm', 'block_stagger_black_cm', 'block_white_cm_halfcm_mm',]

    block_tick = ['blocktick_alternate_black_cm_mm', 'blocktick_stagger_black_cm_halfcm_mm', 'blocktick_stagger_black_cm_mm', 
        'blocktick_stagger_black_inch_8th', 'blocktick_stagger_white_cm_halfcm_mm', 'blocktick_stagger_white_inch_4th_16th', 
        'blocktick_stagger_white_inch_16th', 'blocktick_step_black_cm_halfcm_mm', 'blocktick_step_black_halfinch_4th_16th', 
        'blocktick_step_white_cm_halfcm_mm', 'blocktick_step_white_halfinch_4th_16th',]

    inline_classes = ['tick_black_cm_mm', 'tick_nybg_white_cm_mm', 'tick_black_inch_halfinch_4th_8th_16th_32nd']

    skip_skeleton = ['tick_nybg_white_cm_mm']

    skip_morph_cleaning = ['blocktick_stagger_black_inch_8th', 'blocktick_stagger_black_cm_mm']

    block_regular = ['block_black_cm_halfcm_mm', 'block_mini_black_cm_halfcm_mm', 'block_white_cm_halfcm_mm',]

    metric = ['grid_white_cm', 'tick_black_4thcm', 'tick_black_cm_halfcm_4thcm','tick_black_cm_halfcm', 'tick_black_cm_halfcm_mm', 'tick_black_cm_halfcm_mm_halfmm', 
        'tick_black_cm_mm', 'tick_black_halfcm_mm', 'tick_nybg_white_cm_mm', 'tick_white_cm', 'tick_white_cm_halfcm_mm', 'tick_white_cm_mm', 
        'block_alternate_black_cm', 'block_alternate_white_cm', 'block_black_cm_halfcm_mm', 'block_fieldprism_black_cm', 
        'block_mini_black_cm_halfcm_mm', 'block_stagger_black_cm', 'block_white_cm_halfcm_mm', 'blocktick_alternate_black_cm_mm', 
        'blocktick_stagger_black_cm_halfcm_mm', 'blocktick_stagger_black_cm_mm', 'blocktick_stagger_white_cm_halfcm_mm', 'blocktick_step_black_cm_halfcm_mm',
        'blocktick_step_white_cm_halfcm_mm',]

    standard = ['tick_black_inch_4th_16th', 'tick_black_inch_8th', 'tick_black_inch_halfinch_4th_8th_16th', 'tick_black_inch_halfinch_4th_8th_16th_32nd',
        'tick_white_inch_halfin_4th_8th_16th', 'blocktick_stagger_black_inch_8th', 'blocktick_stagger_white_inch_4th_16th','blocktick_stagger_white_inch_16th',
        'blocktick_step_black_halfinch_4th_16th', 'blocktick_step_white_halfinch_4th_16th',]

    dual = ['tick_black_dual_cm_4th_8th', 'tick_black_dual_cm_inch_halfinch_4th_8th', 'tick_black_dual_cm_mm_inch_8th',]

    units_metric = ['cm', 'halfcm', '4thcm', 'mm', 'halfmm',]
    units_standard = ['inch', 'halfinch', '4th', '8th', '16th', '32nd',]
    units_dual = ['cm', 'halfcm', '4thcm', 'mm', 'halfmm', 'inch', 'halfinch', '4th', '8th', '16th', '32nd',]

    dual_order =  ['halfmm','32nd','mm', '16th', '4thcm', '8th', 'halfcm', '4th', 'cm', 'halfinch', 'inch',]

    summary_message = 'Fail'
    ruler_class_percentage = 0

    def __init__(self, Ruler, predicted_conversion_factor_cm, use_CF_predictor, logger, show_all_logs) -> None:

        self.predicted_conversion_factor_cm = predicted_conversion_factor_cm
        self.use_CF_predictor = use_CF_predictor

        self.summary_message = Ruler.summary_message
        self.ruler_class_percentage = Ruler.ruler_class_percentage

        self.logger = logger
        self.show_all_logs = show_all_logs
        self.Ruler = Ruler            
        self.ruler_class = self.Ruler.ruler_class

        self.conversion_successful = False 
        self.pooled_sd = 0 
        self.summary_image = None
        self.at_least_one_correct_conversion = False
        self.avg_width = 0
        self.best_value = 0 #conversion mean
        self.conversion_data_all = [] # diff from scanlines
        self.conversion_mean = 0 #conversion mean
        self.conversion_mean_n = 0
        self.conversion_mean_n_vals = 0 
        self.cross_validation_count = 0 
        self.data_list = [] # # scanline rows..... # diff from scanlines /->/ for block it will only be the plot points

        ### FIXES
        if self.ruler_class == 'tick_black_4thcm':
            self.ruler_class = 'tick_black_cm_halfcm_4thcm'
            self.Ruler.ruler_class = 'tick_black_cm_halfcm_4thcm'

        elif self.ruler_class == 'tick_black_dual_cm_inch_halfinch_4th_8th':
            self.ruler_class = 'tick_black_dual_cm_halfinch_4th_8th'
            self.Ruler.ruler_class = 'tick_black_dual_cm_halfinch_4th_8th'

        elif self.ruler_class == 'blocktick_stagger_white_inch_16th':
            self.ruler_class = 'blocktick_stagger_white_4th_8th_16th'
            self.Ruler.ruler_class = 'blocktick_stagger_white_4th_8th_16th'

        elif self.ruler_class == 'blocktick_stagger_black_cm_halfcm_mm':
            self.ruler_class = 'blocktick_stagger_black_halfcm_mm'
            self.Ruler.ruler_class = 'blocktick_stagger_black_halfcm_mm'

        elif self.ruler_class == 'blocktick_stagger_white_cm_halfcm_mm':
            self.ruler_class = 'blocktick_stagger_white_halfcm_mm'
            self.Ruler.ruler_class = 'blocktick_stagger_white_halfcm_mm'

        self.ruler_class_parts = self.ruler_class.split('_')

        self.is_ruler = False

        self.cross_validation_count = 0
        self.conversion_mean = 0
        self.conversion_mean_n = 0
        self.conversion_data = None
        self.conversion_data_all = []
        self.unit_list = []

        self.scanSize = 0

        self.is_grid = False
        self.is_ticks_only = False
        self.is_block_only = False
        self.is_block_tick = False

        self.is_block_regular = False

        self.is_inline = False #############################################################################

        self.do_skip_skeleton = False

        self.do_skip_morph_cleaning = False

        self.is_metric = False
        self.is_standard = False
        self.is_dual = False

        self.contains_unit_metric = []
        self.contains_unit_standard = []
        self.contains_unit_dual = []

        self.check_if_ruler()
        self.check_main_path()
        self.check_metric_or_standard()
        self.check_inline()
        self.check_skeleton()
        self.check_morph_cleaning()
        self.check_block_regular()

        self.get_units()

        


    def check_if_ruler(self):
        if self.ruler_class not in self.fail:
            self.is_ruler = True
        else:
            self.is_ruler = False

    def check_main_path(self):
        if self.ruler_class in self.grid:
            self.is_grid = True
        elif self.ruler_class in self.ticks_only:
            self.is_ticks_only = True
        elif self.ruler_class in self.block_only:
            self.is_block_only = True
        elif self.ruler_class in self.block_tick:
            self.is_block_tick = True
        else:
            pass

    def check_metric_or_standard(self):
        if self.ruler_class in self.metric:
            self.is_metric = True
        elif self.ruler_class in self.standard:
            self.is_standard = True
        elif self.ruler_class in self.dual:
            self.is_dual = True
        else:
            pass

    def check_inline(self):
        if self.ruler_class in self.inline_classes:
            self.is_inline = True
        else:
            pass

    def check_skeleton(self):
        if self.ruler_class in self.skip_skeleton:
            self.do_skip_skeleton = True
        elif self.is_block_tick:
            self.do_skip_skeleton = True
        elif self.is_block_only:
            self.do_skip_skeleton = True
        else:
            pass

    def check_morph_cleaning(self):
        if self.ruler_class in self.skip_morph_cleaning:
            self.do_skip_morph_cleaning = True
        else:
            pass

    def check_block_regular(self):
        if self.ruler_class in self.block_regular:
            self.is_block_regular = True
        else:
            pass

    def get_units(self):
        for unit in self.units_metric:
            if unit in self.ruler_class_parts:
                self.contains_unit_metric.append(unit)
            else:
                pass
        for unit in self.units_standard:
            if unit in self.ruler_class_parts:
                self.contains_unit_standard.append(unit)
            else:
                pass

        for unit in self.units_dual:
            if unit in self.ruler_class_parts:
                self.contains_unit_dual.append(unit)
            else:
                pass

        if self.is_metric:
            self.n_units = len(self.contains_unit_metric)

        elif self.is_standard:
            self.n_units = len(self.contains_unit_standard) 

        elif self.is_dual:
            self.n_units = len(self.contains_unit_dual)

    def remove_specks(self, image, min_size=10):
        # Convert the image to binary (black and white)
        binary = image.convert('L').point(lambda x: 0 if x < 128 else 255, '1')
        
        # Convert to numpy array
        np_image = np.array(binary)
        
        # Label connected components
        labeled_array, num_features = ndimage.label(np_image)
        
        # Remove objects smaller than the min_size threshold
        for label in range(1, num_features + 1):
            component = labeled_array == label
            if np.sum(component) < min_size:
                np_image[component] = 0
        
        # Convert back to PIL Image
        result = Image.fromarray(np_image.astype('uint8') * 255)
        return result


    def process_scanline_chunk(self):
        if self.show_all_logs:
            self.logger.debug(f"Starting Scanlines")
        ### should return a list of just the scanlines that
        # 1. pass sanity check
        # 2. not nan
        # 3. are in the dominant mean distance category
        # 4. are in the minimal normalized SD category
        # 5. the union of 3 and 4
        # NOTE: normalized SD seems to be most reliable on getting the actual unit markers and not background noise
        # Union rows is what to compare against looking for cross validataion
        # intersect rows should be the dominant marker
        # data_list is everything
        
        scanSize = 5
        n_clusters = self.n_units + 1

        # cv2.imshow('img', self.Ruler.img_bi_sweep)
        # cv2.waitKey(0)
        if not self.do_skip_skeleton:
            img_skel = skeletonize(self.Ruler.img_bi)
        else:
            img_skel = copy.deepcopy(self.Ruler.img_bi_sweep)
            # img_skel = skeletonize(self.Ruler.img_bi)
       
        img_bi = self.Ruler.img_bi
        # img = self.Ruler.img_bi
        # cv2.imshow('img', img_skel)
        # cv2.waitKey(0)

        '''Stack all 3 versions of the binary images together'''
        img_skel[img_skel<=200] = 0
        img_skel[img_skel>200] = 1

        img_bi[img_bi<=200] = 0
        img_bi[img_bi>200] = 1

        self.Ruler.img_bi_sweep[self.Ruler.img_bi_sweep<=200] = 0
        self.Ruler.img_bi_sweep[self.Ruler.img_bi_sweep>200] = 1

        h,w = img_skel.shape
        n = h % (scanSize *2)
        img_pad_skel = pad_binary_img(img_skel,h,w,n)
        img_pad_bi = pad_binary_img(img_bi,h,w,n)
        img_pad_sweep = pad_binary_img(self.Ruler.img_bi_sweep,h,w,n)

        self.max_dim = max(img_pad_skel.shape)
        self.min_dim = min(img_pad_skel.shape)

        self.size_ratio = np.divide(self.min_dim, self.max_dim)

        img_pad = stack_2_imgs_bi(img_pad_sweep, img_pad_bi)
        img_pad = stack_2_imgs_bi(img_pad, img_pad_skel)



        # Apply despeckling based on the image resolution
        image_resolution = h * w
        if image_resolution > 4e7:  # If resolution is greater than 1 million pixels
            min_size = 40  # Higher threshold for larger images
        elif image_resolution > 1e6:  # If resolution is greater than 1 million pixels
            min_size = 30  # Higher threshold for larger images
        elif image_resolution > 5e5:  # If resolution is between 500k and 1 million pixels
            min_size = 20
        else:  # For smaller images
            min_size = 10

        img_pad_pil = Image.fromarray((img_pad * 255).astype('uint8'))
        despeckled_img = self.remove_specks(img_pad_pil, min_size=min_size)

        # Convert despeckled image back to a numpy array for further processing
        img_pad = np.array(despeckled_img).astype(np.uint8)



        # img_pad_double = img_pad
        h,w = img_pad.shape
        x = np.linspace(0, w, w)

        
        self.Ruler.img_bi_pad = copy.deepcopy(img_pad)

        # self.Ruler.img_bi_pad[self.Ruler.img_bi_pad == 1] = 255
        # cv2.imshow('img', self.Ruler.img_bi_pad)
        # cv2.waitKey(0)
        # best_value, union_rows, intersect_rows, data_list, union_means_list, intersect_means_list = process_scanline_chunk(img_pad, scanSize, h, logger, n_clusters)

        # Divide the chunk into the smaller parts to look for ticks
        means_list = []
        npts_dict = {}
        sd_list = []
        # distance_list = []
        data_list = []
        best_value = 0
        sd_temp_hold = 999
        
        for i in range(int(h / scanSize)):
            chunkAdd = img_pad[scanSize * i: (scanSize * i + scanSize), :]
                
            # Locate the ticks in the chunk
            # Split for inline and non-inline
            # artificially force into pairs for easy use
            if not self.is_inline: #not
                pairs = locate_ticks_centroid(chunkAdd, scanSize, i)
                if self.show_all_logs:
                    self.logger.debug(f"Scanlines locate_ticks_centroid")
            elif self.is_inline:
                pairs = locate_ticks_centroid_inline(chunkAdd, scanSize, i, self.logger, self.max_dim)
                if self.show_all_logs:
                    self.logger.debug(f"Scanlines locate_ticks_centroid_inline")

            for pair in pairs:
                scanlineData = {'index':[],'scanSize':[],'imgChunk':[],'plotPtsX':[],'plotPtsY':[],'plotPtsYoverall':[],'dists':[],'sd':[],'nPeaks':[],'normalizedSD':1000,'gmean':[],'mean':[]}    
                plotPtsX, plotPtsY, distUse, npts, peak_pos, avg_width = pair
                if self.show_all_logs:
                    self.logger.debug(f"pair: {plotPtsX, plotPtsY, distUse, npts, peak_pos, avg_width}")
                if (
                    plotPtsY is not None and 
                    plotPtsX is not None and 
                    distUse is not None and
                    distUse.size > 0 and
                    npts is not None
                ):
                    if len(plotPtsX) == len(plotPtsY):
                        plot_points = list(zip(plotPtsX, plotPtsY))
                        # Check the regularity of the tickmarks and their distances
                        min_pairwise_distance = minimum_pairwise_distance(plotPtsX, plotPtsY)
                        min_pairwise_distance_odd = minimum_pairwise_distance(plotPtsX[1::2], plotPtsY[1::2]) / 2
                        min_pairwise_distance_even = minimum_pairwise_distance(plotPtsX[0::2], plotPtsY[0::2]) / 2
                        min_pairwise_distance_third = minimum_pairwise_distance(plotPtsX[2::3], plotPtsY[2::3]) / 3
                        
                        if self.is_inline:
                            sanity_check = True
                        else:
                            sanity_check = sanity_check_scanlines(
                            min_pairwise_distance,
                            min_pairwise_distance_odd,
                            min_pairwise_distance_even,
                            min_pairwise_distance_third
                            )

                        mean_plus_normsd = np.nanmean(distUse) # + avg_width #+ (3*(np.std(distUse) / np.nanmean(distUse)))

                        sd_temp = (np.std(distUse, ddof=1) / np.nanmean(distUse))
                        if sd_temp < sd_temp_hold and (npts >= 3) and sanity_check:
                            sd_temp_hold = sd_temp
                            self.avg_width = avg_width
                            best_value = np.nanmean(distUse)
                        # Store the scanline data if the tickmarks are regular
                        if sanity_check and not np.isnan(np.nanmean(distUse)):
                            chunkAdd[chunkAdd >= 1] = 255
                            scanlineData = {
                                'index': int(i),
                                'mean': mean_plus_normsd, #np.nanmean(distUse),
                                'normalizedSD': (np.std(distUse, ddof=1) / np.nanmean(distUse)),
                                'nPeaks': npts,
                                'sd': np.std(distUse, ddof=1),
                                'imgChunk': chunkAdd,
                                'plotPtsX': plotPtsX,
                                'plotPtsY': plotPtsY,
                                'plot_points': plot_points,
                                'plotPtsYoverall': (scanSize * i + scanSize) - round(scanSize / 2),
                                'dists': distUse,
                                'gmean': gmean(distUse),
                                'scanSize': int(scanSize),
                                'peak_pos': peak_pos
                            }
                            
                            data_list.append(scanlineData)
                            means_list.append([mean_plus_normsd])
                            sd_list.append([(np.std(distUse) / np.nanmean(distUse))])
                            npts_dict[str(mean_plus_normsd)] = npts

        if len(means_list) >= 2:
            do_continue = True
        else:
            do_continue = False

        if n_clusters > len(means_list):
            n_clusters = len(means_list) - 1
            if n_clusters < 2:
                n_clusters = 2

        if do_continue:
            if self.is_inline: # just keep all the data, skip kmeans
                self.best_value = best_value
                self.union_rows = [data_list[i] for i in range(len(data_list))]
                self.intersect_rows = [data_list[i] for i in range(len(data_list))]
                self.data_list = data_list
                self.union_means_list = [means_list[i] for i in range(len(data_list))]
                self.intersect_means_list = [means_list[i] for i in range(len(data_list))]
                self.npts_dict = npts_dict 

                self.intersect_means_list_indices = [i for i in range(len(data_list))]
                self.union_means_list_indices = [i for i in range(len(data_list))]

                if self.show_all_logs:
                    self.logger.debug(f"Dominant pattern indices - (mean dist) - inline - all ind")
                    self.logger.debug(f"Minimal pattern indices (SD) - inline - all ind")
                    self.logger.debug(f"Union pattern indices - inline - all ind")
                    self.logger.debug(f"Average tick width - {self.avg_width}")

            else:
                # Initialize the k-means model with n_units+1 clusters
                kmeans = KMeans(n_clusters=n_clusters, random_state=2022, n_init=10)
                # Fit the model to the data
                kmeans.fit(np.array(means_list).reshape(-1, 1))
                # Get the cluster centers and labels
                centers = kmeans.cluster_centers_
                labels = kmeans.labels_

                # Determine which cluster has the higher count of data points
                counts = [sum(labels == i) for i in range(n_clusters)]
                dominant_cluster = np.argmax(counts)
                dominant_pattern = labels == dominant_cluster

                dom_pattern = np.where(dominant_pattern)[0]
                if self.show_all_logs:
                    self.logger.debug(f"Dominant pattern indices - {np.where(dominant_pattern)[0]}")


                # Initialize the k-means model with 2 clusters
                kmeans = KMeans(n_clusters=2, random_state=2022, n_init=10)
                # Fit the model to the data
                kmeans.fit(np.array(sd_list).reshape(-1, 1))
                # Get the cluster centers and labels
                centers = kmeans.cluster_centers_
                labels = kmeans.labels_
                # Determine which cluster has the smaller center value
                if centers[0] < centers[1]:
                    minimal_pattern = labels == 0
                else:
                    minimal_pattern = labels == 1
                min_pattern = np.where(minimal_pattern)[0]
                union = np.union1d(dom_pattern, min_pattern)
                union_rows = [data_list[i] for i in union]

                intersection = np.intersect1d(dom_pattern, min_pattern)
                intersect_rows = [data_list[i] for i in intersection]

                union_means_list = [means_list[i] for i in union]
                intersect_means_list = [means_list[i] for i in intersection]

                self.best_value = best_value
                self.union_rows = union_rows
                self.intersect_rows = intersect_rows
                self.data_list = data_list
                self.union_means_list = union_means_list
                self.intersect_means_list = intersect_means_list
                self.npts_dict = npts_dict

                self.intersect_means_list_indices = np.where(minimal_pattern)[0]
                self.union_means_list_indices = np.where(dominant_pattern)[0]
                
                if self.show_all_logs:
                    self.logger.debug(f"Dominant pattern indices - (mean dist){np.where(dominant_pattern)[0]}")
                    self.logger.debug(f"Minimal pattern indices (SD) - {np.where(minimal_pattern)[0]}")
                    self.logger.debug(f"Union pattern indices - {union}")
                    self.logger.debug(f"Average tick width - {self.avg_width}")
        else:
            self.best_value = best_value
            self.union_rows = None
            self.intersect_rows = None
            self.data_list = data_list
            self.union_means_list = None
            self.intersect_means_list = None
            self.npts_dict = npts_dict

            if self.show_all_logs:
                self.logger.debug(f"Not enough scanlines located - only found {len(means_list)} - requires >= 2")
        
    def order_units_small_to_large(self, conversion_board, sorted_union_means_list, units_possible):
        current_val = sorted_union_means_list[0][0]
        # self.prev_val = current_val
        sorted_union_means_list_working = copy.deepcopy(sorted_union_means_list) 
        sorted_union_means_list_working.pop(0)

        # add first val
        conversion_board[self.current_unit].append(current_val)
        self.original_distances_1unit.append(current_val)

        for i, challenge_val in enumerate(sorted_union_means_list_working):
            challenge_val = challenge_val[0]
            # If we're not on the first value, check if it's within the tolerance of the previous value
            if abs(current_val - challenge_val) > self.tolerance: #next value is greater than tolerance
                is_correct_converion = False # reset
                starting_key = self.key_ind # reset

                while not is_correct_converion and starting_key < len(units_possible): # Let's us compare mm to halfcm, mm to cm
                    # Test conversion
                    try:
                        challenge_unit = units_possible[starting_key + 1]
                    except:
                        challenge_unit = units_possible[starting_key]

                    is_correct_converion = self.test_conversion(self.current_unit, [current_val], challenge_unit, [challenge_val], self.tolerance) # current_unit, current_val, challenge_unit, challenge_val

                    if is_correct_converion:
                        self.at_least_one_correct_conversion = True
                        starting_key += 1
                        if starting_key < len(units_possible):
                            self.current_unit = units_possible[starting_key]
                            conversion_board[self.current_unit].append(challenge_val)
                            self.prev_val = challenge_val
                            current_val = challenge_val
                            self.original_distances_1unit.append(challenge_val)
                    else:
                        starting_key += 1
            elif abs(current_val - challenge_val) <= self.tolerance: # values are the same, add to list
                self.cv_list_count += 1
                conversion_board[self.current_unit].append(challenge_val)
                self.original_distances_1unit.append(challenge_val)
                self.original_distances_1unit.append(self.prev_val)

        self.original_distances_1unit = list(set(self.original_distances_1unit))
        return conversion_board, sorted_union_means_list, units_possible
    
    def select_best_conversion_factors(self, conversion_final, conversion_possible):
        tolerance = 0.1 * self.predicted_conversion_factor_cm
        updated_conversion_final = {key: [] for key in conversion_possible.keys()}

        # Check each unit conversion against the tolerance
        for test_unit in conversion_possible.keys():
            for unit, calculated_factors in conversion_final.items():
                if calculated_factors is None:
                    continue
                if isinstance(calculated_factors, list):
                    if len(calculated_factors) == 0:  
                        continue
                else:
                    if calculated_factors.size == 0:  
                        continue

                converted_factors = self.convert_to_cm(test_unit, calculated_factors)

                for original, converted in zip(calculated_factors, converted_factors):
                    difference = abs(self.predicted_conversion_factor_cm - converted)
                    percentage_diff = (difference / self.predicted_conversion_factor_cm) * 100

                    if difference <= tolerance:
                        updated_conversion_final[test_unit].append(original)
                        if self.show_all_logs:
                            self.logger.debug(f"Match found for unit {test_unit}: original {original:.2f}, predicted {self.predicted_conversion_factor_cm:.2f} cm --- difference {percentage_diff:.2f}%")

        # Output debug information
        for unit, matches in updated_conversion_final.items():
            if matches:
                if self.show_all_logs:
                    self.logger.debug(f"Matching unit: {unit} with factors: {matches}")

        # Remove keys with empty list values
        for key in list(updated_conversion_final.keys()):
            if not updated_conversion_final[key]:
                del updated_conversion_final[key]
        
        return updated_conversion_final


    def cross_validate_conversion(self):
        self.conversion_successful = False

        if self.is_metric:
            self.units_possible = self.contains_unit_metric
        elif self.is_standard:
            self.units_possible = self.contains_unit_standard
        elif self.is_dual:
            self.units_possible = self.contains_unit_dual

        if self.show_all_logs:
            self.logger.debug(f"[Cross Validate Conversion] - Units possible - {self.units_possible}")

        # sort the means lists from small to big #tried originally reversed to small
        sorted_union_means_list = sorted(self.union_means_list, reverse=False)
        sorted_intersect_means_list = sorted(self.intersect_means_list, reverse=False)

        # conversion_board_list = []
        remanining = sorted_union_means_list.copy()
        conversion_final = {}
        # conversion_board = {'convert':False, 'cm':[], 'halfcm':[], '4thcm':[], 'mm':[], 'halfmm':[], 'inch':[], 'halfinch':[], '4th':[], '8th':[], '16th':[], '32nd':[]}
        
        '''
        *** where the magic happens ***
        '''
        max_candidate = sorted_union_means_list[0][0]
        max_candidate_n = self.npts_dict[str(max_candidate)]
        min_candidate = sorted_union_means_list[-1][0]
        min_candidate_n = self.npts_dict[str(min_candidate)]

        # largest_unit = self.determine_largest_unit(sorted_intersect_means_list) 
        units_possible = self.units_possible[::-1]
        smallest_unit = units_possible[0]
        if self.show_all_logs:
            self.logger.debug(f"[Cross Validate Conversion] - Smallest unit - {smallest_unit}")

        # conversion_board  = {'convert':False}
        conversion_board  = {}
        if self.is_dual: # for dual need to reorder the units from small to large
            for u in self.dual_order:
                if u in units_possible:
                    conversion_board[u] = []
        else:
            for i in range(0, len(units_possible)):
                conversion_board[units_possible[i]] = []




        self.conversion_board_fresh = copy.deepcopy(conversion_board)

        self.tolerance = 5
        self.at_least_one_correct_conversion = False
        self.exhausted_units_to_test = False
        self.current_unit = smallest_unit
        self.current_val = 0
        self.prev_val = 0
        self.key_ind = 0
        self.original_distances_1unit = []
        sorted_union_means_list_fresh = copy.deepcopy(sorted_union_means_list) 
        unit_try = 0
        did_fail = False
        self.cv_list = []
        self.cv_list_dict = {}
        self.cv_list_count = 0
        self.cv_list_ind = 0
        is_cv_reset = False
        self.conversion_mean_n_vals = 0
        self.pooled_sd = 0
        self.pooled_sd_list = []
        self.cross_validation_count = 0
        '''
        start with smallest value. add next val to that unit IF within tolerance. 
        if next val > tol, then test conversion from previous to current unit. 
        if pass, add that to the next unit
        if fail, skip
        if all fail, then remove first unit's list and start over
        if unit list is exhausted, it will return the last successful list (happens with 'and not did_fail')
        #saves copies of the cv, will use the one with the most valid tolerance values as the export if  fail
        # TODO if mm is put into cm -> make size check based on the old-school dimension check, see how many units fit etc.
        '''
        while not self.at_least_one_correct_conversion:
            
            conversion_board, sorted_union_means_list, units_possible = self.order_units_small_to_large(conversion_board, sorted_union_means_list, units_possible)
            
             # This means that none of the conversions succeeded for the unit. Start over, but remove the smallest unit as a possibility
            if not self.at_least_one_correct_conversion and self.exhausted_units_to_test:
                if self.show_all_logs:
                    self.logger.debug(f"[Cross Validate Conversion] - Reassign - Conversion board units possible {units_possible} - {conversion_board}")
                self.exhausted_units_to_test = False # reset unit count
                unit_try += 1
                try:
                    self.current_unit = units_possible[unit_try]
                except:
                    self.at_least_one_correct_conversion = True # all units / tries are exhausted, give up
                    did_fail = True
                    #find best cv
                    best_cv_ind  = max(self.cv_list_dict, key=self.cv_list_dict.get)
                    conversion_board = copy.deepcopy(self.cv_list[best_cv_ind])

                is_cv_reset = True
                sorted_union_means_list = copy.deepcopy(sorted_union_means_list_fresh) 
            
            # Remove the smallest unit's value and try again
            if not self.at_least_one_correct_conversion and not self.exhausted_units_to_test and not did_fail: 
                # make copy of cv -> this is used if all conversions fail.
                #                   then we keep the smallest, last unit's cv_board
                #                   assuming it to be the smallest unit i.e. mm or 16th....
                self.cv_list.append(copy.deepcopy(conversion_board))
                self.cv_list_dict[self.cv_list_ind] = self.cv_list_count
                self.cv_list_ind += 1
                self.cv_list_count = 0

                conversion_board = conversion_board.clear()
                conversion_board = copy.deepcopy(self.conversion_board_fresh)
                self.current_val = 0
                self.key_ind = unit_try # pins the active unit. as  the unit_try += 1 we keep progressing
                self.original_distances_1unit = []
                if not is_cv_reset: # need to keep the first val in the means list after -> sorted_union_means_list = copy.deepcopy(sorted_union_means_list_fresh) 
                    if len(sorted_union_means_list) > 2:
                        sorted_union_means_list.pop(0)
                    else:
                        self.exhausted_units_to_test = True
                else:
                    is_cv_reset = False
        if did_fail:
            if self.show_all_logs:
                self.logger.debug(f"[Cross Validate Conversion] - Only 1 unit - order_units_small_to_large()")
            # now we need to determine which possible cv_board fits based on units/max_dim


        ### Different paths for if there are multiple possible units vs only one unit
        # if conversion_board['convert'] == True: # If there was at least one validation
        if self.at_least_one_correct_conversion: # If there was at least one validation
            self.conversion_successful = True
            if self.show_all_logs:
                self.logger.debug(f"[Cross Validate Conversion] - Conversion board final - {conversion_board}")

            '''getting the conversion value'''
            # get only the units allowed by the ML prediction
            if len(self.units_possible) > 1:
                for unit, unit_value in conversion_board.items(): # iterate through conversion_board
                    # if unit_value != []:                          # ignore empty units
                    if unit in self.units_possible:           # if unit is in the allowed units
                        # conversion_final[unit] = unit_value
                        # self.cross_validation_count += 1
                        self.unit_list.append(unit)
            else:
                pass

            '''prepping conversion_final'''          
            conversion_final = copy.deepcopy(conversion_board)

            # Remove keys with empty list values
            for key in list(conversion_final.keys()):
                if not conversion_final[key]:
                    del conversion_final[key]

            # remove outliers for each unit
            for unit, unit_value in conversion_final.items():
                if len(unit_value) >= 5:
                    conversion_final[unit] = self.remove_outliers_bias_larger(np.array(unit_value))

            
            ### USE THE PREDICTED CF AS A JUDGE conversion_final should be cm...
            if self.use_CF_predictor:
                conversion_final = self.select_best_conversion_factors(conversion_final, conversion_board)


            # convert all values to cm
            if bool(conversion_final): # if conversion_final is not empty
                conversions_in_cm_nested = []
                for unit, unit_value in conversion_final.items():
                    # convert everything to cm, remove outliers, average
                    conversions_in_cm_nested.append(self.convert_to_cm(unit, unit_value))
                conversions_in_cm = [val for sublist in conversions_in_cm_nested for val in sublist]

                # remove outliers
                if len(conversions_in_cm) >= 5:
                    conversions_in_cm = self.remove_outliers_geomean(conversions_in_cm)

                # get average
                self.cross_validation_count = len(list(conversion_final.keys()))

                self.conversion_mean = np.nanmean(conversions_in_cm) #+ self.avg_width#+ np.nanmean(conversions_in_cm)/50 #2*self.avg_width +
                self.conversion_mean_n = len(conversions_in_cm)
                

                '''getting the data that accompanies the values used to get the conversion value'''
                # Use conversion_final to get the instances from data_list
                if len(self.units_possible) == 1:
                    for ind in self.intersect_means_list_indices:
                        row = self.data_list[ind]
                        unit = self.units_possible[0]
                        if row['mean'] in self.original_distances_1unit:
                            self.conversion_data_all.append({unit:row})
                            self.conversion_mean_n_vals += row['nPeaks']
                            self.pooled_sd_list.append([ np.multiply((row['nPeaks']-1), np.power(row['sd'], 2)) ]) # creat term for pooled sd

                else:
                    for unit, unit_value in conversion_final.items():   ##### getting the n_pts from here self.conversion_mean_n_vals
                        for val in unit_value:
                            for row in self.data_list:
                                if row['mean'] == val:
                                    self.conversion_data_all.append({unit:row})
                                    self.conversion_mean_n_vals += row['nPeaks']
                                    self.pooled_sd_list.append([ np.multiply((row['nPeaks']-1), np.power(row['sd'], 2)) ]) # creat term for pooled sd

                self.calc_pooled_sd()

                self.logger.debug(f"[Cross Validate Conversion] - Conversion data")
                self.logger.debug(f"[Cross Validate Conversion] - !!! Conversion Ratio !!! - 1cm = {np.round(self.conversion_mean, 2)} pixels")
                self.logger.debug(f"[Cross Validate Conversion] - !!! Conversion Ratio !!! - prediction from image's MP - 1cm = {np.round(self.predicted_conversion_factor_cm, 2)} pixels")
                self.logger.debug(f"[Cross Validate Conversion] - !!! Conversion Ratio !!! - used {self.cross_validation_count} units - {list(conversion_final.keys())}")
                self.logger.debug(f"[Cross Validate Conversion] - !!! Conversion Ratio !!! - ratio is average of {self.conversion_mean_n} scanlines")
                self.logger.debug(f"[Cross Validate Conversion] - !!! Conversion Ratio !!! - pooled SD - {self.pooled_sd} ")
                self.message_validation = [f"1cm = {np.round(self.conversion_mean, 2)} pixels",f"Used {self.cross_validation_count} units - {list(conversion_final.keys())}",f"Ratio is average of {self.conversion_mean_n} scanlines and {self.conversion_mean_n_vals} distances"]

                

        else:
            self.logger.debug(f"[Cross Validate Conversion] - Conversion not possible - {conversion_board}")
        self.logger.debug(f"[Cross Validate Conversion] - Done")

    def calc_pooled_sd(self):
        df = self.conversion_mean_n_vals - 2
        numerator = np.sum(self.pooled_sd_list)
        if df == 0:
            df = 0.0000000001
        self.pooled_sd = np.sqrt(np.divide(numerator, df))

    def remove_outliers_bias_larger(self, dist):
        Q1 = np.percentile(dist, 30)
        Q3 = np.percentile(dist, 95)
        IQR = Q3 - Q1
        upper_bound = Q3 + IQR
        lower_bound = Q1 - IQR
        distUse = dist[(dist > lower_bound) & (dist < upper_bound)]
        return distUse

    def determine_largest_unit(self, sorted_union_means_list):
        for value in sorted_union_means_list:
            val = value[0]
            n = self.max_dim/val
            # n = self.npts_dict[str(val)]

            if self.is_metric:
                # 5 cm max ruler
                if self.size_ratio > (1/5):
                    try:
                        if (n-1 * val < self.max_dim) and (n < 7): # cm
                            return self.units_possible[0] 
                        elif (n-1 * val < self.max_dim) and (7 <= n < 20):
                            return self.units_possible[1]
                        else:
                            return 'mm'
                    except:
                        return 'mm'
                elif  (1/20) < self.size_ratio <= (1/5): #15cm
                    try:
                        if (n-1 * val < self.max_dim) and (n < 17): # cm
                            return self.units_possible[0] 
                        elif (n-1 * val < self.max_dim) and (17 <= n < 34):
                            return self.units_possible[1]
                        else:
                            return 'mm'
                    except:
                        return 'mm'
                else:# 30 cm max ruler
                    try:
                        if (n-1 * val < self.max_dim) and (n < 31): # cm
                            return self.units_possible[0] 
                        elif (n-1 * val < self.max_dim) and (31 <= n < 61):
                            return self.units_possible[1]
                        else:
                            return 'mm'
                    except:
                        return 'mm'
        
            elif self.is_standard:
                # 15 inch max ruler
                if (n-1 * val < self.max_dim) and (n < 16): # inch
                    return self.units_possible[0] 
                else:
                    return 'mm'

            elif self.is_dual:
                if self.size_ratio < (1/20):
                    return 'mm'
                else:
                    return 'cm'


    def test_span(self, candidate, cross):
        test_candidate_n = self.npts_dict[str(candidate)]
        test_cross_n = self.npts_dict[str(cross)]

        # How many units fir into the space the points came from
        # if span_x = 150, then 150 units fit into the space

        span_candidate = self.max_dim / candidate
        span_cross = self.max_dim / cross
        # units * pixel length. coverage_if_mm will be less than max_dim IF it's mm
        coverage_if_mm_candidate = candidate * test_candidate_n
        coverage_if_mm_cross = cross * test_cross_n
        # units * pixel length. coverage_if_mm will be less than max_dim IF it's mm
        coverage_if_cm_candidate = candidate * span_candidate * test_candidate_n
        coverage_if_cm_cross = cross * span_cross * test_cross_n

        result_candidate = None
        result_cross = None
        if (coverage_if_mm_candidate < self.max_dim) and (coverage_if_cm_candidate > self.max_dim):
            if span_candidate <= 20:
                result_candidate = 'big'
            else:
                result_candidate = 'small'
        else:
            result_candidate = 'big'

        if (coverage_if_mm_cross < self.max_dim) and (coverage_if_cm_cross > self.max_dim):
            if span_candidate <= 20:
                result_cross = 'big'
            else:
                result_cross = 'small'
        else:
            result_cross = 'big'

        # if result_candidate == 'small': # take remedial action
        #     candidate_unit, cross_unit = self.is_within_tolerance(candidate[0], cross[0])


        return result_candidate, result_cross
    
    def test_conversion(self, current_unit, current_val, challenge_unit, challenge_val, tolerance):
        current_cm = self.convert_to_cm(current_unit, current_val)
        challenge_cm = self.convert_to_cm(challenge_unit, challenge_val)

        if abs(current_cm[0] - challenge_cm[0]) < tolerance:
            return True
        else:
            return False


    def convert_to_cm(self, unit, unit_value):
        if not isinstance(unit_value, list) and not isinstance(unit_value, np.ndarray):
            unit_value = [unit_value]  # Convert scalar to list for uniform processing
    

        unit_value_converted = []

        if unit == '32nd':
            for val in unit_value:
                unit_value_converted.append(float(np.divide(np.multiply(val, 32), 2.54)))
        elif unit == '16th':
            for val in unit_value:
                unit_value_converted.append(float(np.divide(np.multiply(val, 16), 2.54)))
        elif unit == '8th':
            for val in unit_value:
                unit_value_converted.append(float(np.divide(np.multiply(val, 8), 2.54)))
        elif unit == '4th':
            for val in unit_value:
                unit_value_converted.append(float(np.divide(np.multiply(val, 4), 2.54)))
        elif unit == 'halfinch':
            for val in unit_value:
                unit_value_converted.append(float(np.divide(np.multiply(val, 2), 2.54)))
        elif unit == 'inch':
            for val in unit_value:
                unit_value_converted.append(float(np.divide(np.multiply(val, 1), 2.54)))

        elif unit == 'halfmm':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(val, 20)))
        elif unit == 'mm':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(val, 10)))
        elif unit == '4thcm':
            for val in unit_value:    
                unit_value_converted.append(float(np.multiply(val, 4)))
        elif unit == 'halfcm':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(val, 2)))
        elif unit == 'cm':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(val, 1)))

        return unit_value_converted
        
    def remove_outliers_geomean(self, lst):
        if len(lst) >= 2:
            # Find the geometric mean
            geo_mean = math.exp(statistics.mean([math.log(x) for x in lst]))
            
            # Calculate the standard deviation
            std_dev = statistics.stdev(lst)
            
            # Calculate the threshold for outlier removal
            threshold = geo_mean + (0.5 * std_dev)
            
            # Create a new list without outliers
            new_lst = [x for x in lst if x <= threshold]
            
            return new_lst
        else:
            return lst

    def get_unit_color(self, unit):
        # BGR
        # Define color schemes
        color_list = [(0, 80, 0),  # dark green cm
              (0, 0, 100),  # dark Red inch
              (0, 165, 255),  # Orange 4thcm
              (0, 255, 255),  # Yellow halfinch
              (255, 255, 0),  # Cyan
              (255, 165, 0),  # Gold
              (255, 0, 0),  # Blue
              (255, 0, 255),  # Magenta
              (128, 0, 128),  # Purple
              (0, 128, 128),  # Teal
              (0, 191, 255),  # deep sky blue
              (0, 0, 0),  # black
              (255, 255, 255)]  # white
        conversion = [(0, 255, 0), (255, 180, 0), (0, 0, 255)]

        # Assign colors based on unit
        if unit == '32nd':
            return color_list[5]
        elif unit == '16th':
            return color_list[6]
        elif unit == '8th':
            return color_list[9]
        elif unit == '4th':
            return color_list[7]
        elif unit == 'halfinch':
            return color_list[3]
        elif unit == 'inch':
            return color_list[1]

        elif unit == 'halfmm':
            return color_list[10]
        elif unit == 'mm':
            return color_list[4]
        elif unit == '4thcm':
            return color_list[2]
        elif unit == 'halfcm':
            return color_list[8]
        elif unit == 'cm':
            return color_list[0]
        
        elif unit == 'conversion_pred1':
            return color_list[11]
        elif unit == 'conversion_pred2':
            return color_list[12]
        
        elif unit == 'conversion_1cm':
            return conversion[0]
        elif unit == 'conversion_10cm':
            return conversion[1]
        elif unit == 'conversion_1inch':
            return conversion[2]
        
        else:
            raise


    def insert_scanline(self): 
        imgBG = self.Ruler.img_copy

        if self.max_dim < 800:
            sz = 1
        elif self.max_dim < 2000:
            sz = 2
        elif self.max_dim < 4000:
            sz = 3
        else:
            sz = 4

        # Plot all points
        unit_tally = []
        unit_plot = []
        for row in self.conversion_data_all:
            for unit, unit_data in row.items():
                if unit not in unit_tally:
                    unit_tally.append(unit)
                    unit_plot.append(row)

                color = self.get_unit_color(unit)
                x = unit_data['plotPtsX']
                y = unit_data['plotPtsYoverall']

                # Loop through the x and y arrays and plot each point on the image
                for i in range(len(x)):
                    cv2.circle(imgBG, (int(x[i]), int(y)), sz, color, -1, cv2.LINE_AA)


        # Plot 1 unit marker for each identified unit (cm, mm, inch etc)
        # on first iteration, also plot the 1 CM line, thick all green
        plot_conversion = 0
        for row in unit_plot:
            for unit, unit_data in row.items():
                color = self.get_unit_color(unit)
                x = unit_data['plotPtsX']
                y = unit_data['plotPtsYoverall']
                d = unit_data['mean']

                y_pt = y
                imgBG = self.add_unit_marker(imgBG, d, x, 1, y_pt, color)

                if plot_conversion == 0:
                    plot_conversion += 1
                    
                    color = self.get_unit_color('conversion_pred1')
                    imgBG = self.add_unit_marker_guide(imgBG, self.predicted_conversion_factor_cm, [20, int(20+self.max_dim/4), int(20+self.max_dim/2)], 1, self.min_dim+30, color)
                    color = self.get_unit_color('conversion_pred2')
                    imgBG = self.add_unit_marker_guide(imgBG, self.predicted_conversion_factor_cm, [20, int(20+self.max_dim/4), int(20+self.max_dim/2)], 1, self.min_dim+35, color)
                    color = self.get_unit_color('conversion_pred1')
                    imgBG = self.add_unit_marker_guide(imgBG, self.predicted_conversion_factor_cm, [20, int(20+self.max_dim/4), int(20+self.max_dim/2)], 1, self.min_dim+40, color)


                    color = self.get_unit_color('conversion_1cm')
                    imgBG = self.add_unit_marker_guide(imgBG, self.conversion_mean, [20, int(20+self.max_dim/4), int(20+self.max_dim/2)], 1, self.min_dim+10, color)
                    color = self.get_unit_color('conversion_10cm')
                    # imgBG = self.add_unit_marker(imgBG, self.conversion_mean+(self.conversion_mean/50), x, 5, y_pt-10, color)
                    imgBG = self.add_unit_marker_guide(imgBG, self.conversion_mean, [20, int(20+self.max_dim/4), int(20+self.max_dim/2)], 5, self.min_dim, color)
                    if self.is_standard:
                        color = self.get_unit_color('conversion_1inch')
                        # imgBG = self.add_unit_marker(imgBG, self.conversion_mean+(self.conversion_mean/50), x, 5, y_pt-10, color)
                        imgBG = self.add_unit_marker_guide(imgBG, self.conversion_mean, [20, int(20+self.max_dim/4), int(20+self.max_dim/2)], 2.54, self.min_dim+20, color)
       
        # else:
        #     # print(f"{bcolors.WARNING}     No tickmarks found{bcolors.ENDC}")
        #     self.logger.debug(f"No tickmarks found")
        self.Ruler.img_ruler_overlay = imgBG
        
        do_show_combined = False
        if do_show_combined:
            # Load images
            img_bi = self.Ruler.img_bi_pad
            imgBG = imgBG

            # Resize images to have the same width
            width = max(img_bi.shape[1], imgBG.shape[1])
            height = img_bi.shape[0] + imgBG.shape[0]
            img_combined = np.zeros((height, width, 3), dtype=np.uint8)

            # Copy images vertically
            img_combined[:img_bi.shape[0], :img_bi.shape[1]] = cv2.cvtColor(img_bi, cv2.COLOR_GRAY2RGB)
            img_combined[img_bi.shape[0]:, :imgBG.shape[1]] = imgBG

            # Show combined image
            cv2.imshow("Combined", img_combined)
            cv2.waitKey(0)


    def add_unit_marker_guide(self, img_bg, distance, x_coords, factor, y_pt, color):
        # shift_amount = - min(img_bg.shape[0], img_bg.shape[1]) / 10
        thickness = 4 if max(img_bg.shape[0], img_bg.shape[1]) > 1000 else 2
        x_coords.sort()

        first_marker_pos = x_coords[0]
        middle_marker_pos = x_coords[1]
        last_marker_pos = x_coords[2]

        start_positions = [first_marker_pos, middle_marker_pos, last_marker_pos]
        end_positions = [first_marker_pos + int(distance * factor), 
                        middle_marker_pos + int(distance * factor), 
                        last_marker_pos - int(distance * factor)]
    
        do_plot = True
        for pos in range(0,len(start_positions),1):
            y_neg = -1 if pos % 2 else 1
            shift0 = 0 # y_neg * min(img_bg.shape[0], img_bg.shape[1]) / 10
            if do_plot:
                if factor > 1:
                    do_plot = False 
                if (pos % 2) != 0:
                    shift = -1 * shift0
                else:
                    shift = shift0
                for i in range(-thickness,thickness+1):
                    for j in range(start_positions[pos],end_positions[pos],1):
                        try:
                            # 5 pixel thick line
                            if (abs(i) == thickness) | (abs(j) == thickness):
                                img_bg[int(shift+y_pt+i),int(j),0] = color[0]
                                img_bg[int(shift+y_pt+i),int(j),1] = color[1]
                                img_bg[int(shift+y_pt+i),int(j),2] = color[2]
                            else:
                                img_bg[int(shift+y_pt+i),int(j),0] = color[0]
                                img_bg[int(shift+y_pt+i),int(j),1] = color[1]
                                img_bg[int(shift+y_pt+i),int(j),2] = color[2]
                        except:
                            continue

        return img_bg

    def add_unit_marker(self, img_bg, distance, x_coords, factor, y_pt, color):
        # shift_amount = - min(img_bg.shape[0], img_bg.shape[1]) / 10
        thickness = 4 if max(img_bg.shape[0], img_bg.shape[1]) > 1000 else 2
        x_coords.sort()

        try:
            first_marker_pos = int(x_coords[0])
            middle_marker_pos = int(x_coords[int(x_coords.size/2)])
            try:
                last_marker_pos = int(x_coords[-4] - (factor * distance))
            except:
                last_marker_pos = int(x_coords[-2] - (factor * distance))


            start_positions = [first_marker_pos, middle_marker_pos, last_marker_pos]
            end_positions = [first_marker_pos + int(distance * factor), 
                            middle_marker_pos + int(distance * factor), 
                            last_marker_pos - int(distance * factor)]

        except Exception as e:
            if self.show_all_logs:
                self.logger.debug(f"add_unit_marker(): plotting 1 of 3 unit markers. Exception: {e.args[0]}")

            middle_marker_pos = int(x_coords[int(x_coords.size/2)])
            start_positions = [middle_marker_pos]
            end_positions = [middle_marker_pos - int(distance * factor)]

        # do_plot = True
        # for pos in range(len(start_positions)):
        #     if do_plot:
        #         if factor > 1:
        #             do_plot = False
        #         # shift = shift_amount if (pos % 2) != 0 else -1 * shift_amount
        #         y_neg = -1 if pos % 2 else 1
        #         y_shift = (5 * y_neg) + int(y_pt) 
        #         start_point = (int(start_positions[pos]), y_shift)
        #         end_point = (int(end_positions[pos]), y_shift)
        #         cv2.line(img_bg, start_point, end_point, color, thickness, cv2.LINE_AA)
        do_plot = True
        for pos in range(0,len(start_positions),1):
            y_neg = -1 if pos % 2 else 1
            shift0 = 0 # y_neg * min(img_bg.shape[0], img_bg.shape[1]) / 10
            if do_plot:
                if factor > 1:
                    do_plot = False 
                if (pos % 2) != 0:
                    shift = -1 * shift0
                else:
                    shift = shift0
                for i in range(-thickness,thickness+1):
                    for j in range(start_positions[pos],end_positions[pos],1):
                        try:
                            # 5 pixel thick line
                            if (abs(i) == thickness) | (abs(j) == thickness):
                                img_bg[int(shift+y_pt+i),int(j),0] = color[0]
                                img_bg[int(shift+y_pt+i),int(j),1] = color[1]
                                img_bg[int(shift+y_pt+i),int(j),2] = color[2]
                            else:
                                img_bg[int(shift+y_pt+i),int(j),0] = color[0]
                                img_bg[int(shift+y_pt+i),int(j),1] = color[1]
                                img_bg[int(shift+y_pt+i),int(j),2] = color[2]
                        except:
                            continue

        return img_bg



def setup_ruler(Labels, model, device, cfg, Dirs, logger, show_all_logs, RulerCFG, img, img_fname):
    # TODO add the classifier check
    Ruler = RulerImage(img=img, img_fname=img_fname)

    # print(f"{bcolors.BOLD}\nRuler: {img_fname}{bcolors.ENDC}")
    print(f"{bcolors.OKCYAN}Ruler - {img_fname}{bcolors.ENDC}")
    logger.debug(f"Ruler: {img_fname}")
    
    Ruler.ruler_class, Ruler.ruler_class_pred, Ruler.ruler_class_percentage,Ruler.img_type_overlay, summary_message = detect_ruler(logger, show_all_logs, RulerCFG, img, img_fname)

    Ruler.summary_message = summary_message
    
    do_skeletonize = False # TODO change this as needed per class

    Ruler.img_bi = Labels.run_DocEnTR_single(model, device, Ruler.img, do_skeletonize)
    Ruler.img_bi_sweep, pred_class, pred_class_orig, percentage1, level = binary_sweep(RulerCFG, Ruler.img)
    Ruler.img_bi_sweep = cv2.cvtColor(Ruler.img_bi_sweep, cv2.COLOR_RGB2GRAY)
    Ruler.img_bi_sweep = invert_if_white(Ruler.img_bi_sweep)
    if show_all_logs:
        logger.debug(f"[Ruler Binary Sweep] - Pred Class [{pred_class}] - Original Class [{pred_class_orig}] - Confidence [{percentage1}] - Binary Level [{level}]")

    ### Invert ruler if needed
    ruler_class_parts = Ruler.ruler_class.split('_')
    if 'white' in ruler_class_parts:
        Ruler.img_bi = cv2.bitwise_not(Ruler.img_bi)
    
    Ruler.img_bi_inv = cv2.bitwise_not(Ruler.img_bi)
    Ruler.img_bi_inv_sweep = cv2.bitwise_not(Ruler.img_bi_sweep)
    
    Ruler.img_bi_backup = Ruler.img_bi # THIS IS TEMP TODO should be ? maybe --> thresh, Ruler.img_bi_backup = cv2.threshold(Ruler.img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # cv2.imshow('bi',Ruler.img_bi_sweep)
    # cv2.waitKey(0)
    Ruler.img_bi_display = np.array(Ruler.img_bi)
    Ruler.img_bi_display = np.stack((Ruler.img_bi_display,)*3, axis=-1)
    return Ruler

def binary_sweep(RulerCFG, img):
    bi_sweep_img = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.uint8)
    bi_to_process = multi_threshold(img)
    imgs_pass = {}
    ind = 0
    if bi_to_process == []:
        return bi_sweep_img, 'sweep_failed', 'sweep_failed', 'sweep_failed', 'sweep_failed'
    else:
        for i, ruler_bi in enumerate(bi_to_process):
            pred_class, pred_class_orig, percentage1 = detect_bi_sweep(RulerCFG, ruler_bi)
            # cv2.imshow('i',ruler_bi)
            # cv2.waitKey(0)
            # print(f"bi:ruler_bi, pred_class:{pred_class}, pred_class_orig:{pred_class_orig}, percentage1:{percentage1},level {str(i*20)}")
            if pred_class == 'keep':
                imgs_pass[ind] = {'bi':ruler_bi, 'pred_class':pred_class, 'pred_class_orig':pred_class_orig, 'percentage1':percentage1,'level':str(i*20)}
                ind += 1

        if len(imgs_pass.keys()) == 0:
            return bi_sweep_img, 'sweep_failed', 'sweep_failed', 'sweep_failed', 'sweep_failed'
        else:
            if len(imgs_pass.keys()) == 1:
                bi_sweep_img = imgs_pass[0]['bi']
                pred_class = imgs_pass[0]['pred_class']
                pred_class_orig = imgs_pass[0]['pred_class_orig']
                percentage1 = imgs_pass[0]['percentage1']
                level = imgs_pass[0]['level']
            else:
                max_con = 0
                best_i = 0
                for i in range(len(imgs_pass.keys())):
                    p = imgs_pass[i]['percentage1']
                    if p >= max_con:
                        max_con = p
                        best_i = i
                bi_sweep_img = imgs_pass[best_i]['bi']
                pred_class = imgs_pass[best_i]['pred_class']
                pred_class_orig = imgs_pass[best_i]['pred_class_orig']
                percentage1 = imgs_pass[best_i]['percentage1']
                level = imgs_pass[best_i]['level']


            # elif len(imgs_pass.keys()) == 2:
            #     bi_sweep_img = imgs_pass[1]['bi']
            #     pred_class = imgs_pass[1]['pred_class']
            #     pred_class_orig = imgs_pass[1]['pred_class_orig']
            #     percentage1 = imgs_pass[1]['percentage1']
            #     level = imgs_pass[1]['level']
            # else:
            #     ind = int(round((len(imgs_pass.keys()) / 2),0)) - 1
            #     bi_sweep_img = imgs_pass[ind]['bi']
            #     pred_class = imgs_pass[ind]['pred_class']
            #     pred_class_orig = imgs_pass[ind]['pred_class_orig']
            #     percentage1 = imgs_pass[ind]['percentage1']
            #     level = imgs_pass[ind]['level']
            bi_sweep_img[bi_sweep_img<=200] = 0
            bi_sweep_img[bi_sweep_img>200] = 255
            return bi_sweep_img, pred_class, pred_class_orig, percentage1, level

def multi_threshold(img):
    bi_to_process = []
    for idx, i in enumerate(range(0, 255, 20)):
        threshold_value = i
        img_bi = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)[1]

        number_of_white = np.sum(img_bi == 255)
        number_of_black = np.sum(img_bi == 0)

        if (number_of_white != 0) and (number_of_black != 0):
            bi_to_process.append(img_bi)
    return bi_to_process

def invert_if_white(image):
    # Count the number of white and black pixels
    num_white = np.count_nonzero(image == 255)
    num_black = np.count_nonzero(image == 0)
    
    # If there are more white pixels, invert the colors
    if num_white > num_black:
        image = cv2.bitwise_not(image)
    
    return image

def invert_if_black(img):
    # count the number of white and black pixels
    num_white = cv2.countNonZero(img)
    num_black = img.size - num_white
    
    # invert the colors if there are more black pixels than white
    if num_black > num_white:
        img = cv2.bitwise_not(img)
    
    return img

def find_minimal_change_in_binarization(img_gray, version):
    if version == 'block':
        result_list = []

        for idx, i in enumerate(range(0, 255, 10)):
            threshold_value = i
            img_bi = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)[1]
            result = cv2.countNonZero(img_bi)
            result_list.append((threshold_value, result))

        # x = [i[0] for i in result_list]
        y = [i[1] for i in result_list]

        # Calculate the first derivative
        dy = np.diff(y)

        # Calculate the second derivative
        # ddy = np.diff(dy)
        # min_index = np.argmin(dy)
        # min_index = np.argmin(ddy)
        # Find the index of the minimum value of the first derivative
        diffs = [abs(dy[i+5]-dy[i]) for i in range(len(dy)-5)]
        min_index = diffs.index(min(diffs))
        best_threshold = result_list[min_index][0]

        # diffs = [abs(y[i+5]-y[i]) for i in range(len(y)-5)]
        # min_index1 = diffs.index(min(diffs))
        # min_index = diffs.index(min([i for i in diffs if i >= 0.01*max(diffs)]))
        # best_threshold = result_list[min_index][0]
        # Turn this and the commented lines above for testing

        img_bi = cv2.threshold(img_gray, best_threshold, 255, cv2.THRESH_BINARY)[1]
        return img_bi

    elif version == 'tick':
        result_list = []

        for idx, i in enumerate(range(0, 255, 10)):
            threshold_value = i
            img_bi = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)[1]
            result = cv2.countNonZero(img_bi)
            result_list.append((threshold_value, result))

        # x = [i[0] for i in result_list]
        y = [i[1] for i in result_list]

        diffs = [abs(y[i+5]-y[i]) for i in range(len(y)-5)]
        # min_index = diffs.index(min(diffs))
        min_index = diffs.index(min([i for i in diffs if i >= 0.01*max(diffs)]))
        best_threshold = result_list[min_index][0]

        img_bi = cv2.threshold(img_gray, best_threshold, 255, cv2.THRESH_BINARY)[1]
        return img_bi




def detect_ruler(logger, show_all_logs, RulerCFG, ruler_cropped, ruler_name):
    minimum_confidence_threshold = RulerCFG.cfg['leafmachine']['ruler_detection']['minimum_confidence_threshold']
    net = RulerCFG.net_ruler
    
    img = ClassifyRulerImage(ruler_cropped)

    # net = torch.jit.load(os.path.join(modelPath,modelName))
    # net.eval()

    with open(os.path.abspath(RulerCFG.path_to_class_names)) as f:
        classes = [line.strip() for line in f.readlines()]


    out = net(img.img_tensor)
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

    _, index = torch.max(out, 1)
    percentage1 = torch.nn.functional.softmax(out, dim=1)[0] * 100
    percentage1 = round(percentage1[index[0]].item(),2)
    pred_class1 = classes[index[0]]
    # Fix the 4thcm
    # if pred_class1 == 'tick_black_4thcm':
    #     self.Ruler.ruler_class = 'tick_black_cm_halfcm_4thcm'
    pred_class_orig = pred_class1
    

    if (percentage1 < minimum_confidence_threshold) or (percentage1 < (minimum_confidence_threshold*100)):
        pred_class_orig = pred_class1
        pred_class1 = f'fail_thresh_not_met__{pred_class_orig}'

    imgBG = create_overlay_bg(logger, show_all_logs, RulerCFG, img.img_sq)
    addText1 = ''.join(["Class: ", str(pred_class1)])
    if percentage1 < minimum_confidence_threshold:
        addText1 = ''.join(["Class: ", str(pred_class1), '< thresh: ', str(pred_class_orig)])

    addText2 = "Confidence: "+str(percentage1)
    newName = '.'.join([ruler_name ,'jpg'])
    # newName = newName.split(".")[0] + "__overlay.jpg"
    imgOverlay = cv2.putText(img=imgBG, text=addText1, org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(155, 155, 155),thickness=1)
    imgOverlay = cv2.putText(img=imgOverlay, text=addText2, org=(10, 45), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(155, 155, 155),thickness=1)
    if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_validation']:
        cv2.imwrite(os.path.join(RulerCFG.dir_ruler_validation,newName),imgOverlay)

    summary_message = [[''.join(["Class: ", str(pred_class1)])],
                        [''.join(["Confidence: ", str(percentage1), "%"])]]
    message = ''.join(["Class: ", str(pred_class1), " Confidence: ", str(percentage1), "%"])
    # Print_Verbose(RulerCFG.cfg,1,message).green()

    if show_all_logs:
        logger.info(message)
    try:
        torch.cuda.empty_cache()
    except:
        pass
    return pred_class1, pred_class_orig, percentage1, imgOverlay, summary_message

def detect_bi_sweep(RulerCFG, ruler_bi):
    minimum_confidence_threshold = RulerCFG.cfg['leafmachine']['ruler_detection']['minimum_confidence_threshold']
    net = RulerCFG.net_ruler_bi
    
    img = ClassifyRulerImage(ruler_bi)

    # net = torch.jit.load(os.path.join(modelPath,modelName))
    # net.eval()

    with open(os.path.abspath(RulerCFG.path_to_class_names_bi)) as f:
        classes = [line.strip() for line in f.readlines()]


    out = net(img.img_tensor)
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

    _, index = torch.max(out, 1)
    percentage1 = torch.nn.functional.softmax(out, dim=1)[0] * 100
    percentage1 = round(percentage1[index[0]].item(),2)
    pred_class = classes[index[0]]
    # Fix the 4thcm
    # if pred_class1 == 'tick_black_4thcm':
    #     self.Ruler.ruler_class = 'tick_black_cm_halfcm_4thcm'
    pred_class_orig = pred_class
    

    if (percentage1 < minimum_confidence_threshold) or (percentage1 < (minimum_confidence_threshold*100)):
        pred_class_orig = pred_class
        pred_class = f'fail_thresh_not_met__{pred_class_orig}'

    try:
        torch.cuda.empty_cache()
    except:
        pass
    return pred_class, pred_class_orig, percentage1

@dataclass
class RulerConfig:

    path_to_config: str = field(init=False)
    path_to_model: str = field(init=False)
    path_to_class_names: str = field(init=False)

    cfg: str = field(init=False)

    path_ruler_output_parent: str = field(init=False)
    dir_ruler_validation: str = field(init=False)
    dir_ruler_validation_summary: str = field(init=False)
    dir_ruler_processed: str = field(init=False)
    dir_ruler_data: str = field(init=False)

    net_ruler: object = field(init=False)

    def __init__(self, dir_home, Dirs, cfg, device) -> None:
        self.path_to_config = dir_home
        self.cfg = cfg

        self.path_to_model = os.path.join(dir_home,'leafmachine2','machine','ruler_classifier','model')
        self.path_to_class_names = os.path.join(dir_home, 'leafmachine2','machine','ruler_classifier','ruler_classes.txt')
        self.path_to_class_names_bi = os.path.join(dir_home, 'leafmachine2','machine','ruler_classifier','binary_classes.txt')

        self.path_ruler_output_parent = Dirs.ruler_info
        self.dir_ruler_validation = Dirs.ruler_validation
        self.dir_ruler_validation_summary =  Dirs.ruler_validation_summary
        self.dir_ruler_processed =  Dirs.ruler_processed
        self.dir_ruler_data =  Dirs.ruler_data

        # if self.cfg['leafmachine']['ruler_detection']['detect_ruler_type']:
        # try:
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0' # use only GPU 0
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        model_name = self.cfg['leafmachine']['ruler_detection']['ruler_detector']
        
        # torch.cuda.set_device(0)

        # checkpoint = torch.load(os.path.join(self.path_to_model,model_name), map_location='cuda:0')
        # checkpoint['state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}


        # Create the model architecture
        # model = models.resnet18(pretrained=True)
        # Load the state dict into the model
        # model.load_state_dict(checkpoint['state_dict'])

        # model.load_state_dict(checkpoint['model_state_dict'])
        

        try:
            self.net_ruler = torch.jit.load(os.path.join(self.path_to_model,model_name), map_location=f'cuda:{device}')
        except:
            self.net_ruler = torch.jit.load(os.path.join(self.path_to_model,model_name), map_location='cpu')
        self.net_ruler.eval()
        try:
            self.net_ruler.to(f'cuda:{device}') # specify device as 'cuda:0'
        except:
            self.net_ruler.to('cpu')
        # torch.jit.save(self.net_ruler, '/home/brlab/Dropbox/LeafMachine2/leafmachine2/machine/ruler_classifier/model/ruler_classifier_38classes_v-1.pt')
        # torch.save(self.net_ruler.state_dict(), '/home/brlab/Dropbox/LeafMachine2/leafmachine2/machine/ruler_classifier/model/ruler_classifier_38classes_v-1.pt')

        print(f"Loaded ruler classifier network: {os.path.join(self.path_to_model,model_name)}")
        # except:
        #     logger.info("Could not load ruler classifier network")

        model_name_bi = self.cfg['leafmachine']['ruler_detection']['ruler_binary_detector']
        try:
            self.net_ruler_bi = torch.jit.load(os.path.join(self.path_to_model, model_name_bi), map_location=f'cuda:{device}')
        except:
            self.net_ruler_bi = torch.jit.load(os.path.join(self.path_to_model, model_name_bi), map_location='cpu')
        self.net_ruler_bi.eval()
        try:
            self.net_ruler_bi.to(f'cuda:{device}') # specify device as 'cuda:0'
        except:
            self.net_ruler_bi.to('cpu') # specify device as 'cuda:0'
        print(f"Loaded ruler binary classifier network: {os.path.join(self.path_to_model, model_name_bi)}")

@dataclass
class ClassifyRulerImage:
    img_path: None
    img: ndarray = field(init=False)
    img_sq: ndarray = field(init=False)
    img_t: ndarray = field(init=False)
    img_tensor: object = field(init=False)
    transform: object = field(init=False)

    def __init__(self, img) -> None:
        try:
            self.img = img
        except:
            self.img = cv2.imread(self.img_path)
        # self.img_sq = squarify(self.img,showImg=False,makeSquare=True,sz=360) # for model_scripted_resnet_720.pt
        self.img_sq = squarify_tile_four_versions(self.img, showImg=False, makeSquare=True, sz=720) # for 
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.img_t = self.transforms(self.img_sq)
        try:
            self.img_tensor = torch.unsqueeze(self.img_t, 0).cuda()
        except:
            self.img_tensor = torch.unsqueeze(self.img_t, 0).cpu()

@dataclass
class RulerImage:
    img_path: str
    img_fname: str
    summary_message: str

    img: ndarray = field(init=False)

    img_bi: ndarray = field(init=False)
    img_bi_inv: ndarray = field(init=False)
    img_bi_backup: ndarray = field(init=False)

    img_copy: ndarray = field(init=False)
    img_gray: ndarray = field(init=False)
    img_edges: ndarray = field(init=False)
    img_bi_display: ndarray = field(init=False)
    img_bi: ndarray = field(init=False)
    img_best: ndarray = field(init=False)
    img_type_overlay: ndarray = field(init=False)
    img_ruler_overlay: ndarray = field(init=False)
    img_total_overlay: ndarray = field(init=False)
    img_block_overlay: ndarray = field(init=False)

    avg_angle: float = 0
    ruler_class: str = field(init=False)
    ruler_class_pred: str = field(init=False)
    ruler_class_percentage: str = field(init=False)
    

    def __init__(self, img, img_fname) -> None:
        self.img = make_img_hor(img)
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img_copy = self.img.copy()
        self.img_copy = stack_2_imgs(self.img_copy, self.img_copy) # for the double binary [bi, skel]
        self.img_fname = img_fname
        self.summary_message = 'Fail'

class Block:
    def __init__(self, img_bi: ndarray, img_bi_overlay: ndarray, success_sort: str):
        self.img_bi = img_bi
        self.img_bi_overlay = img_bi_overlay
        self.success_sort = success_sort

        # Initialize other properties with default values
        self.img_bi_copy = img_bi.copy()  # Create a copy of img_bi
        self.img_result = None

        self.use_points = []
        self.point_types = []
        self.x_points = []
        self.y_points = []
        self.axis_major_length = []
        self.axis_minor_length = []
        self.conversion_factor = []
        self.conversion_location = []
        self.conversion_location_options = None
        self.conversion_factor_pass = False

        self.largest_blobs = []
        self.remaining_blobs = []

        self.plot_points_1cm = []
        self.plot_points_10cm = []
        self.plot_points = []
        # self.img_bi[self.img_bi < 128] = 0
        # self.img_bi[self.img_bi >= 128] = 255
        # self.img_bi_copy[self.img_bi_copy < 40] = 0
        # self.img_bi_copy[self.img_bi_copy >= 40] = 255
        # self.img_bi_overlay[self.img_bi_overlay < 40] = 0
        # self.img_bi_overlay[self.img_bi_overlay >= 40] = 255

    def whiter_thresh(self) -> None:
        self.img_bi_copy[self.img_bi_copy < 240] = 0
        self.img_bi_copy[self.img_bi_copy >= 240] = 255

'''
####################################
####################################
                Basics
####################################
####################################
'''
# def add_ruler_to_Project(Project, batch, Ruler, BlockCandidate, filename, ruler_crop_name):
#     Project.project_data_list[batch][filename]['Ruler_Info'].append({ruler_crop_name: Ruler})
#     Project.project_data_list[batch][filename]['Ruler_Data'].append({ruler_crop_name: BlockCandidate})

#     # if 'block' in Ruler.ruler_class:
#     #     Project.project_data[filename]['Ruler_Info'].append({ruler_crop_name: Ruler})
#     #     Project.project_data[filename]['Ruler_Data'].append({ruler_crop_name: BlockCandidate})
#     # elif 'tick' in Ruler.ruler_class:
#     #     Project.project_data[filename]['Ruler_Info'].append({ruler_crop_name: Ruler})
#     #     Project.project_data[filename]['Ruler_Data'].append({ruler_crop_name: BlockCandidate})
#     #     print('tick')
#     return Project

def yolo_to_position_ruler(annotation, height, width):
    return ['ruler', 
        int((annotation[1] * width) - ((annotation[3] * width) / 2)), 
        int((annotation[2] * height) - ((annotation[4] * height) / 2)), 
        int(annotation[3] * width) + int((annotation[1] * width) - ((annotation[3] * width) / 2)), 
        int(annotation[4] * height) + int((annotation[2] * height) - ((annotation[4] * height) / 2))]

def make_img_hor(img):
    # Make image horizontal
    try:
        h,w,c = img.shape
    except:
        h,w = img.shape
    if h > w:
        img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def create_overlay_bg_120(logger, show_all_logs, RulerCFG, img):
    try:
        try:
            h,w,_ = img.shape
            imgBG = np.zeros([h+120,w,3], dtype=np.uint8)
            imgBG[:] = 0
        except:
            img = binary_to_color(img)
            h,w,_ = img.shape
            imgBG = np.zeros([h+120,w,3], dtype=np.uint8)
            imgBG[:] = 0

        try:
            imgBG[120:img.shape[0]+120, :img.shape[1],:] = img
        except:
            imgBG[120:img.shape[0]+120, :img.shape[1]] = img

    except Exception as e:
        m = ''.join(['create_overlay_bg() exception: ',e.args[0]])
        # Print_Verbose(RulerCFG.cfg, 2, m).warning()
        if show_all_logs:
            logger.debug(m)
        img = np.stack((img,)*3, axis=-1)
        h,w,_ = img.shape
        imgBG = np.zeros([h+120,w,3], dtype=np.uint8)
        imgBG[:] = 0

        imgBG[120:img.shape[0]+120,:img.shape[1],:] = img
    return imgBG


def create_overlay_bg_3(logger, show_all_logs, RulerCFG, img):
    try:
        try:
            h,w,_ = img.shape
            imgBG = np.zeros([h+170,w,3], dtype=np.uint8)
            imgBG[:] = 0
        except:
            img = binary_to_color(img)
            h,w,_ = img.shape
            imgBG = np.zeros([h+170,w,3], dtype=np.uint8)
            imgBG[:] = 0

        try:
            imgBG[170:img.shape[0]+170, :img.shape[1],:] = img
        except:
            imgBG[170:img.shape[0]+170, :img.shape[1]] = img

    except Exception as e:
        m = ''.join(['create_overlay_bg() exception: ',e.args[0]])
        # Print_Verbose(RulerCFG.cfg, 2, m).warning()
        if show_all_logs:
            logger.debug(m)
        img = np.stack((img,)*3, axis=-1)
        h,w,_ = img.shape
        imgBG = np.zeros([h+170,w,3], dtype=np.uint8)
        imgBG[:] = 0

        imgBG[170:img.shape[0]+170,:img.shape[1],:] = img
    return imgBG

def create_overlay_bg(logger, show_all_logs, RulerCFG, img):
    try:
        try:
            h,w,_ = img.shape
            imgBG = np.zeros([h+60,w,3], dtype=np.uint8)
            imgBG[:] = 0
        except:
            img = binary_to_color(img)
            h,w,_ = img.shape
            imgBG = np.zeros([h+60,w,3], dtype=np.uint8)
            imgBG[:] = 0

        try:
            imgBG[60:img.shape[0]+60, :img.shape[1],:] = img
        except:
            imgBG[60:img.shape[0]+60, :img.shape[1]] = img

    except Exception as e:
        m = ''.join(['create_overlay_bg() exception: ',e.args[0]])
        # Print_Verbose(RulerCFG.cfg, 2, m).warning()
        if show_all_logs:
            logger.debug(m)
        img = np.stack((img,)*3, axis=-1)
        h,w,_ = img.shape
        imgBG = np.zeros([h+60,w,3], dtype=np.uint8)
        imgBG[:] = 0

        imgBG[60:img.shape[0]+60,:img.shape[1],:] = img
    return imgBG

def binary_to_color(binary_image):
    color_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)
    color_image[binary_image == 1] = (255, 255, 255)
    return color_image

def pad_binary_img(img,h,w,n):
    imgBG = np.zeros([h+n,w], dtype=np.uint8)
    imgBG[:] = 0
    imgBG[:h,:w] = img
    return imgBG

def stack_2_imgs(img1,img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    img3 = np.zeros((h1+h2, max(w1,w2),3), dtype=np.uint8)
    img3[:,:] = (0,0,0)

    img3[:h1, :w1,:3] = img1
    try:
        img3[h1:h1+h2, :w2,:3] = img2
    except:
        img3[h1:h1+h2, :w2,:3] = binary_to_color(img2)
    return img3

def stack_2_imgs_bi(img1,img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    img3 = np.zeros((h1+h2, max(w1,w2)), dtype=np.uint8)
    img3[:h1, :w1] = img1
    img3[h1:h1+h2, :w2] = img2
    return img3

def check_ruler_type(ruler_class,option):
    ind = ruler_class.find(option)
    if ind == -1:
        return False
    else:
        return True

def create_white_bg(img,squarifyRatio,h,w):
    w_plus = w
    # if (w_plus % squarifyRatio) != 0:
    # while (w_plus % squarifyRatio) != 0:
    #     w_plus += 1
    
    imgBG = np.zeros([h,w_plus,3], dtype=np.uint8)
    imgBG[:] = 255

    imgBG[:img.shape[0],:img.shape[1],:] = img
    # cv2.imshow('Single Channel Window', imgBG)
    # cv2.waitKey(0)
    return imgBG

def stack_image_quartile_rotate45_cropped_corners(img, q_increment, h, w, showImg):
    # cv2.imshow('Original', img)
    # cv2.waitKey(0)

    rotate_options = [-135, -45, 45, 135]

    imgBG = np.zeros([h*2,h*2,3], dtype=np.uint8)
    imgBG[:] = 255

    increment = 0
    for row in range(0,2):
        for col in range(0,2):
            ONE = (row * h)
            TWO = ((row * h) + h)
            THREE = (col * h)
            FOUR = (col * h) + h

            one = (q_increment*increment)
            two = (q_increment*increment) + h

            if (increment < 3) and (two < w):
                # imgBG[ONE : TWO, THREE : FOUR] = img[:, one : two]
                rotated = imutils.rotate_bound(img[:, one : two], rotate_options[increment])
                # Calculate the center of the rotated image
                center_x = int(rotated.shape[1] / 2)
                center_y = int(rotated.shape[0] / 2)
                # Calculate the coordinates of the top-left corner of the cropped image
                crop_x = max(0, center_x - int(h/2))
                crop_y = max(0, center_y - int(h/2))
                # Crop the rotated image to the desired size
                cropped = rotated[crop_y:crop_y+h, crop_x:crop_x+h]
                imgBG[ONE : TWO, THREE : FOUR] = cropped
            else:
                # imgBG[ONE : TWO, THREE : FOUR] = img[:, w - h : w]
                rotated = imutils.rotate_bound(img[:, w - h : w], rotate_options[increment])
                # Calculate the center of the rotated image
                center_x = int(rotated.shape[1] / 2)
                center_y = int(rotated.shape[0] / 2)
                # Calculate the coordinates of the top-left corner of the cropped image
                crop_x = max(0, center_x - int(h/2))
                crop_y = max(0, center_y - int(h/2))
                # Crop the rotated image to the desired size
                cropped = rotated[crop_y:crop_y+h, crop_x:crop_x+h]
                imgBG[ONE : TWO, THREE : FOUR] = cropped
            increment += 1

    if showImg:
        cv2.imshow('squarify_quartile()', imgBG)
        cv2.waitKey(0)
    return imgBG

def stack_image_quartile_rotate45(img, q_increment, h, w, showImg):
    # cv2.imshow('Original', img)
    # cv2.waitKey(0)

    rotate_options = [-135, -45, 45, 135]

    imgBG = np.zeros([h*2,h*2,3], dtype=np.uint8)
    imgBG[:] = 255

    increment = 0
    for row in range(0,2):
        for col in range(0,2):
            ONE = (row * h)
            TWO = ((row * h) + h)
            THREE = (col * h)
            FOUR = (col * h) + h

            one = (q_increment*increment)
            two = (q_increment*increment) + h

            if (increment < 3) and (two < w):
                # imgBG[ONE : TWO, THREE : FOUR] = img[:, one : two]
                rotated = imutils.rotate_bound(img[:, one : two], rotate_options[increment])
                add_dim1 = rotated.shape[0] - ONE
                add_dim2 = rotated.shape[0] - TWO
                add_dim3 = rotated.shape[0] - THREE
                add_dim4 = rotated.shape[0] - FOUR
                imgBG[ONE : TWO, THREE : FOUR] = cv2.resize(rotated,  (FOUR - THREE, TWO - ONE))
            else:
                # imgBG[ONE : TWO, THREE : FOUR] = img[:, w - h : w]
                rotated = imutils.rotate_bound(img[:, w - h : w], rotate_options[increment])
                imgBG[ONE : TWO, THREE : FOUR] = cv2.resize(rotated,  (FOUR - THREE, TWO - ONE))
            increment += 1


    if showImg:
        cv2.imshow('squarify_quartile()', imgBG)
        cv2.waitKey(0)
    return imgBG

def squarify_maxheight(img, h, w, showImg=False):
    """
    Resizes input image so that height is the maximum and width is adjusted to make the image square.
    """
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    if random.random() < 0.5:
        img = cv2.rotate(img, cv2.ROTATE_180)
    
    resized = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
    if showImg:
        cv2.imshow('squarify_maxheight()', resized)
        cv2.waitKey(0)
    return resized

def stack_image_quartile(img, q_increment, h, w, showImg):
    # cv2.imshow('Original', img)
    # cv2.waitKey(0)

    imgBG = np.zeros([h*2,h*2,3], dtype=np.uint8)
    imgBG[:] = 255

    increment = 0
    for row in range(0,2):
        for col in range(0,2):
            ONE = (row * h)
            TWO = ((row * h) + h)
            THREE = (col * h)
            FOUR = (col * h) + h

            one = (q_increment*increment)
            two = (q_increment*increment) + h

            if (increment < 3) and (two < w):
                imgBG[ONE : TWO, THREE : FOUR] = img[:, one : two]
            else:
                imgBG[ONE : TWO, THREE : FOUR] = img[:, w - h : w]
            increment += 1

    if showImg:
        cv2.imshow('squarify_quartile()', imgBG)
        cv2.waitKey(0)
    return imgBG

def stack_image_nine(img, q_increment, h, w, showImg):
    # cv2.imshow('Original', img)
    # cv2.waitKey(0)

    imgBG = np.zeros([h*3,h*3,3], dtype=np.uint8)
    imgBG[:] = 255

    increment = 0
    for row in range(0,3):
        for col in range(0,3):
            ONE = (row * h)
            TWO = ((row * h) + h)
            THREE = (col * h)
            FOUR = (col * h) + h

            one = (q_increment*increment)
            two = (q_increment*increment) + h

            if (increment < 8) and (two < w):
                imgBG[ONE : TWO, THREE : FOUR] = img[:, one : two]
            else:
                imgBG[ONE : TWO, THREE : FOUR] = img[:, w - h : w]
            increment += 1
            # if showImg:
            #     cv2.imshow('Single Channel Window', imgBG)
            #     cv2.waitKey(0)

    if showImg:
        cv2.imshow('squarify_nine()', imgBG)
        cv2.waitKey(0)
    return imgBG

def stack_image(img,squarifyRatio,h,w_plus,showImg):
    # cv2.imshow('Original', img)
    wChunk = int(w_plus/squarifyRatio)
    hTotal = int(h*squarifyRatio)
    imgBG = np.zeros([hTotal,wChunk,3], dtype=np.uint8)
    imgBG[:] = 255

    wStart = 0
    wEnd = wChunk
    for i in range(1,squarifyRatio+1):
        wStartImg = (wChunk*i)-wChunk
        wEndImg =  wChunk*i
        
        hStart = (i*h)-h
        hEnd = i*h
        # cv2.imshow('Single Channel Window', imgPiece)
        # cv2.waitKey(0)
        imgBG[hStart:hEnd,wStart:wEnd] = img[:,wStartImg:wEndImg]
    if showImg:
        cv2.imshow('squarify()', imgBG)
        cv2.waitKey(0)
    return imgBG

def add_text_to_stacked_img(angle,img, summary_message):
    addText1 = "Angle(deg): "+str(round(angle,3))+' Imgs: Orig,Binary,Skeleton,Validation'
    img = cv2.putText(img=img, text=addText1, org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255),thickness=1)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    summary_message.append([addText1])
    return img, summary_message

def add_text_to_img(text,img):
    addText = text
    img = cv2.putText(img=img, text=addText, org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255),thickness=1)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    return img

def add_text_to_img_list(text,img):
    for i, t in enumerate(text):
        addText = t[0]
        img = cv2.putText(img=img, text=addText, org=(10, (i * 20)+25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255),thickness=1)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
    return img

'''
####################################
####################################
            Squarify
####################################
####################################
'''
def calc_squarify_ratio(img):
    doStack = False
    h,w,c = img.shape

    # Extend width so it's a multiple of h
    ratio = w/h
    ratio_plus = math.ceil(ratio)
    w_plus = ratio_plus*h

    ratio_go = w/h
    if ratio_go > 4:
        doStack = True

    squarifyRatio = 0
    if doStack:
        # print(f'This should equal 0 --> {w_plus % h}')
        for i in range(1,ratio_plus):
            if ((i*h) < (w_plus/i)):
                continue
            else:
                squarifyRatio = i - 1
                break
        # print(f'Optimal stack_h: {squarifyRatio}')
        while (w % squarifyRatio) != 0:
            w += 1
    return doStack,squarifyRatio,w,h

def calc_squarify(img,cuts):
    h,w,c = img.shape
    q_increment = int(np.floor(w / cuts))
    return q_increment,w,h

def squarify(imgSquarify,showImg,makeSquare,sz):
    imgSquarify = make_img_hor(imgSquarify)
    doStack,squarifyRatio,w_plus,h = calc_squarify_ratio(imgSquarify)

    if doStack:
        imgBG = create_white_bg(imgSquarify,squarifyRatio,h,w_plus)
        imgSquarify = stack_image(imgBG,squarifyRatio,h,w_plus,showImg)

    if makeSquare:
        dim = (sz, sz)
        imgSquarify = cv2.resize(imgSquarify, dim, interpolation = cv2.INTER_AREA)
    
    if random.random() < 0.5:
        imgSquarify = cv2.rotate(imgSquarify, cv2.ROTATE_180)

    return imgSquarify

def squarify_rotate45(imgSquarify, showImg, makeSquare, sz, doFlip):
    imgSquarify = make_img_hor(imgSquarify)
    
    # if doFlip:
    #     imgSquarify = cv2.rotate(imgSquarify,cv2.ROTATE_180) 

    q_increment,w,h = calc_squarify(imgSquarify,4)

    imgSquarify = stack_image_quartile_rotate45(imgSquarify, q_increment, h, w, showImg)

    if makeSquare:
        dim = (sz, sz)
        imgSquarify = cv2.resize(imgSquarify, dim, interpolation = cv2.INTER_AREA)
    return imgSquarify

def squarify_quartiles(imgSquarify, showImg, makeSquare, sz, doFlip):
    imgSquarify = make_img_hor(imgSquarify)
    
    if doFlip:
        imgSquarify = cv2.rotate(imgSquarify,cv2.ROTATE_180) 

    q_increment,w,h = calc_squarify(imgSquarify,4)

    imgSquarify = stack_image_quartile(imgSquarify, q_increment, h, w, showImg)

    if makeSquare:
        dim = (sz, sz)
        imgSquarify = cv2.resize(imgSquarify, dim, interpolation = cv2.INTER_AREA)

    if random.random() < 0.5:
        imgSquarify = cv2.rotate(imgSquarify, cv2.ROTATE_180)

    return imgSquarify

def squarify_nine(imgSquarify, showImg, makeSquare, sz):
    imgSquarify = make_img_hor(imgSquarify)

    q_increment,w,h = calc_squarify(imgSquarify,9)

    imgSquarify = stack_image_nine(imgSquarify, q_increment, h, w, showImg)

    if makeSquare:
        dim = (sz, sz)
        imgSquarify = cv2.resize(imgSquarify, dim, interpolation = cv2.INTER_AREA)

    if random.random() < 0.5:
        imgSquarify = cv2.rotate(imgSquarify, cv2.ROTATE_180)

    return imgSquarify

def squarify_tile_four_versions(imgSquarify, showImg, makeSquare, sz):
    h = int(sz*2)
    w = int(sz*2)
    h2 = int(h/2)
    w2 = int(w/2)
    sq1 = squarify(imgSquarify,showImg,makeSquare,sz)
    sq2 = squarify_maxheight(imgSquarify, h/2, w/2, showImg)
    # sq2 = squarify_rotate45(imgSquarify, showImg, makeSquare, sz, doFlip=False)
    sq3 = squarify_quartiles(imgSquarify, showImg, makeSquare, sz, doFlip=showImg)
    sq4 = squarify_nine(imgSquarify, showImg, makeSquare, sz)


    imgBG = np.zeros([h,w,3], dtype=np.uint8)
    imgBG[:] = 255

    imgBG[0:h2, 0:h2 ,:] = sq1
    imgBG[:h2, h2:w ,:] = sq2
    imgBG[h2:w, :h2 ,:] = sq3
    imgBG[h2:w, h2:w ,:] = sq4

    if showImg:
        cv2.imshow('Four versions: squarify(), squarify_quartiles(), squarify_quartiles(rotate180), squarify_nine()', imgBG)
        cv2.waitKey(0)

    return imgBG

'''
####################################
####################################
            Process
####################################
####################################
'''
def straighten_img(logger, show_all_logs, RulerCFG, Ruler, useRegulerBinary, alternate_img, Dirs, do_skip_skeleton, is_block_tick, do_skip_morph_cleaning):
    
    if useRegulerBinary:
        ruler_to_correct = Ruler.img_bi
    else:
        ruler_to_correct = np.uint8(alternate_img) # BlockCandidate.remaining_blobs[0].values


    image_rotated, Ruler.img, Ruler.img_bi_sweep, angle = rotate_bi_image_hor(ruler_to_correct, Ruler.img, Ruler.img_bi_sweep, do_skip_skeleton, is_block_tick, do_skip_morph_cleaning) # rotated_img, rotated_img_rgb, angle

    # update all relevant images - rotate
    # Stack 3 rgb images for the overlay
    Ruler.img_copy = stack_2_imgs(Ruler.img, Ruler.img) # Used to make the overlay
    Ruler.img_copy = stack_2_imgs(Ruler.img_copy, Ruler.img) # Used to make the overlay

    Ruler.img_bi_display = np.array(image_rotated)# Used to make the overlay
    Ruler.img_bi_display = np.stack((Ruler.img_bi_display,)*3, axis=-1)# Used to make the overlay
    Ruler.img_bi = image_rotated # Used to do the actual calculations# Used to make the overlay

    
    if (angle != 0.0) or (angle != 0): # If the rotation was substantial
        Ruler.correction_success = True
        Ruler.avg_angle = angle
    else:
        Ruler.correction_success = False
        Ruler.avg_angle = 0

    ''' exception for grid rulers, revisit
    # Grid rulers will NOT get roatate, assumption is that they are basically straight already
    if check_ruler_type(Ruler.ruler_class,'grid') == False:
        if len(angles) > 0:
            Ruler.avg_angle = np.nanmean(angles)
            imgRotate = ndimage.rotate(Ruler.img,Ruler.avg_angle)
            imgRotate = make_img_hor(imgRotate)
        else:
            Ruler.avg_angle = 0
            imgRotate = Ruler.img
    else: 
        Ruler.avg_angle = 0
        imgRotate = Ruler.img
    '''
    newImg = stack_2_imgs(Ruler.img, Ruler.img_bi_display)

    newImg = create_overlay_bg(logger, show_all_logs, RulerCFG,newImg)
    newImg, Ruler.summary_message = add_text_to_stacked_img(Ruler.avg_angle,newImg, Ruler.summary_message)

    if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_validation']:
        newImg = stack_2_imgs(Ruler.img_type_overlay,newImg)

    Ruler.img_best = image_rotated
    Ruler.img_total_overlay = newImg

    # if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_validation']:
        # cv2.imwrite(os.path.join(Dirs.ruler_validation,'.'.join([Ruler.img_fname, 'jpg'])),Ruler.img_total_overlay)
    if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_processed']:
        cv2.imwrite(os.path.join(Dirs.ruler_processed,'.'.join([Ruler.img_fname, 'jpg'])),Ruler.img_best)
           
    # After saving the edges and imgBi to the compare file, flip for the class
    # Ruler.img_bi = ndimage.rotate(Ruler.img_bi,Ruler.avg_angle)
    # Ruler.img_bi = make_img_hor(Ruler.img_bi)
    ##### Ruler.img_edges = ndimage.rotate(Ruler.img_edges,Ruler.avg_angle) # no
    ###### Ruler.img_edges = make_img_hor(Ruler.img_edges) # no
    Ruler.img_gray = ndimage.rotate(Ruler.img_gray,Ruler.avg_angle)
    Ruler.img_gray = make_img_hor(Ruler.img_gray)

    # cv2.imwrite(os.path.join(RulerCFG.dir_ruler_overlay,'.'.join(['hi', 'jpg'])), Ruler.img_total_overlay)
    return Ruler

def rotate_bi_image_hor(binary_img, rgb_img, img_bi_sweep, remove_text_only, is_block_tick, do_skip_morph_cleaning):
    if not do_skip_morph_cleaning:
        if is_block_tick:
            og_binary_img = copy.deepcopy(binary_img)
            og_img_bi_sweep = copy.deepcopy(img_bi_sweep)

            white_before_bi = cv2.countNonZero(binary_img)
            white_before_sw = cv2.countNonZero(img_bi_sweep)

            # Perform morphological dilation to expand the text regions
            kernel = np.ones((3,3), np.uint8)
            # Perform morphological erosion to shrink the text regions back to their original size
            binary_img = cv2.erode(binary_img, kernel, iterations=1)
            img_bi_sweep = cv2.erode(img_bi_sweep, kernel, iterations=1)
            binary_img = cv2.dilate(binary_img, kernel, iterations=1)
            img_bi_sweep = cv2.dilate(img_bi_sweep, kernel, iterations=1)

            # Find the contours and hierarchy
            contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Find the largest contour by area
            max_area = 0
            largest_contour = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    largest_contour = contour
            # Remove the largest object by filling it with black color
            if largest_contour is not None:
                cv2.drawContours(binary_img, [largest_contour], 0, (0, 0, 0), thickness=cv2.FILLED)

            # Find the contours and hierarchy
            contours, hierarchy = cv2.findContours(img_bi_sweep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Find the largest contour by area
            max_area = 0
            largest_contour = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    largest_contour = contour

            # Remove the largest object by filling it with black color
            if largest_contour is not None:
                cv2.drawContours(img_bi_sweep, [largest_contour], 0, (0, 0, 0), thickness=cv2.FILLED)

            white_after_bi = cv2.countNonZero(binary_img)
            white_after_sw = cv2.countNonZero(img_bi_sweep)

            keep = True
            if white_after_bi < (white_before_bi * 0.05):
                keep = False
            if white_after_sw < (white_before_sw * 0.05):
                keep = False

            if not keep:
                binary_img = copy.deepcopy(og_binary_img)
                img_bi_sweep = copy.deepcopy(og_img_bi_sweep)


        if remove_text_only:
            opened_img = copy.deepcopy(binary_img)
            opened_img = remove_text(opened_img)
            opened_img = cv2.GaussianBlur(opened_img, (3, 3), 0)

            img_bi_sweep = remove_text(img_bi_sweep)
            img_bi_sweep = cv2.GaussianBlur(img_bi_sweep, (3, 3), 0)
            # Find contours of objects in the image
            contours, hierarchy = cv2.findContours(opened_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Loop through the contours
            for contour in contours:
                # Compute the area of the contour
                area = cv2.contourArea(contour)
                rect = cv2.minAreaRect(contour)
                (x, y), (w2, h2), angle = rect
                aspect_ratio = min([w2, h2]) / max([w2, h2])
                if aspect_ratio > (1/1.5):
                    cv2.drawContours(opened_img, [contour], 0, 0, -1)
            contours, hierarchy = cv2.findContours(img_bi_sweep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Loop through the contours
            for contour in contours:
                # Compute the area of the contour
                area = cv2.contourArea(contour)

                rect = cv2.minAreaRect(contour)
                (x, y), (w2, h2), angle = rect
                aspect_ratio = min([w2, h2]) / max([w2, h2])
                if aspect_ratio > (1/1.5):
                    cv2.drawContours(opened_img, [contour], 0, 0, -1)
        else:
            # Clean up the binary image using morphology operations
            # Determine the orientation of the image
            # cv2.imshow('binary_img', binary_img)
            # cv2.waitKey(0)
            
            # Determine the orientation of the image
            (h, w) = binary_img.shape
            max_dim = max([h, w])
            if h > w:
                binary_img = cv2.rotate(binary_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_bi_sweep = cv2.rotate(img_bi_sweep, cv2.ROTATE_90_COUNTERCLOCKWISE)

            img_bi_sweep = remove_text(img_bi_sweep)
            opened_img = remove_text(binary_img)

            # Clean up the binary image using morphology operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
            opened_img = cv2.morphologyEx(opened_img, cv2.MORPH_OPEN, kernel, iterations=1)
            # opened_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel, iterations=1)
            img_bi_sweep = cv2.morphologyEx(img_bi_sweep, cv2.MORPH_OPEN, kernel, iterations=1)
            # img_bi_sweep = cv2.morphologyEx(img_bi_sweep, cv2.MORPH_CLOSE, kernel, iterations=1)

            if max_dim >= 5000:
                opened_img = cv2.GaussianBlur(opened_img, (3, 3), 0)
                img_bi_sweep = cv2.GaussianBlur(img_bi_sweep, (3, 3), 0)
                kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                opened_img = cv2.morphologyEx(opened_img, cv2.MORPH_OPEN, kernel2, iterations=1)
                # opened_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel2, iterations=1)
                img_bi_sweep = cv2.morphologyEx(img_bi_sweep, cv2.MORPH_OPEN, kernel, iterations=2)
                # img_bi_sweep = cv2.morphologyEx(img_bi_sweep, cv2.MORPH_CLOSE, kernel, iterations=2)
            elif max_dim >= 3000:
                opened_img = cv2.GaussianBlur(opened_img, (3, 3), 0)
                img_bi_sweep = cv2.GaussianBlur(img_bi_sweep, (3, 3), 0)
                kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                opened_img = cv2.morphologyEx(opened_img, cv2.MORPH_OPEN, kernel2, iterations=1)
                # opened_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel2, iterations=1)
                img_bi_sweep = cv2.morphologyEx(img_bi_sweep, cv2.MORPH_OPEN, kernel2, iterations=1)
                # img_bi_sweep = cv2.morphologyEx(img_bi_sweep, cv2.MORPH_CLOSE, kernel2, iterations=1)
            elif max_dim >= 1000:
                opened_img = cv2.GaussianBlur(opened_img, (3, 3), 0)
                img_bi_sweep = cv2.GaussianBlur(img_bi_sweep, (3, 3), 0)
                kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                opened_img = cv2.morphologyEx(opened_img, cv2.MORPH_OPEN, kernel2, iterations=1)
                # opened_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel2, iterations=1)
                img_bi_sweep = cv2.morphologyEx(img_bi_sweep, cv2.MORPH_OPEN, kernel2, iterations=1)
                # img_bi_sweep = cv2.morphologyEx(img_bi_sweep, cv2.MORPH_CLOSE, kernel2, iterations=1)
            # closed_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Find contours of objects in the image
            contours, hierarchy = cv2.findContours(opened_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Loop through the contours
            for contour in contours:
                try:
                    # Compute the area of the contour
                    area = cv2.contourArea(contour)
                    rect = cv2.minAreaRect(contour)
                    (x, y), (w2, h2), angle = rect
                    aspect_ratio = min([w2, h2]) / max([w2, h2])

                    if area <= 7:
                        cv2.drawContours(opened_img, [contour], 0, 0, -1)
                        cv2.drawContours(img_bi_sweep, [contour], 0, 0, -1)

                    # peri = cv2.arcLength(contour, True)
                    # approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                    # if approx.shape[0] >= 5:
                    #     cv2.drawContours(opened_img, [contour], 0, 0, -1)
                    #     cv2.drawContours(img_bi_sweep, [contour], 0, 0, -1)

                    if aspect_ratio > (1/7):
                        if max_dim <= 1000:
                            # If the area is 5 pixels or less, set all pixels in the contour to 0
                            if area <= 10:
                                cv2.drawContours(opened_img, [contour], 0, 0, -1)
                        elif max_dim <= 2000:
                            # If the area is 5 pixels or less, set all pixels in the contour to 0
                            if area <= 13:
                                cv2.drawContours(opened_img, [contour], 0, 0, -1)
                        elif max_dim <= 4000:
                            # If the area is 5 pixels or less, set all pixels in the contour to 0
                            if area <= 17:
                                cv2.drawContours(opened_img, [contour], 0, 0, -1)
                        else:
                            if area <= 35:
                                cv2.drawContours(opened_img, [contour], 0, 0, -1)
                    else:
                        if max_dim > 3000:
                            if area <= 35:
                                cv2.drawContours(opened_img, [contour], 0, 0, -1)
                except: 
                    pass
        # opened_img = reduce_to_lines(opened_img)
        # img_bi_sweep = reduce_to_lines(img_bi_sweep)
    else:
        opened_img = binary_img



    LL = max(opened_img.shape) * 0.25   
    lines = cv2.HoughLinesP(opened_img, rho=1, theta=np.pi/180, threshold=25, minLineLength=LL/2, maxLineGap=5)

    angle = 0.0

    # Visualize the largest contours
    # cv2.imshow('largest_contours', img_bi_sweep)
    # cv2.waitKey(0)

    # Determine the rotation angle based on the detected lines
    if lines is not None:
        if len(lines) > 0:
            all_angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)
                all_angles.append(angle)

            angles, counts = np.unique(all_angles, return_counts=True)
            mode_index = np.argmax(counts)
            angle = angles[mode_index]

            # Rotate the image if the angle is not close to zero
            rotated_img = rotate_image_and_expand_bi(opened_img, angle)
            rotated_img_rgb = rotate_image_and_expand(rgb_img, angle)
            img_bi_sweep = rotate_image_and_expand(img_bi_sweep, angle)
        else:
            rotated_img = opened_img.copy()
            rotated_img_rgb = rgb_img
    else:
        rotated_img = opened_img.copy()
        rotated_img_rgb = rgb_img
    
    (h, w) = rotated_img.shape
    angle = math.degrees(angle)
    if h > w:
        if angle < 0:
            angle = angle + 90
            rotated_img = cv2.rotate(rotated_img, cv2.ROTATE_90_CLOCKWISE)
            rotated_img_rgb = cv2.rotate(rotated_img_rgb, cv2.ROTATE_90_CLOCKWISE)
            img_bi_sweep = cv2.rotate(img_bi_sweep, cv2.ROTATE_90_CLOCKWISE)
        else:
            angle = angle - 90
            rotated_img = cv2.rotate(rotated_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rotated_img_rgb = cv2.rotate(rotated_img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_bi_sweep = cv2.rotate(img_bi_sweep, cv2.ROTATE_90_COUNTERCLOCKWISE)


    # cv2.imshow('rot', rotated_img)
    # cv2.waitKey(0)
    # cv2.imshow('bi_remove_text', bi_remove_text)
    # cv2.waitKey(0)

    rotated_img = invert_if_white(rotated_img)
    img_bi_sweep = invert_if_white(img_bi_sweep)

    # cv2.imshow('largest_contours', img_bi_sweep)
    # cv2.waitKey(0)

    return rotated_img, rotated_img_rgb, img_bi_sweep, angle

def reduce_to_lines(img):
    # Apply Canny edge detection
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # Apply Hough Line Transform to detect lines in the image
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=50)

    # Create a mask for the lines that meet our orientation criteria
    mask = np.zeros_like(img)
    for line in lines:
        rho, theta = line[0]
        if (np.abs(theta - np.pi/2) < np.pi/4) or (np.abs(theta) < np.pi/4):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(mask, (x1, y1), (x2, y2), 255, 2)

    # Apply the mask to the original image to keep only the lines that meet our criteria
    img_with_zeros = img.copy()
    img_with_zeros[mask == 0] = 0

    # Display the original and processed images
    cv2.imshow('Original', img)
    cv2.imshow('With Zeros', img_with_zeros)
    cv2.waitKey(0)
    return img_with_zeros

def rotate_image_and_expand_bi(binary_img, angle):
    if abs(angle) >= np.deg2rad(1):
        (h, w) = binary_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, np.degrees(angle), 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated_img = cv2.warpAffine(binary_img, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    else:
        rotated_img = binary_img.copy()

    return rotated_img

def rotate_image_and_expand(rgb_img, angle):
    if abs(angle) >= np.deg2rad(1):
        (h, w) = rgb_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, np.degrees(angle), 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated_img = cv2.warpAffine(rgb_img, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    else:
        rotated_img = rgb_img.copy()

    return rotated_img


'''def rotate_bi_image_hor(binary_img):
    LL = max(binary_img.shape)*0.25
    # cv2.imshow('binary_img',binary_img)
    # cv2.waitKey(0)
    bi_remove_text = binary_img.copy()
    bi_remove_text = remove_text(bi_remove_text)
    # cv2.imshow('bi_remove_text',bi_remove_text)
    # cv2.waitKey(0)
    lines = cv2.HoughLinesP(bi_remove_text, 1, np.pi/180, 50, minLineLength=LL, maxLineGap=2)
    angle = 0.0
    if lines is not None:
        all_angles =[]
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            all_angles.append(angle)

        angles, counts = np.unique(all_angles, return_counts=True)
        mode_index = np.argmax(counts)
        angle = angles[mode_index]
        (h, w) = binary_img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, np.degrees(angle), 1.0)
        if angle >= abs(np.divide(math.pi, 180)): # more than 1 degree, then rotate
            rotated_img = cv2.warpAffine(binary_img, M, (w, h), flags=cv2.INTER_NEAREST)
            # cv2.imshow('bi_remove_text',bi_remove_text)
            # cv2.waitKey(0)
            # cv2.imshow('rotated_img',rotated_img)
            # cv2.waitKey(0)
        else:
            rotated_img = binary_img.copy()
    else:
        rotated_img = binary_img.copy()
    # cv2.imshow('rotated_img',rotated_img)
    # cv2.waitKey(0)
    return rotated_img, angle'''

def remove_text(img):
    img_copy = img.copy()#copy.deepcopy(img)
    img_copy_not = cv2.bitwise_not(img_copy)
    result = [img_copy, img_copy_not]
    result_filled = []
    for img in result:
        # Perform morphological dilation to expand the text regions
        kernel = np.ones((3,3), np.uint8)
        dilation = cv2.dilate(img, kernel, iterations=1)

        # Perform morphological erosion to shrink the text regions back to their original size
        erosion = cv2.erode(dilation, kernel, iterations=1)

        # Find contours in the processed image
        contours, _ = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours to keep only those likely to correspond to text regions
        text_contours = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            if aspect_ratio < 1/3 or aspect_ratio > 3/2:
                continue
            text_contours.append(c)

        # Draw filled contours on the copy of the binary image to fill in the text regions
        result_filled.append(cv2.drawContours(img, text_contours, -1, 255, -1))
    
    diff = [np.count_nonzero(img - img_copy) for img in result_filled]
    idx = np.argmax(diff)
    out = result_filled[idx]
    out = invert_if_white(out)
    return out


def locate_ticks_centroid_inline(chunkAdd,scanSize, i, logger, max_dim):
    tolerance = 5
    props = regionprops_table(label(chunkAdd), properties=('centroid',
                                            'orientation',
                                            'axis_major_length',
                                            'axis_minor_length'))
    props = pd.DataFrame(props)
    # Calculate the average width of the objects
    if np.any(props['axis_major_length']):
        widths = np.sqrt(props['axis_major_length'] * props['axis_minor_length'])
        avg_width = np.nanmean(widths)

        centoid = props['centroid-1']
        peak_pos = np.transpose(np.array(centoid))
        dst_matrix = peak_pos - peak_pos[:, None]
        dst_matrix = dst_matrix[~np.eye(dst_matrix.shape[0],dtype=bool)].reshape(dst_matrix.shape[0],-1)
        if (dst_matrix.shape[0] > 0) and (dst_matrix.shape[1] > 0):
            # Calculate distances
            dist = np.min(np.abs(dst_matrix), axis=1)
            # distUse = dist[dist > 2]

            # Ruler out values larger than the max dim 
            distUse = dist[(dist > 2) & (dist * 10 < max_dim)]

            
            use_kmeans = False
            for i in range(len(distUse)-1):
                val = distUse[0]
                compare = distUse[i]
                if abs(val - compare) > tolerance:
                    use_kmeans = True

            if use_kmeans:
                # Use k-means clustering to find the two dominant patterns
                kmeans = KMeans(n_clusters=2, random_state=2022, n_init=10)
                kmeans.fit(np.array(distUse).reshape(-1, 1))
                labels = kmeans.labels_

                # Determine which cluster has the higher count of data points
                counts = [sum(labels == i) for i in range(2)]
                dominant_cluster = np.argmax(counts)
                
                # Get dominant and subordinate values
                dom_values = distUse[labels == dominant_cluster]
                sub_values = distUse[labels != dominant_cluster]

                # logger.debug(f"Primary pattern values - {list(dom_values)}")
                # logger.debug(f"Secondary pattern values - {list(sub_values)}")

                val_pair = []
                # dom
                if len(dom_values) >= 3:
                    # if len(dom_values) >= 5:
                    #     dom_values = remove_outliers(dom_values)
                    
                    plotPtsX = peak_pos[(dist > 2) & (dist * 10 < max_dim)]
                    plotPtsX = plotPtsX[labels == dominant_cluster]
                    plotPtsY = np.repeat(round((scanSize/2) + (((scanSize * i) + (scanSize * i + scanSize)) / 2)),plotPtsX.size)
                    npts = len(plotPtsY)

                    val_pair.append([plotPtsX,plotPtsY,dom_values,npts,peak_pos,avg_width])
                else:
                    val_pair.append([None,None,None,None,None,None])
                #sub
                if len(sub_values) >= 3:
                    # if len(sub_values) >= 5:
                    #     sub_values = remove_outliers(sub_values)
                    
                    plotPtsX = peak_pos[(dist > 2) & (dist * 10 < max_dim)]
                    plotPtsX = plotPtsX[labels != dominant_cluster]
                    plotPtsY = np.repeat(round((scanSize/2) + (((scanSize * i) + (scanSize * i + scanSize)) / 2)),plotPtsX.size)
                    npts = len(plotPtsY)

                    val_pair.append([plotPtsX,plotPtsY,sub_values,npts,peak_pos,avg_width])
                else:
                    val_pair.append([None,None,None,None,None,None])
                return val_pair
            else:
                dom_values = distUse
                sub_values = []
                # logger.debug(f"Primary pattern values - {list(dom_values)}")
                # logger.debug(f"Secondary pattern values - {list(sub_values)}")

                if len(dom_values) >= 5:
                    distUse = remove_outliers(dom_values)
                
                    plotPtsX = peak_pos[(dist > 2) & (dist * 10 < max_dim)]
                    plotPtsY = np.repeat(round((scanSize/2) + (((scanSize * i) + (scanSize * i + scanSize)) / 2)),plotPtsX.size)
                    npts = len(plotPtsY)
                    return [[plotPtsX,plotPtsY,distUse,npts,peak_pos,avg_width]]
                else:
                    return [[None,None,None,None,None,None]]
        else:
            return [[None,None,None,None,None,None]]
    else:
        return [[None,None,None,None,None,None]]

    #                 return [[plotPtsX,plotPtsY,distUse,npts,peak_pos,avg_width], [None,None,None,None,None,None]]
    #             else:
    #                 return [[None,None,None,None,None,None], [None,None,None,None,None,None]]
    #     else:
    #         return [[None,None,None,None,None,None], [None,None,None,None,None,None]]
    # else:
    #     return [[None,None,None,None,None,None], [None,None,None,None,None,None]]
    # Convert binary image to RGB
    # chunkAdd_rgb = np.stack((chunkAdd*255,)*3, axis=-1).astype(np.uint8)
    # Draw a small circle for each centroid
    # for i in range(len(plotPtsX)):
    #     # Draw a circle around the centroid
    #     cv2.circle(chunkAdd_rgb, (int(plotPtsX[i]), 3), 2, (0, 0, 255), -1)
    # # Show the image
    # cv2.imshow('Centroids', chunkAdd_rgb)
    # cv2.waitKey(0)
    # return None,None,None,None,None,None

def locate_ticks_centroid(chunkAdd,scanSize, i):
    props = regionprops_table(label(chunkAdd), properties=('centroid',
                                            'orientation',
                                            'axis_major_length',
                                            'axis_minor_length'))
    props = pd.DataFrame(props)
    # Calculate the average width of the objects
    if np.any(props['axis_major_length']):
        widths = np.sqrt(props['axis_major_length'] * props['axis_minor_length'])
        avg_width = np.nanmean(widths)

        centoid = props['centroid-1']
        peak_pos = np.transpose(np.array(centoid))
        dst_matrix = peak_pos - peak_pos[:, None]
        dst_matrix = dst_matrix[~np.eye(dst_matrix.shape[0],dtype=bool)].reshape(dst_matrix.shape[0],-1)
        if (dst_matrix.shape[0] > 0) and (dst_matrix.shape[1] > 0):
            dist = np.min(np.abs(dst_matrix), axis=1)
            distUse = dist[dist > 2]

            if len(distUse) >= 5:
                distUse = remove_outliers(distUse)
            
                plotPtsX = peak_pos[dist > 2]
                plotPtsY = np.repeat(round((scanSize/2) + (((scanSize * i) + (scanSize * i + scanSize)) / 2)),plotPtsX.size)
                npts = len(plotPtsY)

                return [[plotPtsX,plotPtsY,distUse,npts,peak_pos,avg_width]]
            else:
                return [[None,None,None,None,None,None]]

        else:
            return [[None,None,None,None,None,None]]
    # Convert binary image to RGB
    # chunkAdd_rgb = np.stack((chunkAdd*255,)*3, axis=-1).astype(np.uint8)
    # Draw a small circle for each centroid
    # for i in range(len(plotPtsX)):
    #     # Draw a circle around the centroid
    #     cv2.circle(chunkAdd_rgb, (int(plotPtsX[i]), 3), 2, (0, 0, 255), -1)
    # # Show the image
    # cv2.imshow('Centroids', chunkAdd_rgb)
    # cv2.waitKey(0)
    return [[None,None,None,None,None,None]]


def remove_outliers(dist):
    '''threshold = 2
    z = np.abs(stats.zscore(dist))
    dist = dist[np.where(z < threshold)]
    threshold = 1
    z = np.abs(stats.zscore(dist))
    dist = dist[np.where(z < threshold)]
    threshold = 1
    z = np.abs(stats.zscore(dist))
    distUse = dist[np.where(z < threshold)]'''
    Q1 = np.percentile(dist, 25)
    Q3 = np.percentile(dist, 75)
    IQR = Q3 - Q1
    upper_bound = Q3 + IQR
    lower_bound = Q1 - IQR
    distUse = dist[(dist > lower_bound) & (dist < upper_bound)]
    return distUse

def locate_tick_peaks(chunk,scanSize,x):
    chunkAdd = [sum(x) for x in zip(*chunk)]
    if scanSize >= 12:
        peaks = find_peaks(chunkAdd,distance=6,height=6)
    elif ((scanSize >= 6)&(scanSize < 12)):
        peaks = find_peaks(chunkAdd,distance=4,height=4)
    else:
        peaks = find_peaks(chunkAdd,distance=3,height=3)
    peak_pos = x[peaks[0]]
    peak_pos = np.array(peak_pos)
    dst_matrix = peak_pos - peak_pos[:, None]
    dst_matrix = dst_matrix[~np.eye(dst_matrix.shape[0],dtype=bool)].reshape(dst_matrix.shape[0],-1)
    dist = np.min(np.abs(dst_matrix), axis=1)
    distUse = dist[dist > 2]

    distUse = remove_outliers(distUse)

    plotPtsX = peak_pos[dist > 2]
    plotPtsY = np.repeat(round(scanSize/2),plotPtsX.size)
    npts = len(plotPtsY)
    # print(x[peaks[0]])
    # print(peaks[1]['peak_heights'])
    # plt.plot(x,chunkAdd)
    # plt.plot(x[peaks[0]],peaks[1]['peak_heights'], "x")
    # plt.show()
    return plotPtsX,plotPtsY,distUse,npts

def skeletonize(img):
    # try:
    #     img = cv2.ximgproc.thinning(img)
    # except:
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     img = cv2.ximgproc.thinning(gray)
    try:
        return cv2.ximgproc.thinning(img)
    except AttributeError:
        warnings.warn("Skeletonization unavailable - cv2.ximgproc.thinning() not available with current cv2 package")
        return img

    '''skel = np.zeros(img.shape,np.uint8)
    size = np.size(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False

    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        # cv2.imshow("skel",skel)
        # cv2.waitKey(0)    
        zeros = size - cv2.countNonZero(img)
        if np.amax(skel) == np.amin(skel):
            done = True
            return img
        else:
            if zeros==size:
                done = True
                return skel'''
        

    

def minimum_pairwise_distance(plotPtsX, plotPtsY):
    points = np.column_stack((plotPtsX, plotPtsY))
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    min_distances = np.min(distances, axis=1)
    min_pairwise_distance = gmean(min_distances)
    return min_pairwise_distance


def standard_deviation_of_pairwise_distance(plotPtsX, plotPtsY):
    x = np.asarray(plotPtsX)
    y = np.asarray(plotPtsY)
    valid_indices = np.where(np.logical_and(np.logical_not(np.isnan(x)), np.logical_not(np.isnan(y))))[0]
    x = x[valid_indices]
    y = y[valid_indices]
    arrmean = np.nanmean(x)
    x = np.asanyarray(x - arrmean)
    return np.sqrt(np.nanmean(x**2))

def sanity_check_scanlines(min_pairwise_distance, min_pairwise_distance_odd, min_pairwise_distance_even, min_pairwise_distance_third):
    if min_pairwise_distance_odd < min_pairwise_distance / 2 or min_pairwise_distance_odd > min_pairwise_distance * 2:
        return False
    if min_pairwise_distance_even < min_pairwise_distance / 2 or min_pairwise_distance_even > min_pairwise_distance * 2:
        return False
    if min_pairwise_distance_third < min_pairwise_distance / 3 or min_pairwise_distance_third > min_pairwise_distance * 3:
        return False
    return True

def verify_cm_vs_mm(scanlineData):
    try:
        max_dim = max(scanlineData.get("imgChunk").shape)
        x = scanlineData.get("peak_pos")
        n = scanlineData.get("nPeaks")
        distUse = scanlineData.get("gmean")

        # How many units fir into the space the points came from
        # if span_x = 150, then 150 units fit into the space
        span_x = (max(x) - min(x)) / distUse
        # units * pixel length. coverage_if_mm will be less than max_dim IF it's mm
        coverage_if_mm = distUse * span_x
        # units * pixel length. coverage_if_mm will be less than max_dim IF it's mm
        coverage_if_cm = distUse * span_x * 10

        # print(span_x)
        if (coverage_if_mm < max_dim) and (coverage_if_cm > max_dim):
            if span_x <= 30:
                return 'cm'
            else:
                return 'mm'
        else:
            return 'cm'
    except:
        return []

def select_best_or_average_factor(BlockCandidate, valid_factors):
    if not valid_factors:
        BlockCandidate.conversion_factor = 0
        BlockCandidate.conversion_location = 'fail'
        BlockCandidate.conversion_factor_pass = False
        BlockCandidate.conversion_location_options = ''
        return
    
    # Determine the factor with the highest count
    # Assuming factors_count and other related data are accessible here
    # This part of the logic needs adjustment based on your actual data
    n_max = max(valid_factors, key=valid_factors.get)
    best_factor = valid_factors[n_max]
    
    # Use a simple list for comparison instead of trying to access non-existent dictionary keys
    values_list = list(valid_factors.values())
    n_greater = len([value for value in values_list if value > best_factor])
    n_lesser = len([value for value in values_list if value < best_factor])

    # If the factor with the highest number of measurements is the outlier, take the average of all factors
    if (n_greater == 0 or n_lesser == 0) and len(values_list) > 1:
        total = sum(values_list)
        BlockCandidate.conversion_factor = total / len(values_list)
        BlockCandidate.conversion_location = 'average'
        BlockCandidate.conversion_factor_pass = True
    else:
        BlockCandidate.conversion_factor = best_factor
        BlockCandidate.conversion_location = n_max
        BlockCandidate.conversion_factor_pass = True  # Assuming all passed factors are valid

    # Generate location options for output or further use
    BlockCandidate.conversion_location_options = ', '.join(valid_factors.keys())
    return BlockCandidate


def calculate_block_conversion_factor(BlockCandidate, nBlockCheck, predicted_conversion_factor_cm, use_CF_predictor):
    factors = {'bigCM':0,'smallCM':0,'halfCM':0,'mm':0}
    factors_count = {'bigCM':0,'smallCM':0,'halfCM':0,'mm':0}
    passFilter = {'bigCM':False,'smallCM':False,'halfCM':False,'mm':False}
    factors_fallback = {'bigCM':0,'smallCM':0,'halfCM':0,'mm':0}

    tolerance = 0.1 * predicted_conversion_factor_cm  # 10% tolerance

    for i in range(0,nBlockCheck):
        if BlockCandidate.use_points[i]:
            X = BlockCandidate.x_points[i].values
            n_measurements = X.size
            axis_major_length = np.nanmean(BlockCandidate.axis_major_length[i].values)
            axis_minor_length = np.nanmean(BlockCandidate.axis_minor_length[i].values)
            dst_matrix = X - X[:, None]
            dst_matrix = dst_matrix[~np.eye(dst_matrix.shape[0],dtype=bool)].reshape(dst_matrix.shape[0],-1)
            dist = np.min(np.abs(dst_matrix), axis=1)
            distUse = dist[dist > 1]

            # Convert everything to CM along the way
            # 'if factors['bigCM'] == 0:' is there to make sure that there are no carry-over values if there were 
            # 2 instances of 'bigCM' coming from determineBlockBlobType()
            if distUse.size > 0:
                distUse_mean = np.nanmean(distUse)
                if BlockCandidate.point_types[i] == 'bigCM':
                    if ((distUse_mean >= 0.8*axis_major_length) & (distUse_mean <= 1.2*axis_major_length)):
                        if factors['bigCM'] == 0:
                            factors['bigCM'] = distUse_mean
                            factors_count['bigCM'] = n_measurements
                            passFilter['bigCM'] = True
                        else:
                            break
                    else: 
                        factors_fallback['bigCM'] = distUse_mean

                elif BlockCandidate.point_types[i] == 'smallCM':
                    if ((distUse_mean >= 0.8*axis_major_length*2) & (distUse_mean <= 1.2*axis_major_length*2)):
                        if factors['smallCM'] ==0:
                            factors['smallCM'] = distUse_mean/2
                            factors_count['smallCM'] = n_measurements
                            passFilter['bigCM'] = True
                        else:
                            break
                    else: 
                        factors_fallback['smallCM'] = distUse_mean/2

                elif BlockCandidate.point_types[i] == 'halfCM':
                    if ((distUse_mean >= 0.8*axis_major_length) & (distUse_mean <= 1.2*axis_major_length)):
                        if factors['halfCM'] ==0:
                            factors['halfCM'] = distUse_mean*2
                            factors_count['halfCM'] = n_measurements
                            passFilter['bigCM'] = True
                        else:
                            break
                    else: 
                        factors_fallback['halfCM'] = distUse_mean*2

                elif BlockCandidate.point_types[i] == 'mm':
                    if ((distUse_mean >= 0.1*axis_minor_length) & (distUse_mean <= 1.1*axis_minor_length)):
                        if factors['mm'] ==0:
                            factors['mm'] = distUse_mean*10
                            factors_count['mm'] = n_measurements
                            passFilter['bigCM'] = True
                        else:
                            break
                    else: 
                        factors['mm'] = 0
                        factors_fallback['mm'] = distUse_mean*10

    # Evaluate which factors are valid based on the predicted conversion factor
    if use_CF_predictor:
        valid_factors = {key: val for key, val in factors.items() if val != 0 and abs(val - predicted_conversion_factor_cm) <= tolerance}
        location_options = ', '.join(valid_factors.keys())
    else:
        valid_factors = factors
        location_options = ', '.join(valid_factors.keys())

    if not valid_factors:
        # Fallback or error handling if no valid factors found
        BlockCandidate.conversion_factor = 0
        BlockCandidate.conversion_location = 'fail'
        BlockCandidate.conversion_factor_pass = False
        BlockCandidate.conversion_location_options = ''
    else:
        # Select the best factor or calculate average if applicable
        BlockCandidate = select_best_or_average_factor(BlockCandidate, valid_factors)
    return BlockCandidate


    # # Remove empty keys from n dict
    # n_max = max(factors_count, key=factors_count.get)
    # best_factor = factors[n_max]
    # n_greater = len([f for f, factor in factors.items() if factor > best_factor])
    # n_lesser = len([f for f, factor in factors.items() if factor < best_factor])
    # location_options = ', '.join([f for f, factor in factors.items() if factor > 0])

    # # If the factor with the higest number of measurements is the outlier, take the average of all factors
    # if ((n_greater == 0) | (n_lesser == 0)):
    #     # Number of keys that = 0
    #     nZero = sum(x == 0 for x in factors.values())
    #     dividend = len(factors) - nZero
    #     # If no blocks pass the filter, return the nMax with a warning 
    #     if dividend == 0:
    #         best_factor_fallback = factors_fallback[n_max]
    #         n_greater = len([f for f, factor in factors_fallback.items() if factor > best_factor_fallback])
    #         n_lesser = len([f for f, factor in factors_fallback.items() if factor < best_factor_fallback])
    #         location_options = ', '.join([f for f, factor in factors_fallback.items() if factor > 0])
    #         if best_factor_fallback > 0:
    #             BlockCandidate.conversion_factor = best_factor_fallback
    #             BlockCandidate.conversion_location = 'fallback'
    #             BlockCandidate.conversion_factor_pass = passFilter[n_max]
    #         # Else complete fail
    #         else: 
    #             BlockCandidate.conversion_factor = 0
    #             BlockCandidate.conversion_location = 'fail'
    #             BlockCandidate.conversion_factor_pass = False
    #     else:
    #         res = sum(factors.values()) / dividend
    #         BlockCandidate.conversion_factor = res
    #         BlockCandidate.conversion_location = 'average'
    #         BlockCandidate.conversion_factor_pass = True
    # # Otherwise use the factor with the most measuements 
    # else:
    #     BlockCandidate.conversion_factor = best_factor
    #     BlockCandidate.conversion_location = n_max
    #     BlockCandidate.conversion_factor_pass = passFilter[n_max]
    # BlockCandidate.conversion_location_options = location_options
    # return BlockCandidate

def sort_blobs_by_size(logger, show_all_logs, RulerCFG, Ruler, predicted_conversion_factor_cm, use_CF_predictor, isStraighten):
    nBlockCheck = 4
    success = True
    tryErode = False

    try: # This gets set the first call, then just leave it alone
        if isStraighten == False:
            # img_best = Ruler.img_best # was causseing issues
            img_best = cv2.cvtColor(Ruler.img_copy, cv2.COLOR_GRAY2RGB) # THIS IS USED FOR THE OVERLAY, NEEDS TO BE RGB, BUT BINARY
        else:
            img_best = cv2.cvtColor(Ruler.img_copy, cv2.COLOR_GRAY2RGB) # THIS IS USED FOR THE OVERLAY, NEEDS TO BE RGB, BUT BINARY
    except:
        img_best = Ruler.img_copy

    # cv2.imshow('img_best', img_best)
    # cv2.waitKey(0)
    # cv2.imshow('Ruler.img_bi', Ruler.img_bi)
    # cv2.waitKey(0)

    BlockCandidate = Block(img_bi=Ruler.img_bi,img_bi_overlay=img_best, success_sort=success)
    try: # Start with 4, reduce by one if fail
        # try: # Normal
        BlockCandidate = remove_small_and_biggest_blobs(BlockCandidate,tryErode)
        for i in range(0,nBlockCheck):
            BlockCandidate = get_biggest_blob(BlockCandidate)
        # except: # Extreme thresholding for whiter rulers
        #     # BlockCandidate.whiter_thresh()
        #     BlockCandidate.img_result = BlockCandidate.img_bi_copy
        #     BlockCandidate = removeSmallAndBiggestBlobs(BlockCandidate,tryErode)
        #     for i in range(0,nBlockCheck):
        #         BlockCandidate = getBiggestBlob(BlockCandidate)
    except:
        try:
            tryErode = True
            del BlockCandidate
            nBlockCheck = 3
            BlockCandidate = Block(img_bi=Ruler.img_bi,img_bi_overlay=img_best, success_sort=success)
            BlockCandidate = remove_small_and_biggest_blobs(BlockCandidate,tryErode)
            for i in range(0,nBlockCheck):
                BlockCandidate = get_biggest_blob(BlockCandidate)
        except:
            success = False
            BlockCandidate = Block(img_bi=Ruler.img_bi,img_bi_overlay=img_best, success_sort=success)
            BlockCandidate.conversion_factor = 0
            BlockCandidate.conversion_location = 'unidentifiable'
            BlockCandidate.conversion_location_options = 'unidentifiable'
            BlockCandidate.success_sort = success
            BlockCandidate.img_bi_overlay = Ruler.img_bi

    # cv2.imshow('img_best2', img_best)
    # cv2.waitKey(0)
    # cv2.imshow('Ruler.img_bi2', Ruler.img_bi)
    # cv2.waitKey(0)

    if success:
        # imgPlot = plt.imshow(img_result)
        for i in range(0,nBlockCheck):
            BlockCandidate = determine_block_blob_type(logger, show_all_logs, RulerCFG,BlockCandidate,i)#BlockCandidate.largest_blobs[0],BlockCandidate.img_bi_overlay)
        if isStraighten == False:
            Ruler.img_block_overlay = BlockCandidate.img_bi_overlay

        BlockCandidate = calculate_block_conversion_factor(BlockCandidate, nBlockCheck, predicted_conversion_factor_cm, use_CF_predictor)  
    BlockCandidate.success_sort = success
    return Ruler, BlockCandidate


def convert_blocks(logger, show_all_logs, RulerCFG,Ruler, predicted_conversion_factor_cm, use_CF_predictor, colorOption,img_fname, Dirs, is_redo):
    # if is_redo:
    #     Ruler.img_bi = Ruler.img_bi_backup
    img_bi = copy.deepcopy(Ruler.img_bi)
    img_bi_sweep = copy.deepcopy(Ruler.img_bi_sweep)

    if colorOption == 'invert':
        img_bi = cv2.bitwise_not(img_bi)
        img_bi_sweep = cv2.bitwise_not(img_bi_sweep)

    # cv2.imshow('bi', img_bi)
    # cv2.imshow('img_bi_sweep', img_bi_sweep)
    # cv2.waitKey(0)

    # happens in Block() class
    # img_bi[img_bi<=200] = 0
    # img_bi[img_bi>200] = 1
    # img_bi_sweep[img_bi_sweep<=200] = 0
    # img_bi_sweep[img_bi_sweep>200] = 1

    max_dim = max(img_bi.shape)
    min_dim = min(img_bi.shape)

    size_ratio = np.divide(min_dim, max_dim)

    # img_pad = stack_2_imgs_bi(img_pad_sweep, img_pad_bi)
    # img_pad = stack_2_imgs_bi(img_pad, img_pad_skel)
    # Ruler.img_bi_pad = copy.deepcopy(img_pad)

    

    Ruler.img_bi = copy.deepcopy(img_bi) # use ML
    Ruler.img_best = copy.deepcopy(img_bi) # use ML
    Ruler.img_copy = copy.deepcopy(img_bi) # use ML
    # Ruler.img_bi = copy.deepcopy(img_bi_sweep) # use sweep
    # Ruler.img_best = copy.deepcopy(img_bi_sweep) # use ML
    # Ruler.img_copy = copy.deepcopy(img_bi_sweep) # use ML


    
    # Straighten the image here using the BlockCandidate.remaining_blobs[0].values
    Ruler,BlockCandidate = sort_blobs_by_size(logger, show_all_logs, RulerCFG, Ruler, predicted_conversion_factor_cm, use_CF_predictor, isStraighten=True) 
    if BlockCandidate.success_sort:
        useRegulerBinary = True
        # Ruler = straighten_img(logger, RulerCFG, Ruler, useRegulerBinary, BlockCandidate.remaining_blobs[0], Dirs, False, False, True)
        del BlockCandidate
        Ruler,BlockCandidate = sort_blobs_by_size(logger, show_all_logs, RulerCFG,Ruler, predicted_conversion_factor_cm, use_CF_predictor, isStraighten=False) 

    
        if BlockCandidate.success_sort: # if this is false, then no marks could be ID'd, will print just the existing Ruler.img_total_overlay
            if BlockCandidate.conversion_location != 'fail':
                BlockCandidate = add_unit_marker_block(BlockCandidate,1)
                BlockCandidate = add_unit_marker_block(BlockCandidate,10)

                add_unit_marker_block(BlockCandidate, 1, is_pred1=True, is_pred2=False, pred_cm=predicted_conversion_factor_cm)
                add_unit_marker_block(BlockCandidate, 1, is_pred1=False, is_pred2=True, pred_cm=predicted_conversion_factor_cm)
                add_unit_marker_block(BlockCandidate, 1, is_pred1=True, is_pred2=False, pred_cm=predicted_conversion_factor_cm)

    message = ''.join(["Angle (deg): ", str(round(Ruler.avg_angle,2))])
    if show_all_logs:
        logger.debug(message)
    # Print_Verbose(RulerCFG.cfg,1,message).cyan()

    # cv2.imshow('img_bi_overlay', BlockCandidate.img_bi_overlay)
    # cv2.waitKey(0)
    # cv2.imshow('img_bi', BlockCandidate.img_bi)
    # cv2.waitKey(0)
    # cv2.imshow('img_bi_copy', BlockCandidate.img_bi_copy)
    # cv2.waitKey(0)

    BlockCandidate.img_bi_overlay = create_overlay_bg_120(logger, show_all_logs, RulerCFG,BlockCandidate.img_bi_overlay)
    if BlockCandidate.conversion_location in ['average','fallback']:
        addText = 'Used: '+BlockCandidate.conversion_location_options+' Factor 1cm: '+str(round(BlockCandidate.conversion_factor,2))
        Ruler.summary_message.append([addText])
    elif BlockCandidate.conversion_location == 'fail':
        addText = 'Used: '+'FAILED'+' Factor 1cm: '+str(round(BlockCandidate.conversion_factor,2))
        Ruler.summary_message.append([addText])
    elif BlockCandidate.conversion_location == 'unidentifiable':
        addText = 'UNIDENTIFIABLE'+' Factor 1cm: '+str(round(BlockCandidate.conversion_factor))
        Ruler.summary_message.append([addText])
    else:
        addText = 'Used: '+BlockCandidate.conversion_location+' Factor 1cm: '+ str(round(BlockCandidate.conversion_factor,2))
        Ruler.summary_message.append([addText])

    BlockCandidate.img_bi_overlay = add_text_to_img_list(Ruler.summary_message,BlockCandidate.img_bi_overlay)#+str(round(scanlineData['gmean'],2)),Ruler.img_block_overlay)
    
    try:
        Ruler.img_total_overlay = stack_2_imgs(Ruler.img_total_overlay,BlockCandidate.img_bi_overlay)
    except:
        Ruler.img_total_overlay = stack_2_imgs(Ruler.img_type_overlay,BlockCandidate.img_bi_overlay)
    Ruler.img_block_overlay = BlockCandidate.img_bi_overlay

    if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_validation_summary']:
        cv2.imwrite(os.path.join(RulerCFG.dir_ruler_validation,'.'.join([img_fname, 'jpg'])),Ruler.img_total_overlay)
        cv2.imwrite(os.path.join(RulerCFG.dir_ruler_validation_summary,'.'.join([img_fname, 'jpg'])),BlockCandidate.img_bi_overlay)
    
    summary_image = copy.deepcopy(BlockCandidate.img_bi_overlay)

    return Ruler, BlockCandidate, summary_image




def add_unit_marker_block(BlockCandidate, multiple, is_pred1=False, is_pred2=False, pred_cm=None):
    COLOR = {'10cm':[0,255,0],
             'cm':[255,0,255],
             'is_pred1':[0,0,0],
             'is_pred2':[255,255,255],
             }
    name = 'cm' if multiple == 1 else '10cm'
    offset = 4 if multiple == 1 else 14
    CF = BlockCandidate.conversion_factor

    if is_pred1:
        offset = 24
        name = 'is_pred1'
        CF = pred_cm
    elif is_pred2:
        offset = 29
        name = 'is_pred2'
        CF = pred_cm
    h, w, _ = BlockCandidate.img_bi_overlay.shape

    if BlockCandidate.conversion_location in ['average','fallback']:
        X = int(round(w/40))
        Y = int(round(h/10))
    else:
        ind = BlockCandidate.point_types.index(BlockCandidate.conversion_location)
        X = int(round(min(BlockCandidate.x_points[ind].values)))
        Y = int(round(np.nanmean(BlockCandidate.y_points[ind].values)))

    start = X
    end = int(round(start+(CF*multiple))) + 1
    if end >= w:
        X = int(round(w/40))
        Y = int(round(h/10))
        start = X
        end = int(round(start+(CF*multiple))) + 1

    plot_points = []
    for j in range(start, end):
        try:
            img_bi_overlay = BlockCandidate.img_bi_overlay
            img_bi_overlay[offset+Y-2:offset+Y+3, j, :] = 0
            img_bi_overlay[offset+Y-1:offset+Y+2, j, :] = COLOR[name]
            plot_points.append([j, offset+Y])
        except:
            continue

    BlockCandidate.img_bi_overlay = img_bi_overlay
    if multiple == 1:
        BlockCandidate.plot_points_1cm = plot_points
    else:
        BlockCandidate.plot_points_10cm = plot_points
    return BlockCandidate



def get_biggest_blob(BlockCandidate):
    img_result = BlockCandidate.img_result
    # cv2.imshow('THIS img',BlockCandidate.img_result)
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(np.uint8(img_result))
    sizes = stats[:, -1]
    sizes = sizes[1:]
    maxBlobSize = max(sizes)
    largestBlobs = np.zeros((img_result.shape))
    remainingBlobs = np.zeros((img_result.shape))
    nb_blobs -= 1
    for blob in range(nb_blobs):
        if (sizes[blob] <= 1.1*maxBlobSize) & ((sizes[blob] >= 0.9*maxBlobSize)):
            # see description of im_with_separated_blobs above
            largestBlobs[im_with_separated_blobs == blob + 1] = 255
        else:
            remainingBlobs[im_with_separated_blobs == blob + 1] = 255
    BlockCandidate.largest_blobs.append(largestBlobs)
    BlockCandidate.remaining_blobs.append(remainingBlobs)
    BlockCandidate.img_result = remainingBlobs
    return BlockCandidate
    
def remove_small_and_biggest_blobs(BlockCandidate,tryErode):
    min_size = 50
    img_bi = BlockCandidate.img_bi
    # cv2.imshow('iimg',img_bi)
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(img_bi, cv2.MORPH_OPEN, kernel)
    if tryErode:
        opening = cv2.bitwise_not(opening)
        opening = cv2.erode(opening,kernel,iterations = 1)
        opening = cv2.dilate(opening,kernel,iterations = 1)
        min_size = 25
        BlockCandidate.img_bi = opening
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(opening)
    sizes = stats[:, -1]
    sizes = sizes[1:]
    maxBlobSize = max(sizes)
    nb_blobs -= 1
    img_result = np.zeros((img_bi.shape))
    # for every component in the image, keep it only if it's above min_size
    for blob in range(nb_blobs):
        if sizes[blob] == maxBlobSize:
            img_result[im_with_separated_blobs == blob + 1] = 0
        elif sizes[blob] >= min_size:
            # see description of im_with_separated_blobs above
            img_result[im_with_separated_blobs == blob + 1] = 255
    BlockCandidate.img_result = img_result
    return BlockCandidate

def add_centroid_to_block_img(imgBG, centroidX, centroidY, ptType):
    COLOR = {'bigCM': [0, 255, 0], 'smallCM': [255, 255, 0], 'halfCM': [0, 127, 255], 'mm': [255, 0, 127]}
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for x in range(0, centroidX.size):
                X = int(round(centroidX.values[x]))
                Y = int(round(centroidY.values[x]))
                if (int(Y+i) < imgBG.shape[0]) and (int(X+j) < imgBG.shape[1]) and (int(Y+i) >= 0) and (int(X+j) >= 0):
                    if (abs(i) == 3) | (abs(j) == 3):
                        imgBG[int(Y+i), int(X+j), 0] = 0
                        imgBG[int(Y+i), int(X+j), 1] = 0
                        imgBG[int(Y+i), int(X+j), 2] = 0
                    else:
                        imgBG[int(Y+i), int(X+j), 0] = COLOR[ptType][0]
                        imgBG[int(Y+i), int(X+j), 1] = COLOR[ptType][1]
                        imgBG[int(Y+i), int(X+j), 2] = COLOR[ptType][2]
                        points.append([j + X, Y + i])
    return imgBG, points

def determine_block_blob_type(logger, show_all_logs, RulerCFG,BlockCandidate,ind):
    largestBlobs = BlockCandidate.largest_blobs[ind]
    img_bi_overlay = BlockCandidate.img_bi_overlay
    # img_bi_overlay = np.stack((img_bi,)*3, axis=-1)
    RATIOS = {'bigCM':1.75,'smallCM':4.5,'halfCM':2.2,'mm':6.8}
    use_points = False
    point_types = 'NA'
    points = []

    props = regionprops_table(label(largestBlobs), properties=('centroid','axis_major_length','axis_minor_length'))
    props = pd.DataFrame(props)
    centoidY = props['centroid-0']
    centoidX = props['centroid-1']
    axis_major_length = props['axis_major_length']
    axis_minor_length = props['axis_minor_length']
    ratio = axis_major_length/axis_minor_length
    if ((ratio.size > 1) & (ratio.size <= 10)):
        ratioM = np.nanmean(ratio)
        if ((ratioM >= (0.9*RATIOS['bigCM'])) & (ratioM <= (1.1*RATIOS['bigCM']))):
            use_points = True
            point_types = 'bigCM'
            img_bi_overlay, points = add_centroid_to_block_img(img_bi_overlay,centoidX,centoidY,point_types)
        elif ((ratioM >= (0.75*RATIOS['smallCM'])) & (ratioM <= (1.25*RATIOS['smallCM']))):
            use_points = True
            point_types = 'smallCM'
            img_bi_overlay, points = add_centroid_to_block_img(img_bi_overlay,centoidX,centoidY,point_types)
        elif ((ratioM >= (0.9*RATIOS['halfCM'])) & (ratioM <= (1.1*RATIOS['halfCM']))):
            use_points = True
            point_types = 'halfCM'
            img_bi_overlay, points = add_centroid_to_block_img(img_bi_overlay,centoidX,centoidY,point_types)
        elif ((ratioM >= (0.9*RATIOS['mm'])) & (ratioM <= (1.1*RATIOS['mm']))):
            use_points = True
            point_types = 'mm'
            img_bi_overlay, points = add_centroid_to_block_img(img_bi_overlay,centoidX,centoidY,point_types)
        message = ''.join(["ratio: ", str(round(ratioM,3)), " use_points: ", str(use_points), " point_types: ", str(point_types)])
        # Print_Verbose(RulerCFG.cfg,2,message).plain()
        if show_all_logs:
            logger.debug(message)
    # plt.imshow(img_bi_overlay)
    BlockCandidate.img_bi_overlay = img_bi_overlay
    BlockCandidate.use_points.append(use_points)
    BlockCandidate.plot_points.append(points)
    BlockCandidate.point_types.append(point_types)
    BlockCandidate.x_points.append(centoidX)
    BlockCandidate.y_points.append(centoidY)
    BlockCandidate.axis_major_length.append(axis_major_length)
    BlockCandidate.axis_minor_length.append(axis_minor_length)
    return BlockCandidate




def find_minimal_change_in_binarization_TESTING(img_gray):
    result_list = []

    # fig, axs = plt.subplots(5, 5, figsize=(20, 20))
    # axs = axs.ravel()

    for idx, i in enumerate(range(0, 255, 10)):
        threshold_value = i
        img_bi = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)[1]
        result = cv2.countNonZero(img_bi)
        result_list.append((threshold_value, result))
        
        # axs[idx-1].imshow(img_bi, cmap='gray')
        # axs[idx-1].set_title(f"Threshold: {threshold_value}")

    # x = [i[0] for i in result_list]
    # y = [i[1] for i in result_list]

    # x = [i[0] for i in result_list]
    y = [i[1] for i in result_list]

    # Calculate the first derivative
    dy = np.diff(y)

    # Calculate the second derivative
    # ddy = np.diff(dy)
    # min_index = np.argmin(dy)
    # min_index = np.argmin(ddy)
    # Find the index of the minimum value of the first derivative
    diffs = [abs(dy[i+5]-dy[i]) for i in range(len(dy)-5)]
    min_index = diffs.index(min(diffs))
    best_threshold = result_list[min_index][0]

    # diffs = [abs(y[i+5]-y[i]) for i in range(len(y)-5)]
    # min_index1 = diffs.index(min(diffs))
    # min_index = diffs.index(min([i for i in diffs if i >= 0.01*max(diffs)]))
    # best_threshold = result_list[min_index][0]
    # Turn this and the commented lines above for testing
    '''
    plt.tight_layout()
    plt.show()
    fig.savefig('bi_panel.pdf')
    plt.close()

    x = [i[0] for i in result_list]
    y = [i[1] for i in result_list]

    diffs = [abs(y[i+5]-y[i]) for i in range(len(y)-5)]
    min_index = diffs.index(min(diffs))


    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel("Threshold value")
    plt.ylabel("Result")
    plt.title("Result vs Threshold Value")
    fig.savefig("bi_plot.pdf")
    plt.close()
    dy = np.gradient(y)
    d2y = np.gradient(dy)

    fig = plt.figure()
    plt.plot(x, dy, label='Derivative')
    plt.plot(x, d2y, label='Second Derivative')
    plt.xlabel("Threshold value")
    plt.ylabel("Result")
    plt.title("Result vs Threshold Value")
    plt.legend()
    fig.savefig("bi_plot_derivative.pdf")
    plt.close()

    # find the median point of result_list where the change between results is the least
    # median_index = 0
    # min_diff = float('inf')
    # diff_list = []
    # for i in range(1, len(result_list) - 1):
    #     diff = abs(result_list[i + 1][1] - result_list[i - 1][1])
    #     diff_list.append(diff)
    #     if diff < min_diff:
    #         median_index = i
    #         min_diff = diff
    '''   
    img_bi = cv2.threshold(img_gray, best_threshold, 255, cv2.THRESH_BINARY)[1]
    return img_bi






@dataclass
class Print_Verbose():
    cfg: str = ''
    indent_level: int = 0
    message: str = ''

    def __init__(self, cfg, indent_level, message) -> None:
        self.cfg = cfg
        self.indent_level = indent_level
        self.message = message

    def bold(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['verbose']:
            print(f"{bcolors.BOLD}{white_space}{self.message}{bcolors.ENDC}")

    def green(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['verbose']:
            print(f"{bcolors.OKGREEN}{white_space}{self.message}{bcolors.ENDC}")

    def cyan(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['verbose']:
            print(f"{bcolors.OKCYAN}{white_space}{self.message}{bcolors.ENDC}")

    def blue(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['verbose']:
            print(f"{bcolors.OKBLUE}{white_space}{self.message}{bcolors.ENDC}")

    def warning(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['verbose']:
            print(f"{bcolors.WARNING}{white_space}{self.message}{bcolors.ENDC}")

    def plain(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['verbose']:
            print(f"{white_space}{self.message}")

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    CEND      = '\33[0m'
    CBOLD     = '\33[1m'
    CITALIC   = '\33[3m'
    CURL      = '\33[4m'
    CBLINK    = '\33[5m'
    CBLINK2   = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK  = '\33[30m'
    CRED    = '\33[31m'
    CGREEN  = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE   = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE  = '\33[36m'
    CWHITE  = '\33[37m'

    CBLACKBG  = '\33[40m'
    CREDBG    = '\33[41m'
    CGREENBG  = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG   = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG  = '\33[46m'
    CWHITEBG  = '\33[47m'

    CGREY    = '\33[90m'
    CRED2    = '\33[91m'
    CGREEN2  = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2   = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2  = '\33[96m'
    CWHITE2  = '\33[97m'

    CGREYBG    = '\33[100m'
    CREDBG2    = '\33[101m'
    CGREENBG2  = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2   = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2  = '\33[106m'
    CWHITEBG2  = '\33[107m'
    CBLUEBG3   = '\33[112m'