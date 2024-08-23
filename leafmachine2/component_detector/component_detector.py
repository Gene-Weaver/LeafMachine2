import os, sys, inspect, json, shutil, cv2, time, glob, sqlite3 #imagesize
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from tqdm import tqdm
from time import perf_counter
import concurrent.futures
from threading import Thread, Lock
from queue import Queue
from collections import defaultdict
import multiprocessing
import torch
from sqlite3 import Error
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio, aiofiles
import numpy as np

currentdir = os.path.dirname(inspect.getfile(inspect.currentframe()))
parentdir = os.path.dirname(currentdir)
sys.path.append(currentdir)
sys.path.append(parentdir)
from detect import run
from landmark_processing import LeafSkeleton
from armature_processing import ArmatureSkeleton

def detect_plant_components(cfg, time_report, logger, dir_home, Project, Dirs):
    t1_start = perf_counter()
    n_images = len(os.listdir(Project.dir_images))
    logger.name = 'Locating Plant Components'
    logger.info(f"Detecting plant components in {n_images} images")

    try:
        dir_exisiting_labels = cfg['leafmachine']['project']['use_existing_plant_component_detections']
    except:
        dir_exisiting_labels = None
    if cfg['leafmachine']['project']['num_workers'] is None:
        num_workers = 1
    else:
        num_workers = int(cfg['leafmachine']['project']['num_workers'])

    # Weights folder base
    dir_weights = os.path.join(dir_home, 'leafmachine2', 'component_detector','runs','train')
    
    # Detection threshold
    threshold = cfg['leafmachine']['plant_component_detector']['minimum_confidence_threshold']

    detector_version = cfg['leafmachine']['plant_component_detector']['detector_version']
    detector_iteration = cfg['leafmachine']['plant_component_detector']['detector_iteration']
    detector_weights = cfg['leafmachine']['plant_component_detector']['detector_weights']
    weights =  os.path.join(dir_weights,'Plant_Detector',detector_version,detector_iteration,'weights',detector_weights)

    do_save_prediction_overlay_images = not cfg['leafmachine']['plant_component_detector']['do_save_prediction_overlay_images']
    ignore_objects = cfg['leafmachine']['plant_component_detector']['ignore_objects_for_overlay']
    ignore_objects = ignore_objects or []

    if dir_exisiting_labels != None:
        logger.info("Loading existing plant labels")
        fetch_labels(dir_exisiting_labels, os.path.join(Dirs.path_plant_components, 'labels'))
        # if n_images <= 4000:
            # logger.debug("Single-threaded create_dictionary_from_txt() n_images <= 4000")
        
        ### CLASSIC
        # A = create_dictionary_from_txt(logger, dir_exisiting_labels, 'Detections_Plant_Components', Project)
        ### SQL
        A = create_dictionary_from_txt_sql(logger, os.path.join(Dirs.path_plant_components, 'labels'), 'Detections_Plant_Components', Project, 'annotations_plant', 'dimensions_plant')

        # else:
            # logger.debug(f"Multi-threaded with ({str(cfg['leafmachine']['project']['num_workers'])}) threads create_dictionary_from_txt() n_images > 4000")
            # A = create_dictionary_from_txt_parallel(logger, cfg, dir_exisiting_labels, 'Detections_Plant_Components', Project)

    else:
        logger.info("Running YOLOv5 to generate plant labels")
        # run(weights = weights,
        #     source = Project.dir_images,
        #     project = Dirs.path_plant_components,
        #     name = Dirs.run_name,
        #     imgsz = (1280, 1280),
        #     nosave = do_save_prediction_overlay_images,
        #     anno_type = 'Plant_Detector',
        #     conf_thres = threshold, 
        #     ignore_objects_for_overlay = ignore_objects,
        #     mode = 'LM2',
        #     LOGGER=logger,)
        source = Project.dir_images
        project = Dirs.path_plant_components
        name = Dirs.run_name
        imgsz = (1280, 1280)
        nosave = do_save_prediction_overlay_images
        anno_type = 'Plant_Detector'
        conf_thres = threshold
        ignore_objects_for_overlay = ignore_objects
        mode = 'LM2'
        LOGGER = logger

        # with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        #     futures = [executor.submit(run_in_parallel, weights, source, project, name, imgsz, nosave, anno_type,
        #                             conf_thres, 10, ignore_objects_for_overlay, mode, LOGGER, i, num_workers) for i in
        #             range(num_workers)]
        #     for future in concurrent.futures.as_completed(futures):
        #         try:
        #             _ = future.result()
        #         except Exception as e:
        #             logger.error(f'Error in thread: {e}')
        #             continue
        distribute_workloads(weights, source, project, name, imgsz, nosave, anno_type, conf_thres, ignore_objects_for_overlay, mode, LOGGER, num_workers)


        t2_stop = perf_counter()
        logger.info(f"[Plant components detection elapsed time] {round(t2_stop - t1_start)} seconds")
        logger.info(f"Threads [{num_workers}]")

        # if len(Project.dir_images) <= 4000:
            # A = create_dictionary_from_txt_parallel(logger, cfg, os.path.join(Dirs.path_plant_components, 'labels'), 'Detections_Plant_Components', Project)
        
        ### CLASSIC
        # A = create_dictionary_from_txt(logger, os.path.join(Dirs.path_plant_components, 'labels'), 'Detections_Plant_Components', Project)
        
        ### SQL
        A = create_dictionary_from_txt_sql(logger, os.path.join(Dirs.path_plant_components, 'labels'), 'Detections_Plant_Components', Project, 'annotations_plant', 'dimensions_plant')

        # else:
            # logger.debug(f"Multi-threaded with ({str(cfg['leafmachine']['project']['num_workers'])}) threads create_dictionary_from_txt() len(Project.dir_images) > 4000")
            # A = create_dictionary_from_txt_parallel(logger, cfg, os.path.join(Dirs.path_plant_components, 'labels'), 'Detections_Plant_Components', Project)
    
    # dict_to_json(Project.project_data, Dirs.path_plant_components, 'Detections_Plant_Components.json')
    
    t1_stop = perf_counter()
    t_pcd = f"[Processing plant components elapsed time] {round(t1_stop - t1_start)} seconds ({round((t1_stop - t1_start)/60)} minutes)"
    logger.info(t_pcd)
    time_report['t_pcd'] = t_pcd
    torch.cuda.empty_cache()
    return Project, time_report
    

def detect_archival_components(cfg, time_report, logger, dir_home, Project, Dirs):
    t1_start = perf_counter()
    n_images = len(os.listdir(Project.dir_images))
    logger.name = 'Locating Archival Components'
    logger.info(f"Detecting archival components in {n_images} images")

    
    try:
        dir_exisiting_labels = cfg['leafmachine']['project']['use_existing_archival_component_detections']
    except:
        dir_exisiting_labels = None
    if cfg['leafmachine']['project']['num_workers'] is None:
        num_workers = 1
    else:
        num_workers = int(cfg['leafmachine']['project']['num_workers'])
    
    # Weights folder base
    dir_weights = os.path.join(dir_home, 'leafmachine2', 'component_detector','runs','train')
    
    # Detection threshold
    threshold = cfg['leafmachine']['archival_component_detector']['minimum_confidence_threshold']

    detector_version = cfg['leafmachine']['archival_component_detector']['detector_version']
    detector_iteration = cfg['leafmachine']['archival_component_detector']['detector_iteration']
    detector_weights = cfg['leafmachine']['archival_component_detector']['detector_weights']
    weights =  os.path.join(dir_weights,'Archival_Detector',detector_version,detector_iteration,'weights',detector_weights)

    do_save_prediction_overlay_images = not cfg['leafmachine']['archival_component_detector']['do_save_prediction_overlay_images']
    ignore_objects = cfg['leafmachine']['archival_component_detector']['ignore_objects_for_overlay']
    ignore_objects = ignore_objects or []


    if dir_exisiting_labels != None:
        logger.info("Loading existing archival labels")
        fetch_labels(dir_exisiting_labels, os.path.join(Dirs.path_archival_components, 'labels'))
        # if n_images <= 4000:
            # logger.debug("Single-threaded create_dictionary_from_txt() n_images <= 4000")
            # A = create_dictionary_from_txt_parallel(logger, cfg, dir_exisiting_labels, 'Detections_Archival_Components', Project)
        ### CLASSIC
        # A = create_dictionary_from_txt(logger, dir_exisiting_labels, 'Detections_Archival_Components', Project)
        
        ### SQL
        A = create_dictionary_from_txt_sql(logger, os.path.join(Dirs.path_archival_components, 'labels'), 'Detections_Archival_Components', Project, 'annotations_archival', 'dimensions_archival')

        # else:
            # logger.debug(f"Multi-threaded with ({str(cfg['leafmachine']['project']['num_workers'])}) threads create_dictionary_from_txt() n_images > 4000")
            # A = create_dictionary_from_txt_parallel(logger, cfg, dir_exisiting_labels, 'Detections_Archival_Components', Project)

    else:
        logger.info("Running YOLOv5 to generate archival labels")
        # run(weights = weights,
        #     source = Project.dir_images,
        #     project = Dirs.path_archival_components,
        #     name = Dirs.run_name,
        #     imgsz = (1280, 1280),
        #     nosave = do_save_prediction_overlay_images,
        #     anno_type = 'Archival_Detector',
        #     conf_thres = threshold, 
        #     ignore_objects_for_overlay = ignore_objects,
        #     mode = 'LM2',
        #     LOGGER=logger)
        # split the image paths into 4 chunks
        source = Project.dir_images
        project = Dirs.path_archival_components
        name = Dirs.run_name
        imgsz = (1280, 1280)
        nosave = do_save_prediction_overlay_images
        anno_type = 'Archival_Detector'
        conf_thres = threshold
        ignore_objects_for_overlay = ignore_objects
        mode = 'LM2'
        LOGGER = logger

        # with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        #     futures = [executor.submit(run_in_parallel, weights, source, project, name, imgsz, nosave, anno_type,
        #                             conf_thres, 10, ignore_objects_for_overlay, mode, LOGGER, i, num_workers) for i in
        #             range(num_workers)]
        #     for future in concurrent.futures.as_completed(futures):
        #         try:
        #             _ = future.result()
        #         except Exception as e:
        #             logger.error(f'Error in thread: {e}')
        #             continue
        distribute_workloads(weights, source, project, name, imgsz, nosave, anno_type, conf_thres, ignore_objects_for_overlay, mode, LOGGER, num_workers)


        t2_stop = perf_counter()
        logger.info(f"[Archival components detection elapsed time] {round(t2_stop - t1_start)} seconds")
        logger.info(f"Threads [{num_workers}]")

        # if n_images <= 4000:
            # logger.debug("Single-threaded create_dictionary_from_txt() n_images <= 4000")
        
        ### CLASSIC
        # A = create_dictionary_from_txt(logger, os.path.join(Dirs.path_archival_components, 'labels'), 'Detections_Archival_Components', Project)
        
        ### SQL
        A = create_dictionary_from_txt_sql(logger, os.path.join(Dirs.path_archival_components, 'labels'), 'Detections_Archival_Components', Project, 'annotations_archival', 'dimensions_archival')
        # else:
            # logger.debug(f"Multi-threaded with ({str(cfg['leafmachine']['project']['num_workers'])}) threads create_dictionary_from_txt() n_images > 4000")
            # A = create_dictionary_from_txt_parallel(logger, cfg, os.path.join(Dirs.path_archival_components, 'labels'), 'Detections_Archival_Components', Project)
    
    # dict_to_json(Project.project_data, Dirs.path_archival_components, 'Detections_Archival_Components.json')

    t1_stop = perf_counter()
    t_acd = f"[Processing archival components elapsed time] {round(t1_stop - t1_start)} seconds ({round((t1_stop - t1_start)/60)} minutes)"
    logger.info(t_acd)
    time_report['t_acd'] = t_acd

    torch.cuda.empty_cache()
    return Project, time_report


def detect_armature_components(cfg, logger, dir_home, Project, Dirs):
    t1_start = perf_counter()
    n_images = len(os.listdir(Project.dir_images))
    logger.name = 'Locating Armature Components'
    logger.info(f"Detecting armature components in {n_images} images")

    if cfg['leafmachine']['project']['num_workers'] is None:
        num_workers = 1
    else:
        num_workers = int(cfg['leafmachine']['project']['num_workers'])

    # Weights folder base
    dir_weights = os.path.join(dir_home, 'leafmachine2', 'component_detector','runs','train')
    
    # Detection threshold
    threshold = cfg['leafmachine']['armature_component_detector']['minimum_confidence_threshold']

    detector_version = cfg['leafmachine']['armature_component_detector']['detector_version']
    detector_iteration = cfg['leafmachine']['armature_component_detector']['detector_iteration']
    detector_weights = cfg['leafmachine']['armature_component_detector']['detector_weights']
    weights =  os.path.join(dir_weights,'Armature_Detector',detector_version,detector_iteration,'weights',detector_weights)

    do_save_prediction_overlay_images = not cfg['leafmachine']['armature_component_detector']['do_save_prediction_overlay_images']
    ignore_objects = cfg['leafmachine']['armature_component_detector']['ignore_objects_for_overlay']
    ignore_objects = ignore_objects or []

    logger.info("Running YOLOv5 to generate armature labels")

    source = Project.dir_images
    project = Dirs.path_armature_components
    name = Dirs.run_name
    imgsz = (1280, 1280)
    nosave = do_save_prediction_overlay_images
    anno_type = 'Armature_Detector'
    conf_thres = threshold
    ignore_objects_for_overlay = ignore_objects
    mode = 'LM2'
    LOGGER = logger

    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     futures = [executor.submit(run_in_parallel, weights, source, project, name, imgsz, nosave, anno_type,
    #                             conf_thres, 10, ignore_objects_for_overlay, mode, LOGGER, i, num_workers) for i in
    #             range(num_workers)]
    #     for future in concurrent.futures.as_completed(futures):
    #         try:
    #             _ = future.result()
    #         except Exception as e:
    #             logger.error(f'Error in thread: {e}')
    #             continue
    distribute_workloads(weights, source, project, name, imgsz, nosave, anno_type, conf_thres, ignore_objects_for_overlay, mode, LOGGER, num_workers)

    t2_stop = perf_counter()
    logger.info(f"[Plant components detection elapsed time] {round(t2_stop - t1_start)} seconds")
    logger.info(f"Threads [{num_workers}]")

    if len(Project.dir_images) <= 4000:
        logger.debug("Single-threaded create_dictionary_from_txt() len(Project.dir_images) <= 4000")
        A = create_dictionary_from_txt(logger, os.path.join(Dirs.path_armature_components, 'labels'), 'Detections_Armature_Components', Project)
    else:
        logger.debug(f"Multi-threaded with ({str(cfg['leafmachine']['project']['num_workers'])}) threads create_dictionary_from_txt() len(Project.dir_images) > 4000")
        A = create_dictionary_from_txt_parallel(logger, cfg, os.path.join(Dirs.path_armature_components, 'labels'), 'Detections_Armature_Components', Project)

    # dict_to_json(Project.project_data, Dirs.path_armature_components, 'Detections_Armature_Components.json')
    
    t1_stop = perf_counter()
    logger.info(f"[Processing armature components elapsed time] {round(t1_stop - t1_start)} seconds")
    torch.cuda.empty_cache()
    return Project


''' RUN IN PARALLEL'''
def distribute_workloads(weights, source, project, name, imgsz, nosave, anno_type, conf_thres, ignore_objects_for_overlay, mode, LOGGER, num_workers):
    num_files = len(os.listdir(source))
    LOGGER.info(f"The number of worker threads: ({num_workers}), number of files ({num_files}).")

    files = [os.path.join(source, f) for f in os.listdir(source) if f.lower().endswith('.jpg')]
    chunk_size = (num_files + num_workers - 1) // num_workers  # Ensuring each worker has something to do, ceiling division

    queue = Queue()
    # Start worker threads
    workers = []
    for _ in range(num_workers):
        t = Thread(target=worker_object_detector, args=(queue, weights, project, name, imgsz, nosave, anno_type, conf_thres, ignore_objects_for_overlay, mode, LOGGER))
        t.start()
        workers.append(t)

    # Enqueue sublists of files
    for i in range(num_workers):
        start = i * chunk_size
        end = min(start + chunk_size, num_files)
        sub_source = files[start:end]
        queue.put(sub_source)

    # Block until all tasks are done
    queue.join()

    # Stop workers
    for _ in range(num_workers):
        queue.put(None)  # send as many None as the number of workers to stop them
    for t in workers:
        t.join()
        
def worker_object_detector(queue, weights, project, name, imgsz, nosave, anno_type, conf_thres, ignore_objects_for_overlay, mode, LOGGER):
    while True:
        sub_source = queue.get()
        if sub_source is None:
            break  # None is the signal to stop processing
        try:
            run(weights=weights,
                source=sub_source,
                project=project,
                name=name,
                imgsz=imgsz,
                nosave=nosave,
                anno_type=anno_type,
                conf_thres=conf_thres,
                ignore_objects_for_overlay=ignore_objects_for_overlay,
                mode=mode,
                LOGGER=LOGGER)
        except Exception as e:
            LOGGER.error(f'Error in processing: {e}')
        queue.task_done()



def run_in_parallel(weights, source, project, name, imgsz, nosave, anno_type, conf_thres, line_thickness, ignore_objects_for_overlay, mode, LOGGER, show_all_logs, chunk, n_workers):
    num_files = len(os.listdir(source))
    LOGGER.info(f"The number of worker threads: ({n_workers}), number of files ({num_files}).")

    chunk_size = len(os.listdir(source)) // n_workers
    start = chunk * chunk_size
    end = start + chunk_size if chunk < (n_workers-1) else len(os.listdir(source))

    sub_source = [os.path.join(source, f) for f in os.listdir(source)[start:end] if f.lower().endswith('.jpg')]

    run(weights=weights,
        source=sub_source,
        project=project,
        name=name,
        imgsz=imgsz,
        nosave=nosave,
        anno_type=anno_type,
        conf_thres=conf_thres,
        ignore_objects_for_overlay=ignore_objects_for_overlay,
        mode=mode,
        LOGGER=LOGGER)

''' RUN IN PARALLEL'''


###### Multi-thread NOTE this works, but unless there are several thousand images, it will be slower
# def process_file(logger, file, dir_components, component, Project, lock):
#     file_name = str(file.split('.')[0])
#     with open(os.path.join(dir_components, file), "r") as f:
#         with lock:
#             Project.project_data[file_name][component] = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]
#             try:
#                 image_path = glob.glob(os.path.join(Project.dir_images, file_name + '.*'))[0]
#                 name_ext = os.path.basename(image_path)
#                 with Image.open(image_path) as im:
#                     _, ext = os.path.splitext(name_ext)
#                     if ext not in ['.jpg']:
#                         im = im.convert('RGB')
#                         im.save(os.path.join(Project.dir_images, file_name) + '.jpg', quality=100)
#                         # file_name += '.jpg'
#                     width, height = im.size
#             except Exception as e:
#                 print(f"Unable to get image dimensions. Error: {e}")
#                 logger.info(f"Unable to get image dimensions. Error: {e}")
#                 width, height = None, None
#             if width and height:
#                 Project.project_data[file_name]['height'] = int(height)
#                 Project.project_data[file_name]['width'] = int(width)


# def create_dictionary_from_txt_parallel(logger, cfg, dir_components, component, Project):
#     if cfg['leafmachine']['project']['num_workers'] is None:
#         num_workers = 4 
#     else:
#         num_workers = int(cfg['leafmachine']['project']['num_workers'])

#     files = [file for file in os.listdir(dir_components) if file.endswith(".txt")]
#     lock = Lock()
#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
#         futures = []
#         for file in files:
#             futures.append(executor.submit(process_file, logger, file, dir_components, component, Project, lock))
#         for future in concurrent.futures.as_completed(futures):
#             pass
#     return Project.project_data
# def create_dictionary_from_txt_parallel(logger, cfg, dir_components, component, Project):
#     if cfg['leafmachine']['project']['num_workers'] is None:
#         num_workers = 4
#     else:
#         num_workers = int(cfg['leafmachine']['project']['num_workers'])

#     files = [file for file in os.listdir(dir_components) if file.endswith(".txt")]
#     lock = Lock()
#     queue = Queue()

#     # Start worker threads
#     workers = []
#     for _ in range(num_workers):
#         t = Thread(target=worker, args=(queue, logger, dir_components, component, Project, lock))
#         t.start()
#         workers.append(t)

#     # Enqueue all files
#     for file in files:
#         queue.put(file)

#     # Block until all tasks are done
#     queue.join()

#     # Stop workers
#     for _ in range(num_workers):
#         queue.put(None)  # send as many None as the number of workers to stop them
#     for t in workers:
#         t.join()

#     return Project.project_data

# def process_file(logger, file, dir_components, component, Project, lock):
#     file_name = str(file.split('.')[0])
#     with open(os.path.join(dir_components, file), "r") as f:
#         with lock:
#             Project.project_data[file_name][component] = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]
#             try:
#                 image_path = glob.glob(os.path.join(Project.dir_images, file_name + '.*'))[0]
#                 name_ext = os.path.basename(image_path)
#                 with Image.open(image_path) as im:
#                     _, ext = os.path.splitext(name_ext)
#                     if ext not in ['.jpg']:
#                         im.convert('RGB').save(os.path.join(Project.dir_images, file_name + '.jpg'), quality=100)
#                     width, height = im.size
#             except Exception as e:
#                 logger.info(f"Unable to get image dimensions. Error: {e}")
#                 width, height = None, None
#             if width and height:
#                 Project.project_data[file_name]['height'] = int(height)
#                 Project.project_data[file_name]['width'] = int(width)

# def worker(queue, logger, dir_components, component, Project, lock):
#     while True:
#         file = queue.get()
#         if file is None:
#             break  # None is the signal to stop processing
#         process_file(logger, file, dir_components, component, Project, lock)
#         queue.task_done()



async def read_file_parallel(file_path):
    async with aiofiles.open(file_path, mode='r') as f:
        return await f.readlines()

def process_image_parallel(file_name, Project, logger):
    result = {}
    try:
        image_path = glob.glob(os.path.join(Project.dir_images, file_name + '.*'))[0]
        name_ext = os.path.basename(image_path)
        with Image.open(image_path) as im:
            _, ext = os.path.splitext(name_ext)
            if ext not in ['.jpg']:
                im = im.convert('RGB')
                im.save(os.path.join(Project.dir_images, file_name) + '.jpg', quality=100)
            width, height = im.size
            result['height'] = int(height)
            result['width'] = int(width)
    except Exception as e:
        logger.info(f"Unable to process image for file {file_name}. Error: {e}")
    return result

def process_file_parallel(file, dir_components, component, Project, logger):
    result = {}
    try:
        if file.endswith(".txt"):
            file_name = str(file.split('.')[0])
            file_path = os.path.join(dir_components, file)
            lines = asyncio.run(read_file_parallel(file_path))
            result[file_name] = {
                component: [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in lines]
            }
            image_result = process_image_parallel(file_name, Project, logger)
            if image_result:
                result[file_name].update(image_result)
    except Exception as e:
        logger.info(f"Unable to process file {file}. Error: {e}")
    return result

def create_dictionary_from_txt_parallel(logger, cfg, dir_components, component, Project):
    num_workers = cfg['leafmachine']['project'].get('num_workers', 4)
    files = [file for file in os.listdir(dir_components) if file.endswith(".txt")]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file_parallel, file, dir_components, component, Project, logger): file for file in files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading Annotations", colour='green'):
            result = future.result()
            if result:
                for file_name, data in result.items():
                    Project.project_data[file_name] = data

    return Project.project_data
######












def create_annotations_table(conn, table_name):
    try:
        sql_create_annotations_table = f"""CREATE TABLE IF NOT EXISTS {table_name} (
                                            id INTEGER PRIMARY KEY,
                                            file_name TEXT NOT NULL,
                                            component TEXT NOT NULL,
                                            annotation TEXT NOT NULL
                                         );"""
        cur = conn.cursor()
        cur.execute(sql_create_annotations_table)
        conn.commit()
    except Error as e:
        print(e)

def create_image_dimensions_table(conn, table_name):
    try:
        sql_create_dimensions_table = f"""CREATE TABLE IF NOT EXISTS {table_name} (
                                            id INTEGER PRIMARY KEY,
                                            file_name TEXT NOT NULL,
                                            width INTEGER,
                                            height INTEGER
                                         );"""
        cur = conn.cursor()
        cur.execute(sql_create_dimensions_table)
        conn.commit()
    except Error as e:
        print(e)

def create_dictionary_from_txt_sql(logger, dir_components, component, ProjectSQL, annotations_table, dimensions_table):
    conn = ProjectSQL.conn

    # Ensure tables exist
    create_annotations_table(conn, annotations_table)
    create_image_dimensions_table(conn, dimensions_table)

    for file in tqdm(os.listdir(dir_components), desc="Loading Annotations", colour='green'):
        if file.endswith(".txt"):
            file_name = str(file.split('.')[0])
            annotations = []
            with open(os.path.join(dir_components, file), "r") as f:
                annotations = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]

            # Insert annotations into the database
            for annotation in annotations:
                annotation_str = ','.join(map(str, annotation))
                sql = f'''INSERT INTO {annotations_table}(file_name, component, annotation)
                         VALUES(?, ?, ?)'''
                cur = conn.cursor()
                cur.execute(sql, (file_name, component, annotation_str))
            conn.commit()

            try:
                image_path = glob.glob(os.path.join(ProjectSQL.dir_images, file_name + '.*'))[0]
                name_ext = os.path.basename(image_path)
                with Image.open(image_path) as im:
                    _, ext = os.path.splitext(name_ext)
                    if ext not in ['.jpg']:
                        im = im.convert('RGB')
                        im.save(os.path.join(ProjectSQL.dir_images, file_name) + '.jpg', quality=100)
                    width, height = im.size
            except Exception as e:
                logger.info(f"Unable to get image dimensions. Error: {e}")
                width, height = None, None

            if width and height:
                # Insert image dimensions into the database
                sql = f'''INSERT INTO {dimensions_table}(file_name, width, height)
                         VALUES(?, ?, ?)'''
                cur.execute(sql, (file_name, width, height))
                conn.commit()

    return None  # No longer returning Project.project_data since data is in SQL now














# Single threaded for non-SQL Project
def create_dictionary_from_txt(logger, dir_components, component, Project):
    # dict_labels = {}
    for file in tqdm(os.listdir(dir_components), desc="Loading Annotations", colour='green'):
        if file.endswith(".txt"):
            file_name = str(file.split('.')[0])
            with open(os.path.join(dir_components, file), "r") as f:
                # dict_labels[file] = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]
                Project.project_data[file_name][component] = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]
                try:
                    image_path = glob.glob(os.path.join(Project.dir_images, file_name + '.*'))[0]
                    name_ext = os.path.basename(image_path)
                    with Image.open(image_path) as im:
                        _, ext = os.path.splitext(name_ext)
                        if ext not in ['.jpg']:
                            im = im.convert('RGB')
                            im.save(os.path.join(Project.dir_images, file_name) + '.jpg', quality=100)
                            # file_name += '.jpg'
                        width, height = im.size
                except Exception as e:
                    # print(f"Unable to get image dimensions. Error: {e}")
                    logger.info(f"Unable to get image dimensions. Error: {e}")
                    width, height = None, None
                if width and height:
                    Project.project_data[file_name]['height'] = int(height)
                    Project.project_data[file_name]['width'] = int(width)
    # for key, value in dict_labels.items():
    #     print(f'{key}  --> {value}')
    return Project.project_data
def create_dictionary_from_txt_SQL(logger, dir_components, component, Project):
    for file in tqdm(os.listdir(dir_components), desc="Loading Annotations", colour='green'):
        if file.endswith(".txt"):
            file_name = str(file.split('.')[0])
            annotations = []
            with open(os.path.join(dir_components, file), "r") as f:
                annotations = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]

            try:
                image_path = glob.glob(os.path.join(Project.dir_images, file_name + '.*'))[0]
                name_ext = os.path.basename(image_path)
                with Image.open(image_path) as im:
                    _, ext = os.path.splitext(name_ext)
                    if ext not in ['.jpg']:
                        im = im.convert('RGB')
                        im.save(os.path.join(Project.dir_images, file_name) + '.jpg', quality=100)
                        image_path = os.path.join(Project.dir_images, file_name) + '.jpg'
                    width, height = im.size
            except Exception as e:
                logger.info(f"Unable to get image dimensions for {file_name}. Error: {e}")
                width, height = None, None

            # Insert the annotations into the archival_components table
            try:
                cur = Project.conn.cursor()
                
                if component == "Detections_Archival_Components":
                    cur.execute('''INSERT INTO archival_components (image_name, component, annotations)
                                VALUES (?, ?, ?)''', (file_name, component, str(annotations)))
                    
                elif component == "Detections_Plant_Components":
                    cur.execute('''INSERT INTO plant_components (image_name, component, annotations)
                                VALUES (?, ?, ?)''', (file_name, component, str(annotations)))
                else:
                    raise

                if width and height:
                    cur.execute('''UPDATE images SET width = ?, height = ? WHERE name = ?''',
                                (width, height, file_name))
                Project.conn.commit()
            except Error as e:
                logger.info(f"Error inserting annotations for {file_name} into database: {e}")
    
    return "Annotations added to the database"



# old below   
'''def create_dictionary_from_txt(dir_components, component, Project):
    # dict_labels = {}
    for file in os.listdir(dir_components):
        if file.endswith(".txt"):
            file_name = str(file.split('.')[0])
            with open(os.path.join(dir_components, file), "r") as f:
                # dict_labels[file] = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]
                Project.project_data[file_name][component] = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]
                try:
                    width, height = imagesize.get(os.path.join(Project.dir_images, '.'.join([file_name,'jpg'])))
                except Exception as e:
                    print(f"Image not in 'jpg' format. Trying 'jpeg'. Note that other formats are not supported.{e}")
                    width, height = imagesize.get(os.path.join(Project.dir_images, '.'.join([file_name,'jpeg'])))
                Project.project_data[file_name]['height'] = int(height)
                Project.project_data[file_name]['width'] = int(width)
    # for key, value in dict_labels.items():
    #     print(f'{key}  --> {value}')
    return Project.project_data'''



def dict_to_json(dict_labels, dir_components, name_json):
    dir_components = os.path.join(dir_components, 'JSON')
    with open(os.path.join(dir_components, name_json), "w") as outfile:
        json.dump(dict_labels, outfile)

def fetch_labels(dir_exisiting_labels, new_dir):
    shutil.copytree(dir_exisiting_labels, new_dir)


'''Landmarks - uses YOLO, but works differently than above. A hybrid between segmentation and component detector'''
def detect_landmarks(cfg, time_report, logger, dir_home, ProjectSQL, batch, n_batches, Batch_Names, Dirs, segmentation_complete):
    start_t = perf_counter()
    logger.name = f'[BATCH {batch+1} Detect Landmarks]'
    logger.info(f'Detecting landmarks for batch {batch+1} of {n_batches}')

    show_all_logs = False

    landmark_whole_leaves = cfg['leafmachine']['landmark_detector']['landmark_whole_leaves']
    landmark_partial_leaves = cfg['leafmachine']['landmark_detector']['landmark_partial_leaves']

    landmarks_whole_leaves_props = {}
    landmarks_whole_leaves_overlay = {}
    landmarks_partial_leaves_props = {}
    landmarks_partial_leaves_overlay = {}

    if landmark_whole_leaves:
        run_landmarks(cfg, logger, show_all_logs, dir_home, ProjectSQL, batch, n_batches, Batch_Names, Dirs, 'Landmarks_Whole_Leaves', segmentation_complete)
    if landmark_partial_leaves:
        run_landmarks(cfg, logger, show_all_logs, dir_home, ProjectSQL, batch, n_batches, Batch_Names, Dirs, 'Landmarks_Partial_Leaves', segmentation_complete)

    # if cfg['leafmachine']['leaf_segmentation']['segment_whole_leaves']:
    #     landmarks_whole_leaves_props_batch, landmarks_whole_leaves_overlay_batch = run_landmarks(Instance_Detector_Whole, Project.project_data_list[batch], 0, 
    #                                                                                 "Segmentation_Whole_Leaf", "Whole_Leaf_Cropped", cfg, Project, Dirs, batch, n_batches)#, start+1, end)
    #     landmarks_whole_leaves_props.update(landmarks_whole_leaves_props_batch)
    #     landmarks_whole_leaves_overlay.update(landmarks_whole_leaves_overlay_batch)
    # if cfg['leafmachine']['leaf_segmentation']['segment_partial_leaves']:
    #     landmarks_partial_leaves_props_batch, landmarks_partial_leaves_overlay_batch = run_landmarks(Instance_Detector_Partial, Project.project_data_list[batch], 1, 
    #                                                                                 "Segmentation_Partial_Leaf", "Partial_Leaf_Cropped", cfg, Project, Dirs, batch, n_batches)#, start+1, end)
    #     landmarks_partial_leaves_props.update(landmarks_partial_leaves_props_batch)
    #     landmarks_partial_leaves_overlay.update(landmarks_partial_leaves_overlay_batch)
    
    end_t = perf_counter()
    t_land = f"[Batch {batch+1}/{n_batches}: Landmark Detection elapsed time] {round(end_t - start_t)} seconds ({round((end_t - start_t)/60)} minutes)"
    logger.info(t_land)
    time_report['t_land'] = t_land
    return time_report


def detect_armature(cfg, logger, dir_home, Project, batch, n_batches, Dirs, segmentation_complete):
    start_t = perf_counter()
    logger.name = f'[BATCH {batch+1} Detect Armature]'
    logger.info(f'Detecting armature for batch {batch+1} of {n_batches}')

    landmark_armature = cfg['leafmachine']['modules']['armature']

    landmarks_armature_props = {}
    landmarks_armature_overlay = {}

    if landmark_armature:
        run_armature(cfg, logger, dir_home, Project, batch, n_batches, Dirs, 'Landmarks_Armature', segmentation_complete)

    end_t = perf_counter()
    logger.info(f'Batch {batch+1}/{n_batches}: Armature Detection Duration --> {round((end_t - start_t)/60)} minutes')
    return Project


def run_armature(cfg, logger, dir_home, Project, batch, n_batches, Dirs, leaf_type, segmentation_complete):
    
    logger.info('Detecting armature landmarks from scratch')
    if leaf_type == 'Landmarks_Armature':
        dir_overlay = os.path.join(Dirs.landmarks_armature_overlay, ''.join(['batch_',str(batch+1)]))

    # if not segmentation_complete: # If segmentation was run, then don't redo the unpack, just do the crop into the temp folder
    if leaf_type == 'Landmarks_Armature': # TODO THE 0 is for prickles. For spines I'll need to add a 1 like with partial_leaves or just do it for all
        Project.project_data_list[batch] = unpack_class_from_components_armature(Project.project_data_list[batch], 0, 'Armature_YOLO', 'Armature_BBoxes', Project)
        Project.project_data_list[batch], dir_temp = crop_images_to_bbox_armature(Project.project_data_list[batch], 0, 'Armature_Cropped', "Armature_BBoxes", Project, Dirs, True, cfg)

    # Weights folder base
    dir_weights = os.path.join(dir_home, 'leafmachine2', 'component_detector','runs','train')
    
    # Detection threshold
    threshold = cfg['leafmachine']['landmark_detector_armature']['minimum_confidence_threshold']

    detector_version = cfg['leafmachine']['landmark_detector_armature']['detector_version']
    detector_iteration = cfg['leafmachine']['landmark_detector_armature']['detector_iteration']
    detector_weights = cfg['leafmachine']['landmark_detector_armature']['detector_weights']
    weights =  os.path.join(dir_weights,'Landmark_Detector_YOLO',detector_version,detector_iteration,'weights',detector_weights)

    do_save_prediction_overlay_images = not cfg['leafmachine']['landmark_detector_armature']['do_save_prediction_overlay_images']
    ignore_objects = cfg['leafmachine']['landmark_detector_armature']['ignore_objects_for_overlay']
    ignore_objects = ignore_objects or []
    if cfg['leafmachine']['project']['num_workers'] is None:
        num_workers = 1
    else:
        num_workers = int(cfg['leafmachine']['project']['num_workers'])

    has_images = False
    if len(os.listdir(dir_temp)) > 0:
        has_images = True
        source = dir_temp
        project = dir_overlay
        name = Dirs.run_name
        imgsz = (1280, 1280)
        nosave = do_save_prediction_overlay_images
        anno_type = 'Armature_Detector'
        conf_thres = threshold
        line_thickness = 2
        ignore_objects_for_overlay = ignore_objects
        mode = 'Landmark'
        LOGGER = logger

        # Initialize a Lock object to ensure thread safety
        lock = Lock()

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(run_in_parallel, weights, source, project, name, imgsz, nosave, anno_type,
                                    conf_thres, line_thickness, ignore_objects_for_overlay, mode, LOGGER, i, num_workers) for i in
                    range(num_workers)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    _ = future.result()
                except Exception as e:
                    logger.error(f'Error in thread: {e}')
                    continue

        with lock:
            if has_images:
                dimensions_dict = get_cropped_dimensions(dir_temp)
                A = add_to_dictionary_from_txt_armature(cfg, logger, Dirs, leaf_type, os.path.join(dir_overlay, 'labels'), leaf_type, Project, dimensions_dict, dir_temp, batch, n_batches)
            else:
                # TODO add empty placeholder to the image data
                pass
    
    # delete the temp dir
    try:
        shutil.rmtree(dir_temp)
    except:
        try:
            time.sleep(5)
            shutil.rmtree(dir_temp)
        except:
            try:
                time.sleep(5)
                shutil.rmtree(dir_temp)
            except:
                pass

    torch.cuda.empty_cache()

    return Project


def run_landmarks(cfg, logger, show_all_logs, dir_home, ProjectSQL, batch, n_batches, Batch_Names, Dirs, leaf_type, segmentation_complete):
    use_existing_landmark_detections = cfg['leafmachine']['landmark_detector']['use_existing_landmark_detections']
    conn = sqlite3.connect(ProjectSQL.database)
    cur = conn.cursor()


    if use_existing_landmark_detections is None:
        logger.info('Detecting landmarks from scratch')

        if leaf_type == 'Landmarks_Whole_Leaves':
            dir_overlay = os.path.join(Dirs.landmarks_whole_leaves_overlay, f'batch_{batch+1}')
            dict_from = 'Whole_Leaf_Cropped'
            dir_leaves = Dirs.whole_leaves
        elif leaf_type == 'Landmarks_Partial_Leaves':
            dir_overlay = os.path.join(Dirs.landmarks_partial_leaves_overlay, f'batch_{batch+1}')
            dict_from = 'Partial_Leaf_Cropped'
            dir_leaves = Dirs.partial_leaves

        # Retrieve cropped images from the SQL database
        placeholders = ','.join(['?'] * len(Batch_Names))
        cur.execute(f"SELECT crop_name, cropped_image FROM {dict_from} WHERE file_name IN (SELECT name FROM images WHERE name IN ({placeholders}))", Batch_Names)
        crops = cur.fetchall()

        if len(crops) > 0:
            has_images = True

            # Continue with the landmark detection as before

            # Weights folder base
            dir_weights = os.path.join(dir_home, 'leafmachine2', 'component_detector', 'runs', 'train')
            
            # Detection threshold
            threshold = cfg['leafmachine']['landmark_detector']['minimum_confidence_threshold']
            detector_version = cfg['leafmachine']['landmark_detector']['detector_version']
            detector_iteration = cfg['leafmachine']['landmark_detector']['detector_iteration']
            detector_weights = cfg['leafmachine']['landmark_detector']['detector_weights']
            weights = os.path.join(dir_weights, 'Landmark_Detector_YOLO', detector_version, detector_iteration, 'weights', detector_weights)

            do_save_prediction_overlay_images = not cfg['leafmachine']['landmark_detector']['do_save_prediction_overlay_images']
            ignore_objects = cfg['leafmachine']['landmark_detector']['ignore_objects_for_overlay'] or []
            num_workers = int(cfg['leafmachine']['project']['num_workers'] or 1)

            source = dir_leaves
            project = dir_overlay
            name = Dirs.run_name
            imgsz = (1280, 1280)
            nosave = do_save_prediction_overlay_images
            anno_type = 'Landmark_Detector'
            conf_thres = threshold
            line_thickness = 2
            mode = 'Landmark'
            LOGGER = logger

            lock = Lock()

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        run_in_parallel, weights, source, project, name, imgsz, nosave, anno_type,
                        conf_thres, line_thickness, ignore_objects, mode, LOGGER, show_all_logs, i, num_workers
                    ) for i in range(num_workers)
                ]
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f'Error in thread: {e}')
                        continue

            with lock:
                if has_images:
                    dimensions_dict = get_cropped_dimensions(dir_leaves)
                    add_landmarks_to_sql(cur, cfg, logger, show_all_logs, Dirs, leaf_type, os.path.join(dir_overlay, 'labels'), dimensions_dict, dir_leaves, batch, n_batches)
                else:
                    # TODO: Handle case with no images
                    pass
    else:
        logger.error('LOADING existing ANNOTATIONS IS NOT SUPPORTED YET')
        # logger.info('Loading existing landmark annotations')
        # dir_temp = os.path.join(use_existing_landmark_detections, f'batch_{batch+1}', 'labels')
        # dimensions_dict = get_cropped_dimensions(dir_temp)
        # add_landmarks_to_sql(cur, cfg, logger, show_all_logs, Dirs, leaf_type, use_existing_landmark_detections, dimensions_dict, dir_temp, batch, n_batches)

    '''in the non-sql version this cropped the leaves to a temp folder that we then delete'''
    # # delete the temp dir
    # try:
    #     shutil.rmtree(dir_temp)
    # except:
    #     try:
    #         time.sleep(5)
    #         shutil.rmtree(dir_temp)
    #     except:
    #         try:
    #             time.sleep(5)
    #             shutil.rmtree(dir_temp)
    #         except:
    #             pass

    torch.cuda.empty_cache()
    conn.close()


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

def add_landmarks_to_sql(cur, cfg, logger, show_all_logs, Dirs, leaf_type, dir_labels, dimensions_dict, dir_temp, batch, n_batches):
    dpi = cfg['leafmachine']['overlay']['overlay_dpi']
    conn = cur.connection  # Ensure you have the connection object
    fig = plt.figure(figsize=(8.27, 11.69), dpi=dpi)  # A4 size, 300 dpi
    row, col = 0, 0

    for file in os.listdir(dir_labels):
        file_name = str(file.split('.')[0])
        file_name_parent = file_name.split('__')[0]

        # Fetch image dimensions from the provided dictionary
        if file_name in dimensions_dict:
            height, width = dimensions_dict[file_name]
        else:
            height, width = None, None  # Handle missing dimensions gracefully

        if file.endswith(".txt"):
            with open(os.path.join(dir_labels, file), "r") as f:
                all_points = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]

                # Create LeafSkeleton object
                Leaf_Skeleton = LeafSkeleton(cfg, logger, show_all_logs, Dirs, leaf_type, all_points, height, width, dir_temp, file_name)

                # Convert numpy objects to JSON serializable formats
                apex_center_serializable = convert_ndarray_to_list(Leaf_Skeleton.apex_center)
                base_center_serializable = convert_ndarray_to_list(Leaf_Skeleton.base_center)
                lamina_tip_serializable = convert_ndarray_to_list(Leaf_Skeleton.lamina_tip)
                lamina_base_serializable = convert_ndarray_to_list(Leaf_Skeleton.lamina_base)
                lamina_fit_serializable = convert_ndarray_to_list(Leaf_Skeleton.lamina_fit)
                width_left_serializable = convert_ndarray_to_list(Leaf_Skeleton.width_left)
                width_right_serializable = convert_ndarray_to_list(Leaf_Skeleton.width_right)
                lobes_serializable = convert_ndarray_to_list(Leaf_Skeleton.lobes)
                midvein_fit_serializable = convert_ndarray_to_list(Leaf_Skeleton.midvein_fit)
                midvein_fit_points_serializable = convert_ndarray_to_list(Leaf_Skeleton.midvein_fit_points)
                ordered_midvein_serializable = convert_ndarray_to_list(Leaf_Skeleton.ordered_midvein)
                ordered_petiole_serializable = convert_ndarray_to_list(Leaf_Skeleton.ordered_petiole)

                # Ensure that `None` values are replaced with a default value
                values = [
                    file_name_parent, 
                    json.dumps(convert_ndarray_to_list(Leaf_Skeleton.all_points)) or '[]',
                    height if height is not None else None,
                    width if width is not None else None,
                    json.dumps(apex_center_serializable) if apex_center_serializable is not None else None,
                    Leaf_Skeleton.apex_angle_degrees if Leaf_Skeleton.apex_angle_degrees is not None else None,
                    json.dumps(base_center_serializable) if base_center_serializable is not None else None,
                    Leaf_Skeleton.base_angle_degrees if Leaf_Skeleton.base_angle_degrees is not None else None,
                    json.dumps(lamina_tip_serializable) if lamina_tip_serializable is not None else None,
                    json.dumps(lamina_base_serializable) if lamina_base_serializable is not None else None,
                    Leaf_Skeleton.lamina_length if Leaf_Skeleton.lamina_length is not None else None,
                    json.dumps(lamina_fit_serializable) if lamina_fit_serializable is not None else None,
                    Leaf_Skeleton.lamina_width if Leaf_Skeleton.lamina_width is not None else None,
                    json.dumps(width_left_serializable) if width_left_serializable is not None else None,
                    json.dumps(width_right_serializable) if width_right_serializable is not None else None,
                    Leaf_Skeleton.lobe_count if Leaf_Skeleton.lobe_count is not None else None,
                    json.dumps(lobes_serializable) if lobes_serializable is not None else None,
                    json.dumps(midvein_fit_serializable) if midvein_fit_serializable is not None else None,
                    json.dumps(midvein_fit_points_serializable) if midvein_fit_points_serializable is not None else None,
                    json.dumps(ordered_midvein_serializable) if ordered_midvein_serializable is not None else None,
                    Leaf_Skeleton.ordered_midvein_length if Leaf_Skeleton.ordered_midvein_length is not None else None,
                    Leaf_Skeleton.has_midvein if Leaf_Skeleton.has_midvein is not None else None,
                    json.dumps(ordered_petiole_serializable) if ordered_petiole_serializable is not None else None,
                    Leaf_Skeleton.ordered_petiole_length if Leaf_Skeleton.ordered_petiole_length is not None else None,
                    Leaf_Skeleton.has_ordered_petiole if Leaf_Skeleton.has_ordered_petiole is not None else None,
                    Leaf_Skeleton.is_split if Leaf_Skeleton.is_split is not None else None,
                    Leaf_Skeleton.has_apex if Leaf_Skeleton.has_apex is not None else None,
                    Leaf_Skeleton.has_base if Leaf_Skeleton.has_base is not None else None,
                    Leaf_Skeleton.has_lamina_tip if Leaf_Skeleton.has_lamina_tip is not None else None,
                    Leaf_Skeleton.has_lamina_base if Leaf_Skeleton.has_lamina_base is not None else None,
                    Leaf_Skeleton.has_lamina_length if Leaf_Skeleton.has_lamina_length is not None else None,
                    Leaf_Skeleton.has_width if Leaf_Skeleton.has_width is not None else None,
                    Leaf_Skeleton.has_lobes if Leaf_Skeleton.has_lobes is not None else None,
                    Leaf_Skeleton.is_complete_leaf if Leaf_Skeleton.is_complete_leaf is not None else None,
                    Leaf_Skeleton.is_leaf_no_width if Leaf_Skeleton.is_leaf_no_width is not None else None
                ]

                # Insert data into the database
                cur.execute(f"""
                    INSERT INTO {leaf_type} 
                    (file_name, all_points, height, width, 
                    apex_center, apex_angle_degrees, 
                    base_center, base_angle_degrees, 
                    lamina_tip, lamina_base, lamina_length, 
                    lamina_fit, lamina_width, width_left, width_right, 
                    lobe_count, lobes, 
                    midvein_fit, midvein_fit_points, ordered_midvein, ordered_midvein_length, has_midvein,
                    ordered_petiole, ordered_petiole_length, has_ordered_petiole, 
                    is_split, has_apex, has_base, 
                    has_lamina_tip, has_lamina_base, has_lamina_length, 
                    has_width, has_lobes, is_complete_leaf, is_leaf_no_width)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, values)

                # Visualize the results (optional)
                # final_add = cv2.cvtColor(Leaf_Skeleton.get_final(), cv2.COLOR_BGR2RGB)
                # ax = fig.add_subplot(5, 3, row * 3 + col + 1)
                # ax.imshow(final_add)
                # ax.axis('off')

                # col += 1
                # if col == 3:
                #     col = 0
                #     row += 1
                # if row == 5:
                #     row = 0
                #     fig = plt.figure(figsize=(8.27, 11.69), dpi=300)  # Create a new page

    conn.commit()



def add_to_dictionary_from_txt_armature(cfg, logger, Dirs, leaf_type, dir_components, component, Project, dimensions_dict, dir_temp, batch, n_batches):
    dpi = cfg['leafmachine']['overlay']['overlay_dpi']
    if leaf_type == 'Landmarks_Armature':
        logger.info(f'Detecting landmarks armature')
        pdf_path = os.path.join(Dirs.landmarks_armature_overlay_QC, ''.join(['landmarks_armature_overlay_QC__',str(batch+1), 'of', str(n_batches), '.pdf']))
        pdf_path_final = os.path.join(Dirs.landmarks_armature_overlay_final, ''.join(['landmarks_armature_overlay_final__',str(batch+1), 'of', str(n_batches), '.pdf']))

    ### FINAL
    # dict_labels = {}
    fig = plt.figure(figsize=(8.27, 11.69), dpi=dpi) # A4 size, 300 dpi
    row, col = 0, 0
    with PdfPages(pdf_path_final) as pdf:
        
        

        for file in os.listdir(dir_components):
            file_name = str(file.split('.')[0])
            file_name_parent = file_name.split('__')[0]

            # Project.project_data_list[batch][file_name_parent][component] = []

            if file_name_parent in Project.project_data_list[batch]:

                

                if file.endswith(".txt"):
                    with open(os.path.join(dir_components, file), "r") as f:
                        all_points = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]
                        # Project.project_data_list[batch][file_name_parent][component][file_name] = all_points

                        height = dimensions_dict[file_name][0]
                        width = dimensions_dict[file_name][1]

                        Armature_Skeleton = ArmatureSkeleton(cfg, logger, Dirs, leaf_type, all_points, height, width, dir_temp, file_name)
                        Project = add_armature_skeleton_to_project(cfg, logger, Project, batch, file_name_parent, component, Dirs, leaf_type, all_points, height, width, dir_temp, file_name, Armature_Skeleton)
                        final_add = cv2.cvtColor(Armature_Skeleton.get_final(), cv2.COLOR_BGR2RGB)

                        # Add image to the current subplot
                        ax = fig.add_subplot(5, 3, row * 3 + col + 1)
                        ax.imshow(final_add)
                        ax.axis('off')

                        col += 1
                        if col == 3:
                            col = 0
                            row += 1
                        if row == 5:
                            row = 0
                            pdf.savefig(fig)  # Save the current page
                            fig = plt.figure(figsize=(8.27, 11.69), dpi=300) # Create a new page
            else:
                pass

        if row != 0 or col != 0:
            pdf.savefig(fig)  # Save the remaining images on the last page


def save_leaf_skeleton_to_sql(cur, leaf_skeleton, file_name, crop_name):
    """
    Save the data from a LeafSkeleton object to the SQL database.
    """
    data_to_save = {
        'ordered_midvein': json.dumps(leaf_skeleton.ordered_midvein),
        'ordered_midvein_length': leaf_skeleton.ordered_midvein_length,
        'ordered_petiole': json.dumps(leaf_skeleton.ordered_petiole),
        'ordered_petiole_length': leaf_skeleton.ordered_petiole_length,
        'apex_center': json.dumps(leaf_skeleton.apex_center),
        'apex_angle_degrees': leaf_skeleton.apex_angle_degrees,
        'base_center': json.dumps(leaf_skeleton.base_center),
        'base_angle_degrees': leaf_skeleton.base_angle_degrees,
        'lamina_tip': json.dumps(leaf_skeleton.lamina_tip),
        'lamina_base': json.dumps(leaf_skeleton.lamina_base),
        'lamina_length': leaf_skeleton.lamina_length,
        'lamina_width': leaf_skeleton.lamina_width,
        'width_left': json.dumps(leaf_skeleton.width_left),
        'width_right': json.dumps(leaf_skeleton.width_right),
        'lobe_count': leaf_skeleton.lobe_count,
        'lobes': json.dumps(leaf_skeleton.lobes),
        'is_split': leaf_skeleton.is_split,
        'is_complete_leaf': leaf_skeleton.is_complete_leaf,
        'is_leaf_no_width': leaf_skeleton.is_leaf_no_width
    }

    cur.execute("""
        INSERT INTO LeafSkeleton (
            file_name, crop_name, leaf_type, ordered_midvein, ordered_midvein_length,
            ordered_petiole, ordered_petiole_length, apex_center, apex_angle_degrees,
            base_center, base_angle_degrees, lamina_tip, lamina_base, lamina_length,
            lamina_width, width_left, width_right, lobe_count, lobes,
            is_split, is_complete_leaf, is_leaf_no_width
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        file_name,
        crop_name,
        leaf_skeleton.leaf_type,
        data_to_save['ordered_midvein'],
        data_to_save['ordered_midvein_length'],
        data_to_save['ordered_petiole'],
        data_to_save['ordered_petiole_length'],
        data_to_save['apex_center'],
        data_to_save['apex_angle_degrees'],
        data_to_save['base_center'],
        data_to_save['base_angle_degrees'],
        data_to_save['lamina_tip'],
        data_to_save['lamina_base'],
        data_to_save['lamina_length'],
        data_to_save['lamina_width'],
        data_to_save['width_left'],
        data_to_save['width_right'],
        data_to_save['lobe_count'],
        data_to_save['lobes'],
        data_to_save['is_split'],
        data_to_save['is_complete_leaf'],
        data_to_save['is_leaf_no_width']
    ))

    # Commit the transaction
    cur.connection.commit()
    '''
def add_to_dictionary_from_txt(cfg, logger, show_all_logs, Dirs, leaf_type, dir_components, component, ProjectSQL, dimensions_dict, dir_temp, batch, n_batches):
    dpi = cfg['leafmachine']['overlay']['overlay_dpi']
    if leaf_type == 'Landmarks_Whole_Leaves':
        logger.info(f'Detecting landmarks whole leaves')
        pdf_path = os.path.join(Dirs.landmarks_whole_leaves_overlay_QC, ''.join(['landmarks_whole_leaves_overlay_QC__',str(batch+1), 'of', str(n_batches), '.pdf']))
        pdf_path_final = os.path.join(Dirs.landmarks_whole_leaves_overlay_final, ''.join(['landmarks_whole_leaves_overlay_final__',str(batch+1), 'of', str(n_batches), '.pdf']))
    elif leaf_type == 'Landmarks_Partial_Leaves':
        logger.info(f'Detecting landmarks partial leaves')
        pdf_path = os.path.join(Dirs.landmarks_partial_leaves_overlay_QC, ''.join(['landmarks_partial_leaves_overlay_QC__',str(batch+1), 'of', str(n_batches), '.pdf']))
        pdf_path_final = os.path.join(Dirs.landmarks_partial_leaves_overlay_final, ''.join(['landmarks_partial_leaves_overlay_final__',str(batch+1), 'of', str(n_batches), '.pdf']))
    elif leaf_type == 'Landmarks_Armature':
        logger.info(f'Detecting landmarks armature')
        pdf_path = os.path.join(Dirs.landmarks_armature_overlay_QC, ''.join(['landmarks_armature_overlay_QC__',str(batch+1), 'of', str(n_batches), '.pdf']))
        pdf_path_final = os.path.join(Dirs.landmarks_armature_overlay_final, ''.join(['landmarks_armature_overlay_final__',str(batch+1), 'of', str(n_batches), '.pdf']))

    ### FINAL
    # dict_labels = {}
    fig = plt.figure(figsize=(8.27, 11.69), dpi=dpi) # A4 size, 300 dpi
    row, col = 0, 0
    # with PdfPages(pdf_path_final) as pdf: #*********************************removed for minimal test
        
    

    for file in os.listdir(dir_components):
        file_name = str(file.split('.')[0])
        file_name_parent = file_name.split('__')[0]

        # Project.project_data_list[batch][file_name_parent][component] = []

        if file_name_parent in Project.project_data_list[batch]:

            

            if file.endswith(".txt"):
                with open(os.path.join(dir_components, file), "r") as f:
                    all_points = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]
                    # Project.project_data_list[batch][file_name_parent][component][file_name] = all_points

                    height = dimensions_dict[file_name][0]
                    width = dimensions_dict[file_name][1]

                    Leaf_Skeleton = LeafSkeleton(cfg, logger, show_all_logs, Dirs, leaf_type, all_points, height, width, dir_temp, file_name)
                    # Project = add_leaf_skeleton_to_project(cfg, logger, Project, batch, file_name_parent, component, Dirs, leaf_type, all_points, height, width, dir_temp, file_name, Leaf_Skeleton)
                    save_leaf_skeleton_to_sql(ProjectSQL.conn.cursor(), Leaf_Skeleton, file_name_parent, file_name)

                    final_add = cv2.cvtColor(Leaf_Skeleton.get_final(), cv2.COLOR_BGR2RGB)

                    # Add image to the current subplot
                    ax = fig.add_subplot(5, 3, row * 3 + col + 1)
                    ax.imshow(final_add)
                    ax.axis('off')

                    col += 1
                    if col == 3:
                        col = 0
                        row += 1
                    if row == 5:
                        row = 0
                        # pdf.savefig(fig)  # Save the current page
                        # fig = plt.figure(figsize=(8.27, 11.69), dpi=300) # Create a new page
        else:
            pass

        # if row != 0 or col != 0: ############################################################################
        #     pdf.savefig(fig)  # Save the remaining images on the last page

    ### QC
    '''
    '''do_save_QC_pdf = False # TODO refine this
    if do_save_QC_pdf:
        # dict_labels = {}
        fig = plt.figure(figsize=(8.27, 11.69), dpi=dpi) # A4 size, 300 dpi
        row, col = 0, 0
        with PdfPages(pdf_path) as pdf:



            for file in os.listdir(dir_components):
                file_name = str(file.split('.')[0])
                file_name_parent = file_name.split('__')[0]

                if file_name_parent in Project.project_data_list[batch]:

                    if file.endswith(".txt"):
                        with open(os.path.join(dir_components, file), "r") as f:
                            all_points = [[int(line.split()[0])] + list(map(float, line.split()[1:])) for line in f]
                            Project.project_data_list[batch][file_name_parent][component][file_name] = all_points

                            height = dimensions_dict[file_name][0]
                            width = dimensions_dict[file_name][1]

                            Leaf_Skeleton = LeafSkeleton(cfg, logger, Dirs, leaf_type, all_points, height, width, dir_temp, file_name)
                            QC_add = cv2.cvtColor(Leaf_Skeleton.get_QC(), cv2.COLOR_BGR2RGB)

                            # Add image to the current subplot
                            ax = fig.add_subplot(5, 3, row * 3 + col + 1)
                            ax.imshow(QC_add)
                            ax.axis('off')

                            col += 1
                            if col == 3:
                                col = 0
                                row += 1
                            if row == 5:
                                row = 0
                                pdf.savefig(fig)  # Save the current page
                                fig = plt.figure(figsize=(8.27, 11.69), dpi=300) # Create a new page
                else:
                    pass

            if row != 0 or col != 0:
                pdf.savefig(fig)  # Save the remaining images on the last page'''


def add_armature_skeleton_to_project(cfg, logger, Project, batch, file_name_parent, component, Dirs, leaf_type, all_points, height, width, dir_temp, file_name, ARM):
    if ARM.is_complete:
        try:
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'armature_status': 'complete'}, {'armature': ARM}]})
        except:
            Project.project_data_list[batch][file_name_parent][component] = []
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'armature_status': 'complete'}, {'armature': ARM}]})

    else:
        try:
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'armature_status': 'incomplete'}, {'armature': ARM}]})
        except:
            Project.project_data_list[batch][file_name_parent][component] = []
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'armature_status': 'incomplete'}, {'armature': ARM}]})


    return Project


def add_leaf_skeleton_to_project(cfg, logger, Project, batch, file_name_parent, component, Dirs, leaf_type, all_points, height, width, dir_temp, file_name, LS):

    if LS.is_complete_leaf:
        try:
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'landmark_status': 'complete_leaf'}, {'landmarks': LS}]})
        except:
            Project.project_data_list[batch][file_name_parent][component] = []
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'landmark_status': 'complete_leaf'}, {'landmarks': LS}]})
        # Project.project_data_list[batch][file_name_parent][component][file_name].update({'landmark_status': 'complete_leaf'})
        # Project.project_data_list[batch][file_name_parent][component][file_name].update({'landmarks': LS})

    elif LS.is_leaf_no_width:
        try:
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'landmark_status': 'leaf_no_width'}, {'landmarks': LS}]})
        except:
            Project.project_data_list[batch][file_name_parent][component] = []
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'landmark_status': 'leaf_no_width'}, {'landmarks': LS}]})
        # Project.project_data_list[batch][file_name_parent][component][file_name].update({'landmark_status': 'leaf_no_width'})
        # Project.project_data_list[batch][file_name_parent][component][file_name].update({'landmarks': LS})

    else:
        try:
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'landmark_status': 'incomplete'}, {'landmarks': LS}]})
        except:
            Project.project_data_list[batch][file_name_parent][component] = []
            Project.project_data_list[batch][file_name_parent][component].append({file_name: [{'landmark_status': 'incomplete'}, {'landmarks': LS}]})

        # Project.project_data_list[batch][file_name_parent][component][file_name].update({'landmark_status': 'incomplete'})
        # Project.project_data_list[batch][file_name_parent][component][file_name].update({'landmarks': LS})

    return Project


'''
self.determine_lamina_length('final') 

# Lamina tip and base
if self.has_lamina_tip:
    cv2.circle(self.image_final, self.lamina_tip, radius=4, color=(0, 255, 0), thickness=2)
    cv2.circle(self.image_final, self.lamina_tip, radius=2, color=(255, 255, 255), thickness=-1)
if self.has_lamina_base:
    cv2.circle(self.image_final, self.lamina_base, radius=4, color=(255, 0, 0), thickness=2)
    cv2.circle(self.image_final, self.lamina_base, radius=2, color=(255, 255, 255), thickness=-1)

# Apex angle
# if self.apex_center != []:
#     cv2.circle(self.image_final, self.apex_center, radius=3, color=(0, 255, 0), thickness=-1)
if self.apex_left != []:
    cv2.circle(self.image_final, self.apex_left, radius=3, color=(255, 0, 0), thickness=-1)
if self.apex_right != []:
    cv2.circle(self.image_final, self.apex_right, radius=3, color=(0, 0, 255), thickness=-1)

# Base angle
# if self.base_center:
#     cv2.circle(self.image_final, self.base_center, radius=3, color=(0, 255, 0), thickness=-1)
if self.base_left:
    cv2.circle(self.image_final, self.base_left, radius=3, color=(255, 0, 0), thickness=-1)
if self.base_right:
    cv2.circle(self.image_final, self.base_right, radius=3, color=(0, 0, 255), thickness=-1)

# Draw line of fit
for point in self.width_infer:


'''









def get_cropped_dimensions(dir_temp):
    dimensions_dict = {}
    for file_name in os.listdir(dir_temp):
        if file_name.endswith(".jpg"):
            img = cv2.imread(os.path.join(dir_temp, file_name))
            height, width, channels = img.shape
            stem = os.path.splitext(file_name)[0]
            dimensions_dict[stem] = (height, width)
    return dimensions_dict

def unpack_class_from_components_armature(dict_big, cls, dict_name_yolo, dict_name_location, Project):
    # Get the dict that contains plant parts, find the whole leaves
    for filename, value in dict_big.items():
        if "Detections_Armature_Components" in value:
            filtered_components = [val for val in value["Detections_Armature_Components"] if val[0] == cls]
            value[dict_name_yolo] = filtered_components

    for filename, value in dict_big.items():
        if "Detections_Armature_Components" in value:
            filtered_components = [val for val in value["Detections_Armature_Components"] if val[0] == cls]
            height = value['height']
            width = value['width']
            converted_list = [[convert_index_to_class_armature(val[0]), int((val[1] * width) - ((val[3] * width) / 2)), 
                                                                int((val[2] * height) - ((val[4] * height) / 2)), 
                                                                int(val[3] * width) + int((val[1] * width) - ((val[3] * width) / 2)), 
                                                                int(val[4] * height) + int((val[2] * height) - ((val[4] * height) / 2))] for val in filtered_components]
            # Verify that the crops are correct
            # img = Image.open(os.path.join(Project., '.'.join([filename,'jpg'])))
            # for d in converted_list:
            #     img_crop = img.crop((d[1], d[2], d[3], d[4]))
            #     img_crop.show() 
            value[dict_name_location] = converted_list
    # print(dict)
    return dict_big

def unpack_class_from_components(dict_big, cls, dict_name_yolo, dict_name_location, Project):
    # Get the dict that contains plant parts, find the whole leaves
    for filename, value in dict_big.items():
        if "Detections_Plant_Components" in value:
            filtered_components = [val for val in value["Detections_Plant_Components"] if val[0] == cls]
            value[dict_name_yolo] = filtered_components

    for filename, value in dict_big.items():
        if "Detections_Plant_Components" in value:
            filtered_components = [val for val in value["Detections_Plant_Components"] if val[0] == cls]
            height = value['height']
            width = value['width']
            converted_list = [[convert_index_to_class(val[0]), int((val[1] * width) - ((val[3] * width) / 2)), 
                                                                int((val[2] * height) - ((val[4] * height) / 2)), 
                                                                int(val[3] * width) + int((val[1] * width) - ((val[3] * width) / 2)), 
                                                                int(val[4] * height) + int((val[2] * height) - ((val[4] * height) / 2))] for val in filtered_components]
            # Verify that the crops are correct
            # img = Image.open(os.path.join(Project., '.'.join([filename,'jpg'])))
            # for d in converted_list:
            #     img_crop = img.crop((d[1], d[2], d[3], d[4]))
            #     img_crop.show() 
            value[dict_name_location] = converted_list
    # print(dict)
    return dict_big


def crop_images_to_bbox_armature(dict_big, cls, dict_name_cropped, dict_from, Project, Dirs, do_upscale=False, cfg=None):
    dir_temp = os.path.join(Dirs.landmarks, 'TEMP_landmarks')
    os.makedirs(dir_temp, exist_ok=True)
    # For each image, iterate through the whole leaves, segment, report data back to dict_plant_components
    for filename, value in dict_big.items():
        value[dict_name_cropped] = []
        if dict_from in value:
            bboxes_whole_leaves = [val for val in value[dict_from] if val[0] == convert_index_to_class_armature(cls)]
            if len(bboxes_whole_leaves) == 0:
                m = str(''.join(['No objects for class ', convert_index_to_class_armature(0), ' were found']))
                # Print_Verbose(cfg, 3, m).plain()
            else:
                try:
                    img = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpg'])))
                    # img = cv2.imread(os.path.join(Project, '.'.join([filename,'jpg']))) # Testing
                except:
                    img = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpeg'])))
                    # img = cv2.imread(os.path.join(Project, '.'.join([filename,'jpeg']))) # Testing
                
                for d in bboxes_whole_leaves:
                    # img_crop = img.crop((d[1], d[2], d[3], d[4])) # PIL
                    img_crop = img[d[2]:d[4], d[1]:d[3]]
                    loc = '-'.join([str(d[1]), str(d[2]), str(d[3]), str(d[4])])
                    # value[dict_name_cropped].append({crop_name: img_crop})
                    if do_upscale:
                        upscale_factor = int(cfg['leafmachine']['landmark_detector_armature']['upscale_factor'])
                        if cls == 0:
                            crop_name = '__'.join([filename,f"PRICKLE-{upscale_factor}x",loc])
                        height, width, _ = img_crop.shape
                        img_crop = cv2.resize(img_crop, ((width * upscale_factor), (height * upscale_factor)), interpolation=cv2.INTER_LANCZOS4)
                    else:
                        if cls == 0:
                            crop_name = '__'.join([filename,'PRICKLE',loc])

                    cv2.imwrite(os.path.join(dir_temp, '.'.join([crop_name,'jpg'])), img_crop)
                    # cv2.imshow('img_crop', img_crop)
                    # cv2.waitKey(0)
                    # img_crop.show() # PIL
    return dict_big, dir_temp


def crop_images_to_bbox(dict_big, cls, dict_name_cropped, dict_from, Project, Dirs):
    dir_temp = os.path.join(Dirs.landmarks, 'TEMP_landmarks')
    os.makedirs(dir_temp, exist_ok=True)
    # For each image, iterate through the whole leaves, segment, report data back to dict_plant_components
    for filename, value in dict_big.items():
        value[dict_name_cropped] = []
        if dict_from in value:
            bboxes_whole_leaves = [val for val in value[dict_from] if val[0] == convert_index_to_class(cls)]
            if len(bboxes_whole_leaves) == 0:
                m = str(''.join(['No objects for class ', convert_index_to_class(0), ' were found']))
                # Print_Verbose(cfg, 3, m).plain()
            else:
                try:
                    img = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpg'])))
                    # img = cv2.imread(os.path.join(Project, '.'.join([filename,'jpg']))) # Testing
                except:
                    img = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpeg'])))
                    # img = cv2.imread(os.path.join(Project, '.'.join([filename,'jpeg']))) # Testing
                
                for d in bboxes_whole_leaves:
                    # img_crop = img.crop((d[1], d[2], d[3], d[4])) # PIL
                    img_crop = img[d[2]:d[4], d[1]:d[3]]
                    loc = '-'.join([str(d[1]), str(d[2]), str(d[3]), str(d[4])])
                    if cls == 0:
                        crop_name = '__'.join([filename,'L',loc])
                    elif cls == 1:
                        crop_name = '__'.join([filename,'PL',loc])
                    elif cls == 2:
                        crop_name = '__'.join([filename,'ARM',loc])
                    # value[dict_name_cropped].append({crop_name: img_crop})
                    cv2.imwrite(os.path.join(dir_temp, '.'.join([crop_name,'jpg'])), img_crop)
                    # cv2.imshow('img_crop', img_crop)
                    # cv2.waitKey(0)
                    # img_crop.show() # PIL
    return dict_big, dir_temp

def convert_index_to_class(ind):
    mapping = {
        0: 'apex_angle',
        1: 'base_angle',
        2: 'lamina_base',
        3: 'lamina_tip',
        4: 'lamina_width',
        5: 'lobe_tip',
        6: 'midvein_trace',
        7: 'petiole_tip',
        8: 'petiole_trace',
    }
    return mapping.get(ind, 'Invalid class').lower()

def convert_index_to_class_armature(ind):
    mapping = {
        0: 'tip',
        1: 'middle',
        2: 'outer',
    }
    return mapping.get(ind, 'Invalid class').lower()
