import os, sys, inspect, json, shutil, cv2, time, glob 
from time import perf_counter
from threading import Thread, Lock
from queue import Queue

currentdir = os.path.dirname(inspect.getfile(inspect.currentframe()))
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentdir))))))
sys.path.append(currentdir)
sys.path.append(parentdir)

from leafmachine2.keypoint_detector.ultralytics.models.yolo.pose.predict import PosePredictor
from leafmachine2.component_detector.component_detector import unpack_class_from_components, crop_images_to_bbox

def detect_landmarks_keypoint(cfg, logger, dir_home, Project, batch, n_batches, Dirs, do_seg):
    start_t = perf_counter()
    logger.name = f'[BATCH {batch+1} Detect Landmarks --- Keypoints]'
    logger.info(f'Detecting landmarks for batch {batch+1} of {n_batches}')

    landmark_whole_leaves = cfg['leafmachine']['landmark_detector']['landmark_whole_leaves']
    landmark_partial_leaves = cfg['leafmachine']['landmark_detector']['landmark_partial_leaves']

    if landmark_whole_leaves:
        run_keypoints(cfg, logger, dir_home, Project, batch, n_batches, Dirs, 'Landmarks_Whole_Leaves')
    if landmark_partial_leaves:
        run_keypoints(cfg, logger, dir_home, Project, batch, n_batches, Dirs, 'Landmarks_Partial_Leaves')
    
    



    




    end_t = perf_counter()
    logger.info(f'Batch {batch+1}/{n_batches}: Landmark Detection Duration --> {round((end_t - start_t)/60)} minutes')
    return Project


def run_keypoints(cfg, logger, dir_home, Project, batch, n_batches, Dirs, leaf_type):
    if leaf_type == 'Landmarks_Whole_Leaves':
        dir_overlay = os.path.join(Dirs.landmarks_whole_leaves_overlay, ''.join(['batch_',str(batch+1)]))
    elif leaf_type == 'Landmarks_Partial_Leaves':
        dir_overlay = os.path.join(Dirs.landmarks_partial_leaves_overlay, ''.join(['batch_',str(batch+1)]))

    # if not segmentation_complete: # If segmentation was run, then don't redo the unpack, just do the crop into the temp folder
    if leaf_type == 'Landmarks_Whole_Leaves':
        Project.project_data_list[batch] = unpack_class_from_components(Project.project_data_list[batch], 0, 'Whole_Leaf_BBoxes_YOLO', 'Whole_Leaf_BBoxes', Project)
        Project.project_data_list[batch], dir_temp = crop_images_to_bbox(Project.project_data_list[batch], 0, 'Whole_Leaf_Cropped', "Whole_Leaf_BBoxes", Project, Dirs)

    elif leaf_type == 'Landmarks_Partial_Leaves':
        Project.project_data_list[batch] = unpack_class_from_components(Project.project_data_list[batch], 1, 'Partial_Leaf_BBoxes_YOLO', 'Partial_Leaf_BBoxes', Project)
        Project.project_data_list[batch], dir_temp = crop_images_to_bbox(Project.project_data_list[batch], 1, 'Partial_Leaf_Cropped', "Partial_Leaf_BBoxes", Project, Dirs)


    save_oriented_images = cfg['leafmachine']['keypoint_detector']['save_oriented_images']
    # Weights folder base
    dir_weights = os.path.join(dir_home, 'leafmachine2', 'keypoint_detector','keypoint_models')
    detector_version = cfg['leafmachine']['keypoint_detector']['detector_version']
    weights =  os.path.join(dir_weights,detector_version,'weights','best.pt')
    

    # Detection threshold
    # threshold = cfg['leafmachine']['landmark_detector']['minimum_confidence_threshold']


    do_save_prediction_overlay_images = not cfg['leafmachine']['landmark_detector']['do_save_prediction_overlay_images']

    if cfg['leafmachine']['project']['num_workers'] is None:
        num_workers = 1
    else:
        num_workers = int(cfg['leafmachine']['project']['num_workers'])

    has_images = False
    if len(os.listdir(dir_temp)) > 0:
        has_images = True
        # run(weights = weights,
        #     source = dir_temp,
        #     project = dir_overlay,
        #     name = Dirs.run_name,
        #     imgsz = (1280, 1280),
        #     nosave = do_save_prediction_overlay_images,
        #     anno_type = 'Landmark_Detector_YOLO',
        #     conf_thres = threshold, 
        #     line_thickness = 2,
        #     ignore_objects_for_overlay = ignore_objects,
        #     mode = 'Landmark')
        source = dir_temp
        project = dir_overlay
        name = Dirs.run_name
        imgsz = (1280, 1280)
        nosave = do_save_prediction_overlay_images
        anno_type = 'Landmark_Detector'
        conf_thres = threshold
        line_thickness = 2
        ignore_objects_for_overlay = ignore_objects
        mode = 'Landmark'
        LOGGER = logger





    # Create dictionary for overrides
    overrides = {
        'model': weights,
        'source': dir_temp,
        'name': detector_version, #'uniform_spaced_oriented_traces_mid15_pet5_clean_640_flipidx_pt2',
        'boxes': False,
        'max_det': 1,
        # 'visualize':True,
        # 'show':True
    }

    # Initialize PosePredictor
    pose_predictor = PosePredictor(Dirs.oriented_cropped_leaves, save_oriented_images, overrides=overrides)