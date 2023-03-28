'''
LeafMachine2 Processes
'''
import os, inspect, sys, logging
from time import perf_counter
currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)
from leafmachine2.component_detector.component_detector import detect_plant_components, detect_archival_components, detect_landmarks
from leafmachine2.segmentation.detectron2.segment_leaves import segment_leaves
# from import  # ruler classifier?
# from import  # landmarks
from leafmachine2.machine.general_utils import get_datetime, load_config_file, report_config, split_into_batches, save_config_file, subset_dir_images, crop_detections_from_images
from leafmachine2.machine.general_utils import print_main_start, print_main_success, print_main_fail, print_main_info, make_file_names_valid, make_images_in_dir_vertical
from leafmachine2.machine.directory_structure import Dir_Structure
from leafmachine2.machine.data_project import Project_Info
from leafmachine2.machine.config import Config
from leafmachine2.machine.build_custom_overlay import build_custom_overlay, build_custom_overlay_parallel
from leafmachine2.machine.utils_ruler import convert_rulers
from leafmachine2.machine.save_data import save_data, merge_csv_files
from leafmachine2.machine.binarize_image_ML import run_binarize
from leafmachine2.machine.LM2_logger import start_logging
from leafmachine2.machine.utils_ruler import convert_rulers_testing, parallel_convert_rulers

def machine(cfg_file_path):
    t_overall = perf_counter()
    # Set LeafMachine2 dir 
    dir_home = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Load config file
    report_config(dir_home, cfg_file_path)
    cfg = load_config_file(dir_home, cfg_file_path)
    # user_cfg = load_config_file(dir_home, cfg_file_path)
    # cfg = Config(user_cfg)

    # Dir structure
    print_main_start("Creating Directory Structure")
    Dirs = Dir_Structure(cfg)

    # logging.info("Hi")
    logger = start_logging(Dirs, cfg)

    # Wrangle images and preprocess
    print_main_start("Gathering Images and Image Metadata")
    Project = Project_Info(cfg, logger, dir_home)

    # Subset dir_images if selected
    Project = subset_dir_images(cfg, Project, Dirs)

    # Save config file
    save_config_file(cfg, logger, Dirs)


    # Detect Archival Components
    print_main_start("Locating Archival Components")
    Project = detect_archival_components(cfg, logger, dir_home, Project, Dirs)

    # Detect Plant Components
    print_main_start("Locating Plant Components")
    Project = detect_plant_components(cfg, logger, dir_home, Project, Dirs)
    
    # Add record data (from GBIF) to the dictionary
    logger.name = 'Project Data'
    logger.info("Adding record data (from GBIF) to the dictionary")
    Project.add_records_to_project_dict()

    # Save cropped detections
    crop_detections_from_images(cfg, logger, dir_home, Project, Dirs)

    # Binarize labels
    run_binarize(cfg, logger, Dirs)
    
    # Split into batches for further processing
    Project, n_batches, m  = split_into_batches(Project, logger, cfg)
    print_main_start(m)
    

    for batch in range(n_batches):
        t_batch_start = perf_counter()
        print_main_info(f'Batch {batch+1} of {n_batches}')
        logger.name = f'[BATCH {batch+1} Convert Rulers]'
        logger.warning(f'Working on batch {batch} of {n_batches}')
        
        
        # Process Rulers
        do_test_ruler_conversion = False
        if do_test_ruler_conversion:
            dir_rulers = 'F:/Rulers_ByType_V2_target'
            Project = convert_rulers_testing(dir_rulers, cfg, logger, dir_home, Project, batch, Dirs)
        else:
            Project = convert_rulers(cfg, logger, dir_home, Project, batch, Dirs)
            # Project = parallel_convert_rulers(cfg, logger, dir_home, Project, batch, Dirs)


        # Segment Whole Leaves
        do_seg = True # Need to know if segmentation has been added to Project[batch] for landmarks
        if do_seg:
            Project = segment_leaves(cfg, logger, dir_home, Project, batch, n_batches, Dirs)
            

        # Landmarks Whole Leaves
        Project = detect_landmarks(cfg, logger, dir_home, Project, batch, n_batches, Dirs, do_seg)


        # Custom Overlay
        build_custom_overlay_parallel(cfg, logger, dir_home, Project, batch, Dirs)


        # Export data to csv and json
        save_data(cfg, logger, dir_home, Project, batch, n_batches, Dirs)


        # Clear Completed Images
        # print(Project.project_data)

        logger.info('Clearing batch data from RAM')
        Project.project_data.clear() 
        Project.project_data_list[batch] = {} 

        # print(Project.project_data)
        # print(Project.project_data_list[batch])
        t_batch_end = perf_counter()
        logger.name = f'[BATCH {batch+1} COMPLETE]'
        logger.info(f"[Batch {batch+1} elapsed time] {round((t_batch_end - t_batch_start)/60)} minutes")


    # Create CSV from the 'Data','Project','Batches' files
    merge_csv_files(Dirs, cfg)
    
    t_overall_s = perf_counter()
    logger.name = 'Run Complete! :)'
    logger.info(f"[Total elapsed time] {round((t_overall_s - t_overall)/60)} minutes")
    # logger.info(f"[Total elapsed time] {round((t_overall_s - t_overall)/60)} minutes") # TODO add stats here


if __name__ == '__main__':    
    # cfg_file_path = 'D:\Dropbox\LeafMachine2\LeafMachine2.yaml'
    # cfg_file_path = 'test_installation'
    cfg_file_path = None
    machine(cfg_file_path)