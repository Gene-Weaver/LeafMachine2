'''
LeafMachine2 Processes
'''
import os, inspect, sys, logging
from time import perf_counter
currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)
from leafmachine2.component_detector.component_detector import detect_plant_components, detect_archival_components
# from leafmachine2.segmentation.detectron2.segment_leaves import segment_leaves
# from import  # ruler classifier?
# from import  # landmarks
from leafmachine2.machine.general_utils import check_for_subdirs, get_datetime, load_config_file, load_config_file_testing, report_config, split_into_batches, save_config_file, crop_detections_from_images
from leafmachine2.machine.general_utils import print_main_start, print_main_success, print_main_fail, print_main_info, make_file_names_valid, make_images_in_dir_vertical
from leafmachine2.machine.directory_structure_R import Dir_Structure
from leafmachine2.machine.data_project import Project_Info
from leafmachine2.machine.config import Config
from leafmachine2.machine.build_custom_overlay import build_custom_overlay, build_custom_overlay_parallel
from leafmachine2.machine.save_data import save_data, merge_csv_files
from leafmachine2.machine.LM2_logger import start_logging
from leafmachine2.machine.fetch_data import fetch_data
from leafmachine2.machine.handle_images import check_image_compliance
from leafmachine2.machine.utils_censor_components import censor_archival_components

def machine(cfg_file_path, dir_home, cfg_test):
    time_report = {}
    t_overall = perf_counter()

    # Load config file
    report_config(dir_home, cfg_file_path, system='CensorArchivalComponents')

    if cfg_test is None:
        cfg = load_config_file(dir_home, cfg_file_path, system='CensorArchivalComponents')
    else:
        cfg = cfg_test 
    # user_cfg = load_config_file(dir_home, cfg_file_path)
    # cfg = Config(user_cfg)

    # Check to see if there are subdirs
    # Yes --> use the names of the subsirs as run_name
    run_name, dirs_list, has_subdirs = check_for_subdirs(cfg)

    for dir_ind, dir_in in enumerate(dirs_list):

        if has_subdirs:
            cfg['leafmachine']['project']['dir_images_local'] = dir_in
            cfg['leafmachine']['project']['run_name'] = run_name[dir_ind]


        # Dir structure
        print_main_start("Creating Directory Structure")
        Dirs = Dir_Structure(cfg)


        # logging.info("Hi")
        logger = start_logging(Dirs, cfg)


        # Check to see if required ML files are ready to use
        ready_to_use = fetch_data(logger, dir_home, cfg_file_path)
        assert ready_to_use, "Required ML files are not ready to use!\nThe download may have failed,\nor\nthe directory structure of LM2 has been altered"


        # Wrangle images and preprocess
        print_main_start("Gathering Images and Image Metadata")


        # Check image dir for correct extensions
        # cfg, original_img_dir, new_tiff_dir = check_image_compliance(cfg, Dirs)
        # if original_img_dir is None, then the images are already jpgs or pngs
        cfg, original_img_dir = check_image_compliance(cfg, Dirs)


        # Create Project
        Project = Project_Info(cfg, logger, dir_home)


        # Save config file
        save_config_file(cfg, logger, Dirs)


        # Detect Archival Components
        print_main_start("Locating Archival Components")
        Project, time_report = detect_archival_components(cfg, time_report, logger, dir_home, Project, Dirs)

        # Detect Plant Components
        # print_main_start("Locating Plant Components")
        # Project, time_report = detect_plant_components(cfg, time_report, logger, dir_home, Project, Dirs)


        # Save cropped detections
        time_report = crop_detections_from_images(cfg, time_report, logger, dir_home, Project, Dirs)

        # If remove archival components is selected
        time_report = censor_archival_components(cfg, time_report, logger, dir_home, Project, Dirs)

        
        if cfg['leafmachine']['overlay']['save_overlay_to_jpgs'] or cfg['leafmachine']['overlay']['save_overlay_to_pdf']:
            # Split into batches for further processing
            Project, n_batches, m  = split_into_batches(Project, logger, cfg)
            print_main_start(m)
            

            for batch in range(n_batches):
                t_batch_start = perf_counter()
                print_main_info(f'Batch {batch+1} of {n_batches}')

                # Custom Overlay
                time_report = build_custom_overlay_parallel(cfg, time_report, logger, dir_home, Project, batch, Dirs)

                logger.info('Clearing batch data from RAM')
                Project.project_data.clear() 
                Project.project_data_list[batch] = {} 

                # print(Project.project_data)
                # print(Project.project_data_list[batch])
                t_batch_end = perf_counter()
                logger.name = f'[BATCH {batch+1} COMPLETE]'
                logger.info(f"[Batch {batch+1} elapsed time] {round((t_batch_end - t_batch_start)/60)} minutes")

        
        t_overall_s = perf_counter()
        t_overall = f"[Total Project elapsed time] {round((t_overall_s - t_overall)/60)} minutes"
        time_report['overall'] = t_overall
        logger.name = 'Run Complete! :)'
        for opt, val in time_report.items():
            logger.info(f"{val}")
        # logger.info(f"[Total elapsed time] {round((t_overall_s - t_overall)/60)} minutes") # TODO add stats here


if __name__ == '__main__':    
    is_test = False

    # Set LeafMachine2 dir 
    dir_home = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    if is_test:
        cfg_file_path = os.path.join(dir_home, 'demo','demo.yaml') #'D:\Dropbox\LeafMachine2\LeafMachine2.yaml'
        # cfg_file_path = 'test_installation'

        cfg_testing = load_config_file_testing(dir_home, cfg_file_path)
        cfg_testing['leafmachine']['project']['dir_images_local'] = os.path.join(dir_home, cfg_testing['leafmachine']['project']['dir_images_local'][0], cfg_testing['leafmachine']['project']['dir_images_local'][1])
        cfg_testing['leafmachine']['project']['dir_output'] = os.path.join(dir_home, cfg_testing['leafmachine']['project']['dir_output'][0], cfg_testing['leafmachine']['project']['dir_output'][1])

        machine(cfg_file_path, dir_home, cfg_testing)
    else:
        cfg_file_path = None
        cfg_testing = None
        machine(cfg_file_path, dir_home, cfg_testing)