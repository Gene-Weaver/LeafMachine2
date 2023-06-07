'''
VoucherVision - based on LeafMachine2 Processes
'''
import os, inspect, sys, logging
from time import perf_counter
currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)
from leafmachine2.component_detector.component_detector import detect_plant_components, detect_archival_components
from leafmachine2.machine.general_utils import check_for_subdirs, load_config_file, load_config_file_testing, report_config, save_config_file, subset_dir_images, crop_detections_from_images_VV
from leafmachine2.machine.general_utils import print_main_start
from leafmachine2.machine.directory_structure_VV import Dir_Structure
from leafmachine2.machine.data_project import Project_Info
from leafmachine2.machine.build_custom_overlay_VV import build_custom_overlay_parallel
from leafmachine2.machine.binarize_image_ML import run_binarize
from leafmachine2.machine.LM2_logger import start_logging
from leafmachine2.machine.fetch_data import fetch_data
from leafmachine2.transcription.run_VoucherVision import VoucherVision

def voucher_vision(cfg_file_path, dir_home, cfg_test):
    t_overall = perf_counter()

    # Load config file
    report_config(dir_home, cfg_file_path, system='VoucherVision')

    if cfg_test is None:
        cfg = load_config_file(dir_home, cfg_file_path, system='VoucherVision')  # For VoucherVision
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
        Project = Project_Info(cfg, logger, dir_home)

        # Save config file
        save_config_file(cfg, logger, Dirs)

        # Detect Archival Components
        print_main_start("Locating Archival Components")
        Project = detect_archival_components(cfg, logger, dir_home, Project, Dirs)

        # Save cropped detections
        crop_detections_from_images_VV(cfg, logger, dir_home, Project, Dirs)

        # Binarize labels
        # run_binarize(cfg, logger, Dirs)
        
        # Custom Overlay
        # build_custom_overlay_parallel(cfg, logger, dir_home, Project, 0, Dirs)

        # Process labels
        # run_voucher_vision(cfg, logger, dir_home, Project, Dirs)
        Voucher_Vision = VoucherVision(cfg, logger, dir_home, Project, Dirs)
        Voucher_Vision.process_specimen_batch()

        t_overall_s = perf_counter()
        logger.name = 'Run Complete! :)'
        logger.info(f"[Total elapsed time] {round((t_overall_s - t_overall)/60)} minutes")

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

        voucher_vision(cfg_file_path, dir_home, cfg_testing)
    else:
        cfg_file_path = None
        cfg_testing = None
        voucher_vision(cfg_file_path, dir_home, cfg_testing)