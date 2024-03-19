'''
LeafMachine2 Processes
'''
import os, inspect, sys, logging, gc
from time import perf_counter
from memory_profiler import profile

currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)
from leafmachine2.component_detector.component_detector import detect_plant_components, detect_archival_components, detect_landmarks, detect_armature_components, detect_armature
from leafmachine2.segmentation.detectron2.segment_leaves import segment_leaves
# from import  # ruler classifier?
# from import  # landmarks
from leafmachine2.machine.general_utils import check_for_subdirs, get_datetime, load_config_file, load_config_file_testing, report_config, split_into_batches, save_config_file, subset_dir_images, crop_detections_from_images, crop_detections_from_images_SpecimenCrop
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
from leafmachine2.machine.fetch_data import fetch_data

# @profile
def machine(cfg_file_path, dir_home, cfg_test, progress_report=None):
    t_overall = perf_counter()

    # Load config file
    report_config(dir_home, cfg_file_path, system='LeafMachine2')
    if progress_report:
        progress_report.update_overall("Loaded config file")

    if cfg_test is None:
        cfg = load_config_file(dir_home, cfg_file_path, system='LeafMachine2')
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
        if progress_report:
            progress_report.update_overall("Create Output Directory Structure")

        # logging.info("Hi")
        logger = start_logging(Dirs, cfg)

        # Check to see if required ML files are ready to use
        ready_to_use = fetch_data(logger, dir_home, cfg_file_path)
        assert ready_to_use, "Required ML files are not ready to use!\nThe download may have failed,\nor\nthe directory structure of LM2 has been altered"
        if progress_report:
            progress_report.update_overall("Validate ML Files")

        # Wrangle images and preprocess
        print_main_start("Gathering Images and Image Metadata")
        Project = Project_Info(cfg, logger, dir_home)
        if progress_report:
            progress_report.update_overall("Create Project Storage Object")

        if not Project.has_valid_images:
            logger.error("No valid images found. Check file extensions.")
            raise FileNotFoundError(f"No valid images found in the specified directory. Invalid files may have been moved to the 'INVALID_FILES' directory.\nSupported file extensions: {Project.file_ext}")
        if progress_report:
            progress_report.update_overall("Validate Input File Names and Types")

        # Subset dir_images if selected
        Project = subset_dir_images(cfg, Project, Dirs)

        # Save config file
        save_config_file(cfg, logger, Dirs)
        if progress_report:
            progress_report.update_overall("Save Copy of Config File")


        # Detect Archival Components
        if progress_report:
            progress_report.update_overall("Detect Archival Components")
        print_main_start("Locating Archival Components")
        Project = detect_archival_components(cfg, logger, dir_home, Project, Dirs)
        
        # Detect Plant Components
        if progress_report:
            progress_report.update_overall("Detect Plant Components")
        print_main_start("Locating Plant Components")
        Project = detect_plant_components(cfg, logger, dir_home, Project, Dirs)
        # Detect Armature Components
        # progress_report.update_overall("Detect Armature Components")
        # print_main_start("Locating Armature Components")
        # Project = detect_armature_components(cfg, logger, dir_home, Project, Dirs)
        # img_crop = img_crop.resize((img_crop.width * 10, img_crop.height * 10), resample=Image.LANCZOS)

        
        # Add record data (from GBIF) to the dictionary
        logger.name = 'Project Data'
        logger.info("Adding record data (from GBIF) to the dictionary")
        Project.add_records_to_project_dict()

        # Save cropped detections
        if progress_report:
            progress_report.update_overall("Crop Individual Objects from Images")
        crop_detections_from_images(cfg, logger, dir_home, Project, Dirs)

        # If Specimen Crop is selected
        if progress_report:
            progress_report.update_overall("SpecimenCrop Images")
        crop_detections_from_images_SpecimenCrop(cfg, logger, dir_home, Project, Dirs)
        
        # Binarize labels
        run_binarize(cfg, logger, Dirs)
        if progress_report:
            progress_report.update_overall("Binarize Labels")
        
        # Split into batches for further processing
        Project, n_batches, m  = split_into_batches(Project, logger, cfg)
        if progress_report:
            progress_report.set_n_batches(n_batches)
        print_main_start(m)
        

        for batch in range(n_batches):
            if progress_report:
                progress_report.update_batch(f"Starting Batch {batch+1} of {n_batches}")

            t_batch_start = perf_counter()
            print_main_info(f'Batch {batch+1} of {n_batches}')

            if cfg['leafmachine']['do']['run_leaf_processing']:
                # Process Rulers
                if progress_report:
                    progress_report.update_batch_part(f"Processing Rulers")
                logger.name = f'[BATCH {batch+1} Convert Rulers]'
                logger.warning(f'Working on batch {batch} of {n_batches}')
                do_test_ruler_conversion = False
                if do_test_ruler_conversion:
                    dir_rulers = 'F:/Rulers_ByType_V2_target'
                    Project = convert_rulers_testing(dir_rulers, cfg, logger, dir_home, Project, batch, Dirs)
                else:
                    Project = convert_rulers(cfg, logger, dir_home, Project, batch, Dirs)
                    # Project = parallel_convert_rulers(cfg, logger, dir_home, Project, batch, Dirs)
                

                # Segment Whole Leaves
                if progress_report:
                    progress_report.update_batch_part(f"Segmenting Leaves")
                do_seg = True # Need to know if segmentation has been added to Project[batch] for landmarks
                if do_seg:
                    Project = segment_leaves(cfg, logger, dir_home, Project, batch, n_batches, Dirs)
                    

                # Landmarks Whole Leaves
                if progress_report:
                    progress_report.update_batch_part(f"Detecting Landmarks")
                Project = detect_landmarks(cfg, logger, dir_home, Project, batch, n_batches, Dirs, do_seg)


                # Landmarks Armature
                # progress_report.update_batch_part(f"Detecting Armature Landmarks")
                # Project = detect_armature(cfg, logger, dir_home, Project, batch, n_batches, Dirs, do_seg)


            # Custom Overlay 
            if progress_report:
                progress_report.update_batch_part(f"Saving Overlay Images")
            build_custom_overlay_parallel(cfg, logger, dir_home, Project, batch, Dirs)


            # Export data to csv and json
            if progress_report:
                progress_report.update_batch_part(f"Saving Data")
            save_data(cfg, logger, dir_home, Project, batch, n_batches, Dirs)


            # Clear Completed Images
            # print(Project.project_data)

            logger.info('Clearing batch data from RAM')
            Project.project_data.clear() 
            Project.project_data_list[batch] = {} 
            
            gc.collect()

            # print(Project.project_data)
            # print(Project.project_data_list[batch])
            t_batch_end = perf_counter()
            logger.name = f'[BATCH {batch+1} COMPLETE]'
            logger.info(f"[Batch {batch+1} elapsed time] {round((t_batch_end - t_batch_start)/60)} minutes")
            if progress_report:
                progress_report.reset_batch_part()

        # Create CSV from the 'Data','Project','Batches' files
        merge_csv_files(Dirs, cfg)
        
        t_overall_s = perf_counter()
        logger.name = 'Run Complete! :)'
        logger.info(f"[Total elapsed time] {round((t_overall_s - t_overall)/60)} minutes")
        # logger.info(f"[Total elapsed time] {round((t_overall_s - t_overall)/60)} minutes") # TODO add stats here
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)



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