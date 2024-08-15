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
from leafmachine2.machine.data_project_sql import Project_Info_SQL, get_database_path, test_sql
from leafmachine2.machine.config import Config
from leafmachine2.machine.build_custom_overlay import build_custom_overlay, build_custom_overlay_parallel
from leafmachine2.machine.utils_ruler import convert_rulers
from leafmachine2.machine.save_data import save_data, merge_csv_files
from leafmachine2.machine.binarize_image_ML import run_binarize
from leafmachine2.machine.LM2_logger import start_logging
from leafmachine2.machine.utils_ruler import convert_rulers_testing #, parallel_convert_rulers
from leafmachine2.machine.fetch_data import fetch_data
from leafmachine2.machine.utils_detect_phenology import detect_phenology
from leafmachine2.machine.utils_censor_components import censor_archival_components

# @profile
def machine(cfg_file_path, dir_home, cfg_test, progress_report=None):
    time_report = {}
    t_overall = perf_counter()

    # Load config file
    report_config(dir_home, cfg_file_path, system='LeafMachine2')
    if progress_report:
        progress_report.update_overall("Loaded config file") # Step 1/13

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
            progress_report.update_overall("Create Output Directory Structure") # Step 2/13

        # logging.info("Hi")
        logger = start_logging(Dirs, cfg)

        # Check to see if required ML files are ready to use
        ready_to_use = fetch_data(logger, dir_home, cfg_file_path)
        assert ready_to_use, "Required ML files are not ready to use!\nThe download may have failed,\nor\nthe directory structure of LM2 has been altered"
        if progress_report:
            progress_report.update_overall("Validate ML Files") # Step 3/13

        # Wrangle images and preprocess
        print_main_start("Gathering Images and Image Metadata")
        Project = Project_Info(cfg, logger, dir_home)
        # ProjectSQL = Project_Info_SQL(cfg, logger, dir_home, Dirs)
        # test_sql(get_database_path(ProjectSQL))

        if progress_report:
            progress_report.update_overall("Created Project Storage Object") # Step 4/13

        # if not Project.has_valid_images:
        # if not ProjectSQL.has_valid_images:
        #     logger.error("No valid images found. Check file extensions.")
        #     raise FileNotFoundError(f"No valid images found in the specified directory. Invalid files may have been moved to the 'INVALID_FILES' directory.\nSupported file extensions: {Project.file_ext}")
        if progress_report:
            progress_report.update_overall("Validate Input File Names and Types")  # Step 5/13

        # Subset dir_images if selected
        # Project = subset_dir_images(cfg, Project, Dirs) # Not used. Need to make a sql version 

        # Save config file
        save_config_file(cfg, logger, Dirs)
        if progress_report:
            progress_report.update_overall("Save Copy of Config File") # Step 6/13


        # Detect Archival Components
        if progress_report:
            progress_report.update_overall("Detect Archival Components") # Step 7/13
        print_main_start("Locating Archival Components")
        Project, time_report = detect_archival_components(cfg, time_report, logger, dir_home, Project, Dirs)
        # ProjectSQL, time_report = detect_archival_components(cfg, time_report, logger, dir_home, ProjectSQL, Dirs)
        # test_sql(get_database_path(ProjectSQL))
        
        # Detect Plant Components
        if progress_report:
            progress_report.update_overall("Detect Plant Components") # Step 8/13
        print_main_start("Locating Plant Components")
        Project, time_report = detect_plant_components(cfg, time_report, logger, dir_home, Project, Dirs)
        # ProjectSQL, time_report = detect_plant_components(cfg, time_report, logger, dir_home, ProjectSQL, Dirs)
        # test_sql(get_database_path(ProjectSQL))

        # Detect Armature Components
        # progress_report.update_overall("Detect Armature Components")  # Step 
        # print_main_start("Locating Armature Components")
        # Project = detect_armature_components(cfg, logger, dir_home, Project, Dirs)
        # img_crop = img_crop.resize((img_crop.width * 10, img_crop.height * 10), resample=Image.LANCZOS)

        
        # Add record data (from GBIF) to the dictionary
        # Turned off as of 5/22/2024. Could be updated though
        # logger.name = 'Project Data'
        # logger.info("Adding record data (from GBIF) to the dictionary")
        # Project.add_records_to_project_dict()

        # Save cropped detections
        if progress_report:
            progress_report.update_overall("Crop Individual Objects from Images") # Step 9/13
        time_report = crop_detections_from_images(cfg, time_report, logger, dir_home, Project, Dirs)
        # time_report = crop_detections_from_images(cfg, time_report, logger, dir_home, ProjectSQL, Dirs)

        # If Specimen Crop is selected
        if progress_report:
            progress_report.update_overall("SpecimenCrop Images") # Step 10/13
        time_report = crop_detections_from_images_SpecimenCrop(cfg, time_report, logger, dir_home, Project, Dirs)

        if progress_report:
            progress_report.update_overall("Detecting Phenology") # Step 11/13
        time_report = detect_phenology(cfg, time_report, logger, dir_home, Project, Dirs)

        if progress_report:
            progress_report.update_overall("Censoring Archival Components") # Step 12/13
        time_report = censor_archival_components(cfg, time_report, logger, dir_home, Project, Dirs)

        
        # Binarize labels
        run_binarize(cfg, logger, Dirs)
        if progress_report:
            progress_report.update_overall("Binarize Labels") # Step 13/13
        
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
                    Project, time_report = convert_rulers_testing(dir_rulers, cfg, time_report, logger, dir_home, Project, batch, Dirs)
                else:
                    Project, time_report = convert_rulers(cfg, time_report, logger, dir_home, Project, batch, Dirs)
                    # Project = parallel_convert_rulers(cfg, logger, dir_home, Project, batch, Dirs)
                

                # Segment Whole Leaves
                if progress_report:
                    progress_report.update_batch_part(f"Segmenting Leaves")
                do_seg = True # Need to know if segmentation has been added to Project[batch] for landmarks
                if do_seg:
                    Project, time_report = segment_leaves(cfg, time_report, logger, dir_home, Project, batch, n_batches, Dirs)
                    

                # Landmarks Whole Leaves
                if progress_report:
                    progress_report.update_batch_part(f"Detecting Landmarks")
                Project, time_report = detect_landmarks(cfg, time_report, logger, dir_home, Project, batch, n_batches, Dirs, do_seg)



                # # Landmarks Whole Leaves
                # if progress_report:
                #     progress_report.update_batch_part(f"Detecting Landmarks - Keypoint")
                # Project = detect_landmarks_keypoint(cfg, logger, dir_home, Project, batch, n_batches, Dirs, do_seg)



                # Landmarks Armature
                # progress_report.update_batch_part(f"Detecting Armature Landmarks")
                # Project = detect_armature(cfg, logger, dir_home, Project, batch, n_batches, Dirs, do_seg)


            # Custom Overlay 
            if progress_report:
                progress_report.update_batch_part(f"Saving Overlay Images")
            time_report = build_custom_overlay_parallel(cfg, time_report, logger, dir_home, Project, batch, Dirs)


            # Export data to csv and json
            if progress_report:
                progress_report.update_batch_part(f"Saving Data")
            time_report = save_data(cfg, time_report, logger, dir_home, Project, batch, n_batches, Dirs)


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

            t_batch = f"[Batch {batch+1} total elapsed time] {round(t_batch_end - t_batch_start)} seconds ({round((t_batch_end - t_batch_start)/60)} minutes)"
            logger.info(t_batch)
            time_report[f't_batch_{batch+1}'] = t_batch

            if progress_report:
                progress_report.reset_batch_part()

        # Create CSV from the 'Data','Project','Batches' files
        merge_csv_files(Dirs, cfg)
        
        t_overall_s = perf_counter()
        t_overall = f"[Total Project elapsed time] {round((t_overall_s - t_overall)/60)} minutes"
        time_report['overall'] = t_overall
        logger.name = 'Run Complete! :)'
        for opt, val in time_report.items():
            logger.info(f"{val}")
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