import os, yaml, platform

def get_default_download_folder():
    system_platform = platform.system()  # Gets the system platform, e.g., 'Linux', 'Windows', 'Darwin'

    if system_platform == "Windows":
        # Typically, the Downloads folder for Windows is in the user's profile folder
        default_output_folder = os.path.join(os.getenv('USERPROFILE'), 'Downloads')
    elif system_platform == "Darwin":
        # Typically, the Downloads folder for macOS is in the user's home directory
        default_output_folder = os.path.join(os.path.expanduser("~"), 'Downloads')
    elif system_platform == "Linux":
        # Typically, the Downloads folder for Linux is in the user's home directory
        default_output_folder = os.path.join(os.path.expanduser("~"), 'Downloads')
    else:
        default_output_folder = "set/path/to/downloads/folder"
        print("Please manually set the output folder")
    return default_output_folder

def build_LM2_config():
    dir_home = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


    # Initialize the base structure
    config_data = {
        'leafmachine': {}
    }

    # Modular sections to be added to 'leafmachine'
    do_section = {
        'check_for_illegal_filenames': True,
        'check_for_corrupt_images_make_vertical': True,
        'run_leaf_processing': True
    }

    print_section = {
        'verbose': True,
        'optional_warnings': True
    }

    logging_section = {
        'log_level': None
    }

    default_output_folder = get_default_download_folder()
    project_section = {
        'dir_output': default_output_folder,
        # 'dir_output': 'D:/D_Desktop/LM2', 
        'run_name': 'test',
        'image_location': 'local',
        'GBIF_mode': 'all',
        'batch_size': 100, #40
        'num_workers': 4, #2
        'num_workers_seg': 4,
        'num_workers_ruler': 4,
        'dir_images_local': '',
        # 'dir_images_local': 'D:\Dropbox\LM2_Env\Image_Datasets\Manuscript_Images',
        'path_combined_csv_local': None,
        'path_occurrence_csv_local': None,
        'path_images_csv_local': None,
        'use_existing_plant_component_detections': '',
        'use_existing_archival_component_detections': '',
        'process_subset_of_images': False,
        'dir_images_subset': '',
        'n_images_per_species': 10,
        'species_list': '',

        # Thresholds for counting and reporting
        'accept_only_ideal_leaves': True, # For the 'has_leaves' variable, this restricts the counts to only whole/ideal leaves
        'minimum_total_reproductive_counts': 0, # If you get false positives, increase this to set a minimum threshold for the is_fertile designation 

        'use_CF_predictor': True,
        
        'censor_archival_components': True,
        'hide_archival_components': ['ruler', 'barcode', 'label', 'colorcard', 'map', 'photo', 'weights',],
        'replacement_color': '#FFFFFF',
    }

    cropped_components_section = {
        'do_save_cropped_annotations': False,
        'save_cropped_annotations': ['label'],
        'save_per_image': False,
        'save_per_annotation_class': True,
        'binarize_labels': False,
        'binarize_labels_skeletonize': False
    }

    modules_section = {
        'armature': False,
        'specimen_crop': False
    }

    data_section = {
        'save_json_rulers': False,
        'save_json_measurements': False,
        'save_individual_csv_files_rulers': False,
        'save_individual_csv_files_measurements': False,
        'save_individual_csv_files_landmarks': False,
        'save_individual_efd_files': False,
        'include_darwin_core_data_from_combined_file': False,
        'do_apply_conversion_factor': True
    }

    overlay_section = {
        'save_overlay_to_pdf': False,
        'save_overlay_to_jpgs': True,
        'overlay_dpi': 300, # Between 100 to 300
        'overlay_background_color': 'black', # Either 'white' or 'black'

        'show_archival_detections': True,
        'show_plant_detections': True,
        'show_segmentations': True,
        'show_landmarks': True,
        'ignore_archival_detections_classes': [],
        'ignore_plant_detections_classes': ['leaf_whole',], # Could also include 'leaf_partial' and others if needed
        'ignore_landmark_classes': [],

        'line_width_archival': 12, # Previous value given was 2
        'line_width_plant': 12, # Previous value given was 6
        'line_width_seg': 12, # 12 is specified as "thick"
        'line_width_efd': 12, # 3 is specified as "thick" but 12 is given here
        'alpha_transparency_archival': 0.3,
        'alpha_transparency_plant': 0,
        'alpha_transparency_seg_whole_leaf': 0.4,
        'alpha_transparency_seg_partial_leaf': 0.3
    }

    plant_component_detector_section = {
        'detector_type': 'Plant_Detector',
        'detector_version': 'PLANT_LeafPriority',
        'detector_iteration': 'PLANT_LeafPriority',
        'detector_weights': 'LeafPriority.pt',
        'minimum_confidence_threshold': 0.5, # Default is 0.5
        'do_save_prediction_overlay_images': True,
        'ignore_objects_for_overlay': [] # 'leaf_partial' can be included if needed
    }

    archival_component_detector_section = {
        'detector_type': 'Archival_Detector',
        'detector_version': 'PREP_final',
        'detector_iteration': 'PREP_final',
        'detector_weights': 'best.pt',
        'minimum_confidence_threshold': 0.5, # Default is 0.5
        'do_save_prediction_overlay_images': True,
        'ignore_objects_for_overlay': []
    }

    armature_component_detector_section = {
        'detector_type': 'Armature_Detector',
        'detector_version': 'ARM_A_1000',
        'detector_iteration': 'ARM_A_1000',
        'detector_weights': 'best.pt',
        'minimum_confidence_threshold': 0.5, # Optionally: 0.2
        'do_save_prediction_overlay_images': True,
        'ignore_objects_for_overlay': []
    }

    landmark_detector_section = {
        'landmark_whole_leaves': True,
        'landmark_partial_leaves': False,
        'detector_type': 'Landmark_Detector_YOLO',
        'detector_version': 'Landmarks',
        'detector_iteration': 'Landmarks_V2',
        'detector_weights': 'best.pt',
        'minimum_confidence_threshold': 0.02,
        'do_save_prediction_overlay_images': True,
        'ignore_objects_for_overlay': [],
        'use_existing_landmark_detections': None, # Example path provided
        'do_show_QC_images': False,
        'do_save_QC_images': True,
        'do_show_final_images': False,
        'do_save_final_images': True
    }

    landmark_detector_armature_section = {
        'upscale_factor': 10,
        'detector_type': 'Landmark_Detector_YOLO',
        'detector_version': 'Landmarks_Arm_A_200',
        'detector_iteration': 'Landmarks_Arm_A_200',
        'detector_weights': 'last.pt',
        'minimum_confidence_threshold': 0.06,
        'do_save_prediction_overlay_images': True,
        'ignore_objects_for_overlay': [],
        'use_existing_landmark_detections': None, # Example path provided
        'do_show_QC_images': True,
        'do_save_QC_images': True,
        'do_show_final_images': True,
        'do_save_final_images': True
    }

    ruler_detection_section = {
        'detect_ruler_type': True,
        'ruler_detector': 'ruler_classifier_38classes_v-1.pt',
        'ruler_binary_detector': 'model_scripted_resnet_720_withCompression.pt',
        'minimum_confidence_threshold': 0.5,
        'save_ruler_validation': False,
        'save_ruler_validation_summary': True,
        'save_ruler_processed': False
    }

    leaf_segmentation_section = {
        'segment_whole_leaves': True,
        'segment_partial_leaves': False,

        'keep_only_best_one_leaf_one_petiole': True,

        'save_segmentation_overlay_images_to_pdf': True,
        'save_each_segmentation_overlay_image': True,
        'save_individual_overlay_images': True, # Not recommended due to potential file count
        'overlay_line_width': 1, # Default is 1

        'use_efds_for_png_masks': False, # Requires calculate_elliptic_fourier_descriptors to be True
        'save_masks_color': True,
        'save_full_image_masks_color': True,
        'save_rgb_cropped_images': True,

        'find_minimum_bounding_box': True,

        'calculate_elliptic_fourier_descriptors': True, # Default is True
        'elliptic_fourier_descriptor_order': 40, # Default is 40

        'segmentation_model': 'Group3_Dataset_100000_Iter_1176PTS_512Batch_smooth_l1_LR00025_BGR', #'GroupB_Dataset_100000_Iter_1176PTS_512Batch_smooth_l1_LR00025_BGR',
        'minimum_confidence_threshold': 0.7, # Alternatively: 0.9
        'generate_overlay': True,
        'overlay_dpi': 300, # Range: 100 to 300
        'overlay_background_color': 'black', # Options: 'white' or 'black',

        'save_oriented_images': True,
        'save_keypoint_overlay': True,
        'save_oriented_mask': True,
        'save_simple_txt': True,
        'detector_version': 'uniform_spaced_oriented_traces_mid15_pet5_clean_640_flipidx_pt2'
    }

    # Add the sections to the 'leafmachine' key
    config_data['leafmachine']['do'] = do_section
    config_data['leafmachine']['print'] = print_section
    config_data['leafmachine']['logging'] = logging_section
    config_data['leafmachine']['project'] = project_section
    config_data['leafmachine']['cropped_components'] = cropped_components_section
    config_data['leafmachine']['modules'] = modules_section
    config_data['leafmachine']['data'] = data_section
    config_data['leafmachine']['overlay'] = overlay_section
    config_data['leafmachine']['plant_component_detector'] = plant_component_detector_section
    config_data['leafmachine']['archival_component_detector'] = archival_component_detector_section
    config_data['leafmachine']['armature_component_detector'] = armature_component_detector_section
    config_data['leafmachine']['landmark_detector'] = landmark_detector_section
    config_data['leafmachine']['landmark_detector_armature'] = landmark_detector_armature_section
    config_data['leafmachine']['ruler_detection'] = ruler_detection_section
    config_data['leafmachine']['leaf_segmentation'] = leaf_segmentation_section

    return config_data, dir_home

def write_config_file(config_data, dir_home, filename="LeafMachine2.yaml"):
    file_path = os.path.join(dir_home, filename)

    # Write the data to a YAML file
    with open(file_path, "w") as outfile:
        yaml.dump(config_data, outfile, default_flow_style=False)

if __name__ == '__main__':
    config_data, dir_home = build_LM2_config()
    write_config_file(config_data, dir_home)

