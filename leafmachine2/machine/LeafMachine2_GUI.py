import streamlit as st
import yaml, os, gc
from PIL import Image
from leafmachine2.machine.LeafMachine2_Config_Builder import build_LM2_config, write_config_file
from leafmachine2.machine.machine import machine
from leafmachine2.machine.general_utils import load_config_file_testing

def save_yaml(data, yaml_file_path):
    with open(yaml_file_path, 'w') as f:
        yaml.dump(data, f)

def load_yaml(yaml_file_path):
    if os.path.exists(yaml_file_path):
        with open(yaml_file_path, 'r') as f:
            return yaml.safe_load(f)
    return None  # Return None if the file doesn't exist

def get_nested(d, *keys):
    try:
        for key in keys:
            d = d[key]
        return d
    except (TypeError, KeyError):
        return None
    
class ProgressReport:
    def __init__(self, overall_bar, batch_bar, batch_part_bar, text_overall, text_batch, text_batch_part, total_batches=10):
        self.overall_bar = overall_bar
        self.batch_bar = batch_bar
        self.batch_part_bar = batch_part_bar
        self.text_overall = text_overall
        self.text_batch = text_batch
        self.text_batch_part = text_batch_part
        self.current_overall_step = 0
        self.total_overall_steps = 13  # number of major steps in machine function
        self.current_batch = 0
        self.total_batches = total_batches
        self.current_batch_part = 0
        self.total_batch_part_steps = 5  # number of major steps in machine function

    def update_overall(self, step_name=""):
        self.current_overall_step += 1
        self.overall_bar.progress(self.current_overall_step / self.total_overall_steps)
        self.text_overall.text(step_name)

    def update_batch(self, step_name=""):
        self.current_batch += 1
        self.batch_bar.progress(self.current_batch / self.total_batches)
        self.text_batch.text(step_name)

    def update_batch_part(self, step_name=""):
        self.current_batch_part += 1
        self.batch_part_bar.progress(self.current_batch_part / self.total_batch_part_steps)
        self.text_batch_part.text(step_name)

    def set_n_batches(self, n_batches):
        self.total_batches = n_batches

    def reset_batch_part(self):
        self.current_batch_part = 0

# Function to show an image given its path
def show_photo(photo_path,paths_images):
    image = Image.open(photo_path)
    st.image(image, caption=photo_path)
    st.write(f"Image Index: {st.session_state.counter + 1}/{len(paths_images)}")


st.set_page_config(layout="wide", page_icon='img/icon.ico', page_title='LeafMachine2')

# Default YAML file path
if 'config' not in st.session_state:
    st.session_state.config, st.session_state.dir_home = build_LM2_config()
# yaml_file_path = "LeafMachine2.yaml"

# Main App
st.title("LeafMachine2")

# Load existing YAML if it exists
# config = load_yaml(yaml_file_path)
# if config is None:
    # config = {}  # Initialize an empty dict to avoid KeyError later

# Define the config structure if it doesn't exist
# config.setdefault('leafmachine', {'do': {}, 'print': {}, 'project': {}, 'modules': {}, 'logging': {},  'data': {}
#                                   ,'cropped_components': {},'data': {},'overlay': {},'plant_component_detector': {},
#                                   'archival_component_detector': {},'armature_component_detector': {},'landmark_detector': {},
#                                   'landmark_detector_armature': {},'ruler_detection': {},'leaf_segmentation': {},
#                                   'images_to_process': {},'component_detector_train': {},'ruler_train': {},})


col_run_1, col_run_2 = st.columns([4,2])
st.write("")
st.write("")
st.write("")
st.write("")
st.header("Progress")
col_run_info_1 = st.columns([1])[0]
st.write("")
st.write("")
st.write("")
st.write("")
st.header("Configuration Settings")

with col_run_info_1:
    # Progress
    st.subheader('Setting Up LM2 Project')
    overall_progress_bar = st.progress(0)
    text_overall = st.empty()  # Placeholder for current step name
    st.subheader('Overall Batch Progress')
    batch_progress_bar = st.progress(0)
    text_batch = st.empty()  # Placeholder for current step name
    st.subheader('Processing Components in Batch')
    batch_part_bar = st.progress(0)
    text_batch_part = st.empty()  # Placeholder for current step name
    progress_report = ProgressReport(overall_progress_bar, batch_progress_bar, batch_part_bar, text_overall, text_batch, text_batch_part)
    
with col_run_1:
    st.subheader('Run LeafMachine2')
    if st.button("Start Processing", type='primary'):
        # First, write the config file.
        write_config_file(st.session_state.config, st.session_state.dir_home)
        
        # Call the machine function.
        machine(None, st.session_state.dir_home, None, progress_report)
        gc.collect()
        st.balloons()

with col_run_2:
    st.subheader('Test LM2')
    if st.button("Test"):
        # First, write the config file.
        write_config_file(st.session_state.config, st.session_state.dir_home)
        # Call the machine function.
        cfg_file_path = os.path.join(st.session_state.dir_home, 'demo','demo.yaml') 

        cfg_testing = load_config_file_testing(st.session_state.dir_home, cfg_file_path)
        cfg_testing['leafmachine']['project']['dir_images_local'] = os.path.join(st.session_state.dir_home, cfg_testing['leafmachine']['project']['dir_images_local'][0], cfg_testing['leafmachine']['project']['dir_images_local'][1])
        cfg_testing['leafmachine']['project']['dir_output'] = os.path.join(st.session_state.dir_home, cfg_testing['leafmachine']['project']['dir_output'][0], cfg_testing['leafmachine']['project']['dir_output'][1])

        machine(cfg_file_path, st.session_state.dir_home, cfg_testing, progress_report)
        gc.collect()
        st.balloons()


# Adding 2 Page Tabs
tab_settings, tab_component, tab_landmark, tab_segmentation, tab_phenology, tab_overlay, tab_training = st.tabs(["Settings", "Component Detector","Landmark Detector","Segmentation", "Phenology", "Image Overlay", "Training Settings"])



# Tab 1: General Settings
with tab_settings:
    st.header('Project')
    col_project_1, col_project_2 = st.columns([4,2])

    st.write("---")
    st.header('Input Images')
    col_input_1, col_input_2 = st.columns([4,2])

    # st.markdown('Process images stored locally')
    col_local_1, col_local_2 = st.columns([4,2])

    # st.markdown('Process images from GBIF')
    col_GBIF_1, col_GBIF_2 = st.columns([4,2])

    st.write("---")
    st.header('Processing Options')
    col_processing_1, _, col_processing_2,col_processing_3 = st.columns([1,2,3,3])

    st.write("---")
    
    col_m1, col_m2 = st.columns(2)

    st.write("---")
    st.header('Cropped Components')   
    st.markdown("This will additionally crop and save the defined components as their own images. For example, selecting 'label' will save every instance of a label as .jpg using the parent image's name. ") 
    st.markdown("The options `Binarize Labels` and `Binarize and skeletonize labels` are generally not required. The Python packages required for binarization may not work correctly with all PCs, but generally work with newer Nvidia GPUs.")
    col_cropped_1, col_cropped_2 = st.columns([2,4])

    st.write("---")
    st.header('Logging and Image Validation')    
    st.markdown("We recommend leaving the default values (all enabled). We require file names to not have special characters and for the image to be in portrait orientation, these options will enable corrections.")
    col_v1, col_v2 = st.columns(2)

    os.path.join(st.session_state.dir_home, )
    ### Project
    with col_project_1:
        st.session_state.config['leafmachine']['project']['run_name'] = st.text_input("Run name", st.session_state.config['leafmachine']['project'].get('run_name', ''))
        st.session_state.config['leafmachine']['project']['dir_output'] = st.text_input("Output directory", st.session_state.config['leafmachine']['project'].get('dir_output', ''))
        # st.session_state.config_alt_path = st.text_input("Load existing config file (provide full path to file)", 'path/to/previous/LeafMachine2.yaml')
        # if st.button('Apply Existing Config'):
        #     ___, st.session_state.dir_home = build_LM2_config()
        #     config_alt = load_yaml(st.session_state.config_alt_path)
        #     if config_alt is not None:
        #         st.session_state.config = config_alt
        #         st.success(':white_check_mark: Config loaded and applied!')
        #         st.rerun()
        #     else:
        #         st.error(':no_entry: Failed to load config :no_entry:')
        # if st.button('Reset Config to Default'):
        #     st.session_state.config, st.session_state.dir_home = build_LM2_config()

    # with col_project_2:
    #     dir_image_out = os.path.join(st.session_state.config['leafmachine']['project']['dir_output'],st.session_state.config['leafmachine']['project']['run_name'],'Summary','Custom_Overlay_Images')
    #     # Initialize the counter in session state to keep track of the current image
    #     if 'counter' not in st.session_state:
    #         st.session_state.counter = 0

    #     # Get list of images in folder
    #     has_images = False
    #     try:
    #         paths_images = [os.path.join(dir_image_out, f) for f in os.listdir(dir_image_out) if os.path.isfile(os.path.join(dir_image_out, f))]
    #         has_images = True
    #     except:
    #         pass

    #     # Check if there are images in the directory
    #     if not has_images:
    #         st.write("")
    #     else:
    #         # Increment and Decrement the counter with 'Next' and 'Previous' buttons
    #         if st.button("Previous"):
    #             st.session_state.counter -= 1  # Decrement the counter
    #             if st.session_state.counter < 0:
    #                 st.session_state.counter = len(paths_images) - 1  # Loop back to the last image

    #         if st.button("Next"):
    #             st.session_state.counter += 1  # Increment the counter
    #             if st.session_state.counter >= len(paths_images):
    #                 st.session_state.counter = 0  # Loop back to the first image

    #         show_photo(paths_images[st.session_state.counter])
            
    ### Input Images
    with col_input_1:
        # st.session_state.config['leafmachine']['project']['image_location'] = st.radio("Image location", ["local", "GBIF"], index=["local", "GBIF"].index(st.session_state.config['leafmachine']['project'].get('image_location', 'local')))
        st.session_state.config['leafmachine']['project']['image_location'] = "local"

    ### Input Images Local
    with col_local_1:
        st.session_state.config['leafmachine']['project']['dir_images_local'] = st.text_input("Full path to images", st.session_state.config['leafmachine']['project'].get('dir_images_local', ''))

    ### Input Images GBIF
    # with col_GBIF_1:
        # st.session_state.config['leafmachine']['project']['GBIF_mode'] = st.selectbox("GBIF mode", ["all", "filter"], index=["all", "filter"].index(st.session_state.config['leafmachine']['project'].get('GBIF_mode', 'all')))
        st.session_state.config['leafmachine']['project']['GBIF_mode'] = "all"

    with col_GBIF_1:
        use_existing_plant_component_detections = st.text_input("Use existing plant component detections (full path to existing `.../Plant_Components/labels` folder)", 
                                                                placeholder="Optional", 
                                                                value=st.session_state.config['leafmachine']['project'].get('use_existing_plant_component_detections', ''))
        use_existing_archival_component_detections = st.text_input("Use existing archival component detections (full path to existing `.../Archival_Components/labels` folder)", 
                                                                placeholder="Optional",
                                                                value=st.session_state.config['leafmachine']['project'].get('use_existing_archival_component_detections', ''))
        
        if use_existing_plant_component_detections == '':
            st.session_state.config['leafmachine']['project']['use_existing_plant_component_detections']  = None
        else:
            st.session_state.config['leafmachine']['project']['use_existing_plant_component_detections'] = use_existing_plant_component_detections
        
        if use_existing_archival_component_detections == '':
            st.session_state.config['leafmachine']['project']['use_existing_archival_component_detections']  = None
        else:
            st.session_state.config['leafmachine']['project']['use_existing_archival_component_detections'] = use_existing_archival_component_detections

    ### Processing Options
    with col_processing_1:
        st.subheader('Compute Options')
        st.session_state.config['leafmachine']['project']['num_workers'] = st.number_input("Number of CPU workers for ACD/PCD, general", value=st.session_state.config['leafmachine']['project'].get('num_workers', 4),help="Start with 4. GPU and CPU limited.")
        st.session_state.config['leafmachine']['project']['num_workers_seg'] = st.number_input("Number of CPU workers for leaf segmentation", value=st.session_state.config['leafmachine']['project'].get('num_workers_seg', 4),help="Start with 4. GPU and CPU limited.")
        st.session_state.config['leafmachine']['project']['num_workers_ruler'] = st.number_input("Number of CPU workers for ruler conversion", value=st.session_state.config['leafmachine']['project'].get('num_workers_ruler', 4),help="Start with 4. CPU limited.")
        st.session_state.config['leafmachine']['project']['batch_size'] = st.number_input("Batch size", value=st.session_state.config['leafmachine']['project'].get('batch_size', 2),help="Determines how many images are processed at a time. These are stored in RAM, so start with 25 or 50 and verify performance.")

    with col_processing_2:
        st.subheader('Optional CSV Files')
        st.session_state.config['leafmachine']['data']['save_individual_csv_files_rulers'] = st.checkbox("Save ruler CSV files per image", st.session_state.config['leafmachine']['data'].get('save_individual_csv_files_rulers', False))
        st.session_state.config['leafmachine']['data']['save_individual_csv_files_measurements'] = st.checkbox("Save measurement CSV files per image", st.session_state.config['leafmachine']['data'].get('save_individual_csv_files_measurements', False))
        st.session_state.config['leafmachine']['data']['save_individual_csv_files_landmarks'] = st.checkbox("Save landmark CSV files per image", st.session_state.config['leafmachine']['data'].get('save_individual_csv_files_landmarks', False))
        st.session_state.config['leafmachine']['data']['save_individual_efd_files'] = st.checkbox("Save EFD CSV files per image", st.session_state.config['leafmachine']['data'].get('save_individual_efd_files', False))


    with col_processing_3:
        st.subheader('Misc')
        st.session_state.config['leafmachine']['data']['include_darwin_core_data_from_combined_file'] = st.checkbox("Attach Darwin Core information to combined output file", st.session_state.config['leafmachine']['data'].get('include_darwin_core_data_from_combined_file', False),disabled=True)
        st.session_state.config['leafmachine']['data']['save_json_rulers'] = st.checkbox("Save ruler data to JSON files", st.session_state.config['leafmachine']['data'].get('save_json_rulers', False),disabled=True)
        st.session_state.config['leafmachine']['data']['save_json_measurements'] = st.checkbox("Save measurements to JSON files", st.session_state.config['leafmachine']['data'].get('save_json_measurements', False),disabled=True)
        

    ### Modules
    with col_m1:
        st.header('Modules')
        st.markdown("If you only need to process the Archival Components, you can disable the `Process leaves` option.")
        st.session_state.config['leafmachine']['do']['run_leaf_processing'] = st.checkbox("Process leaves", st.session_state.config['leafmachine']['do'].get('run_leaf_processing', True))
        st.session_state.config['leafmachine']['modules']['armature'] = st.checkbox("Armature", st.session_state.config['leafmachine']['modules'].get('armature', False),disabled=True)
        st.session_state.config['leafmachine']['modules']['specimen_crop'] = st.checkbox("Specimen Close-up", st.session_state.config['leafmachine']['modules'].get('specimen_crop', False),disabled=True)
        
        default_components = ['ruler', 'barcode', 'label', 'colorcard', 'map', 'photo', 'weights',]
        default_color = '#FFFFFF'  # White color

        st.header('Create images with hidden archival components')
        st.session_state.config['leafmachine']['project']['censor_archival_components'] = st.checkbox("Hide archival components", st.session_state.config['leafmachine']['project'].get('censor_archival_components', True))
        st.session_state.config['leafmachine']['project']['hide_archival_components'] = st.multiselect("Components to hide",  
                ['ruler','barcode','label','colorcard','map','envelope','photo','attached_item','weights',],
                default=default_components)
        # Color picker input for selecting the hide color
        st.session_state.config['leafmachine']['project']['replacement_color'] = st.color_picker(
            "Select replacement color",
            value=default_color
        )

    ### Predictive Ruler
    with col_m2:
        st.header('Predict Conversion Factor')
        st.markdown("If you are processing regular herbarium images, leave this turned on. If you are processing custom images, try it both ways.")
        st.markdown("This will use the image's resolution to predict the conversoin factor. It significantly helps LM2's algorithms calculate the true image-specific conversion factor. When enabled, the predicted value is calculated for each image and reported even when the algorithmic determination fails or when rulers are not present in the image.")
        st.session_state.config['leafmachine']['project']['use_CF_predictor'] = st.checkbox("Use CF predictor", st.session_state.config['leafmachine']['project'].get('use_CF_predictor', True),disabled=False)


    ### cropped_components
    with col_cropped_1:
        st.session_state.config['leafmachine']['cropped_components']['do_save_cropped_annotations'] = st.checkbox("Save cropped components as images", st.session_state.config['leafmachine']['cropped_components'].get('do_save_cropped_annotations', True))
        st.session_state.config['leafmachine']['cropped_components']['save_per_image'] = st.checkbox("Save cropped components grouped by specimen", st.session_state.config['leafmachine']['cropped_components'].get('save_per_image', False))
        st.session_state.config['leafmachine']['cropped_components']['save_per_annotation_class'] = st.checkbox("Save cropped components grouped by type", st.session_state.config['leafmachine']['cropped_components'].get('save_per_annotation_class', True))
        st.session_state.config['leafmachine']['cropped_components']['binarize_labels'] = st.checkbox("Binarize labels", st.session_state.config['leafmachine']['cropped_components'].get('binarize_labels', False))
        st.session_state.config['leafmachine']['cropped_components']['binarize_labels_skeletonize'] = st.checkbox("Binarize and skeletonize labels", st.session_state.config['leafmachine']['cropped_components'].get('binarize_labels_skeletonize', False))
    
    
    with col_cropped_2:
        default_crops = st.session_state.config['leafmachine']['cropped_components'].get('save_cropped_annotations', ['leaf_whole'])
        st.session_state.config['leafmachine']['cropped_components']['save_cropped_annotations'] = st.multiselect("Components to crop",  
                ['ruler', 'barcode','label', 'colorcard','map','envelope','photo','attached_item','weights',
                'leaf_whole', 'leaf_partial', 'leaflet', 'seed_fruit_one', 'seed_fruit_many', 'flower_one', 'flower_many', 'bud','specimen','roots','wood'],default=default_crops)
    ### Logging and Image Validation - col_v1
    with col_v1:
        st.session_state.config['leafmachine']['do']['check_for_illegal_filenames'] = st.checkbox("Check for illegal filenames", st.session_state.config['leafmachine']['do'].get('check_for_illegal_filenames', True))
        st.session_state.config['leafmachine']['do']['check_for_corrupt_images_make_vertical'] = st.checkbox("Check for corrupt images", st.session_state.config['leafmachine']['do'].get('check_for_corrupt_images_make_vertical', True))
        
        st.session_state.config['leafmachine']['print']['verbose'] = st.checkbox("Print verbose", st.session_state.config['leafmachine']['print'].get('verbose', True))
        st.session_state.config['leafmachine']['print']['optional_warnings'] = st.checkbox("Show optional warnings", st.session_state.config['leafmachine']['print'].get('optional_warnings', True))

    with col_v2:
        log_level = st.session_state.config['leafmachine']['logging'].get('log_level', None)
        log_level_display = log_level if log_level is not None else 'default'
        selected_log_level = st.selectbox("Logging Level", ['default', 'DEBUG', 'INFO', 'WARNING', 'ERROR'], index=['default', 'DEBUG', 'INFO', 'WARNING', 'ERROR'].index(log_level_display))
        
        if selected_log_level == 'default':
            st.session_state.config['leafmachine']['logging']['log_level'] = None
        else:
            st.session_state.config['leafmachine']['logging']['log_level'] = selected_log_level
    

with tab_component:
    st.header('Archival Components')
    ACD_version = st.selectbox("Archival Component Detector (ACD) Version", ["Version 2.1", "Version 2.2"])
    
    ACD_confidence_default = int(st.session_state.config['leafmachine']['archival_component_detector']['minimum_confidence_threshold'] * 100)
    ACD_confidence = st.number_input("ACD Confidence Threshold (%)", min_value=0, max_value=100,value=ACD_confidence_default)
    st.session_state.config['leafmachine']['archival_component_detector']['minimum_confidence_threshold'] = float(ACD_confidence/100)

    st.session_state.config['leafmachine']['archival_component_detector']['do_save_prediction_overlay_images'] = st.checkbox("Save Archival Prediction Overlay Images", st.session_state.config['leafmachine']['archival_component_detector'].get('do_save_prediction_overlay_images', True))
    
    st.session_state.config['leafmachine']['archival_component_detector']['ignore_objects_for_overlay'] = st.multiselect("Hide Archival Components in Prediction Overlay Images",  
                ['ruler', 'barcode','label', 'colorcard','map','envelope','photo','attached_item','weights',],
                default=[])

    # Depending on the selected version, set the configuration
    if ACD_version == "Version 2.1":
        st.session_state.config['leafmachine']['archival_component_detector']['detector_type'] = 'Archival_Detector'
        st.session_state.config['leafmachine']['archival_component_detector']['detector_version'] = 'PREP_final'
        st.session_state.config['leafmachine']['archival_component_detector']['detector_iteration'] = 'PREP_final'
        st.session_state.config['leafmachine']['archival_component_detector']['detector_weights'] = 'best.pt'
    elif ACD_version == "Version 2.2": #TODO update this to version 2.2
        st.session_state.config['leafmachine']['archival_component_detector']['detector_type'] = 'Archival_Detector'
        st.session_state.config['leafmachine']['archival_component_detector']['detector_version'] = 'PREP_final'
        st.session_state.config['leafmachine']['archival_component_detector']['detector_iteration'] = 'PREP_final'
        st.session_state.config['leafmachine']['archival_component_detector']['detector_weights'] = 'best.pt'


    st.write("---")    
    st.header('Ruler Detection')
    RULER_version = st.selectbox("Ruler Detector Version", ["Version 2.1", "Version 2.2"])
    
    if RULER_version == "Version 2.1":
        st.session_state.config['leafmachine']['archival_component_detector']['ruler_detector'] = 'ruler_classifier_38classes_v-1.pt'
        st.session_state.config['leafmachine']['archival_component_detector']['ruler_binary_detector'] = 'model_scripted_resnet_720_withCompression.pt'
    elif RULER_version == "Version 2.2": #TODO update this to version 2.2
        st.session_state.config['leafmachine']['archival_component_detector']['ruler_detector'] = 'ruler_classifier_38classes_v-1.pt'
        st.session_state.config['leafmachine']['archival_component_detector']['ruler_binary_detector'] = 'model_scripted_resnet_720_withCompression.pt'

    RULER_confidence_default = int(st.session_state.config['leafmachine']['ruler_detection']['minimum_confidence_threshold'] * 100)
    RULER_confidence = st.number_input("Ruler Confidence Threshold (%)", min_value=0, max_value=100,value=RULER_confidence_default)
    st.session_state.config['leafmachine']['ruler_detection']['minimum_confidence_threshold'] = float(RULER_confidence/100)
    
    st.session_state.config['leafmachine']['ruler_detection']['detect_ruler_type'] = st.checkbox("Detect ruler type", st.session_state.config['leafmachine']['ruler_detection'].get('detect_ruler_type', True),disabled=True)
    st.session_state.config['leafmachine']['ruler_detection']['save_ruler_validation'] = st.checkbox("Save ruler validation", st.session_state.config['leafmachine']['ruler_detection'].get('save_ruler_validation', False))
    st.session_state.config['leafmachine']['ruler_detection']['save_ruler_validation_summary'] = st.checkbox("Save ruler validation summary", st.session_state.config['leafmachine']['ruler_detection'].get('save_ruler_validation_summary', True))
    st.session_state.config['leafmachine']['ruler_detection']['save_ruler_processed'] = st.checkbox("Save processed ruler image", st.session_state.config['leafmachine']['ruler_detection'].get('save_ruler_processed', False))
    
    
    st.write("---")    
    st.header('Plant Components')
    PCD_version = st.selectbox("Plant Component Detector (PCD) Version", ["LeafPriority (Version 2.2)", "Original (Version 2.1)"])
    
    PCD_confidence_default = int(st.session_state.config['leafmachine']['plant_component_detector']['minimum_confidence_threshold'] * 100)
    PCD_confidence = st.number_input("PCD Confidence Threshold (%)", min_value=0, max_value=100,value=PCD_confidence_default)
    st.session_state.config['leafmachine']['plant_component_detector']['minimum_confidence_threshold'] = float(PCD_confidence/100)

    st.session_state.config['leafmachine']['plant_component_detector']['do_save_prediction_overlay_images'] = st.checkbox("Save Plant Prediction Overlay Images", st.session_state.config['leafmachine']['plant_component_detector'].get('do_save_prediction_overlay_images', True))
    
    st.session_state.config['leafmachine']['plant_component_detector']['ignore_objects_for_overlay'] = st.multiselect("Hide Plant Components in Prediction Overlay Images",  
                ['leaf_whole', 'leaf_partial', 'leaflet', 'seed_fruit_one', 'seed_fruit_many', 'flower_one', 'flower_many', 'bud','specimen','roots','wood'],
                default=[])

    # Depending on the selected version, set the configuration
    if PCD_version == "Original (Version 2.1)":
        st.session_state.config['leafmachine']['plant_component_detector']['detector_type'] = 'Plant_Detector'
        st.session_state.config['leafmachine']['plant_component_detector']['detector_version'] = 'PLANT_GroupAB_200'
        st.session_state.config['leafmachine']['plant_component_detector']['detector_iteration'] = 'PLANT_GroupAB_200'
        st.session_state.config['leafmachine']['plant_component_detector']['detector_weights'] = 'best.pt'
    elif PCD_version == "LeafPriority (Version 2.2)": 
        st.session_state.config['leafmachine']['plant_component_detector']['detector_type'] = 'Plant_Detector'
        st.session_state.config['leafmachine']['plant_component_detector']['detector_version'] = 'PLANT_LeafPriority'
        st.session_state.config['leafmachine']['plant_component_detector']['detector_iteration'] = 'PLANT_LeafPriority'
        st.session_state.config['leafmachine']['plant_component_detector']['detector_weights'] = 'LeafPriority.pt'



with tab_landmark:
    st.header('Landmark Detector')

    PLD_version = st.selectbox("Pseudo-Landmark Detector (PLD) Version", ["Version 2.1", "Version 2.2"])

    st.session_state.config['leafmachine']['landmark_detector']['landmark_whole_leaves'] = st.checkbox("Landmark Whole Leaves", st.session_state.config['leafmachine']['landmark_detector'].get('landmark_whole_leaves', True))
    st.session_state.config['leafmachine']['landmark_detector']['landmark_partial_leaves'] = st.checkbox("Landmark Partial Leaves", st.session_state.config['leafmachine']['landmark_detector'].get('landmark_partial_leaves', False),disabled=True)

    PLD_confidence_default = int(st.session_state.config['leafmachine']['landmark_detector']['minimum_confidence_threshold'] * 100)
    PLD_confidence = st.number_input("PLD Confidence Threshold (%)", min_value=0, max_value=100,value=PLD_confidence_default)
    st.session_state.config['leafmachine']['landmark_detector']['minimum_confidence_threshold'] = float(PLD_confidence/100)

    st.session_state.config['leafmachine']['landmark_detector']['do_save_prediction_overlay_images'] = st.checkbox("Save Landmark Prediction Overlay Images", st.session_state.config['leafmachine']['landmark_detector'].get('do_save_prediction_overlay_images', True))
    st.session_state.config['leafmachine']['landmark_detector']['do_save_final_images'] = st.checkbox("Save Final Landmark Overlay Images", st.session_state.config['leafmachine']['landmark_detector'].get('do_save_final_images', True))
    st.session_state.config['leafmachine']['landmark_detector']['do_save_QC_images'] = st.checkbox("Save Landmark QC Overlay Images", st.session_state.config['leafmachine']['landmark_detector'].get('do_save_QC_images', True))

    st.session_state.config['leafmachine']['plant_component_detector']['ignore_objects_for_overlay'] = st.multiselect("Hide Landmark Components in Prediction Overlay Images",  
                ['lamina_trace', 'petiole_trace',],
                default=[], disabled=True)
    
    # Skipped 'use_existing_landmark_detections': [] #TODO
    # Skipped 'do_show_QC_images': False,#TODO
    # Skipped 'do_show_final_images': False,#TODO

    # Depending on the selected version, set the configuration
    if PLD_version == "Version 2.1":
        st.session_state.config['leafmachine']['landmark_detector']['detector_type'] = 'Landmark_Detector_YOLO'
        st.session_state.config['leafmachine']['landmark_detector']['detector_version'] = 'Landmarks'
        st.session_state.config['leafmachine']['landmark_detector']['detector_iteration'] = 'Landmarks'
        st.session_state.config['leafmachine']['landmark_detector']['detector_weights'] = 'best.pt'
    elif PLD_version == "Version 2.2": #TODO update this to version 2.2
        st.session_state.config['leafmachine']['landmark_detector']['detector_type'] = 'Landmark_Detector_YOLO'
        st.session_state.config['leafmachine']['landmark_detector']['detector_version'] = 'Landmarks'
        st.session_state.config['leafmachine']['landmark_detector']['detector_iteration'] = 'Landmarks_V2'
        st.session_state.config['leafmachine']['landmark_detector']['detector_weights'] = 'best.pt'


with tab_segmentation:
    st.header('Leaf Segmentation')
    st.markdown('***Version 2.2*** is new as of April 2024. It is trained on 3x the original dataset and for longer and should produce better masks overall.')
    st.markdown('***Version 2.1*** is the version that came out with the publication for LM2.')

    SEG_version = st.selectbox("Leaf Segmentation (SEG) Version", ["Version 2.2", "Version 2.1"]) # Group3_Dataset_100000_Iter_1176PTS_512Batch_smooth_l1_LR00025_BGR

    SEG_confidence_default = int(st.session_state.config['leafmachine']['leaf_segmentation']['minimum_confidence_threshold'] * 100)
    SEG_confidence = st.number_input("PLD Confidence Threshold (%)", min_value=0, max_value=100,value=SEG_confidence_default)
    st.session_state.config['leafmachine']['leaf_segmentation']['minimum_confidence_threshold'] = float(SEG_confidence/100)

    st.session_state.config['leafmachine']['leaf_segmentation']['segment_whole_leaves'] = st.checkbox("Segment whole leaves", st.session_state.config['leafmachine']['leaf_segmentation'].get('segment_whole_leaves', True))
    st.session_state.config['leafmachine']['leaf_segmentation']['segment_partial_leaves'] = st.checkbox("Segment partial leaves", st.session_state.config['leafmachine']['leaf_segmentation'].get('segment_partial_leaves', False))

    st.session_state.config['leafmachine']['leaf_segmentation']['save_segmentation_overlay_images_to_pdf'] = st.checkbox("Save segmentation overlay images to PDF", st.session_state.config['leafmachine']['leaf_segmentation'].get('save_segmentation_overlay_images_to_pdf', False))
    st.session_state.config['leafmachine']['leaf_segmentation']['save_each_segmentation_overlay_image'] = st.checkbox("Save each segmentation overlay image", st.session_state.config['leafmachine']['leaf_segmentation'].get('save_each_segmentation_overlay_image', True))
    st.session_state.config['leafmachine']['leaf_segmentation']['save_individual_overlay_images'] = st.checkbox("Save individual overlay images", st.session_state.config['leafmachine']['leaf_segmentation'].get('save_individual_overlay_images', True))
    st.session_state.config['leafmachine']['leaf_segmentation']['keep_only_best_one_leaf_one_petiole'] = st.checkbox("Keep best leaf/petiole if conflict", st.session_state.config['leafmachine']['leaf_segmentation'].get('keep_only_best_one_leaf_one_petiole', True),disabled=True)

    st.write("---")    
    st.subheader('Elliptic Fourier Descriptors (EFD)')
    st.session_state.config['leafmachine']['leaf_segmentation']['calculate_elliptic_fourier_descriptors'] = st.checkbox("Calculate elliptic Fourier descriptors", st.session_state.config['leafmachine']['leaf_segmentation'].get('calculate_elliptic_fourier_descriptors', False))
    st.session_state.config['leafmachine']['leaf_segmentation']['elliptic_fourier_descriptor_order'] = st.number_input(
            'Number of EFD harmonics', 
            min_value=1, max_value=40, 
            value=st.session_state.config['leafmachine']['leaf_segmentation'].get('elliptic_fourier_descriptor_order', 40))

    st.write("---")    
    st.subheader('Leaf Segmentation QC Overlay Options')
    st.session_state.config['leafmachine']['leaf_segmentation']['overlay_line_width'] = st.number_input(
            'QC overlay line width', 
            min_value=1, max_value=50, 
            value=st.session_state.config['leafmachine']['leaf_segmentation'].get('overlay_line_width', 1))
    
    st.session_state.config['leafmachine']['leaf_segmentation']['use_efds_for_png_masks'] = st.checkbox("Use EFDs for PNG masks", st.session_state.config['leafmachine']['leaf_segmentation'].get('use_efds_for_png_masks', False))
    st.session_state.config['leafmachine']['leaf_segmentation']['save_masks_color'] = st.checkbox("Save individual masks to PNGs", st.session_state.config['leafmachine']['leaf_segmentation'].get('save_masks_color', True))
    st.session_state.config['leafmachine']['leaf_segmentation']['save_full_image_masks_color'] = st.checkbox("Save full image masks to PNGs", st.session_state.config['leafmachine']['leaf_segmentation'].get('save_full_image_masks_color', True))
    st.session_state.config['leafmachine']['leaf_segmentation']['save_rgb_cropped_images'] = st.checkbox("Save RGB cropped images", st.session_state.config['leafmachine']['leaf_segmentation'].get('save_rgb_cropped_images', True))
    
    st.write("---")    
    st.session_state.config['leafmachine']['leaf_segmentation']['generate_overlay'] = st.checkbox("Save masks/RGB overlay images", st.session_state.config['leafmachine']['leaf_segmentation'].get('generate_overlay', True))
    st.session_state.config['leafmachine']['leaf_segmentation']['overlay_dpi'] = st.number_input(
            'Overlay DPI', 
            min_value=100, max_value=750, 
            value=st.session_state.config['leafmachine']['leaf_segmentation'].get('overlay_dpi', 300))
    
    st.session_state.config['leafmachine']['leaf_segmentation']['overlay_background_color'] = st.selectbox("Leaf segmentation background color", ["black", "white"], index=["black", "white"].index(st.session_state.config['leafmachine']['leaf_segmentation'].get('overlay_background_color', 'black')))

    st.session_state.config['leafmachine']['leaf_segmentation']['save_oriented_images'] = st.checkbox("save_oriented_images", st.session_state.config['leafmachine']['leaf_segmentation'].get('save_oriented_images', True))
    st.session_state.config['leafmachine']['leaf_segmentation']['save_keypoint_overlay'] = st.checkbox("save_keypoint_overlay", st.session_state.config['leafmachine']['leaf_segmentation'].get('save_keypoint_overlay', True))
    st.session_state.config['leafmachine']['leaf_segmentation']['save_oriented_mask'] = st.checkbox("save_oriented_mask", st.session_state.config['leafmachine']['leaf_segmentation'].get('save_oriented_mask', True))
    st.session_state.config['leafmachine']['leaf_segmentation']['save_simple_txt'] = st.checkbox("save_simple_txt", st.session_state.config['leafmachine']['leaf_segmentation'].get('save_simple_txt', True))
    st.session_state.config['leafmachine']['leaf_segmentation']['detector_version'] = st.selectbox("detector_version", ["uniform_spaced_oriented_traces_mid15_pet5_clean_640_flipidx_pt2",], index=["uniform_spaced_oriented_traces_mid15_pet5_clean_640_flipidx_pt2",].index(st.session_state.config['leafmachine']['leaf_segmentation'].get('detector_version', 'uniform_spaced_oriented_traces_mid15_pet5_clean_640_flipidx_pt2')))
        
    
    # Depending on the selected version, set the configuration
    if SEG_version == "Version 2.1":
        st.session_state.config['leafmachine']['leaf_segmentation']['segmentation_model'] = 'GroupB_Dataset_100000_Iter_1176PTS_512Batch_smooth_l1_LR00025_BGR'
    elif SEG_version == "Version 2.2": #TODO update this to version 2.2
        st.session_state.config['leafmachine']['leaf_segmentation']['segmentation_model'] = 'Group3_Dataset_100000_Iter_1176PTS_512Batch_smooth_l1_LR00025_BGR'

with tab_phenology:
    st.header('Phenology')
    st.markdown("Counts number of detected objects from the PCD as a proxy for phenology. This will report counts for all PCD classes, plus a determination of whether the image has leaves and whether the specimen is fertile.")
    st.markdown("We recommend manually verifying images when both 'has_leaves' & 'is_fertile' are false. Flowers and fruits are morphologically diverse and sometimes LM2 may not be able to ID all reproductive structures. For example, bare peduncles are frequently skipped because they closely resemble thin stems; if no other flowers/fruits are apparent, then this could cause a false negative. LM2 has a huge training dataset, but not every flowering/fruiting morphology has been included in the training data yet.")
    st.write("---")    
    st.session_state.config['leafmachine']['project']['accept_only_ideal_leaves'] = st.checkbox("'has_leaves' is determined by the presenece of 'ideal leaves', ignores partial leaves.", st.session_state.config['leafmachine']['project'].get('accept_only_ideal_leaves', True), help="Counts for partial leaves is still provided")
    st.session_state.config['leafmachine']['project']['minimum_total_reproductive_counts'] = st.number_input("Set the minimum required count of reproductive structures for the 'is_fertile' determination to return 'True'.", st.session_state.config['leafmachine']['project'].get('minimum_total_reproductive_counts', 0), help="Default is zero, any reproductive structure counts.")

with tab_overlay:
    st.header('Save Options')
    col_saveoverlay_1, col_saveoverlay_2, _= st.columns([3,3,6])

    st.write("---")
    st.header('Visualizations')
    col_viz_1, col_viz_2, col_viz_3 = st.columns([3,3,6])

    st.header("Style")
    col_style_1, col_style_2, col_style_3 = st.columns([3,3,6])


    with col_saveoverlay_2:
        st.text("Save images to:")
        st.session_state.config['leafmachine']['overlay']['save_overlay_to_pdf'] = st.checkbox("PDF per batch", st.session_state.config['leafmachine']['print'].get('save_overlay_to_pdf', False))
        st.session_state.config['leafmachine']['overlay']['save_overlay_to_jpgs'] = st.checkbox("JPG per specimen", st.session_state.config['leafmachine']['print'].get('save_overlay_to_jpgs', True))
    with col_saveoverlay_1:
        st.session_state.config['leafmachine']['project']['overlay_dpi'] = st.number_input("Overlay resolution (dpi)", value=st.session_state.config['leafmachine']['project'].get('overlay_dpi', 300))
        st.session_state.config['leafmachine']['project']['overlay_background_color'] = st.selectbox("Background color", ["black", "white"], index=["black", "white"].index(st.session_state.config['leafmachine']['project'].get('overlay_background_color', 'black')))

    with col_viz_1:
        st.subheader("Include in overlay image:")
        st.session_state.config['leafmachine']['overlay']['show_archival_detections'] = st.checkbox("Show bboxes for archival components", st.session_state.config['leafmachine']['print'].get('show_archival_detections', True))
        st.session_state.config['leafmachine']['overlay']['show_plant_detections'] = st.checkbox("Show bboxes for plant components", st.session_state.config['leafmachine']['print'].get('show_plant_detections', True))
        st.session_state.config['leafmachine']['overlay']['show_segmentations'] = st.checkbox("Show leaf segmentations", st.session_state.config['leafmachine']['print'].get('show_segmentations', True))
        st.session_state.config['leafmachine']['overlay']['show_landmarks'] = st.checkbox("Save landmarks", st.session_state.config['leafmachine']['print'].get('show_landmarks', True))

    with col_viz_3:
        st.subheader("Hide from overlay image:")
        st.markdown("This hides the bboxes from the ACD and PCD. Default hides the leaf_whole bboxes because leaf_whole already will always have a fitted/rotated bbox. Optionally hide anything else from the ACD/PCD.")
        default_ignore_plant = st.session_state.config['leafmachine']['overlay'].get('save_cropped_annotations', ['leaf_whole'])
        default_ignore_archival = st.session_state.config['leafmachine']['overlay'].get('ignore_archival_detections_classes', [])
        default_ignore_landmark = st.session_state.config['leafmachine']['overlay'].get('ignore_landmark_classes', [])

        st.session_state.config['leafmachine']['overlay']['ignore_plant_detections_classes'] = st.multiselect("Hide plant components",  
                ['leaf_whole', 'leaf_partial', 'leaflet', 'seed_fruit_one', 'seed_fruit_many', 'flower_one', 'flower_many', 'bud','specimen','roots','wood'],
                default=default_ignore_plant)
        
        st.session_state.config['leafmachine']['overlay']['ignore_archival_detections_classes'] = st.multiselect("Hide archival components",  
                ['ruler', 'barcode','label', 'colorcard','map','envelope','photo','attached_item','weights',]
                ,default=default_ignore_archival)
        
        st.session_state.config['leafmachine']['overlay']['ignore_landmark_classes'] = st.multiselect("Hide landmark components",  
                []
                ,default=default_ignore_landmark,disabled=True)
        
    with col_style_1:
        st.subheader("Alpha Transparency ")
        st.session_state.config['leafmachine']['overlay']['alpha_transparency_archival'] = st.number_input(
            'Archival', 
            min_value=0.0, max_value=1.0, 
            value=float(st.session_state.config['leafmachine']['overlay'].get('alpha_transparency_archival', 0.3)))

        # For alpha_transparency_plant
        st.session_state.config['leafmachine']['overlay']['alpha_transparency_plant'] = st.number_input(
            'Plant', 
            min_value=0.0, max_value=1.0, 
            value=float(st.session_state.config['leafmachine']['overlay'].get('alpha_transparency_plant', 0.0)))

        # For alpha_transparency_seg_whole_leaf
        st.session_state.config['leafmachine']['overlay']['alpha_transparency_seg_whole_leaf'] = st.number_input(
            'Segmentation Ideal Leaf', 
            min_value=0.0, max_value=1.0, 
            value=float(st.session_state.config['leafmachine']['overlay'].get('alpha_transparency_seg_whole_leaf', 0.4)))

        # For alpha_transparency_seg_partial_leaf
        st.session_state.config['leafmachine']['overlay']['alpha_transparency_seg_partial_leaf'] = st.number_input(
            'Segmentation Partial Leaf', 
            min_value=0.0, max_value=1.0, 
            value=float(st.session_state.config['leafmachine']['overlay'].get('alpha_transparency_seg_partial_leaf', 0.3)))

        
    with col_style_2:
        st.subheader("Line width")
        default_thick = 12
        st.session_state.config['leafmachine']['overlay']['line_width_archival'] = st.number_input(
            'Archival components', 
            min_value=1, max_value=50, 
            value=st.session_state.config['leafmachine']['overlay'].get('line_width_archival', default_thick))

        st.session_state.config['leafmachine']['overlay']['line_width_plant'] = st.number_input(
            'Plant components', 
            min_value=1, max_value=50, 
            value=st.session_state.config['leafmachine']['overlay'].get('line_width_plant', default_thick))

        st.session_state.config['leafmachine']['overlay']['line_width_seg'] = st.number_input(
            'Segmentation', 
            min_value=1, max_value=50, 
            value=st.session_state.config['leafmachine']['overlay'].get('line_width_seg', default_thick))

        st.session_state.config['leafmachine']['overlay']['line_width_efd'] = st.number_input(
            'Landmarks', 
            min_value=1, max_value=50, 
            value=st.session_state.config['leafmachine']['overlay'].get('line_width_efd', default_thick))














# Tab XXXX: Module Settings
# with tab_training:
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.session_state.config['leafmachine']['modules']['armature'] = st.checkbox("Armature Module", st.session_state.config['leafmachine']['modules'].get('armature', False))
        
#     with col2:
#         st.session_state.config['leafmachine']['modules']['specimen_crop'] = st.checkbox("Specimen Crop Module", st.session_state.config['leafmachine']['modules'].get('specimen_crop', False))
    
#     with col3:
#         st.session_state.config['leafmachine']['data']['save_json_rulers'] = st.checkbox("Save JSON Rulers", st.session_state.config['leafmachine']['data'].get('save_json_rulers', False))

#     with col4:
#         st.session_state.config['leafmachine']['data']['do_apply_conversion_factor'] = st.checkbox("Apply Conversion Factor", st.session_state.config['leafmachine']['data'].get('do_apply_conversion_factor', False))


# # Save the YAML
# if st.button("Save"):
#     save_yaml(st.session_state.config, yaml_file_path)
#     st.success(f"Configurations saved to {yaml_file_path}")
