import streamlit as st
import yaml
import os

def save_yaml(data, yaml_file_path):
    with open(yaml_file_path, 'w') as f:
        yaml.dump(data, f)

def load_yaml(yaml_file_path):
    if os.path.exists(yaml_file_path):
        with open(yaml_file_path, 'r') as f:
            return yaml.safe_load(f)
    return None  # Return None if the file doesn't exist

# Default YAML file path
yaml_file_path = "config.yaml"

st.set_page_config(layout="wide", page_icon='img/icon.ico', page_title='LeafMachine2')


# Main App
st.title("LeafMachine2")

st.markdown("""
This is a Streamlit app for configuring a YAML file.
""")

# Load existing YAML if it exists
config = load_yaml(yaml_file_path)
if config is None:
    config = {}  # Initialize an empty dict to avoid KeyError later

# Define the config structure if it doesn't exist
config.setdefault('leafmachine', {'do': {}, 'print': {}, 'project': {}, 'modules': {}, 'logging': {},  'data': {}
                                  ,'cropped_components': {},'data': {},'overlay': {},'plant_component_detector': {},
                                  'archival_component_detector': {},'armature_component_detector': {},'landmark_detector': {},
                                  'landmark_detector_armature': {},'ruler_detection': {},'leaf_segmentation': {},
                                  'images_to_process': {},'component_detector_train': {},'ruler_train': {},})


# Adding 2 Page Tabs
tab_settings, tab_component, tab_landmark, tab_segmentation, tab_overlay, tab_training = st.tabs(["Settings", "Component Detector","Landmark Detector","Segmentation", "Image Overlay", "Training Settings"])



# Tab 1: General Settings
with tab_settings:
    st.header('Project')
    col_project_1, col_project_2 = st.columns([4,2])

    st.write("---")
    st.header('Input Images')
    col_input_1, col_input_2 = st.columns([4,2])

    st.text('Process images stored locally')
    col_local_1, col_local_2 = st.columns([4,2])

    st.text('Process images from GBIF')
    col_GBIF_1, col_GBIF_2 = st.columns([4,2])

    st.write("---")
    st.header('Processing Options')
    col_processing_1, _, col_processing_2,col_processing_3 = st.columns([1,2,3,3])

    st.write("---")
    st.header('Modules')
    col_m1, col_m2 = st.columns(2)

    st.write("---")
    st.header('Cropped Components')    
    col_cropped_1, col_cropped_2 = st.columns([2,4])

    st.write("---")
    st.header('Logging and Image Validation')    
    col_v1, col_v2 = st.columns(2)

    ### Project
    with col_project_1:
        config['leafmachine']['project']['run_name'] = st.text_input("Run name", config['leafmachine']['project'].get('run_name', ''))
        config['leafmachine']['project']['dir_output'] = st.text_input("Output directory", config['leafmachine']['project'].get('dir_output', ''))
    
    ### Input Images
    with col_input_1:
        config['leafmachine']['project']['image_location'] = st.radio("Image location", ["local", "GBIF"], index=["local", "GBIF"].index(config['leafmachine']['project'].get('image_location', 'local')))

    ### Input Images Local
    with col_local_1:
        config['leafmachine']['project']['dir_images_local'] = st.text_input("Input local images", config['leafmachine']['project'].get('dir_images_local', ''))

    ### Input Images GBIF
    with col_GBIF_1:
        config['leafmachine']['project']['GBIF_mode'] = st.selectbox("GBIF mode", ["all", "filter"], index=["all", "filter"].index(config['leafmachine']['project'].get('GBIF_mode', 'all')))

    ### Processing Options
    with col_processing_1:
        st.subheader('Compute Options')
        config['leafmachine']['project']['num_workers'] = st.number_input("Number of CPU workers", value=config['leafmachine']['project'].get('num_workers', 2))
        config['leafmachine']['project']['batch_size'] = st.number_input("Batch size", value=config['leafmachine']['project'].get('batch_size', 50))

    with col_processing_2:
        st.subheader('Optional CSV Files')
        config['leafmachine']['data']['save_individual_csv_files_rulers'] = st.checkbox("Save ruler CSV files per image", config['leafmachine']['data'].get('save_individual_csv_files_rulers', False))
        config['leafmachine']['data']['save_individual_csv_files_measurements'] = st.checkbox("Save measurement CSV files per image", config['leafmachine']['data'].get('save_individual_csv_files_measurements', False))
        config['leafmachine']['data']['save_individual_csv_files_landmarks'] = st.checkbox("Save landmark CSV files per image", config['leafmachine']['data'].get('save_individual_csv_files_landmarks', False))
        config['leafmachine']['data']['save_individual_efd_files'] = st.checkbox("Save EFD CSV files per image", config['leafmachine']['data'].get('save_individual_efd_files', False))


    with col_processing_3:
        st.subheader('Misc')
        config['leafmachine']['data']['include_darwin_core_data_from_combined_file'] = st.checkbox("Attach Darwin Core information to combined output file", config['leafmachine']['data'].get('include_darwin_core_data_from_combined_file', False),disabled=True)
        config['leafmachine']['data']['save_json_rulers'] = st.checkbox("Save ruler data to JSON files", config['leafmachine']['data'].get('save_json_rulers', False),disabled=True)
        config['leafmachine']['data']['save_json_measurements'] = st.checkbox("Save measurements to JSON files", config['leafmachine']['data'].get('save_json_measurements', False),disabled=True)
        
    ### Modules
    with col_m1:
        config['leafmachine']['do']['run_leaf_processing'] = st.checkbox("Process leaves", config['leafmachine']['do'].get('run_leaf_processing', True))
        config['leafmachine']['modules']['armature'] = st.checkbox("Armature", config['leafmachine']['modules'].get('armature', False),disabled=True)
        config['leafmachine']['modules']['specimen_crop'] = st.checkbox("Specimen Close-up", config['leafmachine']['modules'].get('specimen_crop', False),disabled=True)

    ### cropped_components
    with col_cropped_1:
        config['leafmachine']['cropped_components']['do_save_cropped_annotations'] = st.checkbox("Save cropped components as images", config['leafmachine']['cropped_components'].get('do_save_cropped_annotations', True))
        config['leafmachine']['cropped_components']['save_per_image'] = st.checkbox("Save cropped components grouped by specimen", config['leafmachine']['cropped_components'].get('save_per_image', False))
        config['leafmachine']['cropped_components']['save_per_annotation_class'] = st.checkbox("Save cropped components grouped by type", config['leafmachine']['cropped_components'].get('save_per_annotation_class', True))
        config['leafmachine']['cropped_components']['binarize_labels'] = st.checkbox("Binarize labels", config['leafmachine']['cropped_components'].get('binarize_labels', False))
        config['leafmachine']['cropped_components']['binarize_labels_skeletonize'] = st.checkbox("Binarize and skeletonize labels", config['leafmachine']['cropped_components'].get('binarize_labels_skeletonize', False))
    
    
    with col_cropped_2:
        default_crops = config['leafmachine']['cropped_components'].get('save_cropped_annotations', ['leaf_whole'])
        config['leafmachine']['cropped_components']['save_cropped_annotations'] = st.multiselect("Components to crop",  
                ['ruler', 'barcode','label', 'colorcard','map','envelope','photo','attached_item','weights',
                'leaf_whole', 'leaf_partial', 'leaflet', 'seed_fruit_one', 'seed_fruit_many', 'flower_one', 'flower_many', 'bud','specimen','roots','wood'],default=default_crops)
    ### Logging and Image Validation - col_v1
    with col_v1:
        config['leafmachine']['do']['check_for_illegal_filenames'] = st.checkbox("Check for illegal filenames", config['leafmachine']['do'].get('check_for_illegal_filenames', True))
        config['leafmachine']['do']['check_for_corrupt_images_make_vertical'] = st.checkbox("Check for corrupt images", config['leafmachine']['do'].get('check_for_corrupt_images_make_vertical', True))
        
        config['leafmachine']['print']['verbose'] = st.checkbox("Print verbose", config['leafmachine']['print'].get('verbose', True))
        config['leafmachine']['print']['optional_warnings'] = st.checkbox("Show optional warnings", config['leafmachine']['print'].get('optional_warnings', True))

    with col_v2:
        log_level = config['leafmachine']['logging'].get('log_level', None)
        log_level_display = log_level if log_level is not None else 'default'
        selected_log_level = st.selectbox("Logging Level", ['default', 'DEBUG', 'INFO', 'WARNING', 'ERROR'], index=['default', 'DEBUG', 'INFO', 'WARNING', 'ERROR'].index(log_level_display))
        
        if selected_log_level == 'default':
            config['leafmachine']['logging']['log_level'] = None
        else:
            config['leafmachine']['logging']['log_level'] = selected_log_level
    









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
        config['leafmachine']['overlay']['save_overlay_to_pdf'] = st.checkbox("PDF per batch", config['leafmachine']['print'].get('save_overlay_to_pdf', True))
        config['leafmachine']['overlay']['save_overlay_to_jpgs'] = st.checkbox("JPG per specimen", config['leafmachine']['print'].get('save_overlay_to_jpgs', True))
    with col_saveoverlay_1:
        config['leafmachine']['project']['overlay_dpi'] = st.number_input("Overlay resolution (dpi)", value=config['leafmachine']['project'].get('overlay_dpi', 300))
        config['leafmachine']['project']['overlay_background_color'] = st.selectbox("Background color", ["black", "white"], index=["black", "white"].index(config['leafmachine']['project'].get('overlay_background_color', 'black')))

    with col_viz_1:
        st.subheader("Include in overlay image:")
        config['leafmachine']['overlay']['show_archival_detections'] = st.checkbox("Show bboxes for archival components", config['leafmachine']['print'].get('show_archival_detections', True))
        config['leafmachine']['overlay']['show_plant_detections'] = st.checkbox("Show bboxes for plant components", config['leafmachine']['print'].get('show_plant_detections', True))
        config['leafmachine']['overlay']['show_segmentations'] = st.checkbox("Show leaf segmentations", config['leafmachine']['print'].get('show_segmentations', True))
        config['leafmachine']['overlay']['show_landmarks'] = st.checkbox("Save landmarks", config['leafmachine']['print'].get('show_landmarks', True))

    with col_viz_3:
        st.subheader("Hide from overlay image:")
        default_ignore_plant = config['leafmachine']['overlay'].get('save_cropped_annotations', ['leaf_whole','specimen'])
        default_ignore_archival = config['leafmachine']['overlay'].get('ignore_archival_detections_classes', [])
        default_ignore_landmark = config['leafmachine']['overlay'].get('ignore_landmark_classes', [])

        config['leafmachine']['overlay']['save_cropped_annotations'] = st.multiselect("Hide plant components",  
                ['leaf_whole', 'leaf_partial', 'leaflet', 'seed_fruit_one', 'seed_fruit_many', 'flower_one', 'flower_many', 'bud','specimen','roots','wood'],
                default=default_ignore_plant)
        
        config['leafmachine']['overlay']['ignore_archival_detections_classes'] = st.multiselect("Hide archival components",  
                ['ruler', 'barcode','label', 'colorcard','map','envelope','photo','attached_item','weights',]
                ,default=default_ignore_archival)
        
        config['leafmachine']['overlay']['ignore_landmark_classes'] = st.multiselect("Hide landmark components",  
                []
                ,default=default_ignore_landmark,disabled=True)
        
    with col_style_1:
        st.subheader("Alpha Transparency ")
        config['leafmachine']['overlay']['alpha_transparency_archival'] = st.number_input(
            'Archival', 
            min_value=0.0, max_value=1.0, 
            value=config['leafmachine']['overlay'].get('alpha_transparency_archival', 0.3))

        # For alpha_transparency_plant
        config['leafmachine']['overlay']['alpha_transparency_plant'] = st.number_input(
            'Plant', 
            min_value=0.0, max_value=1.0, 
            value=config['leafmachine']['overlay'].get('alpha_transparency_plant', 0.0))

        # For alpha_transparency_seg_whole_leaf
        config['leafmachine']['overlay']['alpha_transparency_seg_whole_leaf'] = st.number_input(
            'Segmentation Ideal Leaf', 
            min_value=0.0, max_value=1.0, 
            value=config['leafmachine']['overlay'].get('alpha_transparency_seg_whole_leaf', 0.4))

        # For alpha_transparency_seg_partial_leaf
        config['leafmachine']['overlay']['alpha_transparency_seg_partial_leaf'] = st.number_input(
            'Segmentation Partial Leaf', 
            min_value=0.0, max_value=1.0, 
            value=config['leafmachine']['overlay'].get('alpha_transparency_seg_partial_leaf', 0.3))
        
    with col_style_2:
        st.subheader("Line width")
        default_thick = 12
        config['leafmachine']['overlay']['line_width_archival'] = st.number_input(
            'Archival components', 
            min_value=1, max_value=50, 
            value=config['leafmachine']['overlay'].get('line_width_archival', default_thick))

        config['leafmachine']['overlay']['line_width_plant'] = st.number_input(
            'Plant components', 
            min_value=1, max_value=50, 
            value=config['leafmachine']['overlay'].get('line_width_plant', default_thick))

        config['leafmachine']['overlay']['line_width_seg'] = st.number_input(
            'Segmentation', 
            min_value=1, max_value=50, 
            value=config['leafmachine']['overlay'].get('line_width_seg', default_thick))

        config['leafmachine']['overlay']['line_width_efd'] = st.number_input(
            'Landmarks', 
            min_value=1, max_value=50, 
            value=config['leafmachine']['overlay'].get('line_width_efd', default_thick))














# Tab XXXX: Module Settings
with tab_training:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        config['leafmachine']['modules']['armature'] = st.checkbox("Armature Module", config['leafmachine']['modules'].get('armature', False))
        
    with col2:
        config['leafmachine']['modules']['specimen_crop'] = st.checkbox("Specimen Crop Module", config['leafmachine']['modules'].get('specimen_crop', False))
    
    with col3:
        config['leafmachine']['data']['save_json_rulers'] = st.checkbox("Save JSON Rulers", config['leafmachine']['data'].get('save_json_rulers', False))

    with col4:
        config['leafmachine']['data']['do_apply_conversion_factor'] = st.checkbox("Apply Conversion Factor", config['leafmachine']['data'].get('do_apply_conversion_factor', False))

# Save the YAML
if st.button("Save"):
    save_yaml(config, yaml_file_path)
    st.success(f"Configurations saved to {yaml_file_path}")
