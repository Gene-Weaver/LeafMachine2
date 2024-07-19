
from __future__ import annotations
import os, sys, inspect, json, imagesize, shutil
import pandas as pd
from dataclasses import dataclass, field
from time import perf_counter


def save_data(cfg, time_report, logger, dir_home, Project, batch, n_batches, Dirs):
    start_t = perf_counter()
    logger.name = f'[BATCH {batch+1} Save Data]'
    logger.info(f'Saving data for {batch+1} of {n_batches}')

    n_order = cfg['leafmachine']['leaf_segmentation']['elliptic_fourier_descriptor_order'] if cfg['leafmachine']['leaf_segmentation']['elliptic_fourier_descriptor_order'] else 40
    # Generate the column names for efd_coeffs_features
    coeffs_col_names = [f'coeffs_{i}' for i in range(n_order)]

    # Create a new DataFrame with the required columns
    df_project_EFD = pd.DataFrame(columns=['filename', 'image_height', 'image_width','component_name','conversion_factor_applied','conversion_mean','predicted_conversion_factor_cm','annotation_name',
            'efd_order','efd_coeffs_features','efd_a0','efd_c0','efd_scale','efd_angle',
            'efd_phase','efd_area','efd_perimeter','efd_plot_points',] + coeffs_col_names)
    
    df_project_landmarks = pd.DataFrame(columns=[
        'filename', 'image_height', 'image_width','component_name','component_height','component_width','conversion_factor_applied','landmark_status','leaf_type','has_all_landmarks','determined_leftright',
        'lamina_length','lamina_width','ordered_midvein_length','ordered_petiole_length','lobe_count','apex_angle_type','apex_angle_degrees','base_angle_type','base_angle_degrees',
        'has_apex','has_base','has_lamina_base','has_lamina_length','has_lamina_tip','has_lobes','has_midvein','has_ordered_petiole','has_valid_apex_loc','has_valid_base_loc','has_width',
        'apex_center','apex_left','apex_right','base_center','base_left','base_right','lamina_tip','lamina_tip_alt','lamina_base','lamina_base_alt','lobes','midvein_fit_points','ordered_midvein','ordered_petiole','width_left','width_right',
        't_apex_center','t_apex_left','t_apex_right','t_base_center','t_base_left','t_base_right','t_lamina_base','t_lamina_tip','t_midvein','t_midvein_fit_points','t_petiole','t_width_infer','t_width_left','t_width_right',
        'lamina_fit_ax_b','midvein_fit_ax_b',
        'plot_x_shift','plot_y_shift',])

    df_project_rulers = pd.DataFrame(columns=['filename', 'image_height', 'image_width', 'ruler_image_name', 'ruler_location',
            'ruler_success', 'conversion_mean', 'predicted_conversion_factor_cm', 'pooled_sd', 'ruler_class', 'ruler_class_confidence', 
            'units', 'cross_validation_count' ,'n_scanlines' ,'n_data_points_in_avg', 
            'avg_tick_width',])

    seg_column_names = ['filename', 'image_height', 'image_width','component_name','conversion_factor_applied',
            'annotation_name','bbox','bbox_min','rotate_angle','bbox_min_long_side','bbox_min_short_side',
            'area','perimeter','centroid','convex_hull','convexity', 'concavity',
            'circularity','n_pts_in_polygon','aspect_ratio','polygon_closed','polygon_closed_rotated',
            'efd_order','efd_coeffs_features','efd_a0','efd_c0','efd_scale','efd_angle',
            'efd_phase','efd_area','efd_perimeter','efd_plot_points',

            'ruler_image_name', 'ruler_success','conversion_mean', 'predicted_conversion_factor_cm','pooled_sd','ruler_class',
            'ruler_class_confidence','units', 'cross_validation_count','n_scanlines','n_data_points_in_avg','avg_tick_width',]
    
    if cfg['leafmachine']['data']['include_darwin_core_data_from_combined_file']:
        specimen, temp_dict = next(iter(Project.project_data.items())) 
        record_column_names = list(Project.project_data[specimen]['GBIF_Record'].keys())
        column_names = seg_column_names + record_column_names
    else:
        column_names = seg_column_names

    df_project_seg = pd.DataFrame(columns= column_names)

    # Loop through the completed images
    for filename, analysis in Project.project_data_list[batch].items(): # Per image
        # Get ruler info
        df_ruler = None
        df_seg = None

        DATA = Data_Vault(cfg, logger, filename, analysis, Dirs)
        df_ruler = DATA.get_ruler_dataframe()
        df_seg = DATA.get_seg_dataframe()
        df_EFD = DATA.get_EFD_dataframe()
        df_landmarks = DATA.get_landmarks_dataframe()

        df_project_rulers = pd.concat([df_project_rulers, df_ruler], ignore_index=True)
        df_project_seg = pd.concat([df_project_seg, df_seg], ignore_index=True)
        df_project_EFD = pd.concat([df_project_EFD, df_EFD], ignore_index=True)
        df_project_landmarks = pd.concat([df_project_landmarks, df_landmarks], ignore_index=True)


    df_project_rulers.to_csv(os.path.join(Dirs.data_csv_project_batch_ruler, '.'.join([''.join([Dirs.run_name, '__Ruler__', str(batch+1), 'of', str(n_batches)]), 'csv'])), header=True, index=False)
    df_project_seg.to_csv(os.path.join(Dirs.data_csv_project_batch_measurements, '.'.join([''.join([Dirs.run_name, '__Measurements__', str(batch+1), 'of', str(n_batches)]), 'csv'])), header=True, index=False)
    if cfg['leafmachine']['leaf_segmentation']['calculate_elliptic_fourier_descriptors']:
        df_project_EFD.to_csv(os.path.join(Dirs.data_csv_project_batch_EFD, '.'.join([''.join([Dirs.run_name, '__EFD__', str(batch+1), 'of', str(n_batches)]), 'csv'])), header=True, index=False)
    df_project_landmarks.to_csv(os.path.join(Dirs.data_csv_project_batch_landmarks, '.'.join([''.join([Dirs.run_name, '__Landmarks__', str(batch+1), 'of', str(n_batches)]), 'csv'])), header=True, index=False)

    end_t = perf_counter()

    t_save = f"[Batch {batch+1}: Save Data elapsed time] {round(end_t - start_t)} seconds ({round((end_t - start_t)/60)} minutes)"
    logger.info(t_save)
    time_report['t_save'] = t_save
    return time_report


@dataclass
class Data_Vault():
    # Image info
    height: str = None
    width: str = None
    filename: str = None

    # Specimen Record
    specimen_record: list[str] = field(default_factory=list)

    # Plant detections
    detections_plant: list[float] = field(default_factory=list)
    
    # Archival Detections
    detections_archival: list[float] = field(default_factory=list)

    # Ruler Info
    ruler_image_name: str = None
    ruler_class: str = None
    ruler_class_confidence: float = 0.0
    ruler_location: list[int] = field(default_factory=list)

    # Ruler Data
    conversion_factor: float = 0.0
    conversion_determination: str = None
    conversion_inputs: list[str] = field(default_factory=list)
    plot_points: list[int] = field(default_factory=list)
    plot_points_10cm: list[int] = field(default_factory=list)
    plot_points_1cm: list[int] = field(default_factory=list)
    point_types: list[str] = field(default_factory=list)
    n_points: int = 0
    scanline_height: int = 0
    distances_all: list[float] = field(default_factory=list)
    sd: float = 0.0
    conversion_factor_gmean: float = 0.0
    conversion_factor_mean: float = 0.0
    unit: str = None

    df_ruler: list[str] = field(default_factory=list)
    df_seg: list[str] = field(default_factory=list)
    ruler_dict_list: list[str] = field(default_factory=list)
    seg_info_dict: list[str] = field(default_factory=list)
    df_landmarks: list[str] = field(default_factory=list)

    csv_img: str = ''
    df_ruler_use: list[str] = field(default_factory=list)

    def __init__(self, cfg, logger, filename, analysis, Dirs) -> None:
        self.cfg = cfg
        logger.debug(f"[Saving] {filename}")
        # print(filename)
        # if cfg['leafmachine']['data']['save_json']:
        #     self.dict_to_json(analysis, Dirs.data_json, filename)

        # print(analysis)

        # Unpack data
        self.height = self.get_key_value(analysis, 'height', 0)
        self.width = self.get_key_value(analysis, 'width', 0)

        self.specimen_record = self.get_key_value(analysis, 'GBIF_Record')

        self.archival = self.get_key_value(analysis, 'Detections_Archival_Components')
        self.plant = self.get_key_value(analysis, 'Detections_Plant_Components')

        self.ruler_info = self.get_key_value(analysis, 'Ruler_Info')

        self.seg_whole_leaf = self.get_key_value(analysis, 'Segmentation_Whole_Leaf')
        self.seg_partial_leaf = self.get_key_value(analysis, 'Segmentation_Partial_Leaf')

        self.landmarks_whole_leaf = self.get_key_value(analysis, 'Landmarks_Whole_Leaves')
        self.landmarks_partial_leaf = []


        
        self.filename = filename

        self.gather_ruler_info(cfg, Dirs,self.ruler_info) #, self.ruler_data)

        # Pick the best ruler # TODO make sure this is the best methods
        df_ruler_use = None
        df_ruler = self.get_ruler_dataframe()
        if df_ruler.shape[0] == 1:
            df_ruler_use = df_ruler
        else:
            ## *** CODE TO PICK THE BEST....
            if df_ruler['ruler_class_confidence'].notna().any():
                df_ruler_use = df_ruler.sort_values(by='ruler_class_confidence', ascending=False).iloc[0]
            else:
                df_ruler_use = df_ruler.iloc[0]

        df_ruler_use = df_ruler_use.to_dict()

        self.gather_EFD_data(cfg, logger, Dirs, self.seg_whole_leaf, self.seg_partial_leaf, df_ruler_use)
        self.gather_seg_data(cfg, logger, Dirs, self.seg_whole_leaf, self.seg_partial_leaf, df_ruler_use)
        self.gather_landmark_data(cfg, logger, Dirs, self.landmarks_whole_leaf, self.landmarks_partial_leaf, df_ruler_use)

    def get_key_value(self, dictionary, key, default_value=[]):
        return dictionary.get(key, default_value)
    
    def gather_landmark_data(self, cfg, logger, Dirs, landmarks_whole_leaf, landmarks_partial_leaf, df_ruler_use):
        landmarks_list = [landmarks_whole_leaf, landmarks_partial_leaf]
        all_export_data = []

        df_ruler_use = self.ensure_list_values_preruler(df_ruler_use)

        columns_to_extract = ['ruler_image_name', 'ruler_success', 'conversion_mean', 'predicted_conversion_factor_cm', 'pooled_sd', 'ruler_class',
                            'ruler_class_confidence', 'cross_validation_count',
                            'n_scanlines', 'n_data_points_in_avg', 'avg_tick_width',]
        
        column_names = ['filename','image_height','image_width','component_name','component_height', 'component_width','conversion_factor_applied','landmark_status','leaf_type','has_all_landmarks',
            'determined_leftright','lamina_length', 'lamina_width','ordered_midvein_length', 'ordered_petiole_length','lobe_count',
            'apex_angle_type','apex_angle_degrees','base_angle_type','base_angle_degrees','has_apex','has_base','has_lamina_base',
            'has_lamina_length','has_lamina_tip','has_lobes','has_midvein','has_ordered_petiole','has_valid_apex_loc','has_valid_base_loc','has_width',
            'apex_center','apex_left','apex_right','base_center','base_left','base_right','lamina_tip','lamina_tip_alt','lamina_base','lamina_base_alt',
            'lobes','midvein_fit_points','ordered_midvein','ordered_petiole','width_left','width_right','t_apex_center','t_apex_left','t_apex_right',
            't_base_center','t_base_left','t_base_right','t_lamina_base', 't_lamina_tip','t_midvein','t_midvein_fit_points','t_petiole',
            't_width_infer','t_width_left','t_width_right', 'lamina_fit_ax_b','midvein_fit_ax_b', 'plot_x_shift','plot_y_shift', 'ruler_image_name','ruler_success',
            'conversion_mean', 'predicted_conversion_factor_cm','pooled_sd','ruler_class','ruler_class_confidence','units','cross_validation_count','n_scanlines','n_data_points_in_avg','avg_tick_width',
            ]

        # Initialize an empty dictionary to store the extracted values
        extracted_values = {}

        self.df_landmarks = pd.DataFrame(columns=column_names)


        # Use the function to populate the dictionary
        for col in columns_to_extract:
            extracted_values[col] = self.extract_value_from_dataframe(df_ruler_use, col)[0]

        # Handle the special case for 'units'
        extracted_values['units'] = self.extract_value_from_dataframe(df_ruler_use, 'units', check_empty=True)[0]

        # Process each leaf_type_data in seg_data_list
        for leaf_type_data in landmarks_list:
            for i, land_info_part in enumerate(leaf_type_data):
                land_image_name, land_info = next(iter(land_info_part.items()))
                print(land_image_name)

                landmark_status = land_info[0].get('landmark_status', 'NA') if land_info else 'NA'

                LeafSkeleton = land_info[1].get('landmarks', None) if land_info else None

                export_dict = {}

                ### Info
                export_dict['filename'] = self.filename
                export_dict['image_height'] = self.height
                export_dict['image_width'] = self.width
                export_dict['component_name'] = land_image_name
                export_dict['component_height'] = LeafSkeleton.get('height')
                export_dict['component_width'] = LeafSkeleton.get('width')

                if cfg['leafmachine']['data']['do_apply_conversion_factor'] and (df_ruler_use['ruler_success'][0]): # != 'NA' or df_ruler_use['conversion_mean'][0] != 0):
                    do_apply_conversion_factor = True
                else:
                    do_apply_conversion_factor = False

                export_dict['conversion_factor_applied'] = do_apply_conversion_factor

                export_dict['landmark_status'] = landmark_status
                export_dict['leaf_type'] = LeafSkeleton.get('leaf_type')
                export_dict['has_all_landmarks'] = LeafSkeleton.get('is_complete_leaf')
                export_dict['determined_leftright'] = LeafSkeleton.get('is_split')

                ### Measurements
                export_dict['lamina_length'] = LeafSkeleton.get('lamina_length')
                export_dict['lamina_width'] = LeafSkeleton.get('lamina_width')

                export_dict['ordered_midvein_length'] = LeafSkeleton.get('ordered_midvein_length')
                export_dict['ordered_petiole_length'] = LeafSkeleton.get('ordered_petiole_length')

                export_dict['lobe_count'] = LeafSkeleton.get('lobe_count')

                # export_dict['petiole_length'] = LeafSkeleton.get('petiole_length')
                # export_dict['leaf_length'] = LeafSkeleton.get('leaf_length')
                # export_dict['leaf_width'] = LeafSkeleton.get('leaf_width')

                export_dict['apex_angle_type'] = LeafSkeleton.get('apex_angle_type')
                export_dict['apex_angle_degrees'] = round(LeafSkeleton.get('apex_angle_degrees'),2) if LeafSkeleton.get('apex_angle_degrees') is not None else None
                export_dict['base_angle_type'] = LeafSkeleton.get('base_angle_type')
                export_dict['base_angle_degrees'] = round(LeafSkeleton.get('base_angle_degrees'),2) if LeafSkeleton.get('base_angle_degrees') is not None else None

                ### Binary
                export_dict['has_apex'] = LeafSkeleton.get('has_apex')
                export_dict['has_base'] = LeafSkeleton.get('has_base')
                export_dict['has_lamina_base'] = LeafSkeleton.get('has_lamina_base')
                export_dict['has_lamina_length'] = LeafSkeleton.get('has_lamina_length')
                export_dict['has_lamina_tip'] = LeafSkeleton.get('has_lamina_tip')
                export_dict['has_lobes'] = LeafSkeleton.get('has_lobes')
                export_dict['has_midvein'] = LeafSkeleton.get('has_midvein')
                export_dict['has_ordered_petiole'] = LeafSkeleton.get('has_ordered_petiole')
                export_dict['has_valid_apex_loc'] = LeafSkeleton.get('has_valid_apex_loc')
                export_dict['has_valid_base_loc'] = LeafSkeleton.get('has_valid_base_loc')
                export_dict['has_width'] = LeafSkeleton.get('has_width')

                ### Locations
                export_dict['apex_center'] = [LeafSkeleton.get('apex_center')]
                export_dict['apex_left'] = [LeafSkeleton.get('apex_left')]
                export_dict['apex_right'] = [LeafSkeleton.get('apex_right')]

                export_dict['base_center'] = [LeafSkeleton.get('base_center')]
                export_dict['base_left'] = [LeafSkeleton.get('base_left')]
                export_dict['base_right'] = [LeafSkeleton.get('base_right')]

                export_dict['lamina_tip'] = [LeafSkeleton.get('lamina_tip')]
                export_dict['lamina_tip_alt'] = [LeafSkeleton.get('lamina_tip_alternate')]
                export_dict['lamina_base'] = [LeafSkeleton.get('lamina_base')]
                export_dict['lamina_base_alt'] = [LeafSkeleton.get('lamina_base_alternate')]

                export_dict['lobes'] = LeafSkeleton.get('lobes')

                export_dict['midvein_fit_points'] = [LeafSkeleton.get('midvein_fit_points')]

                export_dict['ordered_midvein'] = LeafSkeleton.get('ordered_midvein')
                export_dict['ordered_petiole'] = LeafSkeleton.get('ordered_petiole')

                export_dict['width_left'] = LeafSkeleton.get('width_left') # for lamina_width
                export_dict['width_right'] = LeafSkeleton.get('width_right')

                ### Plot points for full image
                export_dict['t_apex_center'] = LeafSkeleton.get('t_apex_center')
                export_dict['t_apex_left'] = LeafSkeleton.get('t_apex_left')
                export_dict['t_apex_right'] = LeafSkeleton.get('t_apex_right')

                export_dict['t_base_center'] = LeafSkeleton.get('t_base_center')
                export_dict['t_base_left'] = LeafSkeleton.get('t_base_left')
                export_dict['t_base_right'] = LeafSkeleton.get('t_base_right')

                export_dict['t_lamina_base'] = LeafSkeleton.get('t_lamina_base')
                export_dict['t_lamina_tip'] = LeafSkeleton.get('t_lamina_tip')

                export_dict['t_midvein'] = LeafSkeleton.get('t_midvein')
                export_dict['t_midvein_fit_points'] = LeafSkeleton.get('t_midvein_fit_points')

                export_dict['t_petiole'] = LeafSkeleton.get('t_petiole')
                export_dict['t_width_infer'] = LeafSkeleton.get('t_width_infer')
                export_dict['t_width_left'] = LeafSkeleton.get('t_width_left')
                export_dict['t_width_right'] = LeafSkeleton.get('t_width_right')

                ### Line of fit
                lf = LeafSkeleton.get('lamina_fit')
                mf = LeafSkeleton.get('midvein_fit')
                export_dict['lamina_fit_ax_b'] = [lf[0], lf[1]] if lf is not None else None
                export_dict['midvein_fit_ax_b'] = [mf[0], mf[1]] if export_dict['t_midvein_fit_points'] is not None else None
                
                ### Less important
                export_dict['plot_x_shift'] = LeafSkeleton.get('add_x')
                export_dict['plot_y_shift'] = LeafSkeleton.get('add_y')

                # All from ruler data
                export_dict['ruler_image_name'] = self.extract_value_from_dataframe(df_ruler_use, 'ruler_image_name')[0]
                export_dict['ruler_success'] = self.extract_value_from_dataframe(df_ruler_use, 'ruler_success')[0]
                export_dict['conversion_mean'] = self.extract_value_from_dataframe(df_ruler_use, 'conversion_mean')[0]
                export_dict['predicted_conversion_factor_cm'] = self.extract_value_from_dataframe(df_ruler_use, 'predicted_conversion_factor_cm')[0]
                export_dict['pooled_sd'] = self.extract_value_from_dataframe(df_ruler_use, 'pooled_sd')[0]

                export_dict['ruler_class'] = self.extract_value_from_dataframe(df_ruler_use, 'ruler_class')[0]
                export_dict['ruler_class_confidence'] = self.extract_value_from_dataframe(df_ruler_use, 'ruler_class_confidence')[0]
                
                export_dict['units'] = self.extract_value_from_dataframe(df_ruler_use, 'units', check_empty=True)[0]
                export_dict['cross_validation_count'] = self.extract_value_from_dataframe(df_ruler_use, 'cross_validation_count')[0]
                export_dict['n_scanlines'] = self.extract_value_from_dataframe(df_ruler_use, 'n_scanlines')[0]
                export_dict['n_data_points_in_avg'] = self.extract_value_from_dataframe(df_ruler_use, 'n_data_points_in_avg')[0]
                export_dict['avg_tick_width'] = self.extract_value_from_dataframe(df_ruler_use, 'avg_tick_width')[0]

                # Apply conversion factor to the values in the dict
                if do_apply_conversion_factor:
                    # try:
                    export_dict = self.divide_values_length_landmarks(export_dict, df_ruler_use)
                
                export_dict = self.ensure_list_values(export_dict) 
                all_export_data.append(export_dict)

                # Convert all_export_data to a DataFrame
                # print(self.df_landmarks.shape)
                # for key, value in export_dict.items():
                #     print(f"{key}: {len(value) if isinstance(value, list) else 'Not a list'}")

                # print(pd.DataFrame.from_dict(export_dict).shape)

                self.df_landmarks = pd.concat([self.df_landmarks, pd.DataFrame.from_dict(export_dict)], ignore_index=True)

                if cfg['leafmachine']['data']['save_individual_csv_files_landmarks']:
                    self.df_landmarks.to_csv(os.path.join(Dirs.data_csv_individual_landmarks, '.'.join([self.filename, 'csv'])), header=True, index=False)
    
    def divide_values_length_landmarks(self, export_dict, df_ruler_use):
        # Handle cases where the denominator is zero or 'NA'
        if (df_ruler_use['conversion_mean'][0] == 0) or (df_ruler_use['conversion_mean'][0] == 'NA'):
            return export_dict
        else:
            # List of keys we want to divide
            keys_to_divide = ['lamina_length', 'lamina_width', 'ordered_midvein_length', 'ordered_petiole_length']
            # Loop through each key and safely perform the division
            for key in keys_to_divide:
                # Check if the numerator exists and is not None
                if key in export_dict:
                    if export_dict[key] is not None:
                        # Check if the denominator exists, is not None and not zero
                        if 'conversion_mean' in df_ruler_use and df_ruler_use['conversion_mean'][0] and df_ruler_use['conversion_mean'][0] != 0:
                            # Now we know both the numerator and denominator are safe to use
                            export_dict[key] = round(export_dict[key] / df_ruler_use['conversion_mean'][0], 2) if export_dict[key] is not None else None
                    # else: # Keep the original pixel value, 
                    #     export_dict[key] = export_dict[key]
                else:
                    # Initialize to None
                    export_dict[key] = None
        return export_dict


    def gather_EFD_data(self, cfg, logger, Dirs, seg_whole_leaf, seg_partial_leaf, df_ruler_use):
        # Initialize data structures
        efd_data_list = [seg_whole_leaf, seg_partial_leaf]
        self.efd_dict_list = []

        df_ruler_use = self.ensure_list_values_preruler(df_ruler_use)
        
        # Define column names
        efd_column_names = [
            'filename', 'image_height', 'image_width', 'component_name', 'conversion_factor_applied', 'conversion_mean', 'predicted_conversion_factor_cm', 'annotation_name',
            'efd_order', 'efd_coeffs_features', 'efd_a0', 'efd_c0', 'efd_scale', 'efd_angle',
            'efd_phase', 'efd_area', 'efd_perimeter', 'efd_plot_points',
            ]
        
        if cfg['leafmachine']['data']['include_darwin_core_data_from_combined_file']:
            record_column_names = list(self.specimen_record.keys())
            column_names = efd_column_names + record_column_names
        else:
            column_names = efd_column_names

        # Initialize DataFrame
        self.df_EFD = pd.DataFrame(columns=column_names)

        # Handle edge case: No leaf segmentations
        if all(len(data) == 0 for data in efd_data_list):
            return

        # Process each leaf_type_data in seg_data_list
        for leaf_type_data in efd_data_list:
            for i, efd_info_part in enumerate(leaf_type_data):
                efd_image_name, efd_info_dict = next(iter(efd_info_part.items()))
                logger.debug(f"[Leaf EFD] {efd_image_name}")

                n_order = cfg['leafmachine']['leaf_segmentation']['elliptic_fourier_descriptor_order'] if cfg['leafmachine']['leaf_segmentation']['elliptic_fourier_descriptor_order'] else 40
                coeffs_col_names = [f'coeffs_{i}' for i in range(n_order)]  

                # if cfg['leafmachine']['data']['do_apply_conversion_factor'] and df_ruler_use['conversion_mean'][0] != 'NA' or df_ruler_use['conversion_mean'][0] != 0:
                if cfg['leafmachine']['data']['do_apply_conversion_factor'] and (df_ruler_use['ruler_success'][0]):
                    do_apply_conversion_factor = True
                else:
                    do_apply_conversion_factor = False

                # Initialize common parts of dict_EFD
                dict_EFD = {
                    'filename': [self.filename],  # ruler data
                    'image_height': [self.height],  # ruler data
                    'image_width': [self.width],  # ruler data
                }
                
                    
                if cfg['leafmachine']['leaf_segmentation']['calculate_elliptic_fourier_descriptors']:
                    # Handle cases: Data not available
                    if not efd_info_dict:
                        dict_EFD.update({
                            'component_name': ['NA'],
                            'conversion_factor_applied': [do_apply_conversion_factor],
                            'conversion_mean': [df_ruler_use['conversion_mean'][0]],
                            'predicted_conversion_factor_cm': [df_ruler_use['predicted_conversion_factor_cm'][0]],
                            'annotation_name': ['NA'],
                            'efd_order': ['NA'],
                            'efd_order': ['NA'],
                            'efd_coeffs_features': ['too_long'],
                            'efd_a0': ['NA'],
                            'efd_c0': ['NA'],
                            'efd_scale': ['NA'],
                            'efd_angle': ['NA'],
                            'efd_phase': ['NA'],
                            'efd_area': ['NA'],
                            'efd_perimeter': ['NA'],
                            'efd_plot_points': ['NA'],
                        })
                        for col_name in coeffs_col_names:
                            dict_EFD[col_name] = ['NA']

                    # Handle cases: Data available
                    else:
                        for annotation in efd_info_dict:
                            annotation_name, annotation_dict = next(iter(annotation.items()))
                            logger.debug(f'[Annotation] {annotation_name}')

                            dict_EFD.update({
                                'component_name': [efd_image_name],
                                'conversion_factor_applied': [do_apply_conversion_factor],
                                'conversion_mean': [df_ruler_use['conversion_mean'][0]],
                                'predicted_conversion_factor_cm': [df_ruler_use['predicted_conversion_factor_cm'][0]],
                                'annotation_name': [annotation_name],
                                'efd_order': [int(annotation_dict['efds']['coeffs_normalized'].shape[0])],
                                'efd_a0': [float(annotation_dict['efds']['a0'])],
                                'efd_c0': [float(annotation_dict['efds']['c0'])],
                                'efd_scale': [float(annotation_dict['efds']['scale'])],
                                'efd_angle': [float(annotation_dict['efds']['angle'])],
                                'efd_phase': [float(annotation_dict['efds']['phase'])],
                                'efd_area': [float(annotation_dict['efds']['efd_area'])],
                                'efd_perimeter': [float(annotation_dict['efds']['efd_perimeter'])],
                                'efd_plot_points': ['too_long'], #[annotation_dict['efds']['efd_pts_PIL']],
                                'efd_coeffs_features': ['too_long']#,[annotation_dict['efds']['coeffs_features'].tolist()],
                            })
                            for i, coeffs in enumerate(annotation_dict['efds']['coeffs_normalized']):
                                dict_EFD[coeffs_col_names[i]] = [coeffs]
                            # Apply conversion factor to the values in the dict
                            if do_apply_conversion_factor:
                                # try:
                                dict_EFD = self.divide_values_length_efd(dict_EFD, df_ruler_use)
                else:
                    dict_EFD.update({
                            'component_name': ['NA'],
                            'conversion_factor_applied': [do_apply_conversion_factor],
                            'conversion_mean': [df_ruler_use['conversion_mean'][0]],
                            'predicted_conversion_factor_cm': [df_ruler_use['predicted_conversion_factor_cm'][0]],
                            'annotation_name': ['NA'],
                            'efd_order': ['NA'],
                            'efd_order': ['NA'],
                            'efd_coeffs_features': ['too_long'],
                            'efd_a0': ['NA'],
                            'efd_c0': ['NA'],
                            'efd_scale': ['NA'],
                            'efd_angle': ['NA'],
                            'efd_phase': ['NA'],
                            'efd_area': ['NA'],
                            'efd_perimeter': ['NA'],
                            'efd_plot_points': ['NA'],
                        })
                    for col_name in coeffs_col_names:
                        dict_EFD[col_name] = ['NA']

                # Add to seg_dict_list and update DataFrame
                self.efd_dict_list.append(dict_EFD)
                self.df_EFD = pd.concat([self.df_EFD, pd.DataFrame.from_dict(dict_EFD)], ignore_index=True)

        if cfg['leafmachine']['data']['save_json_measurements']:
            anno_list = self.efd_dict_list[0]
            try:
                name_json = ''.join([anno_list['component_name'], '_EFD', '.json'])
            except:
                name_json = ''.join([anno_list['component_name'][0], '_EFD', '.json'])
            with open(os.path.join(Dirs.data_json_measurements, name_json), "w") as outfile:
                json.dump(self.efd_dict_list, outfile)

        if cfg['leafmachine']['data']['save_individual_efd_files']:
            try:
                self.df_EFD.to_csv(os.path.join(Dirs.data_efd_individual_measurements, ''.join([self.filename, '_EFD','.csv'])), header=True, index=False)
            except: 
                self.df_EFD.to_csv(os.path.join(Dirs.data_efd_individual_measurements, ''.join([self.filename[0], '_EFD','.csv'])), header=True, index=False)
        
    def divide_values_length_efd(self, export_dict, df_ruler_use):
        # Handle cases where the denominator is zero or 'NA'
        if (df_ruler_use['conversion_mean'][0] == 0) or (df_ruler_use['conversion_mean'][0] == 'NA'):
            return export_dict
        else:
            # List of keys we want to divide
            keys_to_divide = ['efd_area', 'efd_perimeter',]
            # Loop through each key and safely perform the division
            for key in keys_to_divide:
                # Check if the numerator exists and is not None
                if key in export_dict:
                    if export_dict[key] is not None:
                        # Check if the denominator exists, is not None and not zero
                        if 'conversion_mean' in df_ruler_use and df_ruler_use['conversion_mean'][0] and df_ruler_use['conversion_mean'][0] != 0:
                            # Now we know both the numerator and denominator are safe to use
                            if key == 'efd_area':
                                try:
                                    export_dict[key] = round(export_dict[key][0] / (df_ruler_use['conversion_mean'][0] * df_ruler_use['conversion_mean'][0]), 2) if export_dict[key][0] is not None else None
                                except:
                                    export_dict[key] = round(export_dict[key][0] / (df_ruler_use['conversion_mean'][0][0] * df_ruler_use['conversion_mean'][0][0]), 2) if export_dict[key][0] is not None else None
                            else:
                                try:
                                    export_dict[key] = round(export_dict[key][0] / df_ruler_use['conversion_mean'][0], 2) if export_dict[key][0] is not None else None
                                except:
                                    export_dict[key] = round(export_dict[key][0] / df_ruler_use['conversion_mean'][0][0], 2) if export_dict[key][0] is not None else None
                else:
                    # Initialize to None
                    export_dict[key] = None
        return export_dict
    
    def extract_value_from_dataframe(self, df, column_name, check_empty=False):
        """Extract a value from a DataFrame column and return it in a list format."""
        value = df[column_name]
        if isinstance(value, (list, dict)):
            if check_empty and not value:
                return None
            return value
        else:
            return value
        
    def ensure_list_values(self, input_dict):
        for key, value in input_dict.items():
            if not isinstance(value, list):
                input_dict[key] = [value]
            elif isinstance(value, list) and len(value) > 1:
                input_dict[key] = [value]
            elif isinstance(value, list) and len(value) == 0:
                input_dict[key] = [None]
        return input_dict
    
    def ensure_list_values_preruler(self, input_dict):
        for key, value in input_dict.items():
            if not isinstance(value, (list, dict)):
                input_dict[key] = [value]
            elif isinstance(value, (list, dict)) and len(value) > 1:
                input_dict[key] = value
            elif isinstance(value, (list, dict)) and len(value) == 0:
                input_dict[key] = [None]
        return input_dict

    def gather_seg_data(self, cfg, logger, Dirs, seg_whole_leaf, seg_partial_leaf, df_ruler_use):
        seg_data_list = [seg_whole_leaf, seg_partial_leaf]

        df_ruler_use = self.ensure_list_values_preruler(df_ruler_use)

        ruler_column_names = ['filename', 'image_height', 'image_width', 'component_name','conversion_factor_applied',

            'annotation_name','bbox','bbox_min','rotate_angle','bbox_min_long_side','bbox_min_short_side',
            'area','perimeter','centroid','convex_hull','convexity', 'concavity',
            'circularity','n_pts_in_polygon','aspect_ratio','polygon_closed','polygon_closed_rotated',
            'efd_order','efd_coeffs_features','efd_a0','efd_c0','efd_scale','efd_angle',
            'efd_phase','efd_area','efd_perimeter','efd_plot_points',

            'ruler_image_name', 'ruler_success','conversion_mean', 'predicted_conversion_factor_cm','pooled_sd','ruler_class',
            'ruler_class_confidence','units', 'cross_validation_count','n_scanlines','n_data_points_in_avg','avg_tick_width',]

        if cfg['leafmachine']['data']['include_darwin_core_data_from_combined_file']:
            record_column_names = list(self.specimen_record.keys())
            column_names = ruler_column_names + record_column_names
        else:
            column_names = ruler_column_names

        # if cfg['leafmachine']['data']['do_apply_conversion_factor'] and df_ruler_use['conversion_mean'][0] != 'NA' or df_ruler_use['conversion_mean'][0] != 0:
        if cfg['leafmachine']['data']['do_apply_conversion_factor'] and (df_ruler_use['ruler_success'][0]):
            do_apply_conversion_factor = True
        else:
            do_apply_conversion_factor = False

        df_seg = pd.DataFrame(columns= column_names)
        self.seg_dict_list = []
        if (seg_data_list[0] == []) and (seg_data_list[1] == []): # No leaf segmentaitons
            self.df_seg = df_seg
        else:
            for leaf_type_data in seg_data_list:
                for i in range(len(leaf_type_data)):
                    # Ruler Info
                    seg_info_part = leaf_type_data[i]
                    component_name, seg_info_dict = next(iter(seg_info_part.items())) 
                    # print(component_name)
                    logger.debug(f"[Leaf] {component_name}")

                    if seg_info_dict == []:                      
                        
                        columns_to_extract = [
                            'ruler_image_name', 'ruler_success', 'conversion_mean', 'predicted_conversion_factor_cm', 'pooled_sd', 'ruler_class',
                            'ruler_class_confidence', 'cross_validation_count',
                            'n_scanlines', 'n_data_points_in_avg', 'avg_tick_width'
                        ]

                        # Initialize an empty dictionary to store the extracted values
                        extracted_values = {}

                        # Use the function to populate the dictionary
                        for col in columns_to_extract:
                            extracted_values[col] = self.extract_value_from_dataframe(df_ruler_use, col)[0]

                        # Handle the special case for 'units'
                        extracted_values['units'] = self.extract_value_from_dataframe(df_ruler_use, 'units', check_empty=True)[0]

                        dict_seg = {
                            'filename': [self.filename], # ruler data
                            'image_height': [self.height], # ruler data
                            'image_width': [self.width], # ruler data
                            'component_name': ['NA'],

                            'conversion_factor_applied': [do_apply_conversion_factor],

                            'annotation_name': ['NA'],
                            'bbox': ['NA'],
                            'bbox_min': ['NA'],
                            'rotate_angle': ['NA'],
                            'bbox_min_long_side': ['NA'],
                            'bbox_min_short_side': ['NA'],
                            'area': ['NA'],
                            'perimeter': ['NA'],
                            'centroid': ['NA'],
                            'convex_hull': ['NA'],
                            'convexity': ['NA'],
                            'concavity': ['NA'],
                            'circularity': ['NA'],
                            'n_pts_in_polygon': ['NA'],
                            'aspect_ratio': ['NA'],
                            'polygon_closed': ['NA'],
                            'polygon_closed_rotated': ['NA'],

                            'efd_order': ['NA'],
                            'efd_coeffs_features': ['NA'],
                            'efd_a0': ['NA'],
                            'efd_c0': ['NA'],
                            'efd_scale': ['NA'],
                            'efd_angle': ['NA'],
                            'efd_phase': ['NA'],
                            'efd_area': ['NA'],
                            'efd_perimeter': ['NA'],
                            'efd_plot_points': ['NA'],

                            # All from ruler data
                            'ruler_image_name': self.extract_value_from_dataframe(df_ruler_use, 'ruler_image_name')[0],
                            'ruler_success': self.extract_value_from_dataframe(df_ruler_use, 'ruler_success')[0],
                            'conversion_mean': self.extract_value_from_dataframe(df_ruler_use, 'conversion_mean')[0],
                            'predicted_conversion_factor_cm': self.extract_value_from_dataframe(df_ruler_use, 'predicted_conversion_factor_cm')[0],
                            'pooled_sd': self.extract_value_from_dataframe(df_ruler_use, 'pooled_sd')[0],

                            'ruler_class': self.extract_value_from_dataframe(df_ruler_use, 'ruler_class')[0],
                            'ruler_class_confidence': self.extract_value_from_dataframe(df_ruler_use, 'ruler_class_confidence')[0],
                            
                            'units': self.extract_value_from_dataframe(df_ruler_use, 'units', check_empty=True)[0],
                            'cross_validation_count': self.extract_value_from_dataframe(df_ruler_use, 'cross_validation_count')[0],
                            'n_scanlines': self.extract_value_from_dataframe(df_ruler_use, 'n_scanlines')[0],
                            'n_data_points_in_avg': self.extract_value_from_dataframe(df_ruler_use, 'n_data_points_in_avg')[0],
                            'avg_tick_width': self.extract_value_from_dataframe(df_ruler_use, 'avg_tick_width')[0],
                        }
        
                        dict_seg = self.ensure_list_values(dict_seg)

                        if not cfg['leafmachine']['data']['include_darwin_core_data_from_combined_file']:
                            self.seg_dict_list.append(dict_seg)
                            df_seg = pd.concat([df_seg, pd.DataFrame.from_dict(dict_seg)], ignore_index=True)
                        else:
                            dict_seg2 = dict_seg.copy()
                            dict_seg2.update(self.specimen_record)
                            self.seg_dict_list.append(dict_seg2)

                            dict_seg = pd.DataFrame.from_dict(dict_seg)
                            dict_seg = pd.concat([dict_seg, pd.DataFrame.from_dict(self.specimen_record)], axis = 1)
                            df_seg = pd.concat([df_seg, dict_seg], ignore_index=True)
                    else:
                        for annotation in seg_info_dict:
                            annotation_name, annotation_dict = next(iter(annotation.items())) 
                            # print(annotation_name)
                            logger.debug(f'[Annotation] {annotation_name}')

                            if not cfg['leafmachine']['leaf_segmentation']['calculate_elliptic_fourier_descriptors']:
                                val_efd_order = ['NA']
                                val_efd_coeffs_features = ['NA']
                                val_efd_a0 = ['NA']
                                val_efd_c0 = ['NA']
                                val_efd_scale = ['NA']
                                val_efd_angle = ['NA']
                                val_efd_phase = ['NA']
                                val_efd_area = ['NA']
                                val_efd_perimeter = ['NA']
                                val_efd_plot_points = ['NA']
                            else:
                                val_efd_order = [int(annotation_dict['efds']['coeffs_normalized'].shape[0])]
                                val_efd_coeffs_features = ['too_long']
                                val_efd_a0 = [float(annotation_dict['efds']['a0'])]
                                val_efd_c0 = [float(annotation_dict['efds']['c0'])]
                                val_efd_scale = [float(annotation_dict['efds']['scale'])]
                                val_efd_angle = [float(annotation_dict['efds']['angle'])]
                                val_efd_phase = [float(annotation_dict['efds']['phase'])]
                                val_efd_area = [float(annotation_dict['efds']['efd_area'])]
                                val_efd_perimeter = [float(annotation_dict['efds']['efd_perimeter'])]
                                val_efd_plot_points = ['too_long']

                            dict_seg = {
                                'filename': [self.filename], # ruler data
                                'image_height': [self.height], # ruler data
                                'image_width': [self.width], # ruler data
                                'component_name': [component_name],

                                'conversion_factor_applied': [do_apply_conversion_factor],
                                'annotation_name': [annotation_name],
                                'bbox': [annotation_dict['bbox']],
                                'bbox_min': [annotation_dict['bbox_min']],
                                'rotate_angle': annotation_dict['rotate_angle'],
                                'bbox_min_long_side': [annotation_dict['long']],
                                'bbox_min_short_side': [annotation_dict['short']],
                                'area': [annotation_dict['area']],
                                'perimeter': [annotation_dict['perimeter']],
                                'centroid': [annotation_dict['centroid']],
                                'convex_hull': [round(annotation_dict['convex_hull'], 2) if annotation_dict['convex_hull'] is not None else None],
                                'convexity': [round(annotation_dict['convexity'], 6) if annotation_dict['convexity'] is not None else None],
                                'concavity': [round(annotation_dict['concavity'], 6) if annotation_dict['concavity'] is not None else None],
                                'circularity': [round(annotation_dict['circularity'], 6) if annotation_dict['circularity'] is not None else None],
                                'n_pts_in_polygon': [annotation_dict['degree']],
                                'aspect_ratio': [round(annotation_dict['aspect_ratio'], 6) if annotation_dict['aspect_ratio'] is not None else None],
                                'polygon_closed': ['too_long'], #[out_polygon_closed],
                                'polygon_closed_rotated': ['too_long'], #[out_polygon_closed_rotated],

                                'efd_order': val_efd_order,
                                'efd_coeffs_features': val_efd_coeffs_features, #[annotation_dict['efds']['coeffs_features'].tolist()],
                                'efd_a0': val_efd_a0,
                                'efd_c0': val_efd_c0,
                                'efd_scale': val_efd_scale,
                                'efd_angle': val_efd_angle,
                                'efd_phase': val_efd_phase,
                                'efd_area': val_efd_area,
                                'efd_perimeter': val_efd_perimeter,
                                'efd_plot_points': val_efd_plot_points, #[annotation_dict['efds']['efd_pts_PIL']],

                                # All from ruler data
                                'ruler_image_name': self.extract_value_from_dataframe(df_ruler_use, 'ruler_image_name')[0],
                                'ruler_success': self.extract_value_from_dataframe(df_ruler_use, 'ruler_success')[0],
                                'conversion_mean': self.extract_value_from_dataframe(df_ruler_use, 'conversion_mean')[0],
                                'predicted_conversion_factor_cm': self.extract_value_from_dataframe(df_ruler_use, 'predicted_conversion_factor_cm')[0],
                                'pooled_sd': self.extract_value_from_dataframe(df_ruler_use, 'pooled_sd')[0],

                                'ruler_class': self.extract_value_from_dataframe(df_ruler_use, 'ruler_class')[0],
                                'ruler_class_confidence': self.extract_value_from_dataframe(df_ruler_use, 'ruler_class_confidence')[0],
                                
                                'units': self.extract_value_from_dataframe(df_ruler_use, 'units', check_empty=True)[0],
                                'cross_validation_count': self.extract_value_from_dataframe(df_ruler_use, 'cross_validation_count')[0],
                                'n_scanlines': self.extract_value_from_dataframe(df_ruler_use, 'n_scanlines')[0],
                                'n_data_points_in_avg': self.extract_value_from_dataframe(df_ruler_use, 'n_data_points_in_avg')[0],
                                'avg_tick_width': self.extract_value_from_dataframe(df_ruler_use, 'avg_tick_width')[0],
                            }
                            # Apply conversion factor to the values in the dict
                            if do_apply_conversion_factor:
                                # try:
                                dict_seg = self.divide_values_length(dict_seg, df_ruler_use)
                                # except:
                                    # pass
                                # try:
                                dict_seg = self.divide_values_sq(dict_seg, df_ruler_use)
                                # except:
                                    # pass

                            dict_seg = self.ensure_list_values(dict_seg)
                            
                            if not cfg['leafmachine']['data']['include_darwin_core_data_from_combined_file']:
                                self.seg_dict_list.append(dict_seg)
                                df_seg = pd.concat([df_seg, pd.DataFrame.from_dict(dict_seg).reset_index()], ignore_index=True)
                            else:
                                dict_seg2 = dict_seg.copy()
                                dict_seg2.update(self.specimen_record)
                                self.seg_dict_list.append(dict_seg2)

                                dict_seg = pd.DataFrame.from_dict(dict_seg)
                                dict_seg = pd.concat([dict_seg, pd.DataFrame.from_dict(self.specimen_record).reset_index()], axis = 1)
                                df_seg = pd.concat([df_seg, dict_seg], ignore_index=True)
            
            self.df_seg = df_seg
            if cfg['leafmachine']['data']['save_json_measurements']:
                # for r in range(len(self.seg_dict_list)):
                    # dict_r = self.seg_dict_list[r]
                    # anno_name = ''.join([dict_r['annotation_name'][0].split('_'), str(dict_r['centroid'][0][0]), '-', str(dict_r['centroid'][0][1])])
                    # name_json_part = ''.join([dict_r['component_name'], anno_name])
                anno_list = self.seg_dict_list[0]
                try:
                    name_json = '.'.join([anno_list['component_name'], 'json'])
                except:
                    name_json = '.'.join([anno_list['component_name'][0], 'json'])
                with open(os.path.join(Dirs.data_json_measurements, name_json), "w") as outfile:
                    json.dump(self.seg_dict_list, outfile)

            if cfg['leafmachine']['data']['save_individual_csv_files_measurements']:
                try:
                    self.df_seg.to_csv(os.path.join(Dirs.data_csv_individual_measurements, '.'.join([df_ruler_use['filename'], 'csv'])), header=True, index=False)
                except: 
                    self.df_seg.to_csv(os.path.join(Dirs.data_csv_individual_measurements, '.'.join([df_ruler_use['filename'][0], 'csv'])), header=True, index=False)

    
        
    def gather_ruler_info(self, cfg, Dirs, ruler_info):
        ruler_column_names = ['filename', 'image_height', 'image_width','ruler_image_name', 'ruler_location', 
            'ruler_success', 'conversion_mean', 'predicted_conversion_factor_cm', 'pooled_sd', 'ruler_class', 'ruler_class_confidence', 
            'units', 'cross_validation_count' ,'n_scanlines' ,'n_data_points_in_avg', 
            'avg_tick_width',]

        if cfg['leafmachine']['data']['include_darwin_core_data_from_combined_file']:
            record_column_names = list(self.specimen_record.keys())
            column_names = ruler_column_names + record_column_names
        else:
            column_names = ruler_column_names
        df_ruler = pd.DataFrame(columns=column_names)
        self.ruler_dict_list = []

        if len(ruler_info) == 0:
            dict_ruler = {
                'filename': [self.filename],
                'image_height': [self.height],
                'image_width': [self.width],
                'ruler_image_name': ['no_ruler'],
                'ruler_location': ['NA'],
                

                'ruler_success': ['False'],
                'conversion_mean': ['NA'],
                'predicted_conversion_factor_cm': ['NA'],
                'pooled_sd': ['NA'],

                'ruler_class': ['not_supported'],
                'ruler_class_confidence': [0],

                'units': ['NA'],
                'cross_validation_count': ['NA'],
                'n_scanlines': ['NA'],
                'n_data_points_in_avg': ['NA'],
                'avg_tick_width': ['NA'],

                # 'specimen_record': self.specimen_record,
                # 'detections_plant': self.detections_plant,
                # 'detections_archival': self.detections_archival,
            }
            if not cfg['leafmachine']['data']['include_darwin_core_data_from_combined_file']:
                self.ruler_dict_list.append(dict_ruler)
                df_ruler = pd.concat([df_ruler, pd.DataFrame.from_dict(dict_ruler)], ignore_index=True)
            else:
                dict_ruler2 = dict_ruler.copy()
                dict_ruler2.update(self.specimen_record)
                self.ruler_dict_list.append(dict_ruler2)

                dict_ruler = pd.DataFrame.from_dict(dict_ruler)
                dict_ruler = pd.concat([dict_ruler, pd.DataFrame.from_dict(self.specimen_record)], axis = 1)
                df_ruler = pd.concat([df_ruler, dict_ruler], ignore_index=True)
        else:
            for i in range(len(ruler_info)):
                # Ruler Info
                ruler_info_dict = ruler_info[i]
                ruler_image_name = ruler_info_dict['ruler_image_name']
                
                if ruler_info_dict == []:
                    dict_ruler = {
                        'filename': [self.filename],
                        'image_height': [self.height],
                        'image_width': [self.width],
                        'ruler_image_name': ['no_ruler'],
                        'ruler_location': ['NA'],

                        'ruler_success': ['False'],
                        'conversion_mean': ['NA'],
                        'predicted_conversion_factor_cm': ['NA'],
                        'pooled_sd': ['NA'],

                        'ruler_class': ['not_supported'],
                        'ruler_class_confidence': [0],
                        'units': ['NA'],

                        'cross_validation_count': ['NA'],
                        'n_scanlines': ['NA'],

                        'n_data_points_in_avg': ['NA'],
                        'avg_tick_width': ['NA'],

                        # 'specimen_record': self.specimen_record,
                        # 'detections_plant': self.detections_plant,
                        # 'detections_archival': self.detections_archival,
                    }
                    if not cfg['leafmachine']['data']['include_darwin_core_data_from_combined_file']:
                        self.ruler_dict_list.append(dict_ruler)
                        df_ruler = pd.concat([df_ruler, pd.DataFrame.from_dict(dict_ruler)], ignore_index=True)
                    else:
                        dict_ruler2 = dict_ruler.copy()
                        dict_ruler2.update(self.specimen_record)
                        self.ruler_dict_list.append(dict_ruler2)

                        dict_ruler = pd.DataFrame.from_dict(dict_ruler)
                        dict_ruler = pd.concat([dict_ruler, pd.DataFrame.from_dict(self.specimen_record)], axis = 1)
                        df_ruler = pd.concat([df_ruler, dict_ruler], ignore_index=True)
                else:
                    ruler_info_dict = ruler_info[i]
                    ruler_image_name = ruler_info_dict['ruler_image_name']
                    # ruler_class = ruler_info_dict.ruler_class
                    # ruler_class_confidence = ruler_info_dict.ruler_class_percentage
                    ruler_location = ruler_image_name.split('__')[-1]
                    ruler_location = [int(ruler_location.split('-')[0]),int(ruler_location.split('-')[1]),int(ruler_location.split('-')[2]),int(ruler_location.split('-')[3])]

                    # Ruler Data
                    ruler_data_part = ruler_info[i]
                    # self.ruler_image_name, ruler_data_dict = next(iter(ruler_data_part.items())) 
                        
                    dict_ruler = {
                        'filename': [self.filename],
                        'image_height': [self.height],
                        'image_width': [self.width],
                        'ruler_image_name': [ruler_image_name],
                        'ruler_location': [ruler_location],
                        
                        # All from ruler data
                        'ruler_success': self.extract_value_from_dataframe(ruler_data_part, 'success'),
                        'conversion_mean': self.extract_value_from_dataframe(ruler_data_part, 'conversion_mean'),
                        'predicted_conversion_factor_cm': self.extract_value_from_dataframe(ruler_data_part, 'predicted_conversion_factor_cm'),
                        'pooled_sd': self.extract_value_from_dataframe(ruler_data_part, 'pooled_sd'),

                        'ruler_class': self.extract_value_from_dataframe(ruler_data_part, 'ruler_class'),
                        'ruler_class_confidence': self.extract_value_from_dataframe(ruler_data_part, 'ruler_class_confidence'),
                        
                        'units': self.extract_value_from_dataframe(ruler_data_part, 'units', check_empty=True),
                        'cross_validation_count': self.extract_value_from_dataframe(ruler_data_part, 'cross_validation_count'),
                        'n_scanlines': self.extract_value_from_dataframe(ruler_data_part, 'n_scanlines'),
                        'n_data_points_in_avg': self.extract_value_from_dataframe(ruler_data_part, 'n_data_points_in_avg'),
                        'avg_tick_width': self.extract_value_from_dataframe(ruler_data_part, 'avg_tick_width'),
                    }

                    dict_ruler = self.ensure_list_values(dict_ruler)

                    if not cfg['leafmachine']['data']['include_darwin_core_data_from_combined_file']:
                        self.ruler_dict_list.append(dict_ruler)
                        df_ruler = pd.concat([df_ruler, pd.DataFrame.from_dict(dict_ruler)], ignore_index=True)
                    else:
                        dict_ruler2 = dict_ruler.copy()
                        dict_ruler2.update(self.specimen_record)
                        self.ruler_dict_list.append(dict_ruler2)

                        dict_ruler = pd.DataFrame.from_dict(dict_ruler)
                        dict_ruler = pd.concat([dict_ruler, pd.DataFrame.from_dict(self.specimen_record)], axis = 1)
                        df_ruler = pd.concat([df_ruler, dict_ruler], ignore_index=True)

        self.df_ruler = df_ruler

        if cfg['leafmachine']['data']['save_json_rulers']:
            for r in range(len(self.ruler_dict_list)):
                dict_r = self.ruler_dict_list[r]
                name_json = '.'.join([dict_r['ruler_image_name'][0], 'json'])
                with open(os.path.join(Dirs.data_json_rulers, name_json), "w") as outfile:
                    json.dump(dict_r, outfile)

        if cfg['leafmachine']['data']['save_individual_csv_files_rulers']:
            self.df_ruler.to_csv(os.path.join(Dirs.data_csv_individual_rulers, '.'.join([self.filename, 'csv'])), header=True, index=False)

    def safe_assign_value(self, data_dict, key, default_value):
        try:
            return data_dict[key]
        except KeyError:
            return default_value
    
    def gather_ruler_data(self, cfg, Dirs, ruler_info, ruler_data):
        ruler_column_names = ['filename', 'image_height', 'image_width','ruler_image_name', 
            'conversion_factor', 'conversion_determination', 'conversion_inputs',
            'ruler_class', 'ruler_class_confidence', 'ruler_location','plot_points', 
            'plot_points_1cm', 'plot_points_10cm', 'point_types','n_points','scanline_height',
            'distances_in_average','sd','conversion_factor_gmean','conversion_factor_mean','unit']
        if cfg['leafmachine']['data']['include_darwin_core_data_from_combined_file']:
            record_column_names = list(self.specimen_record.keys())
            column_names = ruler_column_names + record_column_names
        else:
            column_names = ruler_column_names
        df_ruler = pd.DataFrame(columns=column_names)
        self.ruler_dict_list = []

        if len(ruler_info) == 0:
            dict_ruler = {
                'filename': [self.filename],
                'image_height': [self.height],
                'image_width': [self.width],
                'ruler_image_name': ['no_ruler'],

                'conversion_factor': ['NA'],
                'conversion_determination': ['NA'],
                'conversion_inputs': ['NA'],

                'ruler_class': ['not_supported'],
                'ruler_class_confidence': [0],
                'ruler_location': ['NA'],

                'plot_points': ['NA'],
                'plot_points_1cm': ['NA'],
                'plot_points_10cm': ['NA'],
                'point_types': ['NA'],

                'n_points': ['NA'],
                'scanline_height': ['NA'],
                'distances_in_average': ['NA'],
                'sd': ['NA'],
                'conversion_factor_gmean': ['NA'],
                'conversion_factor_mean': ['NA'],
                'unit': ['NA']

                # 'specimen_record': self.specimen_record,
                # 'detections_plant': self.detections_plant,
                # 'detections_archival': self.detections_archival,
            }
            if not cfg['leafmachine']['data']['include_darwin_core_data_from_combined_file']:
                self.ruler_dict_list.append(dict_ruler)
                df_ruler = pd.concat([df_ruler, pd.DataFrame.from_dict(dict_ruler)], ignore_index=True)
            else:
                dict_ruler2 = dict_ruler.copy()
                dict_ruler2.update(self.specimen_record)
                self.ruler_dict_list.append(dict_ruler2)

                dict_ruler = pd.DataFrame.from_dict(dict_ruler)
                dict_ruler = pd.concat([dict_ruler, pd.DataFrame.from_dict(self.specimen_record)], axis = 1)
                df_ruler = pd.concat([df_ruler, dict_ruler], ignore_index=True)

        else:
            for i in range(len(ruler_info)):
                # Ruler Info
                ruler_info_part = ruler_info[i]
                ruler_image_name, ruler_info_dict = next(iter(ruler_info_part.items())) 

                ruler_image_name = ruler_image_name
                
                if ruler_info_dict == []:
                    dict_ruler = {
                        'filename': [self.filename],
                        'image_height': [self.height],
                        'image_width': [self.width],
                        'ruler_image_name': [ruler_image_name],
                        
                        'conversion_factor': ['NA'],
                        'conversion_determination': ['NA'],
                        'conversion_inputs': ['NA'],

                        'ruler_class': ['not_supported'],
                        'ruler_class_confidence': [0],
                        'ruler_location': ['NA'],

                        'plot_points': ['NA'],
                        'plot_points_1cm': ['NA'],
                        'plot_points_10cm': ['NA'],
                        'point_types': ['NA'],

                        'n_points': ['NA'],
                        'scanline_height': ['NA'],
                        'distances_in_average': ['NA'],
                        'sd': ['NA'],
                        'conversion_factor_gmean': ['NA'],
                        'conversion_factor_mean': ['NA'],
                        'unit': ['NA']

                        # 'specimen_record': self.specimen_record,
                        # 'detections_plant': self.detections_plant,
                        # 'detections_archival': self.detections_archival,
                    }
                    if not cfg['leafmachine']['data']['include_darwin_core_data_from_combined_file']:
                        self.ruler_dict_list.append(dict_ruler)
                        df_ruler = pd.concat([df_ruler, pd.DataFrame.from_dict(dict_ruler)], ignore_index=True)
                    else:
                        dict_ruler2 = dict_ruler.copy()
                        dict_ruler2.update(self.specimen_record)
                        self.ruler_dict_list.append(dict_ruler2)

                        dict_ruler = pd.DataFrame.from_dict(dict_ruler)
                        dict_ruler = pd.concat([dict_ruler, pd.DataFrame.from_dict(self.specimen_record)], axis = 1)
                        df_ruler = pd.concat([df_ruler, dict_ruler], ignore_index=True)
                else:
                    ruler_image_name = ruler_image_name
                    ruler_class = ruler_info_dict.ruler_class
                    ruler_class_confidence = ruler_info_dict.ruler_class_percentage
                    ruler_location = ruler_info_dict.img_fname.split('__')[-1]
                    ruler_location = [int(ruler_location.split('-')[0]),int(ruler_location.split('-')[1]),int(ruler_location.split('-')[2]),int(ruler_location.split('-')[3])]

                    # Ruler Data
                    ruler_data_part = ruler_data[i]
                    self.ruler_image_name, ruler_data_dict = next(iter(ruler_data_part.items())) 

                    try:
                        self.plot_points = ruler_data_dict.plot_points# block
                    except: # scanlines
                        try:
                            self.plot_points = []
                            for t in ruler_data_dict['plot_points']:
                                self.plot_points.append([float(x) for x in t])
                        except:
                            self.plot_points = 'NA'

                    attributes_to_assign = [
                        ('conversion_determination', 'conversion_location', 'scanlines'),
                        ('conversion_inputs', 'conversion_location_options', 'scanlines'),
                        ('plot_points_10cm', 'plot_points_10cm', 'NA'),
                        ('plot_points_1cm', 'plot_points_1cm', 'NA'),
                        ('point_types', 'point_types', 'scanlines'),
                        ('n_points', 'nPeaks', -1),
                        ('scanline_height', 'scanSize', -1),
                        ('distances_all', 'dists', 'NA', 'tolist'),
                        ('sd', 'sd', -1),
                        ('conversion_factor_gmean', 'gmean', -1),
                        ('conversion_factor_mean', 'mean', -1),
                        ('unit', 'unit', 'NA')
                    ]

                    for attribute_name, key, default_value, *optional_method in attributes_to_assign:
                        value = self.safe_assign_value(ruler_data_dict, key, default_value)
                        if optional_method:
                            value = getattr(value, optional_method[0])()
                        setattr(self, attribute_name, value)

                    try:
                        self.conversion_factor = ruler_data_dict.conversion_factor # block
                    except:#scanlines
                        if (self.unit == 'mm') and (self.scanline_height != 'NA'):
                            self.conversion_factor = self.conversion_factor_gmean*10
                        elif (self.unit == 'cm') and (self.scanline_height != 'NA'):
                            self.conversion_factor = self.conversion_factor_gmean
                        else:
                            self.conversion_factor = self.conversion_factor_gmean

                    dict_ruler = {
                        'filename': [self.filename],
                        'image_height': [self.height],
                        'image_width': [self.width],
                        'ruler_image_name': [ruler_image_name],

                        'conversion_factor': [self.conversion_factor],
                        'conversion_determination': [self.conversion_determination],
                        'conversion_inputs': [self.conversion_inputs],

                        'ruler_class': [ruler_class],
                        'ruler_class_confidence': [float(ruler_class_confidence)],
                        'ruler_location': [ruler_location],

                        'plot_points': ['skipped'], # [self.plot_points],
                        'plot_points_1cm': ['skipped'], # [self.plot_points_1cm],
                        'plot_points_10cm': ['skipped'], # [self.plot_points_10cm],
                        'point_types': [self.point_types],

                        'n_points': [int(self.n_points)],
                        'scanline_height': [int(self.scanline_height)],
                        'distances_in_average': [self.distances_all],
                        'sd': [float(self.sd)],
                        'conversion_factor_gmean': [float(self.conversion_factor_gmean)],
                        'conversion_factor_mean': [float(self.conversion_factor_mean)],
                        'unit': [self.unit]
                    }
                    if not cfg['leafmachine']['data']['include_darwin_core_data_from_combined_file']:
                        self.ruler_dict_list.append(dict_ruler)
                        df_ruler = pd.concat([df_ruler, pd.DataFrame.from_dict(dict_ruler)], ignore_index=True)
                    else:
                        dict_ruler2 = dict_ruler.copy()
                        dict_ruler2.update(self.specimen_record)
                        self.ruler_dict_list.append(dict_ruler2)

                        dict_ruler = pd.DataFrame.from_dict(dict_ruler)
                        dict_ruler = pd.concat([dict_ruler, pd.DataFrame.from_dict(self.specimen_record)], axis = 1)
                        df_ruler = pd.concat([df_ruler, dict_ruler], ignore_index=True)

        self.df_ruler = df_ruler

        if cfg['leafmachine']['data']['save_json_rulers']:
            for r in range(len(self.ruler_dict_list)):
                dict_r = self.ruler_dict_list[r]
                name_json = '.'.join([dict_r['ruler_image_name'][0], 'json'])
                with open(os.path.join(Dirs.data_json_rulers, name_json), "w") as outfile:
                    json.dump(dict_r, outfile)

        if cfg['leafmachine']['data']['save_individual_csv_files_rulers']:
            self.df_ruler.to_csv(os.path.join(Dirs.data_csv_individual_rulers, '.'.join([self.filename, 'csv'])), header=True, index=False)

        
    def get_ruler_dataframe(self):
        return self.df_ruler

    def get_seg_dataframe(self):
        return self.df_seg

    def get_EFD_dataframe(self):
        return self.df_EFD
    
    def get_landmarks_dataframe(self):
        return self.df_landmarks
    
    def dict_to_json(self, dict_labels, dir_components, name_json):
        # dir_components = os.path.join(dir_components, 'json')
        with open(os.path.join(dir_components, '.'.join([name_json, 'json'])), "w") as outfile:
            json.dump(dict_labels, outfile)

    def divide_values_length(self, dict_seg, df_ruler_use):
        if (df_ruler_use['conversion_mean'][0] == 0) or (df_ruler_use['conversion_mean'][0] == 'NA'):
            return dict_seg
        else:
            # try:
            bbox_min_long_side = round(dict_seg['bbox_min_long_side'][0] / df_ruler_use['conversion_mean'][0], 2) if dict_seg['bbox_min_long_side'][0] is not None else None
            bbox_min_short_side = round(dict_seg['bbox_min_short_side'][0] / df_ruler_use['conversion_mean'][0], 2) if dict_seg['bbox_min_short_side'][0] is not None else None
            perimeter = round(dict_seg['perimeter'][0] / df_ruler_use['conversion_mean'][0], 2) if dict_seg['perimeter'][0] is not None else None
            
            if self.cfg['leafmachine']['leaf_segmentation']['calculate_elliptic_fourier_descriptors']:
                efd_perimeter = round(dict_seg['efd_perimeter'][0] / df_ruler_use['conversion_mean'][0], 2) if dict_seg['efd_perimeter'][0] is not None else None
            else:
                efd_perimeter = 0
            # except:
            #     bbox_min_long_side = round(dict_seg['bbox_min_long_side'][0] / df_ruler_use['conversion_mean'][0][0], 2) if dict_seg['bbox_min_long_side'][0] is not None else None
            #     bbox_min_short_side = round(dict_seg['bbox_min_short_side'][0] / df_ruler_use['conversion_mean'][0][0], 2) if dict_seg['bbox_min_short_side'][0] is not None else None
            #     perimeter = round(dict_seg['perimeter'][0] / df_ruler_use['conversion_mean'][0][0], 2) if dict_seg['perimeter'][0] is not None else None
            #     efd_perimeter = round(dict_seg['efd_perimeter'][0] / df_ruler_use['conversion_mean'][0][0], 2) if dict_seg['efd_perimeter'][0] is not None else None
            dict_seg['bbox_min_long_side'] = bbox_min_long_side
            dict_seg['bbox_min_short_side'] = bbox_min_short_side
            dict_seg['perimeter'] = perimeter
            dict_seg['efd_perimeter'] = efd_perimeter
            return dict_seg
    
    def divide_values_sq(self, dict_seg, df_ruler_use):
        if (df_ruler_use['conversion_mean'][0] == 0) or (df_ruler_use['conversion_mean'][0] == 'NA'):
            return dict_seg
        else:
            # try:
            if self.cfg['leafmachine']['leaf_segmentation']['calculate_elliptic_fourier_descriptors']:
                efd_area = round(dict_seg['efd_area'][0] / (df_ruler_use['conversion_mean'][0] * df_ruler_use['conversion_mean'][0]), 2) if dict_seg['efd_area'][0] is not None else None
            else:
                efd_area = 0
            area = round(dict_seg['area'][0] / (df_ruler_use['conversion_mean'][0] * df_ruler_use['conversion_mean'][0]), 2) if dict_seg['area'][0] is not None else None
            convex_hull = round(dict_seg['convex_hull'][0] / (df_ruler_use['conversion_mean'][0] * df_ruler_use['conversion_mean'][0]), 2) if dict_seg['convex_hull'][0] is not None else None
            # except:
            #     efd_area = round(dict_seg['efd_area'][0] / (df_ruler_use['conversion_mean'][0][0] * df_ruler_use['conversion_mean'][0][0]), 2) if dict_seg['efd_area'][0] is not None else None
            #     area = round(dict_seg['area'][0] / (df_ruler_use['conversion_mean'][0][0] * df_ruler_use['conversion_mean'][0][0]), 2) if dict_seg['area'][0] is not None else None
            #     convex_hull = round(dict_seg['convex_hull'][0] / (df_ruler_use['conversion_mean'][0][0] * df_ruler_use['conversion_mean'][0][0]), 2) if dict_seg['convex_hull'][0] is not None else None

            dict_seg['efd_area'] = [efd_area]
            dict_seg['area'] = [area]
            dict_seg['convex_hull'] = [convex_hull]
            return dict_seg

def merge_csv_files(Dirs, cfg):
    run_name = Dirs.run_name
    
    merge_and_save_csv_files(Dirs, 'data_csv_project_batch_ruler', '_RULER', run_name)
    merge_and_save_csv_files(Dirs, 'data_csv_project_batch_measurements', '_MEASUREMENTS', run_name)

    if cfg['leafmachine']['leaf_segmentation']['calculate_elliptic_fourier_descriptors']:
        merge_and_save_csv_files(Dirs, 'data_csv_project_batch_EFD', '_EFD', run_name)

    if cfg['leafmachine']['landmark_detector']['landmark_whole_leaves'] or cfg['leafmachine']['landmark_detector']['landmark_partial_leaves']:
        merge_and_save_csv_files(Dirs, 'data_csv_project_batch_landmarks', '_LANDMARKS', run_name)
    
def merge_and_save_csv_files(Dirs, dir_attribute, output_suffix, run_name):
    try:
        files = [f for f in os.listdir(getattr(Dirs, dir_attribute)) if f.endswith('.csv')]
    except:
        files = [f for f in os.listdir(Dirs) if f.endswith('.csv')]

    try:
        df_list = [pd.read_csv(os.path.join(getattr(Dirs, dir_attribute), f)) for f in files]
    except:
        df_list = [pd.read_csv(os.path.join(Dirs, f)) for f in files]
        
    df_merged = pd.concat(df_list, ignore_index=True)
    
    try:
        df_merged.to_csv(os.path.join(getattr(Dirs, dir_attribute.replace("_batch", "").strip()), ''.join([run_name, output_suffix, '.csv'])), index=False)
    except:
        alt_run_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(Dirs)))))
        df_merged.to_csv(os.path.join(Dirs, ''.join([alt_run_name, output_suffix, '.csv'])), index=False)


if __name__ == '__main__':
    merge_csv_files('/home/brlab/Dropbox/LM2_Env/Image_Datasets/TEST_LM2/diospyros_10perSpecies_memory/Data/Project/Batch/Measurements', None)