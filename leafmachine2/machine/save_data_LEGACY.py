
from __future__ import annotations
import os, sys, inspect, json, imagesize, shutil
import pandas as pd
from dataclasses import dataclass, field
from time import perf_counter


def save_data(cfg, logger, dir_home, Project, batch, n_batches, Dirs):
    start_t = perf_counter()
    logger.name = f'[BATCH {batch+1} Save Data]'
    logger.info(f'Saving data for {batch+1} of {n_batches}')

    n_order = cfg['leafmachine']['leaf_segmentation']['elliptic_fourier_descriptor_order'] if cfg['leafmachine']['leaf_segmentation']['elliptic_fourier_descriptor_order'] else 40
    # Generate the column names for efd_coeffs_features
    coeffs_col_names = [f'coeffs_{i}' for i in range(n_order)]

    # Create a new DataFrame with the required columns
    df_project_EFD = pd.DataFrame(columns=['filename', 'height', 'width','seg_image_name','annotation_name',
            'efd_order','efd_coeffs_features','efd_a0','efd_c0','efd_scale','efd_angle',
            'efd_phase','efd_area','efd_perimeter','efd_plot_points',] + coeffs_col_names)

    df_project_rulers = pd.DataFrame(columns=['filename', 'ruler_image_name', 'ruler_location', 'height', 'width',
            'success', 'conversion_mean', 'pooled_sd', 'ruler_class', 'ruler_class_confidence', 
            'units', 'cross_validation_count' ,'n_scanlines' ,'n_data_points_in_avg', 
            'avg_tick_width',])

    seg_column_names = ['filename', 'height', 'width',

            'seg_image_name','annotation_name','bbox','bbox_min','rotate_angle','bbox_min_long_side','bbox_min_short_side',
            'area','perimeter','centroid','convex_hull','convexity', 'concavity',
            'circularity','degree','aspect_ratio','polygon_closed','polygon_closed_rotated',
            'efd_order','efd_coeffs_features','efd_a0','efd_c0','efd_scale','efd_angle',
            'efd_phase','efd_area','efd_perimeter','efd_plot_points',

            'ruler_image_name', 'success','conversion_mean','pooled_sd','ruler_class',
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
        df_project_rulers = pd.concat([df_project_rulers, df_ruler], ignore_index=True)
        df_project_seg = pd.concat([df_project_seg, df_seg], ignore_index=True)
        df_project_EFD = pd.concat([df_project_EFD, df_EFD], ignore_index=True)


    df_project_rulers.to_csv(os.path.join(Dirs.data_csv_project_batch_ruler, '.'.join([''.join([Dirs.run_name, '__Ruler__', str(batch+1), 'of', str(n_batches)]), 'csv'])), header=True, index=False)
    df_project_seg.to_csv(os.path.join(Dirs.data_csv_project_batch_measurements, '.'.join([''.join([Dirs.run_name, '__Measurements__', str(batch+1), 'of', str(n_batches)]), 'csv'])), header=True, index=False)
    df_project_EFD.to_csv(os.path.join(Dirs.data_csv_project_batch_EFD, '.'.join([''.join([Dirs.run_name, '__EFD__', str(batch+1), 'of', str(n_batches)]), 'csv'])), header=True, index=False)

    end_t = perf_counter()
    logger.info(f'Batch {batch+1}: Save Data Duration --> {round((end_t - start_t)/60)} minutes')


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

    csv_img: str = ''
    df_ruler_use: list[str] = field(default_factory=list)

    def __init__(self, cfg, logger, filename, analysis, Dirs) -> None:
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

        self.gather_seg_data(cfg, logger, Dirs, self.seg_whole_leaf, self.seg_partial_leaf, df_ruler_use)
        self.gather_EFD_data(cfg, logger, Dirs, self.seg_whole_leaf, self.seg_partial_leaf)
        # self.gather_landmark_data(cfg, logger, Dirs, self.seg_whole_leaf, self.seg_partial_leaf)

    def get_key_value(self, dictionary, key, default_value=[]):
        return dictionary.get(key, default_value)
        
    def gather_EFD_data(self, cfg, logger, Dirs, seg_whole_leaf, seg_partial_leaf):
        seg_data_list = [seg_whole_leaf, seg_partial_leaf]

        ruler_column_names = ['filename', 'height', 'width','seg_image_name','annotation_name',
            'efd_order','efd_coeffs_features','efd_a0','efd_c0','efd_scale','efd_angle',
            'efd_phase','efd_area','efd_perimeter','efd_plot_points',]

        if cfg['leafmachine']['data']['include_darwin_core_data_from_combined_file']:
            record_column_names = list(self.specimen_record.keys())
            column_names = ruler_column_names + record_column_names
        else:
            column_names = ruler_column_names

        self.df_EFD = pd.DataFrame(columns= column_names)
        self.seg_dict_list = []
        if (seg_data_list[0] == []) and (seg_data_list[1] == []): # No leaf segmentaitons
            self.df_EFD = self.df_EFD
        else:
            for leaf_type_data in seg_data_list:
                for i in range(len(leaf_type_data)):
                    # Ruler Info
                    seg_info_part = leaf_type_data[i]
                    seg_image_name, seg_info_dict = next(iter(seg_info_part.items())) 
                    # print(seg_image_name)
                    logger.debug(f"[Leaf EFD] {seg_image_name}")
                    
                    if seg_info_dict == []:                      
                        n_order = cfg['leafmachine']['leaf_segmentation']['elliptic_fourier_descriptor_order'] if cfg['leafmachine']['leaf_segmentation']['elliptic_fourier_descriptor_order'] else 40
                        coeffs_col_names = [f'coeffs_{i}' for i in range(n_order)]  
                        
                        dict_EFD = {
                            'filename': [self.filename], # ruler data
                            'height': [self.height], # ruler data
                            'width': [self.width], # ruler data

                            'seg_image_name': ['NA'],
                            'annotation_name': ['NA'],
                            
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
                            # **{col['name']: [col['value']] for col in coeffs_col_names},
                        }
                        # Add a column for each entry in efd_coeffs_features
                        for i in range(len(coeffs_col_names)):
                            dict_EFD[coeffs_col_names[i]] = ['NA']

                        self.seg_dict_list.append(dict_EFD)
                        self.df_EFD = pd.concat([self.df_EFD, pd.DataFrame.from_dict(dict_EFD)], ignore_index=True)

                    else:
                        for annotation in seg_info_dict:
                            annotation_name, annotation_dict = next(iter(annotation.items())) 
                            # print(annotation_name)
                            logger.debug(f'[Annotation] {annotation_name}')
                            # print(annotation_dict)
                            n_order = cfg['leafmachine']['leaf_segmentation']['elliptic_fourier_descriptor_order'] if cfg['leafmachine']['leaf_segmentation']['elliptic_fourier_descriptor_order'] else 40
                            coeffs_col_names = [f'coeffs_{i}' for i in range(n_order)]                           
                            
                            dict_EFD = {
                                'filename': [self.filename], # ruler data
                                'height': [self.height], # ruler data
                                'width': [self.width], # ruler data

                                'seg_image_name': [seg_image_name],
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
                            }
                            # Add a column for each entry in efd_coeffs_features
                            for i, coeffs in enumerate(annotation_dict['efds']['coeffs_normalized']):
                                dict_EFD[coeffs_col_names[i]] = [coeffs]

                            self.seg_dict_list.append(dict_EFD)
                            self.df_EFD = pd.concat([self.df_EFD, pd.DataFrame.from_dict(dict_EFD)], ignore_index=True)
            
            if cfg['leafmachine']['data']['save_json_measurements']:
                anno_list = self.seg_dict_list[0]
                try:
                    name_json = ''.join([anno_list['seg_image_name'], '_EFD', '.json'])
                except:
                    name_json = ''.join([anno_list['seg_image_name'][0], '_EFD', '.json'])
                with open(os.path.join(Dirs.data_json_measurements, name_json), "w") as outfile:
                    json.dump(self.seg_dict_list, outfile)

            if cfg['leafmachine']['data']['save_individual_csv_files_measurements']:
                try:
                    self.df_EFD.to_csv(os.path.join(Dirs.data_csv_individual_measurements, ''.join([self.filename, '_EFD','.csv'])), header=True, index=False)
                except: 
                    self.df_EFD.to_csv(os.path.join(Dirs.data_csv_individual_measurements, ''.join([self.filename[0], '_EFD','.csv'])), header=True, index=False)

    def gather_seg_data(self, cfg, logger, Dirs, seg_whole_leaf, seg_partial_leaf, df_ruler_use):
        seg_data_list = [seg_whole_leaf, seg_partial_leaf]

        ruler_column_names = ['filename', 'height', 'width',

            'seg_image_name','annotation_name','bbox','bbox_min','rotate_angle','bbox_min_long_side','bbox_min_short_side',
            'area','perimeter','centroid','convex_hull','convexity', 'concavity',
            'circularity','degree','aspect_ratio','polygon_closed','polygon_closed_rotated',
            'efd_order','efd_coeffs_features','efd_a0','efd_c0','efd_scale','efd_angle',
            'efd_phase','efd_area','efd_perimeter','efd_plot_points',

            'ruler_image_name', 'success','conversion_mean','pooled_sd','ruler_class',
            'ruler_class_confidence','units', 'cross_validation_count','n_scanlines','n_data_points_in_avg','avg_tick_width',]

        if cfg['leafmachine']['data']['include_darwin_core_data_from_combined_file']:
            record_column_names = list(self.specimen_record.keys())
            column_names = ruler_column_names + record_column_names
        else:
            column_names = ruler_column_names

        df_seg = pd.DataFrame(columns= column_names)
        self.seg_dict_list = []
        if (seg_data_list[0] == []) and (seg_data_list[1] == []): # No leaf segmentaitons
            self.df_seg = df_seg
        else:
            for leaf_type_data in seg_data_list:
                for i in range(len(leaf_type_data)):
                    # Ruler Info
                    seg_info_part = leaf_type_data[i]
                    seg_image_name, seg_info_dict = next(iter(seg_info_part.items())) 
                    # print(seg_image_name)
                    logger.debug(f"[Leaf] {seg_image_name}")


                    
                    if seg_info_dict == []:                      
                        if isinstance(df_ruler_use['ruler_image_name'], (list, dict)):
                            ruler_image_name = [df_ruler_use['ruler_image_name'][0]]
                        else:
                            ruler_image_name = [df_ruler_use['ruler_image_name']]

                        if isinstance(df_ruler_use['success'], (list, dict)):
                            success = [df_ruler_use['success'][0]]
                        else:
                            success = [df_ruler_use['success']]

                        if isinstance(df_ruler_use['conversion_mean'], (list, dict)):
                            conversion_mean = [df_ruler_use['conversion_mean'][0]]
                        else:
                            conversion_mean = [df_ruler_use['conversion_mean']]

                        if isinstance(df_ruler_use['pooled_sd'], (list, dict)):
                            pooled_sd = [df_ruler_use['pooled_sd'][0]]
                        else:
                            pooled_sd = [df_ruler_use['pooled_sd']]

                        if isinstance(df_ruler_use['ruler_class'], (list, dict)):
                            ruler_class = [df_ruler_use['ruler_class'][0]]
                        else:
                            ruler_class = [df_ruler_use['ruler_class']]

                        if isinstance(df_ruler_use['ruler_class_confidence'], (list, dict)):
                            ruler_class_confidence = [df_ruler_use['ruler_class_confidence'][0]]
                        else:
                            ruler_class_confidence = [df_ruler_use['ruler_class_confidence']]

                        if isinstance(df_ruler_use['units'], (list, dict)) and df_ruler_use['units'] != []:
                            units = [df_ruler_use['units'][0]]
                        else:
                            units = [df_ruler_use['units']]

                        if isinstance(df_ruler_use['cross_validation_count'], (list, dict)):
                            cross_validation_count = [df_ruler_use['cross_validation_count'][0]]
                        else:
                            cross_validation_count = [df_ruler_use['cross_validation_count']]

                        if isinstance(df_ruler_use['n_scanlines'], (list, dict)):
                            n_scanlines = [df_ruler_use['n_scanlines'][0]]
                        else:
                            n_scanlines = [df_ruler_use['n_scanlines']]

                        if isinstance(df_ruler_use['n_data_points_in_avg'], (list, dict)):
                            n_data_points_in_avg = [df_ruler_use['n_data_points_in_avg'][0]]
                        else:
                            n_data_points_in_avg = [df_ruler_use['n_data_points_in_avg']]

                        if isinstance(df_ruler_use['avg_tick_width'], (list, dict)):
                            avg_tick_width = [df_ruler_use['avg_tick_width'][0]]
                        else:
                            avg_tick_width = [df_ruler_use['avg_tick_width']]

                        dict_seg = {
                            'filename': [self.filename], # ruler data
                            'height': [self.height], # ruler data
                            'width': [self.width], # ruler data

                            'seg_image_name': ['NA'],
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
                            'degree': ['NA'],
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
                            'ruler_image_name': ruler_image_name,
                            'success': success,
                            'conversion_mean': conversion_mean,
                            'pooled_sd':pooled_sd,

                            'ruler_class': ruler_class,
                            'ruler_class_confidence': ruler_class_confidence,
                            
                            'units': units,
                            'cross_validation_count': cross_validation_count,
                            'n_scanlines': n_scanlines,
                            'n_data_points_in_avg': n_data_points_in_avg,
                            'avg_tick_width': avg_tick_width,
                        }
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

                            if isinstance(df_ruler_use['ruler_image_name'], (list, dict)):
                                ruler_image_name = [df_ruler_use['ruler_image_name'][0]]
                            else:
                                ruler_image_name = [df_ruler_use['ruler_image_name']]
                            
                            if isinstance(df_ruler_use['success'], (list, dict)):
                                success = [df_ruler_use['success'][0]]
                            else:
                                success = [df_ruler_use['success']]

                            if isinstance(df_ruler_use['conversion_mean'], (list, dict)):
                                conversion_mean = [df_ruler_use['conversion_mean'][0]]
                            else:
                                conversion_mean = [df_ruler_use['conversion_mean']]

                            if isinstance(df_ruler_use['pooled_sd'], (list, dict)):
                                pooled_sd = [df_ruler_use['pooled_sd'][0]]
                            else:
                                pooled_sd = [df_ruler_use['pooled_sd']]

                            if isinstance(df_ruler_use['ruler_class'], (list, dict)):
                                ruler_class = [df_ruler_use['ruler_class'][0]]
                            else:
                                ruler_class = [df_ruler_use['ruler_class']]

                            if isinstance(df_ruler_use['ruler_class_confidence'], (list, dict)):
                                ruler_class_confidence = [df_ruler_use['ruler_class_confidence'][0]]
                            else:
                                ruler_class_confidence = [df_ruler_use['ruler_class_confidence']]

                            if isinstance(df_ruler_use['units'], (list, dict)) and df_ruler_use['units'] != []:
                                units = [df_ruler_use['units'][0]]
                            else:
                                units = [df_ruler_use['units']]

                            if isinstance(df_ruler_use['cross_validation_count'], (list, dict)):
                                cross_validation_count = [df_ruler_use['cross_validation_count'][0]]
                            else:
                                cross_validation_count = [df_ruler_use['cross_validation_count']]

                            if isinstance(df_ruler_use['n_scanlines'], (list, dict)):
                                n_scanlines = [df_ruler_use['n_scanlines'][0]]
                            else:
                                n_scanlines = [df_ruler_use['n_scanlines']]

                            if isinstance(df_ruler_use['n_data_points_in_avg'], (list, dict)):
                                n_data_points_in_avg = [df_ruler_use['n_data_points_in_avg'][0]]
                            else:
                                n_data_points_in_avg = [df_ruler_use['n_data_points_in_avg']]

                            if isinstance(df_ruler_use['avg_tick_width'], (list, dict)):
                                avg_tick_width = [df_ruler_use['avg_tick_width'][0]]
                            else:
                                avg_tick_width = [df_ruler_use['avg_tick_width']]

                            dict_seg = {
                                'filename': [self.filename], # ruler data
                                'height': [self.height], # ruler data
                                'width': [self.width], # ruler data

                                'seg_image_name': [seg_image_name],
                                'annotation_name': [annotation_name],
                                'bbox': [annotation_dict['bbox']],
                                'bbox_min': [annotation_dict['bbox_min']],
                                'rotate_angle': annotation_dict['rotate_angle'],
                                'bbox_min_long_side': [annotation_dict['long']],
                                'bbox_min_short_side': [annotation_dict['short']],
                                'area': [annotation_dict['area']],
                                'perimeter': [annotation_dict['perimeter']],
                                'centroid': [annotation_dict['centroid']],
                                'convex_hull': [annotation_dict['convex_hull']],
                                'convexity': [annotation_dict['convexity']],
                                'concavity': [annotation_dict['concavity']],
                                'circularity': [annotation_dict['circularity']],
                                'degree': [annotation_dict['degree']],
                                'aspect_ratio': [annotation_dict['aspect_ratio']],
                                'polygon_closed': ['too_long'], #[out_polygon_closed],
                                'polygon_closed_rotated': ['too_long'], #[out_polygon_closed_rotated],

                                'efd_order': [int(annotation_dict['efds']['coeffs_normalized'].shape[0])],
                                'efd_coeffs_features': ['too_long'], #[annotation_dict['efds']['coeffs_features'].tolist()],
                                'efd_a0': [float(annotation_dict['efds']['a0'])],
                                'efd_c0': [float(annotation_dict['efds']['c0'])],
                                'efd_scale': [float(annotation_dict['efds']['scale'])],
                                'efd_angle': [float(annotation_dict['efds']['angle'])],
                                'efd_phase': [float(annotation_dict['efds']['phase'])],
                                'efd_area': [float(annotation_dict['efds']['efd_area'])],
                                'efd_perimeter': [float(annotation_dict['efds']['efd_perimeter'])],
                                'efd_plot_points': ['too_long'], #[annotation_dict['efds']['efd_pts_PIL']],

                                # All from ruler data
                                'ruler_image_name': ruler_image_name,
                                'success': success,
                                'conversion_mean': conversion_mean,
                                'pooled_sd':pooled_sd,

                                'ruler_class': ruler_class,
                                'ruler_class_confidence': ruler_class_confidence,
                                
                                'units': units,
                                'cross_validation_count': cross_validation_count,
                                'n_scanlines': n_scanlines,
                                'n_data_points_in_avg': n_data_points_in_avg,
                                'avg_tick_width': avg_tick_width,
                            }
                            # Apply conversion factor to the values in the dict
                            if cfg['leafmachine']['data']['do_apply_conversion_factor']:
                                # try:
                                dict_seg = self.divide_values_length(dict_seg)
                                # except:
                                    # pass
                                # try:
                                dict_seg = self.divide_values_sq(dict_seg)
                                # except:
                                    # pass
                                
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
                    # name_json_part = ''.join([dict_r['seg_image_name'], anno_name])
                anno_list = self.seg_dict_list[0]
                try:
                    name_json = '.'.join([anno_list['seg_image_name'], 'json'])
                except:
                    name_json = '.'.join([anno_list['seg_image_name'][0], 'json'])
                with open(os.path.join(Dirs.data_json_measurements, name_json), "w") as outfile:
                    json.dump(self.seg_dict_list, outfile)

            if cfg['leafmachine']['data']['save_individual_csv_files_measurements']:
                try:
                    self.df_seg.to_csv(os.path.join(Dirs.data_csv_individual_measurements, '.'.join([df_ruler_use['filename'], 'csv'])), header=True, index=False)
                except: 
                    self.df_seg.to_csv(os.path.join(Dirs.data_csv_individual_measurements, '.'.join([df_ruler_use['filename'][0], 'csv'])), header=True, index=False)

    def gather_ruler_info(self, cfg, Dirs, ruler_info):
        ruler_column_names = ['filename', 'ruler_image_name', 'ruler_location', 'height', 'width',
            'success', 'conversion_mean', 'pooled_sd', 'ruler_class', 'ruler_class_confidence', 
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
                'ruler_image_name': ['no_ruler'],
                'ruler_location': ['NA'],
                'height': [self.height],
                'width': [self.width],

                'success': ['False'],
                'conversion_mean': ['NA'],
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
                        'ruler_image_name': ['no_ruler'],
                        'ruler_location': ['NA'],
                        'height': [self.height],
                        'width': [self.width],

                        'success': ['False'],
                        'conversion_mean': ['NA'],
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

                    if isinstance(ruler_data_part['success'], (list, dict)):
                        success = [ruler_data_part['success'][0]]
                    else:
                        success = [ruler_data_part['success']]

                    if isinstance(ruler_data_part['conversion_mean'], (list, dict)):
                        conversion_mean = [ruler_data_part['conversion_mean'][0]]
                    else:
                        conversion_mean = [ruler_data_part['conversion_mean']]

                    if isinstance(ruler_data_part['pooled_sd'], (list, dict)):
                        pooled_sd = [ruler_data_part['pooled_sd'][0]]
                    else:
                        pooled_sd = [ruler_data_part['pooled_sd']]

                    if isinstance(ruler_data_part['ruler_class'], (list, dict)):
                        ruler_class = [ruler_data_part['ruler_class'][0]]
                    else:
                        ruler_class = [ruler_data_part['ruler_class']]

                    if isinstance(ruler_data_part['ruler_class_confidence'], (list, dict)):
                        ruler_class_confidence = [ruler_data_part['ruler_class_confidence'][0]]
                    else:
                        ruler_class_confidence = [ruler_data_part['ruler_class_confidence']]

                    if isinstance(ruler_data_part['units'], (list, dict)) and ruler_data_part['units'] != []:
                        units = [ruler_data_part['units'][0]]
                    else:
                        units = [ruler_data_part['units']]

                    if isinstance(ruler_data_part['cross_validation_count'], (list, dict)):
                        cross_validation_count = [ruler_data_part['cross_validation_count'][0]]
                    else:
                        cross_validation_count = [ruler_data_part['cross_validation_count']]

                    if isinstance(ruler_data_part['n_scanlines'], (list, dict)):
                        n_scanlines = [ruler_data_part['n_scanlines'][0]]
                    else:
                        n_scanlines = [ruler_data_part['n_scanlines']]

                    if isinstance(ruler_data_part['n_data_points_in_avg'], (list, dict)):
                        n_data_points_in_avg = [ruler_data_part['n_data_points_in_avg'][0]]
                    else:
                        n_data_points_in_avg = [ruler_data_part['n_data_points_in_avg']]

                    if isinstance(ruler_data_part['avg_tick_width'], (list, dict)):
                        avg_tick_width = [ruler_data_part['avg_tick_width'][0]]
                    else:
                        avg_tick_width = [ruler_data_part['avg_tick_width']]
                        
                    dict_ruler = {
                        'filename': [self.filename],
                        'ruler_image_name': [ruler_image_name],
                        'ruler_location': [ruler_location],
                        'height': [self.height],
                        'width': [self.width],
                        
                        # All from ruler data
                        'success': success,
                        'conversion_mean': conversion_mean,
                        'pooled_sd':pooled_sd,

                        'ruler_class': ruler_class,
                        'ruler_class_confidence': ruler_class_confidence,
                        
                        'units': units,
                        'cross_validation_count': cross_validation_count,
                        'n_scanlines': n_scanlines,
                        'n_data_points_in_avg': n_data_points_in_avg,
                        'avg_tick_width': avg_tick_width,
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

    def gather_ruler_data(self, cfg, Dirs, ruler_info, ruler_data):
        ruler_column_names = ['filename', 'ruler_image_name', 'height', 'width',
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
                'ruler_image_name': ['no_ruler'],
                'height': [self.height],
                'width': [self.width],

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
                        'ruler_image_name': [ruler_image_name],
                        'height': [self.height],
                        'width': [self.width],

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

                    try:
                        self.conversion_determination = ruler_data_dict.conversion_location# block
                    except:
                        self.conversion_determination = 'scanlines'#scanlines
                    
                    try:
                        self.conversion_inputs = ruler_data_dict.conversion_location_options# block
                    except:
                        self.conversion_inputs = 'scanlines'# scanlines

                    
                    try:
                        self.plot_points_10cm = ruler_data_dict.plot_points_10cm# block
                    except:
                        self.plot_points_10cm = 'NA'# scanlines

                    try:
                        self.plot_points_1cm = ruler_data_dict.plot_points_1cm# block
                    except:
                        self.plot_points_1cm = 'NA'# scanlines
                    
                    try:
                        self.point_types = ruler_data_dict.point_types# block
                    except:
                        self.point_types = 'scanlines'# scanlines
                    
                    try:
                        self.n_points = ruler_data_dict['nPeaks'] #scanlines
                    except:
                        self.n_points = -1 # block
                    
                    try:
                        self.scanline_height = ruler_data_dict['scanSize']#scanlines
                    except:
                        self.scanline_height = -1 # block
                    
                    try:
                        self.distances_all = ruler_data_dict['dists'].tolist()#scanlines
                    except:
                        self.distances_all = 'NA' # block
                    
                    try:
                        self.sd = ruler_data_dict['sd']#scanlines
                    except:
                        self.sd = -1 # block
                    
                    try:
                        self.conversion_factor_gmean = ruler_data_dict['gmean']#scanlines
                    except:
                        self.conversion_factor_gmean = -1 # block
                    
                    try:
                        self.conversion_factor_mean = ruler_data_dict['mean']#scanlines
                    except:
                        self.conversion_factor_mean = -1 # block
                    
                    try:
                        self.unit = ruler_data_dict['unit']#scanlines
                    except:
                        self.unit = 'NA' # block

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
                        'ruler_image_name': [ruler_image_name],
                        'height': [self.height],
                        'width': [self.width],

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


    def dict_to_json(self, dict_labels, dir_components, name_json):
        # dir_components = os.path.join(dir_components, 'json')
        with open(os.path.join(dir_components, '.'.join([name_json, 'json'])), "w") as outfile:
            json.dump(dict_labels, outfile)

    def divide_values_length(self, dict_seg):
        if (dict_seg['conversion_mean'][0] == 0) or (dict_seg['conversion_mean'][0] == 'NA'):
            return dict_seg
        else:
            bbox_min_long_side = dict_seg['bbox_min_long_side'][0] / dict_seg['conversion_mean'][0]
            bbox_min_short_side = dict_seg['bbox_min_short_side'][0] / dict_seg['conversion_mean'][0]
            perimeter = dict_seg['perimeter'][0] / dict_seg['conversion_mean'][0]
            efd_perimeter = dict_seg['efd_perimeter'][0] / dict_seg['conversion_mean'][0]

            dict_seg['bbox_min_long_side'] = [bbox_min_long_side]
            dict_seg['bbox_min_short_side'] = [bbox_min_short_side]
            dict_seg['perimeter'] = [perimeter]
            dict_seg['efd_perimeter'] = [efd_perimeter]
            return dict_seg
    
    def divide_values_sq(self, dict_seg):
        if (dict_seg['conversion_mean'][0] == 0) or (dict_seg['conversion_mean'][0] == 'NA'):
            return dict_seg
        else:
            efd_area = dict_seg['efd_area'][0] / (dict_seg['conversion_mean'][0] * dict_seg['conversion_mean'][0])
            area = dict_seg['area'][0] / (dict_seg['conversion_mean'][0] * dict_seg['conversion_mean'][0])
            convex_hull = dict_seg['convex_hull'][0] / (dict_seg['conversion_mean'][0] * dict_seg['conversion_mean'][0])

            dict_seg['efd_area'] = [efd_area]
            dict_seg['area'] = [area]
            dict_seg['convex_hull'] = [convex_hull]
            return dict_seg

def merge_csv_files(Dirs, cfg):
    '''Ruler'''
    # Get a list of all csv files in the source directory
    try:
        files = [f for f in os.listdir(Dirs.data_csv_project_batch_ruler) if f.endswith('.csv')]
    except:
        files = [f for f in os.listdir(Dirs) if f.endswith('.csv')]

    
    # Read the data from each file into a dataframe
    try:
        df_list = [pd.read_csv(os.path.join(Dirs.data_csv_project_batch_ruler, f)) for f in files]
    except:
        df_list = [pd.read_csv(os.path.join(Dirs, f)) for f in files]

    
    # Concatenate all the dataframes into a single dataframe
    df_merged = pd.concat(df_list, ignore_index=True)
    
    # Save the merged dataframe to a new csv file
    try:
        df_merged.to_csv(os.path.join(Dirs.data_csv_project_ruler, '.'.join([Dirs.run_name, 'csv'])), index=False)
    except:
        run_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(Dirs)))))
        df_merged.to_csv(os.path.join(Dirs, ''.join([run_name,'__RULER', '.csv'])), index=False)


    '''Measurements'''
    # Get a list of all csv files in the source directory
    try:
        files = [f for f in os.listdir(Dirs.data_csv_project_batch_measurements) if f.endswith('.csv')]
    except:
        files = [f for f in os.listdir(Dirs) if f.endswith('.csv')]

    
    # Read the data from each file into a dataframe
    try:
        df_list = [pd.read_csv(os.path.join(Dirs.data_csv_project_batch_measurements, f)) for f in files]
    except:
        df_list = [pd.read_csv(os.path.join(Dirs, f)) for f in files]

    
    # Concatenate all the dataframes into a single dataframe
    df_merged = pd.concat(df_list, ignore_index=True)
    
    # Save the merged dataframe to a new csv file
    try:
        df_merged.to_csv(os.path.join(Dirs.data_csv_project_measurements, '.'.join([Dirs.run_name, 'csv'])), index=False)
    except:
        run_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(Dirs)))))
        df_merged.to_csv(os.path.join(Dirs, ''.join([run_name, '_MEASUREMENTS', '.csv'])), index=False)

    '''EFD'''
    if cfg['leafmachine']['leaf_segmentation']['calculate_elliptic_fourier_descriptors']:
    # Get a list of all csv files in the source directory
        try:
            files = [f for f in os.listdir(Dirs.data_csv_project_batch_EFD) if f.endswith('.csv')]
        except:
            files = [f for f in os.listdir(Dirs) if f.endswith('.csv')]

        
        # Read the data from each file into a dataframe
        try:
            df_list = [pd.read_csv(os.path.join(Dirs.data_csv_project_batch_EFD, f)) for f in files]
        except:
            df_list = [pd.read_csv(os.path.join(Dirs, f)) for f in files]

        
        # Concatenate all the dataframes into a single dataframe
        df_merged = pd.concat(df_list, ignore_index=True)
        
        # Save the merged dataframe to a new csv file
        try:
            df_merged.to_csv(os.path.join(Dirs.data_csv_project_EFD,''.join([Dirs.run_name, '_EFD', '.csv'])), index=False)
        except:
            run_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(Dirs)))))
            df_merged.to_csv(os.path.join(Dirs, ''.join([run_name, '_EFD', '.csv'])), index=False)

if __name__ == '__main__':
    merge_csv_files('/home/brlab/Dropbox/LM2_Env/Image_Datasets/TEST_LM2/diospyros_10perSpecies_memory/Data/Project/Batch/Measurements', None)