from __future__ import annotations
import os, inspect, sys
from dataclasses import dataclass, field
currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)
from leafmachine2.machine.general_utils import validate_dir, get_datetime

@dataclass
class Config():
    
    verbose: bool = True
    optional_warnings: bool = True

    detect_ruler_type: bool = True
    ruler_overlay: bool = True
    ruler_processed: bool = True
    ruler_type_overlay: bool = True
    ruler_data: bool = True

    # Project Output Dir
    dir_output: str = 'LeafMachine2_Output' 
    run_name: str = 'Run'
    
    image_location: str = 'local' 
    GBIF_mode: str = 'all' 

    dir_images_local: str = None 
    path_combined_csv_local: str = None  
    path_occurrence_csv_local: str = None 
    path_images_csv_local: str = None
    
    # Configure Plant Component Detector
    detector_type: str = 'Plant_Detector'
    detector_version: str = 'PLANT_GroupAB_200'
    detector_iteration: str = 'PLANT_GroupAB_200'
    detector_weights: str = 'best.pt'
    minimum_confidence_threshold: float = 0.5
    do_save_prediction_overlay_images: bool = False
    ignore_objects_for_overlay: list[str] = field(default_factory=list) 
    
    # Configure Archival Component Detector
    detector_type: str = 'Archival_Detector' 
    detector_version: str = 'PREP_final'
    detector_iteration: str = 'PREP_final'
    detector_weights: str = 'best.pt'
    minimum_confidence_threshold:float =  0.5
    do_save_prediction_overlay_images: bool = False
    ignore_objects_for_overlay:  list[str] = field(default_factory=list) 

    # Configure Archival Component Detector
    segment_whole_leaves: bool = True
    segment_partial_leaves: bool = True

    keep_only_best_one_leaf_one_petiole: bool = True
    save_segmentation_overlay_images_to_pdf: bool = True
    save_individual_overlay_images: bool = True 
    overlay_line_width: int = 1

    save_masks_color: bool = True
    save_masks_index: bool = True

    calculate_elliptic_fourier_descriptors: bool = True 
    elliptic_fourier_descriptor_order: int = 40 
    
    segmentation_model: str = 'Group1_Dataset_20000_Iter_1176PTS_512Batch_smooth_l1_LR0005_BGR'
    minimum_confidence_threshold: float = 0.5 
    generate_overlay: bool = True
    overlay_dpi: int = 100 
    overlay_background_color: str = 'black' 






    def __init__(self, cfg) -> None:
        self.ignore_objects_for_overlay = ['leaf_partial']


        cfg['leafmachine']['print']['verbose']
        cfg['leafmachine']['print']['optional_warnings']
        verbose: bool = True
        optional_warnings: bool = True

        # detect_ruler_type: bool = True
        # ruler_overlay: bool = True
        # ruler_processed: bool = True
        # ruler_type_overlay: bool = True
        # ruler_data: bool = True

        # Project Output Dir
        cfg['leafmachine']['project']['dir_output']
        cfg['leafmachine']['project']['run_name']
        dir_output: str = 'LeafMachine2_Output' 
        run_name: str = 'Run'
        cfg['leafmachine']['project']['image_location']
        cfg['leafmachine']['project']['GBIF_mode']
        image_location: str = 'local' 
        GBIF_mode: str = 'all' 
        cfg['leafmachine']['project']['dir_images_local']
        cfg['leafmachine']['project']['path_combined_csv_local']
        cfg['leafmachine']['project']['path_occurrence_csv_local']
        cfg['leafmachine']['project']['path_images_csv_local']
        dir_images_local: str = None 
        path_combined_csv_local: str = None  
        path_occurrence_csv_local: str = None 
        path_images_csv_local: str = None
        
        # Configure Plant Component Detector
        cfg['leafmachine']['plant_component_detector']['detector_type']
        cfg['leafmachine']['plant_component_detector']['detector_version']
        cfg['leafmachine']['plant_component_detector']['detector_iteration']
        cfg['leafmachine']['plant_component_detector']['detector_weights']
        cfg['leafmachine']['plant_component_detector']['minimum_confidence_threshold']
        cfg['leafmachine']['plant_component_detector']['do_save_prediction_overlay_images']
        cfg['leafmachine']['plant_component_detector']['ignore_objects_for_overlay']
        detector_type: str = 'Plant_Detector'
        detector_version: str = 'PLANT_GroupAB_200'
        detector_iteration: str = 'PLANT_GroupAB_200'
        detector_weights: str = 'best.pt'
        minimum_confidence_threshold: float = 0.5
        do_save_prediction_overlay_images: bool = False
        ignore_objects_for_overlay: list[str] = field(default_factory=list) 
        
        # Configure Archival Component Detector
        cfg['leafmachine']['archival_component_detector']['detector_type']
        cfg['leafmachine']['archival_component_detector']['detector_version']
        cfg['leafmachine']['archival_component_detector']['detector_iteration']
        cfg['leafmachine']['archival_component_detector']['detector_weights']
        cfg['leafmachine']['archival_component_detector']['minimum_confidence_threshold']
        cfg['leafmachine']['archival_component_detector']['minimum_confidence_threshold']
        cfg['leafmachine']['archival_component_detector']['ignore_objects_for_overlay']
        detector_type: str = 'Archival_Detector' 
        detector_version: str = 'PREP_final'
        detector_iteration: str = 'PREP_final'
        detector_weights: str = 'best.pt'
        minimum_confidence_threshold:float =  0.5
        do_save_prediction_overlay_images: bool = False
        ignore_objects_for_overlay:  list[str] = field(default_factory=list) 

        # Configure Archival Component Detector
        segment_whole_leaves: bool = True
        segment_partial_leaves: bool = True

        keep_only_best_one_leaf_one_petiole: bool = True
        save_segmentation_overlay_images_to_pdf: bool = True
        save_individual_overlay_images: bool = True 
        overlay_line_width: int = 1

        save_masks_color: bool = True
        save_masks_index: bool = True

        calculate_elliptic_fourier_descriptors: bool = True 
        elliptic_fourier_descriptor_order: int = 40 
        
        segmentation_model: str = 'Group1_Dataset_20000_Iter_1176PTS_512Batch_smooth_l1_LR0005_BGR'
        minimum_confidence_threshold: float = 0.5 
        generate_overlay: bool = True
        overlay_dpi: int = 100 
        overlay_background_color: str = 'black' 