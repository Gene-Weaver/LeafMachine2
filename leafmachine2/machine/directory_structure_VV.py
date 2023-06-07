import os, pathlib, sys, inspect
from dataclasses import dataclass, field
currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)
from leafmachine2.machine.general_utils import validate_dir, get_datetime

@dataclass
class Dir_Structure():
    # Home 
    run_name: str = ''
    dir_home: str = ''
    dir_project: str = ''

    # Processing dirs
    path_archival_components: str = ''
    path_config_file: str = ''

    ruler_info: str = ''
    ruler_overlay: str = ''
    ruler_processed: str = ''
    ruler_data: str = ''
    ruler_class_overlay: str = ''
    ruler_validation_summary: str = ''
    ruler_validation: str = ''

    save_per_image: str = ''
    save_per_annotation_class: str = ''
    binarize_labels: str = ''

    # logging
    path_log: str = ''
    
    def __init__(self, cfg) -> None:
        # Home 
        self.run_name = cfg['leafmachine']['project']['run_name']
        self.dir_home = cfg['leafmachine']['project']['dir_output']
        self.dir_project = os.path.join(self.dir_home,self.run_name)
        validate_dir(self.dir_home)
        self.__add_time_to_existing_project_dir()
        validate_dir(self.dir_project)

        # Processing dirs
        self.path_archival_components = os.path.join(self.dir_project,'Archival_Components')
        self.path_config_file = os.path.join(self.dir_project,'Config_File')
        validate_dir(self.path_config_file)

        # Logging
        self.path_log = os.path.join(self.dir_project,'Logs')
        validate_dir(self.path_log)

        self.custom_overlay_pdfs = os.path.join(self.dir_project,'Summary','Custom_Overlay_PDFs')
        self.custom_overlay_images = os.path.join(self.dir_project,'Summary','Custom_Overlay_Images')

        ###
        self.custom_overlay_pdfs = os.path.join(self.dir_project,'Summary','Custom_Overlay_PDFs')
        if cfg['leafmachine']['overlay']['save_overlay_to_pdf']:
            validate_dir(self.custom_overlay_pdfs)

        self.custom_overlay_images = os.path.join(self.dir_project,'Summary','Custom_Overlay_Images')
        if cfg['leafmachine']['overlay']['save_overlay_to_jpgs']:
            validate_dir(self.custom_overlay_images)

        ### Rulers
        self.ruler_info = os.path.join(self.dir_project,'Archival_Components','Ruler_Info')  
        self.ruler_validation_summary =  os.path.join(self.dir_project,'Archival_Components','Ruler_Info', 'Ruler_Validation_Summary')
        self.ruler_validation = os.path.join(self.dir_project,'Archival_Components','Ruler_Info', 'Ruler_Validation')
        self.ruler_processed = os.path.join(self.dir_project,'Archival_Components','Ruler_Info', 'Ruler_Processed')
        validate_dir(self.ruler_info)
        

        validate_dir(self.path_archival_components)
        validate_dir(os.path.join(self.path_archival_components, 'JSON'))

        ### Data
        self.transcription = os.path.join(self.dir_project,'Transcription') 
        validate_dir(self.transcription)
        self.transcription_ind = os.path.join(self.dir_project,'Transcription','Individual') 
        validate_dir(self.transcription_ind)
        self.transcription_ind_helper = os.path.join(self.dir_project,'Transcription','Individual_Helper_Content') 
        validate_dir(self.transcription_ind_helper)


        self.save_per_image = os.path.join(self.dir_project,'Cropped_Images', 'By_Image') 
        self.save_per_annotation_class = os.path.join(self.dir_project,'Cropped_Images', 'By_Class') 
        self.save_per_annotation_class = os.path.join(self.dir_project,'Cropped_Images', 'By_Class') 
        if cfg['leafmachine']['cropped_components']['save_per_image']:
            validate_dir(self.save_per_image)
        if cfg['leafmachine']['cropped_components']['save_per_annotation_class']:
            validate_dir(self.save_per_annotation_class)
        if cfg['leafmachine']['cropped_components']['binarize_labels']:
            validate_dir(self.save_per_annotation_class)
            # self.binarize_labels = os.path.join(self.dir_project,'Cropped_Images', 'By_Class','label_binary') 
            # validate_dir(self.binarize_labels)

    def __add_time_to_existing_project_dir(self) -> None:
        path = pathlib.Path(self.dir_project)
        if path.exists():
            now = get_datetime()
            path = path.with_name(path.name + "_" + now)
            self.run_name = path.name
            path.mkdir()
            self.dir_project = path
        else:
            path.mkdir()
            self.dir_project = path