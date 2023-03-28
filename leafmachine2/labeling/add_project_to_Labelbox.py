# Create new LabelBox dataset
# This script DOES work but MAY error out due to labelbox timeout stuff
# If it fails, try again a few times, deleting the old project and dataset between each try

import os, sys, re, inspect, random
import labelbox
from labelbox.schema.queue_mode import QueueMode
from labelbox import Client, LabelingFrontend, MediaType
import numpy as np
from pathlib import Path

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from machine.general_utils import get_datetime, get_cfg_from_full_path, bcolors
from utils_Labelbox import set_index_for_annotation, parse_name, add_item
from add_dataset_to_Labelbox import upload_dataset_to_LabelBox

def create_datarows(client,dataset_name,dir_containing_original_images,do_sample,sample_size):
    # Create a new dataset
    dataset = client.create_dataset(name=dataset_name)

    sep = "_"
    uploads = []
    for img in os.listdir(dir_containing_original_images):
        id, _name_all = parse_name(img,sep)
        new_item = add_item(dir_containing_original_images,img,id)
        uploads.append(new_item)
    dataset.create_data_rows(uploads)
    data_rows = [dr.uid for dr in list(dataset.export_data_rows())]

    # if do_sample:
    if sample_size == 1:
        return data_rows
    elif sample_size < 1:
        n_keep = int(np.multiply(sample_size,len(os.listdir(dir_containing_original_images))))
        data_rows = random.sample(data_rows, n_keep)
        return data_rows
    elif sample_size > 1:
        if sample_size >= len(os.listdir(dir_containing_original_images)):
            data_rows = random.sample(data_rows, len(os.listdir(dir_containing_original_images)))
        else:
            data_rows = random.sample(data_rows, sample_size)
        return data_rows

def create_Labelbox_project():
    # Read configs
    dir_private = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path_cfg_private = os.path.join(dir_private,'PRIVATE_DATA.yaml')
    cfg_private = get_cfg_from_full_path(path_cfg_private)

    dir_labeling = os.path.dirname(__file__)
    path_cfg = os.path.join(dir_labeling,'config_add_project_to_Labelbox.yaml')
    cfg = get_cfg_from_full_path(path_cfg)

    path_cfg_onto = os.path.join(dir_labeling,'config_Labelbox_ontology_list.yaml')
    cfg_onto = get_cfg_from_full_path(path_cfg_onto)

    LB_API_KEY = cfg_private['labelbox']['LB_API_KEY']
    client = Client(api_key=LB_API_KEY)

    chosen_ontology = cfg['ontology']
    ontology = client.get_ontology(cfg_onto['ontology_name'][chosen_ontology])

    dataset_name = cfg['dataset_name']
    project_name = cfg['project_name'] 
    dir_containing_original_images = cfg['dir_containing_original_images']
    do_sample = cfg['do_randomly_sample_images']
    sample_size = cfg['sample_size']

    # Project Setup
    # ontology_builder = OntologyBuilder(ontology)
    print(f"{bcolors.BOLD}Creating project: {project_name}{bcolors.ENDC}")
    print(f"{bcolors.BOLD}      Uploading images{bcolors.ENDC}")
    new_project = client.create_project(name=project_name, media_type=MediaType.Image, queue_mode=QueueMode.Batch)
    new_project.enable_model_assisted_labeling()

    # Dataset Setup
    dataset = create_datarows(client,dataset_name,dir_containing_original_images,do_sample,sample_size)
    print(f"{bcolors.BOLD}      Images uploaded{bcolors.ENDC}")
    
    # new_project.datasets.connect(dataset)
    new_project.create_batch(''.join(["Initial_batch_",get_datetime()]), # name of the batch
                            dataset, # list of Data Rows
                            2 # priority between 1-5
                            )
    print(f"{bcolors.BOLD}      Attached dataset: {dataset_name}  -->  {project_name}{bcolors.ENDC}")
    
    editor = next(client.get_labeling_frontends(where=LabelingFrontend.name == "Editor"))
    new_project.setup(editor, ontology.normalized)
    print(f"{bcolors.OKGREEN}Project complete!{bcolors.ENDC}")

if __name__ == '__main__':
    create_Labelbox_project()