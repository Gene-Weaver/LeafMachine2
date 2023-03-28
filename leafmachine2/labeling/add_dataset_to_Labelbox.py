# Create new LabelBox dataset

import os, sys, re, inspect
import labelbox
from labelbox.schema.queue_mode import QueueMode
from labelbox import Client, LabelingFrontend, MediaType
import numpy as np
from pathlib import Path

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from machine.general_utils import get_datetime, get_cfg_from_full_path, bcolors, make_file_names_valid
from utils_Labelbox import set_index_for_annotation, parse_name, add_item

def unique(list1):
    x = np.array(list1)
    return np.unique(x)

def parse_name(img,sep):
    id = img.split(".")[0]
    nameAll = id.split("_")[2:]
    nameAll = sep.join(nameAll)
    return id, nameAll

def add_item(rootDir,img,id):
    newItem = {"row_data": os.path.join(rootDir,img),"external_id": id}
    return newItem

def upload_dataset_to_LabelBox(client,dataset_name,dir_containing_original_images):
    img_list = list()
    sep = "_"
    for img in os.listdir(dir_containing_original_images):
        id, _name_all = parse_name(img,sep)
        new_item = add_item(dir_containing_original_images,img,id)
        img_list.append(new_item)

    # Create a new dataset
    dataset = client.create_dataset(name=dataset_name)
    # Bulk add data rows to the dataset
    task = dataset.create_data_rows(img_list)
    task.wait_till_done()
    print(task.status)

def upload_dataset():
    # Read configs
    dir_private = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path_cfg_private = os.path.join(dir_private,'PRIVATE_DATA.yaml')
    cfg_private = get_cfg_from_full_path(path_cfg_private)

    LB_API_KEY = cfg_private['labelbox']['LB_API_KEY']
    client = Client(api_key=LB_API_KEY)

    dir_labeling = os.path.dirname(__file__)
    path_cfg = os.path.join(dir_labeling,'config_add_dataset_to_Labelbox.yaml')
    cfg = get_cfg_from_full_path(path_cfg)

    dataset_name = cfg['dataset_name']
    dir_containing_original_images = cfg['dir_containing_original_images']
    
    upload_dataset_to_LabelBox(client, dataset_name,dir_containing_original_images)

if __name__ == '__main__':
    upload_dataset()