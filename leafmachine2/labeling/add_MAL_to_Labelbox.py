from labelbox import Client, MALPredictionImport, DataRow
from labelbox.data.annotation_types import (
    Label, ImageData, ObjectAnnotation, MaskData,
    Rectangle, Point, Line, Mask, Polygon,
    Radio, Checklist, Text,
    ClassificationAnnotation, ClassificationAnswer
)
from labelbox.data.serialization import NDJsonConverter
from labelbox.schema.media_type import MediaType
import uuid, requests, json, datetime, os, inspect, sys, re, time
import numpy as np
from labelbox.schema.queue_mode import QueueMode
from labelbox import Client, LabelingFrontend, MALPredictionImport, MediaType
from uuid import uuid4
import datetime
import pandas as pd
from PIL import Image # pillow
from typing import Dict, Any, Tuple, List

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from machine.general_utils import get_datetime, get_cfg_from_full_path, bcolors
from utils_Labelbox import get_project_names, get_MAL_ontology, set_index_for_annotation, parse_name, add_item, OPTS, assign_index

def delete_unlabeld_data_rows(ontology, opt, client, existing_project):
    # Setup
    rootDir = opt.DIR_DATASET
    imgList = list()
    sep = "_"

    # Dataset Setup
    dataset = client.create_dataset(name=opt.DS_NAME)
    editor = next(client.get_labeling_frontends(where=LabelingFrontend.name == "Editor"))
    # Project Setup
    if not existing_project:
        # ontology_builder = OntologyBuilder(ontology)
        mal_project = client.create_project(name=opt.PROJECT_NAME,media_type=MediaType.Image,queue_mode=QueueMode.Batch)
        mal_project.enable_model_assisted_labeling()
        mal_project.setup(editor, ontology.normalized)
    else:
        projects = client.get_projects()
        mal_project = 0
        for project in projects:
            if project.name == opt.PROJECT_NAME:
                mal_project = project
                break
    
    uuid_list = []  
    ndjson = []

    '''Get existing uploads -> existing_labels[]'''
    labels = mal_project.export_labels()
    
    unlabeled = mal_project.export_queued_data_rows()
    unlabeled_datarows = []
    for i, row in enumerate(unlabeled):
        print(i)
        try:
            drow = client.get_data_row(row['id'])
            unlabeled_datarows.append(drow)
        except:
            print('skip')
            continue
    DataRow.bulk_delete(unlabeled_datarows)

def upload_MAL(ontology, opt, client, existing_project):
    # Setup
    rootDir = opt.DIR_DATASET
    imgList = list()
    sep = "_"

    # Dataset Setup
    dataset = client.create_dataset(name=opt.DS_NAME)
    editor = next(client.get_labeling_frontends(where=LabelingFrontend.name == "Editor"))
    # Project Setup
    if not existing_project:
        # ontology_builder = OntologyBuilder(ontology)
        mal_project = client.create_project(name=opt.PROJECT_NAME,media_type=MediaType.Image,queue_mode=QueueMode.Batch)
        mal_project.enable_model_assisted_labeling()
        mal_project.setup(editor, ontology.normalized)
    else:
        projects = client.get_projects()
        mal_project = 0
        for project in projects:
            if project.name == opt.PROJECT_NAME:
                mal_project = project
                break
    
    uuid_list = []  
    ndjson = []

    '''Get existing uploads -> existing_labels[]'''
    labels = mal_project.export_labels()
    
    '''Get datarows that are queued by not submitted'''
    unlabeled = mal_project.export_queued_data_rows()
    unlabeled_datarows = []
    unlabeled_names = []
    for i, row in enumerate(unlabeled):
        print(i)
        try:
            drow = client.get_data_row(row['id'])
            unlabeled_datarows.append(drow)
            unlabeled_names.append(row['externalId'])
        except:
            print('skip')
            continue

    try:
        jsonFile = requests.get(labels) 
    except:
        try:
            time.sleep(10)
            labels = mal_project.export_labels()
            jsonFile = requests.get(labels)
        except: 
            time.sleep(10)
            labels = mal_project.export_labels()
            jsonFile = requests.get(labels) 
    project_labels = jsonFile.json()

    n_images_already_in_project = len(project_labels)

    existing_labels = []
    if n_images_already_in_project == 0: # Unstarted datasets
        pass
    else:
        for old_label in project_labels:
            existing_labels.append(old_label['External ID'])
        print(f"{bcolors.OKCYAN}{len(existing_labels)} submitted labels already in Labelbox project {opt.PROJECT_NAME}{bcolors.ENDC}")
        print(f"{bcolors.OKCYAN}{len(unlabeled_names)} images queued (but not submitted) already in Labelbox project {opt.PROJECT_NAME}{bcolors.ENDC}")

    ''''''

    n_total_images = len(os.listdir(rootDir))
    for i_img, img in enumerate(os.listdir(rootDir)):
        if (img.split('.')[0] in existing_labels):
            print(f"{bcolors.OKGREEN}     Image {i_img + 1} / {n_total_images} --> {img.split('.')[0]} already in Labelbox{bcolors.ENDC}")
        elif (img.split('.')[0] in unlabeled_names):
            print(f"{bcolors.OKCYAN}     Image {i_img + 1} / {n_total_images} --> {img.split('.')[0]} already in Labelbox{bcolors.ENDC}")
        else:
            img_path = os.path.abspath(os.path.join(opt.DIR_DATASET,img))

            # Get info from the image, add to dataset
            im = Image.open(img_path)
            width, height = im.size
            id, nameAll = parse_name(img,sep)
            newItem = add_item(rootDir,img,id)
            imgList.append(newItem)

            # image_data = ImageData(uid=id,file_path=os.path.abspath(os.path.join(opt.DIR_DATASET,img)),external_id=id) 
            has_labels = False
            txtFile = os.path.abspath(os.path.join(opt.DIR_LABELS,'labels',id + '.txt'))
            try:
                file = open(txtFile, 'r')
                lines = file.readlines()
                has_labels = True
            except:
                has_labels = False

            # Create the datarow in labelbox
            print(f"{bcolors.BOLD}     Image {i_img + 1} / {n_total_images} --> {id} creating datarow{bcolors.ENDC}")
            data_row = dataset.create_data_row(row_data=img_path,external_id=id)
            print(f"{bcolors.BOLD}     Image {i_img + 1} / {n_total_images} --> {id} datarow created{bcolors.ENDC}")
            uuid_list.append(data_row.uid)
            # labelList = []
            bbox_list = []
            count = 0
            if has_labels:
                onto, thing_classes = get_MAL_ontology(mal_project.uid, client)
                # For each line (label) in the txt file...
                for line in lines:
                    # Read yolov5 prediction and convert to labelbox ndjson format
                    count += 1
                    #print("          Line{}: {}".format(count, line.strip()))
                    lineParse = line.split(" ")
                    objClass = int(lineParse[0])
                    objClass = set_index_for_annotation(objClass,opt.ANNOTYPE)

                    x = float(lineParse[1])
                    y = float(lineParse[2])
                    w = float(lineParse[3])
                    h = float(lineParse[4])
                    left = (x * width) - w/2
                    top = (y * height) - h/2
                    h = h * height
                    w = w * width

                    # for item in onto:
                        # if objClass == item['name']:
                    bbox_annotation = ObjectAnnotation(
                        name = objClass,  # must match your ontology feature's name
                        value=Rectangle(
                            start=Point(x=int(left-w/2), y=int(top-h/2)), # Top left
                            end=Point(x=int(w)+int(left-w/2), y=int(h)+int(top-h/2)), # Bottom right
                        )
                    )   
                    bbox_list.append(bbox_annotation)
                print("          Contained {} predictions".format(count))
            else:
                print(f"{bcolors.BOLD}     Image {i_img + 1} / {n_total_images} --> {id}{bcolors.ENDC}")
                print("          Contained ZERO predictions")
                try:
                    objClass = set_index_for_annotation(objClass,opt.ANNOTYPE)
                except:
                    try:
                        objClass = assign_index(objClass,opt.ANNOTYPE)
                    except:
                        objClass = 'Specimen'

                bbox_annotation = ObjectAnnotation(
                            name = objClass,  # must match your ontology feature's name
                            value=Rectangle(
                                start=Point(x=0, y=0), # Top left
                                end=Point(x=0, y=0), # Bottom right
                            )
                        )   
                bbox_list.append(bbox_annotation)

            
            # Create a Label
            label = Label(
                data=ImageData(uid=data_row.uid),
                annotations = bbox_list
            )

            # Create urls to mask data for upload
            def signing_function(obj_bytes: bytes) -> str:
                url = client.upload_data(content=obj_bytes, sign=True)
                return url

            label.add_url_to_masks(signing_function)

            # Convert our label from a Labelbox class object to the underlying NDJSON format required for upload 
            label_ndjson = list(NDJsonConverter.serialize([label]))
            ndjson.append(label_ndjson)

            mal_project.create_batch(''.join(["Initial_batch_",get_datetime()]), # name of the batch  ### create_batch
                    [data_row.uid], # list of Data Rows
                    2 # priority between 1-5
                    )
            # Upload MAL label for this data row in project
            upload_job = MALPredictionImport.create_from_objects(
                client = client, 
                project_id = mal_project.uid, 
                name="mal_job"+str(uuid.uuid4()), 
                predictions=label_ndjson)

            print("          Errors:", upload_job.errors)

def add_MAL_to_Labelbox():
    # Read configs
    dir_private = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path_cfg_private = os.path.join(dir_private,'PRIVATE_DATA.yaml')
    cfg_private = get_cfg_from_full_path(path_cfg_private)

    dir_labeling = os.path.dirname(__file__)
    path_cfg = os.path.join(dir_labeling,'config_add_MAL_to_Labelbox.yaml')
    cfg = get_cfg_from_full_path(path_cfg)

    path_cfg_onto = os.path.join(dir_labeling,'config_Labelbox_ontology_list.yaml')
    cfg_onto = get_cfg_from_full_path(path_cfg_onto)

    LB_API_KEY = cfg_private['labelbox']['LB_API_KEY']
    client = Client(api_key=LB_API_KEY)

    chosen_ontology = cfg['ontology']
    ontology = client.get_ontology(cfg_onto['ontology_name'][chosen_ontology])

    dir_containing_labels = cfg['dir_containing_labels']
    dir_containing_original_images = cfg['dir_containing_original_images']

    if cfg['include_subdirs']:
        dir_list = os.listdir(dir_containing_labels)
        projectList = get_project_names(client)
        f = 0
        for folder in dir_list:
            if folder in cfg['dirs_to_include']:
                opt = OPTS()
                opt.DS_PREFIX = ''.join([chosen_ontology,'_'])
                opt.DS_DATA = folder
                opt.DS_NAME = ''.join([opt.DS_PREFIX,folder])
                opt.PROJECT_NAME = ''.join([chosen_ontology,'_', folder])
                opt.ANNOTYPE = chosen_ontology

                opt.DIR_DATASET = os.path.join(dir_containing_original_images,folder)
                opt.DIR_LABELS = os.path.join(dir_containing_labels,folder)

                existing_project = False
                if (opt.PROJECT_NAME in projectList):
                    existing_project = True

                if not cfg['add_to_existing_project']:
                    pass
                else:
                    print(f"{bcolors.HEADER}Adding MAL to Labelbox: Project --> {opt.PROJECT_NAME}{bcolors.ENDC}")
                    
                    upload_MAL(ontology, opt, client, existing_project)

                    print(f"{bcolors.OKCYAN}Finished Uploading MAL --> {opt.PROJECT_NAME}{bcolors.ENDC}")
                    f += 1
        print(f"{bcolors.OKGREEN}Finished Uploading MAL for {f} Projects :){bcolors.ENDC}")
    else:
        projectList = get_project_names(client)
        f = 0
        opt = OPTS()
        opt.DS_PREFIX = ''.join([chosen_ontology,'_'])
        opt.DS_DATA = dir_containing_labels
        opt.DS_NAME = ''.join([opt.DS_PREFIX,os.path.basename(dir_containing_original_images)])
        opt.PROJECT_NAME = ''.join([chosen_ontology,'_', os.path.basename(dir_containing_original_images)])
        opt.ANNOTYPE = chosen_ontology

        opt.DIR_DATASET = dir_containing_original_images
        opt.DIR_LABELS = dir_containing_labels

        existing_project = False
        if (opt.PROJECT_NAME in projectList):
            existing_project = True

        if not cfg['add_to_existing_project']:
            pass
        else:
            print(f"{bcolors.HEADER}Adding MAL to Labelbox: Project --> {opt.PROJECT_NAME}{bcolors.ENDC}")

            upload_MAL(ontology, opt, client, existing_project)

            print(f"{bcolors.OKCYAN}Finished Uploading MAL --> {opt.PROJECT_NAME}{bcolors.ENDC}")
            f += 1
        print(f"{bcolors.OKGREEN}Finished Uploading MAL for {f} Projects :){bcolors.ENDC}")
if __name__ == '__main__':
    add_MAL_to_Labelbox()