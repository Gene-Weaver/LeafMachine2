import os
import numpy as np
from labelbox import Client


class OPTS_EXPORT_POINTS():
    client = []
    cfg = []

    # config file
    PROJECT_NAME = []
    REDO = []
    CUMMULATIVE = []
    USE_TEMPLATE_YAML = []
    RESTRICT_ANNOTYPE = []
    IGNORE = []
    INCLUDE = []
    DO_PARTITION_DATA = []
    RATIO = []
    INCLUDE_ANNO = []
    ONLY_REVIEWED = []

    DIR_ROOT = []
    DIR_DATASETS = []
    DIR_JSON = []

    def __init__(self, cfg, client):
        self.client = client
        self.cfg = cfg

        # config file
        self.PROJECT_NAME = cfg['project_name']
        self.REDO = cfg['do_redo_export'] # Redo all folders = 1. Only add new = 0 
        self.CUMMULATIVE = cfg['accumulate_all_projects_in_run'] # 1 = put all labels and images into a single large dir to train everything at once   OR   0 = each project get its own training dir
        self.USE_TEMPLATE_YAML = cfg['use_template_yaml_file']
        self.RESTRICT_ANNOTYPE = cfg['annotation_type_to_export'] # all: ["PREP","PLANT","Morton"]
        self.IGNORE = cfg['ignore_projects']
        self.INCLUDE = cfg['include_projects'] #['PREP_FieldPrism_Training_Outside','PREP_REU_Field_QR-Code-Images','PREP_FieldPrism_Training_Sheets','PREP_FieldPrism_Training_FS-Poor','PREP_FieldPrism_Training_Sheets_QR']# 'PREP_FieldPrism_Outside'    ['PREP_REU_Specimens-Full','PREP_REU_Specimens-Leaves']
        self.DO_PARTITION_DATA = cfg['do_partition_data']
        self.RATIO = cfg['partition_ratio']
        self.INCLUDE_ANNO = cfg['include_annotations']
        self.ONLY_REVIEWED = cfg['fetch_reviewed_annotations_only']

        self.DIR_ROOT = cfg['dir_export_base']
        self.DIR_DATASETS = os.path.join(self.DIR_ROOT,self.PROJECT_NAME)
        self.DIR_JSON = os.path.join(self.DIR_ROOT,self.PROJECT_NAME,'labelbox_json')

class OPTS_EXPORT_SEG():
    client = []
    cfg = []

    # config file
    PROJECT_NAME = []
    REDO = []
    CUMMULATIVE = []
    USE_TEMPLATE_YAML = []
    RESTRICT_ANNOTYPE = []
    IGNORE = []
    INCLUDE = []
    DO_PARTITION_DATA = []
    RATIO = []
    INCLUDE_ANNO = []
    ONLY_REVIEWED = []

    DIR_ROOT = []
    DIR_DATASETS = []
    DIR_JSON = []

    def __init__(self, cfg, client):
        self.client = client
        self.cfg = cfg

        # config file
        self.PROJECT_NAME = cfg['project_name']
        self.REDO = cfg['do_redo_export'] # Redo all folders = 1. Only add new = 0 
        self.CUMMULATIVE = cfg['accumulate_all_projects_in_run'] # 1 = put all labels and images into a single large dir to train everything at once   OR   0 = each project get its own training dir
        self.USE_TEMPLATE_YAML = cfg['use_template_yaml_file']
        self.RESTRICT_ANNOTYPE = cfg['annotation_type_to_export'] # all: ["PREP","PLANT","Morton"]
        self.IGNORE = cfg['ignore_projects']
        self.INCLUDE = cfg['include_projects'] #['PREP_FieldPrism_Training_Outside','PREP_REU_Field_QR-Code-Images','PREP_FieldPrism_Training_Sheets','PREP_FieldPrism_Training_FS-Poor','PREP_FieldPrism_Training_Sheets_QR']# 'PREP_FieldPrism_Outside'    ['PREP_REU_Specimens-Full','PREP_REU_Specimens-Leaves']
        self.DO_PARTITION_DATA = cfg['do_partition_data']
        self.RATIO = cfg['partition_ratio']
        self.INCLUDE_ANNO = cfg['include_annotations']
        self.ONLY_REVIEWED = cfg['fetch_reviewed_annotations_only']

        self.DIR_ROOT = cfg['dir_export_base']
        self.DIR_DATASETS = os.path.join(self.DIR_ROOT,self.PROJECT_NAME)
        self.DIR_JSON = os.path.join(self.DIR_ROOT,self.PROJECT_NAME,'labelbox_json')
        

class OPTS_EXPORT():
    REDO = []
    ZIP = []
    CUMMULATIVE = []
    PROJECT_NAME = []
    USE_TEMPLATE_YAML = []
    RESTRICT_ANNOTYPE = []
    IGNORE = []
    INCLUDE = []
    DO_PARTITION_DATA = []

    DIR_ROOT = []
    DIR_DATASETS = []
    DIR_COLORS = []
    DIR_CROPPED = []

    RATIO = []
    client = []
    cfg = []
    SAVE_COLOR_CSV = []
    INCLUDE_ANNO = []
    max_annotations_saved_per_image_LEAF = []
    max_annotations_saved_per_image_ALL_OTHER_TYPES = []
    do_save_cropped_bboxes_as_jpgs = []

    do_sort_compound_and_simple_leaves =[]

    def __init__(self, cfg, client):
        self.client = client
        self.cfg = cfg

        self.REDO = cfg['do_redo_export'] # Redo all folders = 1. Only add new = 0 
        self.ZIP = cfg['create_zip_of_project']
        self.CUMMULATIVE = cfg['accumulate_all_projects_in_run'] # 1 = put all labels and images into a single large dir to train everything at once   OR   0 = each project get its own training dir
        self.PROJECT_NAME = cfg['project_name']
        self.USE_TEMPLATE_YAML = cfg['use_template_yaml_file']
        self.RESTRICT_ANNOTYPE = cfg['annotation_type_to_export'] # all: ["PREP","PLANT","Morton"]
        self.IGNORE = cfg['ignore_projects']

        self.INCLUDE = cfg['include_projects'] #['PREP_FieldPrism_Training_Outside','PREP_REU_Field_QR-Code-Images','PREP_FieldPrism_Training_Sheets','PREP_FieldPrism_Training_FS-Poor','PREP_FieldPrism_Training_Sheets_QR']# 'PREP_FieldPrism_Outside'    ['PREP_REU_Specimens-Full','PREP_REU_Specimens-Leaves']
        self.DO_PARTITION_DATA = cfg['do_partition_data']
        self.RATIO = cfg['partition_ratio']
        self.ONLY_REVIEWED = cfg['fetch_reviewed_annotations_only']

        self.do_save_cropped_bboxes_as_jpgs = cfg['do_save_cropped_bboxes_as_jpgs']
        self.SAVE_COLOR_CSV = cfg['do_save_colors_to_csv']
        self.INCLUDE_ANNO = cfg['include_annotations']
        # opt.max_annotations_saved_per_image_LEAF = cfg['max_annotations_saved_per_image_LEAF']
        # opt.max_annotations_saved_per_image_ALL_OTHER_TYPES = cfg['max_annotations_saved_per_image_ALL_OTHER_TYPES']

        self.do_sort_compound_and_simple_leaves = cfg['do_sort_compound_and_simple_leaves']

        self.DIR_ROOT = cfg['dir_export_base'] # os.path.abspath(os.path.join('Image_Datasets','GroundTruth_JSON-LB_byAnnoType'))
        self.DIR_DATASETS = os.path.join(self.DIR_ROOT,self.PROJECT_NAME)
        self.DIR_COLORS = os.path.join(self.DIR_ROOT,self.PROJECT_NAME,'colors')

class OPTS():
    DS_PREFIX = []
    DS_DATA = []
    DS_NAME = []
    PROJECT_NAME = []
    DIR_DATASET = []
    DIR_LABELS = []
    ANNOTYPE = []
    def __init__(self):
        self.DS_PREFIX = []
        self.DS_DATA = []
        self.DS_NAME = []
        self.PROJECT_NAME = []
        self.DIR_DATASET = []
        self.DIR_LABELS = []
        self.ANNOTYPE = []

def redo_JSON(fname):
    try:
        doProject = False
        jsonFile = open(fname)
    except:
        doProject = True
    return doProject

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

def validateDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        dirState = 1
    else:
        dirState = 0
    return dirState

def assign_index(cls,annoType):
    cls = cls.lower()
    if annoType == 'PLANT':
        if cls == 'leaf_whole':
            annoInd = 0
        elif cls == 'leaf_partial':
            annoInd = 1
        elif cls == 'leaflet':
            annoInd = 2
        elif cls == 'seed_fruit_one':
            annoInd = 3
        elif cls == 'seed_fruit_many':
            annoInd = 4
        elif cls == 'flower_one':
            annoInd = 5
        elif cls == 'flower_many':
            annoInd = 6
        elif cls == 'bud':
            annoInd = 7
        elif cls == 'specimen':
            annoInd = 8
        elif cls == 'roots':
            annoInd = 9
        elif cls == 'wood':
            annoInd = 10
    elif annoType == 'PREP':
        if cls == 'ruler':
            annoInd = 0
        elif cls == 'barcode':
            annoInd = 1
        elif cls == 'colorcard':
            annoInd = 2
        elif cls == 'label':
            annoInd = 3
        elif cls == 'map':
            annoInd = 4
        elif cls == 'envelope':
            annoInd = 5
        elif cls == 'photo':
            annoInd = 6
        elif cls == 'attached_item':
            annoInd = 7
        elif cls == 'weights':
            annoInd = 8
    return annoInd

def set_index_for_annotation(cls,annoType):
    if annoType == 'PLANT':
        if cls == 0:
            annoInd = 'Leaf_WHOLE'
        elif cls == 1:
            annoInd = 'Leaf_PARTIAL'
        elif cls == 2:
            annoInd = 'Leaflet'
        elif cls == 3:
            annoInd = 'Seed_Fruit_ONE'
        elif cls == 4:
            annoInd = 'Seed_Fruit_MANY'
        elif cls == 5:
            annoInd = 'Flower_ONE'
        elif cls == 6:
            annoInd = 'Flower_MANY'
        elif cls == 7:
            annoInd = 'Bud'
        elif cls == 8:
            annoInd = 'Specimen'
        elif cls == 9:
            annoInd = 'Roots'
        elif cls == 10:
            annoInd = 'Wood'
    elif annoType == 'PREP':
        if cls == 0:
            annoInd = 'Ruler'
        elif cls == 1:
            annoInd = 'Barcode'
        elif cls == 2:
            annoInd = 'Colorcard'
        elif cls == 3:
            annoInd = 'Label'
        elif cls == 4:
            annoInd = 'Map'
        elif cls == 5:
            annoInd = 'Envelope'
        elif cls == 6:
            annoInd = 'Photo'
        elif cls == 7:
            annoInd = 'Attached Item'
        elif cls == 8:
            annoInd = 'Weights'
    return annoInd

def get_MAL_ontology(project_id, client):
    response = client.execute(
                """
                query getOntology (
                    $project_id : ID!){ 
                    project (where: { id: $project_id }) { 
                        ontology { 
                            normalized 
                        } 
                    }
                }
                """,
                {"project_id": project_id})
            
    ontology = response['project']['ontology']['normalized']['tools']

    ##Return list of tools and embed category id to be used to map classname during training and inference
    mapped_ontology = []
    thing_classes = []
    
    i=0
    for item in ontology:
#         if item['tool']=='superpixel' or item['tool']=='rectangle':
        item.update({'category': i})
        mapped_ontology.append(item)
        thing_classes.append(item['name'])
        i=i+1         

    return mapped_ontology, thing_classes

def get_project_names(client):
    projectList = []
    projects = client.get_projects()
    for project in projects:
        projectList.append(project.name)
        #print(project.name)
    return projectList

def get_dataset_names(rootDir):
    nameList = list()
    sep = '_'
    for img in os.listdir(rootDir):
        id, nameAll = parse_name(img,sep)
        nameList.append(nameAll)
    uniqNameList = unique(nameList)


    imgList = list()
    for name in uniqNameList:
        imgList = list()
        for img in os.listdir(rootDir):
            id, nameAll = parse_name(img,sep)

            if nameAll == name:
                newItem = add_item(rootDir,img,id)
                imgList.append(newItem)
    return imgList

def create_annotation_dict(keyList):
    d = {}
    for i in keyList:
        d[i] = 0
    return d