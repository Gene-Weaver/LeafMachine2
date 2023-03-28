# save the labelbox groundtruth overlay images
from cmath import e
from pprint import pprint
import labelbox
from labelbox import Client
from labelbox import Client, OntologyBuilder
from labelbox.data.annotation_types import Geometry # pip install labelbox[data]
from PIL import Image # pillow
import numpy as np
import os, sys, inspect, requests, json, time, math
from pathlib import Path
from dataclasses import dataclass, field
import pandas as pd

import yaml #pyyaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# from utils import make_dirs
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from machine.general_utils import get_cfg_from_full_path, bcolors, validate_dir
from utils_Labelbox import assign_index, redo_JSON, OPTS_EXPORT_POINTS


@dataclass
class Points:
    IMG_FILENAME: str 
    IMG_NAME: str 
    M_LOBE_COUNT: int = 0
    M_DEEPEST_SINUS_ANGLE: int = 0

    LOBE_TIP: list = field(init=False,default_factory=list)
    ANGLE_TYPES: list = field(init=False,default_factory=list)
    LAMINA_TIP: list[tuple] = field(init=False)
    LAMINA_BASE: list[tuple] = field(init=False)
    PETIOLE_TIP: list[tuple] = field(init=False)
    LAMINA_WIDTH: list[tuple] = field(init=False)
    MIDVEIN_TRACE: list[tuple] = field(init=False)
    PETIOLE_TRACE: list[tuple] = field(init=False)
    APEX_ANGLE: list[tuple] = field(init=False)
    BASE_ANGLE: list[tuple] = field(init=False)
    DEEPEST_SINUS_ANGLE: list = field(init=False,default_factory=list)

    CM_1: list[tuple] = field(init=False)
    
    M_LAMINA_LENGTH: float = field(init=False)
    M_MIDVEIN_LENGTH: float = field(init=False)
    M_PETIOLE_TRACE_LENGTH: float = field(init=False)
    M_PETIOLE_LENGTH: float = field(init=False)
    M_LAMINA_WIDTH: float = field(init=False)
    M_APEX_ANGLE: float = field(init=False)
    M_BASE_ANGLE: float = field(init=False)

    M_CM_1: float = field(init=False)

    LOBE_TIP_N: int = field(init=False)
    LAMINA_TIP_N: int = field(init=False)
    LAMINA_BASE_N: int = field(init=False)
    PETIOLE_TIP_N: int = field(init=False)
    LAMINA_WIDTH_N: int = field(init=False)
    MIDVEIN_TRACE_N: int = field(init=False)
    PETIOLE_TRACE_N: int = field(init=False)
    APEX_ANGLE_N: int = field(init=False)
    BASE_ANGLE_N: int = field(init=False)
    DEEPEST_SINUS_ANGLE_N: int = field(init=False)

    # def __post_init__(self) -> None:
    #     self.LOBE_TIP = []
    def total_distance(self,pts):
        total = 0
        for i in range(len(pts) - 1):
            total += math.dist(pts[i],pts[i+1])
        return total

    def find_angle(self,pts,reflex,location):
        isReflex = False
        if location != 'sinus':
            for ans in reflex:
                if location == 'apex':
                    if ans == 'apex_more_than_180':
                        isReflex = True
                elif location == 'base':
                    if ans == 'base_more_than_180':
                        isReflex = True
                elif location == 'sinus':
                    isReflex = False
            a = np.array(pts[0])
            b = np.array(pts[1])
            c = np.array(pts[2])

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            angle = np.degrees(angle)

            if isReflex:
                angle = 360 - angle
            return angle 
        else:
            angles = []
            for pt in pts:
        
                a = np.array(pt[0])
                b = np.array(pt[1])
                c = np.array(pt[2])

                ba = a - b
                bc = c - b

                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                angle = np.arccos(cosine_angle)
                angle = np.degrees(angle)

                if isReflex:
                    angle = 360 - angle
                angles.append(angle)
            return angles 

    def calculate_cm(self):
        # 2 points
        self.M_CM_1 = self.total_distance(self.CM_1)
        print(f'\nM_CM_1: {self.M_CM_1}\n')

    def calculate_measurements(self):
        # 2 points
        try:
            self.M_LAMINA_LENGTH = math.dist(self.LAMINA_TIP[0],self.LAMINA_BASE[0])
            # print(f'\nM_LAMINA_LENGTH: {self.M_LAMINA_LENGTH}')
        except:
            self.M_LAMINA_LENGTH = 0
        try:
            self.M_LAMINA_WIDTH = math.dist(self.LAMINA_WIDTH[0],self.LAMINA_WIDTH[1])
            # print(f'M_LAMINA_WIDTH: {self.M_LAMINA_WIDTH}')
        except:
            self.M_LAMINA_WIDTH = 0
        try:
            self.M_PETIOLE_LENGTH = math.dist(self.LAMINA_BASE[0],self.PETIOLE_TIP[0])
            # print(f'M_PETIOLE_LENGTH: {self.M_PETIOLE_LENGTH}')
        except:
            self.M_PETIOLE_LENGTH = 0

        # Many points
        try:
            self.M_MIDVEIN_LENGTH = self.total_distance(self.MIDVEIN_TRACE)
            # print(f'M_MIDVEIN_LENGTH: {self.M_MIDVEIN_LENGTH}')
        except:
            self.M_MIDVEIN_LENGTH = 0
        try:
            self.M_PETIOLE_TRACE_LENGTH = self.total_distance(self.PETIOLE_TRACE)
            # print(f'M_PETIOLE_TRACE_LENGTH: {self.M_PETIOLE_TRACE_LENGTH}')
        except:
            self.M_PETIOLE_TRACE_LENGTH = 0

        # Angles
        try:
            self.M_APEX_ANGLE = self.find_angle(self.APEX_ANGLE,self.ANGLE_TYPES,'apex')
            # print(f'M_APEX_ANGLE: {self.M_APEX_ANGLE}')
        except:
            self.M_APEX_ANGLE = 0
        try:
            self.M_BASE_ANGLE = self.find_angle(self.BASE_ANGLE,self.ANGLE_TYPES,'base')
            # print(f'M_BASE_ANGLE: {self.M_BASE_ANGLE}\n')
        except:
            self.M_BASE_ANGLE = 0

        try:
            self.M_DEEPEST_SINUS_ANGLE = self.find_angle(self.DEEPEST_SINUS_ANGLE,self.ANGLE_TYPES,'sinus')
            # print(f'M_BASE_ANGLE: {self.M_BASE_ANGLE}\n')
        except:
            self.M_DEEPEST_SINUS_ANGLE = 0

    def export_ruler(self):
        headers = ['img_name','img_filename','1_cm',]
        data = {'img_name':[self.IMG_NAME],'img_filename':[self.IMG_FILENAME],'1_cm':[self.M_CM_1],}
        df = pd.DataFrame(data,headers)
        df = df.iloc[[0]]
        return df

    def export(self,add_pts_counts):
        try:
            # print(self.LAMINA_TIP)
            self.LAMINA_TIP_N = len(self.LAMINA_TIP)
        except:
            self.LAMINA_TIP = 0
            self.LAMINA_TIP_N = 0
        try:
            # print(self.LAMINA_BASE)
            self.LAMINA_BASE_N = len(self.LAMINA_BASE)
        except:
            self.LAMINA_BASE = 0
            self.LAMINA_BASE_N = 0
        try:
            # print(self.PETIOLE_TIP)
            self.PETIOLE_TIP_N = len(self.PETIOLE_TIP)
        except:
            self.PETIOLE_TIP = 0
            self.PETIOLE_TIP_N = 0
        try:
            # print(self.LAMINA_WIDTH)
            self.LAMINA_WIDTH_N = len(self.LAMINA_WIDTH)
        except:
            self.LAMINA_WIDTH = 0
            self.LAMINA_WIDTH_N = 0
        try:
            # print(self.MIDVEIN_TRACE)
            self.MIDVEIN_TRACE_N = len(self.MIDVEIN_TRACE)
        except:
            self.MIDVEIN_TRACE = 0
            self.MIDVEIN_TRACE_N = 0
        try:
            # print(self.PETIOLE_TRACE)
            self.PETIOLE_TRACE_N = len(self.PETIOLE_TRACE)
        except:
            self.PETIOLE_TRACE = 0
            self.PETIOLE_TRACE_N = 0
        try:
            # print(self.APEX_ANGLE)
            self.APEX_ANGLE_N = len(self.APEX_ANGLE)
        except:
            self.APEX_ANGLE = 0
            self.APEX_ANGLE_N = 0
        try:
            # print(self.BASE_ANGLE)
            self.BASE_ANGLE_N = len(self.BASE_ANGLE)
        except:
            self.BASE_ANGLE = 0
            self.BASE_ANGLE_N = 0

        try:
            # print(self.BASE_ANGLE)
            self.DEEPEST_SINUS_ANGLE_N = len(self.DEEPEST_SINUS_ANGLE)
        except:
            self.DEEPEST_SINUS_ANGLE = 0
            self.DEEPEST_SINUS_ANGLE_N = 0

        try:
            # print(self.LOBE_TIP)
            self.LOBE_TIP_N = len(self.LOBE_TIP)
        except:
            self.LOBE_TIP = 0
            self.LOBE_TIP_N = 0

        if add_pts_counts: # will add the counts of the number of points to the dataframe
            headers = ['img_name','img_filename',
                    'lamina_length','midvein_length','petiole_length','petiole_trace_length','lamina_width','apex_angle','base_angle','angle_types','lobe_count',
                    'loc_lobe_tip','loc_lamina_tip','loc_lamina_base','loc_petiole_tip','loc_lamina_width','loc_midvein_trace','loc_petiole_trace','loc_apex_angle','loc_base_angle','loc_deepest_sinus',
                    'loc_lobe_tip_n','loc_lamina_tip_n','loc_lamina_base_n','loc_petiole_tip_n','loc_lamina_width_n','loc_midvein_trace_n','loc_petiole_trace_n','loc_apex_angle_n','loc_base_angle_n','loc_deepest_sinus_n']
            data = {'img_name':[self.IMG_NAME],'img_filename':[self.IMG_FILENAME],
                    'lamina_length':[self.M_LAMINA_LENGTH],'midvein_length':[self.M_MIDVEIN_LENGTH],'petiole_length':[self.M_PETIOLE_LENGTH],'petiole_trace_length':[self.M_PETIOLE_TRACE_LENGTH],'lamina_width':[self.M_LAMINA_WIDTH],
                    'apex_angle':[self.M_APEX_ANGLE],'base_angle':[self.M_BASE_ANGLE],'angle_types':[self.ANGLE_TYPES],'lobe_count':[self.M_LOBE_COUNT],
                    'loc_lobe_tip':[self.LOBE_TIP],'loc_lamina_tip':[self.LAMINA_TIP],'loc_lamina_base':[self.LAMINA_BASE],'loc_petiole_tip':[self.PETIOLE_TIP],'loc_lamina_width':[self.LAMINA_WIDTH],
                    'loc_midvein_trace':[self.MIDVEIN_TRACE],'loc_petiole_trace':[self.PETIOLE_TRACE],'loc_apex_angle':[self.APEX_ANGLE],'loc_base_angle':[self.BASE_ANGLE],'loc_deepest_sinus':[self.DEEPEST_SINUS_ANGLE],
                    'loc_lobe_tip_n':[self.LOBE_TIP_N],'loc_lamina_tip_n':[self.LAMINA_TIP_N],'loc_lamina_base_n':[self.LAMINA_BASE_N],'loc_petiole_tip_n':[self.PETIOLE_TIP_N],'loc_lamina_width_n':[self.LAMINA_WIDTH_N],
                    'loc_midvein_trace_n':[self.MIDVEIN_TRACE_N],'loc_petiole_trace_n':[self.PETIOLE_TRACE_N],'loc_apex_angle_n':[self.APEX_ANGLE_N],'loc_base_angle_n':[self.BASE_ANGLE_N],'loc_deepest_sinus_n':[self.DEEPEST_SINUS_ANGLE_N]}
            df = pd.DataFrame(data,headers)
            df = df.iloc[[0]]
        else: # this is the one that gets saved to csv
            headers = ['img_name','img_filename',
                    'lamina_length','midvein_length','petiole_length','petiole_trace_length','lamina_width','apex_angle','base_angle','angle_types','lobe_count','deepest_sinus',
                    'loc_lobe_tip','loc_lamina_tip','loc_lamina_base','loc_petiole_tip','loc_lamina_width','loc_midvein_trace','loc_petiole_trace','loc_apex_angle','loc_base_angle','loc_deepest_sinus']
            data = {'img_name':[self.IMG_NAME],'img_filename':[self.IMG_FILENAME],
                    'lamina_length':[self.M_LAMINA_LENGTH],'midvein_length':[self.M_MIDVEIN_LENGTH],'petiole_length':[self.M_PETIOLE_LENGTH],'petiole_trace_length':[self.M_PETIOLE_TRACE_LENGTH],'lamina_width':[self.M_LAMINA_WIDTH],
                    'apex_angle':[self.M_APEX_ANGLE],'base_angle':[self.M_BASE_ANGLE],'angle_types':[self.ANGLE_TYPES],'lobe_count':[self.M_LOBE_COUNT],'deepest_sinus':[self.M_DEEPEST_SINUS_ANGLE],
                    'loc_lobe_tip':[self.LOBE_TIP],'loc_lamina_tip':[self.LAMINA_TIP],'loc_lamina_base':[self.LAMINA_BASE],'loc_petiole_tip':[self.PETIOLE_TIP],'loc_lamina_width':[self.LAMINA_WIDTH],
                    'loc_midvein_trace':[self.MIDVEIN_TRACE],'loc_petiole_trace':[self.PETIOLE_TRACE],'loc_apex_angle':[self.APEX_ANGLE],'loc_base_angle':[self.BASE_ANGLE],'loc_deepest_sinus':[self.DEEPEST_SINUS_ANGLE]}
            df = pd.DataFrame(data,headers)
            df = df.iloc[[0]]

        return df

def get_point_locations(project_data,label_path,partition):
    headers = ['filename','count','locations']
    locs = ['loc_lobe_tip','loc_lamina_tip','loc_lamina_base','loc_petiole_tip','loc_lamina_width','loc_midvein_trace','loc_petiole_trace','loc_apex_angle','loc_base_angle','loc_deepest_sinus']
    pt_type = ['lobe_tip','lamina_tip','lamina_base','petiole_tip','lamina_width','midvein_trace','petiole_trace','apex_angle','base_angle','deepest_sinus']
    locs_counts = ['loc_lobe_tip_n','loc_lamina_tip_n','loc_lamina_base_n','loc_petiole_tip_n','loc_lamina_width_n','loc_midvein_trace_n','loc_petiole_trace_n','loc_apex_angle_n','loc_base_angle_n','loc_deepest_sinus_n']
    
    for i, loc in enumerate(locs):
        try: 
            # combine_data = []
            names = project_data['img_filename'].tolist()
            pts = project_data[loc].tolist()
            counts = locs_counts[i]
            n_pts = project_data[counts].tolist()

            
            df = pd.DataFrame()
            df[headers[0]] = names
            df[headers[1]] = n_pts
            df[headers[2]] = pts

            # Remove rows with no labels
            df = df[df['count'] > 0]

            # Remove rows with incorrent number of annotations
            if pt_type[i] in ['lamina_tip','lamina_base','petiole_tip']:
                df = df[df['count'] == 1]
            elif pt_type[i] in ['lamina_width']:
                df = df[df['count'] == 2]
            elif pt_type[i] in ['apex_angle','base_angle','deepest_sinus']:
                df = df[df['count'] == 3]

            if partition == 'train':
                label_path_type = os.path.join(label_path, pt_type[i] + '__' + 'train' + '__gt.csv')
            elif partition == 'val':
                label_path_type = os.path.join(label_path, pt_type[i] + '__' + 'val' + '__gt.csv')
            elif partition == 'test':
                label_path_type = os.path.join(label_path, pt_type[i] + '__' + 'test' + '__gt.csv')
            if os.path.isfile(label_path_type):
                # If the file already exists, append to it
                df.to_csv(label_path_type, mode='a', header=False, index=False)
            else:
                # If the file doesn't exist, create it
                df.to_csv(label_path_type, index=False)
        except:
            print(f"{bcolors.FAIL}\n      Failed to export {loc}{bcolors.ENDC}")

def export_points(opt):
    projects = opt.client.get_projects()
    nProjects = len(list(projects))

    for project in tqdm(projects, desc=f'{bcolors.HEADER}Overall Progress{bcolors.ENDC}',colour="magenta",position=0,total = nProjects):
        print(f"{bcolors.BOLD}\n      Project Name: {project.name} UID: {project.uid}{bcolors.ENDC}")
        print(f"{bcolors.BOLD}      Annotations left to review: {project.review_metrics(None)}{bcolors.ENDC}")

        if project.name in opt.IGNORE:
            continue
        else:
            # if project.name == "PLANT_REU_All_Leaves":
            if project.review_metrics(None) >= 0:#0: 
                sep = '_'
                annoType = project.name.split('_')[0]
                setType = project.name.split('_')[1]
                datasetName = project.name.split('_')[2:]
                datasetName = sep.join(datasetName)

                if annoType in opt.RESTRICT_ANNOTYPE:
                    dirState = False
                    if opt.CUMMULATIVE:
                        # Define JSON name
                        saveNameJSON_LB = '.'.join([os.path.join(opt.DIR_JSON,project.name), 'json'])
                        # Define JSON name, YOLO
                        saveNameJSON_YOLO = opt.DIR_DATASETS
                        saveNameJSON_YOLO_data = os.path.join(opt.DIR_DATASETS,'data')
                        saveNameJSON_YOLO_label = os.path.join(opt.DIR_DATASETS,'images')
                        validate_dir(saveNameJSON_YOLO_data)
                        validate_dir(saveNameJSON_YOLO_label)
                    else:
                        # Define JSON name
                        saveNameJSON_LB = '.'.join([os.path.join(opt.DIR_JSON,opt.PROJECT_NAME,project.name), 'json'])
                        # Define JSON name, YOLO
                        saveNameJSON_YOLO = os.path.join(opt.DIR_DATASETS,opt.PROJECT_NAME,project.name)
                        saveNameJSON_YOLO_data = os.path.join(opt.DIR_DATASETS,opt.PROJECT_NAME,project.name,'data')
                        saveNameJSON_YOLO_label = os.path.join(opt.DIR_DATASETS,opt.PROJECT_NAME,project.name,'images')
                        validate_dir(os.path.join(opt.DIR_JSON,opt.PROJECT_NAME))
                        validate_dir(saveNameJSON_YOLO_data)
                        validate_dir(saveNameJSON_YOLO_label)

                    # If new dir is created, then continue, or continue from REDO
                    # dirState = validateDir(saveDir_LBa)
                    dirState = redo_JSON(saveNameJSON_LB)

                    if opt.REDO or opt.CUMMULATIVE:
                        dirState = True

                    if dirState:
                        # if DO_PARTITION_DATA:
                        #     validateDir_short(os.path.join(saveNameJSON_YOLO,'labels','train'))
                        #     validateDir_short(os.path.join(saveNameJSON_YOLO,'images','train'))
                        #     validateDir_short(os.path.join(saveNameJSON_YOLO,'labels','val'))
                        #     validateDir_short(os.path.join(saveNameJSON_YOLO,'images','val'))
                        #     validateDir_short(os.path.join(saveNameJSON_YOLO,'labels','test'))
                        #     validateDir_short(os.path.join(saveNameJSON_YOLO,'images','test'))
                        # else:
                        #     validateDir_short(os.path.join(saveNameJSON_YOLO,'labels','train'))
                        #     validateDir_short(os.path.join(saveNameJSON_YOLO,'images','train'))

                        # Show the labels
                        labels = project.export_labels()
                        try:
                            jsonFile = requests.get(labels) 
                        except:
                            try:
                                time.sleep(30)
                                labels = project.export_labels()
                                jsonFile = requests.get(labels)
                            except: 
                                time.sleep(30)
                                labels = project.export_labels()
                                jsonFile = requests.get(labels) 
                        jsonFile = jsonFile.json()
                        # print(jsonFile)

                        '''
                        Save JSON file in labelbox format
                        '''
                        # validate_dir(os.path.abspath(os.path.join(saveDir_LB,annoType)))
                        with open(saveNameJSON_LB, 'w', encoding='utf-8') as f:
                            json.dump(jsonFile, f, ensure_ascii=False)
                        '''
                        Convert labelbox JSON to YOLO & split into train/val/test
                        '''
                        # Convert Labelbox JSON labels to YOLO labels
                        names = []  # class names

                        # Reference the original file as it's saved
                        file = saveNameJSON_LB
                        data = jsonFile

                        nImgs = len(data)
                        if nImgs == 0: # Unstarted datasets
                            continue
                        else:
                            if opt.DO_PARTITION_DATA:
                                x = np.arange(0,nImgs)
                                split_size = (1 - float(opt.RATIO))
                                TRAIN,EVAL = train_test_split(x, test_size=split_size, random_state=4)
                                VAL, TEST = train_test_split(EVAL, test_size=0.5, random_state=4)

                            pc = 0
                            cc = "green" if annoType == 'PLANT' else "cyan"
                            
                            project_data = pd.DataFrame()
                            project_data_counts_train = pd.DataFrame()
                            project_data_counts_val = pd.DataFrame()
                            project_data_counts_test = pd.DataFrame()

                            if annoType == "RULER":
                                project_data_path = os.path.join(saveNameJSON_YOLO_data, project.name+'__ConversionFactor.csv')
                            else:
                                project_data_path = os.path.join(saveNameJSON_YOLO_data, project.name+'__Data.csv')



                            for img in tqdm(data, desc=f'{bcolors.BOLD}      Converting  >>>  {file}{bcolors.ENDC}',colour=cc,position=0):
                                im_path = img['Labeled Data']
                                try:
                                    im = Image.open(requests.get(im_path, stream=True).raw if im_path.startswith('http') else im_path)  # open
                                except:
                                    time.sleep(30)
                                    im = Image.open(requests.get(im_path, stream=True).raw if im_path.startswith('http') else im_path)  # open
                                    try:
                                        time.sleep(30)
                                        im = Image.open(requests.get(im_path, stream=True).raw if im_path.startswith('http') else im_path)  # open
                                    except:
                                        time.sleep(30)
                                        im = Image.open(requests.get(im_path, stream=True).raw if im_path.startswith('http') else im_path)  # open
                                width, height = im.size  # image size
                                fname = Path(img['External ID']).with_suffix('.txt').name
                                # if DO_PARTITION_DATA:
                                # label_path_train = os.path.join(saveNameJSON_YOLO,'labels','train')
                                # label_path_val = os.path.join(saveNameJSON_YOLO,'labels','val')
                                # label_path_test = os.path.join(saveNameJSON_YOLO,'labels','test')
                                # validateDir_short(label_path_train)
                                # validateDir_short(label_path_val)
                                # validateDir_short(label_path_test)

                                image_path = os.path.join(saveNameJSON_YOLO_label, img['External ID'])
                                im.save(Path(image_path).with_suffix('.jpg'), quality=100, subsampling=0)
                                    # if pc in TRAIN:
                                    #     # label_path = os.path.join(saveNameJSON_YOLO,'labels','train',fname)
                                    #     image_path = os.path.join(saveNameJSON_YOLO,'images','train',img['External ID'])
                                    #     im.save(Path(image_path).with_suffix('.jpg'), quality=100, subsampling=0) # WW edited this line; added Path(image_path).with_suffix('.jpg')
                                    # elif pc in VAL:
                                    #     # label_path = os.path.join(saveNameJSON_YOLO,'labels','val',fname)
                                    #     image_path = os.path.join(saveNameJSON_YOLO,'images','val',img['External ID'])
                                    #     im.save(Path(image_path).with_suffix('.jpg'), quality=100, subsampling=0) # WW edited this line; added Path(image_path).with_suffix('.jpg')
                                    # elif pc in TEST:
                                    #     # label_path = os.path.join(saveNameJSON_YOLO,'labels','test',fname)
                                    #     image_path = os.path.join(saveNameJSON_YOLO,'images','test',img['External ID'])
                                    #     im.save(Path(image_path).with_suffix('.jpg'), quality=100, subsampling=0) # WW edited this line; added Path(image_path).with_suffix('.jpg')
                                # else:
                                    # label_path = os.path.join(saveNameJSON_YOLO,'labels','train',fname)
                                    # image_path = os.path.join(saveNameJSON_YOLO,'images','train',img['External ID'])
                                    # im.save(Path(image_path).with_suffix('.jpg'), quality=100, subsampling=0) # WW edited this line; added Path(image_path).with_suffix('.jpg')
                                img_name = img['External ID']
                                img_filename = img_name+'.jpg'
                                
                                
                                Labels = Points(IMG_NAME=img_name,IMG_FILENAME=img_filename)
                                for label in img['Label']['objects']:
                                    # single
                                    if label['value'] == 'lamina_tip':
                                        pt_x,pt_y = label['point'].values()
                                        Labels.LAMINA_TIP = [(pt_x,pt_y)]
                                    if label['value'] == 'lamina_base':
                                        pt_x,pt_y = label['point'].values()
                                        Labels.LAMINA_BASE = [(pt_x,pt_y)]
                                    if label['value'] == 'petiole_tip':
                                        pt_x,pt_y = label['point'].values()
                                        Labels.PETIOLE_TIP = [(pt_x,pt_y)]
                                    # list strange
                                    if label['value'] == 'lobe_tip':
                                        pt_x,pt_y = label['point'].values()
                                        try:
                                            lobes = Labels.LOBE_TIP
                                        except:
                                            Labels.LOBE_TIP = []
                                            lobes = Labels.LOBE_TIP
                                        lobes.append((pt_x,pt_y))
                                        Labels.LOBE_TIP = lobes
                                        # Labels.LOBE_TIP.append((pt_x,pt_y))
                                        Labels.M_LOBE_COUNT = Labels.M_LOBE_COUNT + 1


                                    # list
                                    if label['value'] == '1_cm':
                                        trace = []
                                        for row in  label['line']:
                                            pt_x,pt_y = row.values()
                                            trace.append((pt_x,pt_y))
                                        Labels.CM_1 = trace


                                    if label['value'] == 'midvein_trace':
                                        trace = []
                                        for row in  label['line']:
                                            pt_x,pt_y = row.values()
                                            trace.append((pt_x,pt_y))
                                        Labels.MIDVEIN_TRACE = trace
                                    if label['value'] == 'petiole_trace':
                                        trace = []
                                        for row in  label['line']:
                                            pt_x,pt_y = row.values()
                                            trace.append((pt_x,pt_y))
                                        Labels.PETIOLE_TRACE = trace
                                    if label['value'] == 'width':
                                        trace = []
                                        for row in  label['line']:
                                            pt_x,pt_y = row.values()
                                            trace.append((pt_x,pt_y))
                                        Labels.LAMINA_WIDTH = trace
                                    if label['value'] == 'apex_angle':
                                        trace = []
                                        for row in  label['line']:
                                            pt_x,pt_y = row.values()
                                            trace.append((pt_x,pt_y))
                                        Labels.APEX_ANGLE = trace
                                    if label['value'] == 'base_angle':
                                        trace = []
                                        for row in  label['line']:
                                            pt_x,pt_y = row.values()
                                            trace.append((pt_x,pt_y))
                                        Labels.BASE_ANGLE = trace

                                    if label['value'] == 'deepest_sinus_angle':
                                        trace = []
                                        for row in  label['line']:
                                            pt_x,pt_y = row.values()
                                            trace.append((pt_x,pt_y))
                                        Labels.DEEPEST_SINUS_ANGLE.append(trace)
                                        Labels.M_DEEPEST_SINUS_ANGLE = Labels.M_DEEPEST_SINUS_ANGLE + 1
                                # angle type
                                try:
                                    for label in img['Label']['classifications']:
                                        if label['value'] == 'angles':
                                            Labels.ANGLE_TYPES = []
                                            for ans in label['answers']:
                                                answer = ans['value']
                                                # try:
                                                #     all_answers = Labels.ANGLE_TYPES
                                                # except:
                                                #     Labels.ANGLE_TYPES = []
                                                #     all_answers = Labels.ANGLE_TYPES
                                                # all_answers.append(answer)
                                                Labels.ANGLE_TYPES.append(answer)
                                except:
                                    continue
                                    
                                # Calculate
                                if annoType == "RULER":
                                    try:
                                        Labels.calculate_cm()
                                        img_data = Labels.export_ruler()
                                        combine_data = [project_data,img_data]
                                        project_data = pd.concat(combine_data,ignore_index=True)
                                    except:
                                        continue
                                else:
                                    Labels.calculate_measurements()
                                    img_data = Labels.export(add_pts_counts=False)
                                    combine_data = [project_data,img_data]
                                    project_data = pd.concat(combine_data,ignore_index=True)

                                    if opt.DO_PARTITION_DATA:

                                        if pc in TRAIN:
                                            img_data_counts = Labels.export(add_pts_counts=True)
                                            combine_data_counts_train = [project_data_counts_train,img_data_counts]
                                            project_data_counts_train = pd.concat(combine_data_counts_train,ignore_index=True)
                                        elif pc in VAL:
                                            img_data_counts = Labels.export(add_pts_counts=True)
                                            combine_data_counts_val = [project_data_counts_val,img_data_counts]
                                            project_data_counts_val = pd.concat(combine_data_counts_val,ignore_index=True)
                                        elif pc in TEST:
                                            img_data_counts = Labels.export(add_pts_counts=True)
                                            combine_data_counts_test = [project_data_counts_test,img_data_counts]
                                            project_data_counts_test = pd.concat(combine_data_counts_test,ignore_index=True)
                                    

                                    # label_locations = Labels.export_ind_gt_labels_for_PD(label_path_train,label_path_val,label_path_test) label_path
                                    # label_locations = Labels.export_ind_gt_labels_for_PD(label_path)

                                    # LAMINA_TIP: tuple = field(init=False)
                                    # LAMINA_BASE: tuple = field(init=False)
                                    # PETIOLE_TIP: tuple = field(init=False)
                                    # LOBE_TIP: list[tuple] = field(init=False)
                                    # LAMINA_WIDTH: list[tuple] = field(init=False)
                                    # MIDVEIN_TRACE: list[tuple] = field(init=False)
                                    # PETIOLE_TRACE: list[tuple] = field(init=False)
                                    # APEX_ANGLE: list[tuple] = field(init=False)
                                    # BASE_ANGLE: list[tuple] = field(init=False)
                                    # ANGLE_TYPES: str = field(init=False)

                                    # # box
                                    # top, left, h, w = label['bbox'].values()  # top, left, height, width
                                    # xywh = [(left + w / 2) / width, (top + h / 2) / height, w / width, h / height]  # xywh normalized

                                    # # class
                                    # cls = label['value']  # class name
                                    # if cls not in names:
                                    #     names.append(cls)
                                    
                                    # # set the index based on the order of the annotations in LabelBox
                                    # annoInd = setIndexOfAnnotation(cls,annoType)

                                    # line = annoInd, *xywh  # YOLO format (class_index, xywh)
                                    # with open(label_path, 'a') as f:
                                    #     f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                pc += 1
                            # Save dataset.yaml
                            # if USE_TEMPLATE_YAML: # Easy to make manually, then we're not overwriting with less than the max number of 'nc': len(names)
                            #     continue
                            # else:
                            #     sep = ''
                            #     if CUMMULATIVE:
                            #         p0 = os.path.abspath(os.path.join(annoType + CUMMULATIVE_suffix))
                            #     else:
                            #         p0 = os.path.abspath(os.path.join(saveDir_YOLO,annoType,project.name))
                            #     p1 = os.path.join('images','train')
                            #     p2 = os.path.join('images','val') # 'images/val'
                            #     p3 = os.path.join('images','test') #'images/test'
                            #     d = {'path': p0,
                            #         'train': p1,
                            #         'val': p2, 
                            #         'test': p3,
                            #         'nc': len(names),
                            #         'names': names}  # dictionary

                            #     if CUMMULATIVE:
                            #         if ((len(names) == 11) & (annoType == 'PLANT')):
                            #             with open(os.path.join(saveNameJSON_YOLO, Path(annoType + CUMMULATIVE_suffix).with_suffix('.yaml')), 'w') as f:
                            #                 yaml.dump(d, f, sort_keys=False)
                            #         elif ((len(names) == 9) & (annoType == 'PREP')):
                            #             with open(os.path.join(saveNameJSON_YOLO, Path(annoType + CUMMULATIVE_suffix).with_suffix('.yaml')), 'w') as f:
                            #                 yaml.dump(d, f, sort_keys=False)
                            #     else:
                            #         with open(os.path.join(saveNameJSON_YOLO, Path(file).with_suffix('.yaml').name), 'w') as f:
                            #             yaml.dump(d, f, sort_keys=False)

                            # # Zip
                            # if ZIP:
                            #     print(f'Zipping as {saveNameJSON_YOLO}.zip...')
                            #     os.system(f'zip -qr {saveNameJSON_YOLO}.zip {saveNameJSON_YOLO}')
                            # print(f"{bcolors.OKGREEN}      Conversion successful :) {project.name} {project.uid} {bcolors.ENDC}")
                            project_data.to_csv(project_data_path,index=False)

                            get_point_locations(project_data_counts_train,saveNameJSON_YOLO_label,'train')
                            get_point_locations(project_data_counts_val,saveNameJSON_YOLO_label,'val')
                            get_point_locations(project_data_counts_test,saveNameJSON_YOLO_label,'test')
                            

                        

def export_points_labels():
    # Read configs
    dir_private = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path_cfg_private = os.path.join(dir_private,'PRIVATE_DATA.yaml')
    cfg_private = get_cfg_from_full_path(path_cfg_private)

    dir_labeling = os.path.dirname(__file__)
    path_cfg = os.path.join(dir_labeling,'config_export_points_labels_from_Labelbox.yaml')
    cfg = get_cfg_from_full_path(path_cfg)


    LB_API_KEY = cfg_private['labelbox']['LB_API_KEY']
    client = Client(api_key=LB_API_KEY)

    opt = OPTS_EXPORT_POINTS(cfg, client)

    validate_dir(opt.DIR_ROOT)
    validate_dir(opt.DIR_DATASETS)   
    validate_dir(opt.DIR_JSON)   

    print(f"{bcolors.HEADER}Beginning Export for Project {opt.PROJECT_NAME}{bcolors.ENDC}")
    print(f"{bcolors.BOLD}      Labels will go to --> {opt.DIR_DATASETS}{bcolors.ENDC}")
    export_points(opt)
    print(f"{bcolors.OKGREEN}Finished Export :){bcolors.ENDC}")

if __name__ == '__main__':
    export_points_labels()                      