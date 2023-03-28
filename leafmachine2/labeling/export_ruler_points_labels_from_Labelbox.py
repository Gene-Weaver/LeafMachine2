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
import statistics
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

    CM: list[tuple] = field(init=False)
    CM_avg: list[tuple] = field(init=False)
    pooled_sd_value: list[tuple] = field(init=False)
    N: list[tuple] = field(init=False)
    max_dim: list[tuple] = field(init=False)
    rsd: list[tuple] = field(init=False)

    cm_1: list[tuple] = field(init=False)
    cm_half: list[tuple] = field(init=False)
    cm_4th: list[tuple] = field(init=False)
    mm_1: list[tuple] = field(init=False)
    in_1: list[tuple] = field(init=False)
    in_8th: list[tuple] = field(init=False)
    in_4th: list[tuple] = field(init=False)
    in_half: list[tuple] = field(init=False)
    in_16th: list[tuple] = field(init=False)
    mm_half: list[tuple] = field(init=False)
    

    # def __post_init__(self) -> None:
    #     self.LOBE_TIP = []
    def total_distance(self,pts):
        total = 0
        for i in range(len(pts) - 1):
            total += math.dist(pts[i],pts[i+1])
        return total

    def find_angle(self,pts,reflex,location):
        isReflex = False
        for ans in reflex:
            if location == 'apex':
                if ans == 'apex_more_than_180':
                    isReflex = True
            if location == 'base':
                if ans == 'base_more_than_180':
                    isReflex = True
        
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

    def pooled_sd(self, std_devs, sample_sizes):
        k = len(std_devs)
        if k != len(sample_sizes):
            raise ValueError("Length of std_devs and sample_sizes must be equal.")
        numerator = sum([(n-1)*s**2 for n, s in zip(sample_sizes, std_devs)])
        denominator = sum(sample_sizes) - k
        return math.sqrt(numerator / denominator)

    def pairwise_distance(self, coords):
        n = len(coords)
        distances = []
        for i in range(n-1):
            dist = math.sqrt((coords[i][0] - coords[i+1][0])**2 + (coords[i][1] - coords[i+1][1])**2)
            distances.append(dist)
        return distances

    def avg_min_pairwise_distance(self, coords):
        distances = self.pairwise_distance(coords)
        avg_dist = sum(distances) / len(distances)
        std_dev = math.sqrt(sum([(d - avg_dist)**2 for d in distances]) / len(distances))
        return avg_dist, std_dev, len(distances)

    def calculate_cm(self, pt_list, factor, is_metric):
        avg_dist, std_dev, n_pts = self.avg_min_pairwise_distance(pt_list)

        if is_metric:
            self.CM = avg_dist*factor
        else:
            INCH = avg_dist*factor
            self.CM = np.divide(INCH, 2.54)

        return self.CM, std_dev, n_pts

    def export_ruler(self):
        headers = ['img_name','img_filename','cm_1',]
        data = {'img_name':[self.IMG_NAME],'img_filename':[self.IMG_FILENAME],'cm_1':[self.CM],}
        df = pd.DataFrame(data,headers)
        df = df.iloc[[0]]
        return df

    def export_ruler_avg(self):
        headers = ['img_name','img_filename','cm_1_avg','pooled_sd', 'n_pts', 'max_dim','rsd']
        data = {'img_name':[self.IMG_NAME],'img_filename':[self.IMG_FILENAME],'cm_1_avg':[self.CM_avg],'pooled_sd':[self.pooled_sd_value],'n_pts':[self.N],'max_dim':[self.max_dim],'rsd':[self.rsd],}
        df = pd.DataFrame(data,headers)
        df = df.iloc[[0]]
        return df

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
                            project_data_avg = pd.DataFrame()

                            project_data_counts_train = pd.DataFrame()
                            project_data_counts_val = pd.DataFrame()
                            project_data_counts_test = pd.DataFrame()
                            if annoType == "RULER":
                                project_data_path = os.path.join(saveNameJSON_YOLO_data, project.name+'__ConversionFactor.csv')
                                project_data_path_avg = os.path.join(saveNameJSON_YOLO_data, project.name+'__ConversionFactor_avg.csv')
                            else:
                                project_data_path = os.path.join(saveNameJSON_YOLO_data, project.name+'__Data.csv')
                                project_data_path_avg = os.path.join(saveNameJSON_YOLO_data, project.name+'__Data_avg.csv')



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
                                
                                CM_list = []
                                CM_list_sd = []
                                CM_list_n = []
                                rsd = []

                                Labels = Points(IMG_NAME=img_name,IMG_FILENAME=img_filename)
                                for label in img['Label']['objects']:
                                    do_skip = False
                                    if label['value'] == '1_cm':
                                        trace = []
                                        for row in label['line']:
                                            pt_x,pt_y = row.values()
                                            trace.append((pt_x,pt_y))
                                        Labels.cm_1 = trace
                                        Labels.CM, std_dev, n_pts = Labels.calculate_cm(Labels.cm_1, 1, True)

                                    elif label['value'] == 'half_cm':
                                        trace = []
                                        for row in label['line']:
                                            pt_x,pt_y = row.values()
                                            trace.append((pt_x,pt_y))
                                        Labels.cm_half = trace
                                        Labels.CM, std_dev, n_pts = Labels.calculate_cm(Labels.cm_half, 2, True)
                                    
                                    elif label['value'] == '4th_cm':
                                        trace = []
                                        for row in label['line']:
                                            pt_x,pt_y = row.values()
                                            trace.append((pt_x,pt_y))
                                        Labels.cm_4th = trace
                                        Labels.CM, std_dev, n_pts = Labels.calculate_cm(Labels.cm_4th, 4, True)

                                    elif label['value'] == '1_mm':
                                        trace = []
                                        for row in label['line']:
                                            pt_x,pt_y = row.values()
                                            trace.append((pt_x,pt_y))
                                        Labels.mm_1 = trace
                                        Labels.CM, std_dev, n_pts = Labels.calculate_cm(Labels.mm_1, 10, True)

                                    elif label['value'] == 'half_mm':
                                        trace = []
                                        for row in label['line']:
                                            pt_x,pt_y = row.values()
                                            trace.append((pt_x,pt_y))
                                        Labels.mm_half = trace
                                        Labels.CM, std_dev, n_pts = Labels.calculate_cm(Labels.mm_half, 20, True)

                                    elif label['value'] == '1_in':
                                        trace = []
                                        for row in label['line']:
                                            pt_x,pt_y = row.values()
                                            trace.append((pt_x,pt_y))
                                        Labels.in_1 = trace
                                        Labels.CM, std_dev, n_pts = Labels.calculate_cm(Labels.in_1, 1, False)

                                    elif label['value'] == '8th_in':
                                        trace = []
                                        for row in label['line']:
                                            pt_x,pt_y = row.values()
                                            trace.append((pt_x,pt_y))
                                        Labels.in_8th = trace
                                        Labels.CM, std_dev, n_pts = Labels.calculate_cm(Labels.in_8th, 8, False)

                                    elif label['value'] == '4th_in':
                                        trace = []
                                        for row in label['line']:
                                            pt_x,pt_y = row.values()
                                            trace.append((pt_x,pt_y))
                                        Labels.in_4th = trace
                                        Labels.CM, std_dev, n_pts = Labels.calculate_cm(Labels.in_4th, 4, False)

                                    elif label['value'] == 'half_in':
                                        trace = []
                                        for row in label['line']:
                                            pt_x,pt_y = row.values()
                                            trace.append((pt_x,pt_y))
                                        Labels.in_half = trace
                                        Labels.CM, std_dev, n_pts = Labels.calculate_cm(Labels.in_half, 2, False)

                                    elif label['value'] == '16th_in':
                                        trace = []
                                        for row in label['line']:
                                            pt_x,pt_y = row.values()
                                            trace.append((pt_x,pt_y))
                                        Labels.in_16th = trace
                                        Labels.CM, std_dev, n_pts= Labels.calculate_cm(Labels.in_16th, 16, False)

                                    else:
                                        do_skip = True #overall
                                        pass
                                    
                                    if not do_skip:
                                        print(f"img_name {Labels.IMG_NAME} --- 1_cm {Labels.CM}")

                                        CM_list.append(Labels.CM)
                                        CM_list_sd.append(std_dev)
                                        CM_list_n.append(n_pts)
                                        # Calculate
                                        if annoType == "RULER":
                                            # try:
                                            img_data = Labels.export_ruler()
                                            combine_data = [project_data,img_data]
                                            project_data = pd.concat(combine_data,ignore_index=True)
                                            # except:
                                                # continue
                                if CM_list != []:
                                    Labels.pooled_sd_value = Labels.pooled_sd(CM_list_sd, CM_list_n)
                                    Labels.CM_avg = sum(CM_list) / len(CM_list)
                                    Labels.N = sum(CM_list_n)
                                    Labels.rsd = std_dev_and_sd_units(CM_list)
                                else:
                                    Labels.pooled_sd_value = 99999
                                    Labels.CM_avg = 99999
                                    Labels.N = 99999
                                    Labels.rsd = 99999
                                Labels.max_dim = max([im.width, im.height])
                                if annoType == "RULER":
                                    # try:
                                    img_data_avg = Labels.export_ruler_avg()
                                    combine_data_avg = [project_data_avg,img_data_avg]
                                    project_data_avg = pd.concat(combine_data_avg,ignore_index=True)

                                pc += 1

                            project_data.to_csv(project_data_path,index=False)
                            project_data_avg.to_csv(project_data_path_avg,index=False)

def percent_error(actual_values):
    mean = sum(actual_values) / len(actual_values)
    errors = []
    for value in actual_values:
        error = abs(value - mean) / mean * 100
        errors.append(error)
    return errors        

def std_dev_and_sd_units(lst):
    if len(lst) > 1:
        # calculate the standard deviation
        sd = statistics.stdev(lst)

        # calculate the mean
        mean = statistics.mean(lst)

        # calculate the number of units that correspond to one standard deviation
        sd_units = sd / mean * 100
    else:
        sd_units =0

    return sd_units

def export_points_labels():
    # Read configs
    dir_private = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path_cfg_private = os.path.join(dir_private,'PRIVATE_DATA.yaml')
    cfg_private = get_cfg_from_full_path(path_cfg_private)

    dir_labeling = os.path.dirname(__file__)
    path_cfg = os.path.join(dir_labeling,'config_export_ruler_points_labels_from_Labelbox.yaml')
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