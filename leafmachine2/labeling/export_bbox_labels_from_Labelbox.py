# save the labelbox groundtruth overlay images
import time, os, inspect, json, requests, sys
from labelbox import Client, OntologyBuilder
from PIL import Image # pillow
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from machine.general_utils import get_cfg_from_full_path, bcolors, validate_dir
from utils_Labelbox import assign_index, redo_JSON, OPTS_EXPORT
# from utils import make_dirs

def save_color_csv(opt, project):
    print(f"{bcolors.BOLD}      Saving color profile{bcolors.ENDC}")
    # Create a mapping for the colors
    hex_to_rgb = lambda hex_color: tuple(
        int(hex_color[i + 1:i + 3], 16) for i in (0, 2, 4))
    colors = {
        tool.name: hex_to_rgb(tool.color)
        for tool in OntologyBuilder.from_project(project).tools
    }
    csv_name = 'ColorProfile__' + project.name + '.csv'
    csv_name_save = os.path.join(opt.DIR_COLORS,csv_name)
    with open(csv_name_save, 'w') as f:
        for key in colors.keys():
            rgb = colors[key]
            f.write("%s, %s,%s,%s\n" % (key, rgb[0],rgb[1],rgb[2]))



# CHECK !!!!!
# file_stem = img['DataRow ID'] # use img['External ID']
def export_bbox(opt):
    projects = opt.client.get_projects()
    nProjects = len(list(projects))
    for project in tqdm(projects, desc=f'{bcolors.HEADER}Overall Progress{bcolors.ENDC}',colour="magenta",position=0,total = nProjects):
        print(f"{bcolors.BOLD}\n      Project Name: {project.name} UID: {project.uid}{bcolors.ENDC}")
        print(f"{bcolors.BOLD}      Annotations left to review: {project.review_metrics(None)}{bcolors.ENDC}")

        if project.name in opt.IGNORE:
            print(f"{bcolors.BOLD}      Skipping {project.name}{bcolors.ENDC}")
        elif (project.name in opt.INCLUDE) or (opt.INCLUDE == []):
            if project.review_metrics(None) >= 0: 
                sep = '_'
                annoType = project.name.split('_')[0]
                setType = project.name.split('_')[1]
                datasetName = project.name.split('_')[2:]
                datasetName = sep.join(datasetName)

                if annoType in opt.RESTRICT_ANNOTYPE:
                    dirState = False
                    if opt.CUMMULATIVE:
                        # Define JSON name
                        saveDir_LBa = os.path.join(opt.DIR_ROOT, annoType)
                        saveNameJSON_LB = ''.join([os.path.join(opt.DIR_DATASETS, project.name),'.json'])
                        # Define JSON name, YOLO
                        saveNameJSON_YOLO = opt.DIR_DATASETS
                    else:
                        # Define JSON name
                        saveDir_LBa = os.path.join(opt.DIR_ROOT,annoType)
                        saveNameJSON_LB = ''.join([os.path.join(opt.DIR_DATASETS, project.name),'.json'])
                        # Define JSON name, YOLO
                        saveNameJSON_YOLO = os.path.join(opt.DIR_DATASETS,project.name)

                    # If new dir is created, then continue, or continue from REDO
                    # dirState = validateDir(saveDir_LBa)
                    dirState = redo_JSON(saveNameJSON_LB)

                    if opt.REDO:
                        dirState = True

                    if dirState:
                        if opt.do_sort_compound_and_simple_leaves:
                            if opt.DO_PARTITION_DATA:
                                validate_dir(os.path.join(saveNameJSON_YOLO, 'compound','labels','train'))
                                validate_dir(os.path.join(saveNameJSON_YOLO, 'compound','images','train'))
                                validate_dir(os.path.join(saveNameJSON_YOLO, 'compound','labels','val'))
                                validate_dir(os.path.join(saveNameJSON_YOLO, 'compound','images','val'))
                                validate_dir(os.path.join(saveNameJSON_YOLO, 'compound','labels','test'))
                                validate_dir(os.path.join(saveNameJSON_YOLO, 'compound','images','test'))

                                validate_dir(os.path.join(saveNameJSON_YOLO, 'simple','labels','train'))
                                validate_dir(os.path.join(saveNameJSON_YOLO, 'simple','images','train'))
                                validate_dir(os.path.join(saveNameJSON_YOLO, 'simple','labels','val'))
                                validate_dir(os.path.join(saveNameJSON_YOLO, 'simple','images','val'))
                                validate_dir(os.path.join(saveNameJSON_YOLO, 'simple','labels','test'))
                                validate_dir(os.path.join(saveNameJSON_YOLO, 'simple','images','test'))
                            else:
                                validate_dir(os.path.join(saveNameJSON_YOLO, 'compound','labels','train'))
                                validate_dir(os.path.join(saveNameJSON_YOLO, 'compound','images','train'))

                                validate_dir(os.path.join(saveNameJSON_YOLO, 'simple','labels','train'))
                                validate_dir(os.path.join(saveNameJSON_YOLO, 'simple','images','train'))
                        else:
                            if opt.DO_PARTITION_DATA:
                                validate_dir(os.path.join(saveNameJSON_YOLO,'labels','train'))
                                validate_dir(os.path.join(saveNameJSON_YOLO,'images','train'))
                                validate_dir(os.path.join(saveNameJSON_YOLO,'labels','val'))
                                validate_dir(os.path.join(saveNameJSON_YOLO,'images','val'))
                                validate_dir(os.path.join(saveNameJSON_YOLO,'labels','test'))
                                validate_dir(os.path.join(saveNameJSON_YOLO,'images','test'))
                            else:
                                validate_dir(os.path.join(saveNameJSON_YOLO,'labels','train'))
                                validate_dir(os.path.join(saveNameJSON_YOLO,'images','train'))

                        # Show the labels
                        labels = project.export_labels()
                        try:
                            jsonFile = requests.get(labels) 
                        except:
                            try:
                                time.sleep(10)
                                labels = project.export_labels()
                                jsonFile = requests.get(labels)
                            except: 
                                time.sleep(10)
                                labels = project.export_labels()
                                jsonFile = requests.get(labels) 
                        jsonFile = jsonFile.json()
                        # print(jsonFile)

                        '''
                        Save JSON file in labelbox format
                        '''
                        # validate_dir(os.path.join(opt.DIR_ROOT,annoType))
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
                                try:
                                    x = np.arange(0,nImgs)
                                    split_size = (1 - float(opt.RATIO))
                                    TRAIN,EVAL = train_test_split(x, test_size = split_size, random_state = 4)
                                    VAL, TEST = train_test_split(EVAL, test_size = 0.5, random_state = 4)
                                except: # not enough images in project to partition, send all to train
                                    TRAIN = x
                                    VAL = []
                                    TEST = []
                            else:
                                TRAIN = 0
                                VAL = 0 
                                TEST = 0
                            
                            # save the colors to a csv for future refrence 
                            if opt.SAVE_COLOR_CSV:
                                save_color_csv(opt, project)

                            # Saving the labels to a txt file
                            save_labels_to_txt(opt, project, file, annoType, data, saveNameJSON_YOLO, TRAIN, VAL, TEST, names)

                            # Zip
                            if opt.ZIP:
                                print(f'Zipping as {opt.PROJECT_NAME}.zip...')
                                os.system(f'zip -qr {opt.PROJECT_NAME}.zip {opt.PROJECT_NAME}')
                            print(f"{bcolors.OKGREEN}      Conversion successful :) {project.name} {project.uid} {bcolors.ENDC}")
        else:
            print(f"{bcolors.WARNING}      Issue: {project.name} {project.uid} {bcolors.ENDC}")

def check_for_compound(img):
    for label in img['Label']['objects']:
        cls = label['value']
        if cls == 'leaflet':
            return True
        else:
            continue
    return False

def route_compound(opt, img, saveNameJSON_YOLO_orig, project):
    if opt.do_sort_compound_and_simple_leaves:
        is_compound = check_for_compound(img)
        if is_compound:
            saveNameJSON_YOLO = os.path.join(saveNameJSON_YOLO_orig,'compound')
            # cropped
            if opt.CUMMULATIVE:
                path_cropped = os.path.join(opt.DIR_DATASETS, 'compound')
            else:
                path_cropped = os.path.join(opt.DIR_DATASETS, project.name, 'compound')
        else:
            saveNameJSON_YOLO = os.path.join(saveNameJSON_YOLO_orig,'simple')
            # cropped
            if opt.CUMMULATIVE:
                path_cropped = os.path.join(opt.DIR_DATASETS, 'simple')
            else:
                path_cropped = os.path.join(opt.DIR_DATASETS, project.name, 'simple')
    else:
        saveNameJSON_YOLO = saveNameJSON_YOLO_orig
        # cropped
        if opt.CUMMULATIVE:
            path_cropped = opt.DIR_DATASETS
        else:
            path_cropped = os.path.join(opt.DIR_DATASETS, project.name)
    return saveNameJSON_YOLO, path_cropped


def save_labels_to_txt(opt, project, file, annoType, data, saveNameJSON_YOLO_orig, TRAIN, VAL, TEST, names):
    pc = 0
    cc = "green" if annoType == 'PLANT' else "cyan"
    for img in tqdm(data, desc=f'{bcolors.BOLD}      Converting  >>>  {file}{bcolors.ENDC}',colour=cc,position=0):
        if img['Skipped'] == True:
            continue
        elif (img['Reviews'] == []) and (opt.ONLY_REVIEWED):
            continue
        else:
            try:
                # check for compound vs. simple and segregate
                saveNameJSON_YOLO, path_cropped = route_compound(opt, img, saveNameJSON_YOLO_orig, project)

                # Get image
                im_path = img['Labeled Data']
                im = Image.open(requests.get(im_path, stream=True).raw if im_path.startswith('http') else im_path)  # open
                width, height = im.size  # image size
                file_stem = img['External ID'] # use img['External ID'] if the names are normal, img['DataRow ID'] if there are . in the name
                fname = Path(file_stem).with_suffix('.txt').name

                # partition
                if opt.DO_PARTITION_DATA:
                    if pc in TRAIN:
                        label_path = os.path.join(saveNameJSON_YOLO,'labels','train',fname)
                        image_path = os.path.join(saveNameJSON_YOLO,'images','train',file_stem)
                        im.save(Path(image_path).with_suffix('.jpg'), quality=100, subsampling=0) # WW edited this line; added Path(image_path).with_suffix('.jpg')
                    if pc in VAL:
                        label_path = os.path.join(saveNameJSON_YOLO,'labels','val',fname)
                        image_path = os.path.join(saveNameJSON_YOLO,'images','val',file_stem)
                        im.save(Path(image_path).with_suffix('.jpg'), quality=100, subsampling=0) # WW edited this line; added Path(image_path).with_suffix('.jpg')
                    if pc in TEST:
                        label_path = os.path.join(saveNameJSON_YOLO,'labels','test',fname)
                        image_path = os.path.join(saveNameJSON_YOLO,'images','test',file_stem)
                        im.save(Path(image_path).with_suffix('.jpg'), quality=100, subsampling=0) # WW edited this line; added Path(image_path).with_suffix('.jpg')
                else:
                    label_path = os.path.join(saveNameJSON_YOLO,'labels','train',fname)
                    image_path = os.path.join(saveNameJSON_YOLO,'images','train',file_stem)
                    im.save(Path(image_path).with_suffix('.jpg'), quality=100, subsampling=0) # WW edited this line; added Path(image_path).with_suffix('.jpg')

                label_ind = 0
                for label in img['Label']['objects']:
                    label_ind += 1
                    # box
                    top, left, h, w = label['bbox'].values()  # top, left, height, width
                    xywh = [(left + w / 2) / width, (top + h / 2) / height, w / width, h / height]  # xywh normalized

                    # class
                    cls = label['value']  # class name
                    if cls not in names:
                        names.append(cls)
                    
                    # set the index based on the order of the annotations in LabelBox
                    annoInd = assign_index(cls,annoType)

                    line = annoInd, *xywh  # YOLO format (class_index, xywh)
                    with open(label_path, 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')


                    if opt.do_save_cropped_bboxes_as_jpgs:
                        if (cls in opt.INCLUDE_ANNO) or (opt.INCLUDE_ANNO == []):
                            # rect_points = label.value.geometry['coordinates'][0]
                            right = left + w
                            bottom = top + h
                            img_crop = im.crop((left, top, right, bottom))

                            # Upscale acacia prickles/spines
                            if annoType == 'ACACIA':
                                # img_crop = img_crop.resize((img_crop.width * 10, img_crop.height * 10), resample=Image.BICUBIC)
                                img_crop = img_crop.resize((img_crop.width * 10, img_crop.height * 10), resample=Image.LANCZOS)

                            save_crop_name_base = remove_extension(img["External ID"])
                            if opt.DO_PARTITION_DATA:
                                if pc in TRAIN:
                                    save_crop_name = save_crop_name_base + '__' + str(label_ind)
                                    save_crop_dir = os.path.join(path_cropped, cls, 'train')
                                    save_crop_path = os.path.join(save_crop_dir, save_crop_name)
                                    validate_dir(save_crop_dir)
                                    img_crop.save(Path(save_crop_path).with_suffix('.jpg'), quality=100, subsampling=0)

                                if pc in VAL:
                                    save_crop_name = save_crop_name_base + '__' + str(label_ind)
                                    save_crop_dir = os.path.join(path_cropped, cls, 'val')
                                    save_crop_path = os.path.join(save_crop_dir, save_crop_name)
                                    validate_dir(save_crop_dir)
                                    img_crop.save(Path(save_crop_path).with_suffix('.jpg'), quality=100, subsampling=0)
                                if pc in TEST:
                                    save_crop_name = save_crop_name_base + '__' + str(label_ind)
                                    save_crop_dir = os.path.join(path_cropped, cls, 'test')
                                    save_crop_path = os.path.join(save_crop_dir, save_crop_name)
                                    validate_dir(save_crop_dir)
                                    img_crop.save(Path(save_crop_path).with_suffix('.jpg'), quality=100, subsampling=0)
                            else:
                                save_crop_name = save_crop_name_base + '__' + str(label_ind)
                                save_crop_dir = os.path.join(path_cropped, cls)
                                save_crop_path = os.path.join(save_crop_dir, save_crop_name)
                                validate_dir(save_crop_dir)
                                img_crop.save(Path(save_crop_path).with_suffix('.jpg'), quality=100, subsampling=0)
            except:
                pass
            pc += 1

def remove_extension(filename):
    return '.'.join(filename.split('.')[:-1]) if '.' in filename else filename


def export_bbox_labels():
    # Read configs
    dir_private = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path_cfg_private = os.path.join(dir_private,'PRIVATE_DATA.yaml')
    cfg_private = get_cfg_from_full_path(path_cfg_private)

    dir_labeling = os.path.dirname(__file__)
    path_cfg = os.path.join(dir_labeling,'config_export_bbox_labels_from_Labelbox.yaml')
    cfg = get_cfg_from_full_path(path_cfg)


    LB_API_KEY = cfg_private['labelbox']['LB_API_KEY']
    client = Client(api_key=LB_API_KEY)

    opt = OPTS_EXPORT(cfg, client)

    validate_dir(opt.DIR_ROOT)
    validate_dir(opt.DIR_DATASETS)   
    validate_dir(opt.DIR_COLORS)   

    print(f"{bcolors.HEADER}Beginning Export for Project {opt.PROJECT_NAME}{bcolors.ENDC}")
    print(f"{bcolors.BOLD}      Labels will go to --> {opt.DIR_DATASETS}{bcolors.ENDC}")
    export_bbox(opt)
    print(f"{bcolors.OKGREEN}Finished Export :){bcolors.ENDC}")

if __name__ == '__main__':
    export_bbox_labels()