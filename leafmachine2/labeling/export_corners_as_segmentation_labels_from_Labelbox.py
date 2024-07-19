# save the labelbox groundtruth overlay images
import time, os, inspect, json, requests, sys
import labelbox
from labelbox.data.annotation_types import Polygon
from labelbox import Client
from PIL import Image # pillow
import numpy as np
from pathlib import Path
from skimage.measure import find_contours
from shapely.geometry import Polygon, MultiPolygon
import cv2
import io
import yaml #pyyaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from detectron2.structures import BoxMode
from math import dist
import datetime

# pip install git+https://github.com/waspinator/pycococreator.git@0.2.0
from pycococreatortools import pycococreatortools

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from machine.general_utils import get_cfg_from_full_path, bcolors, validate_dir
from utils_Labelbox import assign_index, redo_JSON, OPTS_EXPORT_SEG

# from utils import make_dirs
'''
https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#register-a-coco-format-dataset
https://scikit-image.org/docs/stable/auto_examples/edges/plot_contours.html
https://www.immersivelimit.com/create-coco-annotations-from-scratch
https://patrickwasp.com/create-your-own-coco-style-dataset/
https://github.com/cocodataset/cocoapi/issues/144
https://github.com/waspinator/pycococreator
https://github.com/waspinator/pycococreator/blob/master/examples/shapes/shapes_to_coco.py
https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=U5LhISJqWXgM
'''

def setIndexOfAnnotation(cls):
    if cls == 'Background':
        print("Error: Background not used")
        # annoInd = 1
        # color = [0,0,0]
    elif cls == 'Leaf':
        annoInd = 1
        color = [0,255,46]
    elif cls == 'Petiole':
        annoInd = 2
        color = [255,173,0]
    elif cls == 'Hole':
        annoInd = 3
        color = [255,0,209]
    return annoInd, color

def redoJSON(fname):
    try:
        doProject = False
        jsonFile = open(fname)
    except:
        doProject = True
    return doProject

def convert_PNG_to_COCO_format(img_all_labels, label_img, label_class):
    label_gray = label_img.convert('LA').convert('1')
    # label_bi = label_gray.point( lambda p: 255 if p > 0 else 0 )
    label_bi = np.array(label_gray)
    label_bi = label_bi.astype(np.uint8)

    label_bi[label_bi > 0] = label_class

    img_all_labels = img_all_labels + label_bi
    img_all_labels[img_all_labels > 2] = 3 # holes
    return img_all_labels,label_bi

def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def rotational_sort(xy_coords):
    cx, cy = xy_coords.mean(0)
    x, y = xy_coords.T
    angles = np.arctan2(x-cx, y-cy)
    indices = np.argsort(angles)
    return xy_coords[indices]

def sort_by_dist(xy_coords):
    # sharp corners may cause skipping, resulting in 
    # long distances at the end. So we'll just skip those
    xy_coords_del = xy_coords
    sorted_list = np.zeros((xy_coords.shape[0], 2))
    first = xy_coords[0,:]
    sorted_list[0,:] = first
    xy_coords_del = np.delete(xy_coords, 0, axis=0)
    ind = 0
    while xy_coords_del.shape[0] > 0:
         
        dist_2 = np.sum((xy_coords_del - first)**2, axis=1)
        next = xy_coords_del[np.argmin(dist_2)]   

        xy_coords_del = np.delete(xy_coords_del, np.argmin(dist_2), axis=0)
        
        if dist(first,next) < 5:
            ind += 1
            sorted_list[ind,:] = next
        first = next

    sorted_list = sorted_list[0:ind+1,:]
    return sorted_list

def build_coco_dict():
    INFO = {
        "description": "LeafMachine2 Leaf Segmentation",
        "url": "https://leafmachine.org",
        "version": "1.0.0",
        "year": 2023,
        "contributor": "William Weaver",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [
        {
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }
    ]

    CATEGORIES = [
        {
            'id': 1,
            'name': 'leaf',
            'supercategory': 'wholeleaf',
        },
        {
            'id': 2,
            'name': 'petiole',
            'supercategory': 'wholeleaf',
        },
        {
            'id': 3,
            'name': 'hole',
            'supercategory': 'wholeleaf',
        }
    ]
    coco_output_train = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    coco_output_val = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    return coco_output_train, coco_output_val

def get_label_img_with_retry(anno):
    try:
        label_img = Image.open(requests.get(anno['instanceURI'], stream=True).raw if anno['instanceURI'].startswith('http') else anno['instanceURI'])
        do_continue = True
    except:
        try:
            print('trying again... 1')
            time.sleep(5)
            label_img = Image.open(requests.get(anno['instanceURI'], stream=True).raw if anno['instanceURI'].startswith('http') else anno['instanceURI'])
            do_continue = True
        except:
            try:
                print('trying again... 2')
                time.sleep(5)
                label_img = Image.open(requests.get(anno['instanceURI'], stream=True).raw if anno['instanceURI'].startswith('http') else anno['instanceURI'])
                do_continue = True
            except:
                try:
                    print('trying again... 3')
                    time.sleep(5)
                    label_img = Image.open(requests.get(anno['instanceURI'], stream=True).raw if anno['instanceURI'].startswith('http') else anno['instanceURI'])
                    do_continue = True
                except:
                    do_continue = False
    return label_img, do_continue

def get_contours(labelID, contours, label_class, pc, TRAIN, coco_output_train, image_info_train, VAL, coco_output_val, image_info_val):
    segmentation = []
    segmentations = []
    polygons = []
    for contour in contours:
        labelID +=1
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        if len(contours) == 1:
            # Make a polygon and simplify it
                poly = Polygon(contour)
                poly = poly.simplify(1.0, preserve_topology=False)
                polygons.append(poly)

                multi_poly = MultiPolygon(polygons)
                try:
                    x, y, max_x, max_y = multi_poly.bounds
                    width = max_x - x
                    height = max_y - y
                    bbox = (x, y, max_x, max_y) #bbox = (x, y, width, height)
                    area = multi_poly.area

                    segmentation = np.array(poly.exterior.coords).ravel().tolist()

                    annotation_info = {
                        "id": labelID,
                        "image_id": str(pc),
                        "category_id": label_class,
                        "iscrowd": 0,
                        "area": area,
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [segmentation],
                        "width": width,
                        "height": height,
                    } 
                    
                    if pc in TRAIN:
                        if segmentation is not None:
                            coco_output_train["annotations"].append(annotation_info)
                            image_info_train["annotations"].append(annotation_info)
                    elif pc in VAL:
                        if segmentation is not None:
                            coco_output_val["annotations"].append(annotation_info)
                            image_info_val["annotations"].append(annotation_info)
                except:
                    polygons = None
                    segmentations = None
                    segmentation = None
        elif len(contours) > 1:
            if contour.shape[0] > 50:
                # Make a polygon and simplify it
                poly = Polygon(contour)
                print(poly.geom_type)
                try:
                    poly = poly.simplify(1.0, preserve_topology=False)

                    # try:
                    segmentation = np.array(poly.exterior.coords).ravel().tolist()
                    segmentations.append(segmentation)
                    polygons.append(poly)
                except:
                    poly = max(poly, key=lambda a: a.area)
                    poly = Polygon(poly)
                    print(poly.type)
                    poly = poly.simplify(1.0, preserve_topology=False)

                    # try:
                    segmentation = np.array(poly.exterior.coords).ravel().tolist()
                    segmentations.append(segmentation)
                    polygons.append(poly)

                    # except:
                    #     continue
    return polygons, segmentations, segmentation

def sort_corners(corners_unordered):
    # Sort points by y-coordinate (ascending), then by x-coordinate (ascending)
    corners_sorted = sorted(corners_unordered, key=lambda x: (x[1], x[0]))
    
    # Determine the top and bottom points
    top_points = corners_sorted[:2]
    bottom_points = corners_sorted[2:]
    
    # Sort top points by x-coordinate to get top_left and top_right
    top_left, top_right = sorted(top_points, key=lambda x: x[0])
    
    # Sort bottom points by x-coordinate to get bottom_left and bottom_right
    bottom_right, bottom_left  = sorted(bottom_points, key=lambda x: x[0])
    
    # Return the ordered points
    return [top_left, top_right, bottom_left, bottom_right]

def create_json_packet(contours, polygons, segmentations, segmentation, labelID, pc, label_class, TRAIN, coco_output_train, image_info_train, VAL, coco_output_val, image_info_val):
    if (len(contours) > 1) and (len(polygons) > 0):
        multi_poly = MultiPolygon(polygons)
        if multi_poly.bounds is not None:
            x, y, max_x, max_y = multi_poly.bounds
            width = max_x - x
            height = max_y - y
            bbox = (x, y, max_x, max_y) #bbox = (x, y, width, height)
            area = multi_poly.area

            annotation_info = {
                "id": labelID,
                "image_id": str(pc),
                "category_id": label_class,
                "iscrowd": 0,
                "area": area,
                "bbox": bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": segmentations,
                "width": width,
                "height": height,
            } 
            if pc in TRAIN:
                if segmentation is not None:
                    coco_output_train["annotations"].append(annotation_info)
                    image_info_train["annotations"].append(annotation_info)
            elif pc in VAL:
                if segmentation is not None:
                    coco_output_val["annotations"].append(annotation_info)
                    image_info_val["annotations"].append(annotation_info)
    return coco_output_val, image_info_val

def create_color_png(img_all_labels, n_holes):
    colors_base = [[0,0,0],[0,255,46], [255,173,0], [255,0,209]]
    if n_holes > 0:
        hole_list = [[255,0,209]]
        if n_holes > 1:
            for i in range(1, n_holes):
                hole_list.append([255,0,209])
        colors_base.extend(hole_list)
    colors = np.array(colors_base)
    img_all_labels_color = colors[img_all_labels]
    return img_all_labels_color

def export_segmentation(opt):
    projects = opt.client.get_projects()
    nProjects = len(list(projects))
    for project in tqdm(projects, desc=f'{bcolors.HEADER}Overall Progress{bcolors.ENDC}',colour="magenta",position=0,total = nProjects):
        print(f"{bcolors.BOLD}\n      Project Name: {project.name} UID: {project.uid}{bcolors.ENDC}")
        print(f"{bcolors.BOLD}      Annotations left to review: {project.review_metrics(None)}{bcolors.ENDC}")

        if (project.name in opt.INCLUDE) or (opt.INCLUDE == []):
            # if project.review_metrics(None) >= 0: 
            sep = '_'
            annoType = project.name.split('_')[0]
            setType = project.name.split('_')[1]
            datasetName = project.name.split('_')[2:]
            datasetName = sep.join(datasetName)

            if annoType in opt.RESTRICT_ANNOTYPE:
                dirState = False
                if opt.CUMMULATIVE:
                    # Define JSON name
                    path_LB_json = os.path.join(opt.DIR_JSON,project.name) + '.json'
                    # Define JSON name, YOLO
                    dir_seg = opt.DIR_DATASETS
                    validate_dir(dir_seg)
                else:
                    # Define JSON name
                    path_LB_json = os.path.join(opt.DIR_JSON,project.name) + '.json'
                    # Define JSON name, YOLO
                    dir_seg = os.path.join(opt.DIR_DATASETS,project.name)

                # If new dir is created, then continue, or continue from REDO
                dirState = redoJSON(path_LB_json)

                if opt.REDO:
                    dirState = True

                if dirState:
                    # Get the labels
                    # labels = project.export_labels()
                    # try:
                    #     jsonFile = requests.get(labels) 
                    # except:
                    #     try:
                    #         time.sleep(30)
                    #         labels = project.export_labels()
                    #         jsonFile = requests.get(labels)
                    #     except: 
                    #         time.sleep(30)
                    #         labels = project.export_labels()
                    #         jsonFile = requests.get(labels) 
                    # jsonFile = jsonFile.json()
                    params = {
                            "data_row_details": True,
                            "metadata_fields": True,
                            "attachments": True,
                            "project_details": True,
                            "performance_details": True,
                            "label_details": True,
                            "interpolated_frames": True,
                            "embeddings": True}

                    export_task = project.export_v2(params=params)
                    export_task.wait_till_done()
                    if export_task.errors:
                        print(export_task.errors)
                    export_json = export_task.result
                    # print(export_json)
                    jsonFile = export_json
                    # print(jsonFile)

                    '''
                    Save JSON file in labelbox format
                    '''
                    # with open(path_LB_json, 'w', encoding='utf-8') as f:
                    #     json.dump(jsonFile, f, ensure_ascii=False)
                    '''
                    Convert labelbox JSON to YOLO & split into train/val/test
                    '''
                    nImgs = len(jsonFile)
                    if nImgs == 0: # Unstarted datasets
                        continue
                    else:
                        x = np.arange(0,nImgs)
                        TRAIN,VAL = train_test_split(x, test_size=(1-opt.RATIO), random_state=4)
                        pc = 0
                        cc = "yellow"
                        
                        path_train_mask = os.path.join(dir_seg,'train','masks')
                        path_train_image = os.path.join(dir_seg,'train','images')

                        path_val_mask = os.path.join(dir_seg,'val','masks')
                        path_val_image = os.path.join(dir_seg,'val','images')

                        validate_dir(path_train_mask)
                        validate_dir(path_train_image)
                        validate_dir(path_val_mask)
                        validate_dir(path_val_image)

                        completed_images = os.listdir(path_val_mask)

                        coco_output_train, coco_output_val = build_coco_dict()

                        labelID = 100000
                        for img in tqdm(jsonFile, desc=f'{bcolors.BOLD}      Converting  >>>  {path_LB_json}{bcolors.ENDC}',colour=cc,position=0):
                            img_color_path = img['data_row']['row_data']
                            external_id = img['data_row']['external_id']
                            project_id = img['data_row']['id']
                            for proj in img['projects']:
                                if img['projects'][proj]['labels'][0]['performance_details']['skipped']:
                                    continue

                                #### enable this to only take reviewed images
                                # elif (img['Reviews'] == []) and (opt.ONLY_REVIEWED):
                                    # continue
                                else:   
                                    # if (img['Reviews'] == []):
                                    #     print('UNREVIEWED')
                                    # imgID += 1 
                                    try:
                                        img_color = Image.open(requests.get(img_color_path, stream=True).raw if img_color_path.startswith('http') else img_color_path)  # open
                                    except:
                                        try:
                                            time.sleep(30)
                                            img_color = Image.open(requests.get(img_color_path, stream=True).raw if img_color_path.startswith('http') else img_color_path)  # open
                                        except:
                                            time.sleep(30)
                                            img_color = Image.open(requests.get(img_color_path, stream=True).raw if img_color_path.startswith('http') else img_color_path)  # open
                                    
                                    width, height = img_color.size  # image size
                                    fname_mask = Path(external_id).with_suffix('.png').name
                                    fname_color = Path(external_id).with_suffix('.jpg').name

                                    # fname_mask_compare = '__'.join([fname_mask.split('__')[1], fname_mask.split('__')[2]])

                                    # if fname_mask == 'NY_1931248368_Ulmaceae_Ulmus_americana__39.png':
                                    # if fname_mask_compare not in completed_images:
                                    if pc in TRAIN:
                                        image_info_train = pycococreatortools.create_image_info(str(pc), os.path.basename(fname_color), img_color.size)
                                        image_info_train["annotations"] = []
                                        image_info_val = [] # so the get_contontour() is happy
                                    elif pc in VAL:
                                        image_info_val = pycococreatortools.create_image_info(str(pc), os.path.basename(fname_color), img_color.size)
                                        image_info_val["annotations"] = []
                                        image_info_train = []# so the get_contontour() is happy


                                    img_all_labels = np.zeros((img_color.size[1],img_color.size[0]), np.uint8)#np.zeros((im.size[1],im.size[0])).astype(np.uint8)
                                    n_holes = 0
                                    corners_unordered = []
                                    corners_ordered = []

                                    for proj in img['projects']:
                                        for label in img['projects'][proj]['labels'][0]['annotations']['objects']:
                                            # list
                                            if label['value'] == 'corner':
                                                pt_x,pt_y = label['point'].values()
                                                corners_unordered.append((int(pt_x),int(pt_y)))

                                            if label['value'] == 'corner_infer':
                                                pt_x,pt_y = label['point'].values()
                                                corners_unordered.append((int(pt_x),int(pt_y)))

                                    if corners_unordered:
                                        # Sort and assign the ordered corners
                                        corners_ordered = sort_corners(corners_unordered)

                                        labelID += 1
                                        label_class = 1
                                        label_color = [0, 255, 46]

                                        # Create a blank image
                                        img_all_labels = np.zeros((height, width), dtype=np.uint8)

                                        # Draw the polygon on the blank image
                                        pts = np.array(corners_ordered, dtype=np.int32)
                                        pts = pts.reshape((-1, 1, 2))
                                        cv2.fillPoly(img_all_labels, [pts], label_class)

                                        contours = find_contours(img_all_labels, 0.5, positive_orientation='low')
                                        print(fname_mask)
                                        polygons, segmentations, segmentation = get_contours(labelID, contours, label_class, pc, TRAIN, coco_output_train, image_info_train, VAL, coco_output_val, image_info_val)
                                        
                                        if segmentation:

                                            coco_output_val, image_info_val = create_json_packet(contours, polygons, segmentations, segmentation, labelID, pc, label_class, TRAIN, coco_output_train, image_info_train, VAL, coco_output_val, image_info_val)

                                            # Change 0, 1, 2, 3 to a color
                                            img_all_labels_color = create_color_png(img_all_labels, n_holes)

                                            # route the images
                                            if pc in TRAIN:
                                                img_color.save(os.path.join(path_train_image, fname_color))
                                                cv2.imwrite(os.path.join(path_train_mask, fname_mask), img_all_labels_color)
                                            elif pc in VAL:
                                                img_color.save(os.path.join(path_val_image, fname_color))
                                                cv2.imwrite(os.path.join(path_val_mask, fname_mask), img_all_labels_color)

                                            # route the json packet addition
                                            if pc in TRAIN:
                                                coco_output_train["images"].append(image_info_train)
                                            elif pc in VAL:
                                                coco_output_val["images"].append(image_info_val)
                                pc += 1

                        jsonTrain_Polygons = os.path.join(dir_seg,'train','images',"POLYGONS_train") + '.json'
                        jsonVal_Polygons = os.path.join(dir_seg,'val','images',"POLYGONS_val") + '.json'

                        with open(jsonTrain_Polygons, 'w') as output_json_file:
                            json.dump(coco_output_train, output_json_file)
                        with open(jsonVal_Polygons, 'w') as output_json_file:
                            json.dump(coco_output_val, output_json_file)

def export_segmentation_labels():
    # Read configs
    dir_private = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path_cfg_private = os.path.join(dir_private,'PRIVATE_DATA.yaml')
    cfg_private = get_cfg_from_full_path(path_cfg_private)

    dir_labeling = os.path.dirname(__file__)
    path_cfg = os.path.join(dir_labeling,'config_export_corners_as_segmentation_labels_from_Labelbox.yaml')
    cfg = get_cfg_from_full_path(path_cfg)

    LB_API_KEY = cfg_private['labelbox']['LB_API_KEY']
    client = Client(api_key=LB_API_KEY)

    opt = OPTS_EXPORT_SEG(cfg, client)

    validate_dir(opt.DIR_ROOT)
    validate_dir(opt.DIR_DATASETS)   
    validate_dir(opt.DIR_JSON)   

    print(f"{bcolors.HEADER}Beginning Export for Project {opt.PROJECT_NAME}{bcolors.ENDC}")
    print(f"{bcolors.BOLD}      Labels will go to --> {opt.DIR_DATASETS}{bcolors.ENDC}")
    export_segmentation(opt)
    print(f"{bcolors.OKGREEN}Finished Export :){bcolors.ENDC}")

if __name__ == '__main__':
    export_segmentation_labels()