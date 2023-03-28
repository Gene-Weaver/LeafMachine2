import os, cv2, yaml, math, sys, inspect, imutils
import numpy as np
from numpy import NAN, ndarray
import pandas as pd
from dataclasses import dataclass,field
from scipy import ndimage,stats
from scipy.signal import find_peaks
from scipy.stats.mstats import gmean
from skimage.measure import label, regionprops_table
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)
# from machine.general_utils import print_plain_to_console, print_blue_to_console, print_green_to_console, print_warning_to_console, print_cyan_to_console
# from machine.general_utils import bcolors

def convert_rulers(cfg, dir_home, Project, Dirs):
    RulerCFG = RulerConfig(dir_home, Dirs, cfg)

    for filename, analysis in Project.project_data.items():
        Project.project_data[filename]['Ruler_Info'] = []
        Project.project_data[filename]['Ruler_Data'] = []
        print(filename)
        try:
            full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpg'])))
        except:
            full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpeg'])))

        archival = analysis['Detections_Archival_Components']
        height = analysis['height']
        width = analysis['width']
        ruler_list = [row for row in archival if row[0] == 0]
        # print(ruler_list)
        if len(ruler_list) < 1:
            print('     MAKE THIS HAVE AN EMPTY PLACEHOLDER') # TODO ###################################################################################
        else:
            for ruler in ruler_list:
                ruler_location = yolo_to_position_ruler(ruler, height, width)
                ruler_polygon = [(ruler_location[1], ruler_location[2]), (ruler_location[3], ruler_location[2]), (ruler_location[3], ruler_location[4]), (ruler_location[1], ruler_location[4])]
                print(ruler_polygon)
                x_coords = [x for x, y in ruler_polygon]
                y_coords = [y for x, y in ruler_polygon]

                min_x, min_y = min(x_coords), min(y_coords)
                max_x, max_y = max(x_coords), max(y_coords)

                ruler_cropped = full_image[min_y:max_y, min_x:max_x]
                # img_crop = img[min_y:max_y, min_x:max_x]
                loc = '-'.join([str(min_x), str(min_y), str(max_x), str(max_y)])
                ruler_crop_name = '__'.join([filename,'R',loc])

                # Get the cropped image using cv2.getRectSubPix
                # ruler_cropped = cv2.getRectSubPix(full_image, (int(ruler_location[3] - ruler_location[1]), int(ruler_location[4] - ruler_location[2])), (points[0][0][0], points[0][0][1]))

                Ruler = setup_ruler(RulerCFG, ruler_cropped, ruler_crop_name)
                                
                Ruler, BlockCandidate = convert_pixels_to_metric(RulerCFG,Ruler,ruler_crop_name, Dirs)

                Project = add_ruler_to_Project(Project, Ruler, BlockCandidate, filename, ruler_crop_name)
                print('hi')
    
    
    print('hi')
    
    return Project



@dataclass
class RulerConfig:

    path_to_config: str = field(init=False)
    path_to_model: str = field(init=False)
    path_to_class_names: str = field(init=False)

    cfg: str = field(init=False)

    path_ruler_output_parent: str = field(init=False)
    dir_ruler_class_overlay: str = field(init=False)
    dir_ruler_overlay: str = field(init=False)
    dir_ruler_processed: str = field(init=False)
    dir_ruler_data: str = field(init=False)

    net_ruler: object = field(init=False)

    def __init__(self, dir_home, Dirs, cfg) -> None:
        self.path_to_config = dir_home
        self.cfg = cfg

        self.path_to_model = os.path.join(dir_home,'leafmachine2','machine','ruler_classifier','model')
        self.path_to_class_names = os.path.join(dir_home, 'leafmachine2','machine','ruler_classifier','ruler_classes.txt')

        self.path_ruler_output_parent = Dirs.ruler_info
        self.dir_ruler_class_overlay = Dirs.ruler_class_overlay
        self.dir_ruler_overlay =  Dirs.ruler_overlay
        self.dir_ruler_processed =  Dirs.ruler_processed
        self.dir_ruler_data =  Dirs.ruler_data

        # if self.cfg['leafmachine']['ruler_detection']['detect_ruler_type']:
        try:
            model_name = self.cfg['leafmachine']['ruler_detection']['ruler_detector']
            self.net_ruler = torch.jit.load(os.path.join(self.path_to_model,model_name))
            self.net_ruler.eval()
        except:
            print("Could not load ruler classifier network")

@dataclass
class ClassifyRulerImage:
    img_path: None
    img: ndarray = field(init=False)
    img_sq: ndarray = field(init=False)
    img_t: ndarray = field(init=False)
    img_tensor: object = field(init=False)
    transform: object = field(init=False)

    def __init__(self, img) -> None:
        try:
            self.img = img
        except:
            self.img = cv2.imread(self.img_path)
        self.img_sq = squarify(self.img,showImg=False,makeSquare=True,sz=360)
        # self.img_sq = squarify_tile_four_versions(self.img, showImg=False, makeSquare=True, sz=720) ### new 1-31-23
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.img_t = self.transforms(self.img_sq)
        self.img_tensor = torch.unsqueeze(self.img_t, 0).cuda()

@dataclass
class RulerImage:
    img_path: str
    img_fname: str

    img: ndarray = field(init=False)

    img_copy: ndarray = field(init=False)
    img_gray: ndarray = field(init=False)
    img_edges: ndarray = field(init=False)
    img_bi_display: ndarray = field(init=False)
    img_bi: ndarray = field(init=False)
    img_best: ndarray = field(init=False)
    img_type_overlay: ndarray = field(init=False)
    img_ruler_overlay: ndarray = field(init=False)
    img_total_overlay: ndarray = field(init=False)
    img_block_overlay: ndarray = field(init=False)

    avg_angle: float = 0
    ruler_class: str = field(init=False)
    ruler_class_percentage: str = field(init=False)
    

    def __init__(self, img, img_fname) -> None:
        self.img = make_img_hor(img)
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img_copy = self.img.copy()
        self.img_fname = img_fname

@dataclass
class Block:
    img_bi: ndarray
    img_bi_overlay: ndarray
    img_bi_copy: ndarray = field(init=False)
    img_result: ndarray = field(init=False)
    use_points: list = field(init=False,default_factory=list)
    point_types: list = field(init=False,default_factory=list)
    x_points: list = field(init=False,default_factory=list)
    y_points: list = field(init=False,default_factory=list)
    axis_major_length: list = field(init=False,default_factory=list)
    axis_minor_length: list = field(init=False,default_factory=list)
    conversion_factor: list = field(init=False,default_factory=list)
    conversion_location: list = field(init=False,default_factory=list)
    conversion_location_options: str = field(init=False)
    success_sort: str = field(init=False)

    largest_blobs: list = field(init=False,default_factory=list)
    remaining_blobs: list = field(init=False,default_factory=list)

    plot_points_1cm: list = field(init=False,default_factory=list)
    plot_points_10cm: list = field(init=False,default_factory=list)
    plot_points: list = field(init=False,default_factory=list)

    def __post_init__(self) -> None:
        self.img_bi_copy = self.img_bi
        self.img_bi[self.img_bi < 128] = 0
        self.img_bi[self.img_bi >= 128] = 255
        self.img_bi_copy[self.img_bi_copy < 40] = 0
        self.img_bi_copy[self.img_bi_copy >= 40] = 255

    def whiter_thresh(self) -> None:
        self.img_bi_copy[self.img_bi_copy < 240] = 0
        self.img_bi_copy[self.img_bi_copy >= 240] = 255

'''
####################################
####################################
                Basics
####################################
####################################
'''
def add_ruler_to_Project(Project, Ruler, BlockCandidate, filename, ruler_crop_name):
    if 'block' in Ruler.ruler_class:
        Project.project_data[filename]['Ruler_Info'].append({ruler_crop_name: Ruler})
        Project.project_data[filename]['Ruler_Data'].append({ruler_crop_name: BlockCandidate})

    return Project

def yolo_to_position_ruler(annotation, height, width):
    return ['ruler', 
        int((annotation[1] * width) - ((annotation[3] * width) / 2)), 
        int((annotation[2] * height) - ((annotation[4] * height) / 2)), 
        int(annotation[3] * width) + int((annotation[1] * width) - ((annotation[3] * width) / 2)), 
        int(annotation[4] * height) + int((annotation[2] * height) - ((annotation[4] * height) / 2))]

def make_img_hor(img):
    # Make image horizontal
    try:
        h,w,c = img.shape
    except:
        h,w = img.shape
    if h > w:
        img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def create_overlay_bg(RulerCFG, img):
    try:
        try:
            h,w,_ = img.shape
            imgBG = np.zeros([h+60,w,3], dtype=np.uint8)
            imgBG[:] = 0
        except:
            img = binary_to_color(img)
            h,w,_ = img.shape
            imgBG = np.zeros([h+60,w,3], dtype=np.uint8)
            imgBG[:] = 0

        try:
            imgBG[60:img.shape[0]+60, :img.shape[1],:] = img
        except:
            imgBG[60:img.shape[0]+60, :img.shape[1]] = img

    except Exception as e:
        m = ''.join(['create_overlay_bg() exception: ',e.args[0]])
        Print_Verbose(RulerCFG.cfg, 2, m).warning()
        img = np.stack((img,)*3, axis=-1)
        h,w,_ = img.shape
        imgBG = np.zeros([h+60,w,3], dtype=np.uint8)
        imgBG[:] = 0

        imgBG[60:img.shape[0]+60,:img.shape[1],:] = img
    return imgBG

def binary_to_color(binary_image):
    color_image = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)
    color_image[binary_image == 1] = (255, 255, 255)
    return color_image

def pad_binary_img(img,h,w,n):
    imgBG = np.zeros([h+n,w], dtype=np.uint8)
    imgBG[:] = 0
    imgBG[:h,:w] = img
    return imgBG

def stack_2_imgs(img1,img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    img3 = np.zeros((h1+h2, max(w1,w2),3), dtype=np.uint8)
    img3[:,:] = (255,255,255)

    img3[:h1, :w1,:3] = img1
    try:
        img3[h1:h1+h2, :w2,:3] = img2
    except:
        img3[h1:h1+h2, :w2,:3] = binary_to_color(img2)
    return img3

def check_ruler_type(ruler_class,option):
    ind = ruler_class.find(option)
    if ind == -1:
        return False
    else:
        return True

def create_white_bg(img,squarifyRatio,h,w):
    w_plus = w
    # if (w_plus % squarifyRatio) != 0:
    # while (w_plus % squarifyRatio) != 0:
    #     w_plus += 1
    
    imgBG = np.zeros([h,w_plus,3], dtype=np.uint8)
    imgBG[:] = 255

    imgBG[:img.shape[0],:img.shape[1],:] = img
    # cv2.imshow('Single Channel Window', imgBG)
    # cv2.waitKey(0)
    return imgBG

def stack_image_quartile_rotate45(img, q_increment, h, w, showImg):
    # cv2.imshow('Original', img)
    # cv2.waitKey(0)

    rotate_options = [-135, -45, 45, 135]

    imgBG = np.zeros([h*2,h*2,3], dtype=np.uint8)
    imgBG[:] = 255

    increment = 0
    for row in range(0,2):
        for col in range(0,2):
            ONE = (row * h)
            TWO = ((row * h) + h)
            THREE = (col * h)
            FOUR = (col * h) + h

            one = (q_increment*increment)
            two = (q_increment*increment) + h

            if (increment < 3) and (two < w):
                # imgBG[ONE : TWO, THREE : FOUR] = img[:, one : two]
                rotated = imutils.rotate_bound(img[:, one : two], rotate_options[increment])
                add_dim1 = rotated.shape[0] - ONE
                add_dim2 = rotated.shape[0] - TWO
                add_dim3 = rotated.shape[0] - THREE
                add_dim4 = rotated.shape[0] - FOUR
                imgBG[ONE : TWO, THREE : FOUR] = cv2.resize(rotated,  (FOUR - THREE, TWO - ONE))
            else:
                # imgBG[ONE : TWO, THREE : FOUR] = img[:, w - h : w]
                rotated = imutils.rotate_bound(img[:, w - h : w], rotate_options[increment])
                imgBG[ONE : TWO, THREE : FOUR] = cv2.resize(rotated,  (FOUR - THREE, TWO - ONE))
            increment += 1


    if showImg:
        cv2.imshow('squarify_quartile()', imgBG)
        cv2.waitKey(0)
    return imgBG

def stack_image_quartile(img, q_increment, h, w, showImg):
    # cv2.imshow('Original', img)
    # cv2.waitKey(0)

    imgBG = np.zeros([h*2,h*2,3], dtype=np.uint8)
    imgBG[:] = 255

    increment = 0
    for row in range(0,2):
        for col in range(0,2):
            ONE = (row * h)
            TWO = ((row * h) + h)
            THREE = (col * h)
            FOUR = (col * h) + h

            one = (q_increment*increment)
            two = (q_increment*increment) + h

            if (increment < 3) and (two < w):
                imgBG[ONE : TWO, THREE : FOUR] = img[:, one : two]
            else:
                imgBG[ONE : TWO, THREE : FOUR] = img[:, w - h : w]
            # imgBG[row * h : h * (col + 1), row * h : h * (col + 1)] = img[:, (col * h) + (q_increment * increment) : (col * (q_increment * increment)) + h]
            increment += 1
            # if showImg:
            #     cv2.imshow('Single Channel Window', imgBG)
            #     cv2.waitKey(0)
    # # top left
    # imgBG[0:h , 0:h] = img[:, 0:h]
    # # top right
    # imgBG[0:h, h:h*2] = img[:, q_increment:q_increment+h]
    # # Bottom left
    # imgBG[h:h*2, 0:h] = img[:, 2*q_increment:2*q_increment+h]
    # # Bottom right
    # imgBG[h:h*2, h:h*2] = img[:, w-h:w]

    if showImg:
        cv2.imshow('squarify_quartile()', imgBG)
        cv2.waitKey(0)
    return imgBG

def stack_image_nine(img, q_increment, h, w, showImg):
    # cv2.imshow('Original', img)
    # cv2.waitKey(0)

    imgBG = np.zeros([h*3,h*3,3], dtype=np.uint8)
    imgBG[:] = 255

    increment = 0
    for row in range(0,3):
        for col in range(0,3):
            ONE = (row * h)
            TWO = ((row * h) + h)
            THREE = (col * h)
            FOUR = (col * h) + h

            one = (q_increment*increment)
            two = (q_increment*increment) + h

            if (increment < 8) and (two < w):
                imgBG[ONE : TWO, THREE : FOUR] = img[:, one : two]
            else:
                imgBG[ONE : TWO, THREE : FOUR] = img[:, w - h : w]
            # imgBG[row * h : h * (col + 1), row * h : h * (col + 1)] = img[:, (col * h) + (q_increment * increment) : (col * (q_increment * increment)) + h]
            increment += 1
            # if showImg:
            #     cv2.imshow('Single Channel Window', imgBG)
            #     cv2.waitKey(0)

    if showImg:
        cv2.imshow('squarify_nine()', imgBG)
        cv2.waitKey(0)
    return imgBG

def stack_image(img,squarifyRatio,h,w_plus,showImg):
    # cv2.imshow('Original', img)
    wChunk = int(w_plus/squarifyRatio)
    hTotal = int(h*squarifyRatio)
    imgBG = np.zeros([hTotal,wChunk,3], dtype=np.uint8)
    imgBG[:] = 255

    wStart = 0
    wEnd = wChunk
    for i in range(1,squarifyRatio+1):
        wStartImg = (wChunk*i)-wChunk
        wEndImg =  wChunk*i
        
        hStart = (i*h)-h
        hEnd = i*h
        # imgPiece = img[:,wStartImg:wEndImg]
        # cv2.imshow('Single Channel Window', imgPiece)
        # cv2.waitKey(0)
        imgBG[hStart:hEnd,wStart:wEnd] = img[:,wStartImg:wEndImg]
    if showImg:
        cv2.imshow('squarify()', imgBG)
        cv2.waitKey(0)
    return imgBG

def add_text_to_stacked_img(angle,img):
    addText1 = "Angle(deg):"+str(round(angle,3))+' Imgs:Orig,Binary,Rotated'
    img = cv2.putText(img=img, text=addText1, org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255),thickness=1)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    return img

def add_text_to_img(text,img):
    addText = text
    img = cv2.putText(img=img, text=addText, org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255),thickness=1)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    return img

'''
####################################
####################################
            Squarify
####################################
####################################
'''
def calc_squarify_ratio(img):
    doStack = False
    h,w,c = img.shape

    # Extend width so it's a multiple of h
    ratio = w/h
    ratio_plus = math.ceil(ratio)
    w_plus = ratio_plus*h

    ratio_go = w/h
    if ratio_go > 4:
        doStack = True

    squarifyRatio = 0
    if doStack:
        # print(f'This should equal 0 --> {w_plus % h}')
        for i in range(1,ratio_plus):
            if ((i*h) < (w_plus/i)):
                continue
            else:
                squarifyRatio = i - 1
                break
        # print(f'Optimal stack_h: {squarifyRatio}')
        while (w % squarifyRatio) != 0:
            w += 1
    return doStack,squarifyRatio,w,h

def calc_squarify(img,cuts):
    h,w,c = img.shape
    q_increment = int(np.floor(w / cuts))
    return q_increment,w,h

def squarify(imgSquarify,showImg,makeSquare,sz):
    imgSquarify = make_img_hor(imgSquarify)
    doStack,squarifyRatio,w_plus,h = calc_squarify_ratio(imgSquarify)

    if doStack:
        imgBG = create_white_bg(imgSquarify,squarifyRatio,h,w_plus)
        imgSquarify = stack_image(imgBG,squarifyRatio,h,w_plus,showImg)

    if makeSquare:
        dim = (sz, sz)
        imgSquarify = cv2.resize(imgSquarify, dim, interpolation = cv2.INTER_AREA)

    return imgSquarify

def squarify_rotate45(imgSquarify, showImg, makeSquare, sz, doFlip):
    imgSquarify = make_img_hor(imgSquarify)
    
    # if doFlip:
    #     imgSquarify = cv2.rotate(imgSquarify,cv2.ROTATE_180) 

    q_increment,w,h = calc_squarify(imgSquarify,4)

    # imgBG = create_white_bg(imgSquarify, None, h*2, h*2)
    imgSquarify = stack_image_quartile_rotate45(imgSquarify, q_increment, h, w, showImg)

    if makeSquare:
        dim = (sz, sz)
        imgSquarify = cv2.resize(imgSquarify, dim, interpolation = cv2.INTER_AREA)
    return imgSquarify

def squarify_quartiles(imgSquarify, showImg, makeSquare, sz, doFlip):
    imgSquarify = make_img_hor(imgSquarify)
    
    if doFlip:
        imgSquarify = cv2.rotate(imgSquarify,cv2.ROTATE_180) 

    q_increment,w,h = calc_squarify(imgSquarify,4)

    # imgBG = create_white_bg(imgSquarify, None, h*2, h*2)
    imgSquarify = stack_image_quartile(imgSquarify, q_increment, h, w, showImg)

    if makeSquare:
        dim = (sz, sz)
        imgSquarify = cv2.resize(imgSquarify, dim, interpolation = cv2.INTER_AREA)
    return imgSquarify

def squarify_nine(imgSquarify, showImg, makeSquare, sz):
    imgSquarify = make_img_hor(imgSquarify)

    q_increment,w,h = calc_squarify(imgSquarify,9)

    # imgBG = create_white_bg(imgSquarify, None, h*2, h*2)
    imgSquarify = stack_image_nine(imgSquarify, q_increment, h, w, showImg)

    if makeSquare:
        dim = (sz, sz)
        imgSquarify = cv2.resize(imgSquarify, dim, interpolation = cv2.INTER_AREA)
    return imgSquarify

def squarify_tile_four_versions(imgSquarify, showImg, makeSquare, sz):
    h = int(sz*2)
    w = int(sz*2)
    h2 = int(h/2)
    w2 = int(w/2)
    sq1 = squarify(imgSquarify,showImg,makeSquare,sz)
    sq2 = squarify_rotate45(imgSquarify, showImg, makeSquare, sz, doFlip=False)
    sq3 = squarify_quartiles(imgSquarify, showImg, makeSquare, sz, doFlip=True)
    sq4 = squarify_nine(imgSquarify, showImg, makeSquare, sz)


    imgBG = np.zeros([h,w,3], dtype=np.uint8)
    imgBG[:] = 255

    imgBG[0:h2, 0:h2 ,:] = sq1
    imgBG[:h2, h2:w ,:] = sq2
    imgBG[h2:w, :h2 ,:] = sq3
    imgBG[h2:w, h2:w ,:] = sq4

    if showImg:
        cv2.imshow('Four versions: squarify(), squarify_quartiles(), squarify_quartiles(rotate180), squarify_nine()', imgBG)
        cv2.waitKey(0)

    return imgBG


# def remove_text(imgBi):
#     # reader = easyocr.Reader(['en'])
#     # result = reader.readtext(imgBi,paragraph=False)
#     # df=pd.DataFrame(result)
#     # print(df[1])
#     cv2.imshow("imgBi", imgBi)
#     cv2.waitKey(0)
#     imgBi = cv2.cvtColor(imgBi, cv2.COLOR_BGR2RGB)
#     imgBi = cv2.rotate(imgBi,cv2.ROTATE_180)
#     imgBi= cv2.bitwise_not(imgBi)
#     print(pytesseract.image_to_boxes(imgBi))
#     print(pytesseract.image_to_data(imgBi))
#     text = pytesseract.image_to_string(imgBi)
#     print(text)
#     cv2.imshow("imgBi", imgBi)
#     cv2.waitKey(0)
#     results = pytesseract.image_to_data(imgBi)

'''
####################################
####################################
            Process
####################################
####################################
'''
def straighten_img(RulerCFG, Ruler, useRegulerBinary, alternate_img, Dirs):
    # Ruler = RulerImage(img_path)

    # Ruler.ruler_class,Ruler.ruler_class_percentage,Ruler.img_type_overlay = detectRuler(Ruler.img,
    #         Info.cfg['leafmachine']['print']['ruler_type'],
    #         Info.cfg['leafmachine']['save']['ruler_type_overlay'],
    #         os.path.join(Info.dir_ruler_correction,'ruler_type_overlay'),
    #         Info.path_to_model,
    #         Info.cfg['leafmachine']['model']['ruler_detector'],
    #         Info.path_to_label_names)
    if useRegulerBinary:
        ruler_to_correct = Ruler.img_bi
    else:
        ruler_to_correct = np.uint8(alternate_img) # BlockCandidate.remaining_blobs[0].values

    # if checkRulerType(Ruler.ruler_class,'gray'):
    #     # For gray or tiffen rulers: use --> thresh, img_bi = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    #     Ruler.img_bi = cv2.adaptiveThreshold(Ruler.img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,19)#7,2)
    #     # thresh, img_bi = cv2.threshold(gray, 120, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    # #     cv2.imshow("img_bi", img_bi)
    # #     cv2.waitKey(0)
    # elif checkRulerType(Ruler.ruler_class,'grid'):
    #     # kernel = np.ones((3,3),np.uint8)
    #     Ruler.img_bi = cv2.adaptiveThreshold(Ruler.img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,51,9)
    # elif checkRulerType(Ruler.ruler_class,'tick_black'):
    #     Ruler.img_bi = cv2.adaptiveThreshold(Ruler.img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,9)
    # else:
    #     thresh, Ruler.img_bi = cv2.threshold(Ruler.img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


    # Ruler.img_bi_display = np.array(Ruler.img_bi)
    # Ruler.img_bi_display = np.stack((Ruler.img_bi_display,)*3, axis=-1)

    image_rotated, angle = rotate_bi_image_hor(ruler_to_correct)
    
    '''old method of rotating
    # find edges
    Ruler.img_edges = cv2.Canny(ruler_to_correct, 100, 200, apertureSize=7)

    # Detect lines using hough transform
    cdst = cv2.cvtColor(Ruler.img_edges, cv2.COLOR_GRAY2BGR)

    angles = []
    lines = cv2.HoughLines(Ruler.img_edges, 1, np.pi / 180, 150, 100, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

            deltaY = pt2[1] - pt1[1]
            deltaX = pt2[0] - pt1[0]
            angleInDegrees = math.atan2(deltaY, deltaX) * 180 / np.pi
            angles.append(angleInDegrees)
            # print(angleInDegrees)
            # cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    '''
    if angle > 1: # If the rotation was substantial
        Ruler.correction_success = True
        Ruler.avg_angle = angle
    else:
        Ruler.correction_success = False
        Ruler.avg_angle = 0
        # Ruler.img_best = Ruler.img
        # Ruler.img_total_overlay = newImg

    ''' exception for grid rulers, revisit
    # Grid rulers will NOT get roatate, assumption is that they are basically straight already
    if check_ruler_type(Ruler.ruler_class,'grid') == False:
        if len(angles) > 0:
            Ruler.avg_angle = np.mean(angles)
            imgRotate = ndimage.rotate(Ruler.img,Ruler.avg_angle)
            imgRotate = make_img_hor(imgRotate)
        else:
            Ruler.avg_angle = 0
            imgRotate = Ruler.img
    else: 
        Ruler.avg_angle = 0
        imgRotate = Ruler.img
    '''
    newImg = stack_2_imgs(Ruler.img,Ruler.img_bi_display)
    # newImg = stack_2_imgs(newImg,cdst)
    newImg = stack_2_imgs(newImg,image_rotated)
    newImg = create_overlay_bg(RulerCFG,newImg)
    newImg = add_text_to_stacked_img(Ruler.avg_angle,newImg)
    if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_class_overlay']:
        newImg = stack_2_imgs(Ruler.img_type_overlay,newImg)

    Ruler.img_best = image_rotated
    Ruler.img_total_overlay = newImg

    if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_overlay']:
        cv2.imwrite(os.path.join(Dirs.ruler_overlay,'.'.join([Ruler.img_fname, 'jpg'])),Ruler.img_total_overlay)
    if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_processed']:
        cv2.imwrite(os.path.join(Dirs.ruler_processed,'.'.join([Ruler.img_fname, 'jpg'])),Ruler.img_best)
            # if cfg['leafmachine']['save']['ruler_type_overlay']:
            #     cv2.imwrite(os.path.join(dirSave,'ruler_type_overlay',fName),Ruler.img_type_overlay)
    # else: #len(angles) > 0: == False
    #     Ruler.avg_angle = 0
    #     if Info.cfg['leafmachine']['save']['ruler_type_overlay']:
    #         newImg = stack2Images(Ruler.img_type_overlay,Ruler.img_bi_display)
    #     else:
    #         newImg = stack2Images(Ruler.img,Ruler.img_bi_display)
    #     Ruler.img_best = Ruler.img
    #     Ruler.img_total_overlay = newImg
    #     if Info.cfg['leafmachine']['print']['ruler_correction']:
    #         print(f'Angle: FAIL')
    #     if Info.cfg['leafmachine']['save']['ruler_correction_compare']:
    #         cv2.imwrite(os.path.join(Info.path_ruler_output_parent,'ruler_correction_compare',img_fname),newImg)
    #     if Info.cfg['leafmachine']['save']['ruler_correction']:
    #         cv2.imwrite(os.path.join(Info.path_ruler_output_parent,'ruler_correction','No-Angle-Correction__'+img_fname),Ruler.img)
        # if cfg['leafmachine']['save']['ruler_type_overlay']:
            # cv2.imwrite(os.path.join(dirSave,'ruler_type_overlay',fName),Ruler.img_type_overlay)

    # After saving the edges and imgBi to the compare file, flip for the class
    Ruler.img_bi = ndimage.rotate(Ruler.img_bi,Ruler.avg_angle)
    Ruler.img_bi = make_img_hor(Ruler.img_bi)
    # Ruler.img_edges = ndimage.rotate(Ruler.img_edges,Ruler.avg_angle)
    # Ruler.img_edges = make_img_hor(Ruler.img_edges)
    Ruler.img_gray = ndimage.rotate(Ruler.img_gray,Ruler.avg_angle)
    Ruler.img_gray = make_img_hor(Ruler.img_gray)
    return Ruler

def rotate_bi_image_hor(binary_img):
    cv2.imshow('im',binary_img)
    cv2.waitKey(0)
    lines = cv2.HoughLinesP(binary_img, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    angle = 0.0
    if lines.size > 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle += np.arctan2(y2 - y1, x2 - x1)
        angle /= len(lines)
        (h, w) = binary_img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, np.degrees(angle), 1.0)
        rotated_img = cv2.warpAffine(binary_img, M, (w, h), flags=cv2.INTER_NEAREST)
    else:
        rotated_img = binary_img
    cv2.imshow('im',rotated_img)
    cv2.waitKey(0)
    return rotated_img, angle

def locate_ticks_centroid(chunkAdd,scanSize):
    props = regionprops_table(label(chunkAdd), properties=('centroid',
                                            'orientation',
                                            'axis_major_length',
                                            'axis_minor_length'))
    props = pd.DataFrame(props)
    centoid = props['centroid-1']
    peak_pos = np.transpose(np.array(centoid))
    dst_matrix = peak_pos - peak_pos[:, None]
    dst_matrix = dst_matrix[~np.eye(dst_matrix.shape[0],dtype=bool)].reshape(dst_matrix.shape[0],-1)
    dist = np.min(np.abs(dst_matrix), axis=1)
    distUse = dist[dist > 2]

    distUse = remove_outliers(distUse)
    
    plotPtsX = peak_pos[dist > 2]
    plotPtsY = np.repeat(round(scanSize/2),plotPtsX.size)
    npts = len(plotPtsY)
    return plotPtsX,plotPtsY,distUse,npts

def remove_outliers(dist):
    threshold = 2
    z = np.abs(stats.zscore(dist))
    dist = dist[np.where(z < threshold)]
    threshold = 1
    z = np.abs(stats.zscore(dist))
    dist = dist[np.where(z < threshold)]
    threshold = 1
    z = np.abs(stats.zscore(dist))
    distUse = dist[np.where(z < threshold)]
    return distUse

def locate_tick_peaks(chunk,scanSize,x):
    chunkAdd = [sum(x) for x in zip(*chunk)]
    if scanSize >= 12:
        peaks = find_peaks(chunkAdd,distance=6,height=6)
    elif ((scanSize >= 6)&(scanSize < 12)):
        peaks = find_peaks(chunkAdd,distance=4,height=4)
    else:
        peaks = find_peaks(chunkAdd,distance=3,height=3)
    peak_pos = x[peaks[0]]
    peak_pos = np.array(peak_pos)
    dst_matrix = peak_pos - peak_pos[:, None]
    dst_matrix = dst_matrix[~np.eye(dst_matrix.shape[0],dtype=bool)].reshape(dst_matrix.shape[0],-1)
    dist = np.min(np.abs(dst_matrix), axis=1)
    distUse = dist[dist > 2]

    distUse = remove_outliers(distUse)

    plotPtsX = peak_pos[dist > 2]
    plotPtsY = np.repeat(round(scanSize/2),plotPtsX.size)
    npts = len(plotPtsY)
    # print(x[peaks[0]])
    # print(peaks[1]['peak_heights'])
    # plt.plot(x,chunkAdd)
    # plt.plot(x[peaks[0]],peaks[1]['peak_heights'], "x")
    # plt.show()
    return plotPtsX,plotPtsY,distUse,npts

def scanlines(RulerCFG,img,scanSize):
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    img[img<=200] = 0
    img[img>200] = 1
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    # img = cv2.dilate(img,kernel = np.ones((5,5),np.uint8))
    h,w = img.shape
    n = h % (scanSize *2)
    img_pad = pad_binary_img(img,h,w,n)
    img_pad_double = img_pad
    h,w = img_pad.shape
    x = np.linspace(0, w, w)
    
    scanlineData = {'index':[],'scanSize':[],'imgChunk':[],'plotPtsX':[],'plotPtsY':[],'plotPtsYoverall':[],'dists':[],'sd':[],'nPeaks':[],'normalizedSD':1000,'gmean':[],'mean':[]}    
    for i in range(0,int(h/scanSize)):
        chunkAdd = img_pad[scanSize*i:(scanSize*i+scanSize),:]
        # chunkAdd_open = cv2.morphologyEx(chunkAdd, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        # chunkAdd = cv2.morphologyEx(chunkAdd, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        # chunkAdd = cv2.dilate(chunkAdd,np.ones((3,3),np.uint8),iterations = 1)
        # cv2.imshow("img",np.stack((np.array(chunkAdd),)*3, axis=-1))
        # cv2.waitKey(0)
        try:
            plotPtsX,plotPtsY,distUse,npts = locate_ticks_centroid(chunkAdd,scanSize)
            # plotPtsX,plotPtsY,distUse,npts = locate_tick_peaks(chunkAdd,scanSize,x)
            
            if (distUse.shape[0] >=2) and (npts > 3):
                if (np.std(distUse)/npts < scanlineData['normalizedSD']) and (np.std(distUse)/npts > 0):
                    chunkAdd[chunkAdd >= 1] = 255
                    scanlineData['imgChunk']=chunkAdd
                    scanlineData['plotPtsX']=plotPtsX
                    scanlineData['plotPtsY']=plotPtsY
                    scanlineData['plotPtsYoverall']=(scanSize*i+scanSize)-round(scanSize/2)
                    scanlineData['dists']=distUse
                    scanlineData['sd']=np.std(distUse)
                    scanlineData['nPeaks']=(npts)
                    scanlineData['normalizedSD']=(np.std(distUse)/(npts))
                    scanlineData['gmean']=(gmean(distUse))
                    scanlineData['mean']=(np.mean(distUse))
                    scanlineData['index']=(int(i))
                    scanlineData['scanSize']=(int(scanSize))

                    print_sd = scanlineData.get("sd")
                    print_npts = scanlineData.get("nPeaks")
                    print_distUse = scanlineData.get("gmean")
                    message = "gmean dist: " + str(print_distUse)
                    Print_Verbose(RulerCFG.cfg,2,message).plain()
                    message = "sd/n: " + str(print_sd/print_npts)
                    Print_Verbose(RulerCFG.cfg,2,message).plain()
                    # print(f'gmean: {gmean(distUse)}')
                    # print(f'mean: {np.mean(distUse)}')
                    # print(f'npts: {npts}')
                    # print(f'sd: {np.std(distUse)}')
                    # print(f'sd/n: {(np.std(distUse)/(npts))}\n')
        except Exception as e:
            message = "Notice: Scanline size " + str(scanSize) + " iteration " + str(i) + " skipped:" 
            Print_Verbose(RulerCFG.cfg,2,message,e.args[0]).warning()
            continue
        

    scanSize = scanSize * 2
    for j in range(0,int((h/scanSize))):
        try:
            chunkAdd = img_pad_double[scanSize*j:(scanSize*j+scanSize),:]
            # chunkAdd_open = cv2.morphologyEx(chunkAdd, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
            # chunkAdd = cv2.morphologyEx(chunkAdd, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
            # chunkAdd = cv2.dilate(chunkAdd,np.ones((3,3),np.uint8),iterations = 1)

            plotPtsX,plotPtsY,distUse,npts = locate_ticks_centroid(chunkAdd,scanSize)
            # plotPtsX,plotPtsY,distUse,npts = locate_tick_peaks(chunkAdd,scanSize,x)

            if (distUse.shape[0] >=2) and (npts > 3):
                if (np.std(distUse)/npts < scanlineData['normalizedSD']) and (np.std(distUse)/npts > 0):
                    chunkAdd[chunkAdd > 1] = 255
                    scanlineData['imgChunk']=chunkAdd
                    scanlineData['plotPtsX']=plotPtsX
                    scanlineData['plotPtsY']=plotPtsY
                    scanlineData['plotPtsYoverall']=(scanSize*i+scanSize)-round(scanSize/2)
                    scanlineData['dists']=distUse
                    scanlineData['sd']=np.std(distUse)
                    scanlineData['nPeaks']=(npts)
                    scanlineData['normalizedSD']=(np.std(distUse)/(npts))
                    scanlineData['gmean']=(gmean(distUse))
                    scanlineData['mean']=(np.mean(distUse))
                    scanlineData['index']=(int(j))
                    scanlineData['scanSize']=(int(scanSize))

                    print_sd = scanlineData.get("sd")
                    print_npts = scanlineData.get("nPeaks")
                    print_distUse = scanlineData.get("gmean")
                    message = "gmean dist: " + str(print_distUse)
                    Print_Verbose(RulerCFG.cfg,2,message).plain()
                    message = "sd/n: " + str(print_sd/print_npts)
                    Print_Verbose(RulerCFG.cfg,2,message).plain()
                    # print(f'gmean: {gmean(distUse)}')
                    # print(f'mean: {np.mean(distUse)}')
                    # print(f'npts: {npts}')
                    # print(f'sd: {np.std(distUse)}')
                    # print(f'sd/n: {(np.std(distUse)/(npts))}\n')
        except Exception as e: 
            message = "Notice: Scanline size " + str(scanSize) + " iteration " + str(j) + " skipped:" 
            Print_Verbose(RulerCFG.cfg,2,message,e.args[0]).warning()
            continue
        # print(f'gmean: {gmean(distUse)}')
        # print(f'sd/n: {(np.std(distUse)/npts)}')
        # plt.imshow(chunkAdd)
        # plt.scatter(plotPtsX, plotPtsY,c='r', s=1)
        # plt.show()
    print_sd = scanlineData.get("sd")
    print_npts = scanlineData.get("nPeaks")
    print_distUse = scanlineData.get("gmean")

    try:
        message = "Best ==> distance-gmean: " + str(print_distUse)
        Print_Verbose(RulerCFG.cfg,1,message).blue()
        message = "Best ==> sd/n: " + str(print_sd/print_npts)
        Print_Verbose(RulerCFG.cfg,1,message).blue() 

    except Exception as e: 
        Print_Verbose(RulerCFG.cfg,2,'Pixel to Metric Conversion not possible. Exception: ',e.args[0]).warning()
        pass
    return scanlineData

def calculate_block_conversion_factor(BlockCandidate,nBlockCheck):
    factors = {'bigCM':0,'smallCM':0,'halfCM':0,'mm':0}
    n = {'bigCM':0,'smallCM':0,'halfCM':0,'mm':0}
    passFilter = {'bigCM':False,'smallCM':False,'halfCM':False,'mm':False}
    factors_fallback = {'bigCM':0,'smallCM':0,'halfCM':0,'mm':0}

    for i in range(0,nBlockCheck):
        if BlockCandidate.use_points[i]:
            X = BlockCandidate.x_points[i].values
            n_measurements = X.size
            axis_major_length = np.mean(BlockCandidate.axis_major_length[i].values)
            axis_minor_length = np.mean(BlockCandidate.axis_minor_length[i].values)
            dst_matrix = X - X[:, None]
            dst_matrix = dst_matrix[~np.eye(dst_matrix.shape[0],dtype=bool)].reshape(dst_matrix.shape[0],-1)
            dist = np.min(np.abs(dst_matrix), axis=1)
            distUse = dist[dist > 1]

            # Convert everything to CM along the way
            # 'if factors['bigCM'] == 0:' is there to make sure that there are no carry-over values if there were 
            # 2 instances of 'bigCM' coming from determineBlockBlobType()
            if distUse.size > 0:
                distUse_mean = np.mean(distUse)
                if BlockCandidate.point_types[i] == 'bigCM':
                    if ((distUse_mean >= 0.8*axis_major_length) & (distUse_mean <= 1.2*axis_major_length)):
                        if factors['bigCM'] == 0:
                            factors['bigCM'] = distUse_mean
                            n['bigCM'] = n_measurements
                            passFilter['bigCM'] = True
                        else:
                            break
                    else: 
                        factors_fallback['bigCM'] = distUse_mean

                elif BlockCandidate.point_types[i] == 'smallCM':
                    if ((distUse_mean >= 0.8*axis_major_length*2) & (distUse_mean <= 1.2*axis_major_length*2)):
                        if factors['smallCM'] ==0:
                            factors['smallCM'] = distUse_mean/2
                            n['smallCM'] = n_measurements
                            passFilter['bigCM'] = True
                        else:
                            break
                    else: 
                        factors_fallback['smallCM'] = distUse_mean/2

                elif BlockCandidate.point_types[i] == 'halfCM':
                    if ((distUse_mean >= 0.8*axis_major_length) & (distUse_mean <= 1.2*axis_major_length)):
                        if factors['halfCM'] ==0:
                            factors['halfCM'] = distUse_mean*2
                            n['halfCM'] = n_measurements
                            passFilter['bigCM'] = True
                        else:
                            break
                    else: 
                        factors_fallback['halfCM'] = distUse_mean*2

                elif BlockCandidate.point_types[i] == 'mm':
                    if ((distUse_mean >= 0.1*axis_minor_length) & (distUse_mean <= 1.1*axis_minor_length)):
                        if factors['mm'] ==0:
                            factors['mm'] = distUse_mean*10
                            n['mm'] = n_measurements
                            passFilter['bigCM'] = True
                        else:
                            break
                    else: 
                        factors['mm'] = 0
                        factors_fallback['mm'] = distUse_mean*10
    # Remove empty keys from n dict
    n_max = max(n, key=n.get)
    best_factor = factors[n_max]
    n_greater = len([f for f, factor in factors.items() if factor > best_factor])
    n_lesser = len([f for f, factor in factors.items() if factor < best_factor])
    location_options = ', '.join([f for f, factor in factors.items() if factor > 0])

    # If the factor with the higest number of measurements is the outlier, take the average of all factors
    if ((n_greater == 0) | (n_lesser == 0)):
        # Number of keys that = 0
        nZero = sum(x == 0 for x in factors.values())
        dividend = len(factors) - nZero
        # If no blocks pass the filter, return the nMax with a warning 
        if dividend == 0:
            best_factor_fallback = factors_fallback[n_max]
            n_greater = len([f for f, factor in factors_fallback.items() if factor > best_factor_fallback])
            n_lesser = len([f for f, factor in factors_fallback.items() if factor < best_factor_fallback])
            location_options = ', '.join([f for f, factor in factors_fallback.items() if factor > 0])
            if best_factor_fallback > 0:
                BlockCandidate.conversion_factor = best_factor_fallback
                BlockCandidate.conversion_location = 'fallback'
                BlockCandidate.conversion_factor_pass = passFilter[n_max]
            # Else complete fail
            else: 
                BlockCandidate.conversion_factor = 0
                BlockCandidate.conversion_location = 'fail'
                BlockCandidate.conversion_factor_pass = False
        else:
            res = sum(factors.values()) / dividend
            BlockCandidate.conversion_factor = res
            BlockCandidate.conversion_location = 'average'
            BlockCandidate.conversion_factor_pass = True
    # Otherwise use the factor with the most measuements 
    else:
        BlockCandidate.conversion_factor = best_factor
        BlockCandidate.conversion_location = n_max
        BlockCandidate.conversion_factor_pass = passFilter[n_max]
    BlockCandidate.conversion_location_options = location_options
    return BlockCandidate

def sort_blobs_by_size(RulerCFG, Ruler, isStraighten):
    nBlockCheck = 4
    success = True
    tryErode = False
    if isStraighten == False:
        # img_best = Ruler.img_best # was causseing issues
        img_best = Ruler.img_copy
    else:
        img_best = Ruler.img_copy
    BlockCandidate = Block(img_bi=Ruler.img_bi,img_bi_overlay=img_best)
    try: # Start with 4, reduce by one if fail
        # try: # Normal
        BlockCandidate = remove_small_and_biggest_blobs(BlockCandidate,tryErode)
        for i in range(0,nBlockCheck):
            BlockCandidate = get_biggest_blob(BlockCandidate)
        # except: # Extreme thresholding for whiter rulers
        #     # BlockCandidate.whiter_thresh()
        #     BlockCandidate.img_result = BlockCandidate.img_bi_copy
        #     BlockCandidate = removeSmallAndBiggestBlobs(BlockCandidate,tryErode)
        #     for i in range(0,nBlockCheck):
        #         BlockCandidate = getBiggestBlob(BlockCandidate)
    except:
        try:
            tryErode = True
            del BlockCandidate
            nBlockCheck = 3
            BlockCandidate = Block(img_bi=Ruler.img_bi,img_bi_overlay=img_best)
            BlockCandidate = remove_small_and_biggest_blobs(BlockCandidate,tryErode)
            for i in range(0,nBlockCheck):
                BlockCandidate = get_biggest_blob(BlockCandidate)
        except:
            success = False
            BlockCandidate = Block(img_bi=Ruler.img_bi,img_bi_overlay=img_best)
            BlockCandidate.conversion_factor = 0
            BlockCandidate.conversion_location = 'unidentifiable'
            BlockCandidate.conversion_location_options = 'unidentifiable'
            BlockCandidate.success_sort = success
            BlockCandidate.img_bi_overlay = Ruler.img_bi

    if success:
        # imgPlot = plt.imshow(img_result)
        for i in range(0,nBlockCheck):
            BlockCandidate = determine_block_blob_type(RulerCFG,BlockCandidate,i)#BlockCandidate.largest_blobs[0],BlockCandidate.img_bi_overlay)
        if isStraighten == False:
            Ruler.img_block_overlay = BlockCandidate.img_bi_overlay

        BlockCandidate = calculate_block_conversion_factor(BlockCandidate,nBlockCheck)  
    BlockCandidate.success_sort = success
    return Ruler, BlockCandidate

def convert_ticks(RulerCFG,Ruler,colorOption,img_fname):
    scanSize = 5
    if colorOption == 'black':
        Ruler.img_bi = cv2.bitwise_not(Ruler.img_bi)
    scanlineData = scanlines(RulerCFG,Ruler.img_bi,scanSize)
    Ruler = insert_scanline(RulerCFG,Ruler,scanlineData['imgChunk'],scanlineData['index'],scanlineData['scanSize'],scanlineData['plotPtsX'],scanlineData['plotPtsY'],scanlineData['mean'])
    Ruler.img_ruler_overlay = create_overlay_bg(RulerCFG,Ruler.img_ruler_overlay)

    if scanlineData['gmean'] != []:
        Ruler.img_ruler_overlay = add_text_to_img('GeoMean Pixel Dist Between Pts: '+str(round(scanlineData['gmean'],2)),Ruler.img_ruler_overlay)
    else:
        Ruler.img_ruler_overlay = add_text_to_img('GeoMean Pixel Dist Between Pts: No points found',Ruler.img_ruler_overlay)

    Ruler.img_total_overlay = stack_2_imgs(Ruler.img_total_overlay,Ruler.img_ruler_overlay)

    if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_overlay']:
        cv2.imwrite(os.path.join(RulerCFG.path_ruler_output_parent,RulerCFG.dir_ruler_overlay,img_fname),Ruler.img_total_overlay)
    # createOverlayBG(scanlineData['imgChunk'])
    # stack2Images(img1,img2)
    # addTextToImg(text,img)

def convert_blocks(RulerCFG,Ruler,colorOption,img_fname, Dirs):
    if colorOption == 'invert':
        Ruler.img_bi = cv2.bitwise_not(Ruler.img_bi)
    
    # Straighten the image here using the BlockCandidate.remaining_blobs[0].values
    Ruler,BlockCandidate = sort_blobs_by_size(RulerCFG, Ruler,isStraighten=True) 
    if BlockCandidate.success_sort:
        useRegulerBinary = True
        Ruler = straighten_img(RulerCFG, Ruler, useRegulerBinary, BlockCandidate.remaining_blobs[0], Dirs)
        del BlockCandidate
        Ruler,BlockCandidate = sort_blobs_by_size(RulerCFG,Ruler,isStraighten=False) 

    
        if BlockCandidate.success_sort: # if this is false, then no marks could be ID'd, will print just the existing Ruler.img_total_overlay
            if BlockCandidate.conversion_location != 'fail':
                BlockCandidate = add_unit_marker_block(BlockCandidate,1)
                BlockCandidate = add_unit_marker_block(BlockCandidate,10)

    message = "Angle (deg): " + str(round(Ruler.avg_angle,2))
    Print_Verbose(RulerCFG.cfg,1,message).cyan()

    BlockCandidate.img_bi_overlay = create_overlay_bg(RulerCFG,BlockCandidate.img_bi_overlay)
    if BlockCandidate.conversion_location in ['average','fallback']:
        addText = 'Used: '+BlockCandidate.conversion_location_options+' Factor 1cm: '+str(round(BlockCandidate.conversion_factor,2))
    elif BlockCandidate.conversion_location == 'fail':
        addText = 'Used: '+'FAILED'+' Factor 1cm: '+str(round(BlockCandidate.conversion_factor,2))
    elif BlockCandidate.conversion_location == 'unidentifiable':
        addText = 'UNIDENTIFIABLE'+' Factor 1cm: '+str(round(BlockCandidate.conversion_factor))
    else:
        addText = 'Used: '+BlockCandidate.conversion_location+' Factor 1cm: '+ str(round(BlockCandidate.conversion_factor,2))

    BlockCandidate.img_bi_overlay = add_text_to_img(addText,BlockCandidate.img_bi_overlay)#+str(round(scanlineData['gmean'],2)),Ruler.img_block_overlay)
    try:
        Ruler.img_total_overlay = stack_2_imgs(Ruler.img_total_overlay,BlockCandidate.img_bi_overlay)
    except:
        Ruler.img_total_overlay = stack_2_imgs(Ruler.img_type_overlay,BlockCandidate.img_bi_overlay)
    Ruler.img_block_overlay = BlockCandidate.img_bi_overlay

    if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_overlay']:
        cv2.imwrite(os.path.join(RulerCFG.dir_ruler_overlay,'.'.join([img_fname, 'jpg'])),Ruler.img_total_overlay)

    return Ruler, BlockCandidate


def add_unit_marker(RulerCFG,imgBG,scanSize,dist,X,factor,index):
    X.sort()
    try:
        # Get fist point
        start_f = int(X[5])
        end_f = int(start_f+(dist*factor)) + 1
        # Get middle point
        start_m = int(X[int(X.size/2)])
        end_m = int(start_m+(dist*factor)) + 1
        # get end point
        start_l = int(X[-15])
        end_l = int(start_l+(dist*factor)) + 1

        start = [start_f,start_m,start_l]
        end = [end_f,end_m,end_l]
    except Exception as e:
        Print_Verbose(RulerCFG.cfg,2,'add_unit_marker(): plotting 1 of 3 unit markers. Exception: ',e.args[0]).warning()
        # Get middle point
        start_m = int(X[int(X.size/2)])
        end_m = int(start_m+(dist*factor)) + 1
        start = [start_m]
        end = [end_m]

    for pos in range(0,len(start),1):
        for j in range(start[pos],end[pos],1):
            try:
                # 3 pixel thick line
                imgBG[int(4+(scanSize*index+scanSize)-(scanSize/2)-1),int(j),0] = 255
                imgBG[int(4+(scanSize*index+scanSize)-(scanSize/2)-1),int(j),1] = 0
                imgBG[int(4+(scanSize*index+scanSize)-(scanSize/2)-1),int(j),2] = 255
                imgBG[int(4+(scanSize*index+scanSize)-(scanSize/2)),int(j),0] = 255
                imgBG[int(4+(scanSize*index+scanSize)-(scanSize/2)),int(j),1] = 0
                imgBG[int(4+(scanSize*index+scanSize)-(scanSize/2)),int(j),2] = 255
                imgBG[int(4+(scanSize*index+scanSize)-(scanSize/2)+1),int(j),0] = 255
                imgBG[int(4+(scanSize*index+scanSize)-(scanSize/2)+1),int(j),1] = 0
                imgBG[int(4+(scanSize*index+scanSize)-(scanSize/2)+1),int(j),2] = 255
            except:
                continue
    return imgBG

def add_unit_marker_block(BlockCandidate, multiple):
    COLOR = {'10cm':[0,255,0],'cm':[255,0,255]}
    name = 'cm' if multiple == 1 else '10cm'
    offset = 4 if multiple == 1 else 14
    h, w, _ = BlockCandidate.img_bi_overlay.shape

    if BlockCandidate.conversion_location in ['average','fallback']:
        X = int(round(w/40))
        Y = int(round(h/10))
    else:
        ind = BlockCandidate.point_types.index(BlockCandidate.conversion_location)
        X = int(round(min(BlockCandidate.x_points[ind].values)))
        Y = int(round(np.mean(BlockCandidate.y_points[ind].values)))

    start = X
    end = int(round(start+(BlockCandidate.conversion_factor*multiple))) + 1
    if end >= w:
        X = int(round(w/40))
        Y = int(round(h/10))
        start = X
        end = int(round(start+(BlockCandidate.conversion_factor*multiple))) + 1

    plot_points = []
    for j in range(start, end):
        try:
            img_bi_overlay = BlockCandidate.img_bi_overlay
            img_bi_overlay[offset+Y-2:offset+Y+3, j, :] = 0
            img_bi_overlay[offset+Y-1:offset+Y+2, j, :] = COLOR[name]
            plot_points.append([j, offset+Y])
        except:
            continue

    BlockCandidate.img_bi_overlay = img_bi_overlay
    if multiple == 1:
        BlockCandidate.plot_points_1cm = plot_points
    else:
        BlockCandidate.plot_points_10cm = plot_points
    return BlockCandidate

# def add_unit_marker_block(BlockCandidate,multiple):
#     COLOR = {'10cm':[0,255,0],'cm':[255,0,255]}
#     if multiple == 1:
#         name = 'cm'
#         offest = 4
#     elif multiple == 10:
#         name = '10cm'
#         offest = 14

#     img_bi_overlay = BlockCandidate.img_bi_overlay
#     h,w,_ = img_bi_overlay.shape
#     if BlockCandidate.conversion_location in ['average','fallback']:
#         X = int(round(w/40))
#         Y = int(round(h/10))
#         # Get starting point
#         start = X
#         end = int(round(start+(BlockCandidate.conversion_factor*multiple))) + 1
#     else:
#         ind = BlockCandidate.point_types.index(BlockCandidate.conversion_location)
#         X = int(round(min(BlockCandidate.x_points[ind].values)))
#         Y = int(round(np.mean(BlockCandidate.y_points[ind].values)))
#         # Get starting point
#         start = X
#         end = int(round(start+(BlockCandidate.conversion_factor*multiple))) + 1
#     if end >= w:
#         X = int(round(w/40))
#         Y = int(round(h/10))
#         start = X
#         end = int(round(start+(BlockCandidate.conversion_factor*multiple))) + 1

#     for j in range(start,end,1):
#         try:
#             # 5 pixel thick line
#             img_bi_overlay[int(offest+Y-2),int(j),0] = 0#black
#             img_bi_overlay[int(offest+Y-2),int(j),1] = 0#black
#             img_bi_overlay[int(offest+Y-2),int(j),2] = 0#black
#             img_bi_overlay[int(offest+Y-1),int(j),0] = COLOR[name][0]
#             img_bi_overlay[int(offest+Y-1),int(j),1] = COLOR[name][1]
#             img_bi_overlay[int(offest+Y-1),int(j),2] = COLOR[name][2]
#             img_bi_overlay[int(offest+Y),int(j),0] = COLOR[name][0]
#             img_bi_overlay[int(offest+Y),int(j),1] = COLOR[name][1]
#             img_bi_overlay[int(offest+Y),int(j),2] = COLOR[name][2]
#             img_bi_overlay[int(offest+Y+1),int(j),0] = COLOR[name][0]
#             img_bi_overlay[int(offest+Y+1),int(j),1] = COLOR[name][1]
#             img_bi_overlay[int(offest+Y+1),int(j),2] = COLOR[name][2]
#             img_bi_overlay[int(offest+Y+2),int(j),0] = 0#black
#             img_bi_overlay[int(offest+Y+2),int(j),1] = 0#black
#             img_bi_overlay[int(offest+Y+2),int(j),2] = 0#black
#         except:
#             continue
#     BlockCandidate.img_bi_overlay = img_bi_overlay
#     return BlockCandidate

def insert_scanline(RulerCFG,Ruler,chunk,index,scanSize,X,Y,dist):
    imgBG = Ruler.img_best
    # imgBG[(scanSize*index):((scanSize*index)+scanSize),:] = np.stack((np.array(chunk),)*3, axis=-1)
    for i in range(-2,3):
        for j in range(-2,3):
            for x in X:
                try:
                    if (abs(i) == 2) | (abs(j) == 2):
                        imgBG[int((scanSize*index+scanSize)-(scanSize/2)+i),int(x+j),0] = 0
                        imgBG[int((scanSize*index+scanSize)-(scanSize/2)+i),int(x+j),1] = 0
                        imgBG[int((scanSize*index+scanSize)-(scanSize/2)+i),int(x+j),2] = 0
                    else:
                        imgBG[int((scanSize*index+scanSize)-(scanSize/2)+i),int(x+j),0] = 255
                        imgBG[int((scanSize*index+scanSize)-(scanSize/2)+i),int(x+j),1] = 0
                        imgBG[int((scanSize*index+scanSize)-(scanSize/2)+i),int(x+j),2] = 255
                except:
                    continue
    # print(Ruler.ruler_class)
    if check_ruler_type(Ruler.ruler_class,'AND'):
        '''
        ############################################
        Handle rulers with both metric and imperial
        ############################################
        '''
        imgBG = imgBG
    else:
        if len(X) > 0:
            if check_ruler_type(Ruler.ruler_class,'_16th'):
                factor = 16
                imgBG = add_unit_marker(RulerCFG,imgBG,scanSize,dist,X,factor,index)
            elif check_ruler_type(Ruler.ruler_class,'_8th'):
                factor = 8
                imgBG = add_unit_marker(RulerCFG,imgBG,scanSize,dist,X,factor,index)
            elif check_ruler_type(Ruler.ruler_class,'_halfcm'):
                factor = 20
                imgBG = add_unit_marker(RulerCFG,imgBG,scanSize,dist,X,factor,index)
            elif check_ruler_type(Ruler.ruler_class,'_4thcm'):
                factor = 4
                imgBG = add_unit_marker(RulerCFG,imgBG,scanSize,dist,X,factor,index)
            elif check_ruler_type(Ruler.ruler_class,'_halfmm'):
                factor = 20
                imgBG = add_unit_marker(RulerCFG,imgBG,scanSize,dist,X,factor,index)
            elif check_ruler_type(Ruler.ruler_class,'_halfcm'):
                factor = 2
                imgBG = add_unit_marker(RulerCFG,imgBG,scanSize,dist,X,factor,index)
            elif check_ruler_type(Ruler.ruler_class,'_mm'):
                factor = 10
                imgBG = add_unit_marker(RulerCFG,imgBG,scanSize,dist,X,factor,index)
        else:
            print(f"{bcolors.WARNING}     No tickmarks found{bcolors.ENDC}")
    Ruler.img_ruler_overlay = imgBG
    
    # cv2.imshow("img",imgBG)
    # cv2.waitKey(0)
    return Ruler

def get_biggest_blob(BlockCandidate):
    img_result = BlockCandidate.img_result
    # cv2.imshow('THIS img',BlockCandidate.img_result)
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(np.uint8(img_result))
    sizes = stats[:, -1]
    sizes = sizes[1:]
    maxBlobSize = max(sizes)
    largestBlobs = np.zeros((img_result.shape))
    remainingBlobs = np.zeros((img_result.shape))
    nb_blobs -= 1
    for blob in range(nb_blobs):
        if (sizes[blob] <= 1.1*maxBlobSize) & ((sizes[blob] >= 0.9*maxBlobSize)):
            # see description of im_with_separated_blobs above
            largestBlobs[im_with_separated_blobs == blob + 1] = 255
        else:
            remainingBlobs[im_with_separated_blobs == blob + 1] = 255
    BlockCandidate.largest_blobs.append(largestBlobs)
    BlockCandidate.remaining_blobs.append(remainingBlobs)
    BlockCandidate.img_result = remainingBlobs
    return BlockCandidate
    
def remove_small_and_biggest_blobs(BlockCandidate,tryErode):
    min_size = 50
    img_bi = BlockCandidate.img_bi
    # cv2.imshow('iimg',img_bi)
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(img_bi, cv2.MORPH_OPEN, kernel)
    if tryErode:
        opening = cv2.bitwise_not(opening)
        opening = cv2.erode(opening,kernel,iterations = 1)
        opening = cv2.dilate(opening,kernel,iterations = 1)
        min_size = 25
        BlockCandidate.img_bi = opening
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(opening)
    sizes = stats[:, -1]
    sizes = sizes[1:]
    maxBlobSize = max(sizes)
    nb_blobs -= 1
    img_result = np.zeros((img_bi.shape))
    # for every component in the image, keep it only if it's above min_size
    for blob in range(nb_blobs):
        if sizes[blob] == maxBlobSize:
            img_result[im_with_separated_blobs == blob + 1] = 0
        elif sizes[blob] >= min_size:
            # see description of im_with_separated_blobs above
            img_result[im_with_separated_blobs == blob + 1] = 255
    BlockCandidate.img_result = img_result
    return BlockCandidate

def add_centroid_to_block_img(imgBG, centroidX, centroidY, ptType):
    COLOR = {'bigCM': [0, 255, 0], 'smallCM': [255, 255, 0], 'halfCM': [0, 127, 255], 'mm': [255, 0, 127]}
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for x in range(0, centroidX.size):
                X = int(round(centroidX.values[x]))
                Y = int(round(centroidY.values[x]))
                if (abs(i) == 3) | (abs(j) == 3):
                    imgBG[int(Y+i), int(X+j), 0] = 0
                    imgBG[int(Y+i), int(X+j), 1] = 0
                    imgBG[int(Y+i), int(X+j), 2] = 0
                else:
                    imgBG[int(Y+i), int(X+j), 0] = COLOR[ptType][0]
                    imgBG[int(Y+i), int(X+j), 1] = COLOR[ptType][1]
                    imgBG[int(Y+i), int(X+j), 2] = COLOR[ptType][2]
                    points.append([j + X, Y + i])
    return imgBG, points

# def add_centroid_to_block_img(imgBG,centoidX,centoidY,ptType):
#     COLOR = {'bigCM':[0,255,0],'smallCM':[255,255,0],'halfCM':[0,127,255],'mm':[255,0,127]}
#     for i in range(-3,4):
#         for j in range(-3,4):
#             # print(centoidX.values)
#             for x in range(0,centoidX.size):
#                 # print(centoidX.values[x])
#                 X = int(round(centoidX.values[x]))
#                 Y = int(round(centoidY.values[x]))
#                 # try:
#                 if (abs(i) == 3) | (abs(j) == 3):
#                     imgBG[int(Y+i),int(X+j),0] = 0
#                     imgBG[int(Y+i),int(X+j),1] = 0
#                     imgBG[int(Y+i),int(X+j),2] = 0
#                 else:
#                     imgBG[int(Y+i),int(X+j),0] = COLOR[ptType][0]
#                     imgBG[int(Y+i),int(X+j),1] = COLOR[ptType][1]
#                     imgBG[int(Y+i),int(X+j),2] = COLOR[ptType][2]
#                 # except:
#                     # continue
#     return imgBG

def determine_block_blob_type(RulerCFG,BlockCandidate,ind):
    largestBlobs = BlockCandidate.largest_blobs[ind]
    img_bi_overlay = BlockCandidate.img_bi_overlay
    # img_bi_overlay = np.stack((img_bi,)*3, axis=-1)
    RATIOS = {'bigCM':1.75,'smallCM':4.5,'halfCM':2.2,'mm':6.8}
    use_points = False
    point_types = 'NA'
    points = []

    props = regionprops_table(label(largestBlobs), properties=('centroid','axis_major_length','axis_minor_length'))
    props = pd.DataFrame(props)
    centoidY = props['centroid-0']
    centoidX = props['centroid-1']
    axis_major_length = props['axis_major_length']
    axis_minor_length = props['axis_minor_length']
    ratio = axis_major_length/axis_minor_length
    if ((ratio.size > 1) & (ratio.size <= 10)):
        ratioM = np.mean(ratio)
        if ((ratioM >= (0.9*RATIOS['bigCM'])) & (ratioM <= (1.1*RATIOS['bigCM']))):
            use_points = True
            point_types = 'bigCM'
            img_bi_overlay, points = add_centroid_to_block_img(img_bi_overlay,centoidX,centoidY,point_types)
        elif ((ratioM >= (0.75*RATIOS['smallCM'])) & (ratioM <= (1.25*RATIOS['smallCM']))):
            use_points = True
            point_types = 'smallCM'
            img_bi_overlay, points = add_centroid_to_block_img(img_bi_overlay,centoidX,centoidY,point_types)
        elif ((ratioM >= (0.9*RATIOS['halfCM'])) & (ratioM <= (1.1*RATIOS['halfCM']))):
            use_points = True
            point_types = 'halfCM'
            img_bi_overlay, points = add_centroid_to_block_img(img_bi_overlay,centoidX,centoidY,point_types)
        elif ((ratioM >= (0.9*RATIOS['mm'])) & (ratioM <= (1.1*RATIOS['mm']))):
            use_points = True
            point_types = 'mm'
            img_bi_overlay, points = add_centroid_to_block_img(img_bi_overlay,centoidX,centoidY,point_types)
        message = ''.join(["ratio: ", str(round(ratioM,3)), " use_points: ", str(use_points), " point_types: ", str(point_types)])
        Print_Verbose(RulerCFG.cfg,2,message).plain()
    # plt.imshow(img_bi_overlay)
    BlockCandidate.img_bi_overlay = img_bi_overlay
    BlockCandidate.use_points.append(use_points)
    BlockCandidate.plot_points.append(points)
    BlockCandidate.point_types.append(point_types)
    BlockCandidate.x_points.append(centoidX)
    BlockCandidate.y_points.append(centoidY)
    BlockCandidate.axis_major_length.append(axis_major_length)
    BlockCandidate.axis_minor_length.append(axis_minor_length)
    return BlockCandidate


def convert_pixels_to_metric(RulerCFG, Ruler, img_fname, Dirs):#cfg,Ruler,imgPath,fName,dirSave,dir_ruler_correction,pathToModel,labelNames):

    if check_ruler_type(Ruler.ruler_class,'tick_black'):
        colorOption = 'black'
        Ruler = straighten_img(RulerCFG, Ruler, True, False, Dirs)
        Ruler, BlockCandidate = convert_ticks(RulerCFG, Ruler, colorOption, img_fname)
    elif check_ruler_type(Ruler.ruler_class,'tick_white'):
        colorOption = 'white'
        Ruler = straighten_img(RulerCFG, Ruler, True, False, Dirs)
        Ruler, BlockCandidate = convert_ticks(RulerCFG, Ruler,colorOption, img_fname)


    elif check_ruler_type(Ruler.ruler_class,'block_regular_cm'):
        colorOption = 'invert'
        Ruler, BlockCandidate = convert_blocks(RulerCFG, Ruler, colorOption, img_fname, Dirs)
        return Ruler, BlockCandidate
    elif check_ruler_type(Ruler.ruler_class,'block_invert_cm'):
        colorOption = 'noinvert'
        Ruler, BlockCandidate = convert_blocks(RulerCFG, Ruler, colorOption, img_fname, Dirs)
        return Ruler, BlockCandidate
'''
####################################
####################################
           Main Functions
####################################
####################################
'''
def setup_ruler(RulerCFG, img, img_fname):
    Ruler = RulerImage(img=img, img_fname=img_fname)

    print(f"{bcolors.BOLD}\nRuler: {img_fname}{bcolors.ENDC}")

    Ruler.ruler_class,Ruler.ruler_class_percentage,Ruler.img_type_overlay = detect_ruler(RulerCFG, img, img_fname)
    
    if check_ruler_type(Ruler.ruler_class,'gray'):
        # For gray or tiffen rulers: use --> thresh, img_bi = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        Ruler.img_bi = cv2.adaptiveThreshold(Ruler.img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,19)#7,2)  ### Last used
        # thresh, img_bi = cv2.threshold(gray, 120, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    #     cv2.imshow("img_bi", img_bi)
    #     cv2.waitKey(0)
    elif check_ruler_type(Ruler.ruler_class,'grid'):
        # kernel = np.ones((3,3),np.uint8)
        Ruler.img_bi = cv2.adaptiveThreshold(Ruler.img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,51,9) ### Last used
    elif check_ruler_type(Ruler.ruler_class,'tick_black'):

        Ruler.img_bi = find_minimal_change_in_binarization(Ruler.img_gray)
        # cv2.imshow("Dirty", Ruler.img_bi)
        # cv2.waitKey(0)
        '''
        Ruler.img_bi = cv2.adaptiveThreshold(Ruler.img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,9)

        ##### https://stackoverflow.com/questions/10316057/filling-holes-inside-a-binary-object
        # cv2.imshow("Dirty", Ruler.img_bi)
        # cv2.waitKey(0)
        #####
        contour,hier = cv2.findContours(Ruler.img_bi,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contour:
            cv2.drawContours(Ruler.img_bi,[cnt],0,255,-1)

        gray = cv2.bitwise_not(Ruler.img_bi)
        # cv2.imshow("Clean", Ruler.img_bi)
        # cv2.waitKey(0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        Ruler.img_bi = cv2.morphologyEx(Ruler.img_bi,cv2.MORPH_OPEN,kernel)
        # cv2.imshow("Clean2", Ruler.img_bi)
        # cv2.waitKey(0)
        '''
    else:
        # thresh, Ruler.img_bi = cv2.threshold(Ruler.img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        Ruler.img_bi = find_minimal_change_in_binarization(Ruler.img_gray)

    Ruler.img_bi_display = np.array(Ruler.img_bi)
    Ruler.img_bi_display = np.stack((Ruler.img_bi_display,)*3, axis=-1)
    

    return Ruler

def find_minimal_change_in_binarization(img_gray):
    result_list = []

    # fig, axs = plt.subplots(5, 5, figsize=(20, 20))
    # axs = axs.ravel()

    for idx, i in enumerate(range(0, 255, 10)):
        threshold_value = i
        img_bi = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)[1]
        result = cv2.countNonZero(img_bi)
        result_list.append((threshold_value, result))
        
        # axs[idx-1].imshow(img_bi, cmap='gray')
        # axs[idx-1].set_title(f"Threshold: {threshold_value}")

    x = [i[0] for i in result_list]
    y = [i[1] for i in result_list]

    diffs = [abs(y[i+5]-y[i]) for i in range(len(y)-5)]
    min_index = diffs.index(min(diffs))
    ''' Turn this and the commented lines above for testing
    plt.tight_layout()
    plt.show()
    fig.savefig('bi_panel.pdf')
    plt.close()

    x = [i[0] for i in result_list]
    y = [i[1] for i in result_list]

    diffs = [abs(y[i+5]-y[i]) for i in range(len(y)-5)]
    min_index = diffs.index(min(diffs))


    fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel("Threshold value")
    plt.ylabel("Result")
    plt.title("Result vs Threshold Value")
    fig.savefig("bi_plot.pdf")
    plt.close()
    dy = np.gradient(y)
    d2y = np.gradient(dy)

    fig = plt.figure()
    plt.plot(x, dy, label='Derivative')
    plt.plot(x, d2y, label='Second Derivative')
    plt.xlabel("Threshold value")
    plt.ylabel("Result")
    plt.title("Result vs Threshold Value")
    plt.legend()
    fig.savefig("bi_plot_derivative.pdf")
    plt.close()

    # find the median point of result_list where the change between results is the least
    # median_index = 0
    # min_diff = float('inf')
    # diff_list = []
    # for i in range(1, len(result_list) - 1):
    #     diff = abs(result_list[i + 1][1] - result_list[i - 1][1])
    #     diff_list.append(diff)
    #     if diff < min_diff:
    #         median_index = i
    #         min_diff = diff

    # x = [i[0] for i in result_list]
    y = [i[1] for i in result_list]

    # Calculate the first derivative
    dy = np.diff(y)

    # Calculate the second derivative
    # ddy = np.diff(dy)
    # min_index = np.argmin(dy)
    # min_index = np.argmin(ddy)
    '''
    # Find the index of the minimum value of the first derivative
    
    best_threshold = result_list[min_index][0]
    img_bi = cv2.threshold(img_gray, best_threshold, 255, cv2.THRESH_BINARY)[1]
    return img_bi


def detect_ruler(RulerCFG, ruler_cropped, ruler_name):
    net = RulerCFG.net_ruler
    
    img = ClassifyRulerImage(ruler_cropped)

    # net = torch.jit.load(os.path.join(modelPath,modelName))
    # net.eval()

    with open(os.path.abspath(RulerCFG.path_to_class_names)) as f:
        classes = [line.strip() for line in f.readlines()]


    out = net(img.img_tensor)
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

    _, index = torch.max(out, 1)
    percentage1 = torch.nn.functional.softmax(out, dim=1)[0] * 100
    percentage1 = round(percentage1[index[0]].item(),2)
    pred_class1 = classes[index[0]]

    if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_class_overlay']:
        imgBG = create_overlay_bg(RulerCFG,img.img_sq)
        addText1 = "Class: "+str(pred_class1)
        addText2 = "Certainty: "+str(percentage1)
        newName = '.'.join([ruler_name ,'jpg'])
        # newName = newName.split(".")[0] + "__overlay.jpg"
        imgOverlay = cv2.putText(img=imgBG, text=addText1, org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(155, 155, 155),thickness=1)
        imgOverlay = cv2.putText(img=imgOverlay, text=addText2, org=(10, 45), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(155, 155, 155),thickness=1)
        cv2.imwrite(os.path.join(RulerCFG.dir_ruler_class_overlay,newName),imgOverlay)

    message = "Class: " + str(pred_class1) + " Certainty: " + str(percentage1) + "%"
    Print_Verbose(RulerCFG.cfg,1,message).green()

    return pred_class1,percentage1,imgOverlay









@dataclass
class Print_Verbose():
    cfg: str = ''
    indent_level: int = 0
    message: str = ''

    def __init__(self, cfg, indent_level, message) -> None:
        self.cfg = cfg
        self.indent_level = indent_level
        self.message = message

    def bold(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['verbose']:
            print(f"{bcolors.BOLD}{white_space}{self.message}{bcolors.ENDC}")

    def green(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['verbose']:
            print(f"{bcolors.OKGREEN}{white_space}{self.message}{bcolors.ENDC}")

    def cyan(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['verbose']:
            print(f"{bcolors.OKCYAN}{white_space}{self.message}{bcolors.ENDC}")

    def blue(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['verbose']:
            print(f"{bcolors.OKBLUE}{white_space}{self.message}{bcolors.ENDC}")

    def warning(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['verbose']:
            print(f"{bcolors.WARNING}{white_space}{self.message}{bcolors.ENDC}")

    def plain(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['verbose']:
            print(f"{white_space}{self.message}")

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    CEND      = '\33[0m'
    CBOLD     = '\33[1m'
    CITALIC   = '\33[3m'
    CURL      = '\33[4m'
    CBLINK    = '\33[5m'
    CBLINK2   = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK  = '\33[30m'
    CRED    = '\33[31m'
    CGREEN  = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE   = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE  = '\33[36m'
    CWHITE  = '\33[37m'

    CBLACKBG  = '\33[40m'
    CREDBG    = '\33[41m'
    CGREENBG  = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG   = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG  = '\33[46m'
    CWHITEBG  = '\33[47m'

    CGREY    = '\33[90m'
    CRED2    = '\33[91m'
    CGREEN2  = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2   = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2  = '\33[96m'
    CWHITE2  = '\33[97m'

    CGREYBG    = '\33[100m'
    CREDBG2    = '\33[101m'
    CGREENBG2  = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2   = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2  = '\33[106m'
    CWHITEBG2  = '\33[107m'
    CBLUEBG3   = '\33[112m'