'''
venv requirements:
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip3 install cython
pip3 install opencv-python

git clone https://github.com/facebookresearch/detectron2.git

change:
IN: detectron2/detectron2/layers/csrc/nms_rotated/nms_rotated_cuda.cu
// Copyright (c) Facebook, Inc. and its affiliates.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
//#ifdef WITH_CUDA ***MODIFIED***
#include "../box_iou_rotated/box_iou_rotated_utils.h"
//#endif ***MODIFIED***
// todo avoid this when pytorch supports "same directory" hipification
//#ifdef WITH_HIP ***MODIFIED***
#include "box_iou_rotated/box_iou_rotated_utils.h"
//#endif ***MODIFIED***

pip install -e .
'''

# import some common libraries
import os, json, random, glob, inspect, sys
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from detector import Detector

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

def save_predictions_to_pdf(instance_detector,DIR_MODEL,DATASET,SUBSAMPLE,PDF_NAME,SHOW_IMG):
    with PdfPages(os.path.join(DIR_MODEL,PDF_NAME)) as pdf:
        imgs = os.path.abspath(os.path.join(DATASET,"*.jpg"))
        imgs = glob.glob(imgs)
        n = int(SUBSAMPLE*len(imgs))
        imgs = random.sample(imgs, n)
       
        for img in tqdm(imgs, desc=f'{bcolors.BOLD}Saving {n} images to {PDF_NAME}{bcolors.ENDC}',colour="green",position=0, total=n):
            fig = instance_detector.onImage(os.path.join(DATASET,img),SHOW_IMG)
            pdf.savefig(fig)
    print(f'{bcolors.OKGREEN}Done: {n} images with predictions saved to --> {os.path.join(DIR_MODEL,PDF_NAME)}{bcolors.ENDC}')


def evaluate_model_to_pdf(cfg):
    DIR_ROOT = os.getcwd()
    DIR_MODEL = os.path.join(DIR_ROOT,'detectron2','models',cfg['leafmachine']['segmentation_eval']['model_name'])

    DIR_TRAIN = cfg['leafmachine']['segmentation_train']['dir_images_train']
    DIR_VAL = cfg['leafmachine']['segmentation_train']['dir_images_val']
    DIR_TEST = cfg['leafmachine']['segmentation_train']['dir_images_test']
    
    EVAL_VAL = cfg['leafmachine']['segmentation_eval']['do_eval_val_images']
    EVAL_TRAIN = cfg['leafmachine']['segmentation_eval']['do_eval_training_images']
    EVAL_DIR = cfg['leafmachine']['segmentation_eval']['do_eval_check_directory']

    SHOW_IMG = cfg['leafmachine']['segmentation_eval']['show_image']
    THRESH = cfg['leafmachine']['segmentation_eval']['thresh']

    instance_detector = Detector(DIR_MODEL,THRESH)

    if EVAL_VAL:
        DATASET = DIR_VAL
        SUBSAMPLE = 1
        PDF_NAME = "__".join(['results_overlay_val',instance_detector.model_to_use_name.split(".")[0]])
        PDF_NAME = ".".join([PDF_NAME,"pdf"])
        SHOW_IMG = False
        save_predictions_to_pdf(instance_detector,DIR_MODEL,DATASET,SUBSAMPLE,PDF_NAME,SHOW_IMG)
    if EVAL_TRAIN:
        DATASET = DIR_TRAIN
        SUBSAMPLE = 1
        PDF_NAME = "__".join(['results_overlay_train',instance_detector.model_to_use_name.split(".")[0]])
        PDF_NAME = ".".join([PDF_NAME,"pdf"])
        SHOW_IMG = False
        save_predictions_to_pdf(instance_detector,DIR_MODEL,DATASET,SUBSAMPLE,PDF_NAME,SHOW_IMG)
    if EVAL_DIR:
        DATASET = DIR_TEST
        SUBSAMPLE = 0.01
        try:
            PDF_NAME = "".join(['results_overlay_leafwhole-sample',str(SUBSAMPLE).split(".")[1]])
        except:
            PDF_NAME = "".join(['results_overlay_leafwhole-sample',str(SUBSAMPLE).split(".")[0]])
        PDF_NAME = "__".join([PDF_NAME,instance_detector.model_to_use_name.split(".")[0]])
        PDF_NAME = ".".join([PDF_NAME,"pdf"])
        SHOW_IMG = False
        save_predictions_to_pdf(instance_detector,DIR_MODEL,DATASET,SUBSAMPLE,PDF_NAME,SHOW_IMG)
