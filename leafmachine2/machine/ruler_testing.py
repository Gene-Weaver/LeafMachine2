import os, cv2, yaml, math, sys, inspect, imutils, random
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
from scipy.spatial.distance import cdist
from time import perf_counter
from binarize_image_ML import DocEnTR
from utils_ruler import RulerConfig, setup_ruler, convert_pixels_to_metric

currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)



'''def convert_rulers_testing(dir_rulers, cfg, logger, dir_home, Project, batch, Dirs):
    RulerCFG = RulerConfig(logger, dir_home, Dirs, cfg)
    Labels = DocEnTR()
    model, device = Labels.load_DocEnTR_model(logger)

    for subdir, _, files in os.walk(dir_rulers):
        for img_name in files:
            true_class = os.path.basename(subdir)
            print(true_class)
            path_img = os.path.join(subdir, img_name)
            print(path_img)


            ruler_cropped = cv2.imread(path_img)
            ruler_crop_name = img_name.split('.')[0]

            # Get the cropped image using cv2.getRectSubPix
            # ruler_cropped = cv2.getRectSubPix(full_image, (int(ruler_location[3] - ruler_location[1]), int(ruler_location[4] - ruler_location[2])), (points[0][0][0], points[0][0][1]))

            Ruler = setup_ruler(Labels, model, device, cfg, Dirs, logger, RulerCFG, ruler_cropped, ruler_crop_name)

            Ruler, BlockCandidate = convert_pixels_to_metric(logger, RulerCFG,Ruler,ruler_crop_name, Dirs)

            # Project = add_ruler_to_Project(Project, batch, Ruler, BlockCandidate, filename, ruler_crop_name)
       
    return Project
'''




if __name__ == '__main__':
    dir_rulers = 'F:/Rulers_ByType_V2_sample'
    convert_rulers_testing(dir_rulers)
