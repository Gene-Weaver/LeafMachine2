import os, cv2, yaml, math, sys, inspect, imutils, random, copy
import numpy as np
from numpy import NAN, ndarray
import pandas as pd
from dataclasses import dataclass,field
from scipy import ndimage,stats
from scipy.signal import find_peaks
from scipy.stats.mstats import gmean
from scipy.spatial.distance import pdist, squareform
from skimage.measure import label, regionprops_table
from skimage.util import crop

import torch
import os, argparse, time, copy, cv2, wandb
import torch
from torchvision import *
from sklearn.cluster import KMeans
import statistics

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from time import perf_counter
from binarize_image_ML import DocEnTR

currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)
# from machine.general_utils import print_plain_to_console, print_blue_to_console, print_green_to_console, print_warning_to_console, print_cyan_to_console
# from machine.general_utils import bcolors

def convert_rulers_testing(dir_rulers, cfg, logger, dir_home, Project, batch, Dirs):
    RulerCFG = RulerConfig(logger, dir_home, Dirs, cfg)
    Labels = DocEnTR()
    model, device = Labels.load_DocEnTR_model(logger)

    acc_total = 0
    acc_error = 0
    acc_correct = 0
    incorrect_pair = []

    for subdir, _, files in os.walk(dir_rulers):
        for img_name in files:
            true_class = os.path.basename(subdir)
            # print(true_class)
            if true_class != 'fail':
                path_img = os.path.join(subdir, img_name)
                # print(path_img)

                ruler_cropped = cv2.imread(path_img)
                ruler_crop_name = img_name.split('.')[0]

                Ruler = setup_ruler(Labels, model, device, cfg, Dirs, logger, RulerCFG, ruler_cropped, ruler_crop_name)
                message = ''.join(["True Class: ", str(true_class), "    Pred Class: ",Ruler.ruler_class_pred])
                if Ruler.ruler_class != true_class:
                    acc_total += 1
                    acc_error += 1
                    incorrect_pair.append([true_class, Ruler.ruler_class])
                    Print_Verbose(RulerCFG.cfg,1,message).warning()
                else:
                    acc_total += 1
                    acc_correct += 1
                    Print_Verbose(RulerCFG.cfg,1,message).green()


                Ruler_Info = convert_pixels_to_metric(logger, RulerCFG,Ruler,ruler_crop_name, Dirs)

            # Project = add_ruler_to_Project(Project, batch, Ruler, BlockCandidate, filename, ruler_crop_name)
    
    print(f"Total = {acc_total} Error = {acc_error} Correct = {acc_correct}")
    print(f"Accuracy = {acc_correct/acc_total}")
    print(f"True / Incorrect: {incorrect_pair}")
    return Project

def convert_rulers(cfg, logger, dir_home, Project, batch, Dirs):
    t1_start = perf_counter()
    logger.info(f"Converting Rulers in batch {batch+1}")
    RulerCFG = RulerConfig(logger, dir_home, Dirs, cfg)
    Labels = DocEnTR()
    model, device = Labels.load_DocEnTR_model(logger)


    for filename, analysis in Project.project_data_list[batch].items():
        if len(analysis) != 0:
            Project.project_data_list[batch][filename]['Ruler_Info'] = []
            Project.project_data_list[batch][filename]['Ruler_Data'] = []
            logger.debug(filename)
            try:
                full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpg'])))
            except:
                full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpeg'])))

            try:
                archival = analysis['Detections_Archival_Components']
                has_rulers = True
            except: 
                has_rulers = False

            if has_rulers:
                height = analysis['height']
                width = analysis['width']
                ruler_list = [row for row in archival if row[0] == 0]
                # print(ruler_list)
                if len(ruler_list) < 1:
                    logger.debug('no rulers detected')
                else:
                    for ruler in ruler_list:
                        ruler_location = yolo_to_position_ruler(ruler, height, width)
                        ruler_polygon = [(ruler_location[1], ruler_location[2]), (ruler_location[3], ruler_location[2]), (ruler_location[3], ruler_location[4]), (ruler_location[1], ruler_location[4])]
                        # print(ruler_polygon)
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

                        Ruler = setup_ruler(Labels, model, device, cfg, Dirs, logger, RulerCFG, ruler_cropped, ruler_crop_name)

                        Ruler_Info = convert_pixels_to_metric(logger, RulerCFG,Ruler,ruler_crop_name, Dirs)

                        '''
                        **************************************
                        **************************************
                        **************************************
                        **************************************
                        FINISH THIS. NEED TO EXPORT THE DATA
                        **************************************
                        **************************************
                        **************************************
                        **************************************
                        '''
                        # Project = add_ruler_to_Project(Project, batch, Ruler, BlockCandidate, filename, ruler_crop_name) 
       
    t1_stop = perf_counter()
    logger.info(f"Converting Rulers in batch {batch+1} --- elapsed time: {round(t1_stop - t1_start)} seconds")
    return Project



def convert_pixels_to_metric(logger, RulerCFG, Ruler, img_fname, Dirs):#cfg,Ruler,imgPath,fName,dirSave,dir_ruler_correction,pathToModel,labelNames):
    Ruler_Redo = Ruler

    Ruler_Info = RulerInfo(Ruler, logger)

    
    
    if Ruler_Info.is_ticks_only:
        Ruler = straighten_img(logger, RulerCFG, Ruler, True, False, Dirs)
        Ruler_Info = convert_ticks(logger, Ruler_Info, RulerCFG, Ruler, img_fname, is_redo=False)

    # elif Ruler_Info.is_block_tick:
    #     Ruler = straighten_img(logger, RulerCFG, Ruler, True, False, Dirs)
    #     Ruler_Info = convert_ticks(logger, Ruler_Info, RulerCFG, Ruler, img_fname, is_redo=False)












    '''if check_ruler_type(Ruler.ruler_class,'tick_black'):
        
        colorOption = 'black'
        # colorOption = 'white'
        Ruler = straighten_img(logger, RulerCFG, Ruler, True, False, Dirs)
        Ruler_Out, BlockCandidate = convert_ticks(logger, RulerCFG, Ruler, colorOption, img_fname, is_redo=False)
        if not BlockCandidate['gmean']:
            Ruler_Redo = straighten_img(logger, RulerCFG, Ruler_Redo, True, False, Dirs)
            Ruler_Out, BlockCandidate = convert_ticks(logger, RulerCFG, Ruler_Redo, colorOption, img_fname, is_redo=True)

    elif check_ruler_type(Ruler.ruler_class,'tick_white'):
        colorOption = 'white'
        # colorOption = 'black'
        Ruler = straighten_img(logger, RulerCFG, Ruler, True, False, Dirs)
        Ruler_Out, BlockCandidate = convert_ticks(logger, RulerCFG, Ruler,colorOption, img_fname, is_redo=False)
        if not BlockCandidate['gmean']:
            Ruler_Redo = straighten_img(logger, RulerCFG, Ruler_Redo, True, False, Dirs)
            Ruler_Out, BlockCandidate = convert_ticks(logger, RulerCFG, Ruler_Redo, colorOption, img_fname, is_redo=True)

    elif check_ruler_type(Ruler.ruler_class,'block_regular_cm'):
        colorOption = 'invert'
        Ruler = straighten_img(logger, RulerCFG, Ruler, True, False, Dirs)
        Ruler_Out, BlockCandidate = convert_blocks(logger, RulerCFG, Ruler, colorOption, img_fname, Dirs, is_redo=False)
        if BlockCandidate.conversion_factor <= 0:
            Ruler_Redo = straighten_img(logger, RulerCFG, Ruler_Redo, True, False, Dirs)
            Ruler_Out, BlockCandidate = convert_blocks(logger, RulerCFG, Ruler_Redo, colorOption, img_fname, Dirs, is_redo=True)
    elif check_ruler_type(Ruler.ruler_class,'block_invert_cm'):
        colorOption = 'noinvert'
        Ruler = straighten_img(logger, RulerCFG, Ruler, True, False, Dirs)
        Ruler_Out, BlockCandidate = convert_blocks(logger, RulerCFG, Ruler, colorOption, img_fname, Dirs, is_redo=False)
        if BlockCandidate.conversion_factor <= 0:
            Ruler_Redo = straighten_img(logger, RulerCFG, Ruler_Redo, True, False, Dirs)
            Ruler_Out, BlockCandidate = convert_blocks(logger, RulerCFG, Ruler_Redo, colorOption, img_fname, Dirs, is_redo=True)


    else: # currently unsupported rulers
        Ruler_Out = []
        BlockCandidate = []'''

    return Ruler_Info

def convert_ticks(logger, Ruler_Info, RulerCFG,Ruler,img_fname, is_redo):

    Ruler_Info.process_scanline_chunk()

    Ruler_Info.cross_validate_conversion()
        
    Ruler_Info.insert_scanline()
    # cv2.imshow('img_total_overlay', Ruler_Info.Ruler.img_total_overlay)
    # cv2.waitKey(0)
    # cv2.imshow('img_ruler_overlay', Ruler_Info.Ruler.img_ruler_overlay)
    # cv2.waitKey(0)

    # newImg = create_overlay_bg(logger, RulerCFG,newImg)

    # cv2.imshow('newImg', newImg)
    # cv2.waitKey(0)

    # Ruler_Info.Ruler.img_ruler_overlay = create_overlay_bg(logger, RulerCFG,Ruler_Info.Ruler.img_ruler_overlay)
    validation = stack_2_imgs(cv2.cvtColor(Ruler_Info.Ruler.img_bi_pad, cv2.COLOR_GRAY2RGB), Ruler_Info.Ruler.img_ruler_overlay)
    validation = create_overlay_bg_3(logger, RulerCFG, validation)
    if Ruler_Info.conversion_mean != 0:
        validation = cv2.putText(img=validation, text=Ruler_Info.message_validation[0], org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(155, 155, 155),thickness=1)
        validation = cv2.putText(img=validation, text=Ruler_Info.message_validation[1], org=(10, 45), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(155, 155, 155),thickness=1)
        validation = cv2.putText(img=validation, text=Ruler_Info.message_validation[2], org=(10, 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(155, 155, 155),thickness=1)
    else:
        validation = add_text_to_img('Could not convert: No points found', validation)

    # cv2.imshow('img_total_overlay', newImg)
    # cv2.waitKey(0)
    newImg = stack_2_imgs(Ruler_Info.Ruler.img_total_overlay, validation)
    # cv2.imshow('img_total_overlay', Ruler_Info.Ruler.img_total_overlay)
    # cv2.waitKey(0)
    if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_validation']:
        cv2.imwrite(os.path.join(RulerCFG.dir_ruler_validation,'.'.join([img_fname, 'jpg'])), newImg)
    if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_validation_summary']:
        cv2.imwrite(os.path.join(RulerCFG.dir_ruler_validation_summary,'.'.join([img_fname, 'jpg'])), validation)

    return Ruler_Info
    

'''
####################################
####################################
           Main Functions
####################################
####################################
'''

class RulerInfo:
    fail = ['fail']

    grid = ['grid_white_cm']

    ticks_only = ['tick_black_4thcm', 'tick_black_cm_halfcm_4thcm', 'tick_black_cm_halfcm', 'tick_black_cm_halfcm_mm', 'tick_black_cm_halfcm_mm_halfmm', 'tick_black_cm_mm',
        'tick_black_dual_cm_4th_8th', 'tick_black_dual_cm_inch_halfinch_4th_8th', 'tick_black_dual_cm_mm_inch_8th', 'tick_black_halfcm_mm', 
        'tick_black_inch_4th_16th', 'tick_black_inch_8th', 'tick_black_inch_halfinch_4th_8th_16th', 'tick_black_inch_halfinch_4th_8th_16th_32nd', 
        'tick_nybg_white_cm_mm', 'tick_white_cm', 'tick_white_cm_halfcm_mm', 'tick_white_cm_mm', 'tick_white_inch_halfin_4th_8th_16th',]

    block_only = ['block_alternate_black_cm', 'block_alternate_white_cm', 'block_black_cm_halfcm_mm', 'block_fieldprism_black_cm', 
        'block_mini_black_cm_halfcm_mm', 'block_stagger_black_cm', 'block_white_cm_halfcm_mm',]

    block_tick = ['blocktick_alternate_black_cm_mm', 'blocktick_stagger_black_cm_halfcm_mm', 'blocktick_stagger_black_cm_mm', 
        'blocktick_stagger_black_inch_8th', 'blocktick_stagger_white_cm_halfcm_mm', 'blocktick_stagger_white_inch_4th_16th', 
        'blocktick_stagger_white_inch_16th', 'blocktick_step_black_cm_halfcm_mm', 'blocktick_step_black_halfinch_4th_16th', 
        'blocktick_step_white_cm_halfcm_mm', 'blocktick_step_white_halfinch_4th_16th',]


    metric = ['grid_white_cm', 'tick_black_4thcm', 'tick_black_cm_halfcm_4thcm','tick_black_cm_halfcm', 'tick_black_cm_halfcm_mm', 'tick_black_cm_halfcm_mm_halfmm', 
        'tick_black_cm_mm', 'tick_black_halfcm_mm', 'tick_nybg_white_cm_mm', 'tick_white_cm', 'tick_white_cm_halfcm_mm', 'tick_white_cm_mm', 
        'block_alternate_black_cm', 'block_alternate_white_cm', 'block_black_cm_halfcm_mm', 'block_fieldprism_black_cm', 
        'block_mini_black_cm_halfcm_mm', 'block_stagger_black_cm', 'block_white_cm_halfcm_mm', 'blocktick_alternate_black_cm_mm', 
        'blocktick_stagger_black_cm_halfcm_mm', 'blocktick_stagger_black_cm_mm', 'blocktick_stagger_white_cm_halfcm_mm', 'blocktick_step_black_cm_halfcm_mm',
        'blocktick_step_white_cm_halfcm_mm',]

    standard = ['tick_black_inch_4th_16th', 'tick_black_inch_8th', 'tick_black_inch_halfinch_4th_8th_16th', 'tick_black_inch_halfinch_4th_8th_16th_32nd',
        'tick_white_inch_halfin_4th_8th_16th', 'blocktick_stagger_black_inch_8th', 'blocktick_stagger_white_inch_4th_16th','blocktick_stagger_white_inch_16th',
        'blocktick_step_black_halfinch_4th_16th', 'blocktick_step_white_halfinch_4th_16th',]

    dual = ['tick_black_dual_cm_4th_8th', 'tick_black_dual_cm_inch_halfinch_4th_8th', 'tick_black_dual_cm_mm_inch_8th',]

    units_metric = ['cm', 'halfcm', '4thcm', 'mm', 'halfmm']
    units_standard = ['inch', 'halfinch', '4th', '8th', '16th', '32nd']
    units_dual = ['cm', 'halfcm', '4thcm', 'mm', 'halfmm', 'inch', 'halfinch', '4th', '8th', '16th', '32nd']

    def __init__(self, Ruler, logger) -> None:
        self.logger = logger
        self.Ruler = Ruler            
        self.ruler_class = self.Ruler.ruler_class
        if self.ruler_class == 'tick_black_4thcm':
            self.ruler_class = 'tick_black_cm_halfcm_4thcm'
        self.ruler_class_parts = self.ruler_class.split('_')

        self.is_ruler = False

        self.cross_validation_count = 0
        self.conversion_mean = 0
        self.conversion_mean_n = 0
        self.conversion_data = None
        self.conversion_data_all = []
        self.unit_list = []

        self.scanSize = 0

        self.is_grid = False
        self.is_ticks_only = False
        self.is_block_only = False
        self.is_block_tick = False

        self.is_metric = False
        self.is_standard = False
        self.is_dual = False

        self.contains_unit_metric = []
        self.contains_unit_standard = []
        self.contains_unit_dual = []

        self.check_if_ruler()
        self.check_main_path()
        self.check_metric_or_standard()

        self.get_units()

    def check_if_ruler(self):
        if self.ruler_class not in self.fail:
            self.is_ruler = True
        else:
            self.is_ruler = False

    def check_main_path(self):
        if self.ruler_class in self.grid:
            self.is_grid = True
        elif self.ruler_class in self.ticks_only:
            self.is_ticks_only = True
        elif self.ruler_class in self.block_only:
            self.is_block_only = True
        elif self.ruler_class in self.block_tick:
            self.is_block_tick = True
        else:
            pass

    def check_metric_or_standard(self):
        if self.ruler_class in self.metric:
            self.is_metric = True
        elif self.ruler_class in self.standard:
            self.is_standard = True
        elif self.ruler_class in self.dual:
            self.is_dual = True
        else:
            pass
    
    def get_units(self):
        for unit in self.units_metric:
            if unit in self.ruler_class_parts:
                self.contains_unit_metric.append(unit)
            else:
                pass
        for unit in self.units_standard:
            if unit in self.ruler_class_parts:
                self.contains_unit_standard.append(unit)
            else:
                pass

        for unit in self.units_dual:
            if unit in self.ruler_class_parts:
                self.contains_unit_dual.append(unit)
            else:
                pass

        if self.is_metric:
            self.n_units = len(self.contains_unit_metric)

        elif self.is_standard:
            self.n_units = len(self.contains_unit_standard) 

        elif self.is_dual:
            self.n_units = len(self.contains_unit_dual)

    def process_scanline_chunk(self):
        ### should return a list of just the scanlines that
        # 1. pass sanity check
        # 2. not nan
        # 3. are in the dominant mean distance category
        # 4. are in the minimal normalized SD category
        # 5. the union of 3 and 4
        # NOTE: normalized SD seems to be most reliable on getting the actual unit markers and not background noise
        # Union rows is what to compare against looking for cross validataion
        # intersect rows should be the dominant marker
        # data_list is everything
        
        scanSize = 5
        n_clusters = self.n_units + 1

        img_skel = skeletonize(self.Ruler.img_bi)
        img_bi = self.Ruler.img_bi
        # img = self.Ruler.img_bi

        img_skel[img_skel<=200] = 0
        img_skel[img_skel>200] = 1

        img_bi[img_bi<=200] = 0
        img_bi[img_bi>200] = 1

        h,w = img_skel.shape
        n = h % (scanSize *2)
        img_pad_skel = pad_binary_img(img_skel,h,w,n)
        img_pad_bi = pad_binary_img(img_bi,h,w,n)

        self.max_dim = max(img_pad_skel.shape)
        self.min_dim = min(img_pad_skel.shape)

        self.size_ratio = np.divide(self.min_dim, self.max_dim)

        img_pad = stack_2_imgs_bi(img_pad_bi, img_pad_skel)


        # img_pad_double = img_pad
        h,w = img_pad.shape
        x = np.linspace(0, w, w)

        
        self.Ruler.img_bi_pad = img_pad
        
        # best_value, union_rows, intersect_rows, data_list, union_means_list, intersect_means_list = process_scanline_chunk(img_pad, scanSize, h, logger, n_clusters)

        

        # Divide the chunk into the smaller parts to look for ticks
        means_list = []
        npts_dict = {}
        sd_list = []
        # distance_list = []
        data_list = []
        best_value = 0
        sd_temp_hold = 999
        scanlineData = {'index':[],'scanSize':[],'imgChunk':[],'plotPtsX':[],'plotPtsY':[],'plotPtsYoverall':[],'dists':[],'sd':[],'nPeaks':[],'normalizedSD':1000,'gmean':[],'mean':[]}    
        
        for i in range(int(h / scanSize)):
            chunkAdd = img_pad[scanSize * i: (scanSize * i + scanSize), :]
                
            # Locate the ticks in the chunk
            plotPtsX, plotPtsY, distUse, npts, peak_pos, avg_width = locate_ticks_centroid(chunkAdd, scanSize, i)

            if plotPtsX is not None:
                plot_points = list(zip(plotPtsX, plotPtsY))
                # Check the regularity of the tickmarks and their distances
                min_pairwise_distance = minimum_pairwise_distance(plotPtsX, plotPtsY)
                min_pairwise_distance_odd = minimum_pairwise_distance(plotPtsX[1::2], plotPtsY[1::2]) / 2
                min_pairwise_distance_even = minimum_pairwise_distance(plotPtsX[0::2], plotPtsY[0::2]) / 2
                min_pairwise_distance_third = minimum_pairwise_distance(plotPtsX[2::3], plotPtsY[2::3]) / 3
                sanity_check = sanity_check_scanlines(
                    min_pairwise_distance,
                    min_pairwise_distance_odd,
                    min_pairwise_distance_even,
                    min_pairwise_distance_third
                )
                
                mean_plus_normsd = np.mean(distUse) # + avg_width #+ (3*(np.std(distUse) / np.mean(distUse)))
                # print(f'bigger{mean_plus_normsd} ')
                # print(f'smaller{np.mean(distUse)}')

                sd_temp = (np.std(distUse) / np.mean(distUse))
                if sd_temp < sd_temp_hold and (npts >= 3) and sanity_check:
                    sd_temp_hold = sd_temp
                    self.avg_width = avg_width
                    best_value = np.mean(distUse)
                # Store the scanline data if the tickmarks are regular
                if sanity_check and not np.isnan(np.mean(distUse)):
                    chunkAdd[chunkAdd >= 1] = 255
                    scanlineData = {
                        'index': int(i),
                        'mean': mean_plus_normsd, #np.mean(distUse),
                        'normalizedSD': (np.std(distUse) / np.mean(distUse)),
                        'nPeaks': npts,
                        'sd': np.std(distUse),
                        'imgChunk': chunkAdd,
                        'plotPtsX': plotPtsX,
                        'plotPtsY': plotPtsY,
                        'plot_points': plot_points,
                        'plotPtsYoverall': (scanSize * i + scanSize) - round(scanSize / 2),
                        'dists': distUse,
                        'gmean': gmean(distUse),
                        'scanSize': int(scanSize),
                        'peak_pos': peak_pos
                    }
                    # message = f"gmean dist: {scanlineData['gmean']}"
                    # self.logger.debug(message)
                    # message = f"sd/n: {scanlineData['sd'] / scanlineData['nPeaks']}"
                    # self.logger.debug(message)
                    # message = f"n: {scanlineData['nPeaks']}"
                    # self.logger.debug(message)
                    
                    data_list.append(scanlineData)
                    # distance_list.append([np.mean(distUse), npts])
                    # means_list.append([np.mean(distUse)])
                    means_list.append([mean_plus_normsd])
                    sd_list.append([(np.std(distUse) / np.mean(distUse))])
                    # npts_dict[str(np.mean(distUse))] = npts
                    npts_dict[str(mean_plus_normsd)] = npts
            else:
                message = f"Notice: Scanline size {scanSize} iteration {i} skipped"#: {e.args[0]}"
                self.logger.debug(message)

        if len(means_list) >= 2:
            do_continue = True
        else:
            do_continue = False

        if n_clusters > len(means_list):
            n_clusters = len(means_list) - 1
            if n_clusters < 2:
                n_clusters = 2

        if do_continue:
            # Initialize the k-means model with n_units+1 clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=2022)
            # Fit the model to the data
            kmeans.fit(np.array(means_list).reshape(-1, 1))
            # Get the cluster centers and labels
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_

            # Determine which cluster has the higher count of data points
            counts = [sum(labels == i) for i in range(n_clusters)]
            dominant_cluster = np.argmax(counts)
            dominant_pattern = labels == dominant_cluster

            dom_pattern = np.where(dominant_pattern)[0]
            self.logger.debug(f"Dominant pattern indices: {np.where(dominant_pattern)[0]}")


            # Initialize the k-means model with 2 clusters
            kmeans = KMeans(n_clusters=2, random_state=2022)
            # Fit the model to the data
            kmeans.fit(np.array(sd_list).reshape(-1, 1))
            # Get the cluster centers and labels
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            # Determine which cluster has the smaller center value
            if centers[0] < centers[1]:
                minimal_pattern = labels == 0
            else:
                minimal_pattern = labels == 1
            min_pattern = np.where(minimal_pattern)[0]
            union = np.union1d(dom_pattern, min_pattern)
            union_rows = [data_list[i] for i in union]

            intersection = np.intersect1d(dom_pattern, min_pattern)
            intersect_rows = [data_list[i] for i in intersection]

            union_means_list = [means_list[i] for i in union]
            intersect_means_list = [means_list[i] for i in intersection]

            self.best_value = best_value
            self.union_rows = union_rows
            self.intersect_rows = intersect_rows
            self.data_list = data_list
            self.union_means_list = union_means_list
            self.intersect_means_list = intersect_means_list
            self.npts_dict = npts_dict

            self.intersect_means_list_indices = np.where(minimal_pattern)[0]
            self.union_means_list_indices = np.where(dominant_pattern)[0]

            self.logger.debug(f"Dominant pattern indices - (mean dist){np.where(dominant_pattern)[0]}")
            self.logger.debug(f"Minimal pattern indices (SD) - {np.where(minimal_pattern)[0]}")
            self.logger.debug(f"Union pattern indices - {union}")
            self.logger.debug(f"Average tick width - {self.avg_width}")
        else:
            self.best_value = best_value
            self.union_rows = None
            self.intersect_rows = None
            self.data_list = data_list
            self.union_means_list = None
            self.intersect_means_list = None
            self.npts_dict = npts_dict

            self.logger.debug(f"Not enough points located - only found {len(means_list)} - requires >= 2")
        
    def order_units_small_to_large(self, conversion_board, sorted_union_means_list, units_possible):
        current_val = sorted_union_means_list[0][0]
        # self.prev_val = current_val
        sorted_union_means_list_working = copy.deepcopy(sorted_union_means_list) 
        sorted_union_means_list_working.pop(0)

        # add first val
        conversion_board[self.current_unit].append(current_val)
        self.original_distances_1unit.append(current_val)

        for i, challenge_val in enumerate(sorted_union_means_list_working):
            challenge_val = sorted_union_means_list_working[i][0]
            # If we're not on the first value, check if it's within the tolerance of the previous value
            if abs(current_val - challenge_val) > self.tolerance: #next value is greater than tolerance
                is_correct_converion = False

                # Test conversion
                try:
                    challenge_unit = units_possible[self.key_ind + 1]
                except:
                    challenge_unit = units_possible[self.key_ind]

                is_correct_converion = self.test_conversion(self.current_unit, [current_val], challenge_unit, [challenge_val], self.tolerance) # current_unit, current_val, challenge_unit, challenge_val

                if is_correct_converion:
                    self.at_least_one_correct_conversion = True
                    self.key_ind += 1
                    self.current_unit = units_possible[self.key_ind]
                    conversion_board[self.current_unit].append(challenge_val)
                    self.prev_val = challenge_val
                    current_val = challenge_val
                    self.original_distances_1unit.append(challenge_val)
                else:
                    pass
            elif abs(current_val - challenge_val) <= self.tolerance: # values are the same, add to list
                self.cv_list_count += 1
                conversion_board[self.current_unit].append(challenge_val)
                self.original_distances_1unit.append(challenge_val)
                self.original_distances_1unit.append(self.prev_val)

        self.original_distances_1unit = list(set(self.original_distances_1unit))
        return conversion_board, sorted_union_means_list, units_possible

    def cross_validate_conversion(self):
        self.conversion_successful = False

        if self.is_metric:
            self.units_possible = self.contains_unit_metric
        elif self.is_standard:
            self.units_possible = self.contains_unit_standard
        elif self.is_dual:
            self.units_possible = self.contains_unit_dual

        self.logger.debug(f"[Cross Validate Conversion] - Units possible - {self.units_possible}")

        # TODO is self.union_means_list is empty ***********************************************************************************************************
        # sort the means lists from small to big #tried originally reversed to small
        sorted_union_means_list = sorted(self.union_means_list, reverse=False)
        sorted_intersect_means_list = sorted(self.intersect_means_list, reverse=False)

        # conversion_board_list = []
        remanining = sorted_union_means_list.copy()
        conversion_final = {}
        # conversion_board = {'convert':False, 'cm':[], 'halfcm':[], '4thcm':[], 'mm':[], 'halfmm':[], 'inch':[], 'halfinch':[], '4th':[], '8th':[], '16th':[], '32nd':[]}
        
        '''
        *** where the magic happens ***
        '''
        max_candidate = sorted_union_means_list[0][0]
        max_candidate_n = self.npts_dict[str(max_candidate)]
        min_candidate = sorted_union_means_list[-1][0]
        min_candidate_n = self.npts_dict[str(min_candidate)]

        # largest_unit = self.determine_largest_unit(sorted_intersect_means_list) 
        units_possible = self.units_possible[::-1]
        smallest_unit = units_possible[0]
        self.logger.debug(f"[Cross Validate Conversion] - Smallest unit - {smallest_unit}")

        # conversion_board  = {'convert':False}
        conversion_board  = {}
        for i in range(0, len(units_possible)):
            conversion_board[units_possible[i]] = []
        self.conversion_board_fresh = copy.deepcopy(conversion_board)

        self.tolerance = 5
        self.at_least_one_correct_conversion = False
        self.exhausted_units_to_test = False
        self.current_unit = smallest_unit
        self.current_val = 0
        self.prev_val = 0
        self.key_ind = 0
        self.original_distances_1unit = []
        sorted_union_means_list_fresh = copy.deepcopy(sorted_union_means_list) 
        unit_try = 0
        did_fail = False
        self.cv_list = []
        self.cv_list_dict = {}
        self.cv_list_count = 0
        self.cv_list_ind = 0
        '''
        start with smallest value. add next val to that unit IF within tolerance. 
        if next val > tol, then test conversion from previous to current unit. 
        if pass, add that to the next unit
        if fail, skip
        if all fail, then remove first unit's list and start over
        if unit list is exhausted, it will return the last successful list (happens with 'and not did_fail')
        #saves copies of the cv, will use the one with the most valid tolerance values as the export if  fail
        # TODO if mm is put into cm -> make size check based on the old-school dimension check, see how many units fit etc.
        '''
        while not self.at_least_one_correct_conversion:
            
            conversion_board, sorted_union_means_list, units_possible = self.order_units_small_to_large(conversion_board, sorted_union_means_list, units_possible)
            
            if not self.at_least_one_correct_conversion and self.exhausted_units_to_test: # This means that none of the conversions succeeded. Start over, but remove the smallest unit as a possibility
                self.logger.debug(f"[Cross Validate Conversion] - Conversion board units possible {units_possible} - {conversion_board}")
                self.exhausted_units_to_test = False # reset unit count
                unit_try += 1
                try:
                    self.current_unit = units_possible[unit_try]
                except:
                    self.at_least_one_correct_conversion = True # all units / tries are exhausted, give up
                    did_fail = True
                    #find best cv
                    # best_cv_ind  = max(self.cv_list_dict, key=self.cv_list_dict.get)
                    # conversion_board = copy.deepcopy(self.cv_list[best_cv_ind])

                sorted_union_means_list = copy.deepcopy(sorted_union_means_list_fresh) 

            if not self.at_least_one_correct_conversion and not self.exhausted_units_to_test and not did_fail: # Remove the smallest unit value and try again
                # make copy of cv
                # self.cv_list.append(copy.deepcopy(conversion_board))
                # self.cv_list_dict[self.cv_list_ind] = self.cv_list_count
                # self.cv_list_ind += 1
                # self.cv_list_count = 0

                conversion_board = conversion_board.clear()
                conversion_board = copy.deepcopy(self.conversion_board_fresh)
                self.current_val = 0
                self.key_ind = 0
                self.original_distances_1unit = []
                if len(sorted_union_means_list) > 2:
                    sorted_union_means_list.pop(0)
                else:
                    self.exhausted_units_to_test = True

        if did_fail:
            self.logger.debug(f"[Cross Validate Conversion] - Only 1 unit - order_units_small_to_large()")
            
        '''# assume best val is the biggest unit, then iterate, then shift to next smallest unit etc.
        # if there's only one unit we can skip a lot of the comparison, directly convert to cm
        if len(self.units_possible) == 1:
            original_distances_1unit = []
            for i in range(len(sorted_union_means_list)):
                candidate = sorted_union_means_list[i]

                # remove the candidate for the comparison
                remanining.pop(0)
                for j in range(len(remanining)):
                    cross = remanining[j]
                    candidate_val = self.convert_to_cm(self.units_possible[0], candidate)[0]
                    cross_val = self.convert_to_cm(self.units_possible[0], cross)[0]
                    candidate_unit = 'cm'
                    cross_unit = 'cm'

                    tolerance = 10
                    if (abs(candidate[0] - min(sorted_intersect_means_list)) <= tolerance) and abs(cross[0] -  min(sorted_intersect_means_list)) <= tolerance:
                        original_distances_1unit.append(cross[0]) # keep the og value to use to subset data_list later
                        conversion_board[candidate_unit].append(candidate_val)
                        # conversion_board[cross_unit].append(cross_val)
                        # Remove duplicates
                        conversion_board[candidate_unit] = list(set(conversion_board[candidate_unit]))
                        # conversion_board[cross_unit] = list(set(conversion_board[cross_unit]))
                        conversion_board['convert'] = True
            original_distances_1unit = list(set(original_distances_1unit))
        else:
            for i in range(len(sorted_union_means_list)):
                
                candidate = sorted_union_means_list[i]
                
                remanining.pop(0)
                for j in range(len(remanining)):

                    cross = remanining[j]

                    # if len(self.units_possible) == 1:
                    #     candidate_val = self.convert_to_cm(self.units_possible[0], candidate)[0]
                    #     cross_val = self.convert_to_cm(self.units_possible[0], cross)[0]
                    #     candidate_unit = 'cm'
                    #     cross_unit = 'cm'

                    #     conversion_board[candidate_unit].append(candidate_val)
                    #     conversion_board[cross_unit].append(cross_val)
                    #     # Remove duplicates
                    #     conversion_board[candidate_unit] = list(set(conversion_board[candidate_unit]))
                    #     conversion_board[cross_unit] = list(set(conversion_board[cross_unit]))
                    #     conversion_board['convert'] = True

                    # else:
                    if (largest_unit == 'cm') or (largest_unit == 'inch'):
                        # candidate_unit, cross_unit = self.is_within_tolerance_cm(candidate[0], cross[0]) # first was for big to small AND self.determine_largest_unit(sorted_intersect_means_list)
                        candidate_unit, cross_unit = self.is_within_tolerance_mm(candidate[0], cross[0])
                    elif largest_unit == 'mm':
                        # candidate_unit, cross_unit = self.is_within_tolerance_mm(candidate[0], cross[0])
                        candidate_unit, cross_unit = self.is_within_tolerance_cm(candidate[0], cross[0])
                    else:
                        # candidate_unit, cross_unit = self.is_within_tolerance_cm(candidate[0], cross[0])
                        candidate_unit, cross_unit = self.is_within_tolerance_mm(candidate[0], cross[0])


                    if candidate_unit is not None:
                        if (candidate_unit in self.units_possible) and (cross_unit in self.units_possible):
                            conversion_board[candidate_unit].append(candidate[0])
                            conversion_board[cross_unit].append(cross[0])
                            # Remove duplicates
                            conversion_board[candidate_unit] = list(set(conversion_board[candidate_unit]))
                            conversion_board[cross_unit] = list(set(conversion_board[cross_unit]))
                            conversion_board['convert'] = True
                        elif (candidate_unit not in self.units_possible) and (cross_unit in self.units_possible):
                            conversion_board[cross_unit].append(cross[0])
                            # Remove duplicates
                            conversion_board[cross_unit] = list(set(conversion_board[cross_unit]))
                            conversion_board['convert'] = True'''
        
        ### Different paths for if there are multiple possible units vs only one unit
        # if conversion_board['convert'] == True: # If there was at least one validation
        if self.at_least_one_correct_conversion: # If there was at least one validation
            self.conversion_successful = True
            self.logger.debug(f"[Cross Validate Conversion] - Conversion board final - {conversion_board}")

            '''getting the conversion value'''
            # get only the units allowed by the ML prediction
            if len(self.units_possible) > 1:
                for unit, unit_value in conversion_board.items(): # iterate through conversion_board
                    # if unit_value != []:                          # ignore empty units
                    if unit in self.units_possible:           # if unit is in the allowed units
                        # conversion_final[unit] = unit_value
                        self.cross_validation_count += 1
                        self.unit_list.append(unit)
            else:
                self.cross_validation_count += 1

            conversion_final = copy.deepcopy(conversion_board)

            '''prepping conversion_final'''
            '''if len(self.units_possible) == 1: # skip if only 1 unit
                conversion_final = {} # should all be 'cm', convert the  conversion_board to conversion_final
                # Add only non-empyty non-convert to final
                for key, value in conversion_board.items():
                    if not value:
                        pass
                    else:
                        # ignore
                        if key == 'convert':
                            pass
                        else:
                            conversion_final[key] = value

            # check to see if there are smaller values in larger bins
            else: # delete empty keys for when there are more than 1 possible units
                keys = list(conversion_final.keys())
                for i in range(len(keys) - 1):
                    key1 = keys[i]
                    key2 = keys[i + 1]
                    if (conversion_final[key1] != []) and (conversion_final[key2] != []):
                        if conversion_final[key1][0] < conversion_final[key2][0]:  # Compare the first element of the value list
                            conversion_final[key1] = conversion_final[key2]  # Update the value of the first key
                            conversion_final[key2] = []  # Set the value of the second key to an empty list'''
            
            # # Initialize the variable to store the results
            # do_reorder = False
            # # Loop through the conversion_final dictionary
            # for key in conversion_final:
            #     # Set a tolerance value for merging values
            #     tolerance = 5.0  
            #     # Check if all values in the list are within tolerance of the first value
            #     if all(abs(val - conversion_final[key][0]) <= tolerance for val in conversion_final[key]):
            #         # If all values are within tolerance, print False and break the loop
            #         do_reorder = False
            #     else:
            #         # If at least one value is not within tolerance, print True and set the result variable to True
            #         self.logger.debug(f"[Cross Validate Conversion] - Conversion board - Attempting reorder")
            #         do_reorder = True

            # do_reorder = False
            # if do_reorder:
            #     '''start iterating through the conversion_final keys in reverse order. in the first key, we need to check the 
            #         values inits list. create two temporary empty lists: keep_val and reject_val. in the key's list find the 
            #         smallest value and move all elements in the key's list to keep val and all remaining elements to  reject_val. 
            #         Then add all reject_val elements to the list in the next key's list. Then replace the first key's list with keep_val. 
            #         Reset the two temporary lists. then do everything again for the second key's list, but with one more step: 
            #         remove any elements from the second key's list that fall within the tolerance of the first list.'''
            #     tolerance = 5.0  # Set a tolerance value for merging values
            #     keys = list(conversion_final.keys())  # Get the list of keys
            #     keys.reverse()  # Reverse the list of keys

            #     for i in range(len(keys)):
            #         curr_key = keys[i]  # Current key
            #         curr_list = conversion_final[curr_key]  # Current list
                    
            #         keep_val = []  # Initialize the list to keep
            #         reject_val = []  # Initialize the list to reject
                    
            #         # Find the smallest value in the current list
            #         if len(curr_list) > 0:
            #             min_val = min(curr_list)
                        
            #             # Move all elements in the current list to keep_val or reject_val
            #             for val in curr_list:
            #                 if abs(val - min_val) <= tolerance:
            #                     keep_val.append(val)
            #                 else:
            #                     reject_val.append(val)
                        
            #             # Add all reject_val elements to the next key's list
            #             if i < len(keys) - 1:
            #                 next_key = keys[i+1]  # Next key
            #                 if next_key in conversion_final:
            #                     conversion_final[next_key].extend(reject_val)
            #                 else:
            #                     conversion_final[next_key] = reject_val
                        
            #             # Replace the current key's list with keep_val
            #             conversion_final[curr_key] = keep_val
                        
            #             # Initialize the lists for the next iteration
            #             keep_val = []
            #             reject_val = []
                        
            #             # Remove any elements from the current list that fall within the tolerance of the previous list
            #             if i > 0:
            #                 prev_key = keys[i-1]  # Previous key
            #                 prev_list = conversion_final[prev_key]  # Previous list
                            
            #                 for val in curr_list:
            #                     close_to_prev = False
            #                     for prev_val in prev_list:
            #                         if abs(val - prev_val) < tolerance:
            #                             close_to_prev = True
            #                             break
            #                     if not close_to_prev:
            #                         keep_val.append(val)
                        
            #                 # Replace the current key's list with keep_val
            #                 conversion_final[curr_key] = keep_val
            #     self.logger.debug(f"[Cross Validate Conversion] - Conversion board - Reorder result - {conversion_final}")

            '''NOTE nearly there... need to implement below but just to check to see if the values got moved enough
            ie mm -> halfmm but the val in halfmm is actually a cm...'''
            '''# assume best val is the biggest unit, then iterate, then shift to next smallest unit etc.
            for i in range(len(sorted_union_means_list)):
                
                candidate = sorted_union_means_list[i]
                
                remanining.pop(0)
                for j in range(len(remanining)):

                    cross = remanining[j]

                    if (largest_unit == 'cm') or (largest_unit == 'inch'):
                        candidate_unit, cross_unit = self.is_within_tolerance_cm(candidate[0], cross[0])
                    elif largest_unit == 'mm':
                        candidate_unit, cross_unit = self.is_within_tolerance_mm(candidate[0], cross[0])
                    else:
                        candidate_unit, cross_unit = self.is_within_tolerance_cm(candidate[0], cross[0])


                    if candidate_unit is not None:
                        if (candidate_unit in self.units_possible) and (cross_unit in self.units_possible):
                            conversion_board[candidate_unit].append(candidate[0])
                            conversion_board[cross_unit].append(cross[0])
                            # Remove duplicates
                            conversion_board[candidate_unit] = list(set(conversion_board[candidate_unit]))
                            conversion_board[cross_unit] = list(set(conversion_board[cross_unit]))
                            conversion_board['convert'] = True
                        elif (candidate_unit not in self.units_possible) and (cross_unit in self.units_possible):
                            conversion_board[cross_unit].append(cross[0])
                            # Remove duplicates
                            conversion_board[cross_unit] = list(set(conversion_board[cross_unit]))
                            conversion_board['convert'] = True'''

            # Remove keys with empty list values
            for key in list(conversion_final.keys()):
                if not conversion_final[key]:
                    del conversion_final[key]

            # remove outliers for each unit
            for unit, unit_value in conversion_final.items():
                if len(unit_value) >= 5:
                    conversion_final[unit] = self.remove_outliers_bias_larger(np.array(unit_value))

            # convert all values to cm
            if bool(conversion_final): # if conversion_final is not empty
                conversions_in_cm_nested = []
                for unit, unit_value in conversion_final.items():
                    # convert everything to cm, remove outliers, average
                    conversions_in_cm_nested.append(self.convert_to_cm(unit, unit_value))
                conversions_in_cm = [val for sublist in conversions_in_cm_nested for val in sublist]

                # remove outliers
                if len(conversions_in_cm) >= 5:
                    conversions_in_cm = self.remove_outliers_geomean(conversions_in_cm)

                # get average
                self.conversion_mean = np.mean(conversions_in_cm) #+ np.mean(conversions_in_cm)/50 #2*self.avg_width +
                self.conversion_mean_n = len(conversions_in_cm)
                self.logger.debug(f"[Cross Validate Conversion] - !!! Conversion Ratio !!! - 1cm = {np.round(self.conversion_mean, 2)} pixels")
                self.logger.debug(f"[Cross Validate Conversion] - !!! Conversion Ratio !!! - used {self.cross_validation_count} units - {list(conversion_final.keys())}")
                self.logger.debug(f"[Cross Validate Conversion] - !!! Conversion Ratio !!! - ratio is average of {self.conversion_mean_n} scanlines")
                self.message_validation = [f"1cm = {np.round(self.conversion_mean, 2)} pixels",f"Used {self.cross_validation_count} units - {list(conversion_final.keys())}",f"Ratio is average of {self.conversion_mean_n} scanlines"]

            
                '''getting the data that accompanies the values used to get the conversion value'''
                # Use conversion_final to get the instances from data_list
                if len(self.units_possible) == 1:
                    for ind in self.intersect_means_list_indices:
                        row = self.data_list[ind]
                        unit = self.units_possible[0]
                        if row['mean'] in self.original_distances_1unit:
                            self.conversion_data_all.append({unit:row})
                else:
                    for unit, unit_value in conversion_final.items():
                        for val in unit_value:
                            for row in self.data_list:
                                if row['mean'] == val:
                                    self.conversion_data_all.append({unit:row})
                self.logger.debug(f"[Cross Validate Conversion] - Conversion data")
                
                

        else:
            self.logger.debug(f"[Cross Validate Conversion] - Conversion not possible - {conversion_board}")
        self.logger.debug(f"[Cross Validate Conversion] - Done")

    def remove_outliers_bias_larger(self, dist):
        Q1 = np.percentile(dist, 30)
        Q3 = np.percentile(dist, 95)
        IQR = Q3 - Q1
        upper_bound = Q3 + IQR
        lower_bound = Q1 - IQR
        distUse = dist[(dist > lower_bound) & (dist < upper_bound)]
        return distUse

    def determine_largest_unit(self, sorted_union_means_list):
        for value in sorted_union_means_list:
            val = value[0]
            n = self.max_dim/val
            # n = self.npts_dict[str(val)]

            if self.is_metric:
                # 5 cm max ruler
                if self.size_ratio > (1/5):
                    try:
                        if (n-1 * val < self.max_dim) and (n < 7): # cm
                            return self.units_possible[0] 
                        elif (n-1 * val < self.max_dim) and (7 <= n < 20):
                            return self.units_possible[1]
                        else:
                            return 'mm'
                    except:
                        return 'mm'
                elif  (1/20) < self.size_ratio <= (1/5): #15cm
                    try:
                        if (n-1 * val < self.max_dim) and (n < 17): # cm
                            return self.units_possible[0] 
                        elif (n-1 * val < self.max_dim) and (17 <= n < 34):
                            return self.units_possible[1]
                        else:
                            return 'mm'
                    except:
                        return 'mm'
                else:# 30 cm max ruler
                    try:
                        if (n-1 * val < self.max_dim) and (n < 31): # cm
                            return self.units_possible[0] 
                        elif (n-1 * val < self.max_dim) and (31 <= n < 61):
                            return self.units_possible[1]
                        else:
                            return 'mm'
                    except:
                        return 'mm'
                '''elif (n-1 * val < self.max_dim) and (31 <= n < 310): # mm
                    try:
                        return self.units_possible[1]
                    except:
                        return self.units_possible[0]
                else: # halfmm
                    try:
                        return self.units_possible[2]
                    except:
                        try:
                            return self.units_possible[1]
                        except:
                            return self.units_possible[0]'''

            elif self.is_standard:
                # 15 inch max ruler
                if (n-1 * val < self.max_dim) and (n < 16): # inch
                    return self.units_possible[0] 
                else:
                    return 'mm'
                '''elif (n-1 * val < self.max_dim) and (16 <= n < 40): # halfinch
                    try:
                        return self.units_possible[1]
                    except:
                        return self.units_possible[0]
                else:
                    try:
                        return self.units_possible[2]
                    except:
                        try:
                            return self.units_possible[1]
                        except:
                            return self.units_possible[0]'''

            elif self.is_dual:
                if self.size_ratio < (1/20):
                    return 'mm'
                else:
                    return 'cm'


    def test_span(self, candidate, cross):
        test_candidate_n = self.npts_dict[str(candidate)]
        test_cross_n = self.npts_dict[str(cross)]

        # How many units fir into the space the points came from
        # if span_x = 150, then 150 units fit into the space

        span_candidate = self.max_dim / candidate
        span_cross = self.max_dim / cross
        # units * pixel length. coverage_if_mm will be less than max_dim IF it's mm
        coverage_if_mm_candidate = candidate * test_candidate_n
        coverage_if_mm_cross = cross * test_cross_n
        # units * pixel length. coverage_if_mm will be less than max_dim IF it's mm
        coverage_if_cm_candidate = candidate * span_candidate * test_candidate_n
        coverage_if_cm_cross = cross * span_cross * test_cross_n

        result_candidate = None
        result_cross = None
        if (coverage_if_mm_candidate < self.max_dim) and (coverage_if_cm_candidate > self.max_dim):
            if span_candidate <= 20:
                result_candidate = 'big'
            else:
                result_candidate = 'small'
        else:
            result_candidate = 'big'

        if (coverage_if_mm_cross < self.max_dim) and (coverage_if_cm_cross > self.max_dim):
            if span_candidate <= 20:
                result_cross = 'big'
            else:
                result_cross = 'small'
        else:
            result_cross = 'big'

        # if result_candidate == 'small': # take remedial action
        #     candidate_unit, cross_unit = self.is_within_tolerance(candidate[0], cross[0])


        return result_candidate, result_cross
    
    def test_conversion(self, current_unit, current_val, challenge_unit, challenge_val, tolerance):
        current_cm = self.convert_to_cm(current_unit, current_val)
        challenge_cm = self.convert_to_cm(challenge_unit, challenge_val)

        if abs(current_cm[0] - challenge_cm[0]) < tolerance:
            return True
        else:
            return False


    def convert_to_cm(self, unit, unit_value):
        unit_value_converted = []

        if unit == '32nd':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(np.multiply(val, 32), 2.54)))
        elif unit == '16th':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(np.multiply(val, 16), 2.54)))
        elif unit == '8th':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(np.multiply(val, 8), 2.54)))
        elif unit == '4th':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(np.multiply(val, 4), 2.54)))
        elif unit == 'halfinch':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(np.multiply(val, 2), 2.54)))
        elif unit == 'inch':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(np.multiply(val, 1), 2.54)))

        elif unit == 'halfmm':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(val, 20)))
        elif unit == 'mm':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(val, 10)))
        elif unit == '4thcm':
            for val in unit_value:    
                unit_value_converted.append(float(np.multiply(val, 4)))
        elif unit == 'halfcm':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(val, 2)))
        elif unit == 'cm':
            for val in unit_value:
                unit_value_converted.append(float(np.multiply(val, 1)))

        return unit_value_converted
        
    def remove_outliers_geomean(self, lst):
        if len(lst) >= 2:
            # Find the geometric mean
            geo_mean = math.exp(statistics.mean([math.log(x) for x in lst]))
            
            # Calculate the standard deviation
            std_dev = statistics.stdev(lst)
            
            # Calculate the threshold for outlier removal
            threshold = geo_mean + (0.5 * std_dev)
            
            # Create a new list without outliers
            new_lst = [x for x in lst if x <= threshold]
            
            return new_lst
        else:
            return lst

    def get_unit_color(self, unit):
        # BGR
        # Define color schemes
        color_list = [(0, 80, 0),  # dark green
              (0, 0, 255),  # Red
              (0, 165, 255),  # Orange
              (0, 255, 255),  # Yellow
              (255, 255, 0),  # Cyan
              (255, 165, 0),  # Gold
              (255, 0, 0),  # Blue
              (255, 0, 255),  # Magenta
              (128, 0, 128),  # Purple
              (0, 128, 128),  # Teal
              (0, 191, 255)]  # deep sky blue
        conversion = [(0, 255, 0), (255, 180, 0)]

        # Assign colors based on unit
        if unit == '32nd':
            return color_list[5]
        elif unit == '16th':
            return color_list[6]
        elif unit == '8th':
            return color_list[9]
        elif unit == '4th':
            return color_list[7]
        elif unit == 'halfinch':
            return color_list[3]
        elif unit == 'inch':
            return color_list[1]

        elif unit == 'halfmm':
            return color_list[10]
        elif unit == 'mm':
            return color_list[4]
        elif unit == '4thcm':
            return color_list[2]
        elif unit == 'halfcm':
            return color_list[8]
        elif unit == 'cm':
            return color_list[0]
        
        elif unit == 'conversion_1cm':
            return conversion[0]
        elif unit == 'conversion_10cm':
            return conversion[1]



    def insert_scanline(self): 
        imgBG = self.Ruler.img_copy

        # Plot all points
        unit_tally = []
        unit_plot = []
        for row in self.conversion_data_all:
            for unit, unit_data in row.items():
                if unit not in unit_tally:
                    unit_tally.append(unit)
                    unit_plot.append(row)

                color = self.get_unit_color(unit)
                x = unit_data['plotPtsX']
                y = unit_data['plotPtsY']

                # Loop through the x and y arrays and plot each point on the image
                for i in range(len(x)):
                    cv2.circle(imgBG, (int(x[i]), int(y[i])), 1, color, -1, cv2.LINE_AA)


        # Plot 1 unit marker for each identified unit (cm, mm, inch etc)
        # on first iteration, also plot the 1 CM line, thick all green
        plot_conversion = 0
        for row in unit_plot:
            for unit, unit_data in row.items():
                color = self.get_unit_color(unit)
                x = unit_data['plotPtsX']
                y = unit_data['plotPtsY']
                d = unit_data['mean']

                y_pt = y[0]
                imgBG = self.add_unit_marker(imgBG, d, x, 1, y_pt, color)

                if plot_conversion == 0:
                    plot_conversion += 1
                    color = self.get_unit_color('conversion_1cm')
                    imgBG = self.add_unit_marker(imgBG, self.conversion_mean, x, 1, y_pt-15, color)
                    color = self.get_unit_color('conversion_10cm')
                    # imgBG = self.add_unit_marker(imgBG, self.conversion_mean+(self.conversion_mean/50), x, 5, y_pt-10, color)
                    imgBG = self.add_unit_marker(imgBG, self.conversion_mean, x, 5, y_pt-20, color)
        
       
        # else:
        #     # print(f"{bcolors.WARNING}     No tickmarks found{bcolors.ENDC}")
        #     self.logger.debug(f"No tickmarks found")
        self.Ruler.img_ruler_overlay = imgBG
        
        do_show_combined = False
        if do_show_combined:
            # Load images
            img_bi = self.Ruler.img_bi_pad
            imgBG = imgBG

            # Resize images to have the same width
            width = max(img_bi.shape[1], imgBG.shape[1])
            height = img_bi.shape[0] + imgBG.shape[0]
            img_combined = np.zeros((height, width, 3), dtype=np.uint8)

            # Copy images vertically
            img_combined[:img_bi.shape[0], :img_bi.shape[1]] = cv2.cvtColor(img_bi, cv2.COLOR_GRAY2RGB)
            img_combined[img_bi.shape[0]:, :imgBG.shape[1]] = imgBG

            # Show combined image
            cv2.imshow("Combined", img_combined)
            cv2.waitKey(0)



    def add_unit_marker(self, img_bg, distance, x_coords, factor, y_pt, color):
        # shift_amount = - min(img_bg.shape[0], img_bg.shape[1]) / 10
        thickness = 4 if max(img_bg.shape[0], img_bg.shape[1]) > 1000 else 2
        x_coords.sort()

        try:
            first_marker_pos = int(x_coords[0])
            middle_marker_pos = int(x_coords[int(x_coords.size/2)])
            try:
                last_marker_pos = int(x_coords[-4] - (factor * distance))
            except:
                last_marker_pos = int(x_coords[-2] - (factor * distance))


            start_positions = [first_marker_pos, middle_marker_pos, last_marker_pos]
            end_positions = [first_marker_pos + int(distance * factor), 
                            middle_marker_pos + int(distance * factor), 
                            last_marker_pos - int(distance * factor)]

        except Exception as e:
            self.logger.debug(f"add_unit_marker(): plotting 1 of 3 unit markers. Exception: {e.args[0]}")

            middle_marker_pos = int(x_coords[int(x_coords.size/2)])
            start_positions = [middle_marker_pos]
            end_positions = [middle_marker_pos - int(distance * factor)]

        # do_plot = True
        # for pos in range(len(start_positions)):
        #     if do_plot:
        #         if factor > 1:
        #             do_plot = False
        #         # shift = shift_amount if (pos % 2) != 0 else -1 * shift_amount
        #         y_neg = -1 if pos % 2 else 1
        #         y_shift = (5 * y_neg) + int(y_pt) 
        #         start_point = (int(start_positions[pos]), y_shift)
        #         end_point = (int(end_positions[pos]), y_shift)
        #         cv2.line(img_bg, start_point, end_point, color, thickness, cv2.LINE_AA)
        do_plot = True
        for pos in range(0,len(start_positions),1):
            y_neg = -1 if pos % 2 else 1
            shift0 = 0 # y_neg * min(img_bg.shape[0], img_bg.shape[1]) / 10
            if do_plot:
                if factor > 1:
                    do_plot = False 
                if (pos % 2) != 0:
                    shift = -1 * shift0
                else:
                    shift = shift0
                for i in range(-thickness,thickness+1):
                    for j in range(start_positions[pos],end_positions[pos],1):
                        try:
                            # 5 pixel thick line
                            if (abs(i) == thickness) | (abs(j) == thickness):
                                img_bg[int(shift+y_pt+i),int(j),0] = color[0]
                                img_bg[int(shift+y_pt+i),int(j),1] = color[1]
                                img_bg[int(shift+y_pt+i),int(j),2] = color[2]
                            else:
                                img_bg[int(shift+y_pt+i),int(j),0] = color[0]
                                img_bg[int(shift+y_pt+i),int(j),1] = color[1]
                                img_bg[int(shift+y_pt+i),int(j),2] = color[2]
                        except:
                            continue

        return img_bg



def setup_ruler(Labels, model, device, cfg, Dirs, logger, RulerCFG, img, img_fname):
    # TODO add the classifier check
    Ruler = RulerImage(img=img, img_fname=img_fname)

    # print(f"{bcolors.BOLD}\nRuler: {img_fname}{bcolors.ENDC}")
    logger.debug(f"Ruler: {img_fname}")


    Ruler.ruler_class, Ruler.ruler_class_pred, Ruler.ruler_class_percentage,Ruler.img_type_overlay = detect_ruler(logger, RulerCFG, img, img_fname)
    
    
    # DocEnTr
    # If the run_binarize AND save was already called, just read the image
    '''if cfg['leafmachine']['cropped_components']['binarize_labels'] and cfg['leafmachine']['cropped_components']['do_save_cropped_annotations']:
        try:
            Ruler.img_bi = cv2.imread(os.path.join(Dirs.save_per_annotation_class, 'ruler_binary', '.'.join([img_fname, 'jpg'])))
        except:
            Ruler.img_bi = cv2.imread(os.path.join(Dirs.save_per_image, 'ruler_binary','.'.join([img_fname, 'jpg'])))
    else: # Freshly binarize the image'''
        
    do_skeletonize = False # TODO change this as needed per class

    # Ruler.ruler_class = 'block_regular_cm'

    

    Ruler.img_bi = Labels.run_DocEnTR_single(model, device, Ruler.img, do_skeletonize)

    ### Invert ruler if needed
    ruler_class_parts = Ruler.ruler_class.split('_')
    if 'white' in ruler_class_parts:
        Ruler.img_bi = cv2.bitwise_not(Ruler.img_bi)
    
    Ruler.img_bi_inv = cv2.bitwise_not(Ruler.img_bi)
    

    Ruler.img_bi_backup = Ruler.img_bi # THIS IS TEMP TODO should be ? maybe --> thresh, Ruler.img_bi_backup = cv2.threshold(Ruler.img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # cv2.imshow('bi',Ruler.img_bi)
    # cv2.waitKey(0)
    Ruler.img_bi_display = np.array(Ruler.img_bi)
    Ruler.img_bi_display = np.stack((Ruler.img_bi_display,)*3, axis=-1)
    return Ruler

def invert_if_white(image):
    # Count the number of white and black pixels
    num_white = np.count_nonzero(image == 255)
    num_black = np.count_nonzero(image == 0)
    
    # If there are more white pixels, invert the colors
    if num_white > num_black:
        image = cv2.bitwise_not(image)
    
    return image

def invert_if_black(img):
    # count the number of white and black pixels
    num_white = cv2.countNonZero(img)
    num_black = img.size - num_white
    
    # invert the colors if there are more black pixels than white
    if num_black > num_white:
        img = cv2.bitwise_not(img)
    
    return img

def find_minimal_change_in_binarization(img_gray, version):
    if version == 'block':
        result_list = []

        for idx, i in enumerate(range(0, 255, 10)):
            threshold_value = i
            img_bi = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)[1]
            result = cv2.countNonZero(img_bi)
            result_list.append((threshold_value, result))

        # x = [i[0] for i in result_list]
        y = [i[1] for i in result_list]

        # Calculate the first derivative
        dy = np.diff(y)

        # Calculate the second derivative
        # ddy = np.diff(dy)
        # min_index = np.argmin(dy)
        # min_index = np.argmin(ddy)
        # Find the index of the minimum value of the first derivative
        diffs = [abs(dy[i+5]-dy[i]) for i in range(len(dy)-5)]
        min_index = diffs.index(min(diffs))
        best_threshold = result_list[min_index][0]

        # diffs = [abs(y[i+5]-y[i]) for i in range(len(y)-5)]
        # min_index1 = diffs.index(min(diffs))
        # min_index = diffs.index(min([i for i in diffs if i >= 0.01*max(diffs)]))
        # best_threshold = result_list[min_index][0]
        # Turn this and the commented lines above for testing

        img_bi = cv2.threshold(img_gray, best_threshold, 255, cv2.THRESH_BINARY)[1]
        return img_bi

    elif version == 'tick':
        result_list = []

        for idx, i in enumerate(range(0, 255, 10)):
            threshold_value = i
            img_bi = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)[1]
            result = cv2.countNonZero(img_bi)
            result_list.append((threshold_value, result))

        # x = [i[0] for i in result_list]
        y = [i[1] for i in result_list]

        diffs = [abs(y[i+5]-y[i]) for i in range(len(y)-5)]
        # min_index = diffs.index(min(diffs))
        min_index = diffs.index(min([i for i in diffs if i >= 0.01*max(diffs)]))
        best_threshold = result_list[min_index][0]

        img_bi = cv2.threshold(img_gray, best_threshold, 255, cv2.THRESH_BINARY)[1]
        return img_bi




def detect_ruler(logger, RulerCFG, ruler_cropped, ruler_name):
    minimum_confidence_threshold = RulerCFG.cfg['leafmachine']['ruler_detection']['minimum_confidence_threshold']
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
    # Fix the 4thcm
    # if pred_class1 == 'tick_black_4thcm':
    #     self.Ruler.ruler_class = 'tick_black_cm_halfcm_4thcm'
    pred_class_orig = pred_class1
    

    if (percentage1 < minimum_confidence_threshold) or (percentage1 < (minimum_confidence_threshold*100)):
        pred_class_orig = pred_class1
        pred_class1 = f'fail_thresh_not_met__{pred_class_orig}'

    imgBG = create_overlay_bg(logger, RulerCFG, img.img_sq)
    addText1 = ''.join(["Class: ", str(pred_class1)])
    if percentage1 < minimum_confidence_threshold:
        addText1 = ''.join(["Class: ", str(pred_class1), '< thresh: ', str(pred_class_orig)])

    addText2 = "Certainty: "+str(percentage1)
    newName = '.'.join([ruler_name ,'jpg'])
    # newName = newName.split(".")[0] + "__overlay.jpg"
    imgOverlay = cv2.putText(img=imgBG, text=addText1, org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(155, 155, 155),thickness=1)
    imgOverlay = cv2.putText(img=imgOverlay, text=addText2, org=(10, 45), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(155, 155, 155),thickness=1)
    if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_validation']:
        cv2.imwrite(os.path.join(RulerCFG.dir_ruler_validation,newName),imgOverlay)

    message = ''.join(["Class: ", str(pred_class1), " Certainty: ", str(percentage1), "%"])

    # Print_Verbose(RulerCFG.cfg,1,message).green()

    logger.info(message)
    torch.cuda.empty_cache()
    return pred_class1, pred_class_orig, percentage1, imgOverlay


@dataclass
class RulerConfig:

    path_to_config: str = field(init=False)
    path_to_model: str = field(init=False)
    path_to_class_names: str = field(init=False)

    cfg: str = field(init=False)

    path_ruler_output_parent: str = field(init=False)
    dir_ruler_validation: str = field(init=False)
    dir_ruler_validation_summary: str = field(init=False)
    dir_ruler_processed: str = field(init=False)
    dir_ruler_data: str = field(init=False)

    net_ruler: object = field(init=False)

    def __init__(self, logger, dir_home, Dirs, cfg) -> None:
        self.path_to_config = dir_home
        self.cfg = cfg

        self.path_to_model = os.path.join(dir_home,'leafmachine2','machine','ruler_classifier','model')
        self.path_to_class_names = os.path.join(dir_home, 'leafmachine2','machine','ruler_classifier','ruler_classes.txt')

        self.path_ruler_output_parent = Dirs.ruler_info
        self.dir_ruler_validation = Dirs.ruler_validation
        self.dir_ruler_validation_summary =  Dirs.ruler_validation_summary
        self.dir_ruler_processed =  Dirs.ruler_processed
        self.dir_ruler_data =  Dirs.ruler_data

        # if self.cfg['leafmachine']['ruler_detection']['detect_ruler_type']:
        # try:
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0' # use only GPU 0
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        model_name = self.cfg['leafmachine']['ruler_detection']['ruler_detector']
        
        # torch.cuda.set_device(0)

        # checkpoint = torch.load(os.path.join(self.path_to_model,model_name), map_location='cuda:0')
        # checkpoint['state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}


        # Create the model architecture
        # model = models.resnet18(pretrained=True)
        # Load the state dict into the model
        # model.load_state_dict(checkpoint['state_dict'])

        # model.load_state_dict(checkpoint['model_state_dict'])
        


        self.net_ruler = torch.jit.load(os.path.join(self.path_to_model,model_name), map_location='cuda:0')
        self.net_ruler.eval()
        self.net_ruler.to('cuda:0') # specify device as 'cuda:0'
        # torch.jit.save(self.net_ruler, '/home/brlab/Dropbox/LeafMachine2/leafmachine2/machine/ruler_classifier/model/ruler_classifier_38classes_v-1.pt')
        # torch.save(self.net_ruler.state_dict(), '/home/brlab/Dropbox/LeafMachine2/leafmachine2/machine/ruler_classifier/model/ruler_classifier_38classes_v-1.pt')

        logger.info(f"Loaded ruler classifier network: {os.path.join(self.path_to_model,model_name)}")
        # except:
        #     logger.info("Could not load ruler classifier network")


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
        # self.img_sq = squarify(self.img,showImg=False,makeSquare=True,sz=360) # for model_scripted_resnet_720.pt
        self.img_sq = squarify_tile_four_versions(self.img, showImg=False, makeSquare=True, sz=720) # for 
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.img_t = self.transforms(self.img_sq)
        self.img_tensor = torch.unsqueeze(self.img_t, 0).cuda()

@dataclass
class RulerImage:
    img_path: str
    img_fname: str

    img: ndarray = field(init=False)

    img_bi: ndarray = field(init=False)
    img_bi_inv: ndarray = field(init=False)
    img_bi_backup: ndarray = field(init=False)

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
    ruler_class_pred: str = field(init=False)
    ruler_class_percentage: str = field(init=False)
    

    def __init__(self, img, img_fname) -> None:
        self.img = make_img_hor(img)
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img_copy = self.img.copy()
        self.img_copy = stack_2_imgs(self.img_copy, self.img_copy) # for the double binary [bi, skel]
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
def add_ruler_to_Project(Project, batch, Ruler, BlockCandidate, filename, ruler_crop_name):
    Project.project_data_list[batch][filename]['Ruler_Info'].append({ruler_crop_name: Ruler})
    Project.project_data_list[batch][filename]['Ruler_Data'].append({ruler_crop_name: BlockCandidate})

    # if 'block' in Ruler.ruler_class:
    #     Project.project_data[filename]['Ruler_Info'].append({ruler_crop_name: Ruler})
    #     Project.project_data[filename]['Ruler_Data'].append({ruler_crop_name: BlockCandidate})
    # elif 'tick' in Ruler.ruler_class:
    #     Project.project_data[filename]['Ruler_Info'].append({ruler_crop_name: Ruler})
    #     Project.project_data[filename]['Ruler_Data'].append({ruler_crop_name: BlockCandidate})
    #     print('tick')
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

def create_overlay_bg_3(logger, RulerCFG, img):
    try:
        try:
            h,w,_ = img.shape
            imgBG = np.zeros([h+90,w,3], dtype=np.uint8)
            imgBG[:] = 0
        except:
            img = binary_to_color(img)
            h,w,_ = img.shape
            imgBG = np.zeros([h+90,w,3], dtype=np.uint8)
            imgBG[:] = 0

        try:
            imgBG[90:img.shape[0]+90, :img.shape[1],:] = img
        except:
            imgBG[90:img.shape[0]+90, :img.shape[1]] = img

    except Exception as e:
        m = ''.join(['create_overlay_bg() exception: ',e.args[0]])
        # Print_Verbose(RulerCFG.cfg, 2, m).warning()
        logger.debug(m)
        img = np.stack((img,)*3, axis=-1)
        h,w,_ = img.shape
        imgBG = np.zeros([h+90,w,3], dtype=np.uint8)
        imgBG[:] = 0

        imgBG[90:img.shape[0]+90,:img.shape[1],:] = img
    return imgBG

def create_overlay_bg(logger, RulerCFG, img):
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
        # Print_Verbose(RulerCFG.cfg, 2, m).warning()
        logger.debug(m)
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
    img3[:,:] = (0,0,0)

    img3[:h1, :w1,:3] = img1
    try:
        img3[h1:h1+h2, :w2,:3] = img2
    except:
        img3[h1:h1+h2, :w2,:3] = binary_to_color(img2)
    return img3

def stack_2_imgs_bi(img1,img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    img3 = np.zeros((h1+h2, max(w1,w2)), dtype=np.uint8)
    img3[:h1, :w1] = img1
    img3[h1:h1+h2, :w2] = img2
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

def stack_image_quartile_rotate45_cropped_corners(img, q_increment, h, w, showImg):
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
                # Calculate the center of the rotated image
                center_x = int(rotated.shape[1] / 2)
                center_y = int(rotated.shape[0] / 2)
                # Calculate the coordinates of the top-left corner of the cropped image
                crop_x = max(0, center_x - int(h/2))
                crop_y = max(0, center_y - int(h/2))
                # Crop the rotated image to the desired size
                cropped = rotated[crop_y:crop_y+h, crop_x:crop_x+h]
                imgBG[ONE : TWO, THREE : FOUR] = cropped
            else:
                # imgBG[ONE : TWO, THREE : FOUR] = img[:, w - h : w]
                rotated = imutils.rotate_bound(img[:, w - h : w], rotate_options[increment])
                # Calculate the center of the rotated image
                center_x = int(rotated.shape[1] / 2)
                center_y = int(rotated.shape[0] / 2)
                # Calculate the coordinates of the top-left corner of the cropped image
                crop_x = max(0, center_x - int(h/2))
                crop_y = max(0, center_y - int(h/2))
                # Crop the rotated image to the desired size
                cropped = rotated[crop_y:crop_y+h, crop_x:crop_x+h]
                imgBG[ONE : TWO, THREE : FOUR] = cropped
            increment += 1

    if showImg:
        cv2.imshow('squarify_quartile()', imgBG)
        cv2.waitKey(0)
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

def squarify_maxheight(img, h, w, showImg=False):
    """
    Resizes input image so that height is the maximum and width is adjusted to make the image square.
    """
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    if random.random() < 0.5:
        img = cv2.rotate(img, cv2.ROTATE_180)
    
    resized = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
    if showImg:
        cv2.imshow('squarify_maxheight()', resized)
        cv2.waitKey(0)
    return resized

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
            increment += 1

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
        # cv2.imshow('Single Channel Window', imgPiece)
        # cv2.waitKey(0)
        imgBG[hStart:hEnd,wStart:wEnd] = img[:,wStartImg:wEndImg]
    if showImg:
        cv2.imshow('squarify()', imgBG)
        cv2.waitKey(0)
    return imgBG

def add_text_to_stacked_img(angle,img):
    addText1 = "Angle(deg) "+str(round(angle,3))+' Imgs: Orig,Binary,Skeleton,Validation'
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
    
    if random.random() < 0.5:
        imgSquarify = cv2.rotate(imgSquarify, cv2.ROTATE_180)

    return imgSquarify

def squarify_rotate45(imgSquarify, showImg, makeSquare, sz, doFlip):
    imgSquarify = make_img_hor(imgSquarify)
    
    # if doFlip:
    #     imgSquarify = cv2.rotate(imgSquarify,cv2.ROTATE_180) 

    q_increment,w,h = calc_squarify(imgSquarify,4)

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

    imgSquarify = stack_image_quartile(imgSquarify, q_increment, h, w, showImg)

    if makeSquare:
        dim = (sz, sz)
        imgSquarify = cv2.resize(imgSquarify, dim, interpolation = cv2.INTER_AREA)

    if random.random() < 0.5:
        imgSquarify = cv2.rotate(imgSquarify, cv2.ROTATE_180)

    return imgSquarify

def squarify_nine(imgSquarify, showImg, makeSquare, sz):
    imgSquarify = make_img_hor(imgSquarify)

    q_increment,w,h = calc_squarify(imgSquarify,9)

    imgSquarify = stack_image_nine(imgSquarify, q_increment, h, w, showImg)

    if makeSquare:
        dim = (sz, sz)
        imgSquarify = cv2.resize(imgSquarify, dim, interpolation = cv2.INTER_AREA)

    if random.random() < 0.5:
        imgSquarify = cv2.rotate(imgSquarify, cv2.ROTATE_180)

    return imgSquarify

def squarify_tile_four_versions(imgSquarify, showImg, makeSquare, sz):
    h = int(sz*2)
    w = int(sz*2)
    h2 = int(h/2)
    w2 = int(w/2)
    sq1 = squarify(imgSquarify,showImg,makeSquare,sz)
    sq2 = squarify_maxheight(imgSquarify, h/2, w/2, showImg)
    # sq2 = squarify_rotate45(imgSquarify, showImg, makeSquare, sz, doFlip=False)
    sq3 = squarify_quartiles(imgSquarify, showImg, makeSquare, sz, doFlip=showImg)
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

'''
####################################
####################################
            Process
####################################
####################################
'''
def straighten_img(logger, RulerCFG, Ruler, useRegulerBinary, alternate_img, Dirs):
    
    if useRegulerBinary:
        ruler_to_correct = Ruler.img_bi
    else:
        ruler_to_correct = np.uint8(alternate_img) # BlockCandidate.remaining_blobs[0].values

    image_rotated, Ruler.img, angle = rotate_bi_image_hor(ruler_to_correct, Ruler.img)

    # update all relevant images
    Ruler.img_copy = stack_2_imgs(Ruler.img, Ruler.img) # Used to make the overlay
    Ruler.img_bi_display = np.array(image_rotated)# Used to make the overlay
    Ruler.img_bi_display = np.stack((Ruler.img_bi_display,)*3, axis=-1)# Used to make the overlay
    Ruler.img_bi = image_rotated # Used to do the actual calculations# Used to make the overlay
    
    if (angle != 0.0) or (angle != 0): # If the rotation was substantial
        Ruler.correction_success = True
        Ruler.avg_angle = angle
    else:
        Ruler.correction_success = False
        Ruler.avg_angle = 0

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
    # newImg = stack_2_imgs(newImg,image_rotated)
    newImg = create_overlay_bg(logger, RulerCFG,newImg)
    newImg = add_text_to_stacked_img(Ruler.avg_angle,newImg)
    if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_validation']:
        newImg = stack_2_imgs(Ruler.img_type_overlay,newImg)

    Ruler.img_best = image_rotated
    Ruler.img_total_overlay = newImg

    # if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_validation']:
        # cv2.imwrite(os.path.join(Dirs.ruler_validation,'.'.join([Ruler.img_fname, 'jpg'])),Ruler.img_total_overlay)
    if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_processed']:
        cv2.imwrite(os.path.join(Dirs.ruler_processed,'.'.join([Ruler.img_fname, 'jpg'])),Ruler.img_best)
           
    # After saving the edges and imgBi to the compare file, flip for the class
    # Ruler.img_bi = ndimage.rotate(Ruler.img_bi,Ruler.avg_angle)
    # Ruler.img_bi = make_img_hor(Ruler.img_bi)
    ##### Ruler.img_edges = ndimage.rotate(Ruler.img_edges,Ruler.avg_angle) # no
    ###### Ruler.img_edges = make_img_hor(Ruler.img_edges) # no
    Ruler.img_gray = ndimage.rotate(Ruler.img_gray,Ruler.avg_angle)
    Ruler.img_gray = make_img_hor(Ruler.img_gray)

    # cv2.imwrite(os.path.join(RulerCFG.dir_ruler_overlay,'.'.join(['hi', 'jpg'])), Ruler.img_total_overlay)
    return Ruler

def rotate_bi_image_hor(binary_img, rgb_img):
    # Determine the orientation of the image
    # cv2.imshow('binary_img', binary_img)
    # cv2.waitKey(0)
    
    # Determine the orientation of the image
    (h, w) = binary_img.shape
    max_dim = max([h, w])
    if h > w:
        binary_img = cv2.rotate(binary_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Clean up the binary image using morphology operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)
    # closed_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours of objects in the image
    contours, hierarchy = cv2.findContours(opened_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours
    for contour in contours:
        try:
            # Compute the area of the contour
            area = cv2.contourArea(contour)
            rect = cv2.minAreaRect(contour)
            (x, y), (w2, h2), angle = rect
            aspect_ratio = min([w2, h2]) / max([w2, h2])

            if area <= 7:
                cv2.drawContours(opened_img, [contour], 0, 0, -1)
            if aspect_ratio > (1/7):
                if max_dim <= 1000:
                    # If the area is 5 pixels or less, set all pixels in the contour to 0
                    if area <= 10:
                        cv2.drawContours(opened_img, [contour], 0, 0, -1)
                elif max_dim <= 2000:
                    # If the area is 5 pixels or less, set all pixels in the contour to 0
                    if area <= 13:
                        cv2.drawContours(opened_img, [contour], 0, 0, -1)
                elif max_dim <= 4000:
                    # If the area is 5 pixels or less, set all pixels in the contour to 0
                    if area <= 17:
                        cv2.drawContours(opened_img, [contour], 0, 0, -1)
                else:
                    if area <= 23:
                        cv2.drawContours(opened_img, [contour], 0, 0, -1)
        except: 
            pass
    # Filter out non-rectangular objects
    # contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # rectangular_contours = []
    # for cnt in contours:
    #     peri = cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    #     if len(approx) == 4:
    #         rectangular_contours.append(cnt)

    # Find the largest 50 objects in the binary image
    # areas = [cv2.contourArea(cnt) for cnt in rectangular_contours]
    # sorted_indices = np.argsort(areas)[::-1][:50]
    # largest_contours = [rectangular_contours[i] for i in sorted_indices]

    # Detect lines in the rectangular objects
    # Adjust the HoughLinesP parameters
    LL = max(binary_img.shape) * 0.25
    bi_remove_text = remove_text(opened_img)
    lines = cv2.HoughLinesP(bi_remove_text, rho=1, theta=np.pi/180, threshold=25, minLineLength=LL/2, maxLineGap=5)

    angle = 0.0

    # Visualize the largest contours
    # cv2.imshow('largest_contours', bi_remove_text)
    # cv2.waitKey(0)

    # Determine the rotation angle based on the detected lines
    if lines is not None:
        if len(lines) > 0:
            all_angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1)
                all_angles.append(angle)

            angles, counts = np.unique(all_angles, return_counts=True)
            mode_index = np.argmax(counts)
            angle = angles[mode_index]

            # Rotate the image if the angle is not close to zero
            rotated_img = rotate_image_and_expand_bi(bi_remove_text, angle)
            rotated_img_rgb = rotate_image_and_expand(rgb_img, angle)
        else:
            rotated_img = bi_remove_text.copy()
            rotated_img_rgb = rgb_img
    else:
        rotated_img = bi_remove_text.copy()
        rotated_img_rgb = rgb_img
    
    (h, w) = rotated_img.shape
    angle = math.degrees(angle)
    if h > w:
        if angle < 0:
            angle = angle + 90
            rotated_img = cv2.rotate(rotated_img, cv2.ROTATE_90_CLOCKWISE)
            rotated_img_rgb = cv2.rotate(rotated_img_rgb, cv2.ROTATE_90_CLOCKWISE)
        else:
            angle = angle - 90
            rotated_img = cv2.rotate(rotated_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rotated_img_rgb = cv2.rotate(rotated_img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)


    # cv2.imshow('rot', rotated_img)
    # cv2.waitKey(0)
    # cv2.imshow('bi_remove_text', bi_remove_text)
    # cv2.waitKey(0)

    rotated_img = invert_if_white(rotated_img)

    return rotated_img, rotated_img_rgb, angle

def rotate_image_and_expand_bi(binary_img, angle):
    if abs(angle) >= np.deg2rad(1):
        (h, w) = binary_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, np.degrees(angle), 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated_img = cv2.warpAffine(binary_img, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    else:
        rotated_img = binary_img.copy()

    return rotated_img

def rotate_image_and_expand(rgb_img, angle):
    if abs(angle) >= np.deg2rad(1):
        (h, w) = rgb_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, np.degrees(angle), 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated_img = cv2.warpAffine(rgb_img, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    else:
        rotated_img = rgb_img.copy()

    return rotated_img


'''def rotate_bi_image_hor(binary_img):
    LL = max(binary_img.shape)*0.25
    # cv2.imshow('binary_img',binary_img)
    # cv2.waitKey(0)
    bi_remove_text = binary_img.copy()
    bi_remove_text = remove_text(bi_remove_text)
    # cv2.imshow('bi_remove_text',bi_remove_text)
    # cv2.waitKey(0)
    lines = cv2.HoughLinesP(bi_remove_text, 1, np.pi/180, 50, minLineLength=LL, maxLineGap=2)
    angle = 0.0
    if lines is not None:
        all_angles =[]
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            all_angles.append(angle)

        angles, counts = np.unique(all_angles, return_counts=True)
        mode_index = np.argmax(counts)
        angle = angles[mode_index]
        (h, w) = binary_img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, np.degrees(angle), 1.0)
        if angle >= abs(np.divide(math.pi, 180)): # more than 1 degree, then rotate
            rotated_img = cv2.warpAffine(binary_img, M, (w, h), flags=cv2.INTER_NEAREST)
            # cv2.imshow('bi_remove_text',bi_remove_text)
            # cv2.waitKey(0)
            # cv2.imshow('rotated_img',rotated_img)
            # cv2.waitKey(0)
        else:
            rotated_img = binary_img.copy()
    else:
        rotated_img = binary_img.copy()
    # cv2.imshow('rotated_img',rotated_img)
    # cv2.waitKey(0)
    return rotated_img, angle'''

def remove_text(img):
    img_copy = img.copy()
    img_copy_not = cv2.bitwise_not(img_copy)
    result = [img_copy, img_copy_not]
    result_filled = []
    for img in result:
        # Perform morphological dilation to expand the text regions
        kernel = np.ones((3,3), np.uint8)
        dilation = cv2.dilate(img, kernel, iterations=1)

        # Perform morphological erosion to shrink the text regions back to their original size
        erosion = cv2.erode(dilation, kernel, iterations=1)

        # Find contours in the processed image
        contours, _ = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours to keep only those likely to correspond to text regions
        text_contours = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            if aspect_ratio < 1/3 or aspect_ratio > 3/2:
                continue
            text_contours.append(c)

        # Draw filled contours on the copy of the binary image to fill in the text regions
        result_filled.append(cv2.drawContours(img, text_contours, -1, 255, -1))
    
    diff = [np.count_nonzero(img - img_copy) for img in result_filled]
    idx = np.argmax(diff)
    out = result_filled[idx]
    out = invert_if_white(out)
    return out




def locate_ticks_centroid(chunkAdd,scanSize, i):
    props = regionprops_table(label(chunkAdd), properties=('centroid',
                                            'orientation',
                                            'axis_major_length',
                                            'axis_minor_length'))
    props = pd.DataFrame(props)
    # Calculate the average width of the objects
    if np.any(props['axis_major_length']):
        widths = np.sqrt(props['axis_major_length'] * props['axis_minor_length'])
        avg_width = np.mean(widths)

        centoid = props['centroid-1']
        peak_pos = np.transpose(np.array(centoid))
        dst_matrix = peak_pos - peak_pos[:, None]
        dst_matrix = dst_matrix[~np.eye(dst_matrix.shape[0],dtype=bool)].reshape(dst_matrix.shape[0],-1)
        if (dst_matrix.shape[0] > 0) and (dst_matrix.shape[1] > 0):
            dist = np.min(np.abs(dst_matrix), axis=1)
            distUse = dist[dist > 2]

            if len(distUse) >= 5:
                distUse = remove_outliers(distUse)
            
                plotPtsX = peak_pos[dist > 2]
                plotPtsY = np.repeat(round((scanSize/2) + (((scanSize * i) + (scanSize * i + scanSize)) / 2)),plotPtsX.size)
                npts = len(plotPtsY)

                return plotPtsX,plotPtsY,distUse,npts,peak_pos,avg_width
            else:
                return None,None,None,None,None,None

        else:
            return None,None,None,None,None,None
    # Convert binary image to RGB
    # chunkAdd_rgb = np.stack((chunkAdd*255,)*3, axis=-1).astype(np.uint8)
    # Draw a small circle for each centroid
    # for i in range(len(plotPtsX)):
    #     # Draw a circle around the centroid
    #     cv2.circle(chunkAdd_rgb, (int(plotPtsX[i]), 3), 2, (0, 0, 255), -1)
    # # Show the image
    # cv2.imshow('Centroids', chunkAdd_rgb)
    # cv2.waitKey(0)
    return None,None,None,None,None,None


def remove_outliers(dist):
    '''threshold = 2
    z = np.abs(stats.zscore(dist))
    dist = dist[np.where(z < threshold)]
    threshold = 1
    z = np.abs(stats.zscore(dist))
    dist = dist[np.where(z < threshold)]
    threshold = 1
    z = np.abs(stats.zscore(dist))
    distUse = dist[np.where(z < threshold)]'''
    Q1 = np.percentile(dist, 25)
    Q3 = np.percentile(dist, 75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    distUse = dist[(dist > lower_bound) & (dist < upper_bound)]
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

def skeletonize(img):
    # try:
    #     img = cv2.ximgproc.thinning(img)
    # except:
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     img = cv2.ximgproc.thinning(gray)
    return cv2.ximgproc.thinning(img)

    '''skel = np.zeros(img.shape,np.uint8)
    size = np.size(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False

    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        # cv2.imshow("skel",skel)
        # cv2.waitKey(0)    
        zeros = size - cv2.countNonZero(img)
        if np.amax(skel) == np.amin(skel):
            done = True
            return img
        else:
            if zeros==size:
                done = True
                return skel'''
        

    

def minimum_pairwise_distance(plotPtsX, plotPtsY):
    points = np.column_stack((plotPtsX, plotPtsY))
    distances = cdist(points, points)
    np.fill_diagonal(distances, np.inf)
    min_distances = np.min(distances, axis=1)
    min_pairwise_distance = gmean(min_distances)
    return min_pairwise_distance


def standard_deviation_of_pairwise_distance(plotPtsX, plotPtsY):
    x = np.asarray(plotPtsX)
    y = np.asarray(plotPtsY)
    valid_indices = np.where(np.logical_and(np.logical_not(np.isnan(x)), np.logical_not(np.isnan(y))))[0]
    x = x[valid_indices]
    y = y[valid_indices]
    arrmean = np.mean(x)
    x = np.asanyarray(x - arrmean)
    return np.sqrt(np.mean(x**2))

def sanity_check_scanlines(min_pairwise_distance, min_pairwise_distance_odd, min_pairwise_distance_even, min_pairwise_distance_third):
    if min_pairwise_distance_odd < min_pairwise_distance / 2 or min_pairwise_distance_odd > min_pairwise_distance * 2:
        return False
    if min_pairwise_distance_even < min_pairwise_distance / 2 or min_pairwise_distance_even > min_pairwise_distance * 2:
        return False
    if min_pairwise_distance_third < min_pairwise_distance / 3 or min_pairwise_distance_third > min_pairwise_distance * 3:
        return False
    return True

def verify_cm_vs_mm(scanlineData):
    try:
        max_dim = max(scanlineData.get("imgChunk").shape)
        x = scanlineData.get("peak_pos")
        n = scanlineData.get("nPeaks")
        distUse = scanlineData.get("gmean")

        # How many units fir into the space the points came from
        # if span_x = 150, then 150 units fit into the space
        span_x = (max(x) - min(x)) / distUse
        # units * pixel length. coverage_if_mm will be less than max_dim IF it's mm
        coverage_if_mm = distUse * span_x
        # units * pixel length. coverage_if_mm will be less than max_dim IF it's mm
        coverage_if_cm = distUse * span_x * 10

        # print(span_x)
        if (coverage_if_mm < max_dim) and (coverage_if_cm > max_dim):
            if span_x <= 30:
                return 'cm'
            else:
                return 'mm'
        else:
            return 'cm'
    except:
        return []


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

def sort_blobs_by_size(logger, RulerCFG, Ruler, isStraighten):
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
            BlockCandidate = determine_block_blob_type(logger, RulerCFG,BlockCandidate,i)#BlockCandidate.largest_blobs[0],BlockCandidate.img_bi_overlay)
        if isStraighten == False:
            Ruler.img_block_overlay = BlockCandidate.img_bi_overlay

        BlockCandidate = calculate_block_conversion_factor(BlockCandidate,nBlockCheck)  
    BlockCandidate.success_sort = success
    return Ruler, BlockCandidate


def convert_blocks(logger, RulerCFG,Ruler,colorOption,img_fname, Dirs, is_redo):
    if is_redo:
        Ruler.img_bi = Ruler.img_bi_backup

    if colorOption == 'invert':
        Ruler.img_bi = cv2.bitwise_not(Ruler.img_bi)
    
    # Straighten the image here using the BlockCandidate.remaining_blobs[0].values
    Ruler,BlockCandidate = sort_blobs_by_size(logger, RulerCFG, Ruler,isStraighten=True) 
    if BlockCandidate.success_sort:
        useRegulerBinary = True
        Ruler = straighten_img(logger, RulerCFG, Ruler, useRegulerBinary, BlockCandidate.remaining_blobs[0], Dirs)
        del BlockCandidate
        Ruler,BlockCandidate = sort_blobs_by_size(logger, RulerCFG,Ruler,isStraighten=False) 

    
        if BlockCandidate.success_sort: # if this is false, then no marks could be ID'd, will print just the existing Ruler.img_total_overlay
            if BlockCandidate.conversion_location != 'fail':
                BlockCandidate = add_unit_marker_block(BlockCandidate,1)
                BlockCandidate = add_unit_marker_block(BlockCandidate,10)

    message = ''.join(["Angle (deg): ", str(round(Ruler.avg_angle,2))])
    logger.debug(message)
    # Print_Verbose(RulerCFG.cfg,1,message).cyan()

    BlockCandidate.img_bi_overlay = create_overlay_bg(logger, RulerCFG,BlockCandidate.img_bi_overlay)
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

    if RulerCFG.cfg['leafmachine']['ruler_detection']['save_ruler_validation_summary']:
        cv2.imwrite(os.path.join(RulerCFG.dir_ruler_validation_summary,'.'.join([img_fname, 'jpg'])),Ruler.img_total_overlay)

    return Ruler, BlockCandidate




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
                if (int(Y+i) < imgBG.shape[0]) and (int(X+j) < imgBG.shape[1]) and (int(Y+i) >= 0) and (int(X+j) >= 0):
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

def determine_block_blob_type(logger, RulerCFG,BlockCandidate,ind):
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
        # Print_Verbose(RulerCFG.cfg,2,message).plain()
        logger.debug(message)
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




def find_minimal_change_in_binarization_TESTING(img_gray):
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

    # x = [i[0] for i in result_list]
    # y = [i[1] for i in result_list]

    # x = [i[0] for i in result_list]
    y = [i[1] for i in result_list]

    # Calculate the first derivative
    dy = np.diff(y)

    # Calculate the second derivative
    # ddy = np.diff(dy)
    # min_index = np.argmin(dy)
    # min_index = np.argmin(ddy)
    # Find the index of the minimum value of the first derivative
    diffs = [abs(dy[i+5]-dy[i]) for i in range(len(dy)-5)]
    min_index = diffs.index(min(diffs))
    best_threshold = result_list[min_index][0]

    # diffs = [abs(y[i+5]-y[i]) for i in range(len(y)-5)]
    # min_index1 = diffs.index(min(diffs))
    # min_index = diffs.index(min([i for i in diffs if i >= 0.01*max(diffs)]))
    # best_threshold = result_list[min_index][0]
    # Turn this and the commented lines above for testing
    '''
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
    '''   
    img_bi = cv2.threshold(img_gray, best_threshold, 255, cv2.THRESH_BINARY)[1]
    return img_bi






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