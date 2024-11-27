# save the labelbox groundtruth overlay images
from cmath import e
from pprint import pprint
import labelbox
from labelbox import Client
from labelbox import Client, OntologyBuilder
# from labelbox.data.annotation_types import Geometry # pip install labelbox[data]
from PIL import Image, ImageDraw # pillow
import numpy as np
import os, sys, inspect, requests, json, time, math, random
from pathlib import Path
from dataclasses import dataclass, field
import pandas as pd
from typing import List, Tuple
import csv
import yaml #pyyaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# from utils import make_dirs
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from machine.general_utils import get_cfg_from_full_path, bcolors, validate_dir
from utils_Labelbox import assign_index, redo_JSON, OPTS_EXPORT_POINTS

class KeypointMapping:
    def __init__(self, trace_version):
        if trace_version == 'mid15_pet5':
            self.mapping = {
                'lamina_tip': 0,
                'apex_left': 1,
                'apex_center': 2,
                'apex_right': 3,
                'midvein_0': 4,
                'midvein_1': 5,
                'midvein_2': 6,
                'midvein_3': 7,
                'midvein_4': 8,
                'midvein_5': 9,
                'midvein_6': 10,
                'midvein_7': 11,
                'midvein_8': 12,
                'midvein_9': 13,
                'midvein_10': 14,
                'midvein_11': 15,
                'midvein_12': 16,
                'midvein_13': 17,
                'midvein_14': 18,
                'base_left': 19,
                'base_center': 20,
                'base_right': 21,
                'lamina_base': 22,
                'petiole_0': 23,
                'petiole_1': 24,
                'petiole_2': 25,
                'petiole_3': 26,
                'petiole_4': 27,
                'petiole_tip': 28,
                'width_left': 29,
                'width_right': 30,
            }
        elif trace_version == 'mid30_pet10':
            self.mapping = {
                'lamina_tip': 0,
                'apex_left': 1,
                'apex_center': 2,
                'apex_right': 3,
                'midvein_0': 4,
                'midvein_1': 5,
                'midvein_2': 6,
                'midvein_3': 7,
                'midvein_4': 8,
                'midvein_5': 9,
                'midvein_6': 10,
                'midvein_7': 11,
                'midvein_8': 12,
                'midvein_9': 13,
                'midvein_10': 14,
                'midvein_11': 15,
                'midvein_12': 16,
                'midvein_13': 17,
                'midvein_14': 18,
                'midvein_15': 19,
                'midvein_16': 20,
                'midvein_17': 21,
                'midvein_18': 22,
                'midvein_19': 23,
                'midvein_20': 24,
                'midvein_21': 25,
                'midvein_22': 26,
                'midvein_23': 27,
                'midvein_24': 28,
                'midvein_25': 29,
                'midvein_26': 30,
                'midvein_27': 31,
                'midvein_28': 32,
                'midvein_29': 33,
                'base_left': 34,
                'base_center': 35,
                'base_right': 36,
                'lamina_base': 37,
                'petiole_0': 38,
                'petiole_1': 39,
                'petiole_2': 40,
                'petiole_3': 41,
                'petiole_4': 42,
                'petiole_5': 43,
                'petiole_6': 44,
                'petiole_7': 45,
                'petiole_8': 46,
                'petiole_9': 47,
                'petiole_tip': 48,
                'width_left': 49,
                'width_right': 50,
            }
        elif trace_version == 'corners':
            self.mapping = {
                'top_left': 0,
                'top_right': 1,
                'bottom_left': 2,
                'bottom_right': 3,
            }
        elif trace_version == 'petiole_width':
            self.mapping = {
                'top_left': 0,
                'top_right': 1,
                'bottom_left': 2,
                'bottom_right': 3,
            }
    def get_index(self, keypoint_name):
        return self.mapping.get(keypoint_name, None)

@dataclass
class Points:
    IMG_FILENAME: str 
    IMG_NAME: str 
    VERSION: str

    # New dictionary to hold 52 keypoints
    KEYPOINTS_51: List[List[Tuple[int, int, int]]] = field(default_factory=lambda: [None] * 51)
    KEYPOINTS_31: List[List[Tuple[int, int, int]]] = field(default_factory=lambda: [None] * 31)
    KEYPOINTS_4CORNERS: List[List[Tuple[int, int, int]]] = field(default_factory=lambda: [None] * 4)
    KEYPOINTS: List[tuple] = field(init=False)

    M_LOBE_COUNT: int = 0
    M_DEEPEST_SINUS_ANGLE: int = 0
    M_OUTER: int = 0
    
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

    # Acacia
    TIP: list = field(init=False,default_factory=list)
    TIP_N: int = field(init=False)
    M_TIP: float = field(init=False)
    MIDDLE: list = field(init=False,default_factory=list)
    MIDDLE_N: int = field(init=False)
    M_MIDDLE: float = field(init=False)
    OUTER: list = field(init=False,default_factory=list)
    OUTER_N: int = field(init=False) 

    def __post_init__(self) -> None:
        if self.VERSION == 'mid15_pet5':
            self.KEYPOINTS = self.KEYPOINTS_31
        elif self.VERSION == 'mid30_pet10':
            self.KEYPOINTS = self.KEYPOINTS_51
        elif self.VERSION == 'corners':
            self.KEYPOINTS = self.KEYPOINTS_4CORNERS              
            

    def total_distance(self,pts, is_multi=False):
        if is_multi:
            totals = []
            for pt in pts:
                total = 0
                for i in range(len(pts) - 1):
                    total += math.dist(pts[i],pts[i+1])
                totals.append(total)
            return totals

        else:
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

        # Acacia
        try:
            self.M_MIDDLE = self.total_distance(self.MIDDLE)
            # print(f'M_MIDVEIN_LENGTH: {self.M_MIDVEIN_LENGTH}')
        except:
            self.M_MIDDLE = 0
        try:
            self.M_OUTER = self.total_distance(self.OUTER, is_multi=True)
            # print(f'M_MIDVEIN_LENGTH: {self.M_MIDVEIN_LENGTH}')
        except:
            self.M_OUTER = 0

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

        # ACACIA
        try:
            self.TIP_N = len(self.TIP)
        except:
            self.TIP = 0
            self.TIP_N = 0
        try:
            self.MIDDLE_N = len(self.MIDDLE)
        except:
            self.MIDDLE = 0
            self.MIDDLE_N = 0
        try:
            self.OUTER_N = len(self.OUTER)
        except:
            self.OUTER = 0
            self.OUTER_N = 0

        if add_pts_counts: # will add the counts of the number of points to the dataframe
            headers = ['img_name','img_filename',
                    'lamina_length','midvein_length','petiole_length','petiole_trace_length','lamina_width','apex_angle','base_angle','angle_types','lobe_count',
                    'loc_lobe_tip','loc_lamina_tip','loc_lamina_base','loc_petiole_tip','loc_lamina_width','loc_midvein_trace','loc_petiole_trace','loc_apex_angle','loc_base_angle','loc_deepest_sinus',
                    'loc_tip','loc_middle','loc_outer',
                    'loc_lobe_tip_n','loc_lamina_tip_n','loc_lamina_base_n','loc_petiole_tip_n','loc_lamina_width_n','loc_midvein_trace_n','loc_petiole_trace_n','loc_apex_angle_n','loc_base_angle_n','loc_deepest_sinus_n',
                    'loc_tip_n','loc_middle_n','loc_outer_n',]
            data = {'img_name':[self.IMG_NAME],'img_filename':[self.IMG_FILENAME],
                    'lamina_length':[self.M_LAMINA_LENGTH],'midvein_length':[self.M_MIDVEIN_LENGTH],'petiole_length':[self.M_PETIOLE_LENGTH],'petiole_trace_length':[self.M_PETIOLE_TRACE_LENGTH],'lamina_width':[self.M_LAMINA_WIDTH],
                    'apex_angle':[self.M_APEX_ANGLE],'base_angle':[self.M_BASE_ANGLE],'angle_types':[self.ANGLE_TYPES],'lobe_count':[self.M_LOBE_COUNT],
                    'loc_lobe_tip':[self.LOBE_TIP],'loc_lamina_tip':[self.LAMINA_TIP],'loc_lamina_base':[self.LAMINA_BASE],'loc_petiole_tip':[self.PETIOLE_TIP],'loc_lamina_width':[self.LAMINA_WIDTH],
                    'loc_midvein_trace':[self.MIDVEIN_TRACE],'loc_petiole_trace':[self.PETIOLE_TRACE],'loc_apex_angle':[self.APEX_ANGLE],'loc_base_angle':[self.BASE_ANGLE],'loc_deepest_sinus':[self.DEEPEST_SINUS_ANGLE],
                    'loc_tip':[self.TIP], 'loc_middle':[self.MIDDLE], 'loc_outer':[self.OUTER],
                    'loc_lobe_tip_n':[self.LOBE_TIP_N],'loc_lamina_tip_n':[self.LAMINA_TIP_N],'loc_lamina_base_n':[self.LAMINA_BASE_N],'loc_petiole_tip_n':[self.PETIOLE_TIP_N],'loc_lamina_width_n':[self.LAMINA_WIDTH_N],
                    'loc_midvein_trace_n':[self.MIDVEIN_TRACE_N],'loc_petiole_trace_n':[self.PETIOLE_TRACE_N],'loc_apex_angle_n':[self.APEX_ANGLE_N],'loc_base_angle_n':[self.BASE_ANGLE_N],'loc_deepest_sinus_n':[self.DEEPEST_SINUS_ANGLE_N],
                    'loc_tip_n':[self.TIP_N], 'loc_middle_n':[self.MIDDLE_N], 'loc_outer_n':[self.OUTER_N]}
            df = pd.DataFrame(data,headers)
            df = df.iloc[[0]]
        else: # this is the one that gets saved to csv
            headers = ['img_name','img_filename',
                    'lamina_length','midvein_length','petiole_length','petiole_trace_length','lamina_width','apex_angle','base_angle','angle_types','lobe_count','deepest_sinus',
                    'middle', 'outer',
                    'loc_lobe_tip','loc_lamina_tip','loc_lamina_base','loc_petiole_tip','loc_lamina_width','loc_midvein_trace','loc_petiole_trace','loc_apex_angle','loc_base_angle','loc_deepest_sinus',
                    'loc_tip', 'loc_middle', 'loc_outer']
            data = {'img_name':[self.IMG_NAME],'img_filename':[self.IMG_FILENAME],
                    'lamina_length':[self.M_LAMINA_LENGTH],'midvein_length':[self.M_MIDVEIN_LENGTH],'petiole_length':[self.M_PETIOLE_LENGTH],'petiole_trace_length':[self.M_PETIOLE_TRACE_LENGTH],'lamina_width':[self.M_LAMINA_WIDTH],
                    'apex_angle':[self.M_APEX_ANGLE],'base_angle':[self.M_BASE_ANGLE],'angle_types':[self.ANGLE_TYPES],'lobe_count':[self.M_LOBE_COUNT],'deepest_sinus':[self.M_DEEPEST_SINUS_ANGLE],
                    'middle':[self.M_MIDDLE], 'outer':[self.M_OUTER],
                    'loc_lobe_tip':[self.LOBE_TIP],'loc_lamina_tip':[self.LAMINA_TIP],'loc_lamina_base':[self.LAMINA_BASE],'loc_petiole_tip':[self.PETIOLE_TIP],'loc_lamina_width':[self.LAMINA_WIDTH],
                    'loc_midvein_trace':[self.MIDVEIN_TRACE],'loc_petiole_trace':[self.PETIOLE_TRACE],'loc_apex_angle':[self.APEX_ANGLE],'loc_base_angle':[self.BASE_ANGLE],'loc_deepest_sinus':[self.DEEPEST_SINUS_ANGLE],
                    'loc_tip':[self.TIP], 'loc_middle':[self.MIDDLE], 'loc_outer':[self.OUTER]}
            df = pd.DataFrame(data,headers)
            df = df.iloc[[0]]

        return df
    
    

    
def resize_trace(trace, target_size=30, method='midpoint'):
    if method == 'midpoint':
        return midpoint_trace(trace, target_size)
    elif method == 'uniform':
        return uniformly_respace_trace(trace, target_size)
    elif method == 'curvature':
        return curvature_respace_trace(trace, target_size)
    else:
        raise ValueError("Invalid method")
    
def midpoint_trace(trace, target_size=30):
    n = len(trace)

    # Do nothing if the trace is already the target size
    if n == target_size:
        return trace
    
    # If the trace is larger than the target size, remove points (except first and last)
    elif n > target_size:
        indices_to_remove = random.sample(range(1, n-1), n - target_size)
        indices_to_remove.sort(reverse=True)
        for index in indices_to_remove:
            del trace[index]
        return trace

    # If the trace is smaller than the target size, add points
    else:
        # Calculate distances between each pair of points
        distances = [(euclidean_distance(trace[i], trace[i+1]), i) for i in range(len(trace) - 1)]

        # Sort the segments by their lengths, in descending order
        distances.sort(key=lambda x: x[0], reverse=True)

        while len(trace) < target_size:
            # Get the index of the largest distance
            _, index = distances[0]
            
            # Calculate midpoint
            x1, y1 = trace[index]
            x2, y2 = trace[index + 1]
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            # Insert midpoint into trace
            trace.insert(index + 1, (mid_x, mid_y))

            # Update the distances list
            new_dist1 = euclidean_distance(trace[index], trace[index + 1])
            new_dist2 = euclidean_distance(trace[index + 1], trace[index + 2])
            distances[0] = (new_dist1, index)
            distances.append((new_dist2, index + 1))
            
            # Re-sort the distances
            distances.sort(key=lambda x: x[0], reverse=True)
        
        return trace[:target_size]  # Truncate or extend to target size

    
def euclidean_distance(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def uniformly_respace_trace(trace, target_size=30):
    # applyu midpoint
    trace = midpoint_trace(trace, target_size)
    # Compute total length of trace
    total_length = sum(euclidean_distance(trace[i], trace[i+1]) for i in range(len(trace) - 1))
    
    # Calculate the interval length
    interval_length = total_length / (target_size - 1)

    new_trace = [trace[0]]
    last_point = trace[0]
    remaining_length = interval_length

    for i in range(1, len(trace)):
        segment_length = euclidean_distance(trace[i], last_point)
        
        while segment_length >= remaining_length:
            # Compute the next point on the trace
            ratio = remaining_length / segment_length
            x_new = last_point[0] + ratio * (trace[i][0] - last_point[0])
            y_new = last_point[1] + ratio * (trace[i][1] - last_point[1])
            
            new_trace.append((x_new, y_new))
            
            # Prepare for the next point
            last_point = (x_new, y_new)
            segment_length -= remaining_length
            remaining_length = interval_length
            
        remaining_length -= segment_length
        last_point = trace[i]
        
    return new_trace

def curvature(x1, y1, x2, y2, x3, y3):
    a = euclidean_distance((x1, y1), (x2, y2))
    b = euclidean_distance((x2, y2), (x3, y3))
    c = euclidean_distance((x1, y1), (x3, y3))
    
    # Using Heron's formula to find the area of the triangle
    s = (a + b + c) / 2
    area = math.sqrt(s * (s - a) * (s - b) * (s - c))
    
    # Computing curvature
    k = 4 * area / (a * b * c)
    return k

def curvature_respace_trace(trace, target_size=30):
    n = len(trace)
    curvatures = [0]  # No curvature at start
    
    for i in range(1, n - 1):
        x1, y1 = trace[i - 1]
        x2, y2 = trace[i]
        x3, y3 = trace[i + 1]
        curvatures.append(curvature(x1, y1, x2, y2, x3, y3))

    curvatures.append(0)  # No curvature at end

    # Normalize curvatures to make it a probability distribution
    total_curvature = sum(curvatures)
    if total_curvature == 0:
        return uniformly_respace_trace(trace, target_size)
    
    curvatures = [c / total_curvature for c in curvatures]

    # Calculate the number of points to allocate to each segment based on curvature
    points_to_allocate = [round(c * (target_size - 1)) for c in curvatures]
    total_allocated = sum(points_to_allocate)

    # Adjust the point allocation to ensure exactly target_size points
    while total_allocated < target_size - 1:
        max_curvature_index = curvatures.index(max(curvatures))
        points_to_allocate[max_curvature_index] += 1
        total_allocated += 1
    
    while total_allocated > target_size - 1:
        min_curvature_index = curvatures.index(min(curvatures))
        points_to_allocate[min_curvature_index] -= 1
        total_allocated -= 1

    # Generate the new trace with points distributed based on curvature
    new_trace = []
    for i in range(n - 1):
        x1, y1 = trace[i]
        x2, y2 = trace[i + 1]
        num_points = points_to_allocate[i]

        for j in range(num_points + 1):
            t = j / (num_points + 1)
            x_new = x1 + t * (x2 - x1)
            y_new = y1 + t * (y2 - y1)
            new_trace.append((x_new, y_new))

    return new_trace[:target_size]  # Truncate or extend to target size


def overlay_trace_on_image_lines(image, corners_ordered, width, height, p_width=30):
    draw = ImageDraw.Draw(image)
    color = (255, 0, 0)

    # Draw lines explicitly between each corner
    top_left = (int(corners_ordered[0][0] * width), int(corners_ordered[0][1] * height))
    top_right = (int(corners_ordered[1][0] * width), int(corners_ordered[1][1] * height))
    bottom_left = (int(corners_ordered[2][0] * width), int(corners_ordered[2][1] * height))
    bottom_right = (int(corners_ordered[3][0] * width), int(corners_ordered[3][1] * height))
    
    draw.line([top_left, top_right], fill=color, width=p_width)
    draw.line([top_right, bottom_right], fill=color, width=p_width)
    draw.line([bottom_right, bottom_left], fill=color, width=p_width)
    draw.line([bottom_left, top_left], fill=color, width=p_width)

    return image


def overlay_trace_on_image(image, traces, label, width, height, shade_factor=False, style='visible'):
    draw = ImageDraw.Draw(image)
    p_width = 30

    # Color mapping dictionary
    if style == 'hidden':
        color_map = {
            'top_left': (255, 0, 0),
            'top_right': (255, 0, 0),
            'bottom_right': (255, 0, 0),
            'bottom_left': (255, 0, 0),
        }
    else:
        color_map = {
            'top_left': (0, 255, 255),
            'top_right': (0, 255, 255),
            'bottom_right': (0, 255, 255),
            'bottom_left': (0, 255, 255),
        }

    color = color_map.get(label, (0, 0, 0))

    # Draw lines between the corners to form a rectangle
    for i in range(len(traces)):
        start_point = (int(traces[i][0] * width), int(traces[i][1] * height))
        end_point = (int(traces[(i + 1) % len(traces)][0] * width), int(traces[(i + 1) % len(traces)][1] * height))
        draw.line([start_point, end_point], fill='red', width=p_width)

    for i, (x, y) in enumerate(traces):
        x_pixel = int(x * width)
        y_pixel = int(y * height)
        try:
            draw.ellipse((x_pixel-p_width, y_pixel-p_width, x_pixel+p_width, y_pixel+p_width), fill=color)
        except Exception as e:
            print(f"An error occurred while drawing the ellipse at point ({x_pixel}, {y_pixel}) with color {color}.")
            print(f"Error details: {e}")
    return image


def overlay_trace_on_image_petiole(image, traces, label, width, height):
    draw = ImageDraw.Draw(image)
    p_width = 3

    # Color mapping dictionary
    color_map = {
        'first': (255, 0, 0),
        'second': (0, 255, 255),
    }

    color = color_map.get(label, (0, 0, 0))

    for i, (x, y) in enumerate(traces):
        try:
            draw.ellipse((x-p_width, y-p_width, x+p_width, y+p_width), fill=color)
        except Exception as e:
            print(f"An error occurred while drawing the ellipse at point ({x}, {y}) with color {color}.")
            print(f"Error details: {e}")
    return image

def overlay_trace_on_image_bbox(image, bbox):
    draw = ImageDraw.Draw(image)
    p_width = 5

    # Draw the bounding box if provided
    if bbox:
        top = int(bbox['top'])
        left = int(bbox['left'])
        h = int(bbox['height'])
        w = int(bbox['width'])
        right = left + w
        bottom = top + h
        box_color = (0, 255, 0)  # Green color for bounding box
        try:
            draw.rectangle([(int(left), int(top)), (int(right), int(bottom))], outline=box_color, width=p_width)
        except Exception as e:
            print(f"An error occurred while drawing the bounding box from ({left}, {top}) to ({right}, {bottom}) with color {box_color}.")
            print(f"Error details: {e}")

    return image

def save_image(image, dir_name, img_name):
    # Create the directory if it doesn't exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Create the full path for the image
    img_path = os.path.join(dir_name, img_name)

    # Save the image
    image.save(img_path, 'JPEG')

# def curvature_respace_trace(trace, target_size=30):
#     trace = midpoint_trace(trace, target_size)

#     n = len(trace)
#     curvatures = [0]  # No curvature at start
    
#     for i in range(1, n-1):
#         x1, y1 = trace[i-1]
#         x2, y2 = trace[i]
#         x3, y3 = trace[i+1]
#         curvatures.append(curvature(x1, y1, x2, y2, x3, y3))
    
#     curvatures.append(0)  # No curvature at end

#     # Normalize curvatures to make it a probability distribution
#     total_curvature = sum(curvatures)
#     if total_curvature == 0:
#         return uniformly_respace_trace(trace, target_size)
    
#     curvatures = [c / total_curvature for c in curvatures]

#     # Allocate points based on curvature
#     new_trace = []
#     for i in range(target_size):
#         index = random.choices(range(n), curvatures)[0]
#         new_trace.append(trace[index])
    
#     return new_trace

def sort_corners(corners_unordered):
    # Sort points by y-coordinate (ascending), then by x-coordinate (ascending)
    corners_sorted = sorted(corners_unordered, key=lambda x: (x[1], x[0]))
    
    # Determine the top and bottom points
    top_points = corners_sorted[:2]
    bottom_points = corners_sorted[2:]
    
    # Sort top points by x-coordinate to get top_left and top_right
    top_left, top_right = sorted(top_points, key=lambda x: x[0])
    
    # Sort bottom points by x-coordinate to get bottom_left and bottom_right
    bottom_left, bottom_right = sorted(bottom_points, key=lambda x: x[0])
    
    # Return the ordered points
    return [top_left, top_right, bottom_left, bottom_right]

def get_point_locations(project_data,label_path,partition):
    headers = ['filename','count','locations']
    locs = ['loc_lobe_tip','loc_lamina_tip','loc_lamina_base','loc_petiole_tip','loc_lamina_width','loc_midvein_trace','loc_petiole_trace','loc_apex_angle','loc_base_angle','loc_deepest_sinus','loc_tip', 'loc_middle', 'loc_outer']
    pt_type = ['lobe_tip','lamina_tip','lamina_base','petiole_tip','lamina_width','midvein_trace','petiole_trace','apex_angle','base_angle','deepest_sinus', 'tip', 'middle', 'outer']
    locs_counts = ['loc_lobe_tip_n','loc_lamina_tip_n','loc_lamina_base_n','loc_petiole_tip_n','loc_lamina_width_n','loc_midvein_trace_n','loc_petiole_trace_n','loc_apex_angle_n','loc_base_angle_n','loc_deepest_sinus_n','loc_tip_n', 'loc_middle_n', 'loc_outer_n']
    
    for i, loc in enumerate(locs):
        # try: 
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
        if pt_type[i] in ['lamina_tip','lamina_base','petiole_tip','tip']:
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
        # except:
        #     print(f"{bcolors.FAIL}\n      Failed to export {loc}{bcolors.ENDC}")

def convert_to_yolo_format(pt_x, pt_y, img_width, img_height):
    yolo_x = pt_x / img_width
    yolo_y = pt_y / img_height
    
    return yolo_x, yolo_y

def remove_duplicates_preserving_order(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def remove_extension(filename):
    return '.'.join(filename.split('.')[:-1]) if '.' in filename else filename

def write_to_txt_file(label_path, keypoints):    
    # Open file in write mode
    with open(label_path, "w") as f:
        # Write the first line (zero)
        f.write("0 ")
        
        # Write the second line (bounding box)
        f.write("0.5 0.5 1 1 ")  # whole image in YOLO format

        # Write the third line (keypoints)
        for point_list in keypoints:          
            # Convert each point to YOLO format and append it to a list
            yolo_keypoints = []
            for point in point_list:
                x, y, flag = point
                yolo_keypoints.append(f"{x} {y} {flag} ")
            
            # Write all YOLO keypoints in the third line, separated by spaces
            f.write(" ".join(yolo_keypoints))

def reorder_trace_based_on_reference_point(temp_resized_trace, ref_tip, ref_angle, Labels, loc):
    reordered_trace = temp_resized_trace  # Default to the input trace if no reordering is possible

    ref_1 = None
    lab_1 = None

    if loc == 'apex':
        if hasattr(Labels, 'LAMINA_TIP') and Labels.LAMINA_TIP:
            ref_1 = Labels.LAMINA_TIP[0]
        if hasattr(Labels, 'APEX_ANGLE'):
            lab_1 = Labels.APEX_ANGLE
    elif loc == 'base':
        if hasattr(Labels, 'LAMINA_BASE') and Labels.LAMINA_BASE:
            ref_1 = Labels.LAMINA_BASE[0]
        if hasattr(Labels, 'BASE_ANGLE'):
            lab_1 = Labels.BASE_ANGLE

    # Check for a valid reference point
    if temp_resized_trace and (ref_tip or (ref_angle and len(lab_1) >= 2)):
        ref_tip = ref_1 if ref_tip else None
        ref_angle = lab_1[1] if ref_angle and len(lab_1) >= 2 else None
        ref_point = ref_tip if ref_tip else ref_angle

        if ref_point:
            first_point = temp_resized_trace[0]
            last_point = temp_resized_trace[-1]
            
            distance_first = (first_point[0] - ref_point[0])**2 + (first_point[1] - ref_point[1])**2
            distance_last = (last_point[0] - ref_point[0])**2 + (last_point[1] - ref_point[1])**2

            if distance_last < distance_first:
                reordered_trace = list(reversed(temp_resized_trace))
                print(f"*** {loc} trace order was swapped ***")

    return reordered_trace


def export_points(opt, cfg):
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
                        dir_GT_overlay = os.path.join(opt.DIR_DATASETS,'groundtruth_overlay')
                        validate_dir(saveNameJSON_YOLO_data)
                        validate_dir(saveNameJSON_YOLO_label)
                        validate_dir(dir_GT_overlay)
                    else:
                        # Define JSON name
                        saveNameJSON_LB = '.'.join([os.path.join(opt.DIR_JSON,opt.PROJECT_NAME,project.name), 'json'])
                        # Define JSON name, YOLO
                        saveNameJSON_YOLO = os.path.join(opt.DIR_DATASETS,opt.PROJECT_NAME,project.name)
                        saveNameJSON_YOLO_data = os.path.join(opt.DIR_DATASETS,opt.PROJECT_NAME,project.name,'data')
                        saveNameJSON_YOLO_label = os.path.join(opt.DIR_DATASETS,opt.PROJECT_NAME,project.name,'images')
                        dir_GT_overlay = os.path.join(opt.DIR_DATASETS,opt.PROJECT_NAME,project.name,'groundtruth_overlay')
                        validate_dir(os.path.join(opt.DIR_JSON,opt.PROJECT_NAME))
                        validate_dir(saveNameJSON_YOLO_data)
                        validate_dir(saveNameJSON_YOLO_label)
                        validate_dir(dir_GT_overlay)

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
                        # # labels = project.export()
                        # try:
                        #     jsonFile = requests.get(labels) 
                        # except:
                        #     try:
                        #         time.sleep(30)
                        #         # labels = project.export()
                        #         labels = project.export_v2(params=params)
                        #         jsonFile = requests.get(labels)
                        #     except: 
                        #         time.sleep(30)
                        #         # labels = project.export()
                        #         labels = project.export_v2(params=params)
                        #         jsonFile = requests.get(labels) 
                        # jsonFile = jsonFile.json()
                        # print(jsonFile)



                        '''
                        Save JSON file in labelbox format
                        '''
                        # validate_dir(os.path.abspath(os.path.join(saveDir_LB,annoType)))
                        
                        
                        # with open(saveNameJSON_LB, 'w', encoding='utf-8') as f:
                        #     json.dump(jsonFile, f, ensure_ascii=False)







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


                            csv_file_path = os.path.join(opt.DIR_DATASETS, 'output_distances.csv')
                            # Initialize the CSV file with headers
                            with open(csv_file_path, mode='w', newline='') as csv_file:
                                csv_writer = csv.writer(csv_file)
                                csv_writer.writerow(['image_name', 'leaf_id', 'pixel_distance'])

                            for img_data in tqdm(data, desc=f'{bcolors.BOLD}      Converting  >>>  {file}{bcolors.ENDC}',colour=cc,position=0):
                                im_path = img_data['data_row']['row_data']
                                external_id = img_data['data_row']['external_id']
                                project_id = img_data['data_row']['id']
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
                                imOriginal = im.copy()
                                width, height = im.size  # image size
                                fname = Path(external_id).with_suffix('.txt').name
                                label_path_train = os.path.join(saveNameJSON_YOLO,'labels','train')
                                images_path_train = os.path.join(saveNameJSON_YOLO,'images','train')
                                label_path_val = os.path.join(saveNameJSON_YOLO,'labels','val')
                                images_path_val = os.path.join(saveNameJSON_YOLO,'images','val')
                                label_path_test = os.path.join(saveNameJSON_YOLO,'labels','test')
                                images_path_test = os.path.join(saveNameJSON_YOLO,'images','test')
                                
                                label_path_train_petiole = os.path.join(saveNameJSON_YOLO,'labels_petiole','train')
                                label_path_val_petiole = os.path.join(saveNameJSON_YOLO,'labels_petiole','val')
                                label_path_test_petiole = os.path.join(saveNameJSON_YOLO,'labels_petiole','test')
                                label_path_test_petiole = os.path.join(saveNameJSON_YOLO,'labels_petiole','test')

                                path_cropped = os.path.join(opt.DIR_DATASETS, 'cropped_leaves')

                                if opt.DO_PARTITION_DATA:
                                    
                                    validate_dir(label_path_train)
                                    validate_dir(label_path_val)
                                    validate_dir(label_path_test)
                                    validate_dir(images_path_train)
                                    validate_dir(images_path_val)
                                    validate_dir(images_path_test)

                                    validate_dir(label_path_train_petiole)
                                    validate_dir(label_path_val_petiole)
                                    validate_dir(label_path_test_petiole)
                                    validate_dir(path_cropped)

                                    # image_path = os.path.join(saveNameJSON_YOLO_label, img['External ID'])
                                    # im.save(Path(image_path).with_suffix('.jpg'), quality=100, subsampling=0)
                                    if pc in TRAIN:
                                        # label_path = os.path.join(saveNameJSON_YOLO,'labels','train',fname)
                                        image_path = os.path.join(saveNameJSON_YOLO,'images','train',external_id)
                                        label_path = os.path.join(saveNameJSON_YOLO,'labels','train',fname)
                                        label_path_petiole = os.path.join(saveNameJSON_YOLO,'labels_petiole','train',fname)
                                        im.save(Path(image_path).with_suffix('.jpg'), quality=100, subsampling=0) # WW edited this line; added Path(image_path).with_suffix('.jpg')
                                    elif pc in VAL:
                                        # label_path = os.path.join(saveNameJSON_YOLO,'labels','val',fname)
                                        image_path = os.path.join(saveNameJSON_YOLO,'images','val',external_id)
                                        label_path = os.path.join(saveNameJSON_YOLO,'labels','val',fname)
                                        label_path_petiole = os.path.join(saveNameJSON_YOLO,'labels_petiole','val',fname)
                                        im.save(Path(image_path).with_suffix('.jpg'), quality=100, subsampling=0) # WW edited this line; added Path(image_path).with_suffix('.jpg')
                                    elif pc in TEST:
                                        # label_path = os.path.join(saveNameJSON_YOLO,'labels','test',fname)
                                        image_path = os.path.join(saveNameJSON_YOLO,'images','test',external_id)
                                        label_path = os.path.join(saveNameJSON_YOLO,'labels','test',fname)
                                        label_path_petiole = os.path.join(saveNameJSON_YOLO,'labels_petiole','test',fname)
                                        im.save(Path(image_path).with_suffix('.jpg'), quality=100, subsampling=0) # WW edited this line; added Path(image_path).with_suffix('.jpg')
                                else:
                                    validate_dir(label_path_train)
                                    validate_dir(images_path_train)

                                    label_path = os.path.join(saveNameJSON_YOLO,'labels','train',fname)
                                    label_path_petiole = os.path.join(saveNameJSON_YOLO,'labels_petiole','train',fname)
                                    image_path = os.path.join(saveNameJSON_YOLO,'images','train',external_id)
                                    im.save(Path(image_path).with_suffix('.jpg'), quality=100, subsampling=0) # WW edited this line; added Path(image_path).with_suffix('.jpg')
                                
                                img_name = external_id
                                img_filename = img_name #+'.jpg'
                                save_crop_name_base = remove_extension(img_name)
                                
                                keypoint_mapper = KeypointMapping(trace_version = 'petiole_width') # mid30_pet10 or 
                                Labels = Points(IMG_NAME=img_name,IMG_FILENAME=img_filename,VERSION = 'petiole_width')

                                

                                for proj in img_data['projects']:
                                    leaf_ind = 0
                                    width_ind = 0

                                    # Accumulate distances for each project
                                    distances_data = []

                                    if img_data['projects'][proj]['labels']:
                                        for label in img_data['projects'][proj]['labels'][0]['annotations']['objects']:



                                            if label['value'] == 'width':
                                                width_ind += 1
                                                # Will go to label_path_petiole
                                                L = label['line']

                                                line_0 = L[0]
                                                line_1 = L[1]

                                                line_0_x = line_0['x']
                                                line_0_y = line_0['y']

                                                line_1_x = line_1['x']
                                                line_1_y = line_1['y']

                                                # Calculate Euclidean distance
                                                distance = math.sqrt((line_1_x - line_0_x) ** 2 + (line_1_y - line_0_y) ** 2)

                                                print("Euclidean distance:", distance)

                                                # Append distance data to the list for later saving
                                                save_crop_name = save_crop_name_base + '__' + str(width_ind)
                                                distances_data.append([save_crop_name_base, save_crop_name, distance])

                                                im = overlay_trace_on_image_petiole(im, [(line_0_x, line_0_y)], 'first', width, height)
                                                im = overlay_trace_on_image_petiole(im, [(line_1_x, line_1_y)], 'second', width, height)
                                                

                                            if label['value'] == 'leaf':
                                                leaf_ind += 1
                                                # will go to   label_path

                                                bbox = label['bounding_box']  # top, left, height, width
                                                top = int(bbox['top'])
                                                left = int(bbox['left'])
                                                h = int(bbox['height'])
                                                w = int(bbox['width'])
                                                xywh = [(left + w / 2) / width, (top + h / 2) / height, w / width, h / height]  # xywh normalized

                                                im = overlay_trace_on_image_bbox(im, bbox)

                                                # class
                                                cls = label['value']  # class name
                                                if cls not in names:
                                                    names.append(cls)
                                                
                                                # set the index based on the order of the annotations in LabelBox
                                                annoInd = assign_index(cls,'petiole_width')

                                                line = annoInd, *xywh  # YOLO format (class_index, xywh)
                                                with open(label_path, 'a') as f:
                                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')


                                                # rect_points = label.value.geometry['coordinates'][0]
                                                right = left + w
                                                bottom = top + h
                                                img_crop = imOriginal.crop((left, top, right, bottom))

                                                if opt.DO_PARTITION_DATA:
                                                    if pc in TRAIN:
                                                        save_crop_name = save_crop_name_base + '__' + str(leaf_ind)
                                                        save_crop_dir = os.path.join(path_cropped, cls, 'train')
                                                        save_image(img_crop, save_crop_dir, save_crop_name+'.jpg')
                                                    if pc in VAL:
                                                        save_crop_name = save_crop_name_base + '__' + str(leaf_ind)
                                                        save_crop_dir = os.path.join(path_cropped, cls, 'val')
                                                        save_image(img_crop, save_crop_dir, save_crop_name+'.jpg')
                                                    if pc in TEST:
                                                        save_crop_name = save_crop_name_base + '__' + str(leaf_ind)
                                                        save_crop_dir = os.path.join(path_cropped, cls, 'test')
                                                        save_image(img_crop, save_crop_dir, save_crop_name+'.jpg')
                                                else:
                                                    save_crop_name = save_crop_name_base + '__' + str(leaf_ind)
                                                    save_crop_dir = os.path.join(path_cropped, cls)
                                                    save_image(img_crop, save_crop_dir, save_crop_name+'.jpg')



                                        save_image(im, dir_GT_overlay, img_name+'.jpg')

                                        # Append the accumulated distances to the CSV file
                                        with open(csv_file_path, mode='a', newline='') as csv_file:
                                            csv_writer = csv.writer(csv_file)
                                            csv_writer.writerows(distances_data)
                                            
                                        pc += 1

                            # project_data.to_csv(project_data_path,index=False)

                            # get_point_locations(project_data_counts_train,saveNameJSON_YOLO_label,'train')
                            # get_point_locations(project_data_counts_val,saveNameJSON_YOLO_label,'val')
                            # get_point_locations(project_data_counts_test,saveNameJSON_YOLO_label,'test')
                            

                        

def export_points_labels():
    # Read configs
    dir_private = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path_cfg_private = os.path.join(dir_private,'PRIVATE_DATA.yaml')
    cfg_private = get_cfg_from_full_path(path_cfg_private)

    dir_labeling = os.path.dirname(__file__)
    path_cfg = os.path.join(dir_labeling,'config_export_petiole_width_labels_from_Labelbox.yaml')
    cfg = get_cfg_from_full_path(path_cfg)


    LB_API_KEY = cfg_private['labelbox']['LB_API_KEY']
    client = Client(api_key=LB_API_KEY)

    opt = OPTS_EXPORT_POINTS(cfg, client)

    validate_dir(opt.DIR_ROOT)
    validate_dir(opt.DIR_DATASETS)   
    validate_dir(opt.DIR_JSON)   

    print(f"{bcolors.HEADER}Beginning Export for Project {opt.PROJECT_NAME}{bcolors.ENDC}")
    print(f"{bcolors.BOLD}      Labels will go to --> {opt.DIR_DATASETS}{bcolors.ENDC}")

    export_points(opt, cfg)
    print(f"{bcolors.OKGREEN}Finished Export :){bcolors.ENDC}")

if __name__ == '__main__':
    export_points_labels()                      