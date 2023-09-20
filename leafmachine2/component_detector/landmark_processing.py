import os, math, cv2, random
import numpy as np
from itertools import combinations
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass()
class LeafSkeleton:
    cfg: str 
    Dirs: str 
    leaf_type: str 
    all_points: list
    dir_temp: str
    file_name: str
    width: int
    height: int
    logger: object

    do_show_QC_images: bool = False
    do_save_QC_images: bool = False

    classes: float = None
    points_list: float = None

    image: float = None

    ordered_midvein: float = None
    midvein_fit: float = None
    midvein_fit_points: float = None
    ordered_midvein_length: float = None
    has_midvein = False

    is_split = False

    ordered_petiole: float = None
    ordered_petiole_length: float = None
    has_ordered_petiole = False

    has_apex: bool = False
    apex_left: float = None
    apex_right: float = None
    apex_center: float = None
    apex_angle_type: str = 'NA'
    apex_angle_degrees: float = None

    has_base: bool = False
    base_left: float = None
    base_right: float = None
    base_center: float = None
    base_angle_type: str = 'NA'
    base_angle_degrees: float = None

    has_lamina_tip: bool = False
    lamina_tip: float = None

    has_lamina_base: bool = False
    lamina_base: float = None

    has_lamina_length: bool = False
    lamina_fit: float = None
    lamina_length: float = None

    has_width: bool = False
    lamina_width: float = None
    width_left: float = None
    width_right: float = None

    has_lobes: bool = False
    lobe_count: float = None
    lobes: float = None

    def __init__(self, cfg, logger, Dirs, leaf_type, all_points, height, width, dir_temp, file_name) -> None:
        # Store the necessary arguments as instance attributes
        self.cfg = cfg
        self.Dirs = Dirs
        self.leaf_type = leaf_type
        self.all_points = all_points
        self.height = height
        self.width = width
        self.dir_temp = dir_temp
        self.file_name = file_name

        logger.name = f'[{leaf_type} - {file_name}]'
        self.logger = logger

        self.init_lists_dicts()

        # Setup
        self.set_cfg_values()
        self.define_landmark_classes()

        self.setup_QC_image()
        self.setup_final_image()

        self.parse_all_points()
        self.convert_YOLO_bbox_to_point()

        # Start with ordering the midvein and petiole
        self.order_midvein()
        self.order_petiole()
        # print(self.ordered_midvein)
        
        # Split the image using the midvein IF has_midvein == True
        self.split_image_by_midvein()

        # Process angles IF is_split == True. Need orientation to pick the appropriate pts for angle calcs
        self.determine_apex()
        self.determine_base()

        self.determine_lamina_tip()
        self.determine_lamina_base()
        self.determine_lamina_length('QC')

        self.determine_width()

        self.determine_lobes()
        self.determine_petiole() # straight length of petiole vs. ordered_petiole length which is tracing the petiole

        self.restrictions()

        # creates self.is_complete_leaf = False and self.is_leaf_no_width = False
        # can add less restrictive options later, but for now only very complete leaves will pass
        self.redo_measurements()

        self.create_final_image() 
        self.translate_measurements_to_full_image()

        self.show_QC_image()
        self.show_final_image()
        # self.save_QC_image()
        # print('hi')

    def get(self, attribute, default=None):
        return getattr(self, attribute, default)

    def split_image_by_midvein(self):
        
        if self.has_midvein:
            n_fit = 1

            # Convert the points to a numpy array
            points_arr = np.array(self.ordered_midvein)

            # Fit a line to the points
            self.midvein_fit = np.polyfit(points_arr[:, 0], points_arr[:, 1], n_fit)

            if len(self.midvein_fit) < 1:
                self.midvein_fit = None
            else:
                # Plot a sample of points from along the line
                max_dim = max(self.height, self.width)
                if max_dim < 400:
                    num_points = 40
                elif max_dim < 1000:
                    num_points = 80
                else:
                    num_points = 120

                # Get the endpoints of the line segment that lies within the bounds of the image
                x1 = 0
                y1 = int(self.midvein_fit[0] * x1 + self.midvein_fit[1])
                x2 = self.width - 1
                y2 = int(self.midvein_fit[0] * x2 + self.midvein_fit[1])
                denom = self.midvein_fit[0]
                if denom == 0:
                    denom = 0.0000000001
                if y1 < 0:
                    y1 = 0
                    x1 = int((y1 - self.midvein_fit[1]) / denom)
                if y2 >= self.height:
                    y2 = self.height - 1
                    x2 = int((y2 - self.midvein_fit[1]) / denom)

                # Sample num_points points along the line segment within the bounds of the image
                x_vals = np.linspace(x1, x2, num_points)
                y_vals = self.midvein_fit[0] * x_vals + self.midvein_fit[1]

                # Remove any points that are outside the bounds of the image
                indices = np.where((y_vals >= 0) & (y_vals < self.height))[0]
                x_vals = x_vals[indices]
                y_vals = y_vals[indices]

                # Recompute y-values using the line equation and updated x-values
                y_vals = self.midvein_fit[0] * x_vals + self.midvein_fit[1]

                self.midvein_fit_points = np.column_stack((x_vals, y_vals))
                self.is_split = True

                # Draw line of fit
                for point in self.midvein_fit_points:
                    cv2.circle(self.image, tuple(point.astype(int)), radius=1, color=(255, 255, 255), thickness=-1)


    '''def split_image_by_midvein(self): # cubic
        if self.file_name == 'B_774373024_Ebenaceae_Diospyros_glutinifera__L__469-164-888-632':
            print('hi')
        if self.has_midvein:
            n_fit = 3

            # Convert the points to a numpy array
            points_arr = np.array(self.ordered_midvein)

            # Fit a curve to the points
            self.midvein_fit = np.polyfit(points_arr[:, 0], points_arr[:, 1], n_fit)

            # Plot a sample of points from along the curve
            max_dim = max(self.height, self.width)
            if max_dim < 400:
                num_points = 40
            elif max_dim < 1000:
                num_points = 80
            else:
                num_points = 120

            # Get the endpoints of the curve segment that lies within the bounds of the image
            x1 = 0
            y1 = int(self.midvein_fit[0] * x1**3 + self.midvein_fit[1] * x1**2 + self.midvein_fit[2] * x1 + self.midvein_fit[3])
            x2 = self.width - 1
            y2 = int(self.midvein_fit[0] * x2**3 + self.midvein_fit[1] * x2**2 + self.midvein_fit[2] * x2 + self.midvein_fit[3])

            # Sample num_points y-values that are evenly spaced within the bounds of the image
            y_vals = np.linspace(0, self.height - 1, num_points)

            # Compute the corresponding x-values using the polynomial
            p = np.poly1d(self.midvein_fit)
            x_vals = np.zeros(num_points)
            for i, y in enumerate(y_vals):
                roots = p - y
                real_roots = roots.r[np.isreal(roots.r)].real
                x_val = real_roots[(real_roots >= 0) & (real_roots < self.width)]
                if len(x_val) > 0:
                    x_vals[i] = x_val[0]

            # Remove any points that are outside the bounds of the image
            indices = np.where((y_vals > 0) & (y_vals < self.height-1))[0]
            x_vals = x_vals[indices]
            y_vals = y_vals[indices]

            # Recompute y-values using the polynomial and updated x-values
            y_vals = self.midvein_fit[0] * x_vals**3 + self.midvein_fit[1] * x_vals**2 + self.midvein_fit[2] * x_vals + self.midvein_fit[3]

            self.midvein_fit_points = np.column_stack((x_vals, y_vals))
            self.is_split = True'''




        


        
    def determine_apex(self):
        if self.is_split:
            can_get_angle = False
            if 'apex_angle' in self.points_list:
                if 'lamina_tip' in self.points_list:
                    self.apex_center, self.points_list['apex_angle'] = self.get_closest_point_to_sampled_points(self.points_list['apex_angle'], self.points_list['lamina_tip'])
                    can_get_angle = True
                elif self.midvein_fit_points.shape[0] > 0:
                    self.apex_center, self.points_list['apex_angle'] = self.get_closest_point_to_sampled_points(self.points_list['apex_angle'], self.midvein_fit_points)
                    can_get_angle = True
                
                if can_get_angle:
                    left = []
                    right = []
                    for point in self.points_list['apex_angle']:
                        loc = self.point_position_relative_to_line(point, self.midvein_fit)
                        if loc == 'right':
                            right.append(point)
                        elif loc == 'left':
                            left.append(point)
                    
                    if (left == []) or (right == []):
                        self.has_apex = False
                        if (left == []) and (right != []):
                            self.apex_right, right = self.get_far_point(right, self.apex_center)
                            self.apex_left = None
                        elif (right == []) and (left != []):
                            self.apex_left, left = self.get_far_point(left, self.apex_center)
                            self.apex_right = None
                        else:
                            self.apex_left = None
                            self.apex_right = None
                    else:
                        self.has_apex = True
                        self.apex_left, left = self.get_far_point(left, self.apex_center)
                        self.apex_right, right = self.get_far_point(right, self.apex_center)

                    # print(self.points_list['apex_angle'])   
                    # print(f'apex_center: {self.apex_center} apex_left: {self.apex_left} apex_right: {self.apex_right}')
                    self.logger.debug(f"[apex_angle_list] {self.points_list['apex_angle']}")
                    self.logger.debug(f"[apex_center] {self.apex_center} [apex_left] {self.apex_left} [apex_right] {self.apex_right}")
                    
                    if self.has_apex:
                        self.apex_angle_type, self.apex_angle_degrees = self.determine_reflex(self.apex_left, self.apex_right, self.apex_center)
                        # print(f'angle_type {self.apex_angle_type} angle {self.apex_angle_degrees}')
                        self.logger.debug(f"[angle_type] {self.apex_angle_type} [angle] {self.apex_angle_degrees}")
                    else:
                        self.apex_angle_type = 'NA'
                        self.apex_angle_degrees = None
                        self.logger.debug(f"[angle_type] {self.apex_angle_type} [angle] {self.apex_angle_degrees}")


                    if self.has_apex:
                        if self.apex_center is not None:
                            cv2.circle(self.image, self.apex_center, radius=3, color=(0, 255, 0), thickness=-1)
                        if self.apex_left is not None:
                            cv2.circle(self.image, self.apex_left, radius=3, color=(255, 0, 0), thickness=-1)
                        if self.apex_right is not None:
                            cv2.circle(self.image, self.apex_right, radius=3, color=(0, 0, 255), thickness=-1)
        
    def determine_apex_redo(self):
        self.logger.debug(f"[apex_angle_list REDO] ")
        self.logger.debug(f"[apex_center REDO] {self.apex_center} [apex_left] {self.apex_left} [apex_right] {self.apex_right}")
        
        if self.has_apex:
            self.apex_angle_type, self.apex_angle_degrees = self.determine_reflex(self.apex_left, self.apex_right, self.apex_center)
            self.logger.debug(f"[angle_type REDO] {self.apex_angle_type} [angle] {self.apex_angle_degrees}")
        else:
            self.apex_angle_type = 'NA'
            self.apex_angle_degrees = None
            self.logger.debug(f"[angle_type REDO] {self.apex_angle_type} [angle] {self.apex_angle_degrees}")


        if self.has_apex:
            if self.apex_center is not None:
                cv2.circle(self.image, self.apex_center, radius=11, color=(0, 255, 0), thickness=2)
            if self.apex_left is not None:
                cv2.circle(self.image, self.apex_left, radius=3, color=(255, 0, 0), thickness=-1)
            if self.apex_right is not None:
                cv2.circle(self.image, self.apex_right, radius=3, color=(0, 0, 255), thickness=-1)
    
    def determine_base_redo(self):
        self.logger.debug(f"[base_angle_list REDO] ")
        self.logger.debug(f"[base_center REDO] {self.base_center} [base_left] {self.base_left} [base_right] {self.base_right}")
        
        if self.has_base:
            self.base_angle_type, self.base_angle_degrees = self.determine_reflex(self.base_left, self.base_right, self.base_center)
            self.logger.debug(f"[angle_type REDO] {self.base_angle_type} [angle] {self.base_angle_degrees}")
        else:
            self.base_angle_type = 'NA'
            self.base_angle_degrees = None
            self.logger.debug(f"[angle_type REDO] {self.base_angle_type} [angle] {self.base_angle_degrees}")


        if self.has_base:
            if self.base_center is not None:
                cv2.circle(self.image, self.base_center, radius=11, color=(0, 255, 0), thickness=2)
            if self.base_left is not None:
                cv2.circle(self.image, self.base_left, radius=3, color=(255, 0, 0), thickness=-1)
            if self.base_right is not None:
                cv2.circle(self.image, self.base_right, radius=3, color=(0, 0, 255), thickness=-1)

    def determine_base(self):
        if self.is_split:
            can_get_angle = False
            if 'base_angle' in self.points_list:
                if 'lamina_base' in self.points_list:
                    self.base_center, self.points_list['base_angle'] = self.get_closest_point_to_sampled_points(self.points_list['base_angle'], self.points_list['lamina_base'])
                    can_get_angle = True
                elif self.midvein_fit_points.shape[0] > 0:
                    self.base_center, self.points_list['base_angle'] = self.get_closest_point_to_sampled_points(self.points_list['base_angle'], self.midvein_fit_points)
                    can_get_angle = True
                
                if can_get_angle:
                    left = []
                    right = []
                    for point in self.points_list['base_angle']:
                        loc = self.point_position_relative_to_line(point, self.midvein_fit)
                        if loc == 'right':
                            right.append(point)
                        elif loc == 'left':
                            left.append(point)
                    
                    if (left == []) or (right == []):
                        self.has_base = False
                        if (left == []) and (right != []):
                            self.base_right, right = self.get_far_point(right, self.base_center)
                            self.base_left = None
                        elif (right == []) and (left != []):
                            self.base_left, left = self.get_far_point(left, self.base_center) 
                            self.base_right = None
                        else:
                            self.base_left = None
                            self.base_right = None
                    else:
                        self.has_base = True
                        self.base_left, left = self.get_far_point(left, self.base_center)
                        self.base_right, right = self.get_far_point(right, self.base_center)

                    # print(self.points_list['base_angle'])   
                    # print(f'base_center: {self.base_center} base_left: {self.base_left} base_right: {self.base_right}')
                    self.logger.debug(f"[base_angle_list] {self.points_list['base_angle']}")
                    self.logger.debug(f"[base_center] {self.base_center} [base_left] {self.base_left} [base_right] {self.base_right}")

                    
                    if self.has_base:
                        self.base_angle_type, self.base_angle_degrees = self.determine_reflex(self.base_left, self.base_right, self.base_center)
                        # print(f'angle_type {self.base_angle_type} angle {self.base_angle_degrees}')
                        self.logger.debug(f"[angle_type] {self.base_angle_type} [angle] {self.base_angle_degrees}")
                    else:
                        self.base_angle_type = 'NA'
                        self.base_angle_degrees = None
                        self.logger.debug(f"[angle_type] {self.base_angle_type} [angle] {self.base_angle_degrees}")

                    if self.has_base:
                        if self.base_center:
                            cv2.circle(self.image, self.base_center, radius=3, color=(0, 255, 0), thickness=-1)
                        if self.base_left:
                            cv2.circle(self.image, self.base_left, radius=3, color=(255, 0, 0), thickness=-1)
                        if self.base_right:
                            cv2.circle(self.image, self.base_right, radius=3, color=(0, 0, 255), thickness=-1)
    
    def determine_lamina_tip(self):
        if 'lamina_tip' in self.points_list:
            self.has_lamina_tip = True
            if self.apex_center:
                self.lamina_tip, self.lamina_tip_alternate = self.get_closest_point_to_sampled_points(self.points_list['lamina_tip'], self.apex_center)
            elif len(self.midvein_fit_points) > 0:
                self.lamina_tip, self.lamina_tip_alternate = self.get_closest_point_to_sampled_points(self.points_list['lamina_tip'], self.midvein_fit_points)
            else:
                if len(self.points_list['lamina_tip']) == 1:
                    self.lamina_tip = self.points_list['lamina_tip'][0]
                    self.lamina_tip_alternate = None
                else: # blindly choose the most "central points"
                    centroid = tuple(np.mean(self.points_list['lamina_tip'], axis=0))
                    self.lamina_tip = min(self.points_list['lamina_tip'], key=lambda p: np.linalg.norm(np.array(p) - np.array(centroid)))
                    self.lamina_tip_alternate = None # TODO finish this
            
            # if lamina_tip is closer to midvein_fit_points, then apex_center = lamina_tip
            if self.apex_center and (len(self.midvein_fit_points) > 0):
                d_apex = self.calc_min_distance(self.apex_center, self.midvein_fit_points)
                d_lamina = self.calc_min_distance(self.lamina_tip, self.midvein_fit_points)
                if d_lamina < d_apex:
                    cv2.circle(self.image, self.apex_center, radius=5, color=(255, 255, 255), thickness=3) # white hollow, indicates switch
                    cv2.circle(self.image, self.lamina_tip, radius=3, color=(0, 255, 0), thickness=-1) # repaint the point, indicates switch
                    self.apex_center = self.lamina_tip
                    if self.has_apex:
                        self.apex_angle_type, self.apex_angle_degrees = self.determine_reflex(self.apex_left, self.apex_right, self.apex_center)
        else:
            if self.apex_center:
                self.has_lamina_tip = True
                self.lamina_tip = self.apex_center
                self.lamina_tip_alternate = None
                
        if self.lamina_tip:
            cv2.circle(self.image, self.lamina_tip, radius=5, color=(255, 0, 230), thickness=2) # pink solid
            if self.lamina_tip_alternate:
                for pt in self.lamina_tip_alternate:
                    cv2.circle(self.image, pt, radius=3, color=(255, 0, 230), thickness=-1) # pink hollow

    def determine_lamina_base(self):
        if 'lamina_base' in self.points_list:
            self.has_lamina_base = True
            if self.base_center:
                self.lamina_base, self.lamina_base_alternate = self.get_closest_point_to_sampled_points(self.points_list['lamina_base'], self.base_center)
            elif len(self.midvein_fit_points) > 0:
                self.lamina_base, self.lamina_base_alternate = self.get_closest_point_to_sampled_points(self.points_list['lamina_base'], self.midvein_fit_points)
            else:
                if len(self.points_list['lamina_base']) == 1:
                    self.lamina_base = self.points_list['lamina_base'][0]
                    self.lamina_base_alternate = None
                else: # blindly choose the most "central points"
                    centroid = tuple(np.mean(self.points_list['lamina_base'], axis=0))
                    self.lamina_base = min(self.points_list['lamina_base'], key=lambda p: np.linalg.norm(np.array(p) - np.array(centroid)))
                    self.lamina_base_alternate = None     
            
            # if has_lamina_tip is closer to midvein_fit_points, then base_center = has_lamina_tip
            if self.base_center and (len(self.midvein_fit_points) > 0):
                d_base = self.calc_min_distance(self.base_center, self.midvein_fit_points)
                d_lamina = self.calc_min_distance(self.lamina_base, self.midvein_fit_points)
                if d_lamina < d_base:
                    cv2.circle(self.image, self.base_center, radius=5, color=(255, 255, 255), thickness=3) # white hollow, indicates switch
                    cv2.circle(self.image, self.lamina_base, radius=3, color=(0, 255, 0), thickness=-1) # repaint the point, indicates switch
                    self.base_center = self.lamina_base
                    if self.has_base:
                        self.base_angle_type, self.base_angle_degrees = self.determine_reflex(self.base_left, self.base_right, self.base_center)      
        else:
            if self.base_center:
                self.has_lamina_base = True
                self.lamina_base = self.base_center
                self.lamina_base_alternate = None

        if self.lamina_base:
            cv2.circle(self.image, self.lamina_base, radius=5, color=(0, 100, 255), thickness=2) # orange
            if self.lamina_base_alternate:
                for pt in self.lamina_base_alternate:
                    cv2.circle(self.image, pt, radius=3, color=(0, 100, 255), thickness=-1) # orange hollow

    def determine_lamina_length(self, QC_or_final):
        if self.has_lamina_base and self.has_lamina_tip:
            self.lamina_length = self.distance(self.lamina_base, self.lamina_tip)
            ends = np.array([self.lamina_base, self.lamina_tip])
            self.lamina_fit = np.polyfit(ends[:, 0], ends[:, 1], 1)
            self.has_lamina_length = True
            # r_base = 0
            r_base = 16
            # col = (0, 100, 0)
            col = (0, 0, 0)
            if QC_or_final == 'QC':
                cv2.line(self.image, self.lamina_base, self.lamina_tip, col, 2 + r_base)
            else:
                cv2.line(self.image_final, self.lamina_base, self.lamina_tip, col, 2 + r_base)
        else:
            col = (0, 0, 0)
            r_base = 16
            if self.has_lamina_base and (not self.has_lamina_tip) and self.has_apex: # lamina base and apex center
                self.lamina_length = self.distance(self.lamina_base, self.apex_center)
                ends = np.array([self.lamina_base, self.apex_center])
                self.lamina_fit = np.polyfit(ends[:, 0], ends[:, 1], 1)
                self.has_lamina_length = True
                if QC_or_final == 'QC':
                    cv2.line(self.image, self.lamina_base, self.apex_center, col, 2 + r_base)
                else:
                    cv2.line(self.image, self.lamina_base, self.apex_center, col, 2 + r_base)
            elif self.has_lamina_tip and (not self.has_lamina_base) and self.has_base: # lamina tip and base center
                self.lamina_length = self.distance(self.lamina_tip, self.base_center)
                ends = np.array([self.lamina_tip, self.apex_center])
                self.lamina_fit = np.polyfit(ends[:, 0], ends[:, 1], 1)
                self.has_lamina_length = True
                if QC_or_final == 'QC':
                    cv2.line(self.image, self.lamina_tip, self.apex_center, col, 2 + r_base)
                else:
                    cv2.line(self.image, self.lamina_tip, self.apex_center, col, 2 + r_base)
            elif (not self.has_lamina_tip) and (not self.has_lamina_base) and self.has_apex and self.has_base: # apex center and base center
                self.lamina_length = self.distance(self.apex_center, self.base_center)
                ends = np.array([self.base_center, self.apex_center])
                self.lamina_fit = np.polyfit(ends[:, 0], ends[:, 1], 1)
                self.has_lamina_length = True
                if QC_or_final == 'QC':
                    cv2.line(self.image, self.base_center, self.apex_center, col, 2 + r_base)
                else:
                    cv2.line(self.image, self.base_center, self.apex_center, col, 2 + r_base) # 0, 175, 200
            else:
                self.lamina_length = None
                self.lamina_fit = None
                self.has_lamina_length = False

    def determine_width(self):
        if (('lamina_width' in self.points_list) and ((self.midvein_fit is not None and len(self.midvein_fit) > 0) or (self.lamina_fit is not None))):
            left = []
            right = []
            if len(self.midvein_fit) > 0: # try using the midvein as a reference first
                for point in self.points_list['lamina_width']:
                    loc = self.point_position_relative_to_line(point, self.midvein_fit)

                    if loc == 'right':
                        right.append(point)
                    elif loc == 'left':
                        left.append(point)
            elif len(self.lamina_fit) > 0: # then try just the lamina tip/base
                for point in self.points_list['lamina_width']:
                    loc = self.point_position_relative_to_line(point, self.lamina_fit)

                    if loc == 'right':
                        right.append(point)
                    elif loc == 'left':
                        left.append(point)
            else:
                self.has_width = False
                self.width_left = None
                self.width_right = None
                self.lamina_width = None

            if (left == []) or (right == []) or not self.has_width:
                self.has_width = False
                self.width_left = None
                self.width_right = None
                self.lamina_width = None
            else:
                self.has_width = True
                if len(self.midvein_fit) > 0:
                    self.width_left, self.width_right = self.find_most_orthogonal_vectors(left, right, self.midvein_fit)
                    self.lamina_width = self.distance(self.width_left, self.width_right)
                    self.order_points_plot([self.width_left, self.width_right], 'lamina_width', 'QC')
                else: # get shortest width if the nidvein is absent for comparison
                    self.width_left, self.width_right = self.find_min_width(left, right)
                    self.lamina_width = self.distance(self.width_left, self.width_right)
                    self.order_points_plot([self.width_left, self.width_right], 'lamina_width_alt', 'QC')
        else:
            self.has_width = False
            self.width_left = None
            self.width_right = None
            self.lamina_width = None

    def determine_lobes(self):
        if 'lobe_tip' in self.points_list:
            self.has_lobes = True
            self.lobe_count = len(self.points_list['lobe_tip'])
            self.lobes = self.points_list['lobe_tip']
            for lobe in self.lobes:
                cv2.circle(self.image, tuple(lobe), radius=6, color=(0, 255, 255), thickness=3)

    def determine_petiole(self):
        if 'petiole_tip' in self.points_list:
            self.has_petiole_tip = True

            if len(self.points_list['petiole_tip']) == 1:
                self.petiole_tip = self.points_list['petiole_tip'][0]
                self.petiole_tip_alternate = None
            else: # blindly choose the most "central points"
                centroid = tuple(np.mean(self.points_list['petiole_tip'], axis=0))
                self.petiole_tip = min(self.points_list['petiole_tip'], key=lambda p: np.linalg.norm(np.array(p) - np.array(centroid)))
                self.petiole_tip_alternate = None

            # Straight length of petiole points
            if self.has_ordered_petiole:
                self.petiole_tip_opposite, self.petiole_tip_alternate = self.get_far_point(self.ordered_petiole, self.petiole_tip)
                self.petiole_length = self.distance(self.petiole_tip_opposite, self.petiole_tip)
                self.order_points_plot([self.petiole_tip_opposite, self.petiole_tip], 'petiole_tip', 'QC')
            else:
                self.petiole_tip_opposite = None
                self.petiole_length = None
            
            # Straight length of petiole tip to lamina base
            if self.lamina_base is not None:
                self.petiole_length_to_lamina_base = self.distance(self.lamina_base, self.petiole_tip)
                self.petiole_tip_opposite_alternate = self.lamina_base
                self.order_points_plot([self.petiole_tip_opposite_alternate, self.petiole_tip], 'petiole_tip_alt', 'QC')
            elif self.base_center:
                self.petiole_length_to_lamina_base = self.distance(self.base_center, self.petiole_tip)
                self.petiole_tip_opposite_alternate = self.base_center
                self.order_points_plot([self.petiole_tip_opposite_alternate, self.petiole_tip], 'petiole_tip_alt', 'QC')
            else:
                self.petiole_length_to_lamina_base = None
                self.petiole_tip_opposite_alternate = None

    def redo_measurements(self):
        if self.has_width:
            self.lamina_width = self.distance(self.width_left, self.width_right)
        
        if self.has_ordered_petiole:
            self.ordered_petiole_length, self.ordered_petiole = self.get_length_of_ordered_points(self.ordered_petiole, 'petiole_trace')

        if self.has_midvein:
            self.ordered_midvein_length, self.ordered_midvein = self.get_length_of_ordered_points(self.ordered_midvein, 'midvein_trace')

        if self.has_apex:
            self.apex_angle_type, self.apex_angle_degrees = self.determine_reflex(self.apex_left, self.apex_right, self.apex_center)
        
        if self.has_base:
            self.base_angle_type, self.base_angle_degrees = self.determine_reflex(self.base_left, self.base_right, self.base_center)

        self.determine_lamina_length('final') # Calling just in case, should already be updated

    def translate_measurements_to_full_image(self):
        loc = self.file_name.split('__')[-1]
        self.add_x = int(loc.split('-')[0])
        self.add_y = int(loc.split('-')[1])

        if self.has_base:
            self.t_base_center = [self.base_center[0] + self.add_x, self.base_center[1] + self.add_y]
            self.t_base_left = [self.base_left[0] + self.add_x, self.base_left[1] + self.add_y]
            self.t_base_right = [self.base_right[0] + self.add_x, self.base_right[1] + self.add_y]

        if self.has_apex:
            self.t_apex_center = [self.apex_center[0] + self.add_x, self.apex_center[1] + self.add_y]
            self.t_apex_left = [self.apex_left[0] + self.add_x, self.apex_left[1] + self.add_y]
            self.t_apex_right = [self.apex_right[0] + self.add_x, self.apex_right[1] + self.add_y]

        if self.has_lamina_base:
            self.t_lamina_base = [self.lamina_base[0] + self.add_x, self.lamina_base[1] + self.add_y]
        if self.has_lamina_tip:
            self.t_lamina_tip = [self.lamina_tip[0] + self.add_x, self.lamina_tip[1] + self.add_y]

        if self.has_lobes:
            self.t_lobes = []
            for point in self.lobes:
                new_x = int(point[0]) + self.add_x
                new_y = int(point[1]) + self.add_y
                new_point = [new_x, new_y]
                self.t_lobes.append(new_point)

        if self.has_midvein:
            self.t_midvein_fit_points = []
            for point in self.midvein_fit_points:
                new_x = int(point[0]) + self.add_x
                new_y = int(point[1]) + self.add_y
                new_point = [new_x, new_y]
                self.t_midvein_fit_points.append(new_point)

            self.t_midvein = []
            for point in self.ordered_midvein:
                new_x = int(point[0]) + self.add_x
                new_y = int(point[1]) + self.add_y
                new_point = [new_x, new_y]
                self.t_midvein.append(new_point)
        
        if self.has_ordered_petiole:
            self.t_petiole = []
            for point in self.ordered_petiole:
                new_x = int(point[0]) + self.add_x
                new_y = int(point[1]) + self.add_y
                new_point = [new_x, new_y]
                self.t_petiole.append(new_point)
            
        if self.has_width:
            self.t_width_left = [self.width_left[0] + self.add_x, self.width_left[1] + self.add_y]
            self.t_width_right = [self.width_right[0] + self.add_x, self.width_right[1] + self.add_y]

        if self.width_infer is not None:
            self.t_width_infer = []
            for point in self.width_infer:
                new_x = int(point[0]) + self.add_x
                new_y = int(point[1]) + self.add_y
                new_point = [new_x, new_y]
                self.t_width_infer.append(new_point)

    def create_final_image(self):
        self.is_complete_leaf = False ###########################################################################################################################################################
        self.is_leaf_no_width = False
        # r_base = 0
        r_base = 16
        if (self.has_apex and self.has_base and self.has_ordered_petiole and self.has_midvein and self.has_width):
            self.is_complete_leaf = True

            self.order_points_plot([self.width_left, self.width_right], 'lamina_width', 'final')
            self.order_points_plot(self.ordered_midvein, 'midvein_trace', 'final')
            self.order_points_plot(self.ordered_petiole, 'petiole_trace', 'final')
            self.order_points_plot([self.apex_left, self.apex_center, self.apex_right], self.apex_angle_type, 'final')
            self.order_points_plot([self.base_left, self.base_center, self.base_right], self.base_angle_type, 'final')

            self.determine_lamina_length('final') # try 
            
            
            # Lamina tip and base
            if self.has_lamina_tip:
                cv2.circle(self.image_final, self.lamina_tip, radius=4 + r_base, color=(0, 255, 0), thickness=2)
                cv2.circle(self.image_final, self.lamina_tip, radius=2 + r_base, color=(255, 255, 255), thickness=-1)
            if self.has_lamina_base:
                cv2.circle(self.image_final, self.lamina_base, radius=4 + r_base, color=(255, 0, 0), thickness=2)
                cv2.circle(self.image_final, self.lamina_base, radius=2 + r_base, color=(255, 255, 255), thickness=-1)

            # Apex angle
            # if self.apex_center != []:
            #     cv2.circle(self.image_final, self.apex_center, radius=3, color=(0, 255, 0), thickness=-1)
            if self.apex_left is not None:
                cv2.circle(self.image_final, self.apex_left, radius=3 + r_base, color=(255, 0, 0), thickness=-1)
            if self.apex_right is not None:
                cv2.circle(self.image_final, self.apex_right, radius=3 + r_base, color=(0, 0, 255), thickness=-1)

            # Base angle
            # if self.base_center:
            #     cv2.circle(self.image_final, self.base_center, radius=3, color=(0, 255, 0), thickness=-1)
            if self.base_left:
                cv2.circle(self.image_final, self.base_left, radius=3 + r_base, color=(255, 0, 0), thickness=-1)
            if self.base_right:
                cv2.circle(self.image_final, self.base_right, radius=3 + r_base, color=(0, 0, 255), thickness=-1)

            # Lobes
            if self.has_lobes:
                for lobe in self.lobes:
                    cv2.circle(self.image, tuple(lobe), radius=6 + r_base, color=(0, 255, 255), thickness=3)

        elif self.has_apex and self.has_base and self.has_ordered_petiole and self.has_midvein and (not self.has_width):
            self.is_leaf_no_width = True

            self.order_points_plot(self.ordered_midvein, 'midvein_trace', 'final')
            self.order_points_plot(self.ordered_petiole, 'petiole_trace', 'final')
            self.order_points_plot([self.apex_left, self.apex_center, self.apex_right], self.apex_angle_type, 'final')
            self.order_points_plot([self.base_left, self.base_center, self.base_right], self.base_angle_type, 'final')

            self.determine_lamina_length('final') 

            # Lamina tip and base
            if self.has_lamina_tip:
                cv2.circle(self.image_final, self.lamina_tip, radius=4 + r_base, color=(0, 255, 0), thickness=2)
                cv2.circle(self.image_final, self.lamina_tip, radius=2 + r_base, color=(255, 255, 255), thickness=-1)
            if self.has_lamina_base:
                cv2.circle(self.image_final, self.lamina_base, radius=4 + r_base, color=(255, 0, 0), thickness=2)
                cv2.circle(self.image_final, self.lamina_base, radius=2 + r_base, color=(255, 255, 255), thickness=-1)

            # Apex angle
            # if self.apex_center != []:
            #     cv2.circle(self.image_final, self.apex_center, radius=3, color=(0, 255, 0), thickness=-1)
            if self.apex_left is not None:
                cv2.circle(self.image_final, self.apex_left, radius=3 + r_base, color=(255, 0, 0), thickness=-1)
            if self.apex_right is not None:
                cv2.circle(self.image_final, self.apex_right, radius=3 + r_base, color=(0, 0, 255), thickness=-1)

            # Base angle
            # if self.base_center:
            #     cv2.circle(self.image_final, self.base_center, radius=3, color=(0, 255, 0), thickness=-1)
            if self.base_left:
                cv2.circle(self.image_final, self.base_left, radius=3 + r_base, color=(255, 0, 0), thickness=-1)
            if self.base_right:
                cv2.circle(self.image_final, self.base_right, radius=3 + r_base, color=(0, 0, 255), thickness=-1)

            # Draw line of fit
            for point in self.width_infer:
                point[0] = np.clip(point[0], 0, self.width - 1)
                point[1] = np.clip(point[1], 0, self.height - 1)
                cv2.circle(self.image_final, tuple(point.astype(int)), radius=4 + r_base, color=(0, 0, 255), thickness=-1)

            # Lobes
            if self.has_lobes:
                for lobe in self.lobes:
                    cv2.circle(self.image, tuple(lobe), radius=6 + r_base, color=(0, 255, 255), thickness=3)





    def restrictions(self):
        # self.check_tips()
        self.connect_midvein_to_tips()
        self.connect_petiole_to_midvein()
        self.check_crossing_width()

    def check_tips(self): # TODO need to check the sides to prevent base from ending up on the tip side. just need to check which side of the oredered list to pull from 
        if max([self.height, self.width]) < 200:
            scale_factor = 0.25
        elif max([self.height, self.width]) < 500:
            scale_factor = 0.5
        else:
            scale_factor = 1

        if self.has_lamina_base:
            second_last_dir = np.array(self.ordered_midvein[-1]) - np.array(self.lamina_base)

            end_vector_mag = np.linalg.norm(second_last_dir)
            avg_dist = np.mean([np.linalg.norm(np.array(self.ordered_midvein[i])-np.array(self.ordered_midvein[i-1])) for i in range(1, len(self.ordered_midvein))])

            if (end_vector_mag > (scale_factor * 0.01 * avg_dist * len(self.ordered_midvein))):
                self.lamina_base = self.ordered_midvein[-1]
                cv2.circle(self.image, self.lamina_base, radius=4, color=(0, 0, 0), thickness=-1)
                cv2.circle(self.image, self.lamina_base, radius=8, color=(0, 0, 255), thickness=2)
                self.logger.debug(f'Check Tips - lamina base - made lamina base the last midvein point')
            else:
                self.logger.debug(f'Check Tips - lamina base - kept lamina base')

        
        if self.has_lamina_tip:
            second_last_dir = np.array(self.ordered_midvein[0]) - np.array(self.lamina_tip)

            end_vector_mag = np.linalg.norm(second_last_dir)
            avg_dist = np.mean([np.linalg.norm(np.array(self.ordered_midvein[i])-np.array(self.ordered_midvein[i-1])) for i in range(1, len(self.ordered_midvein))])

            if (end_vector_mag > (scale_factor * 0.01 * avg_dist * len(self.ordered_midvein))):
                self.lamina_tip = self.ordered_midvein[-1]
                cv2.circle(self.image, self.lamina_tip, radius=4, color=(0, 0, 0), thickness=-1)
                cv2.circle(self.image, self.lamina_tip, radius=8, color=(0, 0, 255), thickness=2)
                self.logger.debug(f'Check Tips - lamina tip - made lamina tip the first midvein point')
            else:
                self.logger.debug(f'Check Tips - lamina tip - kept lamina tip')

    def connect_midvein_to_tips(self):
        self.logger.debug(f'Restrictions [Midvein Connect] - connect_midvein_to_tips()')
        if self.has_midvein:
            if self.has_lamina_tip:
                original_lamina_tip = self.lamina_tip

                start_or_end = self.add_tip(self.lamina_tip)
                self.logger.debug(f'Restrictions [Midvein Connect] - Lamina tip [{self.lamina_tip}]')


                self.ordered_midvein, move_midvein = self.check_momentum_complex(self.ordered_midvein, True, start_or_end)
                if move_midvein: # the tip changed the momentum too much
                    self.logger.debug(f'Restrictions [Midvein Connect] - REDO APEX ANGLE - SWAP LAMINA TIP FOR FIRST MIDVEIN POINT')
                    # get midvein point cloases to tip
                    # new_endpoint_side, _ = self.get_closest_point_to_sampled_points(self.ordered_midvein, original_lamina_tip)
                    # new_endpoint, _ = self.get_closest_point_to_sampled_points([self.ordered_midvein[0], self.ordered_midvein[-1]], new_endpoint_side)

                    # change the apex to new endpoint
                    self.lamina_tip = self.ordered_midvein[0]
                    self.apex_center = self.ordered_midvein[0]
                    self.determine_lamina_length('QC')

                    self.determine_apex_redo()
                
                # cv2.imshow('img', self.image)
                # cv2.waitKey(0)
                # self.order_points_plot(self.ordered_midvein, 'midvein_trace')
                self.logger.debug(f'Restrictions [Midvein Connect] - connected lamina tip to midvein')
            else:
                self.logger.debug(f'Restrictions [Midvein Connect] - lacks lamina tip')



            if self.has_lamina_base:
                original_lamina_base = self.lamina_base
                
                start_or_end = self.add_tip(self.lamina_base)
                self.logger.debug(f'Restrictions [Midvein Connect] - Lamina base [{self.lamina_base}]')

                self.ordered_midvein, move_midvein = self.check_momentum_complex(self.ordered_midvein, True, start_or_end)
                if move_midvein: # the tip changed the momentum too much
                    self.logger.debug(f'Restrictions [Midvein Connect] - REDO BASE ANGLE - SWAP LAMINA BASE FOR LAST MIDVEIN POINT')

                    # get midvein point cloases to tip
                    # new_endpoint_side, _ = self.get_closest_point_to_sampled_points(self.ordered_midvein, original_lamina_base)
                    # new_endpoint, _ = self.get_closest_point_to_sampled_points([self.ordered_midvein[0], self.ordered_midvein[-1]], new_endpoint_side)

                    # change the apex to new endpoint
                    self.lamina_base = self.ordered_midvein[-1]
                    self.base_center = self.ordered_midvein[-1]
                    self.determine_lamina_length('QC')

                    self.determine_base_redo()

                # self.order_points_plot(self.ordered_midvein, 'midvein_trace')
                self.logger.debug(f'Restrictions [Midvein Connect] - connected lamina base to midvein')
            else:
                self.logger.debug(f'Restrictions [Midvein Connect] - lacks lamina base')
            

    def connect_petiole_to_midvein(self):
        if self.has_ordered_petiole and self.has_midvein:
            if len(self.ordered_petiole) > 0 and len(self.ordered_midvein) > 0:
                # Find the closest pair of points between ordered_petiole and ordered_midvein
                min_dist = np.inf
                closest_petiole_idx = None
                closest_midvein_idx = None

                for i, petiole_point in enumerate(self.ordered_petiole):
                    for j, midvein_point in enumerate(self.ordered_midvein):
                        # Convert petiole_point and midvein_point to NumPy arrays
                        petiole_point = np.array(petiole_point)
                        midvein_point = np.array(midvein_point)

                        # Calculate the distance between the two points
                        dist = np.linalg.norm(petiole_point - midvein_point)
                        if dist < min_dist:
                            min_dist = dist
                            closest_petiole_idx = i
                            closest_midvein_idx = j

                # Calculate the midpoint between the closest points
                petiole_point = self.ordered_petiole[closest_petiole_idx]
                midvein_point = self.ordered_midvein[closest_midvein_idx]
                midpoint = (int((petiole_point[0] + midvein_point[0]) / 2), int((petiole_point[1] + midvein_point[1]) / 2))

                # Determine whether the midpoint should be added to the beginning or end of each list
                petiole_dist_to_end = np.linalg.norm(np.array(self.ordered_petiole[closest_petiole_idx]) - np.array(self.ordered_petiole[-1]))
                midvein_dist_to_end = np.linalg.norm(np.array(self.ordered_midvein[closest_midvein_idx]) - np.array(self.ordered_midvein[-1]))

                if (petiole_dist_to_end < midvein_dist_to_end):
                    # Add the midpoint to the end of the petiole list and the beginning of the midvein list
                    self.ordered_midvein.insert(0, midpoint)
                    self.ordered_petiole.append(midpoint)
                    self.lamina_base = midpoint
                    cv2.circle(self.image, self.lamina_base, radius=4, color=(0, 255, 0), thickness=-1)
                    cv2.circle(self.image, self.lamina_base, radius=6, color=(0, 0, 0), thickness=2)
                else:
                    # Add the midpoint to the end of the midvein list and the beginning of the petiole list
                    self.ordered_petiole.insert(0, midpoint)
                    self.ordered_midvein.append(midpoint)
                    self.lamina_base = midpoint
                    cv2.circle(self.image, self.lamina_base, radius=4, color=(0, 255, 0), thickness=-1)
                    cv2.circle(self.image, self.lamina_base, radius=6, color=(0, 0, 0), thickness=2)
                # If the momentum changed, then move the apex/base  centers to the begninning/end of the new midvein.
                # self.ordered_midvein, move_midvein = self.check_momentum(self.ordered_midvein, True)
                # self.ordered_petiole, move_petiole = self.check_momentum(self.ordered_petiole, True)

                # if move_midvein or move_petiole:
                    # self.logger.debug(f'')

                self.order_points_plot(self.ordered_midvein, 'midvein_trace', 'QC')
                self.order_points_plot(self.ordered_petiole, 'petiole_trace', 'QC')



    def check_crossing_width(self):
        self.logger.debug(f'Restrictions [Crossing Width Line] - check_crossing_width()')
        self.width_infer = None

        if self.has_width:
            self.logger.debug(f'Restrictions [Crossing Width Line] - has width')
            # Given two points
            x1, y1 = self.width_left
            x2, y2 = self.width_right

            # Calculate the slope and y-intercept
            denom = (x2 - x1)
            if denom == 0:
                denom = 0.00000000001
            m = (y2 - y1) / denom
            b = y1 - m * x1
            line_params = [m, b]

            self.restrict_by_width_relation(line_params)

        elif not self.has_width:
            # generate approximate width line
            self.logger.debug(f'Restrictions [Crossing Width Line] - infer width')
            if self.has_apex and self.has_base:
                line_params = self.infer_width_relation()
                self.restrict_by_width_relation(line_params)

            else:
                self.has_ordered_petiole = False
                self.has_apex = False
                self.has_base = False
                self.has_valid_apex_loc = False
                self.has_valid_base_loc = False
                self.logger.debug(f'Restrictions [Crossing Width Line] - CANNOT VALIDATE APEX, BASE, PETIOLE LOCATIONS')
        
        else:
            self.logger.debug(f'Restrictions [Crossing Width Line] - width fail *** ERROR ***')

    def infer_width_relation(self):
        top = [np.array((self.apex_center[0], self.apex_center[1])), np.array((self.apex_left[0], self.apex_left[1])), np.array((self.apex_right[0], self.apex_right[1]))]
        bottom = [np.array((self.base_center[0], self.base_center[1])), np.array((self.base_left[0], self.base_left[1])), np.array((self.base_right[0], self.base_right[1]))]
        if self.has_ordered_petiole:
            bottom = bottom + [np.array(pt) for pt in self.ordered_petiole]

        if self.has_midvein:
            midvein = np.array(self.ordered_midvein)
            self.logger.debug(f'Restrictions [Crossing Width Line] - infer width - using midvein points')
        else:
            self.logger.debug(f'Restrictions [Crossing Width Line] - infer width - estimating midvein points')
            x_increment = (centroid2[0] - centroid1[0]) / 11
            y_increment = (centroid2[1] - centroid1[1]) / 11
            midvein = []
            for i in range(1, 11):
                x = centroid1[0] + i * x_increment
                y = centroid1[1] + i * y_increment
                midvein.append([x, y])

        # find the centroids of each group of points
        centroid1 = np.mean(top, axis=0)
        centroid2 = np.mean(bottom, axis=0)

        # calculate the midpoint between the centroids
        midpoint = (centroid1 + centroid2) / 2

        # calculate the vector between the centroids
        centroid_vector = centroid2 - centroid1

        # calculate the vector perpendicular to the centroid vector
        perp_vector = np.array([-centroid_vector[1], centroid_vector[0]])

        # normalize the perpendicular vector
        perp_unit_vector = perp_vector / np.linalg.norm(perp_vector)

        # define the length of the line segment
        # line_segment_length = np.linalg.norm(centroid_vector) / 2

        # calculate the maximum length of the line segment that can be drawn inside the image
        max_line_segment_length = min(midpoint[0], midpoint[1], self.width - midpoint[0], self.height - midpoint[1])

        # calculate the step size
        step_size = max_line_segment_length / 5

        # generate 10 points along the line that is perpendicular to the centroid vector and goes through the midpoint
        points = []
        for i in range(-5, 6):
            point = midpoint + i * step_size * perp_unit_vector
            points.append(point)

        # find the equation of the line passing through the midpoint and with the perpendicular unit vector as the slope
        b = midpoint[1] - perp_unit_vector[1] * midpoint[0]
        if perp_unit_vector[0] == 0:
            denom = 0.0000000001
        else:
            denom = perp_unit_vector[0]
        m = perp_unit_vector[1] / denom

        self.width_infer = points
        # Draw line of fit
        for point in points:
            point[0] = np.clip(point[0], 0, self.width - 1)
            point[1] = np.clip(point[1], 0, self.height - 1)
            cv2.circle(self.image, tuple(point.astype(int)), radius=2, color=(0, 0, 255), thickness=-1)

        return [m, b]


    def restrict_by_width_relation(self, line_params):
        '''
        Are the tips on the same side
        '''
        if self.has_lamina_base and self.has_lamina_tip:
            loc_tip = self.point_position_relative_to_line(self.lamina_tip, line_params)
            loc_base = self.point_position_relative_to_line(self.lamina_base, line_params)

            if loc_tip == loc_base:
                self.has_lamina_base = False
                self.has_lamina_tip = False
                
                cv2.circle(self.image, self.lamina_tip, radius=5, color=(0, 0, 0), thickness=2) # pink solid
                cv2.circle(self.image, self.lamina_base, radius=5, color=(0, 0, 0), thickness=2) # purple

                self.logger.debug(f'Restrictions [Lamina Tip/Base] - fail - Lamina tip and base are on same side')
            else:
                self.logger.debug(f'Restrictions [Lamina Tip/Base] - pass - Lamina tip and base are on opposite side')

        '''
        are all apex and base values on their respecitive sides?
        '''
        self.has_valid_apex_loc = False
        self.has_valid_base_loc = False
        apex_side = 'NA'
        base_side = 'NA'
        if self.has_apex:
            loc_left = self.point_position_relative_to_line(self.apex_left, line_params)
            loc_right = self.point_position_relative_to_line(self.apex_right, line_params)
            loc_center = self.point_position_relative_to_line(self.apex_center, line_params)
            if loc_left == loc_right == loc_center: # all the same
                apex_side = loc_center
                self.has_valid_apex_loc = True
            else:
                self.has_valid_apex_loc = False
                self.logger.debug(f'Restrictions [Angles] - has_valid_apex_loc = False, apex loc crosses width')
        else:
            self.logger.debug(f'Restrictions [Angles] - has_valid_apex_loc = False, no apex')

        if self.has_base:
            loc_left_b = self.point_position_relative_to_line(self.base_left, line_params)
            loc_right_b = self.point_position_relative_to_line(self.base_right, line_params)
            loc_center_b = self.point_position_relative_to_line(self.base_center, line_params)
            if loc_left_b == loc_right_b == loc_center_b: # all the same
                base_side = loc_center_b
                self.has_valid_base_loc = True
            else:
                self.logger.debug(f'Restrictions [Angles] - has_valid_base_loc = False, base loc crosses width')
        else:
            self.logger.debug(f'Restrictions [Angles] - has_valid_base_loc = False')

        if self.has_valid_apex_loc and self.has_valid_base_loc and (base_side != apex_side):
            self.logger.debug(f'Restrictions [Angles] - pass - apex and base')
        elif (base_side == apex_side) and (self.has_apex) and (self.has_base):
            self.has_valid_apex_loc = False
            self.has_valid_base_loc = False
            ### This is most restrictive 
            self.has_apex = False
            self.has_base = False

            self.order_points_plot([self.apex_left, self.apex_center, self.apex_right], 'failed_angle', 'QC')
            self.order_points_plot([self.base_left, self.base_center, self.base_right], 'failed_angle', 'QC')

            self.logger.debug(f'Restrictions [Angles] - fail -  apex and base')
        elif (not self.has_valid_apex_loc) and (self.has_apex):
            self.has_apex = False
            self.order_points_plot([self.apex_left, self.apex_center, self.apex_right], 'failed_angle', 'QC')
            self.logger.debug(f'Restrictions [Angles] - fail - apex')

        elif (not self.has_valid_base_loc) and (self.has_base):
            self.has_base = False
            self.order_points_plot([self.base_left, self.base_center, self.base_right], 'failed_angle', 'QC')
            self.logger.debug(f'Restrictions [Angles] - fail - base')
        else:
            self.logger.debug(f'Restrictions [Angles] - no change')
            

        '''
        does the petiole cross the width loc?
        '''
        if self.has_ordered_petiole:
            petiole_check = []
            for point in self.ordered_petiole:
                check_val = self.point_position_relative_to_line(point, line_params)
                petiole_check.append(check_val)
            petiole_check = list(set(petiole_check))
            self.logger.debug(f'Restrictions [Petiole] - petiole set = {petiole_check}')

            if len(petiole_check) == 1:
                self.has_ordered_petiole = True # Keep the petiole
                petiole_check = petiole_check[0]
                self.logger.debug(f'Restrictions [Petiole] - petiole does not cross width - pass')
            else:
                self.has_ordered_petiole = False # Reject the petiole, it crossed the center
                self.logger.debug(f'Restrictions [Petiole] - petiole does cross width - fail')
        else:
            self.logger.debug(f'Restrictions [Petiole] - has_ordered_petiole = False')

        '''
        Is the lamina base on the same side as the petiole?
            happens after the other checks...
        '''
        if self.has_lamina_base and self.has_lamina_tip and self.has_ordered_petiole:
            # base is not on the same side as petiole, swap IF base and tip are already opposite
            if loc_base != petiole_check:
                if loc_base != loc_tip: # make sure that the tips are on opposite sides, if yes, swap the base and tip
                    hold_data = self.lamina_tip
                    self.lamina_tip = self.lamina_base
                    self.lamina_base = hold_data

                    cv2.circle(self.image, self.lamina_tip, radius=9, color=(255, 0, 230), thickness=2) # pink solid
                    cv2.circle(self.image, self.lamina_base, radius=9, color=(0, 100, 255), thickness=2) # purple

                    self.logger.debug(f'Restrictions [Petiole/Lamina Tip Same Side] - pass - swapped lamina tip and lamina base')
                else:
                    self.has_lamina_base = False
                    self.has_lamina_tip = False
                    self.logger.debug(f'Restrictions [Petiole/Lamina Tip Same Side] - fail - lamina base not on same side as petiole, base and tip are on same side')
            else: # base is on correct side
                if loc_base == loc_tip: # base and tip are on the same side. error
                    self.has_lamina_base = False
                    self.has_lamina_tip = False
                    self.logger.debug(f'Restrictions [Petiole/Lamina Tip Same Side] - fail - base and tip are on the same side, but base and petiole are ok')
                else:
                    self.logger.debug(f'Restrictions [Petiole/Lamina Tip Same Side] - pass - no swap')


    def add_tip(self, tip):
        # Calculate the distances between the first and last points in midvein and the new point
        dist_start = math.dist(self.ordered_midvein[0], tip)
        dist_end = math.dist(self.ordered_midvein[-1], tip)

        # Append tip to the beginning of the list if it's closer to the first point, otherwise append it to the end of the list
        if dist_start < dist_end:
            self.ordered_midvein.insert(0, tip)
            start_or_end = 'start'
            self.logger.debug(f'Restrictions [Midvein Connect] - tip added to beginning of ordered_midvein')
        else:
            self.ordered_midvein.append(tip)
            start_or_end = 'end'
            self.logger.debug(f'Restrictions [Midvein Connect] - tip added to end of ordered_midvein')
        return start_or_end

    def find_min_width(self, left, right):
        left_vectors = np.array(left)[:, np.newaxis, :]
        right_vectors = np.array(right)[np.newaxis, :, :]
        distances = np.linalg.norm(left_vectors - right_vectors, axis=2)
        indices = np.unravel_index(np.argmin(distances), distances.shape)
        return left[indices[0]], right[indices[1]]

    def find_most_orthogonal_vectors(self, left, right, midvein_fit):
        left_vectors = np.array(left)[:, np.newaxis, :] - np.array(right)[np.newaxis, :, :]
        right_vectors = -left_vectors
        midvein_vector = np.array(midvein_fit[-1]) - np.array(midvein_fit[0])
        midvein_vector /= np.linalg.norm(midvein_vector)

        dot_products = np.abs(np.sum(left_vectors * midvein_vector, axis=2)) + np.abs(np.sum(right_vectors * midvein_vector, axis=2))
        indices = np.unravel_index(np.argmax(dot_products), dot_products.shape)
        return left[indices[0]], right[indices[1]]

    def determine_reflex(self, apex_left, apex_right, apex_center):
        vector_left_to_center = np.array([apex_center[0] - apex_left[0], apex_center[1] - apex_left[1]])
        vector_right_to_center = np.array([apex_center[0] - apex_right[0], apex_center[1] - apex_right[1]])

        # Calculate the vector pointing to the average midvein trace value
        midvein_trace_arr = np.array([(x, y) for x, y in self.ordered_midvein])
        midvein_trace_avg = midvein_trace_arr.mean(axis=0)
        vector_to_midvein_trace = midvein_trace_avg - np.array(apex_center)

        # Determine whether the angle is reflex or not
        if np.dot(vector_left_to_center, vector_to_midvein_trace) > 0 and np.dot(vector_right_to_center, vector_to_midvein_trace) > 0:
            angle_type = 'reflex'
        else:
            angle_type = 'not_reflex'

        angle = self.calculate_angle(apex_left, apex_center, apex_right)
        if angle_type == 'reflex':
            angle = 360 - angle

        self.order_points_plot([apex_left, apex_center, apex_right], angle_type, 'QC')
        
        return angle_type, angle

    def calculate_angle(self, p1, p2, p3):
        # Calculate the vectors between the points
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # Calculate the dot product and magnitudes of the vectors
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag_v1 == 0:
            mag_v1 = 0.000000001
        if mag_v2 == 0:
            mag_v2 = 0.000000001
        # Calculate the cosine of the angle
        denom = (mag_v1 * mag_v2)
        if denom == 0:
            denom = 0.000000001
        cos_angle = dot_product / denom
        
        # Calculate the angle in radians and degrees
        angle_rad = math.acos(min(max(cos_angle, -1), 1))

        angle_deg = math.degrees(angle_rad)
        
        return angle_deg

    def calc_min_distance(self, point, reference_points):
        # Convert the points and reference points to numpy arrays
        points_arr = np.atleast_2d(point)
        reference_arr = np.array(reference_points)

        # Calculate the distances between each point in "points" and each point in "reference_points"
        dists = np.linalg.norm(points_arr[:, np.newaxis, :] - reference_arr, axis=2)
        distance = np.min(dists, axis=1)
        return distance


    def get_closest_point_to_sampled_points(self, points, reference_points):
        # Convert the points and reference points to numpy arrays
        points_arr = np.array(points)
        reference_arr = np.array(reference_points)

        # Calculate the distances between each point in "points" and each point in "reference_points"
        dists = np.linalg.norm(points_arr[:, np.newaxis, :] - reference_arr, axis=2)
        distances = np.min(dists, axis=1)

        # Get the index of the closest point
        closest_idx = np.argmin(distances)

        # Remove the closest point from the list of points
        return points.pop(closest_idx), points

    def get_far_point(self, points, reference_point):
        # Calculate the distances between each point and the reference points
        distances = [math.dist(point, reference_point) for point in points]

        # Get the index of the closest point
        closest_idx = distances.index(max(distances))
        far_point = points.pop(closest_idx)

        # Remove the closest point from the list of points
        return far_point, points

    '''def point_position_relative_to_line(self, point, line_params):
        # Extract the cubic coefficients from the line parameters
        a, b, c, d = line_params

        # Determine the x-coordinate of the point where it intersects with the line
        # We solve the cubic equation ax^3 + bx^2 + cx + d = y for x, given y = point[1]
        f = lambda x: a*x**3 + b*x**2 + c*x + d - point[1]
        roots = np.roots([a, b, c, d-point[1]])
        real_roots = roots[np.isreal(roots)].real
        if len(real_roots) == 0:
            return "left"  # point is below the curve
        x_intersection = real_roots[0]

        # Determine the midpoint of the line
        mid_x = self.width / 2
        mid_y = self.height / 2

        # Determine if the point is to the left or right of the line
        if self.height > self.width:
            if point[0] < x_intersection:
                return "left"
            else:
                return "right"
        else:
            if point[1] < a*mid_x**3 + b*mid_x**2 + c*mid_x + d:
                return "left"
            else:
                return "right"'''

    def point_position_relative_to_line(self, point, line_params):
        # Extract the slope and y-intercept from the line parameters
        slope, y_intercept = line_params

        if (slope == 0.0) or (slope == 0):
            slope = 0.00000000000001

        # Determine the x-coordinate of the point where it intersects with the line
        x_intersection = (point[1] - y_intercept) / slope

        # Determine the midpoint of the line
        mid_x = self.width / 2
        mid_y = self.height / 2

        # Determine if the point is to the left or right of the line
        if self.height > self.width:
            if point[0] < x_intersection:
                return "left"
            else:
                return "right"
        else:
            if point[1] < slope * (point[0] - mid_x) + mid_y:
                return "left" #below
            else:
                return "right" #above

    def rotate_points(self, points, angle_cw):
        # Calculate the center of the image
        center_x = self.width / 2
        center_y = self.height / 2

        # Translate the points to the center
        translated_points = [(point[0]-center_x, point[1]-center_y) for point in points]

        # Convert the angle to radians
        angle_cw = math.radians(angle_cw)

        # Rotate the points
        rotated_points = [(round(point[0]*math.cos(angle_cw)-point[1]*math.sin(angle_cw)), round(point[0]*math.sin(angle_cw)+point[1]*math.cos(angle_cw))) for point in translated_points]

        # Translate the points back to the original origin
        return [(point[0]+center_x, point[1]+center_y) for point in rotated_points]


    def order_petiole(self):
        if 'petiole_trace' in self.points_list:
            if len(self.points_list['petiole_trace']) >= 5:
                self.logger.debug(f"Ordered Petiole - Raw list contains {len(self.points_list['petiole_trace'])} points - using momentum")
                self.ordered_petiole = self.order_points(self.points_list['petiole_trace'])
                self.ordered_petiole = self.remove_duplicate_points(self.ordered_petiole)

                self.ordered_petiole = self.check_momentum(self.ordered_petiole, False)

                self.order_points_plot(self.ordered_petiole, 'petiole_trace', 'QC')
                self.ordered_petiole_length, self.ordered_petiole = self.get_length_of_ordered_points(self.ordered_petiole, 'petiole_trace')
                self.has_ordered_petiole = True
            elif len(self.points_list['petiole_trace']) >= 2:
                self.logger.debug(f"Ordered Petiole - Raw list contains {len(self.points_list['petiole_trace'])} points - SKIPPING momentum")
                self.ordered_petiole = self.order_points(self.points_list['petiole_trace'])
                self.ordered_petiole = self.remove_duplicate_points(self.ordered_petiole)

                self.order_points_plot(self.ordered_petiole, 'petiole_trace', 'QC')
                self.ordered_petiole_length, self.ordered_petiole = self.get_length_of_ordered_points(self.ordered_petiole, 'petiole_trace')
                self.has_ordered_petiole = True
            else:
                self.logger.debug(f"Ordered Petiole - Raw list contains {len(self.points_list['petiole_trace'])} points - SKIPPING PETIOLE")

    def order_midvein(self):
        if 'midvein_trace' in self.points_list:
            if len(self.points_list['midvein_trace']) >= 5:
                self.logger.debug(f"Ordered Midvein - Raw list contains {len(self.points_list['midvein_trace'])} points - using momentum")
                self.ordered_midvein = self.order_points(self.points_list['midvein_trace'])
                self.ordered_midvein = self.remove_duplicate_points(self.ordered_midvein)

                self.ordered_midvein = self.check_momentum(self.ordered_midvein, False)

                self.order_points_plot(self.ordered_midvein, 'midvein_trace', 'QC')
                self.ordered_midvein_length, self.ordered_midvein = self.get_length_of_ordered_points(self.ordered_midvein, 'midvein_trace')
                self.has_midvein = True
            else:
                self.logger.debug(f"Ordered Midvein - Raw list contains {len(self.points_list['midvein_trace'])} points - SKIPPING MIDVEIN")


    def check_momentum(self, coords, info):
        original_coords = coords
        # find middle index of coordinates
        mid_idx = len(coords) // 2

        # set up variables for running average
        running_avg = np.array(coords[mid_idx-1])
        avg_count = 1

        # iterate over coordinates to check momentum change
        prev_vec = np.array(coords[mid_idx-1]) - np.array(coords[mid_idx-2])
        cur_idx = mid_idx - 1
        while cur_idx >= 0:
            cur_vec = np.array(coords[cur_idx]) - np.array(coords[cur_idx-1])

            # add current point to running average
            running_avg = (running_avg * avg_count + np.array(coords[cur_idx])) / (avg_count + 1)
            avg_count += 1

            # check for momentum change
            if self.check_momentum_change(prev_vec, cur_vec):
                break

            prev_vec = cur_vec
            cur_idx -= 1

        # use running average to check for momentum change
        cur_vec = np.array(coords[cur_idx]) - running_avg
        if self.check_momentum_change(prev_vec, cur_vec):
            cur_idx += 1

        prev_vec = np.array(coords[mid_idx+1]) - np.array(coords[mid_idx])
        cur_idx2 = mid_idx + 1
        while cur_idx2 < len(coords):

            # check if current index is out of range
            if cur_idx2 >= len(coords):
                break

            cur_vec = np.array(coords[cur_idx2]) - np.array(coords[cur_idx2-1])

            # add current point to running average
            running_avg = (running_avg * avg_count + np.array(coords[cur_idx2])) / (avg_count + 1)
            avg_count += 1

            # check for momentum change
            if self.check_momentum_change(prev_vec, cur_vec):
                break

            prev_vec = cur_vec
            cur_idx2 += 1

        # use running average to check for momentum change
        if cur_idx2 < len(coords):
            cur_vec = np.array(coords[cur_idx2]) - running_avg
            if self.check_momentum_change(prev_vec, cur_vec):
                cur_idx2 -= 1

        # remove problematic points and subsequent points from list of coordinates
        new_coords = coords[:cur_idx2] + coords[mid_idx:cur_idx2:-1]
        if info:
            return new_coords, len(original_coords) != len(new_coords)
        else:
            return new_coords

    # define function to check for momentum change
    def check_momentum_change(self, prev_vec, cur_vec):
        dot_product = np.dot(prev_vec, cur_vec)
        prev_norm = np.linalg.norm(prev_vec)
        cur_norm = np.linalg.norm(cur_vec)
        denom = (prev_norm * cur_norm)
        if denom == 0:
            denom = 0.0000000001
        cos_theta = dot_product / denom
        theta = np.arccos(cos_theta)
        return abs(theta) > np.pi / 2

    '''def check_momentum_complex(self, coords, info, start_or_end):
        original_coords = coords
        # find middle index of coordinates
        mid_idx = len(coords) // 2

        # get directional vectors for first-middle, middle-last, and second-first and second-last pairs of points
        first_middle_dir = np.array(coords[1]) - np.array(coords[0])
        middle_last_dir = np.array(coords[-1]) - np.array(coords[-2])
        second_first_dir = np.array(coords[1]) - np.array(coords[2])
        second_last_dir = np.array(coords[-1]) - np.array(coords[-3])

        if start_or_end == 'end':
            # check directional change for first-middle vector
            cur_idx = 2
            while cur_idx < len(coords):
                cur_vec = np.array(coords[cur_idx]) - np.array(coords[cur_idx-1])
                if self.check_momentum_change_complex(first_middle_dir, cur_vec):
                    break
                cur_idx += 1
                
            cur_idx2 = len(coords) - 2

        elif start_or_end == 'start':
            # check directional change for last-middle vector
            cur_idx2 = len(coords)-3
            while cur_idx2 >= 0:
                cur_vec = np.array(coords[cur_idx2]) - np.array(coords[cur_idx2+1])
                if self.check_momentum_change_complex(middle_last_dir, cur_vec):
                    break
                cur_idx2 -= 1
            
            cur_idx = 1

        # check directional change for second-first and second-last vectors
        second_first_change = self.check_momentum_change_complex(second_first_dir, first_middle_dir)
        second_last_change = self.check_momentum_change_complex(second_last_dir, middle_last_dir)

        # remove problematic points and subsequent points from list of coordinates
        if cur_idx <= cur_idx2:
            new_coords = coords[:cur_idx+1] + coords[cur_idx2:mid_idx:-1] + coords[cur_idx+1:cur_idx2+1]
        else:
            new_coords = coords[:mid_idx+1] + coords[cur_idx2:cur_idx:-1] + coords[mid_idx+1:cur_idx2+1]

        self.logger.debug(f'Original midvein points - {self.ordered_midvein}')
        self.logger.debug(f'Momentum midvein points - {new_coords}')
        if info:
            return new_coords, len(original_coords) != len(new_coords) or second_first_change or second_last_change
        else:
            return new_coords'''

    def check_momentum_complex(self, coords, info, start_or_end): # Works, but removes ALL points after momentum change
        original_coords = coords

        if max([self.height, self.width]) < 200:
            scale_factor = 0.25
        elif max([self.height, self.width]) < 500:
            scale_factor = 0.5
        else:
            scale_factor = 1
        self.logger.debug(f'Scale factor - [{scale_factor}]')

        # find middle index of coordinates
        mid_idx = len(coords) // 2

        # get directional vectors for first-middle, middle-last, and second-first and second-last pairs of points
        first_middle_dir = np.array(coords[1]) - np.array(coords[mid_idx])
        middle_last_dir = np.array(coords[mid_idx]) - np.array(coords[-2])
        second_first_dir = np.array(coords[1]) - np.array(coords[0])
        second_last_dir = np.array(coords[-1]) - np.array(coords[-2])

        if start_or_end == 'end':
            # check directional change for first-middle vector
            cur_idx_list = []
            cur_idx = 2
            while cur_idx < len(coords):
                cur_vec = np.array(coords[cur_idx]) - np.array(coords[cur_idx-1])
                if self.check_momentum_change_complex(first_middle_dir, cur_vec):
                    # break
                    cur_idx_list.append(cur_idx)
                cur_idx += 1
            if len(cur_idx_list) > 0:
                cur_idx = max(cur_idx_list)
            else:
                cur_idx = len(coords)
            # remove problematic points and subsequent points from list of coordinates
            end_vector_mag = np.linalg.norm(second_last_dir)
            avg_dist = np.mean([np.linalg.norm(np.array(coords[i])-np.array(coords[i-1])) for i in range(1, len(coords))])
            new_coords = coords

            if (end_vector_mag > (scale_factor * 0.01 * avg_dist * len(new_coords))) and (len(cur_idx_list) > 0):
                # new_coords = coords[:cur_idx+1] + coords[-2:cur_idx:-1][::-1] #coords[-2:cur_idx:-1]
                new_coords = coords[:len(new_coords)-1]# + coords[-2:cur_idx:-1][::-1] #coords[-2:cur_idx:-1]
                self.logger.debug(f'Momentum - removing last point')
            else:
                self.logger.debug(f'Momentum - change not detected, no change')

                

        elif start_or_end == 'start':
            # check directional change for last-middle vector
            cur_idx2_list = []
            cur_idx2 = len(coords)-3
            while cur_idx2 >= 0:
                cur_vec = np.array(coords[cur_idx2]) - np.array(coords[cur_idx2+1])
                if self.check_momentum_change_complex(middle_last_dir, cur_vec):
                    # break
                    cur_idx2_list.append(cur_idx2)
                cur_idx2 -= 1
            if len(cur_idx2_list) > 0:
                cur_idx2 = min(cur_idx2_list)
            else:
                cur_idx2 = 0
            # remove problematic points and subsequent points from list of coordinates
            new_coords = coords
            start_vector_mag = np.linalg.norm(second_first_dir)
            avg_dist = np.mean([np.linalg.norm(np.array(coords[i])-np.array(coords[i-1])) for i in range(1, len(coords))])

            if (start_vector_mag > (scale_factor * 0.01 * avg_dist * len(new_coords))) and (len(cur_idx2_list) > 0):
                # new_coords = coords[:mid_idx+1] + coords[cur_idx2:mid_idx:-1][::-1] # #coords[cur_idx2:mid_idx:-1]
                new_coords = coords[1:]#ur_idx2-1] + coords[mid_idx+1:]
                # new_coords = coords[cur_idx2:mid_idx+1][::-1] + coords[mid_idx+1:]
                self.logger.debug(f'Momentum - removing first point')
            else:
                self.logger.debug(f'Momentum - change not detected, no change')
        else:
            print('hi')

        # check directional change for second-first and second-last vectors
        # second_first_change = self.check_momentum_change_complex(second_first_dir, first_middle_dir)
        # second_last_change = self.check_momentum_change_complex(second_last_dir, middle_last_dir)

        self.logger.debug(f'Original midvein points complex - {start_or_end} - {self.ordered_midvein}')
        self.logger.debug(f'Momentum midvein points complex - {start_or_end} - {new_coords}')
        if info:
            return new_coords, len(original_coords) != len(new_coords) #or second_first_change or second_last_change
        else:
            return new_coords

    '''def check_momentum_complex(self, coords, info, start_or_end): # does not seem to work
        original_coords = coords
        
        # get directional vectors for first-middle, middle-last, and second-first and second-last pairs of points
        first_middle_dir = np.array(coords[1]) - np.array(coords[0])
        middle_last_dir = np.array(coords[-1]) - np.array(coords[-2])
        second_first_dir = np.array(coords[1]) - np.array(coords[2])
        second_last_dir = np.array(coords[-1]) - np.array(coords[-3])
        
        # calculate running average momentum and check if endpoints are different
        if start_or_end == 'end':
            end_vector_mag = np.linalg.norm(first_middle_dir)
            avg_dist = np.mean([np.linalg.norm(np.array(coords[i])-np.array(coords[i-1])) for i in range(1, len(coords))])
            running_avg_momentum = np.mean([np.linalg.norm(np.array(coords[i])-np.array(coords[i-1])) for i in range(len(coords)-10, len(coords))])
            endpoint_diff = np.linalg.norm(np.array(coords[-1])-self.ordered_midvein[-1]) > 0.1*running_avg_momentum
        elif start_or_end == 'start':
            start_vector_mag = np.linalg.norm(middle_last_dir)
            avg_dist = np.mean([np.linalg.norm(np.array(coords[i])-np.array(coords[i-1])) for i in range(1, len(coords))])
            running_avg_momentum = np.mean([np.linalg.norm(np.array(coords[i])-np.array(coords[i-1])) for i in range(10)])
            endpoint_diff = np.linalg.norm(np.array(coords[0])-self.ordered_midvein[0]) > 0.1*running_avg_momentum
            
        # remove problematic points and subsequent points from list of coordinates
        if start_or_end == 'end' and endpoint_diff:
            cur_idx = 2
            while cur_idx < len(coords):
                cur_vec = np.array(coords[cur_idx]) - np.array(coords[cur_idx-1])
                if self.check_momentum_change_complex(first_middle_dir, cur_vec):
                    break
                cur_idx += 1
            new_coords = coords[:cur_idx+1] + coords[-2:cur_idx:-1][::-1]
        elif start_or_end == 'start' and endpoint_diff:
            cur_idx2 = len(coords)-3
            while cur_idx2 >= 0:
                cur_vec = np.array(coords[cur_idx2]) - np.array(coords[cur_idx2+1])
                if self.check_momentum_change_complex(middle_last_dir, cur_vec):
                    break
                cur_idx2 -= 1
            new_coords = coords[:1] + coords[cur_idx2:0:-1][::-1]
        else:
            new_coords = coords

        # check directional change for second-first and second-last vectors
        second_first_change = self.check_momentum_change_complex(second_first_dir, first_middle_dir)
        second_last_change = self.check_momentum_change_complex(second_last_dir, middle_last_dir)

        self.logger.debug(f'Original midvein points - {self.ordered_midvein}')
        self.logger.debug(f'Momentum midvein points - {new_coords}')
        if info:
            return new_coords, len(original_coords) != len(new_coords) #or second_first_change or second_last_change or endpoint_diff
        else:
            return new_coords'''








    # define function to check for momentum change
    def check_momentum_change_complex(self, prev_vec, cur_vec):
        dot_product = np.dot(prev_vec, cur_vec)
        prev_norm = np.linalg.norm(prev_vec)
        cur_norm = np.linalg.norm(cur_vec)
        denom = (prev_norm * cur_norm)
        if denom == 0:
            denom = 0.0000000001
        cos_theta = dot_product / denom
        theta = np.arccos(cos_theta)
        return abs(theta) > np.pi / 2



    def remove_duplicate_points(self, points):
        unique_set = set()
        new_list = []

        for item in points:
            if item not in unique_set:
                unique_set.add(item)
                new_list.append(item)
        return new_list

    def order_points_plot(self, points, version, QC_or_final):
        # thk_base = 0
        thk_base = 16

        if version == 'midvein_trace':
            # color = (0, 255, 0) 
            color = (0, 255, 255)  # yellow
            thick = 2 + thk_base
        elif version == 'petiole_trace':
            color = (255, 255, 0)
            thick = 2 + thk_base
        elif version == 'lamina_width':
            color = (0, 0, 255)
            thick = 2 + thk_base
        elif version == 'lamina_width_alt':
            color = (100, 100, 255)
            thick = 2 + thk_base
        elif version == 'not_reflex':
            color = (200, 0, 123)
            thick = 3 + thk_base
        elif version == 'reflex':
            color = (0, 120, 200)
            thick = 3 + thk_base
        elif version == 'petiole_tip_alt':
            color = (255, 55, 100)
            thick = 1 + thk_base
        elif version == 'petiole_tip':
            color = (100, 255, 55)
            thick = 1 + thk_base
        elif version == 'failed_angle':
            color = (0, 0, 0)
            thick = 3 + thk_base
        # Convert the points to a numpy array and round to integer values
        points_arr = np.round(np.array(points)).astype(int)

        # Draw a green line connecting all of the points
        if QC_or_final == 'QC':
            for i in range(len(points_arr) - 1):
                cv2.line(self.image, tuple(points_arr[i]), tuple(points_arr[i+1]), color, thick)
        else:
            for i in range(len(points_arr) - 1):
                cv2.line(self.image_final, tuple(points_arr[i]), tuple(points_arr[i+1]), color, thick)

        

    
    def get_length_of_ordered_points(self, points, name):
        # if self.file_name == 'B_774373631_Ebenaceae_Diospyros_buxifolia__L__438-687-578-774':
        #     print('hi')
        total_length = 0
        total_length_first_pass = 0
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i+1]
            segment_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            total_length_first_pass += segment_length
        cutoff = total_length_first_pass / 2
        # print(f'Total length of {name}: {total_length_first_pass}')
        # print(f'points length {len(points)}')
        self.logger.debug(f"Total length of {name}: {total_length_first_pass}")
        self.logger.debug(f"Points length {len(points)}")


        # If there are more than 2 points, this will exclude extreme outliers, or
        # misordered points that don't belong
        if len(points) > 2:
            pop_ind = []
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i+1]
                segment_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                if segment_length < cutoff:
                    total_length += segment_length
                else:
                    pop_ind.append(i)
            
            for exclude in pop_ind:
                points.pop(exclude)
            # print(f'Total length of {name}: {total_length}')
            # print(f'Excluded {len(pop_ind)} points')
            # print(f'points length {len(points)}')
            self.logger.debug(f"Total length of {name}: {total_length}")
            self.logger.debug(f"Excluded {len(pop_ind)} points")
            self.logger.debug(f"Points length {len(points)}")

        else:
            total_length = total_length_first_pass

        return total_length, points

    def convert_YOLO_bbox_to_point(self):
        for point_type, bbox in self.points_list.items():
            xy_points = []
            for point in bbox:
                x = point[0]
                y = point[1]
                w = point[2]
                h = point[3]
                x1 = int((x - w/2) * self.width)
                y1 = int((y - h/2) * self.height)
                x2 = int((x + w/2) * self.width)
                y2 = int((y + h/2) * self.height)
                xy_points.append((int((x1+x2)/2), int((y1+y2)/2)))
            self.points_list[point_type] = xy_points
        
    def parse_all_points(self):
        points_list = {}

        for sublist in self.all_points:
            key = sublist[0]
            value = sublist[1:]

            key = self.swap_number_for_string(key)

            if key not in points_list:
                points_list[key] = []
            points_list[key].append(value)

        # print(points_list)
        self.points_list = points_list

    def swap_number_for_string(self, key):
        for k, v in self.classes.items():
            if v == key:
                return k
        return key

    def distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def order_points(self, points):
        # height = max(points, key=lambda point: point[1])[1] - min(points, key=lambda point: point[1])[1]
        # width = max(points, key=lambda point: point[0])[0] - min(points, key=lambda point: point[0])[0]

        if self.height > self.width:
            start_point = min(points, key=lambda point: point[1])
            end_point = max(filter(lambda point: point[0] == max(points, key=lambda point: point[0])[0], points), key=lambda point: point[1])
        else:
            start_point = min(points, key=lambda point: point[0])
            end_point = max(filter(lambda point: point[1] == max(points, key=lambda point: point[1])[1], points), key=lambda point: point[0])

        tour = [start_point]
        unvisited = set(points) - {start_point}

        while unvisited:
            nearest = min(unvisited, key=lambda point: self.distance(tour[-1], point))
            tour.append(nearest)
            unvisited.remove(nearest)

        tour.append(end_point)
        return tour

    def define_landmark_classes(self):
        self.classes = {
            'apex_angle': 0,
            'base_angle': 1,
            'lamina_base': 2,
            'lamina_tip': 3,
            'lamina_width': 4,
            'lobe_tip': 5,
            'midvein_trace': 6,
            'petiole_tip': 7,
            'petiole_trace': 8
            }

    def set_cfg_values(self):
        self.do_show_QC_images = self.cfg['leafmachine']['landmark_detector']['do_show_QC_images']
        self.do_save_QC_images = self.cfg['leafmachine']['landmark_detector']['do_save_QC_images']
        self.do_show_final_images = self.cfg['leafmachine']['landmark_detector']['do_show_final_images']
        self.do_save_final_images = self.cfg['leafmachine']['landmark_detector']['do_save_final_images']

    def setup_QC_image(self):
        self.image = cv2.imread(os.path.join(self.dir_temp, '.'.join([self.file_name, 'jpg'])))

        if self.leaf_type == 'Landmarks_Whole_Leaves':
            self.path_QC_image = os.path.join(self.Dirs.landmarks_whole_leaves_overlay_QC, '.'.join([self.file_name, 'jpg']))
        elif self.leaf_type == 'Landmarks_Partial_Leaves':
            self.path_QC_image = os.path.join(self.Dirs.landmarks_partial_leaves_overlay_QC, '.'.join([self.file_name, 'jpg']))

    def setup_final_image(self):
        self.image_final = cv2.imread(os.path.join(self.dir_temp, '.'.join([self.file_name, 'jpg'])))

        if self.leaf_type == 'Landmarks_Whole_Leaves':
            self.path_image_final = os.path.join(self.Dirs.landmarks_whole_leaves_overlay_final, '.'.join([self.file_name, 'jpg']))
        elif self.leaf_type == 'Landmarks_Partial_Leaves':
            self.path_image_final = os.path.join(self.Dirs.landmarks_partial_leaves_overlay_final, '.'.join([self.file_name, 'jpg']))

    def show_QC_image(self):
        if self.do_show_QC_images:
            cv2.imshow('QC image', self.image)
            cv2.waitKey(0)
    
    def show_final_image(self):
        if self.do_show_final_images:
            cv2.imshow('Final image', self.image_final)
            cv2.waitKey(0)

    def save_QC_image(self):
        if self.do_save_QC_images:
            cv2.imwrite(self.path_QC_image, self.image)

    def get_QC(self):
        return self.image

    def get_final(self):
        return self.image_final

    def init_lists_dicts(self):
        # Initialize all lists and dictionaries
        self.classes = {}
        self.points_list = []
        self.image = []
        self.ordered_midvein = []
        self.midvein_fit = []
        self.midvein_fit_points = []
        self.ordered_petiole = []
        self.apex_left = self.apex_left or None
        self.apex_right = self.apex_right or None
        self.apex_center = self.apex_center or None
        self.base_left = self.base_left or None
        self.base_right = self.base_right or None
        self.base_center = self.base_center or None
        self.lamina_tip = self.lamina_tip or None
        self.lamina_base = self.lamina_base or None
        self.width_left = self.width_left or None
        self.width_right = self.width_right or None


