import os, math, cv2, random
import numpy as np
from itertools import combinations
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Dict
from sklearn.linear_model import LinearRegression
from scipy.optimize import fsolve, minimize


@dataclass()
class ArmatureSkeleton:
    cfg: str 
    Dirs: str 
    leaf_type: str 
    all_points: list
    dir_temp: str
    file_name: str
    width: int
    height: int
    logger: object

    is_complete: bool = False
    keep_going: bool = False
    
    do_show_QC_images: bool = False
    do_save_QC_images: bool = False

    classes: int = 0
    points_list: int = 0

    image: int = 0

    ordered_middle: int = 0
    midvein_fit: int = 0
    midvein_fit_points: int = 0
    ordered_midvein_length: float = 0.0 
    has_middle = False

    has_outer = False
    has_tip = False

    is_split = False

    ordered_petiole: int = 0
    ordered_petiole_length: float = 0.0 
    has_ordered_petiole = False

    has_apex: bool = False
    apex_left: int = 0
    apex_right: int = 0
    apex_center: int = 0
    apex_angle_type: str = 'NA'
    apex_angle_degrees: float = 0.0

    has_base: bool = False
    base_left: int = 0
    base_right: int = 0
    base_center: int = 0
    base_angle_type: str = 'NA'
    base_angle_degrees: float = 0.0

    has_lamina_base: bool = False
    lamina_base: int = 0

    has_lamina_length: bool = False
    lamina_fit: int = 0
    lamina_length: float = 0.0

    has_width: bool = False
    lamina_width: float = 0.0
    width_left: float = 0.0
    width_right: float = 0.0



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

        """ Setup """
        self.set_cfg_values()
        self.define_landmark_classes()

        self.setup_QC_image()
        self.setup_angle_image()
        self.setup_final_image()

        self.parse_all_points()
        self.convert_YOLO_bbox_to_point()

        if (len(self.points_list['outer']) > 6) and (len(self.points_list['middle']) > 3):
            self.keep_going = True

        """ Landmarks """
        if self.keep_going:
            # Start with ordering the midvein and petiole
            self.order_middle()
            # print(self.ordered_midvein)
        if self.keep_going:
            # Split the image using the midvein IF has_midvein == True
            self.split_image_by_middle()
        if self.keep_going:
            self.group_outer_points()
        if self.keep_going:
            # Measure 
            self.measure_armature()
        if self.keep_going:
            # calc tangent angle of outer and inner polys
            self.calc_angle_tangent()
        if self.keep_going:
            self.calc_angle_curl()
        if self.keep_going:
            # self.calc_angle_bend()
            self.calc_curvature_radius()
        if self.keep_going:
            self.calc_direct_length()

            # self.show_QC_image()
            # self.show_angle_image()

            self.is_complete = True # TODO  add ways to set True


    def measure_armature(self):
        # wb = width_base = line between the last outer and inner points
        # Define the line function
        def line_func(x):
            return self.wb_slope * x + self.wb_intercept
        def middle_func(x):
            return self.middle_poly[0]*x**2 + self.middle_poly[1]*x + self.middle_poly[2]
        # Define the difference function
        def line_middle_diff(x):
            return line_func(x) - middle_func(x)
        
        # Convert the points to numpy arrays
        last_point_right = np.array(self.last_point_right)
        last_point_left = np.array(self.last_point_left)

        # Calculate the Euclidean distance between the points
        self.width_base = np.linalg.norm(last_point_right - last_point_left)
        print("The distance between the last points of the right and left segments is:", self.width_base)

        # Intersection of the width and the middlepoly# Draw a line between the last points of the outer_left and outer_right segments
        cv2.line(self.image, (int(self.last_point_left[0]), int(self.last_point_left[1])), (int(self.last_point_right[0]), int(self.last_point_right[1])), gc('white'), thickness=2)
        cv2.line(self.image_angles, (int(self.last_point_left[0]), int(self.last_point_left[1])), (int(self.last_point_right[0]), int(self.last_point_right[1])), color=gc('white'), thickness=2)

        # Calculate the slope and y-intercept of the line
        self.wb_slope = (self.last_point_right[1] - self.last_point_left[1]) / (self.last_point_right[0] - self.last_point_left[0])
        self.wb_intercept = self.last_point_left[1] - self.wb_slope * self.last_point_left[0]

        # Find the intersection point
        intersection_x = fsolve(line_middle_diff, 0)[0]
        intersection_y = line_func(intersection_x)

        self.width_base_inter = [(int(intersection_x), int(intersection_y))]
        # Calculate the midpoint between the last points
        self.width_base_mid = (last_point_right + last_point_left) / 2
        
        cv2.circle(self.image, (int(intersection_x), int(intersection_y)), radius=2, color=gc('green'), thickness=-1)
        cv2.circle(self.image, (int(intersection_x), int(intersection_y)), radius=4, color=gc('black'), thickness=2)
        cv2.circle(self.image, (int(self.width_base_mid[0]), int(self.width_base_mid[1])), radius=2, color=gc('red'), thickness=-1)
        cv2.circle(self.image, (int(self.width_base_mid[0]), int(self.width_base_mid[1])), radius=4, color=gc('black'), thickness=2)

        print("The intersection point of the line and the middle polynomial is:", (intersection_x, intersection_y))

        

    def calc_direct_length(self):
        # Calculate the x-coordinate of the intersection point
        x_intersection = (self.wb_intercept_perpendicular - self.wb_intercept) / (self.wb_slope - self.wb_slope_perpendicular)

        # Calculate the y-coordinate of the intersection point
        y_intersection = self.wb_slope * x_intersection + self.wb_intercept

        # Store the intersection point as self.wb_origin
        self.wb_origin = np.array([x_intersection, y_intersection])

        # Calculate the distance between the intersection point and self.inter_point
        self.length_direct = np.linalg.norm(self.wb_origin - self.inter_point)
        # Plot a 2-pixel thick red line from self.wb_origin to self.inter_point
        cv2.line(self.image_angles, tuple(map(int, self.wb_origin)), tuple(map(int, self.inter_point)), gc('red'), thickness=2)



    def calc_curvature_radius(self):
        def fit_circle_least_squares(points):
            if len(points) <= 1:
                return 0.0, (0, 0)

            def calc_residuals(params, points):
                x0, y0, r = params
                residuals = np.sqrt((points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2) - r
                return residuals

            def objective(params, points):
                return np.sum(calc_residuals(params, points) ** 2)

            x_mean = np.mean(points[:, 0])
            y_mean = np.mean(points[:, 1])
            r_mean = np.mean(np.sqrt((points[:, 0] - x_mean) ** 2 + (points[:, 1] - y_mean) ** 2))
            init_params = [x_mean, y_mean, r_mean]
            
            result = minimize(objective, init_params, args=(points,), method='L-BFGS-B')
            x0, y0, r = result.x

            return r, (x0, y0)
    
        self.radius_middle, center_middle = fit_circle_least_squares(self.ordered_middle_np)
        self.radius_outer_left, center_outer_left = fit_circle_least_squares(self.ordered_outer_left_np)
        self.radius_outer_right, center_outer_right = fit_circle_least_squares(self.ordered_outer_right_np)


        # Plot the circles on self.image_angles
        cv2.circle(self.image_angles, (int(center_middle[0]), int(center_middle[1])), int(self.radius_middle), gc('yellow'), thickness=1)
        cv2.circle(self.image_angles, (int(center_outer_left[0]), int(center_outer_left[1])), int(self.radius_outer_left), gc('pink'), thickness=1)
        cv2.circle(self.image_angles, (int(center_outer_right[0]), int(center_outer_right[1])), int(self.radius_outer_right), gc('cyan'), thickness=1)

        print('hi')
    

    def calc_angle_bend(self):
        print('hi')



    def calc_angle_curl(self):
        # Define the perpendicular line function
        def wb_line_perpendicular(x):
            return self.wb_slope_perpendicular * x + self.wb_intercept_perpendicular


        # Calculate the slope of the line perpendicular to the given line
        self.wb_slope_perpendicular = -1 / self.wb_slope
        # Calculate the y-intercept of the line perpendicular to the given line
        self.wb_intercept_perpendicular = self.inter_point[1] - self.wb_slope_perpendicular * self.inter_point[0]

        # Line fit to first 3 points in self.ordered_middle
        self.middle_tip_poly = np.polyfit(self.ordered_middle_np[0:3, 0], self.ordered_middle_np[0:3, 1], 1)
        middle_tip_slope = self.middle_tip_poly[0]

        # angle between middle_tip fit the curl perpendicular
        theta = math.atan(abs((middle_tip_slope - self.wb_slope_perpendicular) / (1 + self.wb_slope_perpendicular*middle_tip_slope)))

        # Convert the angle to degrees
        self.angle_curl = math.degrees(theta)

        print("The angle between the lines is:", self.angle_curl, "degrees")

        # Draw the tangents at the intersection point
        intersection_point = np.array(self.inter_point_outer_inner, dtype=int)
        length = 50  # Length of the tangent lines

        # Calculate the points for the tangent lines
        curl_tangent_point1 = (intersection_point[0] - length, intersection_point[1] - length * self.wb_slope_perpendicular)
        curl_tangent_point2 = (intersection_point[0] + length, intersection_point[1] + length * self.wb_slope_perpendicular)
        middle_tip_tangent_point1 = (intersection_point[0] - length, intersection_point[1] - length * middle_tip_slope)
        middle_tip_tangent_point2 = (intersection_point[0] + length, intersection_point[1] + length * middle_tip_slope)

        # Convert the points to integers
        curl_tangent_point1 = tuple(map(int, curl_tangent_point1))
        curl_tangent_point2 = tuple(map(int, curl_tangent_point2))
        middle_tip_tangent_point1 = tuple(map(int, middle_tip_tangent_point1))
        middle_tip_tangent_point2 = tuple(map(int, middle_tip_tangent_point2))

        # Draw the tangent lines
        cv2.line(self.image_angles, intersection_point, curl_tangent_point1, gc('teal'), 1)
        cv2.line(self.image_angles, intersection_point, curl_tangent_point2, gc('teal'), 1)
        cv2.line(self.image_angles, intersection_point, middle_tip_tangent_point1, gc('teal'), 1)
        cv2.line(self.image_angles, intersection_point, middle_tip_tangent_point2, gc('teal'), 1)

        # Draw the arc representing the angle
        cv2.ellipse(self.image_angles, tuple(intersection_point), (length, length), 0, 0, self.angle_curl, gc('teal'), 2)
        cv2.ellipse(self.image_angles, tuple(intersection_point), (length, length), 180, 0, self.angle_curl, gc('teal'), 2)

        ### plot the wb_line_perpendicular
        # Calculate the y values for the start and end points of the line
        y_start = max(0, int(wb_line_perpendicular(0)))
        y_end = min(self.height, int(wb_line_perpendicular(self.width)))

        # Define the range of y values for the line
        y_range = np.linspace(y_start, y_end, num=100, dtype=int)  # You can adjust 'num' to control the number of points

        # Draw the dotted gray line
        for i in range(len(y_range) - 1):
            y1, x1 = y_range[i], int((y_range[i] - self.wb_intercept_perpendicular) / self.wb_slope_perpendicular)
            x1 = max(0, min(x1, self.width))  # Keep x1 within the bounds of the image width
            y2, x2 = y_range[i+1], int((y_range[i+1] - self.wb_intercept_perpendicular) / self.wb_slope_perpendicular)
            x2 = max(0, min(x2, self.width))  # Keep x2 within the bounds of the image width

            if i % 2 == 0:  # Change the value of 2 to adjust the spacing between the dots
                cv2.line(self.image_angles, (x1, y1), (x2, y2), gc('white'), 1)




    def calc_angle_tangent(self):
        # Define the polynomial functions
        def left_func(x):
            return self.left_poly[0]*x**2 + self.left_poly[1]*x + self.left_poly[2]

        def right_func(x):
            return self.right_poly[0]*x**2 + self.right_poly[1]*x + self.right_poly[2]

        # Define the difference function
        def left_right_diff(x):
            return left_func(x) - right_func(x)

        # Find the x-coordinate of the intersection point
        intersection_x = fsolve(left_right_diff, 0)[0]

        # Calculate the y-coordinate of the intersection point on the left and right curves
        intersection_y_left = left_func(intersection_x)
        intersection_y_right = right_func(intersection_x)

        # Calculate the derivatives of the polynomials at the intersection point
        left_derivative = 2*self.left_poly[0]*intersection_x + self.left_poly[1]
        right_derivative = 2*self.right_poly[0]*intersection_x + self.right_poly[1]

        # Calculate the angle between the tangents to the polynomials at the intersection point
        theta = math.atan(abs((right_derivative - left_derivative) / (1 + left_derivative*right_derivative)))

        # Convert the angle to degrees
        self.angle_tangent = math.degrees(theta)

        print("The angle between the left and right polynomials at their point of intersection is:", theta, "degrees")

        # Draw the tangents at the intersection point
        intersection_point = np.array([int(intersection_x), int(intersection_y_left + (intersection_y_right - intersection_y_left)/2)])
        length = 30  # Length of the tangent lines

        # Calculate the points for the tangent lines
        left_tangent_point1 = (intersection_point[0] - length, intersection_point[1] - length * left_derivative)
        left_tangent_point2 = (intersection_point[0] + length, intersection_point[1] + length * left_derivative)
        right_tangent_point1 = (intersection_point[0] - length, intersection_point[1] - length * right_derivative)
        right_tangent_point2 = (intersection_point[0] + length, intersection_point[1] + length * right_derivative)

        # Convert the points to integers
        left_tangent_point1 = tuple(map(int, left_tangent_point1))
        left_tangent_point2 = tuple(map(int, left_tangent_point2))
        right_tangent_point1 = tuple(map(int, right_tangent_point1))
        right_tangent_point2 = tuple(map(int, right_tangent_point2))

        # # Draw the tangent lines
        # cv2.line(self.image_angles, intersection_point, left_tangent_point1, gc('yellow'), 1)
        # cv2.line(self.image_angles, intersection_point, left_tangent_point2, gc('yellow'), 1)
        # cv2.line(self.image_angles, intersection_point, right_tangent_point1, gc('yellow'), 1)
        # cv2.line(self.image_angles, intersection_point, right_tangent_point2, gc('yellow'), 1)

        # Draw the arc representing the angle
        cv2.ellipse(self.image_angles, tuple(intersection_point), (length, length), 0, 0, self.angle_tangent, gc('yellow'), 2)
        cv2.ellipse(self.image_angles, tuple(intersection_point), (length, length), 180, 0, self.angle_tangent, gc('yellow'), 2)

        # self.show_angle_image()
        # return theta


    def group_outer_points(self):
        # Split the points into two groups based on their position relative to the line
        self.outer_left = []
        self.outer_right = []

        # if 'tip' in self.points_list:

        for point in self.points_list['outer']:
            x, y = point
            predicted_y = self.predict_y(x)

            if y > predicted_y:
                self.outer_right.append(point)
            else:
                self.outer_left.append(point)

        self.outer_right = np.array(self.outer_right)
        self.outer_left = np.array(self.outer_left)

        if (len(self.outer_right) < 3) or (len(self.outer_left) < 3):
            self.keep_going = False
        else:
            # Plot `outer_left` points in pink
            for point in self.outer_left:
                x, y = point
                cv2.circle(self.image, (x, y), radius=5, color=gc('pink'), thickness=-1)

            # Plot `outer_right` points in cyan
            for point in self.outer_right:
                x, y = point
                cv2.circle(self.image, (x, y), radius=5, color=gc('cyan'), thickness=-1)

            ### outer_left
            self.outer_left = self.order_points(self.outer_left)
            self.outer_left = self.remove_duplicate_points(self.outer_left)
            # self.outer_left = self.check_momentum(self.outer_left, False)
            self.order_points_plot(self.outer_left, 'outer_left', 'final')
            self.order_points_plot(self.outer_left, 'outer_left', 'QC')
            self.outer_left_length, self.outer_left = self.get_length_of_ordered_points(self.outer_left, 'outer_left')
            self.has_outer_left = True
                

            ### outer_right
            self.outer_right = self.order_points(self.outer_right)
            self.outer_right = self.remove_duplicate_points(self.outer_right)
            # self.outer_right = self.check_momentum(self.outer_right, False)
            self.order_points_plot(self.outer_right, 'outer_right', 'final')
            self.order_points_plot(self.outer_right, 'outer_right', 'QC')
            self.outer_right_length, self.outer_right = self.get_length_of_ordered_points(self.outer_right, 'outer_right')
            self.has_middle = True

            print(f"Length outer_left - {self.outer_left_length}")
            print(f"Length outer_right - {self.outer_right_length}")

            self.outer_right_np = np.array(self.outer_right)
            self.outer_left_np = np.array(self.outer_left)
            self.ordered_middle_np = np.array(self.ordered_middle)

            # Fit 2nd order polynomials to the line segments
            self.left_poly = np.polyfit(self.outer_left_np[:, 0], self.outer_left_np[:, 1], 2)
            self.right_poly = np.polyfit(self.outer_right_np[:, 0], self.outer_right_np[:, 1], 2)
            self.middle_poly = np.polyfit(self.ordered_middle_np[:, 0], self.ordered_middle_np[:, 1], 2)


            # Evaluate polynomial coefficients for a range of x values
            x_range = np.linspace(0, self.width, num=100)
            left_line = np.polyval(self.left_poly, x_range)
            right_line = np.polyval(self.right_poly, x_range)
            self.middle_line = np.polyval(self.middle_poly, x_range)

            # Plot lines of fit as white lines
            for i in range(len(x_range)-1):
                cv2.line(self.image, (int(x_range[i]), int(left_line[i])), (int(x_range[i+1]), int(left_line[i+1])), color=gc('gray'), thickness=1)
                cv2.line(self.image, (int(x_range[i]), int(right_line[i])), (int(x_range[i+1]), int(right_line[i+1])), color=gc('white'), thickness=1)
                cv2.line(self.image, (int(x_range[i]), int(self.middle_line[i])), (int(x_range[i+1]), int(self.middle_line[i+1])), color=gc('white'), thickness=2)

            # Define the polynomial functions
            def left_func(x):
                return self.left_poly[0]*x**2 + self.left_poly[1]*x + self.left_poly[2]

            def right_func(x):
                return self.right_poly[0]*x**2 + self.right_poly[1]*x + self.right_poly[2]

            def middle_func(x):
                return self.middle_poly[0]*x**2 + self.middle_poly[1]*x + self.middle_poly[2]

            # Define the difference functions
            def left_middle_diff(x):
                return left_func(x) - middle_func(x)

            def right_middle_diff(x):
                return right_func(x) - middle_func(x)

            def left_right_diff(x):
                return left_func(x) - right_func(x)

            # Find the intersection points
            left_middle_intersection_x = fsolve(left_middle_diff, 0)
            right_middle_intersection_x = fsolve(right_middle_diff, 0)
            left_right_intersection_x = fsolve(left_right_diff, 0)

            left_middle_intersection_y = left_func(left_middle_intersection_x)[0]
            right_middle_intersection_y = right_func(right_middle_intersection_x)[0]
            left_right_intersection_y = left_func(left_right_intersection_x)[0]

            # Keep only points within the image boundaries
            intersection_points = np.array([[left_middle_intersection_x, left_middle_intersection_y], [right_middle_intersection_x, right_middle_intersection_y], [left_right_intersection_x, left_right_intersection_y]])
            intersection_points = intersection_points[(intersection_points[:, 0] >= 0) & (intersection_points[:, 0] <= self.width) & (intersection_points[:, 1] >= 0) & (intersection_points[:, 1] <= self.height)]

            if intersection_points.size == 0:
                self.keep_going = False
            else:
                # Compute the average of the intersection points
                intersection_x = np.mean(intersection_points[:, 0])
                intersection_y = np.mean(intersection_points[:, 1])

                self.inter_point = [int(intersection_x), int(intersection_y)]
                self.inter_point_outer_inner = [int(left_right_intersection_x), int(left_right_intersection_y)]

                # Draw intersection point on the image
                cv2.circle(self.image, (int(intersection_x), int(intersection_y)), radius=5, color=gc('green'), thickness=-1)
                print(f"Length outer_left - {self.outer_left_length}")
                print(f"Length outer_right - {self.outer_right_length}")
                print(f"Intersection point - ({int(intersection_x)}, {int(intersection_y)})")

                # Make the first points be at the tip, last points far away at base
                def reorder_segment(segment, inter):
                    # Convert to numpy arrays for easier manipulation
                    segment = np.array(segment)
                    inter = np.array(inter)

                    # Calculate the Euclidean distance from the INTER point to the first and last points in the segment
                    dist_first = np.linalg.norm(segment[0] - inter)
                    dist_last = np.linalg.norm(segment[-1] - inter)

                    # If the last point is closer to the INTER point than the first point, reverse the order of the segment
                    if dist_last < dist_first:
                        segment = segment[::-1]

                    return segment.tolist()
                
                self.ordered_middle = reorder_segment(self.ordered_middle, self.inter_point)
                self.outer_left = reorder_segment(self.outer_left, self.inter_point)
                self.outer_right = reorder_segment(self.outer_right, self.inter_point)

                self.ordered_outer_right_np = np.array(self.outer_right)
                self.ordered_outer_left_np = np.array(self.outer_left)
                self.ordered_middle_np = np.array(self.ordered_middle)
                
                # Draw a black ring around the last point of the outer_left segment
                self.last_point_left = self.outer_left[-1]
                cv2.circle(self.image, (int(self.last_point_left[0]), int(self.last_point_left[1])), radius=4, color=gc('black'), thickness=2)
                cv2.circle(self.image, (int(self.last_point_left[0]), int(self.last_point_left[1])),  radius=6, color=gc('white'), thickness=2)

                # Draw a black ring around the last point of the outer_right segment
                self.last_point_right = self.outer_right[-1]
                cv2.circle(self.image, (int(self.last_point_right[0]), int(self.last_point_right[1])), radius=4, color=gc('black'), thickness=2)
                cv2.circle(self.image, (int(self.last_point_right[0]), int(self.last_point_right[1])), radius=6, color=gc('white'), thickness=2)

                # self.show_QC_image()
                # print('hi')
        



    def split_image_by_middle(self):
        
        if not self.has_middle:
            self.keep_going = False
        else:
            n_fit = 2

            # Convert the points to a numpy array
            points_arr = np.array(self.ordered_middle)

            # Fit a line to the points
            self.midvein_fit = np.polyfit(points_arr[:, 0], points_arr[:, 1], n_fit)

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
            y1 = int(self.midvein_fit[0] * x1**2 + self.midvein_fit[1] * x1 + self.midvein_fit[2])
            x2 = self.width - 1
            y2 = int(self.midvein_fit[0] * x2**2 + self.midvein_fit[1] * x2 + self.midvein_fit[2])

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
            y_vals = self.midvein_fit[0] * x_vals**2 + self.midvein_fit[1] * x_vals + self.midvein_fit[2]

            # Remove any points that are outside the bounds of the image
            indices = np.where((y_vals >= 0) & (y_vals < self.height))[0]
            x_vals = x_vals[indices]
            y_vals = y_vals[indices]

            # Recompute y-values using the line equation and updated x-values
            y_vals = self.midvein_fit[0] * x_vals + self.midvein_fit[1]

            self.midvein_fit_points = np.column_stack((x_vals, y_vals))
            self.is_split = True

            # Draw line of fit
            # for point in self.midvein_fit_points:
            #     cv2.circle(self.image, tuple(point.astype(int)), radius=1, color=(255, 255, 255), thickness=-1)

    def predict_y(self, x):
        return self.midvein_fit[0] * x**2 + self.midvein_fit[1] * x + self.midvein_fit[2]

    def order_middle(self):
        
        
        if 'middle' not in self.points_list:
            self.keep_going = False
        else:
            if len(self.points_list['middle']) >= 5:
                self.logger.debug(f"Ordered Middle - Raw list contains {len(self.points_list['middle'])} points - using momentum")
                self.ordered_middle = self.order_points(self.points_list['middle'])
                self.ordered_middle = self.remove_duplicate_points(self.ordered_middle)

                self.ordered_middle = self.check_momentum(self.ordered_middle, False)

                self.v_tip = self.find_v_tip(self.points_list['outer'])
                # self.ordered_middle.append(self.v_tip)


                self.order_points_plot(self.ordered_middle, 'middle', 'QC')
                self.ordered_middle_length, self.ordered_middle = self.get_length_of_ordered_points(self.ordered_middle, 'middle')


                self.has_middle = True
            else:
                self.keep_going = False
                self.logger.debug(f"Ordered Middle - Raw list contains {len(self.points_list['middle'])} points - SKIPPING MIDDLE")

    def v_shape_template(self, tip, scale):
        return np.array([
            [tip[0] - scale, tip[1] + scale],
            tip,
            [tip[0] + scale, tip[1] + scale]
        ])

    def error_function(self, params, points):
        tip = params[:2]
        scale = params[2]
        template_points = self.v_shape_template(tip, scale)

        error = 0
        for p in points:
            dist = np.min(np.linalg.norm(template_points - p, axis=1))
            error += dist

        return error

    def find_v_tip(self, points):
        points = np.array(points)
        initial_guess = np.mean(points, axis=0)
        initial_scale = np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)) / 2

        result = minimize(
            self.error_function,
            np.hstack([initial_guess, initial_scale]),
            args=(points,),
            method='Nelder-Mead'
        )

        tip = result.x[:2]
        return tuple(map(int, tip))
    
    def show_QC_image(self):
        if self.do_show_QC_images:
            cv2.imshow('QC image', self.image)
            cv2.waitKey(0)
    
    def show_angle_image(self):
        if self.do_show_QC_images:
            cv2.imshow('Angles image', self.image_angles)
            cv2.waitKey(0)

    def show_final_image(self):
        if self.do_show_final_images:
            cv2.imshow('Final image', self.image_final)
            cv2.waitKey(0)

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
    
    def order_points_plot(self, points, version, QC_or_final):
        # thk_base = 0
        thk_base = 16

        if version == 'middle':
            # color = (0, 255, 0) 
            color = gc('green')  # blue
            thick = 1 #2 + thk_base
        elif version == 'tip':
            color = gc('green')
            thick = 1 #2 + thk_base
        elif version == 'outer':
            color = gc('red')
            thick = 1 #2 + thk_base
        elif version == 'outer_left':
            color = gc('pink')
            thick = 1 #2 + thk_base
        elif version == 'outer_right':
            color = gc('cyan')
            thick = 1 #2 + thk_base

            
        # elif version == 'lamina_width_alt':
        #     color = (100, 100, 255)
        #     thick = 2 + thk_base
        # elif version == 'not_reflex':
        #     color = (200, 0, 123)
        #     thick = 3 + thk_base
        # elif version == 'reflex':
        #     color = (0, 120, 200)
        #     thick = 3 + thk_base
        # elif version == 'petiole_tip_alt':
        #     color = (255, 55, 100)
        #     thick = 1 + thk_base
        # elif version == 'petiole_tip':
        #     color = (100, 255, 55)
        #     thick = 1 + thk_base
        # elif version == 'failed_angle':
        #     color = (0, 0, 0)
        #     thick = 3 + thk_base
        # Convert the points to a numpy array and round to integer values
        points_arr = np.round(np.array(points)).astype(int)

        # Draw a green line connecting all of the points
        if QC_or_final == 'QC':
            for i in range(len(points_arr) - 1):
                cv2.line(self.image, tuple(points_arr[i]), tuple(points_arr[i+1]), color, thick)
        else:
            for i in range(len(points_arr) - 1):
                cv2.line(self.image_final, tuple(points_arr[i]), tuple(points_arr[i+1]), color, thick)
                
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

    def remove_duplicate_points(self, points):
        unique_set = set()
        new_list = []

        for item in points:
            if item not in unique_set:
                unique_set.add(item)
                new_list.append(item)
        return new_list
    
    def distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    ### Shortest distance
    def order_points(self, points):
        points = [tuple(point) for point in points]  # Convert numpy.ndarray points to tuples

        best_tour = None
        shortest_tour_length = float('inf')

        for start_point in points:
            tour = [start_point]
            unvisited = set(points) - {start_point}

            while unvisited:
                nearest = min(unvisited, key=lambda point: self.distance(tour[-1], point))
                tour.append(nearest)
                unvisited.remove(nearest)

            # Calculate the length of the current tour
            tour_length = sum(self.distance(tour[i - 1], tour[i]) for i in range(1, len(tour)))

            # Update the best_tour if the current tour is shorter
            if tour_length < shortest_tour_length:
                shortest_tour_length = tour_length
                best_tour = tour

        return best_tour

    
    ### Smoothest
    '''
    def angle_between_points(self, p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return angle

    def order_points(self, points):
        points = [tuple(point) for point in points]  # Convert numpy.ndarray points to tuples

        best_tour = None
        largest_sum_angles = 0

        for start_point in points:
            tour = [start_point]
            unvisited = set(points) - {start_point}

            while unvisited:
                nearest = min(unvisited, key=lambda point: self.distance(tour[-1], point))
                tour.append(nearest)
                unvisited.remove(nearest)

            # Calculate the sum of angles for the current tour
            sum_angles = sum(self.angle_between_points(tour[i - 1], tour[i], tour[i + 1]) for i in range(1, len(tour) - 1))

            # Update the best_tour if the current tour has a larger sum of angles
            if sum_angles > largest_sum_angles:
                largest_sum_angles = sum_angles
                best_tour = tour

        return best_tour
    '''
    ### ^^^ Smoothest



    
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

    def setup_final_image(self):
        self.image_final = cv2.imread(os.path.join(self.dir_temp, '.'.join([self.file_name, 'jpg'])))

        if self.leaf_type == 'Landmarks_Armature':
            self.path_image_final = os.path.join(self.Dirs.landmarks_armature_overlay_final, '.'.join([self.file_name, 'jpg']))
        
    def setup_QC_image(self):
        self.image = cv2.imread(os.path.join(self.dir_temp, '.'.join([self.file_name, 'jpg'])))

        if self.leaf_type == 'Landmarks_Armature':
            self.path_QC_image = os.path.join(self.Dirs.landmarks_armature_overlay_QC, '.'.join([self.file_name, 'jpg']))

    def setup_angle_image(self):
        self.image_angles = cv2.imread(os.path.join(self.dir_temp, '.'.join([self.file_name, 'jpg'])))

        if self.leaf_type == 'Landmarks_Armature':
            self.path_angles_image = os.path.join(self.Dirs.landmarks_armature_overlay_angles, '.'.join([self.file_name, 'jpg']))

    def define_landmark_classes(self):
        self.classes = {
            'tip': 0,
            'middle': 1,
            'outer': 2,
            }

    def set_cfg_values(self):
        self.do_show_QC_images = self.cfg['leafmachine']['landmark_detector_armature']['do_show_QC_images']
        self.do_save_QC_images = self.cfg['leafmachine']['landmark_detector_armature']['do_save_QC_images']
        self.do_show_final_images = self.cfg['leafmachine']['landmark_detector_armature']['do_show_final_images']
        self.do_save_final_images = self.cfg['leafmachine']['landmark_detector_armature']['do_save_final_images']

    def init_lists_dicts(self):
        # Initialize all lists and dictionaries
        self.classes = {}
        self.points_list = []
        self.image = []


        self.ordered_middle = []

        self.midvein_fit = []
        self.midvein_fit_points = []

        self.outer_right = []
        self.outer_left = []

        # self.ordered_outer_left = []
        # self.ordered_outer_right = []

        self.tip = []

        self.apex_left = []
        self.apex_right = []
        self.apex_center = []
        

        self.base_left = []
        self.base_right = []
        self.base_center = []
        self.lamina_base = []
        self.width_left = []
        self.width_right = []
    
    def get_final(self):
        self.image_final = np.hstack((self.image, self.image_angles))
        return self.image_final
    
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def gc(color):
    colors = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'pink': (255, 0, 255),
        'cyan': (255, 255, 0),
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'gray': (128, 128, 128),
        'orange': (0, 165, 255),
        'purple': (128, 0, 128),
        'lightpink': (203, 192, 255),
        'brown': (42, 42, 165),
        'navy': (128, 0, 0),
        'teal': (128, 128, 0),
    }
    return colors.get(color.lower(), (0, 0, 0))
