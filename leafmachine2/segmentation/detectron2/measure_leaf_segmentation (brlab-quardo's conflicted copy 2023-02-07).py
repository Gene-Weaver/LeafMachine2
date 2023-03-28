import os, json, random, glob, inspect, sys, cv2, itertools
from PIL import Image, ImageDraw
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from dataclasses import dataclass, field
import matplotlib.patches as mpl_patches
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
import math
from shapely.geometry import Polygon, Point, MultiPoint

currentdir = os.path.dirname(inspect.getfile(inspect.currentframe()))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(currentdir)
# from detect import run
sys.path.append(parentdir)

def polygon_properties(polygon, out_label, seg_name_short, cfg, img_cropped):
    do_get_efds = cfg['leafmachine']['leaf_segmentation']['calculate_elliptic_fourier_descriptors']
    efd_order = cfg['leafmachine']['leaf_segmentation']['elliptic_fourier_descriptor_order']
    find_minimum_bounding_box = cfg['leafmachine']['leaf_segmentation']['find_minimum_bounding_box']

    if efd_order is None:
        efd_order = 20
    else:
        efd_order = int(efd_order)

    polygon = np.asarray(polygon)
    first_point = polygon[0]
    polygon_closed = np.vstack([polygon, first_point])

    labels = ['leaf', 'petiole', 'hole']
    name_class = next((label for label in labels if label in out_label), None)  
    # poly_name = '_'.join([name_class, seg_name_short])

    # Create a Shapely Polygon object from the input list of coordinates
    shapely_points = MultiPoint(polygon_closed)
    shapely_polygon = Polygon(polygon_closed)

    polygon_contour = np.array(polygon_closed, dtype=int).reshape((-1,1,2))
    # contour_area = cv2.contourArea(polygon_contour) # same as shapely_polygon.area
    contour_perimeter = cv2.arcLength(polygon_contour, True)
    polygon_approx = np.array(cv2.approxPolyDP(polygon_contour, 0.0064 * contour_perimeter, True)).reshape(-1, 2)
    # Calculate the area of the polygon
    area = shapely_polygon.area
    # Calculate the perimeter of the polygon
    perimeter = shapely_polygon.length
    # Calculate the centroid of the polygon
    cx, cy = shapely_points.centroid.coords.xy
    centroid = (int(cx[0]), int(cy[0]))
    # shapely_polygon.convex_hull
    convex_hull = shapely_points.convex_hull.area
    # Calculate the convexity of the polygon
    try:
        convexity = area / convex_hull 
    except:
        convexity = -1
    # Calculate the concavity of the polygon
    if convexity == -1:
        concavity = -1
    else:
        concavity = 1 - convexity
    # Calculate the circularity of the polygon
    try:
        circularity = (4 * math.pi * area) / (perimeter * perimeter)
    except:
        circularity = -1
    # Calculate the degree of the polygon
    degree = len(polygon_closed)
    # Calculate the aspect ratio of the polygon
    aspect_ratio = shapely_polygon.bounds[2] / shapely_polygon.bounds[3]
    # bounding box
    bbox, polygon_closed_rotated, angle, bbox_min, max_length, min_length = fit_min_bbox(find_minimum_bounding_box, polygon, polygon_approx, int(cx[0]), int(cy[0]), tolerance=2)

    # efds
    dict_efd = calc_efds(do_get_efds, polygon_closed, img_cropped)

    
    top = str(min(bbox, key=lambda x: x[1])[1])
    bottom = str(max(bbox, key=lambda x: x[1])[1])
    left = str(min(bbox, key=lambda x: x[0])[0])
    right = str(max(bbox, key=lambda x: x[0])[0])
    poly_name = '-'.join([top, left, bottom, right])
    poly_name = '_'.join([name_class, poly_name])

    # return a dictionary of the properties
    props = {
        'bbox': bbox,
        'bbox_min': bbox_min, 
        'rotate_angle': angle,
        'long': max_length,
        'short': min_length,
        'efds': dict_efd,
        'area': area,
        'perimeter': perimeter,
        'centroid': centroid,
        'convex_hull': convex_hull,
        'convexity': convexity,
        'concavity': concavity,
        'circularity': circularity,
        'degree': degree,
        'aspect_ratio': aspect_ratio,
        'polygon_closed': polygon_closed,
        'polygon': polygon,
        'polygon_closed_rotated': polygon_closed_rotated
    }
    out_dict = {poly_name: props}
    return out_dict, bbox_min

def fit_min_bbox(find_minimum_bounding_box, polygon, polygon_approx, c_x, c_y, tolerance=2):
    polygon_orig = polygon
    # polygon_vert, angle_vert = make_polygon_vertical(polygon, c_x, c_y)
    x, y, w, h = cv2.boundingRect(polygon)
    rect_orig = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    edge_lengths = [math.sqrt((rect_orig[i][0]-rect_orig[i+1][0])**2 + (rect_orig[i][1]-rect_orig[i+1][1])**2) for i in range(3)]
    max_length = max(edge_lengths)
    min_length = min(edge_lengths)
    (cir_x, cir_y), radius = cv2.minEnclosingCircle(polygon)
    angle = 0

    if find_minimum_bounding_box:
        if abs(max_length - (2 * radius)) < tolerance:
            return rect_orig, polygon_orig, angle, rect_orig, max_length, min_length
        else:
            # angle = -90
            # rotated_polygon = polygon
            while abs(max_length - (2 * radius)) >= tolerance:
                if angle > 180:
                    tolerance += 2
                    angle = 0
                angle += 1
                rotated_polygon = rotate_polygon_by_angle(polygon_approx, angle, c_x, c_y)
                x, y, w, h = cv2.boundingRect(np.asarray(rotated_polygon).astype(int))
                rect = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                edge_lengths = [math.sqrt((rect[i][0]-rect[i+1][0])**2 + (rect[i][1]-rect[i+1][1])**2) for i in range(3)]
                max_length = max(edge_lengths)
                min_length = min(edge_lengths)
                if tolerance > 4:
                    return rect_orig, polygon_orig, 0, rect_orig, max_length, min_length
            # return rect_orig, rotate_polygon_by_angle(rotated_polygon, -angle, c_x, c_y), angle, rotate_polygon_by_angle(rect, -angle, c_x, c_y), max_length, min_length
            rotated_polygon = rotate_polygon_by_angle(polygon, angle, c_x, c_y)
            return rect_orig, rotate_polygon_by_angle(rotated_polygon, -angle, c_x, c_y), angle, rotate_polygon_by_angle(rect, -angle, c_x, c_y), max_length, min_length
    else:
        return rect_orig, polygon_orig, angle, rect_orig, 2*radius, None

def rotate_polygon_by_angle(polygon, angle, c_x, c_y):
    # Translate polygon to origin
    translated_polygon = [(x-c_x, y-c_y) for x, y in polygon]
    # Rotate the polygon by the given angle
    rotated_polygon = [(x*math.cos(math.radians(angle)) - y*math.sin(math.radians(angle)),
                        x*math.sin(math.radians(angle)) + y*math.cos(math.radians(angle))) for x, y in translated_polygon]
    # Translate polygon back
    rotated_polygon = [(x+c_x, y+c_y) for x, y in rotated_polygon]
    return rotated_polygon

def make_polygon_vertical(polygon, c_x, c_y, angle=None):
    # Translate polygon to origin
    translated_polygon = [(x-c_x, y-c_y) for x, y in polygon]

    # Find the angle of the primary axis of the polygon
    x1, y1, x2, y2 = translated_polygon[0][0], translated_polygon[0][1], translated_polygon[1][0], translated_polygon[1][1]
    angle = math.atan2(y2 - y1, x2 - x1)

    # Rotate the polygon by the angle of the primary axis
    rotated_polygon = [(x*math.cos(angle) - y*math.sin(angle),
                        x*math.sin(angle) + y*math.cos(angle)) for x, y in translated_polygon]

    # Translate polygon back
    rotated_polygon = [(x+c_x, y+c_y) for x, y in rotated_polygon]
    return rotated_polygon, angle

def calc_efds(do_get_efds, polygon, img_cropped):
    if not do_get_efds:
        return []
    else:
        # Create a cloased polygon
        polygon = np.asarray(polygon)
        first_point = polygon[0]
        closed_polygon = polygon#np.vstack([polygon, first_point])

        # Get efds
        coeffs_normalized, coeffs_orig, transformations = elliptic_fourier_descriptors(np.squeeze(closed_polygon), order=40)
        # Retrieve transformation params
        scale = transformations[0]
        angle = transformations[1]
        phase = transformations[2]

        coeffs_features = coeffs_normalized.flatten()[3:]
        a0, c0 = calculate_dc_coefficients(closed_polygon)
        fig_efd_cv2, pts_contour, pts_efd, pts_efd_PIL = plot_efd(coeffs_orig, (a0,c0), img_cropped, closed_polygon, 300, scale, angle, phase)

        efd_area = cv2.contourArea(closed_polygon)
        efd_perimeter = cv2.arcLength(closed_polygon, True)

        dict_efd = {}
        dict_efd["coeffs_normalized"] = coeffs_normalized
        dict_efd["coeffs_features"] = coeffs_features
        dict_efd["a0"] = a0
        dict_efd["c0"] = c0
        dict_efd["scale"] = scale
        dict_efd["angle"] = angle
        dict_efd["phase"] = phase
        dict_efd["efd_fig_cv2"] = fig_efd_cv2
        dict_efd["original_contour"] = pts_contour
        dict_efd["efd_pts"] = pts_efd
        dict_efd["efd_pts_PIL"] = pts_efd_PIL
        dict_efd["efd_area"] = efd_area
        dict_efd["efd_perimeter"] = efd_perimeter
        return dict_efd

'''We use the library pyefd for Fourier analysis
BUT we could not get it to work as is, so we modified the functions here

from pyefd import normalize_efd, elliptic_fourier_descriptors, calculate_dc_coefficients

Here are the modified versions:
    replacing division with np.divide()
    changing zeros to: dt[dt == 0] = 1e-9
'''

def normalize_efd(coeffs, size_invariant=True, return_transformation=False):
    coeffs_orig = coeffs
    # Make the coefficients have a zero phase shift from
    # the first major axis. Theta_1 is that shift angle.
    theta_1 = 0.5 * np.arctan2(
        2 * ((coeffs[0, 0] * coeffs[0, 1]) + (coeffs[0, 2] * coeffs[0, 3])),
        (
            (coeffs[0, 0] ** 2)
            - (coeffs[0, 1] ** 2)
            + (coeffs[0, 2] ** 2)
            - (coeffs[0, 3] ** 2)
        ),
    )
    # Rotate all coefficients by theta_1.
    for n in range(1, coeffs.shape[0] + 1):
        coeffs[n - 1, :] = np.dot(
            np.array(
                [
                    [coeffs[n - 1, 0], coeffs[n - 1, 1]],
                    [coeffs[n - 1, 2], coeffs[n - 1, 3]],
                ]
            ),
            np.array(
                [
                    [np.cos(n * theta_1), -np.sin(n * theta_1)],
                    [np.sin(n * theta_1), np.cos(n * theta_1)],
                ]
            ),
        ).flatten()

    # Make the coefficients rotation invariant by rotating so that
    # the semi-major axis is parallel to the x-axis.
    psi_1 = np.arctan2(coeffs[0, 2], coeffs[0, 0])
    psi_rotation_matrix = np.array(
        [[np.cos(psi_1), np.sin(psi_1)], [-np.sin(psi_1), np.cos(psi_1)]]
    )
    # Rotate all coefficients by -psi_1.
    for n in range(1, coeffs.shape[0] + 1):
        coeffs[n - 1, :] = psi_rotation_matrix.dot(
            np.array(
                [
                    [coeffs[n - 1, 0], coeffs[n - 1, 1]],
                    [coeffs[n - 1, 2], coeffs[n - 1, 3]],
                ]
            )
        ).flatten()

    size = coeffs[0, 0]
    if size_invariant:
        # Obtain size-invariance by normalizing.
        coeffs /= np.abs(size)

    if return_transformation:
        return coeffs, coeffs_orig, (size, psi_1, theta_1)
    else:
        return coeffs, coeffs_orig

def elliptic_fourier_descriptors(contour, order=10, normalize=True, return_transformation=True):

    dxy = np.diff(contour, axis=0)
    dt = np.sqrt((dxy ** 2).sum(axis=1))
    dt[dt == 0] = 1e-9
    t = np.concatenate([([0.0]), np.cumsum(dt)])
    
    T = t[-1]

    phi = np.divide((2 * np.pi * t), T)

    orders = np.arange(1, order + 1)
    consts = np.divide(T, (2 * orders * orders * np.pi * np.pi))
    phi = phi * orders.reshape((order, -1))

    d_cos_phi = np.cos(phi[:, 1:]) - np.cos(phi[:, :-1])
    d_sin_phi = np.sin(phi[:, 1:]) - np.sin(phi[:, :-1])

    a = consts * np.sum((np.divide(dxy[:, 0], dt)) * d_cos_phi, axis=1)
    b = consts * np.sum((np.divide(dxy[:, 0], dt)) * d_sin_phi, axis=1)
    c = consts * np.sum((np.divide(dxy[:, 1], dt)) * d_cos_phi, axis=1)
    d = consts * np.sum((np.divide(dxy[:, 1], dt)) * d_sin_phi, axis=1)

    coeffs = np.concatenate(
        [
            a.reshape((order, 1)),
            b.reshape((order, 1)),
            c.reshape((order, 1)),
            d.reshape((order, 1)),
        ],
        axis=1,
    )
    coeffs_original = coeffs.copy()
    if normalize:
        coeffs_normmalized, coeffs_2, trans = normalize_efd(coeffs, return_transformation=return_transformation)

    return coeffs_normmalized, coeffs_original, trans

def calculate_dc_coefficients(contour):

    dxy = np.diff(contour, axis=0)
    dt = np.sqrt((dxy ** 2).sum(axis=1))
    dt[dt == 0] = 1e-9
    t = np.concatenate([([0.0]), np.cumsum(dt)])
    T = t[-1]

    xi = np.cumsum(dxy[:, 0]) - np.divide(dxy[:, 0], dt) * t[1:]
    A0 = np.divide(1, T) * np.sum((np.divide(dxy[:, 0], (2 * dt)) * np.diff(t ** 2)) + xi * dt)
    delta = np.cumsum(dxy[:, 1]) - np.divide(dxy[:, 1], dt) * t[1:]
    C0 = np.divide(1, T) * np.sum((np.divide(dxy[:, 1], (2 * dt)) * np.diff(t ** 2)) + delta * dt)

    # A0 and CO relate to the first point of the contour array as origin.
    # Adding those values to the coefficients to make them relate to true origin.
    return contour[0, 0] + A0, contour[0, 1] + C0

def plot_efd(coeffs, locus, image, contour, samp, scale, angle, theta):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    t = np.linspace(0, 1.0, samp)
    xt = np.ones((samp,)) * locus[0]
    yt = np.ones((samp,)) * locus[1]

    for n in range(coeffs.shape[0]):
        xt += (coeffs[n, 0] * np.cos(2 * (n + 1) * np.pi * t)) + (
            coeffs[n, 1] * np.sin(2 * (n + 1) * np.pi * t)
        )
        yt += (coeffs[n, 2] * np.cos(2 * (n + 1) * np.pi * t)) + (
            coeffs[n, 3] * np.sin(2 * (n + 1) * np.pi * t)
        )
    
    coords = np.array(np.column_stack((xt.astype(int), yt.astype(int))))
    coords_PIL = list(map(lambda x: (int(x[0]), int(x[1])), zip(xt, yt)))


    color = (0, 255, 0)
    thickness = 2
    isClosed = True
    image = cv2.polylines(image, [contour], isClosed, color, thickness)
    color = (0, 0, 255)
    image = cv2.polylines(image, [coords], isClosed, color, thickness)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    return image, contour, coords, coords_PIL
   