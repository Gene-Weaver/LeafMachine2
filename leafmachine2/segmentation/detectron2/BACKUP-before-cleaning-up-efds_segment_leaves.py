import os, json, random, glob, inspect, sys, cv2
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
from sklearn.linear_model import RANSACRegressor

from detector import Detector
currentdir = os.path.dirname(inspect.getfile(inspect.currentframe()))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(currentdir)
# from detect import run
sys.path.append(parentdir)
# from machine.general_utils import Print_Verbose

def segment_leaves(cfg, dir_home, Project, Dirs, dict_plant_components): 
    # See convert_index_to_class(ind) for list of ind -> cls
    dict_plant_components = unpack_class_from_components(dict_plant_components, 0, 'Whole_Leaf_BBoxes_YOLO', 'Whole_Leaf_BBoxes', Project)
    dict_plant_components = unpack_class_from_components(dict_plant_components, 1, 'Partial_Leaf_BBoxes_YOLO', 'Partial_Leaf_BBoxes', Project)

    # Crop the images to bboxes
    dict_plant_components = crop_images_to_bbox(dict_plant_components, 0, 'Whole_Leaf_Cropped', "Whole_Leaf_BBoxes", Project)
    dict_plant_components = crop_images_to_bbox(dict_plant_components, 1, 'Partial_Leaf_Cropped', "Partial_Leaf_BBoxes", Project)

    # Run the leaf instance segmentation operations
    dir_seg_model = os.path.join(dir_home, 'leafmachine2', 'segmentation', 'models',cfg['leafmachine']['leaf_segmentation']['segmentation_model'])
    Instance_Detector = Detector(dir_seg_model, cfg['leafmachine']['leaf_segmentation']['minimum_confidence_threshold'])

    if cfg['leafmachine']['leaf_segmentation']['segment_whole_leaves']:
        segment_images(Instance_Detector, dict_plant_components, 0, "Segmentation_Whole_Leaf", "Whole_Leaf_Cropped", cfg)
    if cfg['leafmachine']['leaf_segmentation']['segment_partial_leaves']:
        segment_images(Instance_Detector, dict_plant_components, 1, "Segmentation_Partial_Leaf", "Partial_Leaf_Cropped", cfg)

    


    # if cfg['leafmachine']['leaf_segmentation']['segment_whole_leaves']:
    
    # print(dict_plant_components)


    # imgs = os.path.abspath(os.path.join(DATASET,"*.jpg"))
    # imgs = glob.glob(imgs)
    # n = int(SUBSAMPLE*len(imgs))
    # imgs = random.sample(imgs, n)
    
    # for img in tqdm(imgs, desc=f'{bcolors.BOLD}Saving {n} images to {PDF_NAME}{bcolors.ENDC}',colour="green",position=0, total=n):
    #     fig = instance_detector.onImage(os.path.join(DATASET,img),SHOW_IMG)
    #     pdf.savefig(fig)

def segment_images(Instance_Detector, dict, cls, dict_name_seg, dict_from, cfg):
    generate_overlay = cfg['leafmachine']['leaf_segmentation']['generate_overlay']
    overlay_dpi = cfg['leafmachine']['leaf_segmentation']['overlay_dpi']
    bg_color = cfg['leafmachine']['leaf_segmentation']['overlay_background_color']

    for filename, value in dict.items():
        value[dict_name_seg] = []
        if value[dict_from] is not []:
            for cropped in value[dict_from]:
                for seg_name, img_cropped in cropped.items():
                    
                    print(seg_name)
                    # cropped_overlay = []

                    # Segment!
                    fig, out_polygons, out_bboxes, out_labels, out_color = Instance_Detector.segment(img_cropped, generate_overlay, overlay_dpi, bg_color)
                    
                    create_overlay_and_calculate_props(seg_name, img_cropped, out_polygons, out_labels, out_color, cfg)

                    # value[dict_name_seg].append({seg_name: })
                    
                    


                    # value[dict_name_seg].append({seg_name: })

def create_overlay_and_calculate_props(seg_name, img_cropped, out_polygons, out_labels, out_color, cfg):
    cropped_overlay = img_cropped
    # cropped_overlay = Image.fromarray(cv2.cvtColor(cropped_overlay, cv2.COLOR_BGR2RGB))
    cropped_overlay = Image.fromarray(cropped_overlay)
    draw = ImageDraw.Draw(cropped_overlay, "RGBA")

    # parse seg_name
    seg_name_short = seg_name.split("__")[2]

    # List of instances
    detected_components = []
    for i, polys in enumerate(out_polygons):
        # PIL
        # fill_color = (color_rgb[2], color_rgb[1], color_rgb[0], 127)
        # outline_color = (color_rgb[2], color_rgb[1], color_rgb[0])

        # cv2
        color_rgb = tuple(map(lambda x: int(x*255), out_color[i]))
        fill_color = (color_rgb[0], color_rgb[1], color_rgb[2], 127)
        outline_color = (color_rgb[0], color_rgb[1], color_rgb[2])
        
        max_poly = get_largest_polygon(polys)#, value['height'], value['width'])
        # cropped_overlay = plot_polygons_on_image(max_poly, cropped_overlay, color_rgb)#, 0.4) # cv2 implementation

        # calculate props
        component, bbox = polygon_properties(max_poly, out_labels[i], seg_name_short, cfg, img_cropped)
        detected_components.append(component)

        # draw poly
        draw.polygon(max_poly, fill=fill_color, outline=outline_color)
        draw.rectangle(bbox, outline=outline_color)
    # PIL
    # cropped_overlay.show() # wrong colors without changing to RGB
    # cv2
    cropped_overlay = np.array(cropped_overlay)
    # cv2.imshow('img_crop', cropped_overlay)
    # cv2.waitKey(0)

def polygon_properties(polygon, out_label, seg_name_short, cfg, img_cropped):
    do_get_efds = cfg['leafmachine']['leaf_segmentation']['calculate_elliptic_fourier_descriptors']
    efd_order = cfg['leafmachine']['leaf_segmentation']['elliptic_fourier_descriptor_order']
    if efd_order is None:
        efd_order = 20
    else:
        efd_order = int(efd_order)

    polygon = np.asarray(polygon)
    first_point = polygon[0]
    polygon = np.vstack([polygon, first_point])

    labels = ['leaf', 'petiole', 'hole']
    name_class = next((label for label in labels if label in out_label), None)  
    poly_name = '_'.join([name_class, seg_name_short])

    # Create a Shapely Polygon object from the input list of coordinates
    shapely_points = MultiPoint(polygon)
    shapely_polygon = Polygon(polygon)

    polygon_contour = np.array(polygon, dtype=np.int32).reshape((-1,1,2))
    contour_area = cv2.contourArea(polygon_contour)
    contour_perimeter = cv2.arcLength(polygon_contour, True)
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
    convexity = area / convex_hull 
    # Calculate the concavity of the polygon
    concavity = 1 - convexity
    # Calculate the circularity of the polygon
    circularity = (4 * math.pi * area) / (perimeter * perimeter)
    # Calculate the degree of the polygon
    degree = len(polygon)
    # Calculate the aspect ratio of the polygon
    aspect_ratio = shapely_polygon.bounds[2] / shapely_polygon.bounds[3]
    # bounding box
    bbox = get_bounding_box(polygon)
    bbox = ((int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])))
    # min bbox
    bbox_min = shapely_points.minimum_rotated_rectangle
    bbox_min_x, bbox_min_y = bbox_min.exterior.coords.xy
    bbox_min_coords = ((int(bbox_min_x[0]), int(bbox_min_y[0])), (int(bbox_min_x[1]), int(bbox_min_y[1])), (int(bbox_min_x[2]), int(bbox_min_y[2])), (int(bbox_min_x[3]), int(bbox_min_y[3])))
    edge_length = (Point(bbox_min_x[0], bbox_min_y[0]).distance(Point(bbox_min_x[1], bbox_min_y[1])), Point(bbox_min_x[1], bbox_min_y[1]).distance(Point(bbox_min_x[2], bbox_min_y[2])))
    length = max(edge_length)
    width = min(edge_length)
    # efds
    if do_get_efds:
        test_efd_algorithm(polygon, img_cropped)
        efds = elliptic_fourier_descriptors(polygon)
    else:
        efds = []
    # return a dictionary of the properties
    props = {
        'bbox': bbox,
        'bbox_min': bbox_min_coords, 
        'length': length,
        'width': width,
        'efds': efds,
        'area': area,
        'perimeter': perimeter,
        'centroid': centroid,
        'convex_hull': convex_hull,
        'convexity': convexity,
        'concavity': concavity,
        'circularity': circularity,
        'degree': degree,
        'aspect_ratio': aspect_ratio
    }
    out_dict = {poly_name: props}

    return out_dict, bbox

def get_bounding_box(polygon):
    x_coords = [x for x, y in polygon]
    y_coords = [y for x, y in polygon]
    left = min(x_coords)
    right = max(x_coords)
    top = min(y_coords)
    bottom = max(y_coords)
    return (left, top, right, bottom)

# def elliptic_fourier_descriptors(polygon):
#     # Get the x and y coordinates of the polygon
#     x = [point[0] for point in polygon]
#     y = [point[1] for point in polygon]

#     # Calculate the contour length
#     n = len(x)
#     perimeter = 0.0
#     for i in range(n-1):
#         perimeter += np.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2)

#     # Calculate the EFDs
#     t = np.linspace(0, 2*np.pi, n)
#     efds = []
#     for k in range(1, n):
#         c_k = 0.0
#         s_k = 0.0
#         for i in range(n):
#             c_k += np.cos(k*t[i])*x[i] + np.sin(k*t[i])*y[i]
#             s_k += np.sin(k*t[i])*x[i] - np.cos(k*t[i])*y[i]
#         c_k /= n
#         s_k /= n
#         efds.append((c_k, s_k))
#     efds_clean = remove_outliers(efds)
#     return efds_clean

def remove_outliers(polygon):
    x = [point[0] for point in polygon]
    y = [point[1] for point in polygon]
    
    x_median = np.median(x)
    y_median = np.median(y)
    
    x_std = np.std(x)
    y_std = np.std(y)
    
    cleaned_polygon = []
    for point in polygon:
        x_val, y_val = point
        if abs(x_val - x_median) <= 2*x_std and abs(y_val - y_median) <= 2*y_std:
            cleaned_polygon.append(point)
    
    return cleaned_polygon

def test_efd_algorithm(polygon, img_cropped):
    polygon = np.asarray(polygon)
    first_point = polygon[0]
    closed_polygon = np.vstack([polygon, first_point])
    polygon_contour = np.array(polygon, dtype=np.int32).reshape((-1,1,2))
    # print(closed_polygon)

    # closed_polygon = np.swapaxes(closed_polygon,0,1)
    # print(closed_polygon)

    coeffs, transformations = elliptic_fourier_descriptors(np.squeeze(closed_polygon), order=40)
    scaling = transformations[0]
    angle = transformations[1]
    coeffs_normalized = normalize_efd(coeffs)
    coeffs_features = coeffs.flatten()[3:]
    a0, c0 = calculate_dc_coefficients(closed_polygon)
    plot_efd(coeffs, (a0,c0), img_cropped, closed_polygon, 300, scaling, angle)




    # Add the first point to the end of the polygon to close it
    polygon = np.asarray(polygon)
    first_point = polygon[0]
    closed_polygon = np.vstack([polygon, first_point])

    # Calculate the efds
    efds = elliptic_fourier_descriptors(closed_polygon)
    # Reconstruct the polygon from the efds
    reconstructed_polygon = reconstruct_polygon_from_efds(efds)
    # reconstructed_polygon = reconstruct_polygon_from_efds2(efds, closed_polygon)

    # Plot the original polygon and the reconstructed polygon

    """# Reconstruct the polygon from the EFDs
    reconstructed_polygon = reconstruct_polygon_from_efds(efds)

    # Plot the original polygon and the reconstructed polygon
    plt.plot(polygon[:,0], polygon[:,1], 'b-', label='Original Polygon')
    plt.plot(reconstructed_polygon[:,0], reconstructed_polygon[:,1], 'r-', label='Reconstructed Polygon')
    plt.legend()
    plt.show()"""

    plt.plot(polygon[:, 0], polygon[:, 1], 'g-', label='Original')
    plt.plot(reconstructed_polygon[:, 0], reconstructed_polygon[:, 1], 'r-', label='Reconstructed')
    plt.legend()
    plt.title("Original polygon vs Reconstructed polygon")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")

    # Save the plot to a file
    plt.savefig("original_vs_reconstructed.png")
    plt.show()

def reconstruct_polygon_from_efds(efds):
    n = len(efds)
    t = np.linspace(0, 2*np.pi, n)
    x = []
    y = []
    for i in range(n):
        x_val = 0
        y_val = 0
        for j in range(n-1):
            c_k, s_k = efds[j]
            x_val += (c_k * np.cos(j*t[i]) - s_k * np.sin(j*t[i]))
            y_val += (c_k * np.sin(j*t[i]) + s_k * np.cos(j*t[i]))
        x.append(x_val)
        y.append(y_val)
    return np.column_stack((x, y))

# def reconstruct_polygon_from_efds2(efds, polygon):
#     x = [point[0] for point in polygon]
#     y = [point[1] for point in polygon]
#     n = len(x)
#     perimeter = 0.0
#     for i in range(n-1):
#         perimeter += np.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2)
#     # Calculate the EFDs
#     t = np.linspace(0, 2*np.pi, n)
#     x_recon = []
#     y_recon = []
#     for i in range(n):
#         x_val = 0
#         y_val = 0
#         for j in range(len(efds)):
#             c_k, s_k = efds[j]
#             x_val += c_k * np.cos(j*t[i]) - s_k * np.sin(j*t[i])
#             y_val += c_k * np.sin(j*t[i]) + s_k * np.cos(j*t[i])
#         x_recon.append(x_val)
#         y_recon.append(y_val)
#     return np.column_stack((x_recon, y_recon))





# def test_efd_algorithm(polygon):
#     # Create a triangle polygon
#     # polygon = [(0, 0), (0, 1), (1, 0)]
#     polygon = np.asarray(polygon)
#     first_point = polygon[0]
#     closed_polygon = np.vstack([polygon, first_point])
#     # Calculate the efds
#     efds = elliptic_fourier_descriptors(polygon)

#     # Plot the original polygon in polar coordinates
#     plt.subplot(1, 2, 1, polar=True)
#     theta = np.linspace(0, 2*np.pi, len(polygon))
#     plt.polar(theta, polygon[:, 0], "b-")
#     plt.title("Original Polygon")

#     # Plot the efds in polar coordinates
#     plt.subplot(1, 2, 2, polar=True)
#     theta = np.linspace(0, 2*np.pi, len(efds))
#     plt.polar(theta, efds, "r-")
#     plt.title("EFDs")

#     # Save the plot to a file
#     plt.savefig("efds_triangle.png")
#     # plt.show()

def crop_images_to_bbox(dict, cls, dict_name_cropped, dict_from, Project):
    # For each image, iterate through the whole leaves, segment, report data back to dict_plant_components
    for filename, value in dict.items():
        value[dict_name_cropped] = []
        bboxes_whole_leaves = [val for val in value[dict_from] if val[0] == convert_index_to_class(cls)]
        if len(bboxes_whole_leaves) == 0:
            m = str(''.join(['No objects for class ', convert_index_to_class(0), ' were found']))
            # Print_Verbose(cfg, 3, m).plain()
        else:
            try:
                img = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpg'])))
                # img = cv2.imread(os.path.join(Project, '.'.join([filename,'jpg']))) # Testing
            except:
                img = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpeg'])))
                # img = cv2.imread(os.path.join(Project, '.'.join([filename,'jpeg']))) # Testing
            
            for d in bboxes_whole_leaves:
                # img_crop = img.crop((d[1], d[2], d[3], d[4])) # PIL
                img_crop = img[d[2]:d[4], d[1]:d[3]]
                loc = '-'.join([str(d[1]), str(d[2]), str(d[3]), str(d[4])])
                crop_name = '__'.join([filename,'L',loc])
                value[dict_name_cropped].append({crop_name: img_crop})
                # cv2.imshow('img_crop', img_crop)
                # cv2.waitKey(0)
                # img_crop.show() # PIL
    return dict

def unpack_class_from_components(dict, cls, dict_name_yolo, dict_name_location, Project):
    # Get the dict that contains plant parts, find the whole leaves
    for filename, value in dict.items():
        filtered_components = [val for val in value["Plant_Components"] if val[0] == cls]
        value[dict_name_yolo] = filtered_components

    for filename, value in dict.items():
        filtered_components = [val for val in value["Plant_Components"] if val[0] == cls]
        height = value['height']
        width = value['width']
        converted_list = [[convert_index_to_class(val[0]), int((val[1] * width) - ((val[3] * width) / 2)), 
                                                            int((val[2] * height) - ((val[4] * height) / 2)), 
                                                            int(val[3] * width) + int((val[1] * width) - ((val[3] * width) / 2)), 
                                                            int(val[4] * height) + int((val[2] * height) - ((val[4] * height) / 2))] for val in filtered_components]
        # Verify that the crops are correct
        # img = Image.open(os.path.join(Project., '.'.join([filename,'jpg'])))
        # for d in converted_list:
        #     img_crop = img.crop((d[1], d[2], d[3], d[4]))
        #     img_crop.show() 
        value[dict_name_location] = converted_list
    # print(dict)
    return dict

def plot_polygons_on_image(polygons, img, color):
    for polygon in polygons:
        # convert the list of points to a numpy array of shape Nx1x2
        polygon = np.array(polygon, dtype=np.int32).reshape((-1,1,2))
        # draw the polygon on the image
        cv2.polylines(img, [polygon], True, color, 2)
    # show the image with the polygons
    # cv2.imshow("Polygons", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img

def get_largest_polygon(polygons):
    # polygons = max(polygons, key=len)
    # Keep the polygon that has the most points
    polygons = [polygon for polygon in polygons if len(polygon) == max([len(p) for p in polygons])]
    # convert the list of polygons to a list of contours
    contours = [np.array(polygon, dtype=np.int32).reshape((-1,1,2)) for polygon in polygons]
    # filter the contours to only closed contours
    closed_contours = [c for c in contours if cv2.isContourConvex(c)]
    if len(closed_contours)>0:
        # sort the closed contours by area
        closed_contours = sorted(closed_contours, key=cv2.contourArea, reverse=True)
        # take the largest closed contour
        largest_closed_contour = closed_contours[0]
    else:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_closed_contour = contours[0]
    largest_polygon = [tuple(i[0]) for i in largest_closed_contour]
    return largest_polygon

def convert_index_to_class(ind):
    mapping = {
        0: 'Leaf_WHOLE',
        1: 'Leaf_PARTIAL',
        2: 'Leaflet',
        3: 'Seed_Fruit_ONE',
        4: 'Seed_Fruit_MANY',
        5: 'Flower_ONE',
        6: 'Flower_MANY',
        7: 'Bud',
        8: 'Specimen',
        9: 'Roots',
        10: 'Wood'
    }
    return mapping.get(ind, 'Invalid class').lower()


'''We use the library pyefd for Fourier analysis
BUT we could not get it to work as is, so we modified the functions here

from pyefd import normalize_efd, elliptic_fourier_descriptors, calculate_dc_coefficients

Here are the modified versions:
    replacing division with np.divide()
    changing zeros to: dt[dt == 0] = 1e-9
'''

def normalize_efd(coeffs, size_invariant=True, return_transformation=False):
    """Normalizes an array of Fourier coefficients.

    See [#a]_ and [#b]_ for details.

    :param numpy.ndarray coeffs: A ``[n x 4]`` Fourier coefficient array.
    :param bool size_invariant: If size invariance normalizing should be done as well.
        Default is ``True``.
    :param bool return_transformation: If the normalization parametres should be returned.
        Default is ``False``.
    :return: The normalized ``[n x 4]`` Fourier coefficient array and optionally the
        transformation parametres ``scale``, :math:`psi_1` (rotation) and :math:`theta_1` (phase)
    :rtype: :py:class:`numpy.ndarray` or (:py:class:`numpy.ndarray`, (float, float, float))

    """
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
        return coeffs, (size, psi_1, theta_1)
    else:
        return coeffs

def elliptic_fourier_descriptors(
    contour, order=10, normalize=True, return_transformation=True
):
    """Calculate elliptical Fourier descriptors for a contour.

    :param numpy.ndarray contour: A contour array of size ``[M x 2]``.
    :param int order: The order of Fourier coefficients to calculate.
    :param bool normalize: If the coefficients should be normalized;
        see references for details.
    :param bool return_transformation: If the normalization parametres should be returned.
        Default is ``False``.
    :return: A ``[order x 4]`` array of Fourier coefficients and optionally the
        transformation parametres ``scale``, ``psi_1`` (rotation) and ``theta_1`` (phase)
    :rtype: ::py:class:`numpy.ndarray` or (:py:class:`numpy.ndarray`, (float, float, float))

    """
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

    if normalize:
        coeffs = normalize_efd(coeffs, return_transformation=return_transformation)

    return coeffs

def calculate_dc_coefficients(contour):
    """Calculate the :math:`A_0` and :math:`C_0` coefficients of the elliptic Fourier series.

    :param numpy.ndarray contour: A contour array of size ``[M x 2]``.
    :return: The :math:`A_0` and :math:`C_0` coefficients.
    :rtype: tuple

    """
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

def plot_efd(coeffs, locus, image, contour, samp, scaling, angle):
    """Plot a ``[2 x (N / 2)]`` grid of successive truncations of the series.

    .. note::

        Requires `matplotlib <http://matplotlib.org/>`_!

    :param numpy.ndarray coeffs: ``[N x 4]`` Fourier coefficient array.
    :param list, tuple or numpy.ndarray locus:
        The :math:`A_0` and :math:`C_0` elliptic locus in [#a]_ and [#b]_.
    :param int n: Number of points to use for plotting of Fourier series.

    """
    # image = None
    # contour = None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x_min, x_max = np.min(contour[:, 0]), np.max(contour[:, 0])
    y_min, y_max = np.min(contour[:, 1]), np.max(contour[:, 1])

    x_diff = np.divide((x_max - x_min), 2)
    y_diff = np.divide((y_max - y_min), 2)

    scaling_factor = np.divide(max(x_diff, y_diff),  max(x_diff, y_diff))


    # y_diff = 300
    # x_diff = 300

    N = coeffs.shape[0]
    N_half = int(np.ceil(np.divide(N, 1)))
    N_half = 1
    n_rows = 1

    t = np.linspace(0, 1.0, samp)
    xt = np.ones((samp,)) * locus[1]
    yt = np.ones((samp,)) * locus[0]

    for n in range(coeffs.shape[0]):
        # n=39
        xt += (coeffs[n, 0] * np.cos(2 * (n + 1) * np.pi * t)) * scaling + (
            coeffs[n, 1] * np.sin(2 * (n + 1) * np.pi * t) * scaling
        )
        yt += (coeffs[n, 2] * np.cos(2 * (n + 1) * np.pi * t)) * scaling + (
            coeffs[n, 3] * np.sin(2 * (n + 1) * np.pi * t) * scaling
        )

    ## yt = -1 * yt + locus[0] + locus[0]
    
    # Rotate the coords
    xt = -1 * xt + locus[1] + locus[1]

    radi = math.radians(angle)
    yt = (xt - locus[1])*np.cos(-angle) - (yt - locus[0])*np.sin(-angle) + locus[0]
    xt = (xt - locus[1])*np.sin(-angle) + (yt - locus[0])*np.cos(-angle) + locus[1]

    # xt = -1 * xt + locus[1] + locus[1]
    yt = -1 * yt + locus[0] + locus[0]

    # ax = plt.subplot2grid((n_rows, N_half), (n // N_half, n % N_half))
    ax = plt.subplot(1, 1, 1)
    ax.set_title(str(n + 1))

    if image is not None:
        # A background image of shape [rows, cols] gets transposed
        # by imshow so that the first dimension is vertical
        # and the second dimension is horizontal.
        # This implies swapping the x and y axes when plotting a curve.
        if contour is not None:
            ax.plot(contour[:, 0], contour[:, 1], "g", linewidth=1)
        
        ax.plot(yt, xt, "r", linewidth=1)
        ax.imshow(image, plt.cm.gray)
    else:
        # Without a background image, no transpose is implied.
        # This case is useful when (x,y) point clouds
        # without relation to an image are to be handled.
        if contour is not None:
            ax.plot(contour[:, 1], contour[:, 0], "g", linewidth=1)
        ax.plot(yt, xt, "r", linewidth=1)
        ax.axis("equal")
        
    plt.savefig("efds_example.png", dpi=300)
    # plt.show()

# def get_largest_polygon2(polygons, height, width):
#     # Keep the polygon that has the most points
#     polygons = [polygon for polygon in polygons if len(polygon) == max([len(p) for p in polygons])]
#     # convert the list of polygons to a list of contours
#     contours = [np.array(polygon, dtype=np.int32).reshape((-1,1,2)) for polygon in polygons]
#     # filter the contours to only closed contours
#     closed_contours = [c for c in contours if cv2.isContourConvex(c)]
#     if len(closed_contours)>0:
#         # sort the closed contours by area
#         closed_contours = sorted(closed_contours, key=cv2.contourArea, reverse=True)
#         # take the largest closed contour
#         largest_closed_contour = closed_contours[0]
#     else:
#         contours = sorted(contours, key=cv2.contourArea, reverse=True)
#         largest_closed_contour = contours[0]
#     # Create a black image with the same dimensions as the input image
#     img = np.zeros((height, width), dtype=np.uint8)
#     # Draw all the contours in the black image
#     cv2.drawContours(img, contours, -1, 255, -1)
#     # Find all the contours in the black image
#     all_contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     # Filter out the contours that are inside the largest closed contour
#     inner_contours = [c for c in all_contours if cv2.contourArea(c) < cv2.contourArea(largest_closed_contour)]
#     # Sort the inner contours by area
#     inner_contours = sorted(inner_contours, key=cv2.contourArea, reverse=True)
#     # Return the largest hole contour
#     if len(inner_contours)>0:
#         largest_hole_contour = inner_contours[0]
#         # Convert the largest hole contour back to the polygon format
#         largest_polygon = [tuple(i[0]) for i in largest_hole_contour]
#     else:
#         largest_polygon = [tuple(i[0]) for i in largest_closed_contour]
#     return largest_polygon

if __name__ == '__main__':
    with open('D:/Dropbox/LM2_Env/Image_Datasets/TEST_LM2/TEST_2023_01_24__16-03-18/Plant_Components/json/Plant_Components.json') as json_file:
        dict_plant_components = json.load(json_file)
    segment_leaves([], 'D:\Dropbox\LeafMachine2', 'D:\Dropbox\LM2_Env\Image_Datasets\SET_Acacia\Images_GBIF_Acacia_Prickles', [], dict_plant_components)