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
from io import BytesIO

currentdir = os.path.dirname(inspect.getfile(inspect.currentframe()))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(currentdir)
# from detect import run
sys.path.append(parentdir)
# from machine.general_utils import Print_Verbose
from detector import Detector

def segment_leaves(cfg, dir_home, Project, Dirs, dict_plant_components): 
    batch_size = cfg['leafmachine']['project']['batch_size']

    # See convert_index_to_class(ind) for list of ind -> cls
    dict_plant_components = unpack_class_from_components(dict_plant_components, 0, 'Whole_Leaf_BBoxes_YOLO', 'Whole_Leaf_BBoxes', Project)
    dict_plant_components = unpack_class_from_components(dict_plant_components, 1, 'Partial_Leaf_BBoxes_YOLO', 'Partial_Leaf_BBoxes', Project)

    # Crop the images to bboxes
    dict_plant_components = crop_images_to_bbox(dict_plant_components, 0, 'Whole_Leaf_Cropped', "Whole_Leaf_BBoxes", Project)
    dict_plant_components = crop_images_to_bbox(dict_plant_components, 1, 'Partial_Leaf_Cropped', "Partial_Leaf_BBoxes", Project)

    # Run the leaf instance segmentation operations
    dir_seg_model = os.path.join(dir_home, 'leafmachine2', 'segmentation', 'models',cfg['leafmachine']['leaf_segmentation']['segmentation_model'])
    Instance_Detector = Detector(dir_seg_model, cfg['leafmachine']['leaf_segmentation']['minimum_confidence_threshold'])

    segment_whole_leaves_props = {}
    segment_whole_leaves_overlay = {}
    segment_partial_leaves_props = {}
    segment_partial_leaves_overlay = {}
    for i in range(0, len(dict_plant_components), batch_size):
        start = i
        end = i+batch_size
        dict_plant_components_batch = dict(itertools.islice(dict_plant_components.items(), i, i+batch_size))
        end = len(dict_plant_components) if len(dict_plant_components_batch) != (end - start) else end
        if cfg['leafmachine']['leaf_segmentation']['segment_whole_leaves']:
            segment_whole_leaves_props_batch, segment_whole_leaves_overlay_batch = segment_images(Instance_Detector, dict_plant_components_batch, 0, "Segmentation_Whole_Leaf", "Whole_Leaf_Cropped", cfg, Project, Dirs, start+1, end)
        segment_whole_leaves_props.update(segment_whole_leaves_props_batch)
        segment_whole_leaves_overlay.update(segment_whole_leaves_overlay_batch)
        if cfg['leafmachine']['leaf_segmentation']['segment_partial_leaves']:
            segment_partial_leaves_props_batch, segment_partial_leaves_overlay_batch = segment_images(Instance_Detector, dict_plant_components_batch, 1, "Segmentation_Partial_Leaf", "Partial_Leaf_Cropped", cfg, Project, Dirs, start+1, end)
        segment_partial_leaves_props.update(segment_partial_leaves_props_batch)
        segment_partial_leaves_overlay.update(segment_partial_leaves_overlay_batch)
    
    size_of_dict = sys.getsizeof(segment_whole_leaves_props)
    print(size_of_dict)
    size_of_dict = sys.getsizeof(segment_whole_leaves_overlay)
    print(size_of_dict)
    size_of_dict = sys.getsizeof(segment_partial_leaves_props)
    print(size_of_dict)
    size_of_dict = sys.getsizeof(segment_partial_leaves_overlay)
    print(size_of_dict)
    print("hi")






def segment_images(Instance_Detector, dict_objects, leaf_type, dict_name_seg, dict_from, cfg, Project, Dirs, start, end):
    generate_overlay = cfg['leafmachine']['leaf_segmentation']['generate_overlay']
    overlay_dpi = cfg['leafmachine']['leaf_segmentation']['overlay_dpi']
    bg_color = cfg['leafmachine']['leaf_segmentation']['overlay_background_color']
    keep_best = cfg['leafmachine']['leaf_segmentation']['keep_only_best_one_leaf_one_petiole']
    save_overlay_pdf = cfg['leafmachine']['leaf_segmentation']['save_segmentation_overlay_images_to_pdf']
    save_individual_overlay_images = cfg['leafmachine']['leaf_segmentation']['save_individual_overlay_images']
    save_ind_masks_color = cfg['leafmachine']['leaf_segmentation']['save_masks_color']
    # save_ind_masks_index = cfg['leafmachine']['leaf_segmentation']['save_masks_index']
    save_full_image_masks_color = cfg['leafmachine']['leaf_segmentation']['save_full_image_masks_color']
    # save_full_image_masks_index = cfg['leafmachine']['leaf_segmentation']['save_full_image_masks_index']
    use_efds_for_masks = cfg['leafmachine']['leaf_segmentation']['use_efds_for_masks']
    save_rgb_cropped_images = cfg['leafmachine']['leaf_segmentation']['save_rgb_cropped_images']

    filenames = []
    full_images = []
    full_masks = []

    seg_overlay_name = '_'.join([dict_name_seg,'Overlay'])
    seg_overlay = {}
    seg_overlay_data = {}

    for filename, value in dict_objects.items(): # Whole image
        value[dict_name_seg] = []
        seg_overlay[filename] = []
        seg_overlay_data[filename] = []

        try:
            # full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpg'])))
            full_image = cv2.cvtColor(cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpg']))), cv2.COLOR_BGR2RGB)
        except:
            # full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpeg'])))
            full_image = cv2.cvtColor(cv2.imread(os.path.join(Project.dir_images, '.'.join([filename,'jpeg']))), cv2.COLOR_BGR2RGB)

        full_size = full_image.shape
        if save_full_image_masks_color:
            # Create a black image
            full_mask = Image.new('RGB', (full_size[1], full_size[0]), color=(0, 0, 0))
        else: 
            full_mask = []

        if value[dict_from] is not []:
            for cropped in value[dict_from]: # Individual leaf
                for seg_name, img_cropped in cropped.items():
                    print(seg_name)
                    seg_name_short = seg_name.split("__")[2]
                    # cropped_overlay = []


                    # Segment!
                    fig, out_polygons, out_bboxes, out_labels, out_color = Instance_Detector.segment(img_cropped, generate_overlay, overlay_dpi, bg_color)
                    
                    if len(out_polygons) > 0: # Success
                        if keep_best:
                            out_polygons, out_bboxes, out_labels, out_color = keep_rows(out_polygons, out_bboxes, out_labels, out_color, get_string_indices(out_labels))
                        
                        
                        detected_components, cropped_overlay, overlay_data = create_overlay_and_calculate_props(seg_name, img_cropped, out_polygons, out_labels, out_color, cfg)
                        # full_image = create_insert_legacy(full_image, cropped_overlay, seg_name_short)
                        full_image = create_insert(full_image, overlay_data, seg_name_short, cfg)

                        cropped_overlay_size = cropped_overlay.shape

                    else: # Fail
                        detected_components = []
                        cropped_overlay = []
                        overlay_data = []
                        cropped_overlay_size = []

                    value[dict_name_seg].append({seg_name: detected_components})
                    seg_overlay[filename].append({seg_name: cropped_overlay})
                    seg_overlay_data[filename].append({seg_name: overlay_data})

                    save_rgb_cropped(save_rgb_cropped_images, seg_name, img_cropped, leaf_type, Dirs)

                    save_individual_segmentations(save_individual_overlay_images, dict_name_seg, seg_name, cropped_overlay, Dirs)

                    full_mask = save_masks_color(save_ind_masks_color, save_full_image_masks_color, use_efds_for_masks, full_mask, overlay_data, cropped_overlay_size, full_size, seg_name, seg_name_short, leaf_type, Dirs)
                    # save_masks_index()
                    
        save_full_masks(save_full_image_masks_color, full_mask, filename, leaf_type, Dirs)
        
        filenames.append(filename)
        
        if save_overlay_pdf:
            full_images.append(full_image)

        if save_full_image_masks_color:
            full_masks  .append(full_mask)
        
    save_full_image_segmentations(save_overlay_pdf, dict_name_seg, full_images, filenames, Dirs, cfg, start, end)
    return dict_objects, seg_overlay

def save_rgb_cropped(save_rgb_cropped_images, seg_name, img_cropped, leaf_type, Dirs):
    if save_rgb_cropped_images:
        if leaf_type == 0:
            cv2.imwrite(os.path.join(Dirs.whole_leaves, '.'.join([seg_name, 'jpg'])), img_cropped)
        elif leaf_type == 1:
            cv2.imwrite(os.path.join(Dirs.partial_leaves, '.'.join([seg_name, 'jpg'])), img_cropped)


def save_masks_color(save_individual_masks_color, save_full_image_masks_color, use_efds_for_masks, full_mask, overlay_data, cropped_overlay_size, full_size, seg_name, seg_name_short, leaf_type, Dirs):
    if len(overlay_data) > 0:
        # unpack
        overlay_poly, overlay_efd, overlay_rect, overlay_color = overlay_data

        if use_efds_for_masks:
            use_polys = overlay_efd
        else:
            use_polys = overlay_poly
        
        if save_individual_masks_color:
            # Create a black image
            img = Image.new('RGB', (cropped_overlay_size[1], cropped_overlay_size[0]), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)

            if use_polys != []:
                for i, poly in enumerate(use_polys):
                    this_color = overlay_color[i]
                    cls, this_color = next(iter(this_color.items()))
                    # Set the color for the polygon based on its class
                    if 'leaf' in cls:
                        color = [46, 255, 0]
                    elif 'petiole' in cls:
                        color = [0, 173, 255]
                    elif 'hole' in cls:
                        color = [209, 0, 255]
                    else:
                        color = [255, 255, 255]
                    # Draw the filled polygon on the image
                    draw.polygon(poly, fill=tuple(color))
            if leaf_type == 0:
                img.save(os.path.join(Dirs.segmentation_masks_color_whole_leaves, '.'.join([seg_name, 'png'])))
            elif leaf_type == 1:
                img.save(os.path.join(Dirs.segmentation_masks_color_partial_leaves, '.'.join([seg_name, 'png'])))

        if save_full_image_masks_color:
            origin_x = int(seg_name_short.split('-')[0])
            origin_y = int(seg_name_short.split('-')[1])

            # Create a black image
            draw = ImageDraw.Draw(full_mask)

            if use_polys != []:
                for i, poly in enumerate(use_polys):
                    this_color = overlay_color[i]
                    poly = [(x+origin_x, y+origin_y) for x, y in poly]
                    cls, this_color = next(iter(this_color.items()))
                    # Set the color for the polygon based on its class
                    if 'leaf' in cls:
                        color = [46, 255, 0]
                    elif 'petiole' in cls:
                        color = [0, 173, 255]
                    elif 'hole' in cls:
                        color = [209, 0, 255]
                    else:
                        color = [255,255,255]
                    # Draw the filled polygon on the image
                    draw.polygon(poly, fill=tuple(color))
    return full_mask

def save_full_masks(save_full_image_masks_color, full_mask, filename, leaf_type, Dirs):
    if save_full_image_masks_color:
        if leaf_type == 0:
            full_mask.save(os.path.join(Dirs.segmentation_masks_full_image_color_whole_leaves, '.'.join([filename, 'png'])))
        elif leaf_type == 1:
            full_mask.save(os.path.join(Dirs.segmentation_masks_full_image_color_partial_leaves, '.'.join([filename, 'png'])))
    

def create_insert(full_image, overlay_data, seg_name_short, cfg):
    width = cfg['leafmachine']['leaf_segmentation']['overlay_line_width']
    origin_x = int(seg_name_short.split('-')[0])
    origin_y = int(seg_name_short.split('-')[1])
    # unpack
    overlay_poly, overlay_efd, overlay_rect, overlay_color = overlay_data
    # fill_color = overlay_color[0][0]
    # outline_color =  overlay_color[0][1]

    # initialize
    full_image = np.asarray(full_image)
    full_image = Image.fromarray(full_image)
    draw = ImageDraw.Draw(full_image, "RGBA")

    if overlay_poly != []:
        for i, poly in enumerate(overlay_poly):
            this_color = overlay_color[i]
            key, this_color = next(iter(this_color.items()))
            # outline_color =(this_color[1][2],this_color[1][1],this_color[1][0])
            # fill_color = (this_color[0][2],this_color[0][1],this_color[0][0],this_color[0][3])
            outline_color = (this_color[1][2],this_color[1][1],this_color[1][0])
            fill_color = (this_color[0][2],this_color[0][1],this_color[0][0],this_color[0][3])

            poly = [(x+origin_x, y+origin_y) for x, y in poly]
            # poly = np.asarray(poly)
            # first_point = poly[0]
            # poly_closed = np.vstack((poly, first_point))
            draw.polygon(poly, fill=fill_color, outline=outline_color, width=width)

    if overlay_rect != []:
        for i, rect in enumerate(overlay_rect):
            this_color = overlay_color[i]
            key, this_color = next(iter(this_color.items()))
            outline_color = (this_color[1][2],this_color[1][1],this_color[1][0])

            rect = [(x+origin_x, y+origin_y) for x, y in rect]
            draw.rectangle(rect, fill=None, outline=outline_color, width=width)
    
    if overlay_efd != []:
        for efd in overlay_efd:
            efd = [(x+origin_x, y+origin_y) for x, y in efd]
            draw.polygon(efd, fill=None, outline=(135,30,210), width=width)

    return full_image

def create_insert_legacy(full_image, cropped_overlay, seg_name_short):
    # Get the coordinates from the seg_name_short string
    x1, y1, x2, y2 = seg_name_short.split("-")
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    full_image[y1:y2, x1:x2] = cv2.cvtColor(cropped_overlay, cv2.COLOR_BGR2RGB)
    # cv2.imshow('full_image', full_image)
    # cv2.waitKey(0)
    return full_image

def save_individual_segmentations(save_individual_overlay_images, dict_name_seg, seg_name, cropped_overlay, Dirs):
    if save_individual_overlay_images:
        if dict_name_seg == "Segmentation_Whole_Leaf":
            cv2.imwrite(os.path.join(Dirs.segmentation_whole_leaves, '.'.join([seg_name, 'jpg'])), cropped_overlay)
        elif dict_name_seg == "Segmentation_Partial_Leaf":
            cv2.imwrite(os.path.join(Dirs.segmentation_partial_leaves, '.'.join([seg_name, 'jpg'])), cropped_overlay)

def save_full_image_segmentations(save_overlay_pdf, dict_name_seg, full_images, filenames, Dirs, cfg, start, end):
    color_bg = cfg['leafmachine']['leaf_segmentation']['overlay_background_color']
    overlay_dpi = cfg['leafmachine']['leaf_segmentation']['overlay_dpi']
    if color_bg == 'black':
        color_text = 'white'
    else:
        color_text = 'black'

    if save_overlay_pdf:
        pdf_name = os.path.join(Dirs.segmentation_overlay_pdfs, ''.join([dict_name_seg, '_',str(start), 'to',str(end),'.pdf']))
        with PdfPages(pdf_name) as pdf:
            for idx, img in enumerate(full_images):
                # Create a new figure
                fig = plt.figure()
                fig.set_facecolor(color_bg)
                plt.tight_layout(pad=0)
                # plt.subplots_adjust(left=1, right=1, bottom=1, top=1)
                # Add the image to the figure
                plt.imshow(img)
                # plt.annotate(xy=(0, 0), xycoords='axes fraction', fontsize=6,
                #             xytext=(1, 1), textcoords='offset points',
                #             ha='left', va='bottom')
                plt.suptitle(filenames[idx], fontsize=10, y=0.95, color=color_text)
                # Save the current figure to the PDF
                pdf.savefig(fig, dpi=overlay_dpi)
                plt.close()

# def create_image_report(full_image):
#     doc = SimpleDocTemplate("combined_images.pdf", pagesize=landscape(letter))

#     # Open the full image
#     full_image = Image.open("full_image.jpg")

#     # Create an empty list for the smaller images
#     small_images = []

#     # Open and add the smaller images to the list
#     for i in range(0, 50):
#         small_image = Image.open("small_image_{}.jpg".format(i))
#         small_images.append(small_image)

#     # Create a table to hold the smaller images
#     table_data = []
#     for i in range(0, len(small_images), 5):
#         row = small_images[i:i+5]
#         table_data.append(row)

#     table = Table(table_data)

#     # Add the full image to the left side of the table
#     table.insert(0, PDF_Image(full_image, width=full_image.width, height=full_image.height))

#     # Apply any desired styling to the table
#     table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"),
#     ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
#     ("GRID", (0, 0), (-1, -1), 1, colors.black)
#     ]))
#     doc.build([table])
#     doc.save()
#     doc.close()

def keep_rows(list1, list2, list3, list4, string_indices):
    leaf_index, petiole_index, hole_indices = string_indices
    # All
    if (leaf_index is not None) and (petiole_index is not None) and (hole_indices is not None):
        indices_to_keep = [leaf_index, petiole_index] + hole_indices
    # No holes
    elif (leaf_index is not None) and (petiole_index is not None) and (hole_indices is None):
        indices_to_keep = [leaf_index, petiole_index]
    # Only leaves
    elif (leaf_index is not None) and (petiole_index is None) and (hole_indices is None):
        indices_to_keep = [leaf_index]
    # Only petiole
    elif (leaf_index is None) and (petiole_index is not None) and (hole_indices is None):
        indices_to_keep = [petiole_index]
    # Only hole
    elif (leaf_index is None) and (petiole_index is None) and (hole_indices is not None):
        indices_to_keep = [hole_indices]
    # Only petiole and hole
    elif (leaf_index is None) and (petiole_index is not None) and (hole_indices is not None):
        indices_to_keep =  [petiole_index] + hole_indices
    # Only leaves and hole
    elif (leaf_index is not None) and (petiole_index is None) and (hole_indices is not None):
        indices_to_keep =  [leaf_index] + hole_indices
    list1 = [list1[i] for i in indices_to_keep]
    list2 = [list2[i] for i in indices_to_keep]
    list3 = [list3[i] for i in indices_to_keep]
    list4 = [list4[i] for i in indices_to_keep]
    return list1, list2, list3, list4

def get_string_indices(strings):
    leaf_strings = [s for s in strings if s.startswith('leaf')]
    petiole_strings = [s for s in strings if s.startswith('petiole')]
    hole_strings = [s for s in strings if s.startswith('hole')]

    if len(leaf_strings) > 0:
        leaf_value = max([int(s.split(' ')[1].replace('%','')) for s in leaf_strings])
        leaf_index = strings.index([s for s in leaf_strings if int(s.split(' ')[1].replace('%','')) == leaf_value][0])
    else:
        leaf_index = None

    if len(petiole_strings) > 0:
        petiole_value = max([int(s.split(' ')[1].replace('%','')) for s in petiole_strings])
        petiole_index = strings.index([s for s in petiole_strings if int(s.split(' ')[1].replace('%','')) == petiole_value][0])
    else:
        petiole_index = None

    if len(hole_strings) > 0:
        hole_indices = [i for i, s in enumerate(strings) if s.startswith('hole')]
    else:
        hole_indices = None

    return leaf_index, petiole_index, hole_indices

def create_overlay_and_calculate_props(seg_name, img_cropped, out_polygons, out_labels, out_color, cfg):
    width = cfg['leafmachine']['leaf_segmentation']['overlay_line_width']
    do_get_efds = cfg['leafmachine']['leaf_segmentation']['calculate_elliptic_fourier_descriptors']
    cropped_overlay = img_cropped
    # cropped_overlay = Image.fromarray(cv2.cvtColor(cropped_overlay, cv2.COLOR_BGR2RGB))
    cropped_overlay = Image.fromarray(cropped_overlay)
    draw = ImageDraw.Draw(cropped_overlay, "RGBA")

    # parse seg_name
    seg_name_short = seg_name.split("__")[2]

    # List of instances
    detected_components = []
    overlay_poly = []
    overlay_efd = []
    overlay_rect = []
    overlay_color = []
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
        if max_poly is None:
            component = None
        else:
            component, bbox = polygon_properties(max_poly, out_labels[i], seg_name_short, cfg, img_cropped)
            detected_components.append(component)

            # draw poly
            draw.polygon(max_poly, fill=fill_color, outline=outline_color, width=width)
            draw.rectangle(bbox, outline=outline_color, width=width)
            overlay_rect.append(bbox)
            overlay_poly.append(max_poly)
            overlay_color.append({out_labels[i]: [fill_color, outline_color]})

            if '__L__' in seg_name:
                if do_get_efds:
                    _, value = next(iter(component.items()))
                    efd = value['efds']['pts_efd_PIL']
                    # efd = efd['pts_efd']
                    draw.polygon(efd, fill=None, outline=(210,30,135), width=width)
                    overlay_efd.append(efd)
    # PIL               
    # cropped_overlay.show() # wrong colors without changing to RGB
    # cv2
    cropped_overlay = np.array(cropped_overlay)
    overlay_data = [overlay_poly, overlay_efd, overlay_rect, overlay_color]
    # cv2.imshow('img_crop', cropped_overlay)
    # cv2.waitKey(0)
    return detected_components, cropped_overlay, overlay_data

def polygon_properties(polygon, out_label, seg_name_short, cfg, img_cropped):
    do_get_efds = cfg['leafmachine']['leaf_segmentation']['calculate_elliptic_fourier_descriptors']
    efd_order = cfg['leafmachine']['leaf_segmentation']['elliptic_fourier_descriptor_order']
    if efd_order is None:
        efd_order = 20
    else:
        efd_order = int(efd_order)

    polygon = np.asarray(polygon)
    first_point = polygon[0]
    polygon_closed = np.vstack([polygon, first_point])

    labels = ['leaf', 'petiole', 'hole']
    name_class = next((label for label in labels if label in out_label), None)  
    poly_name = '_'.join([name_class, seg_name_short])

    # Create a Shapely Polygon object from the input list of coordinates
    shapely_points = MultiPoint(polygon_closed)
    shapely_polygon = Polygon(polygon_closed)

    polygon_contour = np.array(polygon_closed, dtype=np.int32).reshape((-1,1,2))
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
    degree = len(polygon_closed)
    # Calculate the aspect ratio of the polygon
    aspect_ratio = shapely_polygon.bounds[2] / shapely_polygon.bounds[3]
    # bounding box
    bbox = get_bounding_box(polygon_closed)
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
        dict_efd = calc_efds(polygon_closed, img_cropped)
    else:
        dict_efd = []
    # return a dictionary of the properties
    props = {
        'bbox': bbox,
        'bbox_min': bbox_min_coords, 
        'length': length,
        'width': width,
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
        'polygon': polygon
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

def calc_efds(polygon, img_cropped):
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

    dict_efd = {}
    dict_efd["coeffs_normalized"] = coeffs_normalized
    dict_efd["coeffs_features"] = coeffs_features
    dict_efd["a0"] = a0
    dict_efd["c0"] = c0
    dict_efd["scale"] = scale
    dict_efd["angle"] = angle
    dict_efd["phase"] = phase
    dict_efd["fig_efd_cv2"] = fig_efd_cv2
    dict_efd["pts_contour"] = pts_contour
    dict_efd["pts_efd"] = pts_efd
    dict_efd["pts_efd_PIL"] = pts_efd_PIL
    return dict_efd

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
                if cls == 0:
                    crop_name = '__'.join([filename,'L',loc])
                elif cls == 1:
                    crop_name = '__'.join([filename,'PL',loc])
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
    try:
        # polygons = max(polygons, key=len)
        # Keep the polygon that has the most points
        polygons = [polygon for polygon in polygons if len(polygon) == max([len(p) for p in polygons])]
        # convert the list of polygons to a list of contours
        contours = [np.array(polygon, dtype=np.int32).reshape((-1,1,2)) for polygon in polygons]
        # filter the contours to only closed contours
        closed_contours = [c for c in contours if cv2.isContourConvex(c)]
        if len(closed_contours) > 0:
            # sort the closed contours by area
            closed_contours = sorted(closed_contours, key=cv2.contourArea, reverse=True)
            # take the largest closed contour
            largest_closed_contour = closed_contours[0]
        else:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_closed_contour = contours[0]
        largest_polygon = [tuple(i[0]) for i in largest_closed_contour]
    except:
        largest_polygon = None
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

''''''
if __name__ == '__main__':
    with open('D:/Dropbox/LM2_Env/Image_Datasets/TEST_LM2/TEST_2023_01_24__16-03-18/Plant_Components/json/Plant_Components.json') as json_file:
        dict_plant_components = json.load(json_file)
    segment_leaves([], 'D:\Dropbox\LeafMachine2', 'D:\Dropbox\LM2_Env\Image_Datasets\SET_Acacia\Images_GBIF_Acacia_Prickles', [], dict_plant_components)
''''''