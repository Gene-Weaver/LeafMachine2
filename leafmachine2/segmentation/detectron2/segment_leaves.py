import os, json, random, glob, inspect, sys, cv2, itertools, torch
from timeit import default_timer as timer
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
from time import perf_counter
from shapely.geometry import Polygon, Point, MultiPoint
from io import BytesIO
import concurrent.futures
import threading

currentdir = os.path.dirname(inspect.getfile(inspect.currentframe()))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(currentdir)
# from detect import run
sys.path.append(parentdir)
# from machine.general_utils import Print_Verbose
from measure_leaf_segmentation import polygon_properties
from detector import Detector_LM2

def segment_leaves(cfg, logger, dir_home, Project, batch, n_batches, Dirs): 
    start_t = perf_counter()
    logger.name = f'[BATCH {batch+1} Segment Leaves]'
    logger.info(f'Segmenting leaves for batch {batch+1} of {n_batches}')

    # batch_size = cfg['leafmachine']['project']['batch_size']
    if cfg['leafmachine']['project']['num_workers'] is None:
        num_workers = 1
    else:
        num_workers = int(cfg['leafmachine']['project']['num_workers'])

    # See convert_index_to_class(ind) for list of ind -> cls
    Project.project_data_list[batch] = unpack_class_from_components(Project.project_data_list[batch], 0, 'Whole_Leaf_BBoxes_YOLO', 'Whole_Leaf_BBoxes', Project)
    Project.project_data_list[batch] = unpack_class_from_components(Project.project_data_list[batch], 1, 'Partial_Leaf_BBoxes_YOLO', 'Partial_Leaf_BBoxes', Project)

    # Crop the images to bboxes
    Project.project_data_list[batch] = crop_images_to_bbox(Project.project_data_list[batch], 0, 'Whole_Leaf_Cropped', "Whole_Leaf_BBoxes", Project)
    Project.project_data_list[batch] = crop_images_to_bbox(Project.project_data_list[batch], 1, 'Partial_Leaf_Cropped', "Partial_Leaf_BBoxes", Project)

    # Run the leaf instance segmentation operations
    dir_seg_model = os.path.join(dir_home, 'leafmachine2', 'segmentation', 'models',cfg['leafmachine']['leaf_segmentation']['segmentation_model'])
    Instance_Detector_Whole = Detector_LM2(logger, dir_seg_model, cfg['leafmachine']['leaf_segmentation']['minimum_confidence_threshold'], 0)
    Instance_Detector_Partial = Detector_LM2(logger, dir_seg_model, cfg['leafmachine']['leaf_segmentation']['minimum_confidence_threshold'], 1)

    # Old batch method ...
    # segment_whole_leaves_props = {}
    # segment_whole_leaves_overlay = {}
    # segment_partial_leaves_props = {}
    # segment_partial_leaves_overlay = {}
    # if batch_size is None:
    #     batch_size = len(Project.project_data_list[batch])
    # for i in range(0, len(Project.project_data_list[batch]), batch_size):
    #     start = i
    #     end = i+batch_size
    #     dict_plant_components_batch = dict(itertools.islice(Project.project_data.items(), i, i+batch_size))
    #     end = len(Project.project_data) if len(dict_plant_components_batch) != (end - start) else end
    segment_whole_leaves_props = {}
    segment_whole_leaves_overlay = {}
    segment_partial_leaves_props = {}
    segment_partial_leaves_overlay = {}
    if cfg['leafmachine']['leaf_segmentation']['segment_whole_leaves']:
        logger.info(f'Segmenting whole leaves')
        segment_whole_leaves_props_batch, segment_whole_leaves_overlay_batch = segment_images_parallel(logger, Instance_Detector_Whole, Project.project_data_list[batch], 0, "Segmentation_Whole_Leaf", "Whole_Leaf_Cropped", cfg, Project, Dirs, batch, n_batches, num_workers)#, start+1, end)
        segment_whole_leaves_props.update(segment_whole_leaves_props_batch)
        segment_whole_leaves_overlay.update(segment_whole_leaves_overlay_batch)

    if cfg['leafmachine']['leaf_segmentation']['segment_partial_leaves']:
        logger.info(f'Segmenting partial leaves')
        segment_partial_leaves_props_batch, segment_partial_leaves_overlay_batch = segment_images_parallel(logger, Instance_Detector_Partial, Project.project_data_list[batch], 1, "Segmentation_Partial_Leaf", "Partial_Leaf_Cropped", cfg, Project, Dirs, batch, n_batches, num_workers)#, start+1, end)
        segment_partial_leaves_props.update(segment_partial_leaves_props_batch)
        segment_partial_leaves_overlay.update(segment_partial_leaves_overlay_batch)
    
    # dict_part_names = ['Segmentation_Whole_Leaf_Props', 'Segmentation_Whole_Leaf_Overlay', 'Segmentation_Partial_Leaf_Props', 'Segmentation_Partial_Leaf_Overlay']
    # for img in Project.project_data:
    #     for i, dict_part in enumerate([segment_whole_leaves_props, segment_whole_leaves_overlay, segment_partial_leaves_props, segment_partial_leaves_overlay]):
    #         Project.project_data[img].update({dict_part_names[i]: dict_part[img]})

    # size_of_dict = sys.getsizeof(segment_whole_leaves_props)
    # print(size_of_dict)
    # size_of_dict = sys.getsizeof(segment_whole_leaves_overlay)
    # print(size_of_dict)
    # size_of_dict = sys.getsizeof(segment_partial_leaves_props)
    # print(size_of_dict)
    # size_of_dict = sys.getsizeof(segment_partial_leaves_overlay)
    # print(size_of_dict)
    # logger.debug()
    
    end_t = perf_counter()
    # print(f'Batch {batch+1}/{n_batches}: Leaf Segmentation Duration --> {round((end_t - start_t)/60)} minutes')
    logger.info(f'Batch {batch+1}/{n_batches}: Leaf Segmentation Duration --> {round((end_t - start_t)/60)} minutes')
    return Project


''' SEGMENT PARALLEL'''
def segment_images_parallel(logger, Instance_Detector, dict_objects, leaf_type, dict_name_seg, dict_from, cfg, Project, Dirs, batch, n_batches, num_workers):
    
    seg_overlay = {}
    seg_overlay_data = {}

    # Define a lock object
    lock = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            dict_objects_chunk = dict(list(dict_objects.items())[i::num_workers])
            futures.append(executor.submit(segment_images, logger, Instance_Detector, dict_objects_chunk, leaf_type, dict_name_seg, dict_from, cfg, Project, Dirs, batch, n_batches, lock))

        for future in concurrent.futures.as_completed(futures):
            dict_objects, seg_overlay_chunk = future.result()
            for filename, value in dict_objects.items():
                seg_overlay[filename] = seg_overlay_chunk[filename]
                seg_overlay_data[filename] = seg_overlay_chunk[filename]

                # filenames.append(filename)

                # if save_overlay_pdf:
                #     full_images.append(Image.open(os.path.join(Dirs.path_segmentation_images, seg_overlay_name, filename)))

                # if save_full_image_masks_color:
                #     full_masks.append(Image.open(os.path.join(Dirs.path_segmentation_images, seg_overlay_name, filename.split(".")[0] + "_mask.png")))

    # save_full_image_segmentations(save_overlay_pdf, dict_name_seg, full_images, filenames, Dirs, cfg, batch, n_batches)#, start, end)
    torch.cuda.empty_cache()
    return dict_objects, seg_overlay#, seg_overlay_data
''' SEGMENT PARALLEL'''



def segment_images(logger, Instance_Detector, dict_objects, leaf_type, dict_name_seg, dict_from, cfg, Project, Dirs, batch, n_batches, lock):#, start, end):
    generate_overlay = cfg['leafmachine']['leaf_segmentation']['generate_overlay']
    overlay_dpi = cfg['leafmachine']['leaf_segmentation']['overlay_dpi']
    bg_color = cfg['leafmachine']['leaf_segmentation']['overlay_background_color']
    keep_best = cfg['leafmachine']['leaf_segmentation']['keep_only_best_one_leaf_one_petiole']
    save_overlay_pdf = cfg['leafmachine']['leaf_segmentation']['save_segmentation_overlay_images_to_pdf']
    save_each_segmentation_overlay_image = cfg['leafmachine']['leaf_segmentation']['save_segmentation_overlay_images_to_pdf']
    save_individual_overlay_images = cfg['leafmachine']['leaf_segmentation']['save_individual_overlay_images']
    save_ind_masks_color = cfg['leafmachine']['leaf_segmentation']['save_masks_color']
    # save_ind_masks_index = cfg['leafmachine']['leaf_segmentation']['save_masks_index']
    save_full_image_masks_color = cfg['leafmachine']['leaf_segmentation']['save_full_image_masks_color']
    # save_full_image_masks_index = cfg['leafmachine']['leaf_segmentation']['save_full_image_masks_index']
    use_efds_for_masks = cfg['leafmachine']['leaf_segmentation']['use_efds_for_png_masks']
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
                    # print(seg_name)
                    logger.debug(f'segmenting - {seg_name}')

                    seg_name_short = seg_name.split("__")[2]
                    # cropped_overlay = []


                    # Segment!
                    # fig, out_polygons, out_bboxes, out_labels, out_color = Instance_Detector.segment(img_cropped, generate_overlay, overlay_dpi, bg_color)
                    try:
                        out_polygons, out_bboxes, out_labels, out_color = Instance_Detector.segment(img_cropped, generate_overlay, overlay_dpi, bg_color)
                    except:
                        detected_components = []
                        cropped_overlay = []
                        overlay_data = []
                        cropped_overlay_size = []
                        out_polygons = []
                    
                    if len(out_polygons) > 0: # Success
                        if keep_best:
                            out_polygons, out_bboxes, out_labels, out_color = keep_rows(out_polygons, out_bboxes, out_labels, out_color, get_string_indices(out_labels))
                        
                        if (out_polygons is None) and (out_bboxes is None) and (out_labels is None) and (out_color is None):
                            detected_components = []
                            cropped_overlay = []
                            overlay_data = []
                            cropped_overlay_size = []
                        else:
                            detected_components, cropped_overlay, overlay_data = create_overlay_and_calculate_props(seg_name, img_cropped, out_polygons, out_labels, out_color, cfg)
                            # full_image = create_insert_legacy(full_image, cropped_overlay, seg_name_short)
                            full_image = create_insert(full_image, overlay_data, seg_name_short, cfg)

                            cropped_overlay_size = cropped_overlay.shape

                    else: # Fail
                        detected_components = []
                        cropped_overlay = []
                        overlay_data = []
                        cropped_overlay_size = []

                    with lock:
                        value[dict_name_seg].append({seg_name: detected_components})
                        seg_overlay[filename].append({seg_name: cropped_overlay})
                        seg_overlay_data[filename].append({seg_name: overlay_data})

                        save_rgb_cropped(save_rgb_cropped_images, seg_name, img_cropped, leaf_type, Dirs)

                        save_individual_segmentations(save_individual_overlay_images, dict_name_seg, seg_name, cropped_overlay, Dirs)

                        full_mask = save_masks_color(save_ind_masks_color, save_full_image_masks_color, use_efds_for_masks, full_mask, overlay_data, cropped_overlay_size, full_size, seg_name, seg_name_short, leaf_type, Dirs)

        save_full_masks(save_full_image_masks_color, full_mask, filename, leaf_type, Dirs)
        save_full_overlay_images(save_each_segmentation_overlay_image, full_image, filename, leaf_type, Dirs)

        filenames.append(filename)
        
        if save_overlay_pdf:
            full_images.append(full_image)

        if save_full_image_masks_color:
            full_masks.append(full_mask)
        
    save_full_image_segmentations(save_overlay_pdf, dict_name_seg, full_images, filenames, Dirs, cfg, batch, n_batches, lock)#, start, end)
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
                    if leaf_type == 0:
                        if 'leaf' in cls:
                            color = [46, 255, 0]
                        elif 'petiole' in cls:
                            color = [0, 173, 255]
                        elif 'hole' in cls:
                            color = [209, 0, 255]
                        else:
                            color = [255, 255, 255]
                    elif leaf_type == 1:
                        if 'leaf' in cls:
                            color = [0, 200, 255]
                        elif 'petiole' in cls:
                            color = [255, 140, 0]
                        elif 'hole' in cls:
                            color = [200, 0, 255]
                        else:
                            color = [255, 255, 255]
                    # Draw the filled polygon on the image
                    draw.polygon(poly, fill=tuple(color))
            if leaf_type == 0:
                img.save(os.path.join(Dirs.segmentation_masks_color_whole_leaves, '.'.join([seg_name, 'png'])))
            elif leaf_type == 1:
                img.save(os.path.join(Dirs.segmentation_masks_color_partial_leaves, '.'.join([seg_name, 'png'])))

        if save_full_image_masks_color:
            if '-' in seg_name_short:
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
                        if leaf_type == 0:
                            if 'leaf' in cls:
                                color = [46, 255, 0]
                            elif 'petiole' in cls:
                                color = [0, 173, 255]
                            elif 'hole' in cls:
                                color = [209, 0, 255]
                            else:
                                color = [255,255,255]
                        elif leaf_type == 1:
                            if 'leaf' in cls:
                                color = [0, 200, 255]
                            elif 'petiole' in cls:
                                color = [255, 140, 0]
                            elif 'hole' in cls:
                                color = [200, 0, 255]
                            else:
                                color = [255, 255, 255]
                        # Draw the filled polygon on the image
                        draw.polygon(poly, fill=tuple(color))
    return full_mask

def save_full_masks(save_full_image_masks_color, full_mask, filename, leaf_type, Dirs):
    if save_full_image_masks_color:
        if leaf_type == 0:
            full_mask.save(os.path.join(Dirs.segmentation_masks_full_image_color_whole_leaves, '.'.join([filename, 'png'])))
        elif leaf_type == 1:
            full_mask.save(os.path.join(Dirs.segmentation_masks_full_image_color_partial_leaves, '.'.join([filename, 'png'])))

def save_full_overlay_images(save_each_segmentation_overlay_image, full_image, filename, leaf_type, Dirs):
    if save_each_segmentation_overlay_image:
        if leaf_type == 0:
            try:
                full_image.save(os.path.join(Dirs.segmentation_overlay_whole_leaves, '.'.join([filename, 'jpg'])))
            except:
                full_image = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(Dirs.segmentation_overlay_whole_leaves, '.'.join([filename, 'jpg'])) , full_image)
        elif leaf_type == 1:
            try:
                full_image.save(os.path.join(Dirs.segmentation_overlay_partial_leaves, '.'.join([filename, 'jpg'])))
            except:
                full_image = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(Dirs.segmentation_overlay_partial_leaves, '.'.join([filename, 'jpg'])) , full_image)

    

def create_insert(full_image, overlay_data, seg_name_short, cfg):
    if '-' in seg_name_short:
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
                draw.polygon(rect, fill=None, outline=outline_color, width=width)
        
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

def save_full_image_segmentations(save_overlay_pdf, dict_name_seg, full_images, filenames, Dirs, cfg, batch, n_batches, lock):#, start, end):
    color_bg = cfg['leafmachine']['leaf_segmentation']['overlay_background_color']
    overlay_dpi = cfg['leafmachine']['leaf_segmentation']['overlay_dpi']

    if color_bg == 'black':
        color_text = 'white'
    else:
        color_text = 'black'

    '''if save_overlay_pdf:
        pdf_name = os.path.join(Dirs.segmentation_overlay_pdfs, ''.join([dict_name_seg, '_',str(batch+1), 'of',str(n_batches),'.pdf']))
        with PdfPages(pdf_name) as pdf:
            for idx, img in enumerate(full_images):
                # Create a new figure
                fig = plt.figure()'''
    if save_overlay_pdf:
        pdf_name = os.path.join(Dirs.segmentation_overlay_pdfs, ''.join([dict_name_seg, '_',str(batch+1), 'of',str(n_batches),'.pdf']))
        with PdfPages(pdf_name) as pdf:
            for idx, img in enumerate(full_images):
                # Acquire the lock before accessing the list
                with lock:
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
        indices_to_keep = hole_indices
    # Only petiole and hole
    elif (leaf_index is None) and (petiole_index is not None) and (hole_indices is not None):
        indices_to_keep =  [petiole_index] + hole_indices
    # Only holes and no leaves or petiole
    elif (leaf_index is None) and (petiole_index is None) and (hole_indices is not None):
        indices_to_keep = hole_indices
    # Only leaves and hole
    elif (leaf_index is not None) and (petiole_index is None) and (hole_indices is not None):
        indices_to_keep =  [leaf_index] + hole_indices
    else:
        indices_to_keep = None

    # get empty list1 values []
    indices_empty = [i for i, lst in enumerate(list1) if not lst]
    indices_to_keep = [i for i in indices_to_keep if i not in indices_empty]

    if indices_to_keep is not None:
        list1 = [list1[i] for i in indices_to_keep]
        list2 = [list2[i] for i in indices_to_keep]
        list3 = [list3[i] for i in indices_to_keep]
        list4 = [list4[i] for i in indices_to_keep]
        return list1, list2, list3, list4
    else:
        return None, None, None, None

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
            draw.polygon(bbox, outline=outline_color, width=width)
            overlay_rect.append(bbox)
            overlay_poly.append(max_poly)
            overlay_color.append({out_labels[i]: [fill_color, outline_color]})

            if '__L__' in seg_name:
                if do_get_efds:
                    _, value = next(iter(component.items()))
                    efd = value['efds']['efd_pts_PIL']
                    # efd = efd['pts_efd']
                    draw.polygon(efd, fill=None, outline=(210,30,135), width=width)
                    overlay_efd.append(efd)
            elif '__PL__' in seg_name:
                if do_get_efds:
                    _, value = next(iter(component.items()))
                    efd = value['efds']['efd_pts_PIL']
                    # efd = efd['pts_efd']
                    draw.polygon(efd, fill=None, outline=(255,120,0), width=width)
                    overlay_efd.append(efd)
    # PIL               
    # cropped_overlay.show() # wrong colors without changing to RGB
    # cv2
    cropped_overlay = np.array(cropped_overlay)
    overlay_data = [overlay_poly, overlay_efd, overlay_rect, overlay_color]
    # cv2.imshow('img_crop', cropped_overlay)
    # cv2.waitKey(0)
    return detected_components, cropped_overlay, overlay_data



def crop_images_to_bbox(dict, cls, dict_name_cropped, dict_from, Project):
    # For each image, iterate through the whole leaves, segment, report data back to dict_plant_components
    for filename, value in dict.items():
        value[dict_name_cropped] = []
        if dict_from in value:
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
        if "Detections_Plant_Components" in value:
            filtered_components = [val for val in value["Detections_Plant_Components"] if val[0] == cls]
            value[dict_name_yolo] = filtered_components

    for filename, value in dict.items():
        if "Detections_Plant_Components" in value:
            filtered_components = [val for val in value["Detections_Plant_Components"] if val[0] == cls]
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



''''''
if __name__ == '__main__':
    with open('D:/Dropbox/LM2_Env/Image_Datasets/TEST_LM2/TEST_2023_01_24__16-03-18/Plant_Components/json/Plant_Components.json') as json_file:
        dict_plant_components = json.load(json_file)
    segment_leaves([], 'D:\Dropbox\LeafMachine2', 'D:\Dropbox\LM2_Env\Image_Datasets\SET_Acacia\Images_GBIF_Acacia_Prickles', [], dict_plant_components)
''''''