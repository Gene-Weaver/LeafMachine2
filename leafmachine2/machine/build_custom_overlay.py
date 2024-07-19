import os, json, random, inspect, sys, cv2, itertools
from PIL import Image, ImageDraw
from dataclasses import dataclass, field
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from time import perf_counter
import concurrent.futures
from threading import Lock

currentdir = os.path.dirname(inspect.getfile(inspect.currentframe()))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(currentdir)
sys.path.append(parentdir)
# from segmentation.detectron2.segment_leaves import create_insert

def build_custom_overlay_parallel(cfg, time_report, logger, dir_home, Project, batch, Dirs):
    start_t = perf_counter()
    logger.name = f'[BATCH {batch+1} Build Custom Overlay]'
    logger.info(f'Creating overlay for batch {batch+1}')

    line_w_archival = cfg['leafmachine']['overlay']['line_width_archival']
    line_w_plant = cfg['leafmachine']['overlay']['line_width_plant']
    show_archival = cfg['leafmachine']['overlay']['show_archival_detections']
    ignore_archival = cfg['leafmachine']['overlay']['ignore_archival_detections_classes']
    show_plant = cfg['leafmachine']['overlay']['show_plant_detections']
    ignore_plant = cfg['leafmachine']['overlay']['ignore_plant_detections_classes']
    show_segmentations = cfg['leafmachine']['overlay']['show_segmentations']
    show_landmarks = cfg['leafmachine']['overlay']['show_landmarks']
    ignore_landmarks = cfg['leafmachine']['overlay']['ignore_landmark_classes']

    lock = Lock()  # Create a lock object

    if cfg['leafmachine']['project']['num_workers'] is None:
        num_workers = 1
    else:
        num_workers = int(cfg['leafmachine']['project']['num_workers'])

    filenames = []
    overlay_images = []
    ruler_images = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for filename, analysis in Project.project_data_list[batch].items():
            logger.info(f'Creating overlay for {filename}')
            futures.append(executor.submit(process_file, Project, filename, analysis, line_w_archival, show_archival, ignore_archival, line_w_plant, show_plant, ignore_plant, show_segmentations, show_landmarks, cfg, Dirs, lock))  # Pass the lock object to the process_file function

        logger.info(f'Merging results from {num_workers} workers')
        for future in concurrent.futures.as_completed(futures):
            filename, image_overlay, ruler_img = future.result()
            # save_overlay_images_to_jpg(image_overlay, filename, Dirs, cfg)  # Use the lock object when writing to the file
            filenames.append(filename)
            overlay_images.append(image_overlay)
            ruler_images.append(ruler_img)

    logger.info(f'Saving batch {batch+1} overlay images to PDF')
    save_custom_overlay_to_PDF(filenames, overlay_images, ruler_images, batch, Dirs, Project, cfg)
    end_t = perf_counter()

    t_overlay = f"[Batch {batch+1}: Build Custom Overlay elapsed time] {round(end_t - start_t)} seconds ({round((end_t - start_t)/60)} minutes)"
    logger.info(t_overlay)
    time_report['t_overlay'] = t_overlay
    return time_report


def process_file(Project, filename, analysis, line_w_archival, show_archival, ignore_archival, line_w_plant, show_plant, ignore_plant, show_segmentations, show_landmarks, cfg, Dirs, lock):

    if 'height' in analysis:
        height = analysis['height']
    else:
        height = 0

    if 'width' in analysis:
        width = analysis['width']
    else:
        width = 0

    if 'Detections_Archival_Components' in analysis:
        archival = analysis['Detections_Archival_Components']
    else:
        archival = []

    if 'Detections_Plant_Components' in analysis:
        plant = analysis['Detections_Plant_Components']
    else:
        plant = []

    if cfg['leafmachine']['leaf_segmentation']['segment_whole_leaves']:
        if 'Segmentation_Whole_Leaf' in analysis:
            Segmentation_Whole_Leaf = analysis['Segmentation_Whole_Leaf']
        else:
            Segmentation_Whole_Leaf = []
    else:
        Segmentation_Whole_Leaf = []

    if cfg['leafmachine']['leaf_segmentation']['segment_partial_leaves']:
        if 'Segmentation_Partial_Leaf' in analysis:
            Segmentation_Partial_Leaf = analysis['Segmentation_Partial_Leaf']
        else:
            Segmentation_Partial_Leaf = []
    else:
        Segmentation_Partial_Leaf = []

    if cfg['leafmachine']['landmark_detector']['landmark_whole_leaves']:
        if 'Landmarks_Whole_Leaves' in analysis:
            Landmarks_Whole_Leaves = analysis['Landmarks_Whole_Leaves']
        else:
            Landmarks_Whole_Leaves = []
    else:
        Landmarks_Whole_Leaves = []

    if cfg['leafmachine']['landmark_detector']['landmark_partial_leaves']:
        if 'Landmarks_Partial_Leaves' in analysis:
            Landmarks_Partial_Leaves = analysis['Segmentation_Partial_Leaf']
        else:
            Landmarks_Partial_Leaves = []
    else:
        Landmarks_Partial_Leaves = []


    if 'Ruler_Info' in analysis:
        Ruler_Images = analysis['Ruler_Info']
    else:
        Ruler_Images = []


    try:
        image = Image.open(os.path.join(Project.dir_images, '.'.join([filename, 'jpg'])))
    except:
        image = Image.open(os.path.join(Project.dir_images, '.'.join([filename, 'jpeg'])))

    image_overlay = image
    if image_overlay.mode != 'RGB':
        image_overlay = image_overlay.convert('RGB')

    with lock:

        image_overlay = add_archival_detections(image_overlay, archival, height, width, line_w_archival, show_archival, ignore_archival, cfg)

        image_overlay = add_plant_detections(image_overlay, plant, height, width, line_w_plant, show_plant, ignore_plant, cfg)

        image_overlay = add_segmentations(image_overlay, Segmentation_Whole_Leaf, Segmentation_Partial_Leaf, show_segmentations, cfg)

        image_overlay = add_landmarks(image_overlay, Landmarks_Whole_Leaves, Landmarks_Partial_Leaves, show_landmarks, cfg)

        ruler_img = get_ruler_images(Ruler_Images, cfg)

        save_overlay_images_to_jpg(image_overlay, filename, Dirs, cfg)

    return filename, image_overlay, ruler_img


def build_custom_overlay(cfg, logger, dir_home, Project, batch, Dirs):
    start_t = perf_counter()
    logger.name = f'[BATCH {batch+1} Build Custom Overlay]'
    logger.info(f'Creating overlay for {batch+1}')

    line_w_archival = cfg['leafmachine']['overlay']['line_width_archival']
    line_w_plant = cfg['leafmachine']['overlay']['line_width_plant']
    show_archival = cfg['leafmachine']['overlay']['show_archival_detections']
    ignore_archival = cfg['leafmachine']['overlay']['ignore_archival_detections_classes']
    show_plant = cfg['leafmachine']['overlay']['show_plant_detections']
    ignore_plant = cfg['leafmachine']['overlay']['ignore_plant_detections_classes']
    show_segmentations = cfg['leafmachine']['overlay']['show_segmentations']
    show_landmarks = cfg['leafmachine']['overlay']['show_landmarks']
    ignore_landmarks = cfg['leafmachine']['overlay']['ignore_landmark_classes']

    filenames = []
    overlay_images = []
    for filename, analysis in Project.project_data_list[batch].items():
        logger.info(f'Creating overlay for {filename}')
        # print(filename)
        # print(analysis)
        if 'height' in analysis:
            height = analysis['height']
        else:
            height = 0

        if 'width' in analysis:
            width = analysis['width']
        else:
            width = 0

        if 'Detections_Archival_Components' in analysis:
            archival = analysis['Detections_Archival_Components']
        else:
            archival = []

        if 'Detections_Plant_Components' in analysis:
            plant = analysis['Detections_Plant_Components']
        else:
            plant = []

        # Whole_Leaf_BBoxes = analysis['Whole_Leaf_BBoxes']
        # Whole_Leaf_BBoxes_YOLO = analysis['Whole_Leaf_BBoxes_YOLO']
        # Whole_Leaf_Cropped = analysis['Whole_Leaf_Cropped']
        
        # Partial_Leaf_BBoxes_YOLO = analysis['Partial_Leaf_BBoxes_YOLO']
        # Partial_Leaf_BBoxes = analysis['Partial_Leaf_BBoxes']
        # Partial_Leaf_Cropped = analysis['Partial_Leaf_Cropped']

        if cfg['leafmachine']['leaf_segmentation']['segment_whole_leaves']:
            if 'Segmentation_Whole_Leaf' in analysis:
                Segmentation_Whole_Leaf = analysis['Segmentation_Whole_Leaf']
            else:
                Segmentation_Whole_Leaf = []
        else:
            Segmentation_Whole_Leaf = []

        if cfg['leafmachine']['leaf_segmentation']['segment_partial_leaves']:
            if 'Segmentation_Partial_Leaf' in analysis:
                Segmentation_Partial_Leaf = analysis['Segmentation_Partial_Leaf']
            else:
                Segmentation_Partial_Leaf = []
        else:
            Segmentation_Partial_Leaf = []

        if cfg['leafmachine']['landmark_detector']['landmark_whole_leaves']:
            if 'Landmarks_Whole_Leaves' in analysis:
                Landmarks_Whole_Leaves = analysis['Landmarks_Whole_Leaves']
            else:
                Landmarks_Whole_Leaves = []
        else:
            Landmarks_Whole_Leaves = []

        if cfg['leafmachine']['landmark_detector']['landmark_partial_leaves']:
            if 'Landmarks_Partial_Leaves' in analysis:
                Landmarks_Partial_Leaves = analysis['Segmentation_Partial_Leaf']
            else:
                Landmarks_Partial_Leaves = []
        else:
            Landmarks_Partial_Leaves = []
        
        try:
            image = Image.open(os.path.join(Project.dir_images, '.'.join([filename, 'jpg'])))
        except:
            image = Image.open(os.path.join(Project.dir_images, '.'.join([filename, 'jpeg'])))

        image_overlay = image
        if image_overlay.mode != 'RGB':
            image_overlay = image_overlay.convert('RGB')

        image_overlay = add_archival_detections(image_overlay, archival, height, width, line_w_archival, show_archival, ignore_archival, cfg)

        image_overlay = add_plant_detections(image_overlay, plant, height, width, line_w_plant, show_plant, ignore_plant, cfg)

        image_overlay = add_segmentations(image_overlay, Segmentation_Whole_Leaf, Segmentation_Partial_Leaf, show_segmentations, cfg)

        image_overlay = add_landmarks(image_overlay, Landmarks_Whole_Leaves, Landmarks_Partial_Leaves, show_landmarks, cfg)
        
        # add_efds()

        # add_landmarks()

        # create_panel() # with individual leaves inside a table to the right of the full image
        #images in panel have more info printed with them 

        save_overlay_images_to_jpg(image_overlay, filename, Dirs, cfg)

        filenames.append(filename)
        overlay_images.append(image_overlay)

    # save_custom_overlay_to_PDF(filenames, overlay_images, batch, Dirs, Project, cfg)
    end_t = perf_counter()
    logger.info(f'Batch {batch+1}: Build Custom Overlay Duration --> {round((end_t - start_t)/60)} minutes')
    

def add_landmarks(image_overlay, Landmarks_Whole_Leaves, Landmarks_Partial_Leaves, show_landmarks, cfg):
    if show_landmarks:
        if cfg['leafmachine']['landmark_detector']['landmark_whole_leaves']:
            leaf_type = 0
            for object in Landmarks_Whole_Leaves:
                for seg_name, overlay_data in object.items():
                    seg_name_short = seg_name.split("__")[-1]

                    # status = overlay_data[0]
                    data = overlay_data[1]['landmarks']
                    status = overlay_data[0]['landmark_status']
                    # if status == 'incomplete':
                    #     pass
                    # else:
                    image_overlay = insert_landmark(image_overlay, data, seg_name_short, cfg)

                    '''for part in overlay_data:
                        key, overlay_data_insert = next(iter(part.items()))   
                        overlay_poly = overlay_data_insert['polygon_closed']
                        overlay_rect = overlay_data_insert['bbox_min']
                        overlay_efd = overlay_data_insert['efds']['efd_pts_PIL']

                        if 'leaf' in key:
                            c_outline, c_fill = get_color('seg_leaf_whole', 'SEG_WHOLE', cfg)
                        elif 'petiole' in key:
                            c_outline, c_fill = get_color('seg_leaf_whole_petiole', 'SEG_WHOLE', cfg)
                        elif 'hole' in key:
                            c_outline, c_fill = get_color('seg_hole', 'SEG_WHOLE', cfg)

                        overlay_data_insert = [overlay_poly, overlay_efd, overlay_rect, c_outline, c_fill]
                        image_overlay = insert_seg(image_overlay, overlay_data_insert, seg_name_short, cfg)'''
            # im_show = cv2.cvtColor(np.array(image_overlay), cv2.COLOR_RGB2BGR)
            # cv2.imshow('overlay', im_show)
            # cv2.waitKey(0)
        if cfg['leafmachine']['landmark_detector']['landmark_partial_leaves']: # TODO finish this
            leaf_type = 1
            for object in Landmarks_Partial_Leaves:
                for seg_name, overlay_data in object.items():
                    seg_name_short = seg_name.split("__")[-1]
                    for part in overlay_data:
                        key, overlay_data_insert = next(iter(overlay_data[0].items()))   
                        overlay_poly = overlay_data_insert['polygon_closed']
                        overlay_rect = overlay_data_insert['bbox_min']
                        overlay_efd = overlay_data_insert['efds']['efd_pts_PIL']

                        if 'leaf' in key:
                            c_outline, c_fill = get_color('seg_leaf_partial', 'SEG_PARTIAL', cfg)
                        elif 'petiole' in key:
                            c_outline, c_fill = get_color('seg_leaf_partial_petiole', 'SEG_PARTIAL', cfg)
                        elif 'hole' in key:
                            c_outline, c_fill = get_color('seg_hole', 'SEG_PARTIAL', cfg)

                        overlay_data_insert = [overlay_poly, overlay_efd, overlay_rect, c_outline, c_fill]
                        image_overlay = insert_seg(image_overlay, overlay_data_insert, seg_name_short, cfg) ### *** TODO
            # im_show = cv2.cvtColor(np.array(image_overlay), cv2.COLOR_RGB2BGR)
            # cv2.imshow('overlay', im_show)
            # cv2.waitKey(0)
    return image_overlay

def add_segmentations(image_overlay, Segmentation_Whole_Leaf, Segmentation_Partial_Leaf, show_segmentations, cfg):
    if show_segmentations:
        if cfg['leafmachine']['leaf_segmentation']['segment_whole_leaves']:
            leaf_type = 0
            for object in Segmentation_Whole_Leaf:
                for seg_name, overlay_data in object.items():
                    seg_name_short = seg_name.split("__")[-1]
                    for part in overlay_data:
                        key, overlay_data_insert = next(iter(part.items()))   
                        overlay_poly = overlay_data_insert['polygon_closed']
                        overlay_rect = overlay_data_insert['bbox_min']
                        if cfg['leafmachine']['leaf_segmentation']['calculate_elliptic_fourier_descriptors']:
                            overlay_efd = overlay_data_insert['efds']['efd_pts_PIL']
                        else:
                            overlay_efd = None

                        if 'leaf' in key:
                            c_outline, c_fill = get_color('seg_leaf_whole', 'SEG_WHOLE', cfg)
                        elif 'petiole' in key:
                            c_outline, c_fill = get_color('seg_leaf_whole_petiole', 'SEG_WHOLE', cfg)
                        elif 'hole' in key:
                            c_outline, c_fill = get_color('seg_hole', 'SEG_WHOLE', cfg)

                        overlay_data_insert = [overlay_poly, overlay_efd, overlay_rect, c_outline, c_fill]
                        image_overlay = insert_seg(image_overlay, overlay_data_insert, seg_name_short, cfg)
            # im_show = cv2.cvtColor(np.array(image_overlay), cv2.COLOR_RGB2BGR)
            # cv2.imshow('overlay', im_show)
            # cv2.waitKey(0)
        if cfg['leafmachine']['leaf_segmentation']['segment_partial_leaves']:
            leaf_type = 1
            for object in Segmentation_Partial_Leaf:
                for seg_name, overlay_data in object.items():
                    seg_name_short = seg_name.split("__")[-1]
                    for part in overlay_data:
                        key, overlay_data_insert = next(iter(overlay_data[0].items()))   
                        overlay_poly = overlay_data_insert['polygon_closed']
                        overlay_rect = overlay_data_insert['bbox_min']
                        overlay_efd = overlay_data_insert['efds']['efd_pts_PIL']

                        if 'leaf' in key:
                            c_outline, c_fill = get_color('seg_leaf_partial', 'SEG_PARTIAL', cfg)
                        elif 'petiole' in key:
                            c_outline, c_fill = get_color('seg_leaf_partial_petiole', 'SEG_PARTIAL', cfg)
                        elif 'hole' in key:
                            c_outline, c_fill = get_color('seg_hole', 'SEG_PARTIAL', cfg)

                        overlay_data_insert = [overlay_poly, overlay_efd, overlay_rect, c_outline, c_fill]
                        image_overlay = insert_seg(image_overlay, overlay_data_insert, seg_name_short, cfg)
            # im_show = cv2.cvtColor(np.array(image_overlay), cv2.COLOR_RGB2BGR)
            # cv2.imshow('overlay', im_show)
            # cv2.waitKey(0)
    return image_overlay

def insert_landmark(full_image, data, seg_name_short, cfg):
    width = cfg['leafmachine']['overlay']['line_width_seg'] + 2
    # origin_x = int(seg_name_short.split('-')[0])
    # origin_y = int(seg_name_short.split('-')[1])
    # initialize
    full_image = np.asarray(full_image)
    full_image = Image.fromarray(full_image)
    draw = ImageDraw.Draw(full_image, "RGBA")

    A, B = check_lamina_length(data)
    if (A is not None) and (B is not None):
        order_points_plot([A, B], 'lamina_length', full_image, draw, width)

    if hasattr(data, 't_midvein_fit_points'):
        if (data.t_midvein_fit_points != []):
            order_points_plot(data.t_midvein_fit_points, 'midvein_fit', full_image, draw, width)

    if hasattr(data, 't_apex_center'):
        if (data.t_apex_center != []) and (data.has_apex):
            order_points_plot([data.t_apex_left, data.t_apex_center, data.t_apex_right], data.apex_angle_type, full_image, draw, width)
    
    if hasattr(data, 't_base_center'):
        if (data.t_base_center != []) and (data.has_base):
            order_points_plot([data.t_base_left, data.t_base_center, data.t_base_right], data.base_angle_type, full_image, draw, width)

    if data.has_width:
        if hasattr(data, 't_width_left') and hasattr(data, 't_width_right'):
            if (data.t_width_left != []) and (data.t_width_right != []):
                order_points_plot([data.t_width_left, data.t_width_right], 'lamina_width', full_image, draw, width)
    else:
        if hasattr(data, 't_width_infer'):
            if data.t_width_infer != []:
                order_points_plot(data.t_width_infer, 'infer_width', full_image, draw, width)

    if hasattr(data, 't_midvein'):
        if (data.t_midvein != []) and (data.has_midvein != []):
            order_points_plot(data.t_midvein, 'midvein_trace', full_image, draw, width)
    
    if hasattr(data, 't_petiole'):
        if (data.t_petiole != []) and (data.has_ordered_petiole != []):
            order_points_plot(data.t_petiole, 'petiole_trace', full_image, draw, width)

    if hasattr(data, 't_lobes'):
        if (data.t_lobes != []) and (data.has_lobes != []):
            order_points_plot(data.t_lobes, 'lobes', full_image, draw, width)

    # Lamina tip and base
    if hasattr(data, 't_lamina_tip'):
        if (data.t_lamina_tip != []) and (data.has_lamina_tip != []):
            draw.ellipse((data.t_lamina_tip[0]-width, data.t_lamina_tip[1]-width, data.t_lamina_tip[0]+width, data.t_lamina_tip[1]+width), fill=(0, 255, 0, 255), outline=(0, 0, 0, 255))
    if hasattr(data, 't_lamina_base'):
        if (data.t_lamina_base != []) and (data.has_lamina_base != []):
            draw.ellipse((data.t_lamina_base[0]-width, data.t_lamina_base[1]-width, data.t_lamina_base[0]+width, data.t_lamina_base[1]+width), fill=(255, 0, 0, 255), outline=(0, 0, 0, 255))

        # Apex angle
    if hasattr(data, 't_apex_left'):
        if (data.t_apex_left != []) and (data.has_apex != []):
            draw.ellipse((data.t_apex_left[0]-width, data.t_apex_left[1]-width, data.t_apex_left[0]+width, data.t_apex_left[1]+width), fill=(255, 0, 0, 255))
    if hasattr(data, 't_apex_right'):
        if (data.t_apex_right != []) and (data.has_apex != []):
            draw.ellipse((data.t_apex_right[0]-width, data.t_apex_right[1]-width, data.t_apex_right[0]+width, data.t_apex_right[1]+width), fill=(0, 0, 255, 255))

        # Base angle
    if hasattr(data, 't_base_left'):
        if (data.t_base_left != []) and (data.has_base != []):
            draw.ellipse((data.t_base_left[0]-width, data.t_base_left[1]-width, data.t_base_left[0]+width, data.t_base_left[1]+width), fill=(255, 0, 0, 255))
    if hasattr(data, 't_base_right'):
        if (data.t_base_right != []) and (data.has_base != []):
            draw.ellipse((data.t_base_right[0]-width, data.t_base_right[1]-width, data.t_base_right[0]+width, data.t_base_right[1]+width), fill=(0, 0, 255, 255))

    return full_image


def order_points_plot(points, version, full_image, draw, thick):
    if version == 'midvein_trace':
        # color = (0, 255, 0)
        color = (0, 0, 0)
        # thick = 2
    elif version == 'petiole_trace':
        color = (0, 255, 255)
        # thick = 2
    elif version == 'lamina_width':
        color = (255, 0, 0)
    elif version == 'lamina_length':
        color = (255, 255, 255)
        # thick = 2
    elif version == 'lamina_width_alt':
        color = (255, 100, 100)
    elif version == 'infer_width':
        color = (255, 100, 100)
    elif version == 'midvein_fit':
        color = (255, 255, 255)
        thick = 2
    elif version == 'not_reflex':
        color = (255, 0, 200)
        # thick = 3
    elif version == 'reflex':
        color = (255, 120, 0)
        # thick = 3
    elif version == 'petiole_tip_alt':
        color = (100, 55, 255)
        # thick = 1
    elif version == 'petiole_tip':
        color = (55, 255, 100)
        # thick = 1
    elif version == 'failed_angle':
        color = (0, 0, 0)
        # thick = 3
    elif version == 'lobes':
        color = (0, 30, 255)
        # thick = 3

    # Convert the points to a list of tuples and round to integer values
    points_list = [tuple(np.round(p).astype(int)) for p in points]

    if version == 'infer_width':
        for p in points_list:
            draw.ellipse((p[0]-thick, p[1]-thick, p[0]+thick, p[1]+thick), fill=color+(255,))
    elif version == 'midvein_fit':
        for p in points_list:
            draw.ellipse((p[0]-thick, p[1]-thick, p[0]+thick, p[1]+thick), fill=color+(255,))
    elif version == 'lamina_length':
        for i in range(len(points_list) - 1):
            draw.line([points_list[i], points_list[i+1]], fill=color, width=thick)
    elif version == 'lobes':
        for i in range(len(points_list) - 1):
            lobe = points_list[i]
            draw.ellipse((lobe[0]-thick*3, lobe[1]-thick*3, lobe[0]+thick*3, lobe[1]+thick*3), outline=color, width=int(thick/2))
            draw.ellipse((lobe[0]-thick, lobe[1]-thick, lobe[0]+thick, lobe[1]+thick), outline=color, width=thick)
    else:
        for i in range(len(points_list) - 1):
            draw.line([points_list[i], points_list[i+1]], fill=color+(255,), width=thick)

    return full_image

def check_lamina_length(data):
    if data.has_lamina_base and data.has_lamina_tip:
        return data.t_lamina_base, data.t_lamina_tip
    else:
        if data.has_lamina_base and (not data.has_lamina_tip) and data.has_apex: # lamina base and apex center
            return data.t_lamina_base, data.t_apex_center
        elif data.has_lamina_tip and (not data.has_lamina_base) and data.has_base: # lamina tip and base center
            return data.t_lamina_tip, data.t_apex_center
        elif (not data.has_lamina_tip) and (not data.has_lamina_base) and data.has_apex and data.has_base: # apex center and base center
            return data.t_base_center, data.t_apex_center
        else:
            return None, None

def insert_seg(full_image, overlay_data, seg_name_short, cfg):
    width = cfg['leafmachine']['overlay']['line_width_seg']
    width_efd = cfg['leafmachine']['overlay']['line_width_efd']
    origin_x = int(seg_name_short.split('-')[0])
    origin_y = int(seg_name_short.split('-')[1])
    # unpack
    overlay_poly, overlay_efd, overlay_rect, c_outline, c_fill = overlay_data
    # fill_color = overlay_color[0][0]
    # outline_color =  overlay_color[0][1]

    # initialize
    full_image = np.asarray(full_image)
    full_image = Image.fromarray(full_image)
    draw = ImageDraw.Draw(full_image, "RGBA")

    if len(overlay_poly) != 0:
        overlay_poly = [(x+origin_x, y+origin_y) for x, y in overlay_poly]
        draw.polygon(overlay_poly, fill=c_fill, outline=c_outline, width=width)
        
    if len(overlay_rect) != 0:
        overlay_rect = [(x+origin_x, y+origin_y) for x, y in overlay_rect]
        draw.polygon(overlay_rect, fill=None, outline=c_outline, width=width)
        
    if overlay_efd:
        if len(overlay_efd) != 0:
            overlay_efd = [(x+origin_x, y+origin_y) for x, y in overlay_efd]
            draw.polygon(overlay_efd, fill=None, outline=(135,30,210), width=width_efd)
        

    return full_image

def add_plant_detections(image_overlay, plant, height, width, line_w, show_plant, ignore_plant, cfg):
    if show_plant:
        draw = ImageDraw.Draw(image_overlay, "RGBA")

        for annotation in plant:
            anno = yolo_to_position(annotation, height, width, 'PLANT')
            if anno[0] not in ignore_plant:
                polygon = [(anno[1], anno[2]), (anno[3], anno[2]), (anno[3], anno[4]), (anno[1], anno[4])]
                c_outline, c_fill = get_color(anno[0], 'PLANT', cfg)
                draw.polygon(polygon, fill=c_fill, outline=c_outline, width=line_w)
        # im_show = cv2.cvtColor(np.array(image_overlay), cv2.COLOR_RGB2BGR)
        # cv2.imshow('overlay', im_show)
        # cv2.waitKey(0)
    return image_overlay

def add_archival_detections(image_overlay, archival, height, width, line_w, show_archival, ignore_archival, cfg):
    if show_archival:
        draw = ImageDraw.Draw(image_overlay, "RGBA")

        for annotation in archival:
            anno = yolo_to_position(annotation, height, width, 'ARCHIVAL')
            if anno[0] not in ignore_archival:
                polygon = [(anno[1], anno[2]), (anno[3], anno[2]), (anno[3], anno[4]), (anno[1], anno[4])]
                c_outline, c_fill = get_color(anno[0], 'ARCHIVAL', cfg)
                draw.polygon(polygon, fill=c_fill, outline=c_outline, width=line_w)
        # im_show = cv2.cvtColor(np.array(image_overlay), cv2.COLOR_RGB2BGR)
        # cv2.imshow('overlay', im_show)
        # cv2.waitKey(0)
    return image_overlay

def get_ruler_images(ruler_info, cfg):
    ruler_composite = None

    for row in ruler_info:
        ruler_img = row['summary_img']
        ruler_composite = stack_images(ruler_composite, ruler_img)
        
    return ruler_composite
        
def stack_images(img1, img2):
    # If img1 is empty, use the dimensions of img2
    if img1 is None:
        return img2

    else:
        # Get the dimensions of the images
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape

        # Calculate the dimensions of the stacked image
        h_stacked = h1 + h2
        w_stacked = max(w1, w2)

        # Create a blank black image for the stacked image
        stacked_img = np.zeros((h_stacked, w_stacked, 3), dtype=np.uint8)

        # Insert the first image at the top
        stacked_img[:h1, :w1, :] = img1

        # Insert the second image at the bottom
        stacked_img[h_stacked-h2:h_stacked, :w2, :] = img2

        return stacked_img

def save_overlay_images_to_jpg(full_image, filename, Dirs, cfg):
    save_each_segmentation_overlay_image = cfg['leafmachine']['overlay']['save_overlay_to_jpgs']
    if save_each_segmentation_overlay_image:
        try:
            full_image.save(os.path.join(Dirs.custom_overlay_images, '.'.join([filename, 'jpg'])))
        except:
            full_image = cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(Dirs.custom_overlay_images, '.'.join([filename, 'jpg'])) , full_image)
        
def save_custom_overlay_to_PDF(filenames, full_images, ruler_images, batch, Dirs, Project, cfg):
    color_bg = cfg['leafmachine']['overlay']['overlay_background_color']
    overlay_dpi = cfg['leafmachine']['overlay']['overlay_dpi']
    save_overlay_pdf = cfg['leafmachine']['overlay']['save_overlay_to_pdf']
    batch_size = cfg['leafmachine']['project']['batch_size']

    if save_overlay_pdf:
        if color_bg == 'black':
            color_text = 'white'
        else:
            color_text = 'black'
            
        if batch_size is None:
            batch_size = len(filenames)
        for i in range(0, len(filenames), batch_size):
            start = batch*batch_size
            end = batch*batch_size + batch_size
            # if end > len(os.listdir(Project.dir_images)):
            #     end = int(len(os.listdir(Project.dir_images)))

            filenames_batch = list(itertools.islice(filenames, i, i+batch_size))
            full_images_batch = list(itertools.islice(full_images, i, i+batch_size))
            ruler_images_batch = list(itertools.islice(ruler_images, i, i+batch_size))
            if len(filenames_batch) != batch_size:
                end = batch*batch_size + len(filenames)

            pdf_name = os.path.join(Dirs.custom_overlay_pdfs, ''.join(['Custom_Overlay', '_',str(start+1), 'to',str(end),'.pdf']))
            # with PdfPages(pdf_name) as pdf:
            #     for idx, img in enumerate(full_images_batch):
            #         # Create a new figure
            #         fig = plt.figure(dpi=overlay_dpi)
            #         fig.set_size_inches(8.5, 11)
            #         fig.set_facecolor(color_bg)
            #         plt.tight_layout(pad=0)
            #         # Add the image to the figure
            #         plt.imshow(img)
            #         plt.suptitle(filenames[idx], fontsize=10, y=0.95, color=color_text)
            #         # Save the current figure to the PDF
            #         pdf.savefig(fig)
            #         plt.close()
            # Define the size of the left and right columns
            fig_width = 11
            fig_height = 11
            left_column_size = 0.6
            right_column_size = 0.4
            # Create the PDF
            with PdfPages(pdf_name) as pdf:
                for idx, img in enumerate(full_images_batch):
                    # Create a new figure
                    fig = plt.figure(dpi=overlay_dpi)
                    fig.set_size_inches(fig_width, fig_height)
                    fig.set_facecolor(color_bg)
                    
                    # Create the left subplot for the full-sized image
                    left_ax = fig.add_axes([0, 0, left_column_size, 0.95])
                    left_ax.imshow(img)
                    left_ax.set_xticks([])
                    left_ax.set_yticks([])
                    left_ax.set_anchor('NW')  # right-justify the image
                    
                    # Create the right subplot for the ruler image
                    right_ax = fig.add_axes([left_column_size, 0, right_column_size, 0.95])
                    if ruler_images[idx] is not None:
                        right_ax.imshow(cv2.cvtColor(ruler_images[idx], cv2.COLOR_BGR2RGB))
                    right_ax.set_xticks([])
                    right_ax.set_yticks([])
                    right_ax.set_anchor('NW')  # left-justify the image
                    # right_ax.set_title(filenames[idx], fontsize=10, color=color_text, y=0.95)  # move the title to the top
                    # Add the image name to the title
                    plt.suptitle(filenames[idx], fontsize=10, y=0.97, color=color_text)
                    
                    # Save the figure to the PDF
                    pdf.savefig(fig)
                    
                    # Close the figure
                    plt.close()
        
def yolo_to_position(annotation, height, width, anno_type):
    return [set_index_for_annotation(annotation[0], anno_type), 
        int((annotation[1] * width) - ((annotation[3] * width) / 2)), 
        int((annotation[2] * height) - ((annotation[4] * height) / 2)), 
        int(annotation[3] * width) + int((annotation[1] * width) - ((annotation[3] * width) / 2)), 
        int(annotation[4] * height) + int((annotation[2] * height) - ((annotation[4] * height) / 2))]
     
@dataclass
class Colors:
    alpha: int = 127

    seg_leaf_whole: tuple = (46,255,0)
    seg_leaf_partial: tuple = (0,200,255)
    seg_leaf_whole_petiole: tuple = (255, 0, 150) #(0,173,255)
    seg_leaf_partial_petiole: tuple = (90, 0, 75) #(255,140,0)
    seg_hole: tuple = (200,0,255)

    seg_leaf_whole_fill: tuple = (46,255,0, 127)
    seg_leaf_partial_fill: tuple = (0,200,255, 127)
    seg_leaf_whole_petiole_fill: tuple = (255, 0, 150, 127)#(0,173,255, 127)
    seg_leaf_partial_petiole_fill: tuple = (90, 0, 75, 127) #(255,140,0, 127)
    seg_hole_fill: tuple = (200,0,255, 127)

    ruler: tuple = (255, 0, 70)
    barcode: tuple = (0, 137, 65)
    colorcard: tuple = (242, 255, 0)
    label: tuple = (0, 0, 255)
    map: tuple = (0, 251, 255)
    envelope: tuple = (163, 0, 89)
    photo: tuple = (255, 205, 220)
    attached: tuple = (255, 172, 40)
    weights: tuple = (140, 140, 140)

    ruler_fill: tuple = (255, 0, 70, 127)
    barcode_fill: tuple = (0, 137, 65, 127)
    colorcard_fill: tuple = (242, 255, 0, 127)
    label_fill: tuple = (0, 0, 255, 127)
    map_fill: tuple = (0, 251, 255, 127)
    envelope_fill: tuple = (163, 0, 89, 127)
    photo_fill: tuple = (255, 205, 220, 127)
    attached_fill: tuple = (255, 172, 40, 127)
    weights_fill: tuple = (140, 140, 140, 127)

    leaf_whole: tuple = (0, 255, 55)
    leaf_partial: tuple = (0, 255, 250)
    leaflet: tuple = (255, 203, 0)
    seed_fruit_one: tuple = (252, 255, 0)
    seed_fruit_many: tuple = (0, 0, 80)
    flower_one: tuple = (255, 52, 255)
    flower_many: tuple = (154, 0, 255)
    bud: tuple = (255, 0, 9)
    specimen: tuple = (0, 0, 0)
    roots: tuple = (255, 134, 0)
    wood: tuple = (144, 22, 22)

    leaf_whole_fill: tuple = (0, 255, 55, 127)
    leaf_partial_fill: tuple = (0, 255, 250, 127)
    leaflet_fill: tuple = (255, 203, 0, 127)
    seed_fruit_one_fill: tuple = (252, 255, 0, 127)
    seed_fruit_many_fill: tuple = (0, 0, 80, 127)
    flower_one_fill: tuple = (255, 52, 255, 127)
    flower_many_fill: tuple = (154, 0, 255, 127)
    bud_fill: tuple = (255, 0, 9, 127)
    specimen_fill: tuple = (0, 0, 0, 127)
    roots_fill: tuple = (255, 134, 0, 127)
    wood_fill: tuple = (144, 22, 22, 127)

    def __init__(self, alpha):
        alpha = int(np.multiply(alpha, 255))
        self.ruler_fill = (255, 0, 70, alpha)
        self.barcode_fill = (0, 137, 65, alpha)
        self.colorcard_fill = (242, 255, 0, alpha)
        self.label_fill = (0, 0, 255, alpha)
        self.map_fill = (0, 251, 255, alpha)
        self.envelope_fill = (163, 0, 89, alpha)
        self.photo_fill = (255, 205, 220, alpha)
        self.attached_fill = (255, 172, 40, alpha)
        self.weights_fill = (140, 140, 140, alpha)
        self.leaf_whole_fill = (0, 255, 55, alpha)
        self.leaf_partial_fill = (0, 255, 250, alpha)
        self.leaflet_fill = (255, 203, 0, alpha)
        self.seed_fruit_one_fill = (252, 255, 0, alpha)
        self.seed_fruit_many_fill = (0, 0, 80, alpha)
        self.flower_one_fill = (255, 52, 255, alpha)
        self.flower_many_fill = (154, 0, 255, alpha)
        self.bud_fill = (255, 0, 9, alpha)
        self.specimen_fill = (0, 0, 0, alpha)
        self.roots_fill = (255, 134, 0, alpha)
        self.wood_fill = (144, 22, 22, alpha)
        self.seg_leaf_whole_fill = (46,255,0, alpha)
        self.seg_leaf_partial_fill = (0,200,255, alpha)
        self.seg_leaf_whole_petiole_fill = (255, 0, 150, alpha)
        self.seg_leaf_partial_petiole_fill = (90, 0, 75, alpha)
        self.seg_hole_fill = (200,0,255, alpha)

def get_color(anno, a_type, cfg):
    if a_type == 'ARCHIVAL':
        alpha = cfg['leafmachine']['overlay']['alpha_transparency_archival']
    elif a_type == 'PLANT':
        alpha = cfg['leafmachine']['overlay']['alpha_transparency_plant']
    elif a_type == 'SEG_WHOLE':
        alpha = cfg['leafmachine']['overlay']['alpha_transparency_seg_whole_leaf']
    elif a_type == 'SEG_PARTIAL':
        alpha = cfg['leafmachine']['overlay']['alpha_transparency_seg_partial_leaf']

    if alpha is None:
        alpha = 0.5
    Color = Colors(alpha)
    color_dict = {
        "ruler": (Color.ruler, Color.ruler_fill),
        "barcode": (Color.barcode, Color.barcode_fill),
        "colorcard": (Color.colorcard, Color.colorcard_fill),
        "label": (Color.label, Color.label_fill),
        "map": (Color.map, Color.map_fill),
        "envelope": (Color.envelope, Color.envelope_fill),
        "photo": (Color.photo, Color.photo_fill),
        "attached_item": (Color.attached, Color.attached_fill),
        "weights": (Color.weights, Color.weights_fill),
        "leaf_whole": (Color.leaf_whole, Color.leaf_whole_fill),
        "leaf_partial": (Color.leaf_partial, Color.leaf_partial_fill),
        "leaflet": (Color.leaflet, Color.leaflet_fill),
        "seed_fruit_one": (Color.seed_fruit_one, Color.seed_fruit_one_fill),
        "seed_fruit_many": (Color.seed_fruit_many, Color.seed_fruit_many_fill),
        "flower_one": (Color.flower_one, Color.flower_one_fill),
        "flower_many": (Color.flower_many, Color.flower_many_fill),
        "bud": (Color.bud, Color.bud_fill),
        "specimen": (Color.specimen, Color.specimen_fill),
        "roots": (Color.roots, Color.roots_fill),
        "wood": (Color.wood, Color.wood_fill),
        "seg_leaf_whole": (Color.seg_leaf_whole , Color.seg_leaf_whole_fill),
        "seg_leaf_partial": (Color.seg_leaf_partial , Color.seg_leaf_partial_fill),
        "seg_leaf_whole_petiole": (Color.seg_leaf_whole_petiole , Color.seg_leaf_whole_petiole_fill),
        "seg_leaf_partial_petiole": (Color.seg_leaf_partial_petiole , Color.seg_leaf_partial_petiole_fill),
        "seg_hole": (Color.seg_hole , Color.seg_hole_fill)
    }
    return color_dict[anno.lower()][0], color_dict[anno.lower()][1]

def set_index_for_annotation(cls,annoType):
    if annoType == 'PLANT':
        if cls == 0:
            annoInd = 'Leaf_WHOLE'
        elif cls == 1:
            annoInd = 'Leaf_PARTIAL'
        elif cls == 2:
            annoInd = 'Leaflet'
        elif cls == 3:
            annoInd = 'Seed_Fruit_ONE'
        elif cls == 4:
            annoInd = 'Seed_Fruit_MANY'
        elif cls == 5:
            annoInd = 'Flower_ONE'
        elif cls == 6:
            annoInd = 'Flower_MANY'
        elif cls == 7:
            annoInd = 'Bud'
        elif cls == 8:
            annoInd = 'Specimen'
        elif cls == 9:
            annoInd = 'Roots'
        elif cls == 10:
            annoInd = 'Wood'
    elif annoType == 'ARCHIVAL':
        if cls == 0:
            annoInd = 'Ruler'
        elif cls == 1:
            annoInd = 'Barcode'
        elif cls == 2:
            annoInd = 'Colorcard'
        elif cls == 3:
            annoInd = 'Label'
        elif cls == 4:
            annoInd = 'Map'
        elif cls == 5:
            annoInd = 'Envelope'
        elif cls == 6:
            annoInd = 'Photo'
        elif cls == 7:
            annoInd = 'Attached_item'
        elif cls == 8:
            annoInd = 'Weights'
    return annoInd.lower()