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

def build_custom_overlay_parallel(cfg, logger, dir_home, Project, batch, Dirs):
    start_t = perf_counter()
    logger.name = f'[BATCH {batch+1} Build Custom Overlay]'
    logger.info(f'Creating overlay for batch {batch+1}')

    line_w_archival = cfg['leafmachine']['overlay']['line_width_archival']
    show_archival = cfg['leafmachine']['overlay']['show_archival_detections']
    ignore_archival = cfg['leafmachine']['overlay']['ignore_archival_detections_classes']

    lock = Lock()  # Create a lock object

    if cfg['leafmachine']['project']['num_workers'] is None:
        num_workers = 1
    else:
        num_workers = int(cfg['leafmachine']['project']['num_workers'])

    filenames = []
    overlay_images = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for filename, analysis in Project.project_data.items():
            logger.info(f'Creating overlay for {filename}')
            futures.append(executor.submit(process_file, Project, filename, analysis, line_w_archival, show_archival, ignore_archival, cfg, Dirs, lock))  # Pass the lock object to the process_file function

        logger.info(f'Merging results from {num_workers} workers')
        for future in concurrent.futures.as_completed(futures):
            filename, image_overlay = future.result()
            # save_overlay_images_to_jpg(image_overlay, filename, Dirs, cfg)  # Use the lock object when writing to the file
            filenames.append(filename)
            overlay_images.append(image_overlay)


    logger.info(f'Saving batch {batch+1} overlay images to PDF')
    save_custom_overlay_to_PDF(filenames, overlay_images, batch, Dirs, Project, cfg)
    end_t = perf_counter()
    logger.info(f'Batch {batch+1}: Build Custom Overlay Duration --> {round((end_t - start_t)/60)} minutes')


def process_file(Project, filename, analysis, line_w_archival, show_archival, ignore_archival, cfg, Dirs, lock):

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


    try:
        image = Image.open(os.path.join(Project.dir_images, '.'.join([filename, 'jpg'])))
    except:
        image = Image.open(os.path.join(Project.dir_images, '.'.join([filename, 'jpeg'])))

    image_overlay = image
    if image_overlay.mode != 'RGB':
        image_overlay = image_overlay.convert('RGB')

    with lock:

        image_overlay = add_archival_detections(image_overlay, archival, height, width, line_w_archival, show_archival, ignore_archival, cfg)

        save_overlay_images_to_jpg(image_overlay, filename, Dirs, cfg)

    return filename, image_overlay


def build_custom_overlay(cfg, logger, dir_home, Project, batch, Dirs):
    start_t = perf_counter()
    logger.name = f'[BATCH {batch+1} Build Custom Overlay]'
    logger.info(f'Creating overlay for {batch+1}')

    line_w_archival = cfg['leafmachine']['overlay']['line_width_archival']
    show_archival = cfg['leafmachine']['overlay']['show_archival_detections']
    ignore_archival = cfg['leafmachine']['overlay']['ignore_archival_detections_classes']
   
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

        try:
            image = Image.open(os.path.join(Project.dir_images, '.'.join([filename, 'jpg'])))
        except:
            image = Image.open(os.path.join(Project.dir_images, '.'.join([filename, 'jpeg'])))

        image_overlay = image
        if image_overlay.mode != 'RGB':
            image_overlay = image_overlay.convert('RGB')

        image_overlay = add_archival_detections(image_overlay, archival, height, width, line_w_archival, show_archival, ignore_archival, cfg)

        save_overlay_images_to_jpg(image_overlay, filename, Dirs, cfg)

        filenames.append(filename)
        overlay_images.append(image_overlay)

    # save_custom_overlay_to_PDF(filenames, overlay_images, batch, Dirs, Project, cfg)
    end_t = perf_counter()
    logger.info(f'Batch {batch+1}: Build Custom Overlay Duration --> {round((end_t - start_t)/60)} minutes')
    





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
        
def save_custom_overlay_to_PDF(filenames, full_images, batch, Dirs, Project, cfg):
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