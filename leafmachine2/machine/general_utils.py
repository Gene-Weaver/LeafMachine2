import os, yaml, datetime, argparse, re, cv2, random, shutil
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import concurrent.futures
from time import perf_counter

# https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal

def validate_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_cfg_from_full_path(path_cfg):
    with open(path_cfg, "r") as ymlfile:
        cfg = yaml.full_load(ymlfile)
    return cfg

def load_cfg(pathToCfg):
    try:
        with open(os.path.join(pathToCfg,"LeafMachine2.yaml"), "r") as ymlfile:
            cfg = yaml.full_load(ymlfile)
    except:
        with open(os.path.join(os.path.dirname(os.path.dirname(pathToCfg)),"LeafMachine2.yaml"), "r") as ymlfile:
            cfg = yaml.full_load(ymlfile)
    return cfg

def import_csv(full_path):
    csv_data = pd.read_csv(full_path,sep=',',header=0, low_memory=False, dtype=str)
    return csv_data

def import_tsv(full_path):
    csv_data = pd.read_csv(full_path,sep='\t',header=0, low_memory=False, dtype=str)
    return csv_data

def parse_cfg():
    parser = argparse.ArgumentParser(
            description='Parse inputs to read  config file',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    optional_args = parser._action_groups.pop()
    required_args = parser.add_argument_group('MANDATORY arguments')
    required_args.add_argument('--path-to-cfg',
                                type=str,
                                required=True,
                                help='Path to config file - LeafMachine.yaml. Do not include the file name, just the parent dir.')

    parser._action_groups.append(optional_args)
    args = parser.parse_args()
    return args

def get_datetime():
    day = "_".join([str(datetime.datetime.now().strftime("%Y")),str(datetime.datetime.now().strftime("%m")),str(datetime.datetime.now().strftime("%d"))])
    time = "-".join([str(datetime.datetime.now().strftime("%H")),str(datetime.datetime.now().strftime("%M")),str(datetime.datetime.now().strftime("%S"))])
    new_time = "__".join([day,time])
    return new_time

def save_config_file(cfg, logger, Dirs):
    logger.info("Save config file")
    name_yaml = ''.join([Dirs.run_name,'.yaml'])
    write_yaml(cfg, os.path.join(Dirs.path_config_file, name_yaml))

def write_yaml(cfg, path_cfg):
    with open(path_cfg, 'w') as file:
        yaml.dump(cfg, file)

def split_into_batches(Project, logger, cfg):
    logger.name = 'Creating Batches'
    n_batches, n_images = Project.process_in_batches(cfg)
    m = f'Created {n_batches} Batches to Process {n_images} Images'
    logger.info(m)
    return Project, n_batches, m 

def make_images_in_dir_vertical(dir_images_unprocessed, cfg):
    if cfg['leafmachine']['do']['check_for_corrupt_images_make_vertical']:
        n_rotate = 0
        n_corrupt = 0
        n_total = len(os.listdir(dir_images_unprocessed))
        for image_name_jpg in tqdm(os.listdir(dir_images_unprocessed), desc=f'{bcolors.BOLD}     Checking Image Dimensions{bcolors.ENDC}',colour="cyan",position=0,total = n_total):
            if image_name_jpg.endswith((".jpg",".JPG",".jpeg",".JPEG")):
                try:
                    image = cv2.imread(os.path.join(dir_images_unprocessed, image_name_jpg))
                    h, w, img_c = image.shape
                    image, img_h, img_w, did_rotate = make_image_vertical(image, h, w, do_rotate_180=False)
                    if did_rotate:
                        n_rotate += 1
                    cv2.imwrite(os.path.join(dir_images_unprocessed,image_name_jpg), image)
                except:
                    n_corrupt +=1
                    os.remove(os.path.join(dir_images_unprocessed, image_name_jpg))
            # TODO check that below works as intended 
            elif image_name_jpg.endswith((".tiff",".tif",".png",".PNG",".TIFF",".TIF",".jp2",".JP2",".bmp",".BMP",".dib",".DIB")):
                try:
                    image = cv2.imread(os.path.join(dir_images_unprocessed, image_name_jpg))
                    h, w, img_c = image.shape
                    image, img_h, img_w, did_rotate = make_image_vertical(image, h, w, do_rotate_180=False)
                    if did_rotate:
                        n_rotate += 1
                    image_name_jpg = '.'.join([image_name_jpg.split('.')[0], 'jpg'])
                    cv2.imwrite(os.path.join(dir_images_unprocessed,image_name_jpg), image)
                except:
                    n_corrupt +=1
                    os.remove(os.path.join(dir_images_unprocessed, image_name_jpg))
        m = ''.join(['Number of Images Rotated: ', str(n_rotate)])
        Print_Verbose(cfg, 2, m).bold()
        m2 = ''.join(['Number of Images Corrupted: ', str(n_corrupt)])
        if n_corrupt > 0:
            Print_Verbose(cfg, 2, m2).warning
        else:
            Print_Verbose(cfg, 2, m2).bold

def make_image_vertical(image, h, w, do_rotate_180):
    did_rotate = False
    if do_rotate_180:
        # try:
        image = cv2.rotate(image, cv2.ROTATE_180)
        img_h, img_w, img_c = image.shape
        did_rotate = True
        # print("      Rotated 180")
    else:
        if h < w:
            # try:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            img_h, img_w, img_c = image.shape
            did_rotate = True
            # print("      Rotated 90 CW")
        elif h >= w:
            image = image
            img_h = h
            img_w = w
            # print("      Not Rotated")
    return image, img_h, img_w, did_rotate
    

def make_image_horizontal(image, h, w, do_rotate_180):
    if h > w:
        if do_rotate_180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), w, h, True
    return image, w, h, False

def make_images_in_dir_horizontal(dir_images_unprocessed, cfg):
    # if cfg['leafmachine']['do']['check_for_corrupt_images_make_horizontal']:
    n_rotate = 0
    n_corrupt = 0
    n_total = len(os.listdir(dir_images_unprocessed))
    for image_name_jpg in tqdm(os.listdir(dir_images_unprocessed), desc=f'{bcolors.BOLD}     Checking Image Dimensions{bcolors.ENDC}', colour="cyan", position=0, total=n_total):
        if image_name_jpg.endswith((".jpg",".JPG",".jpeg",".JPEG")):
            try:
                image = cv2.imread(os.path.join(dir_images_unprocessed, image_name_jpg))
                h, w, img_c = image.shape
                image, img_h, img_w, did_rotate = make_image_horizontal(image, h, w, do_rotate_180=False)
                if did_rotate:
                    n_rotate += 1
                cv2.imwrite(os.path.join(dir_images_unprocessed,image_name_jpg), image)
            except:
                n_corrupt +=1
                os.remove(os.path.join(dir_images_unprocessed, image_name_jpg))
        # TODO check that below works as intended 
        elif image_name_jpg.endswith((".tiff",".tif",".png",".PNG",".TIFF",".TIF",".jp2",".JP2",".bmp",".BMP",".dib",".DIB")):
            try:
                image = cv2.imread(os.path.join(dir_images_unprocessed, image_name_jpg))
                h, w, img_c = image.shape
                image, img_h, img_w, did_rotate = make_image_horizontal(image, h, w, do_rotate_180=False)
                if did_rotate:
                    n_rotate += 1
                image_name_jpg = '.'.join([image_name_jpg.split('.')[0], 'jpg'])
                cv2.imwrite(os.path.join(dir_images_unprocessed,image_name_jpg), image)
            except:
                n_corrupt +=1
                os.remove(os.path.join(dir_images_unprocessed, image_name_jpg))
    m = ''.join(['Number of Images Rotated: ', str(n_rotate)])
    print(m)
    # Print_Verbose(cfg, 2, m).bold()
    m2 = ''.join(['Number of Images Corrupted: ', str(n_corrupt)])
    print(m2)


@dataclass
class Print_Verbose_Error():
    cfg: str = ''
    indent_level: int = 0
    message: str = ''
    error: str = ''

    def __init__(self, cfg,indent_level,message,error) -> None:
        self.cfg = cfg
        self.indent_level = indent_level
        self.message = message
        self.error = error

    def print_error_to_console(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['optional_warnings']:
            print(f"{bcolors.FAIL}{white_space}{self.message} ERROR: {self.error}{bcolors.ENDC}")

    def print_warning_to_console(self):
        white_space = " " * 5 * self.indent_level
        if self.cfg['leafmachine']['print']['optional_warnings']:
            print(f"{bcolors.WARNING}{white_space}{self.message} ERROR: {self.error}{bcolors.ENDC}")

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

def print_main_start(message):
    indent_level = 1
    white_space = " " * 5 * indent_level
    end = " " * int(80 - len(message) - len(white_space))
    # end_white_space = " " * end
    blank = " " * 80
    print(f"{bcolors.CBLUEBG2}{blank}{bcolors.ENDC}")
    print(f"{bcolors.CBLUEBG2}{white_space}{message}{end}{bcolors.ENDC}")
    print(f"{bcolors.CBLUEBG2}{blank}{bcolors.ENDC}")

def print_main_success(message):
    indent_level = 1
    white_space = " " * 5 * indent_level
    end = " " * int(80 - len(message) - len(white_space))
    blank = " " * 80
    # end_white_space = " " * end
    print(f"{bcolors.CGREENBG2}{blank}{bcolors.ENDC}")
    print(f"{bcolors.CGREENBG2}{white_space}{message}{end}{bcolors.ENDC}")
    print(f"{bcolors.CGREENBG2}{blank}{bcolors.ENDC}")

def print_main_warn(message):
    indent_level = 1
    white_space = " " * 5 * indent_level
    end = " " * int(80 - len(message) - len(white_space))
    # end_white_space = " " * end
    blank = " " * 80
    print(f"{bcolors.CYELLOWBG2}{blank}{bcolors.ENDC}")
    print(f"{bcolors.CYELLOWBG2}{white_space}{message}{end}{bcolors.ENDC}")
    print(f"{bcolors.CYELLOWBG2}{blank}{bcolors.ENDC}")

def print_main_fail(message):
    indent_level = 1
    white_space = " " * 5 * indent_level
    end = " " * int(80 - len(message) - len(white_space))
    # end_white_space = " " * end
    blank = " " * 80
    print(f"{bcolors.CREDBG2}{blank}{bcolors.ENDC}")
    print(f"{bcolors.CREDBG2}{white_space}{message}{end}{bcolors.ENDC}")
    print(f"{bcolors.CREDBG2}{blank}{bcolors.ENDC}")

def print_main_info(message):
    indent_level = 2
    white_space = " " * 5 * indent_level
    end = " " * int(80 - len(message) - len(white_space))
    # end_white_space = " " * end
    print(f"{bcolors.CGREYBG}{white_space}{message}{end}{bcolors.ENDC}")
    
def report_config(dir_home, cfg_file_path):
    print_main_start("Loading Configuration File")
    if cfg_file_path == None:
        print_main_info(''.join([os.path.join(dir_home, 'LeafMachine2.yaml')]))
    elif cfg_file_path == 'test_installation':
        print_main_info(''.join([os.path.join(dir_home, 'demo','LeafMachine2_demo.yaml')]))
    else:
        print_main_info(cfg_file_path)

def make_file_names_valid(dir, cfg):
    if cfg['leafmachine']['do']['check_for_illegal_filenames']:
        n_total = len(os.listdir(dir))
        for file in tqdm(os.listdir(dir), desc=f'{bcolors.HEADER}     Removing illegal characters from file names{bcolors.ENDC}',colour="cyan",position=0,total = n_total):
            name = Path(file).stem
            ext = Path(file).suffix
            name_cleaned = re.sub(r"[^a-zA-Z0-9_-]","-",name)
            name_new = ''.join([name_cleaned,ext])
            i = 0
            try:
                os.rename(os.path.join(dir,file), os.path.join(dir,name_new))
            except:
                while os.path.exists(os.path.join(dir,name_new)):
                    i += 1
                    name_new = '_'.join([name_cleaned, str(i), ext])
                os.rename(os.path.join(dir,file), os.path.join(dir,name_new))

def load_config_file(dir_home, cfg_file_path):
    if cfg_file_path == None: # Default path
        return load_cfg(dir_home)
    else:
        if cfg_file_path == 'test_installation':
            path_cfg = os.path.join(dir_home,'demo','LeafMachine2_demo.yaml')                     # TODO make the demo yaml
            return get_cfg_from_full_path(path_cfg)
        else: # Custom path
            return get_cfg_from_full_path(cfg_file_path)

def subset_dir_images(cfg, Project, Dirs):
    if cfg['leafmachine']['project']['process_subset_of_images']:
        dir_images_subset = cfg['leafmachine']['project']['dir_images_subset']
        num_images_per_species = cfg['leafmachine']['project']['n_images_per_species']
        if cfg['leafmachine']['project']['species_list'] is not None:
            species_list = import_csv(cfg['leafmachine']['project']['species_list'])
            species_list = species_list.iloc[:, 0].tolist()
        else:
            species_list = None

        validate_dir(dir_images_subset)

        species_counts = {}
        filenames = os.listdir(Project.dir_images)
        random.shuffle(filenames)
        for filename in filenames:
            species_name = filename.split('.')[0]
            species_name = species_name.split('_')[2:]
            species_name = '_'.join([species_name[0], species_name[1], species_name[2]])

            if (species_list is None) or ((species_name in species_list) and (species_list is not None)):
            
                if species_name not in species_counts:
                    species_counts[species_name] = 0
                
                if species_counts[species_name] < num_images_per_species:
                    species_counts[species_name] += 1
                    src_path = os.path.join(Project.dir_images, filename)
                    dest_path = os.path.join(dir_images_subset, filename)
                    shutil.copy(src_path, dest_path)
        
        Project.dir_images = dir_images_subset
        
        subset_csv_name = os.path.join(Dirs.dir_images_subset, '.'.join([Dirs.run_name, 'csv']))
        df = pd.DataFrame({'species_name': list(species_counts.keys()), 'count': list(species_counts.values())})
        df.to_csv(subset_csv_name, index=False)
        return Project
    else:
        return Project

'''# Define function to be executed by each worker
def worker_crop(rank, cfg, dir_home, Project, Dirs):
    # Set worker seed based on rank
    np.random.seed(rank)
    # Call function for this worker
    crop_detections_from_images(cfg, dir_home, Project, Dirs)

def crop_detections_from_images(cfg, dir_home, Project, Dirs):
    num_workers = 6
    
    # Initialize and start worker processes
    processes = []
    for rank in range(num_workers):
        p = mp.Process(target=worker_crop, args=(rank, cfg, dir_home, Project, Dirs))
        p.start()
        processes.append(p)

    # Wait for all worker processes to finish
    for p in processes:
        p.join()'''

def crop_detections_from_images_worker(filename, analysis, Project, Dirs, save_per_image, save_per_class, save_list, binarize_labels):
    try:
        full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpg'])))
    except:
        full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpeg'])))

    try:
        archival = analysis['Detections_Archival_Components']
        has_archival = True
    except: 
        has_archival = False

    try:
        plant = analysis['Detections_Plant_Components']
        has_plant = True
    except: 
        has_plant = False

    if has_archival and (save_per_image or save_per_class):
        crop_component_from_yolo_coords('ARCHIVAL', Dirs, analysis, archival, full_image, filename, save_per_image, save_per_class, save_list)
    if has_plant and (save_per_image or save_per_class):
        crop_component_from_yolo_coords('PLANT', Dirs, analysis, plant, full_image, filename, save_per_image, save_per_class, save_list)


def crop_detections_from_images(cfg, logger, dir_home, Project, Dirs, batch_size=50):
    t2_start = perf_counter()
    logger.name = 'Crop Components'
    
    if cfg['leafmachine']['cropped_components']['do_save_cropped_annotations']:
        detections = cfg['leafmachine']['cropped_components']['save_cropped_annotations']
        logger.info(f"Cropping {detections} components from images")

        save_per_image = cfg['leafmachine']['cropped_components']['save_per_image']
        save_per_class = cfg['leafmachine']['cropped_components']['save_per_annotation_class']
        save_list = cfg['leafmachine']['cropped_components']['save_cropped_annotations']
        binarize_labels = cfg['leafmachine']['cropped_components']['binarize_labels']
        if cfg['leafmachine']['project']['batch_size'] is None:
            batch_size = 50
        else:
            batch_size = int(cfg['leafmachine']['project']['batch_size'])
        if cfg['leafmachine']['project']['num_workers'] is None:
            num_workers = 4 
        else:
            num_workers = int(cfg['leafmachine']['project']['num_workers'])

        if binarize_labels:
            save_per_class = True

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(0, len(Project.project_data), batch_size):
                batch = list(Project.project_data.items())[i:i+batch_size]
                # print(f'Cropping Detections from Images {i} to {i+batch_size}')
                logger.info(f'Cropping {detections} from images {i} to {i+batch_size} [{len(Project.project_data)}]')
                for filename, analysis in batch:
                    if len(analysis) != 0:
                        futures.append(executor.submit(crop_detections_from_images_worker, filename, analysis, Project, Dirs, save_per_image, save_per_class, save_list, binarize_labels))

                for future in concurrent.futures.as_completed(futures):
                    pass
                futures.clear()

    t2_stop = perf_counter()
    logger.info(f"Save cropped components --- elapsed time: {round(t2_stop - t2_start)} seconds")
'''
# Single threaded
def crop_detections_from_images(cfg, dir_home, Project, Dirs):
    if cfg['leafmachine']['cropped_components']['do_save_cropped_annotations']:
        save_per_image = cfg['leafmachine']['cropped_components']['save_per_image']
        save_per_class = cfg['leafmachine']['cropped_components']['save_per_annotation_class']
        save_list = cfg['leafmachine']['cropped_components']['save_cropped_annotations']
        binarize_labels = cfg['leafmachine']['cropped_components']['binarize_labels']
        if binarize_labels:
            save_per_class = True

        for filename, analysis in  tqdm(Project.project_data.items(), desc=f'{bcolors.BOLD}     Cropping Detections from Images{bcolors.ENDC}',colour="cyan",position=0,total = len(Project.project_data.items())):
            if len(analysis) != 0:
                try:
                    full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpg'])))
                except:
                    full_image = cv2.imread(os.path.join(Project.dir_images, '.'.join([filename, 'jpeg'])))

                try:
                    archival = analysis['Detections_Archival_Components']
                    has_archival = True
                except: 
                    has_archival = False

                try:
                    plant = analysis['Detections_Plant_Components']
                    has_plant = True
                except: 
                    has_plant = False

                if has_archival and (save_per_image or save_per_class):
                    crop_component_from_yolo_coords('ARCHIVAL', Dirs, analysis, archival, full_image, filename, save_per_image, save_per_class, save_list)
                if has_plant and (save_per_image or save_per_class):
                    crop_component_from_yolo_coords('PLANT', Dirs, analysis, plant, full_image, filename, save_per_image, save_per_class, save_list)
'''

def crop_component_from_yolo_coords(anno_type, Dirs, analysis, all_detections, full_image, filename, save_per_image, save_per_class, save_list):
    height = analysis['height']
    width = analysis['width']
    if len(all_detections) < 1:
        print('     MAKE THIS HAVE AN EMPTY PLACEHOLDER') # TODO ###################################################################################
    else:
        for detection in all_detections:
            detection_class = detection[0]
            detection_class = set_index_for_annotation(detection_class, anno_type)

            if (detection_class in save_list) or ('save_all' in save_list):

                location = yolo_to_position_ruler(detection, height, width)
                ruler_polygon = [(location[1], location[2]), (location[3], location[2]), (location[3], location[4]), (location[1], location[4])]

                x_coords = [x for x, y in ruler_polygon]
                y_coords = [y for x, y in ruler_polygon]

                min_x, min_y = min(x_coords), min(y_coords)
                max_x, max_y = max(x_coords), max(y_coords)

                detection_cropped = full_image[min_y:max_y, min_x:max_x]
                loc = '-'.join([str(min_x), str(min_y), str(max_x), str(max_y)])
                detection_cropped_name = '.'.join(['__'.join([filename, detection_class, loc]), 'jpg'])

                # save_per_image
                if (detection_class in save_list) and save_per_image:
                    dir_destination = os.path.join(Dirs.save_per_image, filename, detection_class)
                    # print(os.path.join(dir_destination,detection_cropped_name))
                    validate_dir(dir_destination)
                    cv2.imwrite(os.path.join(dir_destination,detection_cropped_name), detection_cropped)
                    
                # save_per_class
                if (detection_class in save_list) and save_per_class:
                    dir_destination = os.path.join(Dirs.save_per_annotation_class, detection_class)
                    # print(os.path.join(dir_destination,detection_cropped_name))
                    validate_dir(dir_destination)
                    cv2.imwrite(os.path.join(dir_destination,detection_cropped_name), detection_cropped)
            else:
                # print(f'detection_class: {detection_class} not in save_list: {save_list}')
                pass

def yolo_to_position_ruler(annotation, height, width):
    return ['ruler', 
        int((annotation[1] * width) - ((annotation[3] * width) / 2)), 
        int((annotation[2] * height) - ((annotation[4] * height) / 2)), 
        int(annotation[3] * width) + int((annotation[1] * width) - ((annotation[3] * width) / 2)), 
        int(annotation[4] * height) + int((annotation[2] * height) - ((annotation[4] * height) / 2))]


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
# def set_yaml(path_to_yaml, value):
#     with open('file_to_edit.yaml') as f:
#         doc = yaml.load(f)

#     doc['state'] = state

#     with open('file_to_edit.yaml', 'w') as f:
#         yaml.dump(doc, f)