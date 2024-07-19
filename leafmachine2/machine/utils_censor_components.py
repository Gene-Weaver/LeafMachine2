import os
import glob
from time import perf_counter
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor

class HideComponents:
    def __init__(self, cfg, Dirs, Project, base_dir, label_files):
        self.cfg = cfg
        self.Dirs = Dirs
        self.Project = Project
        self.base_dir = base_dir
        self.label_files = label_files
        self.class_names = {
            0: 'ruler',
            1: 'barcode',
            2: 'colorcard',
            3: 'label',
            4: 'map',
            5: 'envelope',
            6: 'photo',
            7: 'attached_item',
            8: 'weights',
        }
        # Initialize counts as a dict of dicts
        self.file_counts = {}
    
    def hex_to_bgr(self, hex_color):
        """Convert hex color to BGR format."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

    def remove(self):
        HIDE = self.cfg['leafmachine']['project']['hide_archival_components']
        HIDE_indices = {index for index, name in self.class_names.items() if name in HIDE}
        COLOR = self.hex_to_bgr(self.cfg['leafmachine']['project']['replacement_color'])  # hex to BGR

        for file_path in self.label_files:
            file_name = os.path.basename(file_path).replace('.txt', '')
            # Initialize count for each class for this file
            self.file_counts[file_name] = {name: 0 for name in self.class_names.values()}

            # Try to load the image
            try:
                full_image = cv2.imread(os.path.join(self.Project.dir_images, f"{file_name}.jpg"))
            except:
                full_image = cv2.imread(os.path.join(self.Project.dir_images, f"{file_name}.jpeg"))
            
            if full_image is None:
                continue

            height, width, _ = full_image.shape

            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.split()
                    class_index = int(parts[0])
                    if class_index in HIDE_indices:
                        x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])
                        x_center *= width
                        y_center *= height
                        bbox_width *= width
                        bbox_height *= height

                        x_min = int(x_center - bbox_width / 2)
                        y_min = int(y_center - bbox_height / 2)
                        x_max = int(x_center + bbox_width / 2)
                        y_max = int(y_center + bbox_height / 2)

                        # Replace with the selected color
                        full_image[y_min:y_max, x_min:x_max] = COLOR

                    class_name = self.class_names.get(class_index)
                    if class_name:
                        self.file_counts[file_name][class_name] += 1

            save_path = os.path.join(self.Dirs.censor_archival_components, f"{file_name}.jpg")
            cv2.imwrite(save_path, full_image)

def process_files(cfg, Dirs, Project, base_dir, label_files):
    detector = HideComponents(cfg, Dirs, Project, base_dir, label_files)
    detector.remove()

def censor_archival_components(cfg, time_report, logger, dir_home, Project, Dirs):
    t2_start = perf_counter()
    logger.name = f'Removing Archival Components --- {Dirs.path_archival_components}'
    
    path_archival_labels = os.path.join(Dirs.path_archival_components, 'labels')
    label_files = glob.glob(os.path.join(path_archival_labels, '*.txt'))

    # Split the list of label files into chunks for each worker
    num_workers = os.cpu_count()  # Number of available CPU cores
    chunks = [label_files[i::num_workers] for i in range(num_workers)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_files, cfg, Dirs, Project, path_archival_labels, chunk) for chunk in chunks]
        for future in futures:
            future.result()  # Wait for all futures to complete

    # return counts
    t2_stop = perf_counter()
    t_remove = f"[Removing Archival Components elapsed time] {round(t2_stop - t2_start)} seconds ({round((t2_stop - t2_start)/60)} minutes)"
    logger.info(t_remove)
    time_report['t_remove'] = t_remove
    return time_report
