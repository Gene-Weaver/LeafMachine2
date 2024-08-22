import os, sqlite3
import glob
from time import perf_counter
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor

class HideComponents:
    def __init__(self, cfg, Dirs, base_dir, database_path):
        self.cfg = cfg
        self.Dirs = Dirs
        self.base_dir = base_dir
        self.database_path = database_path
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
        self.file_counts = {}

    def hex_to_bgr(self, hex_color):
        """Convert hex color to BGR format."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

    def remove(self):
        HIDE = self.cfg['leafmachine']['project']['hide_archival_components']
        HIDE_indices = {index for index, name in self.class_names.items() if name in HIDE}
        COLOR = self.hex_to_bgr(self.cfg['leafmachine']['project']['replacement_color'])  # hex to BGR

        # Create a new database connection in the worker process
        conn = sqlite3.connect(self.database_path)
        cur = conn.cursor()

        cur.execute("SELECT name, width, height FROM images")
        image_data = cur.fetchall()

        for file_name, width, height in image_data:
            # Initialize count for each class for this file
            self.file_counts[file_name] = {name: 0 for name in self.class_names.values()}

            # Try to load the image
            try:
                image_path = glob.glob(os.path.join(self.base_dir, file_name + '.*'))[0]
                full_image = cv2.imread(image_path)
            except:
                raise FileNotFoundError(f"Could not load image for {file_name}")

            # Fetch annotations from the SQL database
            cur.execute("SELECT annotation FROM annotations_archival WHERE file_name = ?", (file_name,))
            annotations = cur.fetchall()

            for annotation in annotations:
                # Split the annotation string to extract the values
                class_index, x_center, y_center, bbox_width, bbox_height = map(float, annotation[0].split(','))

                class_index = int(class_index)
                if class_index in HIDE_indices:
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

        conn.close()  # Close the connection when done

def process_files(cfg, Dirs, base_dir, database_path):
    detector = HideComponents(cfg, Dirs, base_dir, database_path)
    detector.remove()

def censor_archival_components(cfg, time_report, logger, dir_home, ProjectSQL, Dirs):
    t2_start = perf_counter()
    logger.name = f'Censoring Archival Components --- {ProjectSQL.dir_images}'

    # Get the database path from ProjectSQL
    database_path = ProjectSQL.database

    # Parallel processing
    num_workers = os.cpu_count()  # Number of available CPU cores
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_files, cfg, Dirs, ProjectSQL.dir_images, database_path) for _ in range(num_workers)]
        for future in futures:
            future.result()  # Wait for all futures to complete

    t2_stop = perf_counter()
    t_remove = f"[Censoring Archival Components elapsed time] {round(t2_stop - t2_start)} seconds ({round((t2_stop - t2_start)/60)} minutes)"
    logger.info(t_remove)
    time_report['t_remove'] = t_remove
    return time_report