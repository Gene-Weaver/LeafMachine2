import os, time, sys, inspect, shutil, random, sqlite3
import pandas as pd
from sqlite3 import Error
from PIL import Image, ImageFile
from dataclasses import dataclass, field
from tqdm import tqdm
import concurrent.futures


currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)
from leafmachine2.machine.general_utils import import_csv, import_tsv
from leafmachine2.machine.general_utils import Print_Verbose, print_main_warn, print_main_success, make_file_names_valid, make_images_in_dir_vertical
from leafmachine2.machine.utils_GBIF import generate_image_filename
from leafmachine2.downloading.download_from_GBIF_all_images_in_file import download_all_images_from_GBIF_LM2

ImageFile.LOAD_TRUNCATED_IMAGES = True

@dataclass
class Project_Info_SQL():
    batch_size: int = 50
    image_location: str = ''
    dir_images: str = ''
    path_csv_combined: str = ''
    path_csv_occ: str = ''
    path_csv_img: str = ''
    csv_combined: str = ''
    csv_occ: str = ''
    csv_img: str = ''
    database: str = ''
    conn: object = field(init=False)

    def __init__(self, cfg, logger, dir_home, Dirs) -> None:
        logger.name = 'Project Info'
        logger.info("Gathering Images and Image Metadata")

        self.batch_size = cfg['leafmachine']['project']['batch_size']
        self.image_location = cfg['leafmachine']['project']['image_location']
        self.Dirs = Dirs
        self.database = Dirs.database
        

        self.valid_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF']

        if self.image_location in ['local','l','L','Local']:
            print(f"Directory for images: {self.dir_images}")  # Add this line to log the directory being used
            self.__import_local_files(cfg, logger)
        elif self.image_location in ['GBIF','g','G','gbif']:
            self.__import_GBIF_files_post_download(cfg, logger, dir_home)

        make_file_names_valid(self.dir_images, cfg)
        make_images_in_dir_vertical(self.dir_images, cfg)

        self.create_tables_if_not_exist()
        self.__make_project_dict()
        self.populate_image_dimensions()  # Populate dimensions after creating the tables

    def create_tables_if_not_exist(self, retries=5, delay=1):
        
        # for attempt in range(retries):
            # try:
        self.conn = self.create_connection()
        time.sleep(0.1)
        self.conn.execute('PRAGMA journal_mode=WAL;')
        
        # Ensure the connection is valid before proceeding
        if self.conn is None:
            print("Database connection is not established.")
            return
    
        cur = self.conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id integer PRIMARY KEY,
                name text NOT NULL,
                path text NOT NULL,
                valid integer,
                width integer,
                height integer
            )
        """)
        # Assert images table was created
        assert self.table_exists('images'), "Table 'images' was not created."
    
        cur.execute("""
            CREATE TABLE IF NOT EXISTS annotations_archival (
                id INTEGER PRIMARY KEY,
                file_name TEXT NOT NULL,
                component TEXT NOT NULL,
                annotation TEXT NOT NULL
            )
        """)
        assert self.table_exists('annotations_archival'), "Table 'annotations_archival' was not created."

        cur.execute("""
            CREATE TABLE IF NOT EXISTS annotations_plant (
                id INTEGER PRIMARY KEY,
                file_name TEXT NOT NULL,
                component TEXT NOT NULL,
                annotation TEXT NOT NULL
            )
        """)
        assert self.table_exists('annotations_plant'), "Table 'annotations_plant' was not created."

        cur.execute("""
            CREATE TABLE IF NOT EXISTS dimensions_archival (
                id INTEGER PRIMARY KEY,
                file_name TEXT NOT NULL,
                width INTEGER,
                height INTEGER
            )
        """)
        assert self.table_exists('dimensions_archival'), "Table 'dimensions_archival' was not created."

        cur.execute("""
            CREATE TABLE IF NOT EXISTS dimensions_plant (
                id INTEGER PRIMARY KEY,
                file_name TEXT NOT NULL,
                width INTEGER,
                height INTEGER
            )
        """)
        assert self.table_exists('dimensions_plant'), "Table 'dimensions_plant' was not created."


        cur.execute("""
            CREATE TABLE IF NOT EXISTS ruler_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                ruler_image_name TEXT,
                success BOOLEAN,
                conversion_mean REAL,
                predicted_conversion_factor_cm REAL,
                pooled_sd REAL,
                ruler_class TEXT,
                ruler_class_confidence REAL,
                units TEXT,
                cross_validation_count INTEGER,
                n_scanlines INTEGER,
                n_data_points_in_avg INTEGER,
                avg_tick_width REAL,
                plot_points BLOB
            )
        """)
        # summary_img BLOB
        assert self.table_exists('ruler_data'), "Table 'ruler_data' was not created."

        # Table for Whole_Leaf_BBoxes
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Whole_Leaf_BBoxes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                class INTEGER,
                x_min INTEGER,
                y_min INTEGER,
                x_max INTEGER,
                y_max INTEGER
            )
        """)
        assert self.table_exists('Whole_Leaf_BBoxes'), "Table 'Whole_Leaf_BBoxes' was not created."

        # Table for Partial_Leaf_BBoxes
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Partial_Leaf_BBoxes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                class INTEGER,
                x_min INTEGER,
                y_min INTEGER,
                x_max INTEGER,
                y_max INTEGER
            )
        """)
        assert self.table_exists('Partial_Leaf_BBoxes'), "Table 'Partial_Leaf_BBoxes' was not created."

        # Create the 'Whole_Leaf_Cropped' and 'Partial_Leaf_Cropped' tables
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Whole_Leaf_Cropped (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                crop_name TEXT NOT NULL,
                cropped_image BLOB NOT NULL
            )
        """)
        assert self.table_exists('Whole_Leaf_Cropped'), "Table 'Whole_Leaf_Cropped' was not created."
    
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Partial_Leaf_Cropped (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                crop_name TEXT NOT NULL,
                cropped_image BLOB NOT NULL
            )
        """)
        assert self.table_exists('Partial_Leaf_Cropped'), "Table 'Partial_Leaf_Cropped' was not created."

        # Create the 'Segmentation_Whole_Leaf' and 'Segmentation_Partial_Leaf' tables
        # overlay_data TEXT NOT NULL,    went after colors
        # 
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Segmentation_Whole_Leaf (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                crop_name TEXT NOT NULL,
                object_name TEXT,
                overlay_data TEXT,
                polygons TEXT NOT NULL,
                bboxes TEXT NOT NULL,
                labels TEXT NOT NULL,
                colors TEXT NOT NULL,
                bbox TEXT,
                bbox_min TEXT,
                rotate_angle REAL,
                bbox_min_long_side REAL,
                bbox_min_short_side REAL,
                efd_coeffs_features TEXT,
                efd_a0 TEXT,
                efd_c0 TEXT,
                efd_scale TEXT,
                efd_angle REAL,
                efd_phase TEXT,
                efd_area REAL,
                efd_perimeter REAL,
                efd_overlay TEXT,
                area REAL,
                perimeter REAL, 
                centroid TEXT,
                convex_hull REAL,
                convexity REAL,
                concavity REAL,
                circularity REAL,
                n_pts_in_polygon REAL,
                aspect_ratio REAL,
                polygon_closed TEXT,
                polygon_closed_rotated TEXT
            )
        """)
        assert self.table_exists('Segmentation_Whole_Leaf'), "Table 'Segmentation_Whole_Leaf' was not created."

        cur.execute("""
            CREATE TABLE IF NOT EXISTS Segmentation_Partial_Leaf (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                crop_name TEXT NOT NULL,
                object_name TEXT,
                overlay_data TEXT,
                polygons TEXT NOT NULL,
                bboxes TEXT NOT NULL,
                labels TEXT NOT NULL,
                colors TEXT NOT NULL,
                bbox TEXT,
                bbox_min TEXT,
                rotate_angle REAL,
                bbox_min_long_side REAL,
                bbox_min_short_side REAL,
                efd_coeffs_features TEXT,
                efd_a0 TEXT,
                efd_c0 TEXT,
                efd_scale TEXT,
                efd_angle REAL,
                efd_phase TEXT,
                efd_area REAL,
                efd_perimeter REAL,
                efd_overlay TEXT,
                area REAL,
                perimeter REAL, 
                centroid TEXT,
                convex_hull REAL,
                convexity REAL,
                concavity REAL,
                circularity REAL,
                n_pts_in_polygon REAL,
                aspect_ratio REAL,
                polygon_closed TEXT,
                polygon_closed_rotated TEXT
            )
        """)
        assert self.table_exists('Segmentation_Partial_Leaf'), "Table 'Segmentation_Partial_Leaf' was not created."

        # Create tables for storing landmarks data
        cur.execute("""
            CREATE TABLE IF NOT EXISTS Landmarks_Whole_Leaves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                crop_name TEXT NOT NULL,
                all_points TEXT NOT NULL,
                height INTEGER,
                width INTEGER,
                apex_center TEXT,
                apex_left TEXT,
                apex_right TEXT,
                apex_angle_degrees REAL,
                apex_angle_type TEXT,
                base_center TEXT,
                base_left TEXT,
                base_right TEXT,
                base_angle_degrees REAL,
                base_angle_type TEXT,
                lamina_tip TEXT,
                lamina_base TEXT,
                lamina_length REAL,
                lamina_fit TEXT,
                lamina_width REAL,
                width_left TEXT,
                width_right TEXT,
                lobe_count INTEGER,
                lobes TEXT,
                midvein_fit TEXT,
                midvein_fit_points TEXT,
                ordered_midvein TEXT,
                ordered_midvein_length REAL,
                has_midvein INTEGER,
                ordered_petiole TEXT,
                ordered_petiole_length REAL,
                has_ordered_petiole INTEGER,
                is_split INTEGER,
                has_apex INTEGER,
                has_base INTEGER,
                has_lamina_tip INTEGER,
                has_lamina_base INTEGER,
                has_lamina_length INTEGER,
                has_width INTEGER,
                has_lobes INTEGER,
                is_complete_leaf INTEGER,
                is_leaf_no_width INTEGER,
                t_base_center TEXT, 
                t_base_left TEXT, 
                t_base_right TEXT, 
                t_apex_center TEXT, 
                t_apex_left TEXT, 
                t_apex_right TEXT, 
                t_lamina_base TEXT, 
                t_lamina_tip TEXT,
                t_lobes TEXT, 
                t_midvein TEXT, 
                t_midvein_fit_points TEXT, 
                t_petiole TEXT, 
                t_width_left TEXT, 
                t_width_right TEXT, 
                t_width_infer TEXT
            )
        """)
        assert self.table_exists('Landmarks_Whole_Leaves'), "Table 'Landmarks_Whole_Leaves' was not created."

        cur.execute("""
            CREATE TABLE IF NOT EXISTS Landmarks_Partial_Leaves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                crop_name TEXT NOT NULL,
                all_points TEXT NOT NULL,
                height INTEGER,
                width INTEGER,
                apex_center TEXT,
                apex_left TEXT,
                apex_right TEXT,
                apex_angle_degrees REAL,
                apex_angle_type TEXT,
                base_center TEXT,
                base_left TEXT,
                base_right TEXT,
                base_angle_degrees REAL,
                base_angle_type TEXT,
                lamina_tip TEXT,
                lamina_base TEXT,
                lamina_length REAL,
                lamina_fit TEXT,
                lamina_width REAL,
                width_left TEXT,
                width_right TEXT,
                lobe_count INTEGER,
                lobes TEXT,
                midvein_fit TEXT,
                midvein_fit_points TEXT,
                ordered_midvein TEXT,
                ordered_midvein_length REAL,
                has_midvein INTEGER,
                ordered_petiole TEXT,
                ordered_petiole_length REAL,
                has_ordered_petiole INTEGER,
                is_split INTEGER,
                has_apex INTEGER,
                has_base INTEGER,
                has_lamina_tip INTEGER,
                has_lamina_base INTEGER,
                has_lamina_length INTEGER,
                has_width INTEGER,
                has_lobes INTEGER,
                is_complete_leaf INTEGER,
                is_leaf_no_width INTEGER,
                t_base_center TEXT, 
                t_base_left TEXT, 
                t_base_right TEXT, 
                t_apex_center TEXT, 
                t_apex_left TEXT, 
                t_apex_right TEXT, 
                t_lamina_base TEXT, 
                t_lamina_tip TEXT,
                t_lobes TEXT, 
                t_midvein TEXT, 
                t_midvein_fit_points TEXT, 
                t_petiole TEXT, 
                t_width_left TEXT, 
                t_width_right TEXT, 
                t_width_infer TEXT
            )
        """)
        assert self.table_exists('Landmarks_Partial_Leaves'), "Table 'Landmarks_Partial_Leaves' was not created."

        cur.execute("""
            CREATE TABLE IF NOT EXISTS Keypoints_Data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT,
                crop_name TEXT,
                keypoints TEXT,
                angle REAL,
                tip TEXT,
                base TEXT,
                distance_lamina REAL, 
                distance_width REAL, 
                distance_petiole REAL,
                distance_midvein_span REAL, 
                distance_petiole_span REAL,
                trace_midvein_distance REAL, 
                trace_petiole_distance REAL, 
                apex_angle REAL, 
                apex_is_reflex INTEGER, 
                base_angle REAL, 
                base_is_reflex INTEGER
            )
        """)
        assert self.table_exists('Keypoints_Data'), "Table 'Keypoints_Data' was not created."

        # Commit changes and close the cursor
        self.conn.commit()
            # except sqlite3.OperationalError as e:
            #     if "locked" in str(e):
            #         print(f"Database is locked, retrying in {delay} seconds...")
            #         time.sleep(delay)
            #     else:
            #         raise
            # except sqlite3.Error as e:
            #     print(f"Error creating table 'images': {e}")
            #     self.conn.rollback()  # Rollback the transaction in case of error
            #     # try:
            #     #     self.conn.commit()
            #     #     cur.close()
            #     # except:
            #     #     pass
            #     # print(f"An error occurred while creating tables: {e}")
            # finally:
            #     if self.conn:
            #         self.conn.close()
            #         try:
            #             cur.close()  # Close the cursor in the finally block to ensure it's closed no matter what
            #         except:
            #             pass

    def table_exists(self, table_name):
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        return cur.fetchone() is not None

    # def populate_image_dimensions(self):
    #     cur = self.conn.cursor()

    #     # Fetch all images that don't have width and height set
    #     cur.execute("SELECT id, path FROM images WHERE width IS NULL OR height IS NULL")
    #     images = cur.fetchall()

    #     for img_id, img_path in images:
    #         try:
    #             with Image.open(img_path) as img:
    #                 width, height = img.size
    #                 cur.execute("UPDATE images SET width = ?, height = ? WHERE id = ?", (width, height, img_id))
    #                 self.conn.commit()
    #         except Exception as e:
    #             print(f"Error processing image {img_path}: {e}")
    
    def populate_image_dimensions(self, batch_size=100):
        cur = self.conn.cursor()

        # Fetch all images that don't have width and height set
        cur.execute("SELECT id, path FROM images WHERE width IS NULL OR height IS NULL")
        images = cur.fetchall()
        cur.close()

        # Split images into batches for parallel processing
        image_batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]

        db_path = self.database
        
        # Use ProcessPoolExecutor for parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(process_image_batch, batch, db_path)
                for batch in image_batches
            ]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Populating Image Dimensions", colour='green'):
                future.result()  # Catch exceptions from workers


    def create_connection(self, retries=5, delay=0.1):
        conn = None
        self.database = os.path.abspath(self.database)  # Ensure the database path is absolute
        
        if not os.path.exists(self.database):
            print(f"Creating new project database: {self.database}")
            print(f"Populating database with image names and dimensions")
        
        for attempt in range(retries):
            try:
                # Use mode=rwc to allow reading, writing, and creation of the file if it doesn't exist
                conn = sqlite3.connect(f'file:{self.database}?mode=rwc', uri=True, timeout=10)
                return conn
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e):
                    print(f"Database is locked, retrying in {delay} seconds... (Attempt {attempt+1}/{retries})")
                    time.sleep(delay)
                else:
                    print(f"An error occurred: {e}")
                    break
        return conn

    def create_tables(self):
        try:
            sql_create_images_table = """ CREATE TABLE IF NOT EXISTS images (
                                            id integer PRIMARY KEY,
                                            name text NOT NULL,
                                            path text NOT NULL,
                                            valid integer,
                                            width integer,
                                            height integer
                                        ); """
            cur = self.conn.cursor()
            cur.execute(sql_create_images_table)
            self.conn.commit()
        except Error as e:
            print(e)


    def create_archival_components_table(self):
        try:
            sql_create_archival_components_table = """ CREATE TABLE IF NOT EXISTS archival_components (
                                                        id integer PRIMARY KEY,
                                                        image_name text NOT NULL,
                                                        component text NOT NULL,
                                                        annotations text NOT NULL,
                                                        FOREIGN KEY(image_name) REFERENCES images(name)
                                                    ); """
            cur = self.conn.cursor()
            cur.execute(sql_create_archival_components_table)
            self.conn.commit()
        except Error as e:
            print(e)

    def create_plant_components_table(self):
        try:
            sql_create_plant_components_table = """ CREATE TABLE IF NOT EXISTS plant_components (
                                                        id integer PRIMARY KEY,
                                                        image_name text NOT NULL,
                                                        component text NOT NULL,
                                                        annotations text NOT NULL,
                                                        FOREIGN KEY(image_name) REFERENCES images(name)
                                                    ); """
            cur = self.conn.cursor()
            cur.execute(sql_create_plant_components_table)
            self.conn.commit()
        except Error as e:
            print(e)

    def insert_image(self, name, path, valid):
        sql = ''' INSERT INTO images(name,path,valid)
                  VALUES(?,?,?) '''
        try:
            cur = self.conn.cursor()
            cur.execute(sql, (name, path, valid))
            self.conn.commit()
            # print(f"Inserted image into database: {name}, {path}, {valid}")  # Log each database insertion
            return cur.lastrowid
        except Error as e:
            print(f"Error inserting image {name} into database: {e}")

    @property
    def has_valid_images(self):
        return self.check_for_images()

    @property
    def file_ext(self):
        return f"{['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF']}"

    def check_for_images(self):
        for filename in os.listdir(self.dir_images):
            if filename.endswith(tuple(self.valid_extensions)):
                return True
        return False

    def __create_combined_csv(self):
        self.csv_img = self.csv_img.rename(columns={"gbifID": "gbifID_images"})
        self.csv_img = self.csv_img.rename(columns={"identifier": "url"})
        combined = pd.merge(self.csv_img, self.csv_occ, left_on='gbifID_images', right_on='gbifID')
        names_list = combined.apply(generate_image_filename, axis=1, result_type='expand')
        selected_columns = names_list.iloc[:, [7, 0, 1]]
        selected_columns.columns = ['fullname', 'filename_image', 'filename_image_jpg']
        self.csv_combined = pd.concat([selected_columns, combined], axis=1)
        new_name = ''.join(['combined_', os.path.basename(self.path_csv_occ).split('.')[0], '_', os.path.basename(self.path_csv_img).split('.')[0], '.csv'])
        self.path_csv_combined = os.path.join(os.path.dirname(self.path_csv_occ), new_name)
        self.csv_combined.to_csv(self.path_csv_combined, mode='w', header=True, index=False)
        return self.path_csv_combined

    def __import_local_files(self, cfg, logger):
        if cfg['leafmachine']['project']['dir_images_local'] is None:
            self.dir_images = None
        else:
            self.dir_images = cfg['leafmachine']['project']['dir_images_local']
            print(f"Local images directory: {self.dir_images}")
        try:
            if cfg['leafmachine']['project']['path_combined_csv_local'] is None:
                self.csv_combined = None
                self.path_csv_combined = None
            else:
                self.path_csv_combined = cfg['leafmachine']['project']['path_combined_csv_local']
                self.csv_combined = import_csv(self.path_csv_combined)
            if cfg['leafmachine']['project']['path_occurrence_csv_local'] is None:
                self.csv_occ = None
                self.path_csv_occ = None
            else:
                self.path_csv_occ = cfg['leafmachine']['project']['path_occurrence_csv_local']
                self.csv_occ = import_csv(self.path_csv_occ)
            if cfg['leafmachine']['project']['path_images_csv_local'] is None:
                self.path_csv_img = None
                self.path_csv_img = None
            else:
                self.path_csv_img = cfg['leafmachine']['project']['path_images_csv_local']
                self.csv_img = import_csv(self.path_csv_img)

            if self.csv_combined is None:
                if cfg['leafmachine']['project']['path_combined_csv_local'] is not None:
                    logger.info('Combined CSV file not provided, creating it now...')
                    location = self.__create_combined_csv()
                    logger.info(''.join(['Combined CSV --> ', location]))
                else:
                    logger.info('Combined CSV file not available or provided. Skipped record import.')
            else:
                logger.info(''.join(['Combined CSV --> ', self.path_csv_combined]))
        except Exception as e:
            logger.error(f"Error importing local files: {e}")
        logger.info(''.join(['Image Directory --> ', self.dir_images]))

    def __import_GBIF_files_post_download(self, cfg, logger, dir_home):
        print_main_warn('Downloading Images from GBIF...')
        logger.info('Downloading Images from GBIF...')
        self.cfg_images = download_all_images_from_GBIF_LM2(dir_home, cfg['leafmachine']['project']['GBIF_mode'])
        self.dir_images = self.cfg_images['dir_destination_images']
        self.path_csv = self.cfg_images['dir_destination_csv']
        print_main_success(''.join(['Images saved to --> ', self.dir_images]))
        logger.info(''.join(['Images saved to --> ', self.dir_images]))

        self.path_csv_combined = os.path.join(self.path_csv, self.cfg_images['filename_combined'])
        self.path_csv_occ = os.path.join(self.path_csv, self.cfg_images['filename_occ'])
        self.path_csv_img = os.path.join(self.path_csv, self.cfg_images['filename_img'])

        if 'txt' in (self.cfg_images['filename_occ'].split('.')[1] or self.cfg_images['filename_img'].split('.')[1]):
            self.csv_combined = import_tsv(self.path_csv_combined)
        else:
            self.csv_combined = import_csv(self.path_csv_combined)

    def process_in_batches(self, cfg):
        batch_size = cfg['leafmachine']['project']['batch_size']
        self.project_data_list = []
        keys = list(self.project_data.keys())
        num_batches = len(keys) // batch_size + 1
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch_keys = keys[start:end]
            batch = {key: self.project_data[key] for key in batch_keys}
            self.project_data_list.append(batch)
        return num_batches, len(self.project_data)

    def __make_project_dict(self):
        self.project_data = {}
        invalid_dir = None

        for img in os.listdir(self.dir_images):
            img_split, ext = os.path.splitext(img)
            if ext in self.valid_extensions:
                try:
                    with Image.open(os.path.join(self.dir_images, img)) as im:
                        _, ext = os.path.splitext(img)
                        if ext not in ['.jpg']:
                            im = im.convert('RGB')
                            new_img_name = ''.join([img_split, '.jpg'])
                            im.save(os.path.join(self.dir_images, new_img_name), quality=100)
                            img = new_img_name
                    img_name = os.path.splitext(img)[0]
                    self.project_data[img_split] = {}
                    self.insert_image(img_name, os.path.join(self.dir_images, img), 1)
                except Exception as e:
                    print(f"Error processing image {img}: {e}")
            else:
                if invalid_dir is None:
                    invalid_dir = os.path.join(os.path.dirname(self.dir_images), 'INVALID_FILES')
                    os.makedirs(invalid_dir, exist_ok=True)
                if not os.path.exists(os.path.join(invalid_dir, img)):
                    shutil.move(os.path.join(self.dir_images, img), os.path.join(invalid_dir, img))
                self.insert_image(img_split, os.path.join(invalid_dir, img), 0)

    def add_records_to_project_dict(self):
        for img in os.listdir(self.dir_images):
            if img.endswith(".jpg") or img.endswith(".jpeg"):
                img_name = str(img.split('.')[0])
                try:
                    self.project_data[img_name]['GBIF_Record'] = self.__get_data_from_combined(img_name)
                except:
                    self.project_data[img_name]['GBIF_Record'] = None

    def __get_data_from_combined(self, img_name):
        df = pd.DataFrame(self.csv_combined)
        row = df[df['filename_image'] == img_name].head(1).to_dict()
        return row

    def close_connection(self):
        if self.conn:
            self.conn.close()

def process_image_batch(image_batch, db_path, batch_commit_size=10, retries=5, delay=0.5):
    REQ_PIXEL_HEIGHT = 3400
    conn = sqlite3.connect(db_path, timeout=30)  # Increased timeout for database lock
    cur = conn.cursor()
    
    batch_updates = []
    
    for img_id, img_path in image_batch:
        width, height = fast_image_size(img_path)
        if width is not None and height is not None:  # Only proceed if valid dimensions are returned
            longest_side = max(width, height)
            if longest_side < REQ_PIXEL_HEIGHT:
                new_width, new_height = resize_image_to_min_length(img_path, width, height, REQ_PIXEL_HEIGHT)
                # If resize happens, we update dimensions
                width, height = new_width, new_height

            batch_updates.append((width, height, img_id))
        else:
            print(f"Warning: Unable to get dimensions for image {img_path}")
            
        # Perform batch updates when the size reaches the batch_commit_size
        if len(batch_updates) >= batch_commit_size:
            for attempt in range(retries):
                try:
                    cur.executemany("UPDATE images SET width = ?, height = ? WHERE id = ?", batch_updates)
                    conn.commit()
                    batch_updates.clear()  # Clear the batch after commit
                    break  # Exit retry loop after success
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e):
                        print(f"Database is locked, retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
                        time.sleep(delay)  # Wait and retry if database is locked
                    else:
                        print(f"Error updating database for batch: {e}")
                        break
    
    # Commit any remaining updates
    if batch_updates:
        try:
            cur.executemany("UPDATE images SET width = ?, height = ? WHERE id = ?", batch_updates)
            conn.commit()
        except sqlite3.OperationalError as e:
            print(f"Final batch update failed due to database lock: {e}")
    
    cur.close()
    conn.close()

def resize_image_to_min_length(img_path, width, height, required_length):
    # Determine the longest side and calculate the scaling factor
    if width > height:
        scale_factor = required_length / width
        new_width = required_length
        new_height = int(height * scale_factor)
    else:
        scale_factor = required_length / height
        new_height = required_length
        new_width = int(width * scale_factor)

    # Resize the image and save it
    try:
        img_small = Image.open(img_path)
        new_img = img_small.resize((new_width, new_height), Image.LANCZOS)
        new_img.save(img_path)  # Overwrite the original image or save to a new path if needed
        print(f"Image {img_path} resized from h:{height} w:{width} >>> TO >>> h:{new_height} w:{new_width}")

    except Exception as e:
        print(f"Error resizing image {img_path}: {e}")

    return new_width, new_height

def fast_image_size(img_path):
    parser = ImageFile.Parser()
    try:
        # First try the fast method by parsing chunks of the file
        with open(img_path, 'rb') as f:
            while True:
                chunk = f.read(1024)
                if not chunk:
                    break
                parser.feed(chunk)
                if parser.image:
                    return parser.image.size
    except Exception as e:
        print(f"Fast method failed for image {img_path}: {e}")
    
    # Fallback to PIL if the fast method fails
    try:
        with Image.open(img_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"PIL fallback also failed for image {img_path}: {e}")
    
    # If both methods fail, return None, None
    return None, None


def get_database_path(project_info):
    return project_info.database

class Project_Stats():
    specimens = 0
    rulers = 0

    def __init__(self, cfg, logger, dir_home) -> None:
        logger.name = 'Project Info'
        logger.info("Gathering Images and Image Metadata")

# def test_sql(database):
#     try:
#         conn = sqlite3.connect(database)
#         cur = conn.cursor()

#         # Print first two entries from images table
#         cur.execute("SELECT * FROM images LIMIT 2")
#         image_rows = cur.fetchall()
        
#         print("\nImages Table:\n")
#         for row in image_rows:
#             print(row)
#             image_name = row[1]
            
#             # Print associated archival components
#             cur.execute("SELECT * FROM archival_components WHERE image_name = ?", (image_name,))
#             archival_rows = cur.fetchall()
#             print(f"Archival Components for {image_name}:")
#             for archival_row in archival_rows:
#                 print(archival_row)

#             # Print associated plant components
#             cur.execute("SELECT * FROM plant_components WHERE image_name = ?", (image_name,))
#             plant_rows = cur.fetchall()
#             print(f"Plant Components for {image_name}:")
#             for plant_row in plant_rows:
#                 print(plant_row)
            
#             print(f"\n")
        
#         conn.close()
#     except Error as e:
#         print(e)
def test_sql(database, n_rows=1):
    try:
        conn = sqlite3.connect(database)
        cur = conn.cursor()

        # List all tables in the database
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cur.fetchall()

        print("\nDatabase Diagnostics\n" + "="*50)

        # Loop through all tables and print their contents
        for table_name in tables:
            table_name = table_name[0]
            print(f"\nTable: {table_name}\n" + "-"*50)

            # Get the table's column names and types
            cur.execute(f"PRAGMA table_info({table_name})")
            columns_info = cur.fetchall()
            columns = [column[1] for column in columns_info]
            column_types = {column[1]: column[2] for column in columns_info}  # Map column name to type
            print(f"Columns: {columns}")

            # Fetch and print the specified number of rows from the table
            cur.execute(f"SELECT * FROM {table_name} LIMIT ?", (n_rows,))
            rows = cur.fetchall()

            if rows:
                # Determine the width of each column for pretty printing
                col_widths = []
                for i, col in enumerate(columns):
                    if column_types[col] == 'BLOB':
                        col_widths.append(5)  # Fixed small width for BLOB columns
                    else:
                        max_width = max(len(str(item)) for item in [col] + [row[i] for row in rows])
                        col_widths.append(max_width)

                # Print column headers
                header = " | ".join(f"{col_name:<{col_width}}" for col_name, col_width in zip(columns, col_widths))
                print(header)
                print("-" * len(header))

                # Print each row with aligned columns
                for row in rows:
                    row_str = " | ".join(
                        f"{'TRUE' if column_types[columns[i]] == 'BLOB' and item else 'FALSE' if column_types[columns[i]] == 'BLOB' else str(item):<{col_width}}"
                        for i, (item, col_width) in enumerate(zip(row, col_widths))
                    )
                    print(row_str)
            else:
                print("No data found in this table.")

        conn.close()

    except sqlite3.Error as e:
        print(f"Error testing SQL database: {e}")