import os
import sys
import inspect
import shutil
import random
import pandas as pd
import sqlite3
from sqlite3 import Error
from PIL import Image, ImageFile
from dataclasses import dataclass, field
from tqdm import tqdm

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
        self.conn = self.create_connection(self.database)

        self.valid_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF']

        if self.image_location in ['local','l','L','Local']:
            print(f"Directory for images: {self.dir_images}")  # Add this line to log the directory being used
            self.__import_local_files(cfg, logger)
        elif self.image_location in ['GBIF','g','G','gbif']:
            self.__import_GBIF_files_post_download(cfg, logger, dir_home)

        make_file_names_valid(self.dir_images, cfg)
        make_images_in_dir_vertical(self.dir_images, cfg)

        self.create_tables()
        # self.create_archival_components_table()  # Add this line to create the archival_components table
        # self.create_plant_components_table()  # Add this line to create the archival_components table
        self.__make_project_dict()
        self.populate_image_dimensions()  # Populate dimensions after creating the tables
        self.create_tables_if_not_exist()

    def create_tables_if_not_exist(self):
        try:
            cur = self.conn.cursor()

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

            # Create the 'Whole_Leaf_Cropped' and 'Partial_Leaf_Cropped' tables
            cur.execute("""
                CREATE TABLE IF NOT EXISTS Whole_Leaf_Cropped (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT NOT NULL,
                    crop_name TEXT NOT NULL,
                    cropped_image BLOB NOT NULL
                )
            """)
        
            cur.execute("""
                CREATE TABLE IF NOT EXISTS Partial_Leaf_Cropped (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT NOT NULL,
                    crop_name TEXT NOT NULL,
                    cropped_image BLOB NOT NULL
                )
            """)

            # Create the 'Segmentation_Whole_Leaf' and 'Segmentation_Partial_Leaf' tables
            cur.execute("""
                CREATE TABLE IF NOT EXISTS Segmentation_Whole_Leaf (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT NOT NULL,
                    crop_name TEXT NOT NULL,
                    segmentation_data TEXT NOT NULL,
                    polygons TEXT NOT NULL,
                    bboxes TEXT NOT NULL,
                    labels TEXT NOT NULL,
                    colors TEXT NOT NULL,
                    overlay_data TEXT NOT NULL
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS Segmentation_Partial_Leaf (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT NOT NULL,
                    crop_name TEXT NOT NULL,
                    segmentation_data TEXT NOT NULL,
                    polygons TEXT NOT NULL,
                    bboxes TEXT NOT NULL,
                    labels TEXT NOT NULL,
                    colors TEXT NOT NULL,
                    overlay_data TEXT NOT NULL
                )
            """)


            # Commit changes
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"An error occurred while creating tables: {e}")


    def populate_image_dimensions(self):
        cur = self.conn.cursor()

        # Fetch all images that don't have width and height set
        cur.execute("SELECT id, path FROM images WHERE width IS NULL OR height IS NULL")
        images = cur.fetchall()

        for img_id, img_path in images:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    cur.execute("UPDATE images SET width = ?, height = ? WHERE id = ?", (width, height, img_id))
                    self.conn.commit()
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

    def create_connection(self, db_file):
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            return conn
        except Error as e:
            print(e)
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
            print(f"Inserted image into database: {name}, {path}, {valid}")  # Log each database insertion
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