import os, sys, inspect, shutil
from dataclasses import dataclass, field
import pandas as pd
currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)
from leafmachine2.machine.general_utils import import_csv, import_tsv
from leafmachine2.machine.general_utils import Print_Verbose, print_main_warn, print_main_success, make_file_names_valid, make_images_in_dir_vertical
from leafmachine2.machine.utils_GBIF import generate_image_filename
from leafmachine2.downloading.download_from_GBIF_all_images_in_file import download_all_images_from_GBIF_LM2
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

@dataclass
class Project_Info():
    batch_size: int = 50

    image_location: str = ''

    dir_images: str = ''

    project_data: object = field(init=False)
    project_data_list: object = field(init=False)

    path_csv_combined: str = ''
    path_csv_occ: str = ''
    path_csv_img: str = ''    
    csv_combined: str = ''
    csv_occ: str = ''
    csv_img: str = ''

    has_valid_images: bool = True

    def __init__(self, cfg, logger, dir_home) -> None:
        logger.name = 'Project Info'
        logger.info("Gathering Images and Image Metadata")

        self.batch_size = cfg['leafmachine']['project']['batch_size']

        self.image_location = cfg['leafmachine']['project']['image_location']
        
        self.valid_extensions = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF']

        # If project is local, expect:
        #       dir with images
        #       path to images.csv
        #       path to occ.csv
        #   OR  path to combined.csv
        if self.image_location in ['local','l','L','Local']:
            self.__import_local_files(cfg, logger)

        # If project is GBIF, expect:
        #       Darwin Core Images (or multimedia.txt) and Occurrences file pair, either .txt or .csv
        elif self.image_location in ['GBIF','g','G','gbif']:
            self.__import_GBIF_files_post_download(cfg, logger, dir_home)

        # self.__make_project_dict() #, self.batch_size)

        # Make sure image file names are legal
        make_file_names_valid(self.dir_images, cfg)
        
        # Make all images vertical
        make_images_in_dir_vertical(self.dir_images, cfg)

        self.__make_project_dict() #, self.batch_size)



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
        # print(self.csv_img.head(5))

        combined = pd.merge(self.csv_img, self.csv_occ, left_on='gbifID_images', right_on='gbifID')
        # print(combined.head(5))
        names_list = combined.apply(generate_image_filename, axis=1, result_type='expand')
        # print(names_list.head(5))
        # Select columns 7, 0, 1
        selected_columns = names_list.iloc[:,[7,0,1]]
        # Rename columns
        selected_columns.columns = ['fullname','filename_image','filename_image_jpg']
        # print(selected_columns.head(5))
        self.csv_combined = pd.concat([selected_columns, combined], axis=1)
        # print(self.csv_combined.head(5))
        new_name = ''.join(['combined_', os.path.basename(self.path_csv_occ).split('.')[0], '_', os.path.basename(self.path_csv_img).split('.')[0], '.csv'])
        self.path_csv_combined = os.path.join(os.path.dirname(self.path_csv_occ), new_name)
        self.csv_combined.to_csv(self.path_csv_combined, mode='w', header=True, index=False)
        return self.path_csv_combined

    def __import_local_files(self, cfg, logger):
        # Images
        if cfg['leafmachine']['project']['dir_images_local'] is None:
            self.dir_images = None
        else:
            self.dir_images = cfg['leafmachine']['project']['dir_images_local']
        
        # CSV import
        # Combined
        try:
            if cfg['leafmachine']['project']['path_combined_csv_local'] is None:
                self.csv_combined = None
                self.path_csv_combined = None
            else:
                self.path_csv_combined = cfg['leafmachine']['project']['path_combined_csv_local']
                self.csv_combined = import_csv(self.path_csv_combined)
            # Occurrence
            if cfg['leafmachine']['project']['path_occurrence_csv_local'] is None:
                self.csv_occ = None
                self.path_csv_occ = None
            else:
                self.path_csv_occ = cfg['leafmachine']['project']['path_occurrence_csv_local']
                self.csv_occ = import_csv(self.path_csv_occ)
            # Images/metadata
            if cfg['leafmachine']['project']['path_images_csv_local'] is None:
                self.path_csv_img = None
                self.path_csv_img = None
            else:
                self.path_csv_img = cfg['leafmachine']['project']['path_images_csv_local']
                self.csv_img = import_csv(self.path_csv_img)

            # Create combined if it's missing
            if self.csv_combined is None:
                if cfg['leafmachine']['project']['path_combined_csv_local'] is not None:
                    # Print_Verbose(cfg, 2, 'Combined CSV file not provided, creating it now...').bold()
                    logger.info('Combined CSV file not provided, creating it now...')
                    location = self.__create_combined_csv()
                    # Print_Verbose(cfg, 2, ''.join(['Combined CSV --> ',location])).green()
                    logger.info(''.join(['Combined CSV --> ',location]))

                else:
                    # Print_Verbose(cfg, 2, 'Combined CSV file not available or provided. Skipped record import.').bold()
                    logger.info('Combined CSV file not available or provided. Skipped record import.')
            else:
                # Print_Verbose(cfg, 2, ''.join(['Combined CSV --> ',self.path_csv_combined])).green()
                logger.info(''.join(['Combined CSV --> ',self.path_csv_combined]))
        except:
            pass

        # Print_Verbose(cfg, 2, ''.join(['Image Directory --> ',self.dir_images])).green()
        logger.info(''.join(['Image Directory --> ',self.dir_images]))


    
    def __import_GBIF_files_post_download(self, cfg, logger, dir_home):
        # Download the images from GBIF
        # This pulls from /LeafMachine2/configs/config_download_from_GBIF_all_images_in_file or filter
        print_main_warn('Downloading Images from GBIF...')
        logger.info('Downloading Images from GBIF...')
        self.cfg_images = download_all_images_from_GBIF_LM2(dir_home, cfg['leafmachine']['project']['GBIF_mode'])
        self.dir_images = self.cfg_images['dir_destination_images']
        self.path_csv = self.cfg_images['dir_destination_csv']
        print_main_success(''.join(['Images saved to --> ',self.dir_images]))
        logger.info(''.join(['Images saved to --> ',self.dir_images]))


        self.path_csv_combined = os.path.join(self.path_csv, self.cfg_images['filename_combined'])
        self.path_csv_occ = os.path.join(self.path_csv, self.cfg_images['filename_occ'])
        self.path_csv_img = os.path.join(self.path_csv, self.cfg_images['filename_img'])

        if 'txt' in (self.cfg_images['filename_occ'].split('.')[1] or self.cfg_images['filename_img'].split('.')[1]):
            self.csv_combined = import_tsv(self.path_csv_combined)
            # self.csv_occ = import_tsv(self.path_csv_occ)
            # self.csv_img = import_tsv(self.path_csv_img)
        else:
            self.csv_combined = import_csv(self.path_csv_combined)
            # self.csv_occ = import_csv(self.path_csv_occ)
            # self.csv_img = import_csv(self.path_csv_img)

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
    
    def process_in_batches_spoof(self, cfg):
        self.project_data_list = []
        keys = list(self.project_data.keys())
        batch_size = len(keys)  # Entire dataset as one batch

        # Directly use the original input data
        num_batches = 1  # Only one batch since we're not splitting the data
        start = 0
        end = batch_size
        batch_keys = keys[start:end]
        batch = {key: self.project_data[key] for key in batch_keys}
        self.project_data_list.append(batch)
        return num_batches, len(self.project_data)


    # Original
    '''def __make_project_dict(self):
        self.project_data = {}
        for img in os.listdir(self.dir_images):
            if (img.endswith(".jpg") or img.endswith(".jpeg")):
                img_name = str(img.split('.')[0])
                self.project_data[img_name] = {}
    '''
    # def __make_project_dict(self): # This DELETES the invalid file, not safe
    #     self.project_data = {}
    #     for img in os.listdir(self.dir_images):
    #         img_split, ext = os.path.splitext(img)
    #         if ext.lower() in self.valid_extensions:
    #             with Image.open(os.path.join(self.dir_images, img)) as im:
    #                 _, ext = os.path.splitext(img)
    #                 if ext != '.jpg':
    #                     im = im.convert('RGB')
    #                     im.save(os.path.join(self.dir_images, img_split) + '.jpg', quality=100)
    #                     img += '.jpg'
    #                     os.remove(os.path.join(self.dir_images, ''.join([img_split, ext])))
    #             img_name = os.path.splitext(img)[0]
    #             self.project_data[img_split] = {}

    # def __make_project_dict(self):
    #     self.project_data = {}
    #     invalid_dir = None

    #     for img in os.listdir(self.dir_images):
    #         img_split, ext = os.path.splitext(img)
    #         if ext in self.valid_extensions:
    #             with Image.open(os.path.join(self.dir_images, img)) as im:
    #                 _, ext = os.path.splitext(img)
    #                 if ext not in ['.jpg']:
    #                     im = im.convert('RGB')
    #                     new_img_name = ''.join([img_split, '.jpg'])
    #                     im.save(os.path.join(self.dir_images, new_img_name), quality=100)
    #                     self.project_data[img_split] = {}

    #                     # Prepare the INVALID_FILES directory
    #                     # if invalid_dir is None:
    #                     #     invalid_dir = os.path.join(os.path.dirname(self.dir_images), 'INVALID_FILES')
    #                     #     os.makedirs(invalid_dir, exist_ok=True)

    #                     # # Copy the original file to the INVALID_FILES directory instead of moving it
    #                     # invalid_file_path = os.path.join(invalid_dir, img)
    #                     # if not os.path.exists(invalid_file_path):
    #                     #     shutil.copy2(os.path.join(self.dir_images, img), invalid_file_path)

    #                     # Update the img variable to the new image name
    #                     img = new_img_name
    #             img_name = os.path.splitext(img)[0]
    #             self.project_data[img_split] = {}
    #         else:
    #             # if the file has an invalid extension, move it to the INVALID_FILE directory
    #             if invalid_dir is None:
    #                 invalid_dir = os.path.join(os.path.dirname(self.dir_images), 'INVALID_FILES')
    #                 os.makedirs(invalid_dir, exist_ok=True)

    #             # skip if the file already exists in the INVALID_FILE directory
    #             if not os.path.exists(os.path.join(invalid_dir, img)):
    #                 shutil.move(os.path.join(self.dir_images, img), os.path.join(invalid_dir, img))
    def __make_project_dict(self):
        self.project_data = {}
        converted_dir = None
        contains_non_jpg = False

        # Check if any non-JPG files are present
        for img in os.listdir(self.dir_images):
            img_split, ext = os.path.splitext(img)
            if ext.lower() not in ['.jpg',]:
                contains_non_jpg = True
                break

        # If there are non-JPG files, create the 'converted_to_jpg' directory
        if contains_non_jpg:
            converted_dir = os.path.join(self.dir_images, 'converted_to_jpg')
            os.makedirs(converted_dir, exist_ok=True)

        for img in os.listdir(self.dir_images):
            img_split, ext = os.path.splitext(img)
            ext = ext.lower()

            if ext in self.valid_extensions:
                img_path = os.path.join(self.dir_images, img)

                if ext not in ['.jpg', '.jpeg']:
                    # Convert the image to JPG
                    with Image.open(img_path) as im:
                        im = im.convert('RGB')
                        new_img_name = f"{img_split}.jpg"
                        new_img_path = os.path.join(converted_dir, new_img_name)
                        im.save(new_img_path, quality=100)
                    self.project_data[img_split] = {}
                else:
                    # Copy the JPG image to the new directory if necessary
                    if contains_non_jpg:
                        new_img_path = os.path.join(converted_dir, img)
                        shutil.copy2(img_path, new_img_path)
                    self.project_data[img_split] = {}

        # Update self.dir_images to point to the new directory if it was created
        if contains_non_jpg:
            self.dir_images = converted_dir

    def add_records_to_project_dict(self):
        for img in os.listdir(self.dir_images):
            if (img.endswith(".jpg") or img.endswith(".jpeg")):
                img_name = str(img.split('.')[0])
                try:
                    self.project_data[img_name]['GBIF_Record'] = self.__get_data_from_combined(img_name)
                except:
                    self.project_data[img_name]['GBIF_Record'] = None

    def __get_data_from_combined(self, img_name):
        df = pd.DataFrame(self.csv_combined)
        row = df[df['filename_image'] == img_name].head(1).to_dict()
        return row


class Project_Stats():
    specimens = 0
    
    rulers = 0
    

    def __init__(self, cfg, logger, dir_home) -> None:
        logger.name = 'Project Info'
        logger.info("Gathering Images and Image Metadata")