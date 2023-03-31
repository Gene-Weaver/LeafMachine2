import os, time, requests, yaml, re, csv, sys, inspect
from dataclasses import dataclass, field
# from difflib import diff_bytes
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from torch import ge
from re import S
from threading import Lock

currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)
from concurrent.futures import ThreadPoolExecutor as th


from leafmachine2.machine.general_utils import bcolors, validate_dir

'''
For download parallelization, I followed this guide https://rednafi.github.io/digressions/python/2020/04/21/python-concurrent-futures.html
'''

'''
####################################################################################################
Read config files
####################################################################################################
'''
def get_cfg_from_full_path(path_cfg):
    with open(path_cfg, "r") as ymlfile:
        cfg = yaml.full_load(ymlfile)
    return cfg

'''
Classes
'''
@dataclass
class ImageCandidate:
    cfg: str = ''
    herb_code: str = '' 
    specimen_id: str = ''
    family: str = ''
    genus: str = ''
    species: str = ''
    fullname: str = ''

    filename_image: str = ''
    filename_image_jpg: str = ''

    url: str = ''
    headers_occ: str = ''
    headers_img: str = ''

    occ_row: list = field(init=False,default_factory=None)
    image_row: list = field(init=False,default_factory=None)


    def __init__(self, cfg, image_row, occ_row, url, lock):
        self.headers_occ =  list(occ_row.columns.values)
        self.headers_img = list(occ_row.columns.values)
        self.occ_row = occ_row # pd.DataFrame(data=occ_row,columns=self.headers_occ)
        self.image_row = image_row # pd.DataFrame(data=image_row,columns=self.headers_img)
        self.url = url
        self.cfg = cfg

        self.filename_image, self.filename_image_jpg, self.herb_code, self.specimen_id, self.family, self.genus, self.species, self.fullname = generate_image_filename(occ_row)
        self.download_image(lock)

    def download_image(self, lock) -> None:
        dir_destination = self.cfg['dir_destination_images']
        MP_low = self.cfg['MP_low']
        MP_high = self.cfg['MP_high']
        # Define URL get parameters
        sep = '_'
        session = requests.Session()
        retry = Retry(connect=1) #2, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        print(f"{bcolors.BOLD}      {self.fullname}{bcolors.ENDC}")
        print(f"{bcolors.BOLD}           URL: {self.url}{bcolors.ENDC}")
        try:
            response = session.get(self.url, stream=True, timeout=1.0)
            img = Image.open(response.raw)
            self._save_matching_image(img, MP_low, MP_high, dir_destination, lock)
            print(f"{bcolors.OKGREEN}                SUCCESS{bcolors.ENDC}")
        except Exception as e: 
            print(f"{bcolors.FAIL}                SKIP No Connection or ERROR --> {e}{bcolors.ENDC}")
            print(f"{bcolors.WARNING}                Status Code --> {response.status_code}{bcolors.ENDC}")
            print(f"{bcolors.WARNING}                Reasone --> {response.reason}{bcolors.ENDC}")

    def _save_matching_image(self, img, MP_low, MP_high, dir_destination, lock) -> None:
        img_mp, img_w, img_h = check_image_size(img)
        if img_mp < MP_low:
            print(f"{bcolors.WARNING}                SKIP < {MP_low}MP: {img_mp}{bcolors.ENDC}")

        elif MP_low <= img_mp <= MP_high:
            image_path = os.path.join(dir_destination,self.filename_image_jpg)
            img.save(image_path)

            #imgSaveName = pd.DataFrame({"image_path": [image_path]})
            self._add_occ_and_img_data(lock)

            print(f"{bcolors.OKGREEN}                Regular MP: {img_mp}{bcolors.ENDC}")
            print(f"{bcolors.OKGREEN}                Image Saved: {image_path}{bcolors.ENDC}")

        elif img_mp > MP_high:
            if self.cfg['do_resize']:
                [img_w, img_h] = calc_resize(img_w, img_h)
                newsize = (img_w, img_h)
                img = img.resize(newsize)
                image_path = os.path.join(dir_destination,self.filename_image_jpg)
                img.save(image_path)

                #imgSaveName = pd.DataFrame({"imgSaveName": [imgSaveName]})
                self._add_occ_and_img_data(lock)
                
                print(f"{bcolors.OKGREEN}                {MP_high}MP+ Resize: {img_mp}{bcolors.ENDC}")
                print(f"{bcolors.OKGREEN}                Image Saved: {image_path}{bcolors.ENDC}")
            else:
                print(f"{bcolors.OKCYAN}                {MP_high}MP+ Resize: {img_mp}{bcolors.ENDC}")
                print(f"{bcolors.OKCYAN}                SKIP: {image_path}{bcolors.ENDC}")
    
    def _add_occ_and_img_data(self, lock) -> None:
        self.image_row = self.image_row.to_frame().transpose().rename(columns={"identifier": "url"}) 
        self.image_row = self.image_row.rename(columns={"gbifID": "gbifID_images"}) 

        new_data = {'fullname': [self.fullname], 'filename_image': [self.filename_image], 'filename_image_jpg': [self.filename_image_jpg]}
        new_data = pd.DataFrame(data=new_data)

        all_data = [new_data.reset_index(), self.image_row.reset_index(), self.occ_row.reset_index()]
        combined = pd.concat(all_data,ignore_index=False, axis=1)

        w_1 = new_data.shape[1] + 1
        w_2 = self.image_row.shape[1] + 1
        w_3 = self.occ_row.shape[1]

        combined.drop([combined.columns[0], combined.columns[w_1], combined.columns[w_1 + w_2]], axis=1, inplace=True)
        headers = np.hstack((new_data.columns.values, self.image_row.columns.values, self.occ_row.columns.values))
        combined.columns = headers
        self._append_combined_occ_image(self.cfg, combined, lock)

    def _append_combined_occ_image(self, cfg, combined, lock) -> None:
        path_csv_combined = os.path.join(cfg['dir_destination_csv'], cfg['filename_combined'])
        with lock:
            try: 
                # Add row once the file exists
                csv_combined = pd.read_csv(path_csv_combined,dtype=str)
                combined.to_csv(path_csv_combined, mode='a', header=False, index=False)
                print(f'{bcolors.OKGREEN}       Added 1 row to combined CSV: {path_csv_combined}{bcolors.ENDC}')

            except Exception as e:
                print(f"{bcolors.WARNING}       Initializing new combined .csv file: [occ,images]: {path_csv_combined}{bcolors.ENDC}")
                combined.to_csv(path_csv_combined, mode='w', header=True, index=False)



@dataclass
class ImageCandidateMulti:
    cfg: str = ''
    herb_code: str = '' 
    specimen_id: str = ''
    family: str = ''
    genus: str = ''
    species: str = ''
    fullname: str = ''

    filename_image: str = ''
    filename_image_jpg: str = ''

    url: str = ''
    headers_occ: str = ''
    headers_img: str = ''

    occ_row: list = field(init=False,default_factory=None)
    image_row: list = field(init=False,default_factory=None)


    def __init__(self, cfg, image_row, occ_row, url, dir_destination, lock):
        self.headers_occ =  list(occ_row.columns.values)
        self.headers_img = list(occ_row.columns.values)
        self.occ_row = occ_row # pd.DataFrame(data=occ_row,columns=self.headers_occ)
        self.image_row = image_row # pd.DataFrame(data=image_row,columns=self.headers_img)
        self.url = url
        self.cfg = cfg

        self.filename_image, self.filename_image_jpg, self.herb_code, self.specimen_id, self.family, self.genus, self.species, self.fullname = generate_image_filename(occ_row)
        self.download_image(dir_destination, lock)

    def download_image(self, dir_destination, lock) -> None:
        # dir_destination = self.cfg['dir_destination_images']
        MP_low = self.cfg['MP_low']
        MP_high = self.cfg['MP_high']
        # Define URL get parameters
        sep = '_'
        session = requests.Session()
        retry = Retry(connect=1) #2, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        print(f"{bcolors.BOLD}      {self.fullname}{bcolors.ENDC}")
        print(f"{bcolors.BOLD}           URL: {self.url}{bcolors.ENDC}")
        try:
            response = session.get(self.url, stream=True, timeout=1.0)
            img = Image.open(response.raw)
            self._save_matching_image(img, MP_low, MP_high, dir_destination, lock)
            print(f"{bcolors.OKGREEN}                SUCCESS{bcolors.ENDC}")
        except Exception as e: 
            print(f"{bcolors.FAIL}                SKIP No Connection or ERROR --> {e}{bcolors.ENDC}")
            print(f"{bcolors.WARNING}                Status Code --> {response.status_code}{bcolors.ENDC}")
            print(f"{bcolors.WARNING}                Reasone --> {response.reason}{bcolors.ENDC}")

    def _save_matching_image(self, img, MP_low, MP_high, dir_destination, lock) -> None:
        img_mp, img_w, img_h = check_image_size(img)
        if img_mp < MP_low:
            print(f"{bcolors.WARNING}                SKIP < {MP_low}MP: {img_mp}{bcolors.ENDC}")

        elif MP_low <= img_mp <= MP_high:
            image_path = os.path.join(dir_destination,self.filename_image_jpg)
            img.save(image_path)

            #imgSaveName = pd.DataFrame({"image_path": [image_path]})
            self._add_occ_and_img_data(lock)

            print(f"{bcolors.OKGREEN}                Regular MP: {img_mp}{bcolors.ENDC}")
            print(f"{bcolors.OKGREEN}                Image Saved: {image_path}{bcolors.ENDC}")

        elif img_mp > MP_high:
            if self.cfg['do_resize']:
                [img_w, img_h] = calc_resize(img_w, img_h)
                newsize = (img_w, img_h)
                img = img.resize(newsize)
                image_path = os.path.join(dir_destination,self.filename_image_jpg)
                img.save(image_path)

                #imgSaveName = pd.DataFrame({"imgSaveName": [imgSaveName]})
                self._add_occ_and_img_data(lock)
                
                print(f"{bcolors.OKGREEN}                {MP_high}MP+ Resize: {img_mp}{bcolors.ENDC}")
                print(f"{bcolors.OKGREEN}                Image Saved: {image_path}{bcolors.ENDC}")
            else:
                print(f"{bcolors.OKCYAN}                {MP_high}MP+ Resize: {img_mp}{bcolors.ENDC}")
                print(f"{bcolors.OKCYAN}                SKIP: {image_path}{bcolors.ENDC}")
    
    def _add_occ_and_img_data(self, lock) -> None:
        self.image_row = self.image_row.to_frame().transpose().rename(columns={"identifier": "url"}) 
        self.image_row = self.image_row.rename(columns={"gbifID": "gbifID_images"}) 

        new_data = {'fullname': [self.fullname], 'filename_image': [self.filename_image], 'filename_image_jpg': [self.filename_image_jpg]}
        new_data = pd.DataFrame(data=new_data)

        all_data = [new_data.reset_index(), self.image_row.reset_index(), self.occ_row.reset_index()]
        combined = pd.concat(all_data,ignore_index=False, axis=1)

        w_1 = new_data.shape[1] + 1
        w_2 = self.image_row.shape[1] + 1
        w_3 = self.occ_row.shape[1]

        combined.drop([combined.columns[0], combined.columns[w_1], combined.columns[w_1 + w_2]], axis=1, inplace=True)
        headers = np.hstack((new_data.columns.values, self.image_row.columns.values, self.occ_row.columns.values))
        combined.columns = headers
        self._append_combined_occ_image(self.cfg, combined, lock)

    def _append_combined_occ_image(self, cfg, combined, lock) -> None:
        path_csv_combined = os.path.join(cfg['dir_destination_csv'], cfg['filename_combined'])
        with lock:
            try: 
                # Add row once the file exists
                csv_combined = pd.read_csv(path_csv_combined,dtype=str)
                combined.to_csv(path_csv_combined, mode='a', header=False, index=False)
                print(f'{bcolors.OKGREEN}       Added 1 row to combined CSV: {path_csv_combined}{bcolors.ENDC}')

            except Exception as e:
                print(f"{bcolors.WARNING}       Initializing new combined .csv file: [occ,images]: {path_csv_combined}{bcolors.ENDC}")
                combined.to_csv(path_csv_combined, mode='w', header=True, index=False)

'''
####################################################################################################
General Functions
####################################################################################################
'''
# If image is larger than MP max, downsample to have long side = 5000
def calc_resize(w,h):
    if h > w:
        ratio = h/w
        new_h = 5000
        new_w = round(5000/ratio)
    elif w >= h:
        ratio = w/h
        new_w = 5000
        new_h = round(5000/ratio)
    return new_w, new_h

def check_image_size(img):
    [img_w, img_h] = img.size
    img_mp = round(img_w * img_h / 1000000,1)
    return img_mp, img_w, img_h

def check_n_images_in_group(detailedOcc,N):
    fam = detailedOcc['fullname'].unique()
    for f in fam:
        ct = len(detailedOcc[detailedOcc['fullname'].str.match(f)])
        if ct == N:
            print(f"{bcolors.OKGREEN}{f}: {ct}{bcolors.ENDC}")
        else:
            print(f"{bcolors.FAIL}{f}: {ct}{bcolors.ENDC}")



'''
####################################################################################################
Functions for --> download_GBIF_from_user_file.py
####################################################################################################
'''

# def download_subset_images_user_file(dir_home,dir_destination,n_already_downloaded,MP_low,MP_high,wishlist,filename_occ,filename_img):
#     # (dirWishlists,dirNewImg,alreadyDownloaded,MP_Low,MP_High,wishlist,aggOcc_filename,aggImg_filename):
#     sep = '_'
#     aggOcc = pd.DataFrame()
#     aggImg = pd.DataFrame()

#     # Define URL get parameters
#     session = requests.Session()
#     retry = Retry(connect=1) #2, backoff_factor=0.5)
#     adapter = HTTPAdapter(max_retries=retry)
#     session.mount('http://', adapter)
#     session.mount('https://', adapter)

#     listMax = wishlist.shape[0]
#     for index, spp in wishlist.iterrows():
#         imageFound = False
#         currentFamily = spp['family']
#         # currentSpecies = spp['genus'] + ' ' + spp['species']
#         currentFullname = spp['fullname']
#         currentURL = spp['url']
#         currentBarcode = spp['barcode']
#         currentHerb = spp['herbCode']
#         print(f"{bcolors.BOLD}Family: {currentFamily}{bcolors.ENDC}")
#         print(f"{bcolors.BOLD}      {currentFullname}{bcolors.ENDC}")
#         print(f"{bcolors.BOLD}      In Download List: {index} / {listMax}{bcolors.ENDC}")

#         imgFilename = [currentHerb, currentBarcode, currentFullname]
#         imgFilename = sep.join(imgFilename)
#         imgFilenameJPG = imgFilename + ".jpg"
#         print(f"{bcolors.BOLD}           URL: {currentURL}{bcolors.ENDC}")
#         try:
#             img = Image.open(session.get(currentURL, stream=True, timeout=1.0).raw)
#             imageFound, alreadyDownloaded, aggOcc, aggImg = save_matching_image_user_file(alreadyDownloaded,img,MP_Low,MP_High,dirNewImg,imgFilenameJPG)
#             print(f"{bcolors.OKGREEN}                SUCCESS{bcolors.ENDC}")
#         except Exception as e: 
#             print(f"{bcolors.WARNING}                SKIP No Connection or ERROR{bcolors.ENDC}")


#     aggOcc.to_csv(os.path.join(dir_home,aggOcc_filename),index=False)
#     aggImg.to_csv(os.path.join(dir_home,aggImg_filename),index=False)

#     return alreadyDownloaded, aggOcc, aggImg


# Return entire row of file_to_search that matches the gbif_id, else return []
def find_gbifID(gbif_id,file_to_search):
    row_found = file_to_search.loc[file_to_search['gbifID'].astype(str).str.match(str(gbif_id)),:]
    if row_found.empty:
        print(f"{bcolors.WARNING}      gbif_id: {gbif_id} not found in occurrences file{bcolors.ENDC}")
        row_found = None
    else:
        print(f"{bcolors.OKGREEN}      gbif_id: {gbif_id} successfully found in occurrences file{bcolors.ENDC}")
    return row_found

def validate_herb_code(occ_row):
    # print(occ_row)
    # Herbarium codes are not always in the correct column, we need to find the right one
    try:
        opts = [occ_row['institutionCode'],
            occ_row['institutionID'],
            occ_row['ownerInstitutionCode'],
            occ_row['collectionCode'],
            occ_row['publisher'],
            occ_row['occurrenceID']]
        opts = [item for item in opts if not(pd.isnull(item.values)) == True]
    except:
        opts = [str(occ_row['institutionCode']),
            str(occ_row['institutionID']),
            str(occ_row['ownerInstitutionCode']),
            str(occ_row['collectionCode']),
            str(occ_row['publisher']),
            str(occ_row['occurrenceID'])]
        opts = pd.DataFrame(opts)
        opts = opts.dropna()
        opts = opts.apply(lambda x: x[0]).tolist()

    opts_short = []

    for word in opts:
        #print(word)
        if len(word) <= 8:
            if word is not None:
                opts_short = opts_short + [word]

    if len(opts_short) == 0:
        try:
            herb_code = occ_row['publisher'].values[0].replace(" ","-")
        except:
            try:
                herb_code = occ_row['publisher'].replace(" ","-")
            except:
                herb_code = "ERROR"
    try:
        inst_ID = occ_row['institutionID'].values[0]
        occ_ID = occ_row['occurrenceID'].values[0]
    except:
        inst_ID = occ_row['institutionID']
        occ_ID = occ_row['occurrenceID']
    if inst_ID == "UBC Herbarium":
        herb_code = "UBC"
    elif inst_ID == "Naturalis Biodiversity Center":
        herb_code = "L"
    elif inst_ID == "Forest Herbarium Ibadan (FHI)":
        herb_code = "FHI"
    elif 'id.luomus.fi' in occ_ID:
        herb_code = "FinBIF"
    else:
        if len(opts_short) > 0:
            herb_code = opts_short[0]

    try:
        herb_code = herb_code.values[0]
    except:
        herb_code = herb_code

    # Specific cases that require manual overrides
    # If you see an herbarium DWC file with a similar error, add them here
    if herb_code == "Qarshi-Botanical-Garden,-Qarshi-Industries-Pvt.-Ltd,-Pakistan":
        herb_code = "Qarshi-Botanical-Garden"
    elif herb_code == "12650":
        herb_code = "SDSU"
    elif herb_code == "322":
        herb_code = "SDSU"
    elif herb_code == "GC-University,-Lahore":
        herb_code = "GC-University-Lahore"
    elif herb_code == "Institute-of-Biology-of-Komi-Scientific-Centre-of-the-Ural-Branch-of-the-Russian-Academy-of-Sciences":
        herb_code = "Komi-Scientific-Centre"
    
    return herb_code

def remove_illegal_chars(text):
    cleaned = re.sub(r"[^a-zA-Z0-9_-]","",text)
    return cleaned

def keep_first_word(text):
    if (' ' in text) == True:
        cleaned = text.split(' ')[0]
    else:
        cleaned = text
    return cleaned

# Create a filename for the downloaded image
# In the case sensitive format:
#        HERBARIUM_barcode_Family_Genus_species.jpg
def generate_image_filename(occ_row):
    herb_code = remove_illegal_chars(validate_herb_code(occ_row))
    try:
        specimen_id = str(occ_row['gbifID'].values[0])
        family = remove_illegal_chars(occ_row['family'].values[0])
        genus = remove_illegal_chars(occ_row['genus'].values[0])
        species = remove_illegal_chars(keep_first_word(occ_row['specificEpithet'].values[0]))
    except:
        specimen_id = str(occ_row['gbifID'])
        family = remove_illegal_chars(occ_row['family'])
        genus = remove_illegal_chars(occ_row['genus'])
        species = remove_illegal_chars(keep_first_word(occ_row['specificEpithet']))
    fullname = '_'.join([family, genus, species])

    filename_image = '_'.join([herb_code, specimen_id, fullname])
    filename_image_jpg = '.'.join([filename_image, 'jpg'])

    return filename_image, filename_image_jpg, herb_code, specimen_id, family, genus, species, fullname

def read_DWC_file(cfg):
    dir_home = cfg['dir_home']
    filename_occ = cfg['filename_occ']
    filename_img = cfg['filename_img']
    # read the images.csv or occurences.csv file. can be txt ro csv
    occ_df = ingest_DWC(filename_occ,dir_home)
    images_df = ingest_DWC(filename_img,dir_home)
    return occ_df, images_df

def read_DWC_file_multiDirs(cfg, dir_sub):
    filename_occ = cfg['filename_occ']
    filename_img = cfg['filename_img']
    # read the images.csv or occurences.csv file. can be txt ro csv
    occ_df = ingest_DWC(filename_occ,dir_sub)
    images_df = ingest_DWC(filename_img,dir_sub)
    return occ_df, images_df

def ingest_DWC(DWC_csv_or_txt_file,dir_home):
    if DWC_csv_or_txt_file.split('.')[1] == 'txt':
        df = pd.read_csv(os.path.join(dir_home,DWC_csv_or_txt_file), sep="\t",header=0, low_memory=False, dtype=str)
    elif DWC_csv_or_txt_file.split('.')[1] == 'csv':
        df = pd.read_csv(os.path.join(dir_home,DWC_csv_or_txt_file), sep=",",header=0, low_memory=False, dtype=str)
    else:
        print(f"{bcolors.FAIL}DWC file {DWC_csv_or_txt_file} is not '.txt' or '.csv' and was not opened{bcolors.ENDC}")
    return df
    
'''
#######################################################################
Main function for the config_download_from_GBIF_all_images_in_file.yml
see yml for details
#######################################################################
'''
def download_all_images_in_images_csv_multiDirs(cfg):
    dir_destination_parent = cfg['dir_destination_images']
    dir_destination_csv = cfg['dir_destination_csv']
    # (dirWishlists,dirNewImg,alreadyDownloaded,MP_Low,MP_High,aggOcc_filename,aggImg_filename):
    

    # Get DWC files
    for dir_DWC, dirs_sub, __ in os.walk(cfg['dir_home']):
        for dir_sub in dirs_sub:
            dir_home = os.path.join(dir_DWC, dir_sub)
            dir_destination = os.path.join(dir_destination_parent, dir_sub)

            validate_dir(dir_destination)
            validate_dir(dir_destination_csv)

            occ_df, images_df = read_DWC_file_multiDirs(cfg, dir_home)

            # Report summary
            print(f"{bcolors.BOLD}Beginning of images file:{bcolors.ENDC}")
            print(images_df.head())
            print(f"{bcolors.BOLD}Beginning of occurrence file:{bcolors.ENDC}")
            print(occ_df.head())

            # Ignore problematic Herbaria
            if cfg['ignore_banned_herb']:
                for banned_url in cfg['banned_url_stems']:
                    images_df = images_df[~images_df['identifier'].str.contains(banned_url, na=False)]
            
            # Report summary
            n_imgs = images_df.shape[0]
            n_occ = occ_df.shape[0]
            print(f"{bcolors.BOLD}Number of images in images file: {n_imgs}{bcolors.ENDC}")
            print(f"{bcolors.BOLD}Number of occurrence to search through: {n_occ}{bcolors.ENDC}")

            results = process_image_batch_multiDirs(cfg, images_df, occ_df, dir_destination)


def download_all_images_in_images_csv(cfg):
    dir_destination = cfg['dir_destination_images']
    dir_destination_csv = cfg['dir_destination_csv']
    # (dirWishlists,dirNewImg,alreadyDownloaded,MP_Low,MP_High,aggOcc_filename,aggImg_filename):
    validate_dir(dir_destination)
    validate_dir(dir_destination_csv)

    # Get DWC files
    occ_df, images_df = read_DWC_file(cfg)

    # Report summary
    print(f"{bcolors.BOLD}Beginning of images file:{bcolors.ENDC}")
    print(images_df.head())
    print(f"{bcolors.BOLD}Beginning of occurrence file:{bcolors.ENDC}")
    print(occ_df.head())

    # Ignore problematic Herbaria
    if cfg['ignore_banned_herb']:
        for banned_url in cfg['banned_url_stems']:
            images_df = images_df[~images_df['identifier'].str.contains(banned_url, na=False)]
    
    # Report summary
    n_imgs = images_df.shape[0]
    n_occ = occ_df.shape[0]
    print(f"{bcolors.BOLD}Number of images in images file: {n_imgs}{bcolors.ENDC}")
    print(f"{bcolors.BOLD}Number of occurrence to search through: {n_occ}{bcolors.ENDC}")

    results = process_image_batch(cfg, images_df, occ_df)

def process_image_batch(cfg, images_df, occ_df):
    futures_list = []
    results = []

    # single threaded, useful for debugging
    # for index, image_row in images_df.iterrows():
    #     futures = process_each_image_row( cfg, image_row, occ_df)
    #     futures_list.append(futures)
    # for future in futures_list:
    #     try:
    #         result = future.result(timeout=60)
    #         results.append(result)
    #     except Exception:
    #         results.append(None)
    lock = Lock() 

    with th(max_workers=13) as executor:
        for index, image_row in images_df.iterrows():
            futures = executor.submit(process_each_image_row, cfg, image_row, occ_df, lock)
            futures_list.append(futures)

        for future in futures_list:
            try:
                result = future.result(timeout=60)
                results.append(result)
            except Exception:
                results.append(None)
    return results

def process_image_batch_multiDirs(cfg, images_df, occ_df, dir_destination):
    futures_list = []
    results = []

    lock = Lock() 

    with th(max_workers=13) as executor:
        for index, image_row in images_df.iterrows():
            futures = executor.submit(process_each_image_row_multiDirs, cfg, image_row, occ_df, dir_destination, lock)
            futures_list.append(futures)

        for future in futures_list:
            try:
                result = future.result(timeout=60)
                results.append(result)
            except Exception:
                results.append(None)
    return results


def process_each_image_row(cfg, image_row, occ_df, lock):
    print(f"{bcolors.BOLD}Working on image: {image_row['gbifID']}{bcolors.ENDC}")
    gbif_id = image_row['gbifID']
    gbif_url = image_row['identifier']

    occ_row = find_gbifID(gbif_id,occ_df)

    if occ_row is not None:
        ImageInfo = ImageCandidate(cfg, image_row, occ_row, gbif_url, lock)
        # ImageInfo.download_image(cfg, occ_row, image_row)
    else:
        pass

def process_each_image_row_multiDirs(cfg, image_row, occ_df, dir_destination, lock):
    print(f"{bcolors.BOLD}Working on image: {image_row['gbifID']}{bcolors.ENDC}")
    gbif_id = image_row['gbifID']
    gbif_url = image_row['identifier']

    occ_row = find_gbifID(gbif_id,occ_df)

    if occ_row is not None:
        ImageInfo = ImageCandidateMulti(cfg, image_row, occ_row, gbif_url, dir_destination, lock)
    else:
        pass