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
import threading
from queue import Queue
from threading import Thread, Lock
from random import shuffle
from collections import defaultdict
import streamlit
import random
from tqdm import tqdm
from bs4 import BeautifulSoup



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
        # self.headers_occ =  list(occ_row.columns.values)
        # self.headers_img = list(image_row.columns.values)
        self.headers_occ = occ_row
        self.headers_img = image_row
        self.occ_row = occ_row # pd.DataFrame(data=occ_row,columns=self.headers_occ)
        self.image_row = image_row # pd.DataFrame(data=image_row,columns=self.headers_img)
        self.url = url
        self.cfg = cfg

        self.filename_image, self.filename_image_jpg, self.herb_code, self.specimen_id, self.family, self.genus, self.species, self.fullname = generate_image_filename(occ_row)
        self.download_image(lock)

    # def download_image(self, lock) -> None:
    #     dir_destination = self.cfg['dir_destination_images']
    #     MP_low = self.cfg['MP_low']
    #     MP_high = self.cfg['MP_high']
    #     # Define URL get parameters
    #     sep = '_'
    #     session = requests.Session()
    #     retry = Retry(connect=1) #2, backoff_factor=0.5)
    #     adapter = HTTPAdapter(max_retries=retry)
    #     session.mount('http://', adapter)
    #     session.mount('https://', adapter)

    #     print(f"{bcolors.BOLD}      {self.fullname}{bcolors.ENDC}")
    #     print(f"{bcolors.BOLD}           URL: {self.url}{bcolors.ENDC}")
    #     try:
    #         response = session.get(self.url, stream=True, timeout=1.0)
    #         img = Image.open(response.raw)
    #         self._save_matching_image(img, MP_low, MP_high, dir_destination, lock)
    #         print(f"{bcolors.OKGREEN}                SUCCESS{bcolors.ENDC}")
    #     except Exception as e: 
    #         print(f"{bcolors.FAIL}                SKIP No Connection or ERROR --> {e}{bcolors.ENDC}")
    #         print(f"{bcolors.WARNING}                Status Code --> {response.status_code}{bcolors.ENDC}")
    #         print(f"{bcolors.WARNING}                Reason --> {response.reason}{bcolors.ENDC}")
    
    # Worked fine, tried the SLTP version
    # def download_image(self, lock) -> None:
    #     dir_destination = self.cfg['dir_destination_images']
    #     MP_low = self.cfg['MP_low']
    #     MP_high = self.cfg['MP_high']
        
    #     # Set up a session with retry strategy
    #     session = requests.Session()
    #     retries = Retry(connect=3, backoff_factor=1)
    #     adapter = HTTPAdapter(max_retries=retries)
    #     session.mount('http://', adapter)
    #     session.mount('https://', adapter)

    #     print(f"{bcolors.BOLD}      {self.fullname}{bcolors.ENDC}")
    #     print(f"{bcolors.BOLD}           URL: {self.url}{bcolors.ENDC}")

    #     try:
    #         response = session.get(self.url, stream=True, timeout=1.0)
    #         response.raise_for_status()  # Check for HTTP errors

    #         # Check if the content-type of the response is an image
    #         if 'image' in response.headers['Content-Type']:
    #             img = Image.open(response.raw)
    #             self._save_matching_image(img, MP_low, MP_high, dir_destination, lock)
    #             print(f"{bcolors.OKGREEN}                SUCCESS{bcolors.ENDC}")
    #         else:
    #             # The content is not an image, use BeautifulSoup to find the image link
    #             soup = BeautifulSoup(response.content, 'html.parser')
    #             image_link = soup.find('a', text='Open Large Image')
    #             if image_link and 'href' in image_link.attrs:
    #                 image_url = image_link['href']
    #                 # Ensure the URL is absolute
    #                 image_url = requests.compat.urljoin(response.url, image_url)
    #                 # Fetch and save the image
    #                 image_response = session.get(image_url, stream=True)
    #                 image_response.raise_for_status()
    #                 img = Image.open(image_response.raw)
    #                 self._save_matching_image(img, MP_low, MP_high, dir_destination, lock)
    #                 print(f"{bcolors.OKGREEN}                SUCCESS{bcolors.ENDC}")

    #     except requests.exceptions.HTTPError as http_err:
    #         print(f"{bcolors.FAIL}                HTTP Error --> {http_err}{bcolors.ENDC}")

    #     except requests.exceptions.ConnectionError as conn_err:
    #         # Handle connection-related errors, ignore if you don't want to print them
    #         print(f"{bcolors.FAIL}                HTTP Error --> {http_err}{bcolors.ENDC}")
    #         pass

    #     except Exception as e:
    #         # This will ignore the "No active exception to reraise" error
    #         if str(e) != "No active exception to reraise":
    #             print(f"{bcolors.FAIL}                SKIP --- No Connection or Rate Limited --> {e}{bcolors.ENDC}")

    #     finally:
    #         # Randomized delay
    #         time.sleep(1 + random.uniform(0, 1))
    def download_image(self, lock) -> None:
        import logging
        import http.client as http_client
        import requests, time, random, certifi


        http_client.HTTPConnection.debuglevel = 1

        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)
        requests_log = logging.getLogger("requests.packages.urllib3")
        requests_log.setLevel(logging.DEBUG)
        requests_log.propagate = True



        dir_destination = self.cfg['dir_destination_images']
        MP_low = self.cfg['MP_low']
        MP_high = self.cfg['MP_high']
        
        # Set up a session with retry strategy
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        session.headers.update(headers)
        session.verify = certifi.where()
        retries = Retry(connect=3, backoff_factor=1)
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        print(f"{bcolors.BOLD}      {self.fullname}{bcolors.ENDC}")
        print(f"{bcolors.BOLD}           URL: {self.url}{bcolors.ENDC}")
        
        

        try:
            response = session.get(self.url, stream=True, timeout=5.0, verify=False)
            response.raise_for_status()  # Check for HTTP errors

            img = Image.open(response.raw)

            was_saved = self._save_matching_image(img, MP_low, MP_high, dir_destination, lock)  # TODO make this occ + img code work for MICH *and* GBIF, right now they are seperate 
            
            if not was_saved:
                raise ImageSaveError(f"Failed to save the image: {self.url}")

            print(f"{bcolors.OKGREEN}                SUCCESS{bcolors.ENDC}")
            self.download_success = True

        except ImageSaveError as e:
            print(f"{bcolors.FAIL}                {e}{bcolors.ENDC}")

        except requests.exceptions.HTTPError as http_err:
            print(f"{bcolors.FAIL}                HTTP Error --> {http_err}{bcolors.ENDC}")

        except requests.exceptions.ConnectionError as conn_err:
            # Handle connection-related errors, ignore if you don't want to print them
            pass

        except Exception as e:
            # This will ignore the "No active exception to reraise" error
            if str(e) != "No active exception to reraise":
                print(f"{bcolors.FAIL}                SKIP --- No Connection or Rate Limited --> {e}{bcolors.ENDC}")

        finally:
            # Randomized delay
            time.sleep(2 + random.uniform(0, 2))

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
        if 'gbifID' in self.image_row.columns:
            id_column = 'gbifID'
            self.image_row = self.image_row.rename(columns={id_column: 'gbifID_images'}) 
        # If 'gbifID' is not a key, check if 'id' is a key
        elif 'id' in self.image_row.columns:
            id_column = 'id'
            self.image_row = self.image_row.rename(columns={id_column: 'id_images'}) 
        else:
            raise

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


def create_subset_file(cfg):
    # Read the original files
    occ_df, images_df = read_DWC_file(cfg)

    # Generate 'fullname' for each row
    occ_df['fullname'] = occ_df.apply(lambda row: generate_image_filename2(row)[7], axis=1)

    # Initialize a dictionary to keep track of counts for each fullname
    fullname_counts = {}

    # Prepare DataFrame for the subset
    subset_rows = []

    # Process chunks based on unique values in 'specificEpithet'
    for fullname, group_df in tqdm(occ_df.groupby('fullname'), desc="Processing fullnames"):
        # Shuffle the group DataFrame
        shuffled_group_df = group_df.sample(frac=1, random_state=2023).reset_index(drop=True)

        for _, row in tqdm(shuffled_group_df.iterrows(), total=shuffled_group_df.shape[0], desc=f"Processing rows for {fullname}", leave=False):
            # Check if fullname has reached the limit of 20
            if fullname_counts.get(fullname, 0) < 10:
                subset_rows.append(row)
                fullname_counts[fullname] = fullname_counts.get(fullname, 0) + 1

    # Convert subset rows to a DataFrame
    subset_df = pd.DataFrame(subset_rows)

    # Define the new filename
    original_filename = cfg['filename_occ']
    base, extension = os.path.splitext(original_filename)
    new_filename = f"{base}_subset{extension}"

    # Write to a new CSV file
    subset_df.to_csv(os.path.join(cfg['dir_home'], new_filename), sep='\t', index=False)

    return new_filename

class ImageSaveError(Exception):
    """Custom exception for image saving errors."""
    pass

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

    download_success: bool = False


    def __init__(self, cfg, image_row, occ_row, url, dir_destination, lock):
        # Convert the Series to a DataFrame with one row
        try:
            # Now, you can access columns and data as you would in a DataFrame
            self.headers_occ = occ_row
            self.headers_img = image_row
        except Exception as e:
            print(f"Exception occurred: {e}")

        
        self.occ_row = occ_row # pd.DataFrame(data=occ_row,columns=self.headers_occ)
        self.image_row = image_row # pd.DataFrame(data=image_row,columns=self.headers_img)
        self.url = url
        self.cfg = cfg

        self.filename_image, self.filename_image_jpg, self.herb_code, self.specimen_id, self.family, self.genus, self.species, self.fullname = generate_image_filename2(occ_row)

        self.download_success = self.download_image(dir_destination, lock)



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
            return True
        except Exception as e: 
            print(f"{bcolors.FAIL}                SKIP No Connection or ERROR --> {e}{bcolors.ENDC}")
            print(f"{bcolors.WARNING}                Status Code --> {response.status_code}{bcolors.ENDC}")
            print(f"{bcolors.WARNING}                Reasone --> {response.reason}{bcolors.ENDC}")
            return False

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
        
        if 'gbifID' in self.image_row.columns:
            id_column = 'gbifID'
            self.image_row = self.image_row.rename(columns={id_column: 'gbifID_images'}) 
        # If 'gbifID' is not a key, check if 'id' is a key
        elif 'id' in self.image_row.columns:
            id_column = 'id'
            self.image_row = self.image_row.rename(columns={id_column: 'id_images'}) 
        else:
            raise


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

class SharedCounter:
    def __init__(self):
        self.img_count_dict = {}
        self.lock = Lock()
    
    def increment(self, key, value=1):
        with self.lock:
            self.img_count_dict[key] = self.img_count_dict.get(key, 0) + value

    def get_count(self, key):
        with self.lock:
            return self.img_count_dict.get(key, 0)



@dataclass
class ImageCandidateCustom:
    cfg: str = ''
    # herb_code: str = '' 
    # specimen_id: str = ''
    # family: str = ''
    # genus: str = ''
    # species: str = ''
    fullname: str = ''

    filename_image: str = ''
    filename_image_jpg: str = ''

    url: str = ''
    # headers_occ: str = ''
    headers_img: str = ''

    # occ_row: list = field(init=False,default_factory=None)
    image_row: list = field(init=False,default_factory=None)


    def __init__(self, cfg, image_row, url, col_name, lock):
        # self.headers_occ =  list(occ_row.columns.values)
        # self.headers_img = list(image_row.columns.values)
        self.image_row = image_row # pd.DataFrame(data=image_row,columns=self.headers_img)

        self.url = url
        self.cfg = cfg
        self.col_name = col_name

        self.fullname = image_row[col_name]
        self.filename_image = image_row[col_name]
        self.filename_image_jpg = ''.join([image_row[col_name], '.jpg'])
        
        self.download_image(lock)

    # def download_image(self, lock) -> None:
    #     dir_destination = self.cfg['dir_destination_images']
    #     MP_low = self.cfg['MP_low']
    #     MP_high = self.cfg['MP_high']
    #     # Define URL get parameters
    #     sep = '_'
    #     session = requests.Session()
    #     retry = Retry(connect=1) #2, backoff_factor=0.5)
    #     adapter = HTTPAdapter(max_retries=retry)
    #     session.mount('http://', adapter)
    #     session.mount('https://', adapter)

    #     print(f"{bcolors.BOLD}      {self.fullname}{bcolors.ENDC}")
    #     print(f"{bcolors.BOLD}           URL: {self.url}{bcolors.ENDC}")
    #     try:
    #         response = session.get(self.url, stream=True, timeout=1.0)
    #         img = Image.open(response.raw)
    #         self._save_matching_image(img, MP_low, MP_high, dir_destination, lock)
    #         print(f"{bcolors.OKGREEN}                SUCCESS{bcolors.ENDC}")
    #     except Exception as e: 
    #         print(f"{bcolors.FAIL}                SKIP No Connection or ERROR --> {e}{bcolors.ENDC}")
    #         print(f"{bcolors.WARNING}                Status Code --> {response.status_code}{bcolors.ENDC}")
    #         print(f"{bcolors.WARNING}                Reasone --> {response.reason}{bcolors.ENDC}")
    def download_image(self, lock) -> None:
        dir_destination = self.cfg['dir_destination_images']
        MP_low = self.cfg['MP_low']
        MP_high = self.cfg['MP_high']
        
        # Set up a session with retry strategy
        session = requests.Session()
        retries = Retry(connect=3, backoff_factor=1)
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        print(f"{bcolors.BOLD}      {self.fullname}{bcolors.ENDC}")
        print(f"{bcolors.BOLD}           URL: {self.url}{bcolors.ENDC}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'DNT': '1',  # Do Not Track Request Header
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-User': '?1',
            'Sec-Fetch-Dest': 'document'
        }


        try:
            response = session.get(self.url, headers=headers, stream=True, timeout=1.0, verify=False)
            response.raise_for_status()  # Check for HTTP errors

            img = Image.open(response.raw)

            self._save_matching_image(img, MP_low, MP_high, dir_destination, lock)  # TODO make this occ + img code work for MICH *and* GBIF, right now they are seperate 
            print(f"{bcolors.OKGREEN}                SUCCESS{bcolors.ENDC}")

            self.download_success = True


        except requests.exceptions.HTTPError as http_err:
            print(f"{bcolors.FAIL}                HTTP Error --> {http_err}{bcolors.ENDC}")

        except requests.exceptions.ConnectionError as conn_err:
            # Handle connection-related errors, ignore if you don't want to print them
            pass

        except Exception as e:
            # This will ignore the "No active exception to reraise" error
            if str(e) != "No active exception to reraise":
                print(f"{bcolors.FAIL}                SKIP --- No Connection or Rate Limited --> {e}{bcolors.ENDC}")

        finally:
            # Randomized delay
            time.sleep(2 + random.uniform(0, 2))

    def _save_matching_image(self, img, MP_low, MP_high, dir_destination, lock) -> None:
        img_mp, img_w, img_h = check_image_size(img)
        if img_mp < MP_low:
            print(f"{bcolors.WARNING}                SKIP < {MP_low}MP: {img_mp}{bcolors.ENDC}")

        elif MP_low <= img_mp <= MP_high:
            image_path = os.path.join(dir_destination,self.filename_image_jpg)
            img.save(image_path)

            print(f"{bcolors.OKGREEN}                Regular MP: {img_mp}{bcolors.ENDC}")
            print(f"{bcolors.OKGREEN}                Image Saved: {image_path}{bcolors.ENDC}")

        elif img_mp > MP_high:
            if self.cfg['do_resize']:
                [img_w, img_h] = calc_resize(img_w, img_h)
                newsize = (img_w, img_h)
                img = img.resize(newsize)
                image_path = os.path.join(dir_destination,self.filename_image_jpg)
                img.save(image_path)

                print(f"{bcolors.OKGREEN}                {MP_high}MP+ Resize: {img_mp}{bcolors.ENDC}")
                print(f"{bcolors.OKGREEN}                Image Saved: {image_path}{bcolors.ENDC}")
            else:
                print(f"{bcolors.OKCYAN}                {MP_high}MP+ Resize: {img_mp}{bcolors.ENDC}")
                print(f"{bcolors.OKCYAN}                SKIP: {image_path}{bcolors.ENDC}")


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
    # Check if 'gbifID' is a key in the DataFrame
    if 'gbifID' in file_to_search.columns:
        row_found = file_to_search.loc[file_to_search['gbifID'].astype(str).str.match(str(gbif_id)), :]
        
    # If 'gbifID' is not a key, check if 'id' is a key
    elif 'id' in file_to_search.columns:
        row_found = file_to_search.loc[file_to_search['id'].astype(str).str.match(str(gbif_id)), :]
    elif 'coreid' in file_to_search.columns:
        row_found = file_to_search.loc[file_to_search['coreid'].astype(str).str.match(str(gbif_id)), :]
    # If neither 'gbifID' nor 'id' is a key, raise an error
    else:
        raise KeyError("Neither 'gbifID' nor 'id' found in the column names for the Occurrences file")


    if row_found.empty:
        print(f"{bcolors.WARNING}      gbif_id: {gbif_id} not found in occurrences file{bcolors.ENDC}")
        return None
    else:
        print(f"{bcolors.OKGREEN}      gbif_id: {gbif_id} successfully found in occurrences file{bcolors.ENDC}")
        return row_found

def validate_herb_code(occ_row):
    possible_keys = ['institutionCode', 'institutionID', 'ownerInstitutionCode', 
                    'collectionCode', 'publisher', 'occurrenceID']
    # print(occ_row)
    # Herbarium codes are not always in the correct column, we need to find the right one
    # try:
    #     opts = [occ_row['institutionCode'],
    #         occ_row['institutionID'],
    #         occ_row['ownerInstitutionCode'],
    #         occ_row['collectionCode'],
    #         occ_row['publisher'],
    #         occ_row['occurrenceID']]
    #     opts = [item for item in opts if not(pd.isnull(item.values)) == True]
    # except:
    #     opts = [str(occ_row[key]) for key in possible_keys if key in occ_row and not pd.isnull(occ_row[key])]  ######### TODO see if this should be the default
    #     opts = pd.DataFrame(opts)
    #     opts = opts.dropna()
    #     opts = opts.apply(lambda x: x[0]).tolist()
    opts = []
    for key in possible_keys:
        if key in occ_row:
            value = occ_row[key]
            if isinstance(value, pd.Series):
                # Iterate through each element in the Series
                for item in value:
                    if pd.notnull(item) and isinstance(item, str):
                        opts.append(item)
            else:
                # Handle the case where value is not a Series
                if pd.notnull(value) and isinstance(value, str):
                    opts.append(value)

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
        try:
            inst_ID = occ_row['institutionID']
            occ_ID = occ_row['occurrenceID']
        
            occ_ID = str(occ_row['occID']) if 'occID' in occ_row and pd.notna(occ_row['occID']) else "" ############## new NOTE
        except:
            inst_ID = ''
            occ_ID = occ_row['occurrenceID']
        
            occ_ID = str(occ_row['occID']) if 'occID' in occ_row and pd.notna(occ_row['occID']) else "" ############## new NOTE

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
    if 'gbifID' in occ_row:
        id_column = 'gbifID'
    # If 'gbifID' is not a key, check if 'id' is a key
    elif 'id' in occ_row:
        id_column = 'id'
    elif 'coreid' in occ_row:
        id_column = 'coreid'
    else:
        raise

    herb_code = remove_illegal_chars(validate_herb_code(occ_row))
    try:
        specimen_id = str(occ_row[id_column].values[0])
        family = remove_illegal_chars(occ_row['family'].values[0])
        genus = remove_illegal_chars(occ_row['genus'].values[0])
        species = remove_illegal_chars(keep_first_word(occ_row['specificEpithet'].values[0]))
    except:
        specimen_id = str(occ_row[id_column])
        family = remove_illegal_chars(occ_row['family'])
        genus = remove_illegal_chars(occ_row['genus'])
        species = remove_illegal_chars(keep_first_word(occ_row['specificEpithet']))
    fullname = '_'.join([family, genus, species])

    filename_image = '_'.join([herb_code, specimen_id, fullname])
    filename_image_jpg = '.'.join([filename_image, 'jpg'])

    return filename_image, filename_image_jpg, herb_code, specimen_id, family, genus, species, fullname

def generate_image_filename2(occ_row):
    if 'gbifID' in occ_row:
        id_column = 'gbifID'
    # If 'gbifID' is not a key, check if 'id' is a key
    elif 'id' in occ_row:
        id_column = 'id'
    elif 'coreid' in occ_row:
        id_column = 'coreid'
    else:
        raise

    herb_code = remove_illegal_chars(validate_herb_code(occ_row))
    try:
        # Assuming gbifID is a string, no need for .values[0]
        specimen_id = str(occ_row[id_column])
        family = remove_illegal_chars(occ_row['family'])
        genus = remove_illegal_chars(occ_row['genus'])
        # Convert to string in case of float and use keep_first_word
        specificEpithet = str(occ_row['specificEpithet']) if pd.notna(occ_row['specificEpithet']) else ""
        species = remove_illegal_chars(keep_first_word(specificEpithet))
    except Exception as e:
        # Handle exceptions or log errors as needed
        print(f"Error processing row: {e}")
        # Set default values or handle the error as appropriate
        specimen_id, family, genus, species = "", "", "", ""
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
    file_path = os.path.join(dir_home, DWC_csv_or_txt_file)
    file_extension = DWC_csv_or_txt_file.split('.')[1]

    try:
        if file_extension == 'txt':
            df = pd.read_csv(file_path, sep="\t", header=0, low_memory=False, dtype=str)
        elif file_extension == 'csv':
            # Attempt to read with comma separator
            try:
                df = pd.read_csv(file_path, sep=",", header=0, low_memory=False, dtype=str)
            except pd.errors.ParserError:
                try:
                    # If failed, try with a different separator, e.g., semicolon
                    df = pd.read_csv(file_path, sep="\t", header=0, low_memory=False, dtype=str)
                except:
                    try:
                        df = pd.read_csv(file_path, sep="|", header=0, low_memory=False, dtype=str)
                    except:
                        df = pd.read_csv(file_path, sep=";", header=0, low_memory=False, dtype=str)
        else:
            print(f"{bcolors.FAIL}DWC file {DWC_csv_or_txt_file} is not '.txt' or '.csv' and was not opened{bcolors.ENDC}")
            return None
    except Exception as e:
        print(f"Error while reading file: {e}")
        return None

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
    n_already_downloaded = cfg['n_already_downloaded']
    n_max_to_download = cfg['n_max_to_download']
    n_imgs_per_species = cfg['n_imgs_per_species']
    MP_low = cfg['MP_low']
    MP_high = cfg['MP_high']
    do_shuffle_occurrences = cfg['do_shuffle_occurrences']

    shared_counter = SharedCounter() 

    # (dirWishlists,dirNewImg,alreadyDownloaded,MP_Low,MP_High,aggOcc_filename,aggImg_filename):
    

    # Get DWC files
    for dir_DWC, dirs_sub, __ in os.walk(cfg['dir_home']):
        for dir_sub in dirs_sub:
            dir_home = os.path.join(dir_DWC, dir_sub)
            dir_destination = os.path.join(dir_destination_parent, dir_sub)

            validate_dir(dir_destination)
            validate_dir(dir_destination_csv)

            occ_df, images_df = read_DWC_file_multiDirs(cfg, dir_home)

            # Shuffle the order of the occurrences DataFrame if the flag is set
            if do_shuffle_occurrences:
                occ_df = occ_df.sample(frac=1).reset_index(drop=True)

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

            results = process_image_batch_multiDirs(cfg, images_df, occ_df, dir_destination, shared_counter, n_imgs_per_species, do_shuffle_occurrences)


def download_all_images_in_images_csv(cfg):
    dir_destination = cfg['dir_destination_images']
    dir_destination_csv = cfg['dir_destination_csv']

    # (dirWishlists,dirNewImg,alreadyDownloaded,MP_Low,MP_High,aggOcc_filename,aggImg_filename):
    validate_dir(dir_destination)
    validate_dir(dir_destination_csv)
    
    if cfg['is_custom_file']:
        download_from_custom_file(cfg)
    else:
        # Get DWC files
        occ_df, images_df = read_DWC_file(cfg)

        if occ_df is not None:

            # Report summary
            print(f"{bcolors.BOLD}Beginning of images file:{bcolors.ENDC}")
            print(images_df.head())
            print(f"{bcolors.BOLD}Beginning of occurrence file:{bcolors.ENDC}")
            print(occ_df.head())

            # Ignore problematic Herbaria
            if cfg['ignore_banned_herb']:
                for banned_url in cfg['banned_url_stems']:
                    images_df = images_df[~images_df['identifier'].str.contains(banned_url, na=False)]


            if not cfg['is_custom_file']:
                ### TODO NEW, needs to match the gbif version
                # Find common 'id' values in both dataframes
                # common_ids = set(occ_df['id']).intersection(set(images_df['coreid']))
                # common_ids = set(occ_df['gbifID']).intersection(set(images_df['gbifID']))

                # Filter both dataframes to keep only rows with these common 'id' values
                # occ_df_filtered = occ_df[occ_df['id'].isin(common_ids)]
                # images_df = images_df[images_df['coreid'].isin(common_ids)]
                # images_df = images_df[images_df['gbifID'].isin(common_ids)]

                # Ensure the IDs are of the same type, generally string is safer if IDs are alphanumeric
                occ_df['gbifID'] = occ_df['gbifID'].astype(str)
                images_df['gbifID'] = images_df['gbifID'].astype(str)

                # Filter images DataFrame based on occurrence DataFrame's IDs
                images_df = images_df[images_df['gbifID'].isin(occ_df['gbifID'])]
            
            # Report summary
            n_imgs = images_df.shape[0]
            n_occ = occ_df.shape[0]
            print(f"{bcolors.BOLD}Number of images in images file: {n_imgs}{bcolors.ENDC}")
            print(f"{bcolors.BOLD}Number of occurrence to search through: {n_occ}{bcolors.ENDC}")

            results = process_image_batch(cfg, images_df, occ_df)
        else:
            pass

# def process_image_batch(cfg, images_df, occ_df):
#     futures_list = []
#     results = []
#     lock = Lock() 

#     # single threaded, useful for debugging
#     # for index, image_row in images_df.iterrows():
#     #     futures = process_each_image_row( cfg, image_row, occ_df, lock)
#     #     futures_list.append(futures)
#     # for future in futures_list:
#     #     try:
#     #         result = future.result(timeout=60)
#     #         results.append(result)
#     #     except Exception:
#     #         results.append(None)


#     with th(max_workers=cfg['n_threads']) as executor:
#         for index, image_row in images_df.iterrows():
#             futures = executor.submit(process_each_image_row, cfg, image_row, occ_df, lock)
#             futures_list.append(futures)

#         for future in futures_list:
#             try:
#                 result = future.result(timeout=60)
#                 results.append(result)
#             except Exception:
#                 results.append(None)
#     return results
        
def worker_download_standard(queue, cfg, occ_df, results, lock):
    while True:
        image_row = queue.get()
        if image_row is None:
            break  # None is the signal to stop processing
        try:
            result = process_each_image_row(cfg, image_row, occ_df, lock)
            results.append(result)
        except Exception as e:
            print(f"Error processing image: {e}")
            results.append(None)
        queue.task_done()

def process_image_batch(cfg, images_df, occ_df):
    num_workers = cfg['n_threads']
    queue = Queue()
    results = []
    lock = Lock()

    # Start worker threads
    threads = []
    for _ in range(num_workers):
        t = Thread(target=worker_download_standard, args=(queue, cfg, occ_df, results, lock))
        t.start()
        threads.append(t)

    # Enqueue tasks
    for index, image_row in images_df.iterrows():
        queue.put(image_row)

    # Block until all tasks are done
    queue.join()

    # Stop workers
    for _ in range(num_workers):
        queue.put(None)  # Send as many None as the number of workers to stop them
    for t in threads:
        t.join()

    return results

def process_each_image_row(cfg, image_row, occ_df, lock):
    id_column = next((col for col in ['gbifID', 'id', 'coreid'] if col in image_row), None)
    if id_column is None:
        raise KeyError("No valid ID column found in image row.")
    
    print(f"Working on image: {image_row[id_column]}")
    gbif_id = image_row[id_column]
    if cfg['is_custom_file']:
        gbif_url = image_row[cfg['custom_url_column_name']]
    else:
        gbif_url = image_row['identifier']

    occ_row = find_gbifID(gbif_id, occ_df)

    if occ_row is not None:
        ImageInfo = ImageCandidate(cfg, image_row, occ_row, gbif_url, lock)
        return pd.DataFrame(occ_row)
    else:
        pass


def process_image_batch_multiDirs(cfg, images_df, occ_df, dir_destination, shared_counter, n_imgs_per_species, do_shuffle_occurrences):
    futures_list = []
    results = []

    lock = Lock()

    if do_shuffle_occurrences:
        images_df = images_df.sample(frac=1).reset_index(drop=True)

    # Partition occ_df based on the first word of the 'specificEpithet' column
    partition_dict = defaultdict(list)
    for index, row in occ_df.iterrows():
        first_word = row['specificEpithet']  # Assuming keep_first_word is defined
        partition_dict[first_word].append(row)

    # Convert lists to DataFrames
    for key in partition_dict.keys():
        partition_dict[key] = pd.DataFrame(partition_dict[key])

    num_workers = 13

    with th(max_workers=num_workers) as executor:
        for specific_epithet, partition in partition_dict.items():
            future = executor.submit(process_occ_chunk_multiDirs, cfg, images_df, partition, dir_destination, shared_counter, n_imgs_per_species, do_shuffle_occurrences, lock)
            futures_list.append(future)

        for future in futures_list:
            try:
                result = future.result(timeout=60)
                results.append(result)
            except Exception:
                results.append(None)
    return results

def process_occ_chunk_multiDirs(cfg, images_df, occ_chunk, dir_destination, shared_counter, n_imgs_per_species, do_shuffle_occurrences, lock):
    results = []
    for index, occ_row in occ_chunk.iterrows():
        result = process_each_occ_row_multiDirs(cfg, images_df, occ_row, dir_destination, shared_counter, n_imgs_per_species, do_shuffle_occurrences, lock)
        results.append(result)
    return results

def process_each_occ_row_multiDirs(cfg, images_df, occ_row, dir_destination, shared_counter, n_imgs_per_species, do_shuffle_occurrences, lock):
    if 'gbifID' in occ_row:
        id_column = 'gbifID'
    # If 'gbifID' is not a key, check if 'id' is a key
    elif 'id' in occ_row:
        id_column = 'id'
    elif 'coreid' in occ_row:
        id_column = 'coreid'
    else:
        raise

    print(f"{bcolors.BOLD}Working on occurrence: {occ_row[id_column]}{bcolors.ENDC}")
    gbif_id = occ_row[id_column]
    
    image_row = find_gbifID_in_images(gbif_id, images_df)  # New function to find the image_row

    if image_row is not None:
        filename_image, filename_image_jpg, herb_code, specimen_id, family, genus, species, fullname = generate_image_filename(occ_row)  
        
        current_count = shared_counter.get_count(fullname)

        # If the fullname is not in the counter yet, increment it
        if current_count == 0:
            shared_counter.increment(fullname)
            
        print(shared_counter.get_count(fullname))
        if shared_counter.get_count(fullname) > n_imgs_per_species:
            print(f"Reached image limit for {fullname}. Skipping.")
            return
        else:
        
            gbif_url = image_row['identifier']

            image_candidate = ImageCandidateMulti(cfg, image_row, occ_row, gbif_url, dir_destination, lock)
            if image_candidate.download_success:  
                shared_counter.increment(fullname)
    else:
        pass

def find_gbifID_in_images(gbif_id, images_df):
    if 'gbifID' in images_df.columns:
        id_column = 'gbifID'
    # If 'gbifID' is not a key, check if 'id' is a key
    elif 'id' in images_df.columns:
        id_column = 'id'
    elif 'coreid' in  images_df.columns:
        id_column = 'coreid'
    else:
        raise

    image_row = images_df[images_df[id_column] == gbif_id]
    if image_row.empty:
        return None
    return image_row.iloc[0]


def process_each_image_row_multiDirs(cfg, image_row, occ_df, dir_destination, shared_counter, n_imgs_per_species, do_shuffle_occurrences, lock):
    if 'gbifID' in image_row:
        id_column = 'gbifID'
    # If 'gbifID' is not a key, check if 'id' is a key
    elif 'id' in image_row:
        id_column = 'id'
    elif 'coreid' in image_row:
        id_column = 'coreid'
    else:
        raise
    
    print(f"{bcolors.BOLD}Working on image: {image_row[id_column]}{bcolors.ENDC}")
    gbif_id = image_row[id_column]
    gbif_url = image_row['identifier']

    occ_row = find_gbifID(gbif_id,occ_df)

    if occ_row is not None:
        filename_image, filename_image_jpg, herb_code, specimen_id, family, genus, species, fullname = generate_image_filename(occ_row)  
        
        current_count = shared_counter.get_count(fullname)

        # If the fullname is not in the counter yet, increment it
        if current_count == 0:
            shared_counter.increment(fullname)
            
        print(shared_counter.get_count(fullname))
        if shared_counter.get_count(fullname) > n_imgs_per_species:
            print(f"Reached image limit for {fullname}. Skipping.")
            return
        
        image_candidate = ImageCandidateMulti(cfg, image_row, occ_row, gbif_url, dir_destination, lock)
        if image_candidate.download_success:  
            shared_counter.increment(fullname)
    else:
        pass


# def process_each_image_row(cfg, image_row, occ_df, lock):
#     if 'gbifID' in image_row:
#         id_column = 'gbifID'
#     # If 'gbifID' is not a key, check if 'id' is a key
#     elif 'id' in image_row:
#         id_column = 'id'
#     elif 'coreid' in image_row:
#         id_column = 'coreid'
#     else:
#         raise

#     print(f"{bcolors.BOLD}Working on image: {image_row[id_column]}{bcolors.ENDC}")
#     gbif_id = image_row[id_column]
#     if cfg['is_custom_file']:
#         gbif_url = image_row[cfg['custom_url_column_name']]
#     else:
#         gbif_url = image_row['identifier'] 

#     occ_row = find_gbifID(gbif_id,occ_df)

#     if occ_row is not None:
#         ImageInfo = ImageCandidate(cfg, image_row, occ_row, gbif_url, lock)
#         return pd.DataFrame(occ_row)
#         # ImageInfo.download_image(cfg, occ_row, image_row)
#     else:
#         pass

def download_from_custom_file(cfg):
    # Get DWC files
    images_df = read_custom_file(cfg)

    col_url = cfg['col_url']
    col_name = cfg['col_name']
    if col_url == None:
        col_url = 'identifier'
    else:
        col_url = col_url

    # Report summary
    print(f"{bcolors.BOLD}Beginning of images file:{bcolors.ENDC}")
    print(images_df.head())

    # Ignore problematic Herbaria
    if cfg['ignore_banned_herb']:
        for banned_url in cfg['banned_url_stems']:
            images_df = images_df[~images_df[col_url].str.contains(banned_url, na=False)]
    
    # Report summary
    n_imgs = images_df.shape[0]
    print(f"{bcolors.BOLD}Number of images in images file: {n_imgs}{bcolors.ENDC}")

    results = process_custom_image_batch(cfg, images_df)

def read_custom_file(cfg):
    dir_home = cfg['dir_home']
    filename_img = cfg['filename_img']
    # read the images.csv or occurences.csv file. can be txt ro csv
    images_df = ingest_DWC(filename_img,dir_home)
    return images_df

# def ingest_DWC(DWC_csv_or_txt_file,dir_home):
#     if DWC_csv_or_txt_file.split('.')[1] == 'txt':
#         df = pd.read_csv(os.path.join(dir_home,DWC_csv_or_txt_file), sep="\t",header=0, low_memory=False, dtype=str)
#     elif DWC_csv_or_txt_file.split('.')[1] == 'csv':
#         df = pd.read_csv(os.path.join(dir_home,DWC_csv_or_txt_file), sep=",",header=0, low_memory=False, dtype=str)
#     else:
#         print(f"{bcolors.FAIL}DWC file {DWC_csv_or_txt_file} is not '.txt' or '.csv' and was not opened{bcolors.ENDC}")
#     return df
'''
# Works, but could be faster
def process_custom_image_batch(cfg, images_df):
    futures_list = []
    results = []

    lock = Lock() 

    with th(max_workers=13) as executor:
        for index, image_row in images_df.iterrows():
            futures = executor.submit(process_each_custom_image_row, cfg, image_row, lock)
            futures_list.append(futures)

        for future in futures_list:
            try:
                result = future.result(timeout=60)
                results.append(result)
            except Exception:
                results.append(None)
    return results

def process_each_custom_image_row(cfg, image_row, lock):
    col_url = cfg['col_url']
    col_name = cfg['col_name']

    if col_url == None:
        col_url = 'identifier'
    else:
        col_url = col_url

    gbif_url = image_row[col_url] 

    print(f"{bcolors.BOLD}Working on image: {image_row[col_name]}{bcolors.ENDC}")
    if image_row is not None:
        ImageInfo = ImageCandidateCustom(cfg, image_row, gbif_url, col_name, lock)
    else:
        pass
    
'''

def worker_custom(queue, cfg, results, lock):
    while True:
        image_row = queue.get()
        if image_row is None:
            break  # None is the signal to stop processing
        try:
            result = process_each_custom_image_row(cfg, image_row, lock)
            results.append(result)
        except Exception as e:
            print(f"Error processing image: {e}")
            results.append(None)
        queue.task_done()

def process_custom_image_batch(cfg, images_df):
    num_workers = 13  # you can adjust this number based on your system's capabilities or load
    queue = Queue()
    results = []
    lock = Lock()

    # Start worker threads
    threads = []
    for _ in range(num_workers):
        t = Thread(target=worker_custom, args=(queue, cfg, results, lock))
        t.start()
        threads.append(t)

    # Enqueue tasks
    for index, image_row in images_df.iterrows():
        queue.put(image_row)

    # Block until all tasks are done
    queue.join()

    # Stop workers
    for _ in range(num_workers):
        queue.put(None)  # send as many None as the number of workers to stop them
    for t in threads:
        t.join()

    return results

def process_each_custom_image_row(cfg, image_row, lock):
    col_url = cfg['col_url'] if cfg['col_url'] else 'identifier'
    gbif_url = image_row[col_url]

    print(f"Working on image: {image_row[cfg['col_name']]}")
    if image_row is not None:
        ImageInfo = ImageCandidateCustom(cfg, image_row, gbif_url, cfg['col_name'], lock)
        return ImageInfo
    else:
        pass