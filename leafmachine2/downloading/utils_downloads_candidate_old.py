import os
import time
import requests
import re
import sys
import inspect
import random
import logging
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
import http.client as http_client
from urllib3.util import Retry
from aiohttp import ClientSession, ClientTimeout, ClientResponseError, ClientConnectionError
from PIL import Image
import asyncio
from io import BytesIO
from aiohttp_retry import RetryClient, ExponentialRetry
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.common.exceptions import (
    WebDriverException, NoSuchElementException, TimeoutException, 
    StaleElementReferenceException, ElementNotInteractableException, 
    ElementClickInterceptedException, InvalidElementStateException, 
    NoSuchFrameException, NoSuchWindowException
)
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)

from leafmachine2.machine.general_utils import bcolors


@dataclass
class ImageCandidate:
    cfg: dict
    image_row: pd.Series
    occ_row: pd.Series
    url: str
    lock: asyncio.Lock
    failure_log: dict

    filename_image: str = field(init=False)
    filename_image_jpg: str = field(init=False)
    herb_code: str = field(init=False)
    specimen_id: str = field(init=False)
    family: str = field(init=False)
    genus: str = field(init=False)
    species: str = field(init=False)
    fullname: str = field(init=False)

    headers_occ: list = field(init=False)
    headers_img: list = field(init=False)
    download_success: bool = field(default=False, init=False)

    def __post_init__(self):
        self.headers_occ = self.occ_row
        self.headers_img = self.image_row
        self.filename_image, self.filename_image_jpg, self.herb_code, self.specimen_id, self.family, self.genus, self.species, self.fullname = generate_image_filename(self.occ_row)

    async def download_image(self, session: ClientSession, n_queue ,logging_enabled=False) -> None:
        self.logging_enabled = logging_enabled
        # Check if the URL is valid
        if ((not isinstance(self.url, str) or pd.isna(self.url)) and n_queue != 0):
            print(f"{bcolors.WARNING}                Invalid URL --> {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
            self.download_success = False
            return
        else:
            if self.logging_enabled:
                http_client.HTTPConnection.debuglevel = 1
                logging.basicConfig(level=logging.DEBUG)
                requests_log = logging.getLogger("aiohttp.client")
                requests_log.setLevel(logging.CRITICAL)
                requests_log.propagate = True
                print(f"{bcolors.BOLD}      {self.fullname}{bcolors.ENDC}")
                print(f"{bcolors.BOLD}           URL: {self.url}{bcolors.ENDC}")    

            dir_destination = self.cfg['dir_destination_images']
            MP_low = self.cfg['MP_low']
            MP_high = self.cfg['MP_high']


            # session = requests.Session()
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
            }
            
            await asyncio.sleep(random.uniform(0, 5))  # Random delay between 0 and 5 seconds

            try:
                async with session.get(self.url, headers=headers, timeout=ClientTimeout(total=10.0)) as response:
                    response.raise_for_status()
                    img_data = await response.read()
                    img = Image.open(BytesIO(img_data))

                    was_saved = await self._save_matching_image(img, MP_low, MP_high, dir_destination)
                    print(f"{bcolors.OKGREEN}                SUCCESS 1st{bcolors.ENDC}")
                    self.download_success = True
                
                if not self.download_success:
                    retry_options = ExponentialRetry(attempts=3)
                    async with RetryClient(session, retry_options=retry_options) as retry_session:
                        async with retry_session.get(self.url, timeout=ClientTimeout(total=10.0)) as response:
                            response.raise_for_status()
                            img_data = await response.read()
                            img = Image.open(BytesIO(img_data))

                            was_saved = await self._save_matching_image(img, MP_low, MP_high, dir_destination)
                            print(f"{bcolors.OKGREEN}                SUCCESS 2nd{bcolors.ENDC}")
                            self.download_success = True

            except ImageSaveError as e:
                print(f"{bcolors.WARNING}                {e} --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
                # self._log_failure("ImageSaveError", self.url, e)
                # await self.retry_download_image()
                self.download_success = False

            except ClientResponseError as http_err:
                if http_err.status == 404:
                    print(f"{bcolors.WARNING}                404 Error (Not Found) --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
                    self.download_success = '404'
                else:
                    print(f"{bcolors.WARNING}                HTTP Error --> {http_err} --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
                self.download_success = False

            except ClientConnectionError as conn_err:
                print(f"{bcolors.WARNING}                Connection Error --> {conn_err} --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
                # self._log_failure("ConnectionError", self.url, conn_err)
                # await self.retry_download_image()
                self.download_success = False

            except Exception as e:
                if str(e) != "No active exception to reraise":
                    print(f"{bcolors.WARNING}                No Connection or Rate Limited --> {e} --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
                    # self._log_failure("OtherError", self.url, e)
                # await self.retry_download_image()
                self.download_success = False

            # finally:
                # await asyncio.sleep(2 + random.uniform(0, 2))

    # async def retry_download_image(self) -> None:
    #     for attempt in range(5):  # Retry up to 5 times
    #         print(f"Retry attempt {attempt + 1}")

    #         await asyncio.sleep(random.uniform(0, 5) * 2**attempt)  # Exponential backoff with random delay

    #         # Use Selenium for retries
    #         success = await self.download_image_with_selenium()
    #         if success:
    #             self._log_failure("PassedOnRetry", self.url, f"Success on retry attempt {attempt + 1} with Selenium")
    #             return
            
    #     self._log_failure("FailedOnRetry", self.url, f"Failed on retry attempt {attempt + 1} with Selenium")
    #     print(f"{bcolors.FAIL}                All retry attempts failed for URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")

    async def download_image_with_selenium(self, driver, index, n_queue, timeout_duration=10) -> bool:
        
        if ((not isinstance(self.url, str) or pd.isna(self.url)) and n_queue != 0):
            print(f"{bcolors.BOLD}                Invalid URL --> {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
            # self._log_failure("InvalidURL", self.url, "URL is not a valid string")
            self.download_success = False
            return False

        # print(f"{bcolors.BOLD}      {self.fullname}{bcolors.ENDC}")
        # print(f"{bcolors.BOLD}           URL: {self.url}{bcolors.ENDC}")
        print(f"{bcolors.CWHITEBG}                RETRY [i {index}] [Q {n_queue}] {self.url} GBIF ID: {self.fullname}{bcolors.ENDC}")

        dir_destination = self.cfg['dir_destination_images']
        MP_low = self.cfg['MP_low']
        MP_high = self.cfg['MP_high']

        try:
            driver.set_page_load_timeout(timeout_duration)
            driver.implicitly_wait(timeout_duration)
            driver.get(self.url)
            await asyncio.sleep(3)  # Wait for the page to load

            wait = WebDriverWait(driver, timeout_duration)
            img_element = wait.until(EC.presence_of_element_located((By.TAG_NAME, 'img')))
            img_url = img_element.get_attribute('src')

            response = requests.get(img_url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))

            was_saved = await self._save_matching_image(img, MP_low, MP_high, dir_destination, is_retry=True)
            if not was_saved:
                print(f"Failed to save the image RETRY: {self.url}, GBIF ID: {self.fullname}")
                return False

            print(f"{bcolors.OKCYAN}                SUCCESS RETRY [Q {n_queue}]{bcolors.ENDC}")
            self.download_success = True
            return True
    
        except (WebDriverException, NoSuchElementException, TimeoutException, 
            StaleElementReferenceException, ElementNotInteractableException, 
            ElementClickInterceptedException, InvalidElementStateException, 
            NoSuchFrameException, NoSuchWindowException) as e:
            print(f"{bcolors.CREDBG}                Selenium Exception --> {e} --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
            self._log_failure(type(e).__name__, self.url, str(e))
            self.download_success = False
            return False

        except ImageSaveError as e:
            print(f"{bcolors.CREDBG}                {e} --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
            self._log_failure("ImageSaveError", self.url, e)
            self.download_success = False
            return False

        except Exception as e:
            print(f"{bcolors.CREDBG}                SKIP --- No Connection or Rate Limited --> {e} --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
            self._log_failure("OtherError", self.url, e)
            self.download_success = False
            return False



    def _log_failure(self, error_type, url, error):
        server = url.split('/')[2] if isinstance(url, str) else "invalid_url"
        if server not in self.failure_log:
            self.failure_log[server] = []
        self.failure_log[server].append({
            "error_type": error_type,
            "url": url,
            "error": str(error)
        })
        if self.logging_enabled:
            print(f"Failure log updated: {self.failure_log}")



    async def _save_matching_image(self, img, MP_low, MP_high, dir_destination,is_retry=False) -> bool:
        color = bcolors.OKCYAN if is_retry else bcolors.OKGREEN

        img_mp, img_w, img_h = check_image_size(img)
        if img_mp < MP_low:
            print(f"{bcolors.WARNING}                SKIP < {MP_low}MP: {img_mp}{bcolors.ENDC}")
            print(f"{bcolors.WARNING}                URL: {self.url}{bcolors.ENDC}")
            return False

        if MP_low <= img_mp <= MP_high:
            image_path = os.path.join(dir_destination, self.filename_image_jpg)
            img.save(image_path)
            await self._add_occ_and_img_data()
            print(f"{color}                Regular MP: {img_mp}{bcolors.ENDC}")
            print(f"{color}                URL: {self.url}{bcolors.ENDC}")
            print(f"{color}                Image Saved: {image_path}{bcolors.ENDC}")
            return True

        if img_mp > MP_high and self.cfg['do_resize']:
            img_w, img_h = calc_resize(img_w, img_h)
            img = img.resize((img_w, img_h))
            image_path = os.path.join(dir_destination, self.filename_image_jpg)
            img.save(image_path)
            await self._add_occ_and_img_data()
            print(f"{color}                {MP_high}MP+ Resize: {img_mp}{bcolors.ENDC}")
            print(f"{color}                URL: {self.url}{bcolors.ENDC}")
            print(f"{color}                Image Saved: {image_path}{bcolors.ENDC}")
            return True

        print(f"{bcolors.CVIOLETBG}                {MP_high}MP+ Resize: {img_mp}{bcolors.ENDC}")
        print(f"{bcolors.CVIOLETBG}                URL: {self.url}{bcolors.ENDC}")
        print(f"{bcolors.CVIOLETBG}                SKIP: {image_path}{bcolors.ENDC}")
        return False

    async def _add_occ_and_img_data(self) -> None:
        self.image_row = self.image_row.to_frame().transpose().rename(columns={"identifier": "url"})
        id_column = 'gbifID' if 'gbifID' in self.image_row.columns else 'id'
        self.image_row = self.image_row.rename(columns={id_column: f'{id_column}_images'})

        new_data = pd.DataFrame({
            'fullname': [self.fullname],
            'filename_image': [self.filename_image],
            'filename_image_jpg': [self.filename_image_jpg]
        })

        combined = pd.concat([new_data.reset_index(drop=True), self.image_row.reset_index(drop=True), self.occ_row.reset_index(drop=True)], axis=1)
        combined.columns = np.hstack((new_data.columns.values, self.image_row.columns.values, self.occ_row.columns.values))
        await self._append_combined_occ_image(combined)

    async def _append_combined_occ_image(self, combined) -> None:
        path_csv_combined = os.path.join(self.cfg['dir_destination_csv'], self.cfg['filename_combined'])
        async with self.lock:
            try:
                combined.to_csv(path_csv_combined, mode='a', header=False, index=False)
                print(f'{bcolors.OKGREEN}       Added 1 row to combined CSV: {path_csv_combined}{bcolors.ENDC}')
            except Exception as e:
                print(f"{bcolors.WARNING}       Initializing new combined .csv file: [occ, images]: {path_csv_combined}{bcolors.ENDC}")
                combined.to_csv(path_csv_combined, mode='w', header=True, index=False)

class ImageSaveError(Exception):
    """Custom exception for image saving errors."""
    pass

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
def validate_herb_code(occ_row):
    possible_keys = ['institutionCode', 'institutionID', 'ownerInstitutionCode', 
                    'collectionCode', 'publisher', 'occurrenceID']
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
            response = session.get(self.url, headers=headers, stream=True, timeout=5.0, verify=False)
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

