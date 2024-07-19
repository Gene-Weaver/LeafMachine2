import os, base64, cv2
import time
import requests, ssl, aiohttp
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
from aiohttp import ClientSession, ClientTimeout, ClientResponseError, ClientConnectionError, TCPConnector
from PIL import Image, UnidentifiedImageError
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
from selenium.webdriver.chrome.options import Options
from concurrent.futures import ThreadPoolExecutor

from rich.console import Console, Group
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from threading import Thread, Lock
from queue import Queue, Empty
from bs4 import BeautifulSoup

# from scraperapi import ScraperAPIClient # pip install scraperapi-sdk

currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)

Image.MAX_IMAGE_PIXELS = None

from leafmachine2.machine.general_utils import bcolors

# Initialize ScraperAPI client
SCRAPERAPI_KEY = ''


USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15; rv:86.0) Gecko/20100101 Firefox/86.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:90.0) Gecko/20100101 Firefox/90.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.192 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; AS; rv:11.0) like Gecko',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
]

# def fetch_image_with_proxy_func(url):
#     headers = {
#         'User-Agent': random.choice(USER_AGENTS),
#         'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
#         'Referer': 'https://www.google.com/',
#         "Accept-Language": "en-US,en;q=0.9",
#         "DNT": "1",
#         "Connection": "keep-alive",
#     }

#     proxies = {
#         "https": f"http://scraperapi:{SCRAPERAPI_KEY}@proxy-server.scraperapi.com:8001"
#     }

#     # try:
#     #     response = requests.get(url, headers=headers, proxies=proxies, verify=False, allow_redirects=True, timeout=60)
#     #     response.raise_for_status()  # Check if the request was successful
#     #     return response.content
#     # except requests.exceptions.RequestException as e:
#     #     print(f"Request failed: {e}")
#     #     return None
#     session = requests.Session()
#     session.headers.update(headers)

#     # Initial request to get cookies
#     initial_response = session.get(url, proxies=proxies, verify=False, allow_redirects=True, timeout=60)
#     if initial_response.status_code != 200:
#         print(f"Initial request failed: {initial_response.status_code}")
#         return None

#     cookies = initial_response.cookies.get_dict()

#     try:
#         time.sleep(random.uniform(1, 5))
#         response = session.get(url, proxies=proxies, cookies=cookies, verify=False, allow_redirects=True, timeout=60)
#         response.raise_for_status()  # Check if the request was successful
#         return response.content
#     except requests.exceptions.RequestException as e:
#         print(f"Request failed: {e}")
#         return None

        
def fetch_image_with_proxy_func(url):
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Referer': 'https://www.google.com/',
        "Accept-Language": "en-US,en;q=0.9",
        "DNT": "1",
        "Connection": "keep-alive",
    }

    # Standard proxy URL
    scraperapi_url = f"http://api.scraperapi.com?api_key={SCRAPERAPI_KEY}&url={url}"

    # Premium proxy URL
    scraperapi_url_premium = f"http://api.scraperapi.com?api_key={SCRAPERAPI_KEY}&url={url}&premium=true"

    session = requests.Session()
    session.headers.update(headers)

    # Introduce a random delay to avoid detection
    time.sleep(random.uniform(1, 5))

    try:
        # Initial request without premium proxy
        initial_response = session.get(scraperapi_url, verify=False, allow_redirects=True, timeout=60)
        initial_response.raise_for_status()

        # If the initial request is successful, use the response directly
        if initial_response.status_code == 200:
            return initial_response.content

    except requests.exceptions.RequestException as e:
        print(f"Initial request failed: {e}")

    # If the initial request fails, try with premium proxy and cookies
    try:
        # Introduce another delay before the next request
        time.sleep(random.uniform(1, 5))

        # Get cookies with premium proxy
        initial_response_premium = session.get(scraperapi_url_premium, verify=False, allow_redirects=True, timeout=60)
        initial_response_premium.raise_for_status()

        cookies = initial_response_premium.cookies.get_dict()
        cookies['random_cookie'] = str(random.randint(1, 1000000))

        # Introduce another delay before the actual image request
        time.sleep(random.uniform(1, 5))
        response = session.get(scraperapi_url_premium, headers=headers, cookies=cookies, verify=False, allow_redirects=True, timeout=60)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Request with premium proxy failed: {e}")
        return None

    
@dataclass
class ImageCandidate:
    cfg: dict
    # image_row: pd.DataFrame
    # occ_row: pd.DataFrame
    occ_row: pd.DataFrame
    image_row: pd.DataFrame
    url: str
    lock: asyncio.Lock
    failure_log: dict
    download_tracker: dict  # Shared dictionary
    completed_tracker: list  # Shared list
    banned_url_tracker: list # Shared list
    banned_url_counts_tracker: dict # Shared dictionary
    # semaphore_scraperapi: asyncio.Semaphore
    # pp: PrettyPrint # self.pp.print()  ALSO enable stuff in process_batch()

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

    need_image: bool = field(default=False, init=False)

    def __post_init__(self):
        self.headers_occ = self.occ_row
        self.headers_img = self.image_row
        self.filename_image, self.filename_image_jpg, self.herb_code, self.specimen_id, self.family, self.genus, self.species, self.fullname = generate_image_filename(self.occ_row)
            
        self.n_to_download = self.cfg['n_to_download']
    
    def get_url_stem(self, url):
        if isinstance(url, str):
            return url.split('/')[2]  # Extracts the URL stem (domain)
        return None
    
    async def download_image(self, session: ClientSession, n_queue, logging_enabled=False, timeout=20) -> None:
        if self.fullname is None:
            self.download_success = 'skip'
            return
        
        # current_count = self.download_tracker.get(self.fullname, 0)
        current_count = self.download_tracker.get(self.fullname, 0)


        if current_count >= self.n_to_download:
            print(f"      Skipping download for {self.fullname} as max number of images {current_count}/{self.n_to_download} have already been downloaded.")
            self.download_success = True
            return  
        
        # Check if the URL stem is in the banned list
        url_stem = self.get_url_stem(self.url)

        
        
        self.logging_enabled = logging_enabled
        # Check if the URL is valid
        if ((not isinstance(self.url, str) or pd.isna(self.url)) and n_queue != 0):
            print(f"{bcolors.BOLD}      Invalid URL --> {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
            self.download_success = False
            return 

        if 'manifest' in self.url:
            print(f"{bcolors.BOLD}      Manifest URL detected --> {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
            self.download_success = 'skip'
            return
        # elif url_stem is not None and url_stem in self.banned_url_tracker:
        #     print(f"{bcolors.CBLUEBG3}      URL stem is banned --> {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
        #     self.download_success = 'skip'
        #     return 
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

            headers = {
                'User-Agent': random.choice(USER_AGENTS),  # Rotate user agent
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Referer': 'https://www.google.com/',  # Spoof the referrer
            }

            await asyncio.sleep(random.uniform(0, 20))  # Random delay between 0 and 5 seconds

            try:
                # Attempt to download the image directly
                async with session.get(self.url, headers=headers, timeout=ClientTimeout(total=timeout)) as response:
                    response.raise_for_status()
                    content_type = response.headers.get('content-type')

                    if 'image' in content_type:
                        img_data = await response.read()
                        img = Image.open(BytesIO(img_data))

                        ### Exit early if the count has increased while working
                        current_count = self.download_tracker.get(self.fullname, 0)
                        if current_count >= self.n_to_download:
                            print(f"Skipping URGENT download for {self.fullname} as max number of images {current_count}/{self.n_to_download} have already been downloaded.")
                            self.download_success = True
                            return True

                        was_saved, reason = await self._save_matching_image(img, MP_low, MP_high, dir_destination)
                        if reason == "too_small" and 'too_small' in self.banned_url_counts_tracker:
                            self.banned_url_counts_tracker['too_small'] += 1
                        elif reason == "too_small":
                            self.banned_url_counts_tracker['too_small'] = 1

                        if was_saved:
                            print(f"{bcolors.OKGREEN}      SUCCESS 1st{bcolors.ENDC}")
                            self.download_success = True
                            await self._add_occ_and_img_data()
                        else:
                            print(f"{bcolors.WARNING}      FOUND BUT NOT SAVED{bcolors.ENDC}")
                            self.download_success = False

                    # if not self.download_success:
                    #     retry_options = ExponentialRetry(attempts=3)
                    #     async with RetryClient(session, retry_options=retry_options) as retry_session:
                    #         async with retry_session.get(self.url, headers=headers, timeout=ClientTimeout(total=10.0)) as response:
                    #             response.raise_for_status()
                    #             content_type = response.headers.get('content-type')

                    #             if 'image' in content_type:
                    #                 img_data = await response.read()
                    #                 img = Image.open(BytesIO(img_data))

                    #                 was_saved, reason = await self._save_matching_image(img, MP_low, MP_high, dir_destination)
                    #                 print(f"{bcolors.OKGREEN}                SUCCESS 2nd{bcolors.ENDC}")
                    #                 self.download_success = True

            except ImageSaveError as e:
                print(f"{bcolors.WARNING}      {e} --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
                self.download_success = False
                # await asyncio.sleep(random.uniform(1, 5))  # Random delay between 1 and 5 seconds

            except ClientResponseError as http_err:
                if http_err.status == 404:
                    print(f"{bcolors.WARNING}404 Error (Not Found) --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
                    self.download_success = 'skip' # To cause it to not be tried again
                elif http_err.status == 503:
                    print(f"{bcolors.WARNING}503 Error (Service Unavailable) --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
                    self.download_success = 'skip' # To cause it to not be tried again
                else:
                    print(f"{bcolors.WARNING}HTTP Error --> {http_err} --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
                    self.download_success = 'skip' # To cause it to not be tried again


            except ClientConnectionError as conn_err:
                print(f"{bcolors.WARNING}      Connection Error --> {conn_err} --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
                self.download_success = False
                # await asyncio.sleep(random.uniform(1, 5))  # Random delay between 1 and 5 seconds

            except Exception as e:
                if str(e) != "No active exception to reraise":
                    print(f"{bcolors.WARNING}      No Connection or Rate Limited --> {e} --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
                self.download_success = False
                # await asyncio.sleep(random.uniform(1, 5))  # Random delay between 1 and 5 seconds

        if self.download_success and self.download_success != 'skip':
            async with self.lock:
                current_count = self.download_tracker.get(self.fullname, 0)
                updated_count = current_count + 1
                self.download_tracker.update({self.fullname: updated_count})
                print(f"{bcolors.OKGREEN}      SUCCESS {self.fullname} has {updated_count} images{bcolors.ENDC}")

    async def download_base64_image(self, url):
        was_saved = False
        try:
            header, encoded = url.split(",", 1)
            data = base64.b64decode(encoded)
            img = Image.open(BytesIO(data))
            dir_destination = self.cfg['dir_destination_images']
            MP_low = self.cfg['MP_low']
            MP_high = self.cfg['MP_high']
            
            was_saved, reason = await self._save_matching_image(img, MP_low, MP_high, dir_destination)
            self.download_success = True if was_saved else False
        except Exception as e:
            print(f"Error saving base64 image: {e}")
            self.download_success = False
        return was_saved, reason
    
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

    async def download_image_with_selenium(self, driver, index, n_queue, timeout_duration=90, max_strikes=50) -> bool:
        print(f"{bcolors.CWHITEBG}      RETRY [i {index}] [Q {n_queue}] {self.url} GBIF ID: {self.fullname}{bcolors.ENDC}")
        self.download_success = False

        current_count = self.download_tracker.get(self.fullname, 0)
        url_stem = self.get_url_stem(self.url)

        if current_count >= self.n_to_download or self.fullname in self.completed_tracker:
            print(f"Skipping download for {self.fullname} as max number of images {current_count}/{self.n_to_download} have already been downloaded.")
            self.download_success = True
            return 
        
        if current_count < self.n_to_download:
            if ((not isinstance(self.url, str) or pd.isna(self.url)) and n_queue != 0):
                print(f"{bcolors.BOLD}      Invalid URL --> {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
                self.download_success = 'skip'
                return 
            
            async with self.lock:
                if url_stem in self.banned_url_tracker:
                    print(f"{bcolors.CGREYBG}      ******URL stem is banned --> {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
                    self.download_success = 'skip'
                    return 

            # Ensure the URL is a valid image URL and not a user agent string
            if "user-agent" in self.url.lower():
                print(f"{bcolors.BOLD}      Invalid URL detected (User-Agent) --> {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
                self.download_success = False
                return 


            dir_destination = self.cfg['dir_destination_images']
            MP_low = self.cfg['MP_low']
            MP_high = self.cfg['MP_high']

            retries = 1
            for attempt in range(retries):
                
                await asyncio.sleep(10 * (attempt + 1))  # Increase wait time with each attempt
                try:
                    was_saved = False
                    reason = None
                    driver.set_page_load_timeout(timeout_duration)
                    driver.implicitly_wait(timeout_duration)

                    # Set random User-Agent
                    user_agent = random.choice(USER_AGENTS)
                    driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": user_agent})
                    driver.execute_cdp_cmd('Network.setExtraHTTPHeaders', {
                        "headers": {
                            "Accept-Language": "en-US,en;q=0.9",
                            "Referer": "https://www.google.com/",
                            "DNT": "1",
                            "Connection": "keep-alive",
                        }
                    })
                
                    driver.get(self.url)

                    wait = WebDriverWait(driver, timeout_duration)
                    img_element = wait.until(EC.presence_of_element_located((By.TAG_NAME, 'img')))
                    img_url = img_element.get_attribute('src')

                    # Handle redirects by checking the current URL
                    current_url = driver.current_url
                    if current_url != self.url and not current_url.lower().startswith("http://user-agent"):
                        print(f"{bcolors.CVIOLETBG}      Redirected to {current_url} from {self.url}{bcolors.ENDC}")
                        self.url = current_url
                        img_element = wait.until(EC.presence_of_element_located((By.TAG_NAME, 'img')))
                        img_url = img_element.get_attribute('src')

                    if img_url.startswith('data:image'):
                        was_saved, reason = await self.download_base64_image(img_url)
                    else:
                        response = requests.get(img_url)
                        response.raise_for_status()
                        img = Image.open(BytesIO(response.content))

                        ### Exit early if the count has increased while working
                        current_count = self.download_tracker.get(self.fullname, 0)
                        if current_count >= self.n_to_download:
                            print(f"Skipping urgent download for {self.fullname} as max number of images {current_count}/{self.n_to_download} have already been downloaded.")
                            self.download_success = True
                            return 

                        was_saved, reason = await self._save_matching_image(img, MP_low, MP_high, dir_destination, is_retry=True)
                    
                    if reason == "too_small" and 'too_small' in self.banned_url_counts_tracker:
                        self.banned_url_counts_tracker['too_small_selenium'] += 1
                    elif reason == "too_small":
                        self.banned_url_counts_tracker['too_small_selenium'] = 1

                    if was_saved:
                        await self._add_occ_and_img_data() # IF saved, add row to csv
                        print(f"{bcolors.OKCYAN}      SUCCESS RETRY attempt {attempt + 1} [Q {n_queue}]{bcolors.ENDC}")
                        self.download_success = True
                        if self.download_success:
                            async with self.lock:
                                current_count = self.download_tracker.get(self.fullname, 0)
                                updated_count = current_count + 1
                                self.download_tracker.update({self.fullname: updated_count})
                                print(f"{bcolors.OKCYAN}      SUCCESS {self.fullname} has {updated_count} images{bcolors.ENDC}")
                        return 

                    else:
                        print(f"Failed to save the image RETRY: {self.url}, GBIF ID: {self.fullname}")


                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 503:
                        print(f"{bcolors.WARNING}      SKIP all retries due to 503 error --> {e} --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
                        self._log_failure("HTTPError503", self.url, e)
                        return   # Skip all retries if 503 error occurs

                except (WebDriverException, NoSuchElementException, TimeoutException, 
                        StaleElementReferenceException, ElementNotInteractableException, 
                        ElementClickInterceptedException, InvalidElementStateException, 
                        NoSuchFrameException, NoSuchWindowException) as e:
                    if attempt == retries - 1:
                        print(f"{bcolors.FAIL}      Selenium Exception {attempt + 1} --> {e} --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
                    else:
                        print(f"{bcolors.OKBLUE}      Selenium Exception {attempt + 1} --> {e} --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")

                except ImageSaveError as e:
                    if attempt == retries - 1:
                        print(f"{bcolors.FAIL}      attempt {attempt + 1} {e} --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
                    else:
                        print(f"{bcolors.OKBLUE}      attempt {attempt + 1} {e} --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")

                except Exception as e:
                    if attempt == retries - 1:
                        print(f"{bcolors.FAIL}      SKIP attempt {attempt + 1} --- No Connection or Rate Limited --> {e} --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
                    else:
                        print(f"{bcolors.OKBLUE}      SKIP attempt {attempt + 1} --- No Connection or Rate Limited --> {e} --> URL: {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
            
                async with self.lock:
                    if url_stem in self.banned_url_tracker:
                        print(f"{bcolors.CGREYBG}      ***URL stem is banned --> {self.url}, GBIF ID: {self.fullname}{bcolors.ENDC}")
                        self.download_success = 'skip'
                        return 
                
            
                # self.download_success = 'proxy'
                # return 
                # If all retries with Selenium fail, attempt to download using ScraperAPI
                print(f"{bcolors.BOLD}      All retries with Selenium failed. Attempting to download using ScraperAPI...{bcolors.ENDC}")
                MP_low = self.cfg['MP_low']
                MP_high = self.cfg['MP_high']
                dir_destination = self.cfg['dir_destination_images']

                for attempt in range(3):  # Retry up to 3 times for ScraperAPI requests
                    try:
                        
                        # async with semaphore_scraperapi:
                            # image_data = fetch_image_with_proxy_func(self.url)
                        image_data = fetch_image_with_proxy_func(self.url)
                        if image_data:
                            # Check if image_data is base64 encoded
                            if image_data.startswith(b'data:image'):
                                was_saved, reason = await self.download_base64_image(image_data.decode('utf-8'))
                            else:
                                try:
                                    img = Image.open(BytesIO(image_data))
                                except UnidentifiedImageError as e:
                                    print(f"{bcolors.CREDBG}      Error identifying image file with proxy: {e} URL: {self.url}{bcolors.ENDC}")
                                    self.download_success = 'skip'
                                    break  # Skip further processing for this image

                                was_saved, reason = await self._save_matching_image(img, MP_low, MP_high, dir_destination, is_retry=True)

                        if reason == "too_small" and 'too_small' in self.banned_url_counts_tracker:
                            self.banned_url_counts_tracker['too_small_proxy'] += 1
                        elif reason == "too_small":
                            self.banned_url_counts_tracker['too_small_proxy'] = 1
                                
                        if was_saved:
                            await self._add_occ_and_img_data()
                            self.download_success = True

                            async with self.lock:
                                current_count = self.download_tracker.get(self.fullname, 0)
                                updated_count = current_count + 1
                                self.download_tracker.update({self.fullname: updated_count})
                                print(f"{bcolors.CGREENBG2}      SUCCESS WITH PROXY {self.fullname} has {updated_count} images  URL: {self.url}{bcolors.ENDC}")
                            return True
                    except requests.exceptions.RequestException as e:
                        print(f"{bcolors.CREDBG}      Request failed: {e} URL: {self.url}{bcolors.ENDC}")
                    except Exception as e:
                        print(f"{bcolors.CREDBG}      Error in proxy attempt {attempt + 1}: {e}{self.banned_url_counts_tracker[url_stem]} URL: {self.url}{bcolors.ENDC}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff


                print(f"{bcolors.CBEIGE}      ADDING TO BANNED URL LIST{bcolors.ENDC}")
                # Update banned_url_counts_tracker
                async with self.lock:
                    if url_stem in self.banned_url_counts_tracker:
                        self.banned_url_counts_tracker[url_stem] += 1
                    else:
                        self.banned_url_counts_tracker[url_stem] = 1
                    print(f"{bcolors.CREDBG}      URL stem {url_stem} COUNT {self.banned_url_counts_tracker[url_stem]} URL: {self.url}{bcolors.ENDC}")

                    # Check if the count exceeds the threshold
                    if self.banned_url_counts_tracker[url_stem] > max_strikes:
                        if url_stem not in self.banned_url_tracker:
                            self.banned_url_tracker.append(url_stem)
                            print(f"{bcolors.CREDBG}      URL stem {url_stem} added to banned_url_tracker{bcolors.ENDC}")
                    print(f"{bcolors.CREDBG}      banned_url_tracker --> {self.banned_url_tracker} URL: {self.url}{bcolors.ENDC}")

             
                print(f"{bcolors.CREDBG}      Failed to download image even with ScraperAPI: {self.url}{bcolors.ENDC}")
                return 

        self.download_success = 'skip'
        return 

    
    async def download_image_with_proxy(self, session, max_strikes=50) -> bool:
        # If all retries with Selenium fail, attempt to download using ScraperAPI
        print(f"{bcolors.BOLD}      All retries with Selenium failed. Attempting to download using ScraperAPI...{bcolors.ENDC}")
        MP_low = self.cfg['MP_low']
        MP_high = self.cfg['MP_high']
        dir_destination = self.cfg['dir_destination_images']
        url_stem = self.get_url_stem(self.url)

        for attempt in range(3):  # Retry up to 3 times for ScraperAPI requests
            try:
                image_data = fetch_image_with_proxy_func(self.url)
                if image_data:
                    try:
                        img = Image.open(BytesIO(image_data))
                    except UnidentifiedImageError as e:
                        print(f"{bcolors.CREDBG}      Error identifying image file with proxy: {e}{bcolors.ENDC}")
                        self.download_success = 'skip'
                        break  # Skip further processing for this image

                    was_saved, reason = await self._save_matching_image(img, MP_low, MP_high, dir_destination, is_retry=True)
                    if reason == "too_small" and 'too_small' in self.banned_url_counts_tracker:
                        self.banned_url_counts_tracker['too_small_proxy'] += 1
                    elif reason == "too_small":
                        self.banned_url_counts_tracker['too_small_proxy'] = 1

                    if was_saved:
                        await self._add_occ_and_img_data()
                        self.download_success = True

                        async with self.lock:
                            current_count = self.download_tracker.get(self.fullname, 0)
                            updated_count = current_count + 1
                            self.download_tracker.update({self.fullname: updated_count})
                            print(f"{bcolors.CGREENBG2}      SUCCESS WITH PROXY {self.fullname} has {updated_count} images{bcolors.ENDC}")
                        return True
            except requests.exceptions.RequestException as e:
                print(f"{bcolors.CREDBG}      Request failed: {e}{bcolors.ENDC}")
            except Exception as e:
                print(f"{bcolors.CREDBG}      Error in proxy attempt {attempt + 1}: {e}{self.banned_url_counts_tracker[url_stem]}{bcolors.ENDC}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

        print(f"{bcolors.CBEIGE}      ADDING TO BANNED URL LIST{bcolors.ENDC}")
        # Update banned_url_counts_tracker
        async with self.lock:
            if url_stem in self.banned_url_counts_tracker:
                self.banned_url_counts_tracker[url_stem] += 1
            else:
                self.banned_url_counts_tracker[url_stem] = 1
            print(f"{bcolors.CREDBG}      URL stem {url_stem} COUNT {self.banned_url_counts_tracker[url_stem]}{bcolors.ENDC}")

            # Check if the count exceeds the threshold
            if self.banned_url_counts_tracker[url_stem] > max_strikes:
                if url_stem not in self.banned_url_tracker:
                    self.banned_url_tracker.append(url_stem)
                    print(f"{bcolors.CREDBG}      URL stem {url_stem} added to banned_url_tracker{bcolors.ENDC}")
            print(f"{bcolors.CREDBG}      banned_url_tracker --> {self.banned_url_tracker}{bcolors.ENDC}")

        print(f"{bcolors.CREDBG}      Failed to download image even with ScraperAPI: {self.url}{bcolors.ENDC}")
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



    async def _save_matching_image(self, img, MP_low, MP_high, dir_destination, is_retry=False):
        reason = None
        color = bcolors.OKCYAN if is_retry else bcolors.OKGREEN

        img_mp, img_w, img_h = check_image_size(img)
        if img_mp < MP_low:
            print(f"{bcolors.WARNING}      SKIP < {MP_low}MP: {img_mp}{bcolors.ENDC}")
            print(f"{bcolors.WARNING}      URL: {self.url}{bcolors.ENDC}")
            reason = "too_small"
            return False, reason

        if MP_low <= img_mp <= MP_high:
            image_path = os.path.join(dir_destination, self.filename_image_jpg)
            img.save(image_path)
            # await self._add_occ_and_img_data()
            print(f"{color}      Regular MP: {img_mp}{bcolors.ENDC}")
            print(f"{color}      URL: {self.url}{bcolors.ENDC}")
            print(f"{color}      Image Saved: {image_path}{bcolors.ENDC}")
            reason = "regular"
            return True, reason

        if img_mp > MP_high and self.cfg['do_resize']:
            img_w, img_h = calc_resize(img_w, img_h)
            img = img.resize((img_w, img_h))
            image_path = os.path.join(dir_destination, self.filename_image_jpg)
            img.save(image_path)
            # await self._add_occ_and_img_data()
            print(f"{color}      {MP_high}MP+ Resize: {img_mp}{bcolors.ENDC}")
            print(f"{color}      URL: {self.url}{bcolors.ENDC}")
            print(f"{color}      Image Saved: {image_path}{bcolors.ENDC}")
            reason = "resize"
            return True, reason

        print(f"{bcolors.CVIOLETBG}      {MP_high}MP+ Resize: {img_mp}{bcolors.ENDC}")
        print(f"{bcolors.CVIOLETBG}      URL: {self.url}{bcolors.ENDC}")
        print(f"{bcolors.CVIOLETBG}      SKIP: {image_path}{bcolors.ENDC}")
        reason = "too_large_skip_resize"
        return False, reason

    async def _add_occ_and_img_data(self) -> None:
        if not isinstance(self.occ_row, pd.DataFrame):
            # Convert Series to DataFrame
            self.occ_row = self.occ_row.to_frame().T

        try:
            # Ensure self.image_row is a DataFrame
            if not isinstance(self.image_row, pd.DataFrame):
                # Convert Series to DataFrame
                self.image_row = self.image_row.to_frame().T
            # Rename columns for DataFrame
            self.image_row = self.image_row.rename(columns={"identifier": "url"})
        except Exception as e:
            print(f"Error during renaming or conversion: {e}")

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
            file_exists = os.path.isfile(path_csv_combined)
            try:
                combined.to_csv(path_csv_combined, mode='a', header=not file_exists, index=False)
                print(f'{bcolors.OKGREEN}      Added data to combined CSV: {path_csv_combined}{bcolors.ENDC}')
            except Exception as e:
                print(f"{bcolors.WARNING}      Error appending to combined CSV: {path_csv_combined}{bcolors.ENDC}")
                print(f"{bcolors.WARNING}      {e}{bcolors.ENDC}")

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

    # # Display image using OpenCV
    # img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # cv2.imshow('Image', img_cv)
    # cv2.waitKey(0)  # Wait for a key press to close the image window
    # cv2.destroyAllWindows()
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
        specimen_id = str(occ_row[id_column].iloc[0])
        family = remove_illegal_chars(occ_row['family'].iloc[0])
        genus = remove_illegal_chars(occ_row['genus'].iloc[0])
        specificEpithet = str(occ_row['specificEpithet'].iloc[0]) if pd.notna(occ_row['specificEpithet'].iloc[0]) else ""
        species = remove_illegal_chars(keep_first_word(specificEpithet))

        # specimen_id = str(occ_row[id_column].values[0])
        # family = remove_illegal_chars(occ_row['family'].values[0])
        # genus = remove_illegal_chars(occ_row['genus'].values[0])
        # specificEpithet = str(occ_row['specificEpithet']) if pd.notna(occ_row['specificEpithet']) else ""
        # species = remove_illegal_chars(keep_first_word(specificEpithet))
        ## species = remove_illegal_chars(keep_first_word(occ_row['specificEpithet'].values[0]))
    except Exception as e:
        # If any of these are missing, it is skipped. So it MUST be ID'd to species level
        try:
            specimen_id = str(occ_row[id_column])
            family = remove_illegal_chars(occ_row['family'])
            genus = remove_illegal_chars(occ_row['genus'])
            species = remove_illegal_chars(keep_first_word(occ_row['specificEpithet']))
        except:
            specimen_id = ""
            family = ""
            genus = ""
            species = ""
            # print(e)
            return None, None, None, None, None, None, None, None
            
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













class PrettyPrint:
    def __init__(self, buffer_size=50, buffer_size_c=40, taxa_column_width=50):
        self.console = Console(record=True)
        self.completed_tasks = {}  # Dictionary to track unique taxa and their counts
        self.completed_order = []  # List to maintain the order of tasks
        self.log_queue = Queue()
        self.log_buffer = []
        self.buffer_size = buffer_size
        self.buffer_size_c = buffer_size_c
        self.taxa_column_width = taxa_column_width
        self.global_task_counter = 0  # Global counter for all tasks added
        self.lock = Lock()
        self.layout = Layout()
        self.layout.split_row(
            Layout(name="Completed", size=50, minimum_size=50),  # Fixed size for the left panel
            Layout(name="Logs", ratio=1)
        )
        self.live = Live(self.layout, console=self.console, refresh_per_second=1)
        self.stop_event = False
        self.update_thread = Thread(target=self.update_live_panel)
        self.update_thread.start()

    def add_completed(self, task):
        with self.lock:
            if task not in self.completed_tasks:
                self.global_task_counter += 1
                self.completed_tasks[task] = self.global_task_counter
                self.completed_order.append(task)
                if len(self.completed_order) > self.buffer_size_c:
                    oldest_task = self.completed_order.pop(0)
                    del self.completed_tasks[oldest_task]
                self.layout["Completed"].update(self._render_table())

    def print(self, message):
        self.log_queue.put(message)

    def update_live_panel(self):
        while not self.stop_event:
            try:
                message = self.log_queue.get(timeout=0.5)
                with self.lock:
                    self.log_buffer.append(message)
                    if len(self.log_buffer) > self.buffer_size:
                        self.log_buffer.pop(0)  # Remove the oldest message
                    log_messages = "\n".join(self.log_buffer)
                    formatted_log = self.format_log(log_messages)
                    self.layout["Logs"].update(Panel(formatted_log, title="Log"))
            except Empty:
                continue

    def format_taxa(self, task):
        if len(task) > self.taxa_column_width:
            return task[:self.taxa_column_width].ljust(self.taxa_column_width)
        return task.ljust(self.taxa_column_width)

    def format_log(self, log_messages):
        # Ensure that each line in log_messages is properly wrapped within the panel width
        formatted_lines = []
        for line in log_messages.split("\n"):
            if len(line) > 120:
                wrapped_lines = [line[i:i+120] for i in range(0, len(line), 120)]
                formatted_lines.extend(wrapped_lines)
            else:
                formatted_lines.append(line)
        return "\n".join(formatted_lines)

    def _render_table(self):
        table = Table(title="COMPLETED")
        table.add_column("Count", justify="right", style="cyan")
        table.add_column("Taxa", justify="left", style="magenta")

        for task in self.completed_order:
            count = self.completed_tasks[task]
            table.add_row(str(count), task)
        
        return table

    def start(self):
        self.live.start()

    def stop(self):
        self.stop_event = True
        self.update_thread.join()
        self.live.stop()



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
            img.verify()
            
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

    # async def download_base64_image(self, url):
    #     was_saved = False
    #     try:
    #         header, encoded = url.split(",", 1)
    #         data = base64.b64decode(encoded)
    #         img = Image.open(BytesIO(data))
    #         dir_destination = self.cfg['dir_destination_images']
    #         MP_low = self.cfg['MP_low']
    #         MP_high = self.cfg['MP_high']
            
    #         was_saved = await self._save_matching_image(img, MP_low, MP_high, dir_destination)
    #         self.download_success = True if was_saved else False
    #     except Exception as e:
    #         print(f"Error saving base64 image: {e}")
    #         self.download_success = False
    #     return was_saved
            
    # def _save_matching_image(self, img, MP_low, MP_high, dir_destination, lock) -> None:
    #     img_mp, img_w, img_h = check_image_size(img)
    #     if img_mp < MP_low:
    #         print(f"{bcolors.WARNING}                SKIP < {MP_low}MP: {img_mp}{bcolors.ENDC}")

    #     elif MP_low <= img_mp <= MP_high:
    #         image_path = os.path.join(dir_destination,self.filename_image_jpg)
    #         img.save(image_path)

    #         print(f"{bcolors.OKGREEN}                Regular MP: {img_mp}{bcolors.ENDC}")
    #         print(f"{bcolors.OKGREEN}                Image Saved: {image_path}{bcolors.ENDC}")

    #     elif img_mp > MP_high:
    #         if self.cfg['do_resize']:
    #             [img_w, img_h] = calc_resize(img_w, img_h)
    #             newsize = (img_w, img_h)
    #             img = img.resize(newsize)
    #             image_path = os.path.join(dir_destination,self.filename_image_jpg)
    #             img.save(image_path)

    #             print(f"{bcolors.OKGREEN}                {MP_high}MP+ Resize: {img_mp}{bcolors.ENDC}")
    #             print(f"{bcolors.OKGREEN}                Image Saved: {image_path}{bcolors.ENDC}")
    #         else:
    #             print(f"{bcolors.OKCYAN}                {MP_high}MP+ Resize: {img_mp}{bcolors.ENDC}")
    #             print(f"{bcolors.OKCYAN}                SKIP: {image_path}{bcolors.ENDC}")
    async def _save_matching_image(self, img, MP_low, MP_high, dir_destination, is_retry=False) -> bool:
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

