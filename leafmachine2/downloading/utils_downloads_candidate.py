import os, base64, cv2, csv
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
from urllib.parse import urljoin, urlparse, urlunparse
from requests.exceptions import ConnectionError, Timeout, HTTPError, RequestException
import threading

# from scraperapi import ScraperAPIClient # pip install scraperapi-sdk

currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)

Image.MAX_IMAGE_PIXELS = None

from leafmachine2.machine.general_utils import bcolors, get_cfg_from_full_path
# Initialize ScraperAPI client

try:
    # Attempt to load the private data file
    cfg_private = get_cfg_from_full_path(os.path.join(parentdir, 'PRIVATE_DATA.yaml'))
except FileNotFoundError:
    # Raise an error if the file is not found
    raise FileNotFoundError("The private data file 'PRIVATE_DATA.yaml' cannot be found. This file is required for scraperAPI to function.")
except Exception as e:
    # Handle any other unexpected exceptions
    raise RuntimeError(f"An unexpected error occurred while trying to load 'PRIVATE_DATA.yaml': {e}")
    

# List of available API keys
# SCRAPERAPI_KEYS = [
#     cfg_private['SCRAPERAPI_KEY1'],
#     cfg_private['SCRAPERAPI_KEY2'],
# ]
SCRAPERAPI_KEYS = [
    cfg_private['SCRAPERAPI_KEY3'],
]

BANNED = ['www.herbariumhamburgense.de', 'imagens4.jbrj.gov.br', 'imagens1.jbrj.gov.br', 
              'arbmis.arcosnetwork.org', '128.171.206.220', 'ia801503.us.archive.org', 'procyon.acadiau.ca',
              'www.inaturalist.org']
TIMEOUT = 80
USER_AGENTS = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
        ]
# Create an SSL context that ignores SSL certificate verification
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

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

        
# def fetch_image_with_proxy_func(url):
#     headers = {
#         'User-Agent': random.choice(USER_AGENTS),
#         'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
#         'Referer': 'https://www.google.com/',
#         "Accept-Language": "en-US,en;q=0.9",
#         "DNT": "1",
#         "Connection": "keep-alive",
#     }

#     # Standard proxy URL
#     payload = 
#     scraperapi_url = f"http://api.scraperapi.com?api_key={SCRAPERAPI_KEY}&url={url}"

#     # Premium proxy URL
#     scraperapi_url_premium = f"http://api.scraperapi.com?api_key={SCRAPERAPI_KEY}&url={url}&premium=true"

#     session = requests.Session()
#     session.headers.update(headers)

#     # Introduce a random delay to avoid detection
#     time.sleep(random.uniform(1, 5))

#     try:
#         # Initial request without premium proxy
#         initial_response = session.get(scraperapi_url, verify=False, allow_redirects=True, timeout=60)
#         initial_response.raise_for_status()

#         # If the initial request is successful, use the response directly
#         if initial_response.status_code == 200:
#             return initial_response.content

#     except requests.exceptions.RequestException as e:
#         print(f"Initial request failed: {e}")

#     # If the initial request fails, try with premium proxy and cookies
#     try:
#         # Introduce another delay before the next request
#         time.sleep(random.uniform(1, 5))

#         # Get cookies with premium proxy
#         initial_response_premium = session.get(scraperapi_url_premium, verify=False, allow_redirects=True, timeout=60)
#         initial_response_premium.raise_for_status()

#         cookies = initial_response_premium.cookies.get_dict()
#         cookies['random_cookie'] = str(random.randint(1, 1000000))

#         # Introduce another delay before the actual image request
#         time.sleep(random.uniform(1, 5))
#         response = session.get(scraperapi_url_premium, headers=headers, cookies=cookies, verify=False, allow_redirects=True, timeout=60)
#         response.raise_for_status()
#         return response.content
#     except requests.exceptions.RequestException as e:
#         print(f"Request with premium proxy failed: {e}")
#         return None
def fetch_image_with_proxy_func(url, premium=True, retries=1, render_js=False, country_code='us'):
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Referer': 'https://www.google.com/',
        "Accept-Language": "en-US,en;q=0.9",
        "DNT": "1",
        "Connection": "keep-alive",
    }

    # Randomly pick one of the API keys
    SCRAPERAPI_KEY = random.choice(SCRAPERAPI_KEYS)
    
    params = {
        'api_key': SCRAPERAPI_KEY,
        'url': url,
        'keep_headers': 'true',
        'render_js': str(render_js).lower(),  # Convert boolean to 'true'/'false' for API
        'premium': str(premium).lower(),  # Optional premium proxy
    }

    if country_code:
        params['country_code'] = country_code

    # Optional parameters: device type, JS rendering, geotargeting, etc.
    scraperapi_url = "http://api.scraperapi.com/"

    session = requests.Session()
    session.headers.update(headers)

    # Exponential backoff retry logic
    for attempt in range(1, retries + 1):
        try:
            # Introduce a random delay to avoid detection
            time.sleep(random.uniform(1, 5))

            # Make the request to ScraperAPI
            print(f"      Attempt {attempt}: Fetching URL [{url}] with{' premium' if premium else ''} proxy...")
            response = session.get(scraperapi_url, params=params, verify=False, allow_redirects=True, timeout=TIMEOUT)

            # Check if the response is successful
            if response.status_code == 200:
                return response.content  # Return the image binary content

            # If the status code isn't 200, log it and continue retrying
            print(f"      Attempt {attempt} failed with status code {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"      Attempt {attempt} failed: {e}")

        # Increase delay between retries (exponential backoff)
        backoff_delay = 2 ** attempt
        print(f"      Retrying in {backoff_delay} seconds...")
        time.sleep(backoff_delay)

    print(f"      All {retries} attempts failed to fetch the URL.")
    return None
    
@dataclass
class ImageCandidate:
    cfg: dict
    # image_row: pd.DataFrame
    # occ_row: pd.DataFrame
    occ_row: pd.DataFrame
    image_row: pd.DataFrame
    url: str
    
    failure_log: dict
    download_tracker: dict  # Shared dictionary
    completed_tracker: list  # Shared list
    banned_url_tracker: list # Shared list
    banned_url_counts_tracker: dict # Shared dictionary
    # semaphore_scraperapi: asyncio.Semaphore
    # pp: PrettyPrint # self.pp.print()  ALSO enable stuff in process_batch()
    # Initialize the asyncio.Lock with default_factory
    # lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # sync_lock: threading.Lock = field(default_factory=threading.Lock)
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
        self.lock = asyncio.Lock()  # Ensure it's correctly set here
        self.sync_lock = threading.Lock()  # Ensure it's correctly set here

        self.failure_csv_lock = threading.Lock()

        self.n_to_download = self.cfg['n_to_download']
        self.taxonomic_level = self.cfg['taxonomic_level']
        self.retries = 0
        self.backoff = 2  # Start with a 2-second delay for exponential backoff
        self.max_retries=2
        self.dir_destination = self.cfg['dir_destination_images']
        self.MP_low = self.cfg['MP_low']
        self.MP_high = self.cfg['MP_high']

        self.failure_csv_path = self.cfg['failure_csv_path']

        if self.taxonomic_level == 'family':
            self.taxonomic_unit = self.family
        elif self.taxonomic_level == 'genus':
            self.taxonomic_unit = self.genus
        else:
            self.taxonomic_unit = self.fullname
        

    
    async def get_domain(self, url):
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        return domain
    
    def get_domain_sync(self, url):
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        return domain
    
    async def is_valid_url(self):
        if not isinstance(self.url, str) or not self.url.strip():
            return False
        parsed_url = urlparse(self.url)
        return bool(parsed_url.scheme and parsed_url.netloc)
        
    async def convert_drive_url_to_direct(self, drive_url):
        """Convert a Google Drive shareable URL to a direct download link."""
        file_id = drive_url.split('/d/')[1].split('/')[0]  # Extract the file ID
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    
    async def check_for_google_drive_url(self):
        # Handle Google Drive links
        if "drive.google.com" in self.url:
            print(f"Google Drive URL detected: {self.url}")
            self.url = await self.convert_drive_url_to_direct(self.url)

    async def fetch_largest_image_from_info(self, image_service_url, headers, session):
        """Fetch the largest available image from the external info.json."""
        try:
            info_json_url = f"{image_service_url}/info.json"
            
            # Request the info.json file
            async with session.get(info_json_url, headers=headers, timeout=TIMEOUT) as response:
                if response.status == 200:
                    info_data = await response.json()

                    # Find the largest size in the 'sizes' array
                    if 'sizes' in info_data:
                        largest_size = max(info_data['sizes'], key=lambda x: x['width'])
                        width = largest_size['width']
                        height = largest_size['height']
                        
                        # Construct the full image URL for the largest size
                        largest_image_url = f"{image_service_url}/full/{width},{height}/0/default.jpg"
                        
                        return largest_image_url
                    else:
                        # If no sizes array, fall back to full image
                        return f"{image_service_url}/full/full/0/default.jpg"
                else:
                    print(f"      Failed to fetch info.json: {response.status} from {image_service_url}")
        except Exception as e:
            print(f"      Error fetching info.json from {image_service_url}: {e}")
        
        return None

    async def iiif_parse(self, manifest_url):
        domain = await self.get_domain(manifest_url)

        # Define custom headers with a random User-Agent
        headers = {
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Referer': 'https://www.google.com/',
            'Accept-Language': 'en-US,en;q=0.9',
            'DNT': '1',  # Do Not Track
            'Connection': 'keep-alive',
        }

        #"""Parse the IIIF manifest to extract the associated image URL."""
        try:
            # Make the GET request with custom headers
                    

            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                async with session.get(manifest_url, headers=headers, timeout=TIMEOUT) as response:

                    if response.status == 200:
                        manifest_data = await response.json()

                        # First check if this is the original structure with 'dwc:associatedMedia'
                        for item in manifest_data.get('metadata', []):
                            if item['label'] == 'dwc:associatedMedia':
                                associated_media_url = item['value']
                                # Handle cases where the value might contain an anchor tag with a URL
                                if 'href' in associated_media_url:
                                    associated_media_url = associated_media_url.split('"')[1]
                                return associated_media_url

                        # If no 'dwc:associatedMedia' was found, check the structure for the second case
                        if 'items' in manifest_data:
                            for canvas in manifest_data['items']:
                                for annotation_page in canvas.get('items', []):
                                    for annotation in annotation_page.get('items', []):
                                        if annotation.get('motivation') == 'painting' and 'body' in annotation:
                                            image_url = annotation['body']['id']

                                            # Check if there's an ImageService3 reference
                                            if 'service' in annotation['body']:
                                                image_service_url = annotation['body']['service'][0]['id']
                                                # Fetch the external info.json for image sizes
                                                return await self.fetch_largest_image_from_info(image_service_url, headers, session)

                                            return image_url

                    else:
                        print(f"      Failed to fetch manifest: {response.status} domain: {domain}")
        except Exception as e:
            print(f"      Error parsing manifest: domain: {domain} error: {e}")
        
        return None

    async def check_for_iiif_manifest_url(self):
        # Handle special case for URLs containing "manifest.json"
        if ("manifest.json" in self.url) or ('manifest' in self.url):
            print(f"      Manifest URL detected: {self.url}")
            # Parse the IIIF manifest to get the actual image URL
            self.url = await self.iiif_parse(self.url)
            if not self.url:
                # return None, "Failed to extract image from IIIF manifest"   
                return False
            else:
                # return 200, "Extracted image from IIIF manifest"   
                return True
        return True
        
    async def get_driver_with_random_user_agent(self):
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
        ]

        options = Options()
        user_agent = random.choice(user_agents)
        options.add_argument(f"user-agent={user_agent}")
        options.headless = True  # Run headless, without a GUI

        driver = uc.Chrome(options=options)
        driver.set_page_load_timeout(TIMEOUT)
        driver.set_script_timeout(TIMEOUT)
        return driver

    async def handle_consent_with_selenium(self, consent_url):
        if 'arctos' in consent_url:
            try:
                driver = await self.get_driver_with_random_user_agent()
                driver.get(consent_url)

                # Wait for the page to load (adjust as necessary)
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//input[@value='I agree, continue']"))
                )


                # Interact with the consent buttons
                agree_button = driver.find_element(By.XPATH, "//input[@value='I agree, continue']")
                agree_button.click()

                # Wait for the page to redirect and ensure the new URL is loaded
                WebDriverWait(driver, 10).until(EC.url_changes(consent_url))

                # Capture the current page URL after the redirect
                redirect_url = driver.current_url
                print(f"      Redirect URL after consent: {redirect_url}")
                
                return redirect_url  # Return the redirect URL for further use

            except Exception as e:
                print(f"      An error occurred while handling consent: {e}")
                return None
            finally:
                driver.quit()
        else:
            print(f"      Only [arctos] server is currently supported for 401, automate agreement errors")
    
    def handle_consent_with_selenium_sync(self, consent_url, driver):
        if 'arctos' in consent_url:
            try:
                driver.get(consent_url)

                # Wait for the page to load (adjust as necessary)
                WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//input[@value='I agree, continue']"))
                )


                # Interact with the consent buttons
                agree_button = driver.find_element(By.XPATH, "//input[@value='I agree, continue']")
                agree_button.click()

                # Wait for the page to redirect and ensure the new URL is loaded
                WebDriverWait(driver, 10).until(EC.url_changes(consent_url))

                # Capture the current page URL after the redirect
                redirect_url = driver.current_url
                print(f"      Redirect URL after consent: {redirect_url}")
                
                return redirect_url  # Return the redirect URL for further use

            except Exception as e:
                print(f"      An error occurred while handling consent: {e}")
                return None
        else:
            print(f"      Only [arctos] server is currently supported for 401, automate agreement errors")

    async def is_base64_image(self, image_data):
        if isinstance(image_data, str) and image_data.startswith('data:image/'):
            return True
        return False

    async def decode_base64_image(self, image_data):
        # The base64 image data typically follows this pattern: data:image/{type};base64,{encoded_data}
        header, encoded_data = image_data.split(',', 1)
        
        # Determine the extension from the base64 header
        if 'image/jpeg' in header or 'image/jpg' in header:
            ext = 'jpg'
        elif 'image/png' in header:
            ext = 'png'
        elif 'image/jfif' in header:
            ext = 'jfif'
        elif 'image/tiff' in header:
            ext = 'tiff'
        else:
            ext = 'unknown'

        return base64.b64decode(encoded_data), ext

    # async def save_base64_image(self, image_data, target_path):
    #     """Save a decoded base64 image to a file."""
    #     decoded_image, ext = await self.decode_base64_image(image_data)

    #     # Save the image to the specified path
    #     with open(target_path, 'wb') as img_file:
    #         img_file.write(decoded_image)
    #     print(f"Base64 image saved to {target_path}")

    async def download_image_from_dynamic_page(self, page_url):
        #"""Extracts the high-resolution image or canvas image using Selenium."""
        driver = await self.get_driver_with_random_user_agent()
        try:
            driver.get(page_url)

            # Try to extract the high-resolution image from the hidden input field
            try:
                txt_file_name = driver.find_element(By.ID, "txtFileName").get_attribute("value")
                
                # Dynamically construct the full URL based on the page_url
                high_res_image_url = urljoin(page_url, txt_file_name)
                print(f"            High-resolution image URL extracted: {high_res_image_url}")

                # Download the high-resolution image using the extracted URL
                async with aiohttp.ClientSession() as session:
                    async with session.get(high_res_image_url) as response:
                        if response.status == 200:
                            image_data = await response.read()  
                            image_data = BytesIO(await response.read())
                            return image_data, 'jpg', high_res_image_url
                        else:
                            print(f"            Failed to download high-resolution image. Status code: {response.status}")
            except NoSuchElementException:
                print("            High-resolution image not found, falling back to canvas extraction.")

            # Fall back to canvas image extraction if high-res image is not available
            try:
                canvas_element = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, 'canvas'))  # Adjust the selector if necessary
                )
                print("            Canvas element found, extracting image...")

                # Use JavaScript to extract the data from the canvas
                image_url = driver.execute_script("""
                    var canvas = arguments[0];
                    return canvas.toDataURL('image/jpeg');
                """, canvas_element)

                # Handle Base64 encoded data URL
                if image_url.startswith("data:image"):
                    ext = image_url.split(";")[0].split("/")[1]  # Extract extension from the data URL
                    print(f"            Extracted image format: {ext}")

                    # Extract the base64-encoded part
                    base64_data = image_url.split(",")[1]
                    image_data = BytesIO(base64.b64decode(base64_data))  # Convert to a BytesIO object

                    return image_data, ext, page_url  # Return image data and extension for further processing
                else:
                    print("            No valid image data found in the canvas element.")
                    return None, None, page_url

            except TimeoutException:
                print("            Timed out waiting for the canvas element to load.")
                return None, None, page_url

        except Exception as e:
            print(f"            Error extracting image: {e}")
            return None, None, page_url

        finally:
            driver.quit()  # No need to await here, just call the method

    async def make_https(self, url):
        """Converts an http URL to https if it's not already https."""
        parsed_url = urlparse(url)
        if parsed_url.scheme == 'http':
            # Rebuild the URL with https instead of http
            https_url = parsed_url._replace(scheme='https')
            return urlunparse(https_url)
        return url  # Return the URL unchanged if it is already https

    async def log_failed_download(self, reason):
        """
        Append a failed download entry to the failure CSV file.
        Args:
            reason (str): Reason for the failure.
        """
        print(f"{bcolors.HEADER}LOGGING FAILURE{bcolors.ENDC}")
        with self.failure_csv_lock:
            try:
                with open(self.failure_csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([self.filename_image_jpg, self.url, reason])
                    print(f"{bcolors.FAIL}Logged failure: {self.filename_image_jpg}, Reason: {reason}{bcolors.ENDC}")
            except Exception as e:
                print(f"{bcolors.FAIL}Failed to log failure: {e}{bcolors.ENDC}")



    async def handle_response(self, session, response, n_queue):
        if response.status == 404:
            # return response.status, f"Error 404"
            self.download_success = 'skip'
            await self.log_failed_download("404 Not Found")
            print(f"            Skipping Error 404 URL {self.url}")
            return

        if response.status == 403:
            self.download_success = 'skip'
            await self.log_failed_download("403 Forbidden")
            print(f"            Skipping Error 403 URL {self.url}")
            return
        

        # Check if we received a 401 status code and handle consent
        if response.status == 401:
            self.initial_code = response.status
            print(f"      401 Unauthorized detected. Handling consent...  {self.url}")
            if self.url:
                redirect_url = await self.handle_consent_with_selenium(self.url)
                if redirect_url:
                    response = session.get(redirect_url, timeout=TIMEOUT)
                    print(f"            Retrying with the redirect URL {redirect_url}")
                else:
                    print(f"            Failed to retrieve the redirect URL after consent.  SKIPPING {self.url}")
                    await self.log_failed_download("401 Unauthorized - Failed to retrieve the redirect URL after consent")
                    # return response.status, f"Failed to download image. Status code: {response.status}"
                    self.download_success = 'skip'
                    # self.download_success = False
                    return
            else:
                await self.log_failed_download("401 Unauthorized - No URL")
                self.download_success = 'skip'
                return
            


        # Check if the image URL is base64-encoded
        if await self.is_base64_image(self.url):
            print("      Base64-encoded image detected.")
            decoded_image, ext = await self.decode_base64_image(self.url)
            # Convert to a PIL Image
            try:
                img = Image.open(BytesIO(decoded_image))
            except:
                print(f"            Failed to save binary image, GBIF ID: {self.taxonomic_unit}  {self.url}")
                self.download_success = 'skip'
                return

            was_saved, reason = await self._save_matching_image(img, self.MP_low, self.MP_high, self.dir_destination)
            if was_saved:
                await self._add_occ_and_img_data() # IF saved, add row to csv
                print(f"{bcolors.OKGREEN}      SUCCESS binary image attempt {self.retries + 1} [Q {n_queue}]  {self.url}{bcolors.ENDC}")
                self.download_success = True
                if self.download_success:
                    async with self.lock:
                        current_count = self.download_tracker.get(self.taxonomic_unit, 0)
                        updated_count = current_count + 1
                        self.download_tracker.update({self.taxonomic_unit: updated_count})
                        print(f"{bcolors.OKGREEN}      SUCCESS {self.taxonomic_unit} has {updated_count} images  {self.url}{bcolors.ENDC}")
                return 
            else:
                print(f"      Failed to save binary image, GBIF ID: {self.taxonomic_unit} {self.url}")
                await self.log_failed_download(reason)
                self.download_success = 'skip'
                return



        # Get the content type to figure out the image format for non-base64 images
        content_type = response.headers.get('Content-Type')
        if not content_type:
            content_type = 'missing'

        if 'text/html' in content_type:
            print(f"      HTML content detected. Parsing for image... {self.url}")

            # Try to extract the image from the dynamic page
            image_data, ext, new_dynamic_url = await self.download_image_from_dynamic_page(self.url)

            if image_data:
                # ext = "jpg"  # Ensure extension is jpg
                # target_path = os.path.join(target_dir, f"{fname}.{ext}")

                # Convert and save the image as JPEG
                img = Image.open(image_data)
                img = img.convert("RGB")  # Convert to RGB if needed
                # img.save(target_path, 'JPEG', quality=95)
                was_saved, reason = await self._save_matching_image(img, self.MP_low, self.MP_high, self.dir_destination)

                if was_saved:
                    if new_dynamic_url != self.url:
                        # return 200, f"Downloaded image. Extracted HIGHRES url from dynamic image viewer."
                        await self._add_occ_and_img_data() # IF saved, add row to csv
                        print(f"{bcolors.OKGREEN}            Downloaded image. Extracted HIGHRES url from dynamic image viewer. {self.retries + 1} [Q {n_queue}] {self.url}{bcolors.ENDC}")
                        self.download_success = True
                        if self.download_success:
                            async with self.lock:
                                current_count = self.download_tracker.get(self.taxonomic_unit, 0)
                                updated_count = current_count + 1
                                self.download_tracker.update({self.taxonomic_unit: updated_count})
                                print(f"{bcolors.OKGREEN}      SUCCESS {self.taxonomic_unit} has {updated_count} images {self.url}{bcolors.ENDC}")
                        return 
                    else:
                        # return 200, f"Downloaded image. Extracted canvas LOWRES url from dynamic image viewer."
                        await self._add_occ_and_img_data() # IF saved, add row to csv
                        print(f"{bcolors.OKGREEN}            Downloaded image. Extracted canvas LOWRES url from dynamic image viewer. {self.retries + 1} [Q {n_queue}] {self.url}{bcolors.ENDC}")
                        self.download_success = True
                        if self.download_success:
                            async with self.lock:
                                current_count = self.download_tracker.get(self.taxonomic_unit, 0)
                                updated_count = current_count + 1
                                self.download_tracker.update({self.taxonomic_unit: updated_count})
                                print(f"{bcolors.OKGREEN}      SUCCESS {self.taxonomic_unit} has {updated_count} images {self.url}{bcolors.ENDC}")
                        return 
                else:
                    print(f"      Failed to save image from dynamic viewer, GBIF ID: {self.taxonomic_unit} {self.url}")
                    await self.log_failed_download(reason)
                    self.download_success = 'skip'
                    return
            else:
                print(f"      Failed to save image from dynamic viewer, no image_data, GBIF ID: {self.taxonomic_unit} {self.url}")
                await self.log_failed_download("No image data in dynamic page")
                self.download_success = 'skip'
                return

        elif 'image/jpeg' in content_type or 'image/jpg' in content_type:
            ext = 'jpg'
        elif 'image/png' in content_type:
            ext = 'png'
        elif 'image/jfif' in content_type:
            ext = 'jfif'
        elif 'image/tiff' in content_type:
            ext = 'tiff'
        else:
            ext = 'unknown'

        print(f"            Image format detected: {ext} {self.url}")

        # Use a placeholder filename based on the image URL and content type
        # image_name = f"{fname}.jpg"
        # target_path = os.path.join(target_dir, image_name)

        # Read the image content from the response
        image_data = await response.read()  # Now image_data is a bytes object
        image_data = BytesIO(image_data)
        
        try:
            # Open the image and convert it to JPEG format if necessary
            img = Image.open(image_data)
            if ext != 'jpg':
                # target_path = os.path.join(target_dir, fname + f'_converted_from_{ext}' + ".jpg")
                img = img.convert("RGB")  # Convert to RGB if needed (e.g., for PNG or TIFF)
            
            # Save the image as JPEG
            # img.save(target_path, 'JPEG', quality=95)
            was_saved, reason = await self._save_matching_image(img, self.MP_low, self.MP_high, self.dir_destination)

            # print(f"      Image successfully saved as JPG to {target_path}")
            # return response.status, f"Downloaded image. Final - {response.status} Initial - {initial_code}"
            if was_saved:
                await self._add_occ_and_img_data() # IF saved, add row to csv
                print(f"{bcolors.OKGREEN}      Downloaded image. Final - {response.status} Initial - {self.initial_code} {self.retries + 1} [Q {n_queue}] {self.url}{bcolors.ENDC}")
                self.download_success = True
                if self.download_success:
                    async with self.lock:
                        current_count = self.download_tracker.get(self.taxonomic_unit, 0)
                        updated_count = current_count + 1
                        self.download_tracker.update({self.taxonomic_unit: updated_count})
                        print(f"{bcolors.OKGREEN}      SUCCESS {self.taxonomic_unit} has {updated_count} images {self.url}{bcolors.ENDC}")
                return 
            else:
                print(f"      Failed to save image status.code {response.status}, REASON [{reason}], GBIF ID: {self.taxonomic_unit} {self.url}")
                await self.log_failed_download(f"Failed to save image status.code {response.status}, REASON [{reason}], GBIF ID: {self.taxonomic_unit} {self.url}")

                if reason == "too_small" and 'too_small' in self.banned_url_counts_tracker:
                    self.banned_url_counts_tracker['too_small_direct'] += 1
                    self.download_success = 'skip'
                elif reason == "too_small":
                    self.banned_url_counts_tracker['too_small_direct'] = 1
                    self.download_success = 'skip'
                else:
                    if reason in self.banned_url_counts_tracker:
                        self.banned_url_counts_tracker[reason] += 1
                    else:
                        self.banned_url_counts_tracker[reason] = 1
                    self.download_success = 'skip'
                    # self.download_success = False
                return
            
        except Exception as e:
            print(f"      Failed to save image status.code {response.status}, {e}, GBIF ID: {self.taxonomic_unit} {self.url}")
            # self.download_success = False
            self.download_success = 'skip'
            return

    async def iiif_parse(manifest_url):
        parsed_url = urlparse(manifest_url)
        domain = parsed_url.netloc  # Extract the domain (netloc)

        # Define custom headers with a random User-Agent
        headers = {
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Referer': 'https://www.google.com/',
            'Accept-Language': 'en-US,en;q=0.9',
            'DNT': '1',  # Do Not Track
            'Connection': 'keep-alive',
        }

        """Parse the IIIF manifest to extract the associated image URL."""
        try:
            # Make the GET request with custom headers
            response = requests.get(manifest_url, headers=headers)
            if response.status_code == 200:
                manifest_data = response.json()

                # First check if this is the original structure with 'dwc:associatedMedia'
                for item in manifest_data.get('metadata', []):
                    if item['label'] == 'dwc:associatedMedia':
                        associated_media_url = item['value']
                        # Handle cases where the value might contain an anchor tag with a URL
                        if 'href' in associated_media_url:
                            associated_media_url = associated_media_url.split('"')[1]
                        return associated_media_url

                # If no 'dwc:associatedMedia' was found, check the structure for the second case
                if 'items' in manifest_data:
                    for canvas in manifest_data['items']:
                        for annotation_page in canvas.get('items', []):
                            for annotation in annotation_page.get('items', []):
                                if annotation.get('motivation') == 'painting' and 'body' in annotation:
                                    image_url = annotation['body']['id']
                                    return image_url

            else:
                print(f"Failed to fetch manifest: {response.status_code} domain: {domain}")
        except Exception as e:
            print(f"Error parsing manifest: domain: {domain} error: {e}")
        
        return None




    async def download_image(self, session: ClientSession, n_queue, logging_enabled=False, timeout=TIMEOUT) -> None:
        # SETTINGS
        self.logging_enabled = logging_enabled 
        
        # Skip GBIF Taxa if not ID'd to species level
        if self.fullname is None:
            self.download_success = 'skip'
            return
        
        current_count = self.download_tracker.get(self.taxonomic_unit, 0)

        # COUNT
        if current_count >= self.n_to_download:
            print(f"      Skipping download for {self.taxonomic_unit} as max number of images {current_count}/{self.n_to_download} have already been downloaded.")
            self.download_success = True
            return  
        
        # Check if the URL stem is in the banned list
        if not await self.is_valid_url():
            print(f"{bcolors.BOLD}      Invalid URL --> {self.url}, GBIF ID: {self.taxonomic_unit}{bcolors.ENDC}")
            self.download_success = False
            return 

        # Check for banned domain
        domain = await self.get_domain(self.url)
        if (domain in BANNED) or (domain in self.banned_url_tracker):
            self.download_success = 'skip'
            return
        
        # Log
        if self.logging_enabled:
            http_client.HTTPConnection.debuglevel = 1
            logging.basicConfig(level=logging.DEBUG)
            requests_log = logging.getLogger("aiohttp.client")
            requests_log.setLevel(logging.CRITICAL)
            requests_log.propagate = True
            print(f"{bcolors.BOLD}      {self.taxonomic_unit}{bcolors.ENDC}")
            print(f"{bcolors.BOLD}           URL: {self.url}{bcolors.ENDC}")

        await asyncio.sleep(random.uniform(0, 5))  # Random delay between 0 and 5 seconds

        await self.check_for_google_drive_url() # updates self.url
        
        if 'arctos' in self.url:
            print(f'Is arctos {self.url}')
            self.url = await self.make_https(self.url)

        if ("manifest.json" in self.url) or ('manifest' in self.url):
            print(f"IIIF Manifest URL detected: {self.url}")
            # Parse the IIIF manifest to get the actual image URL
            self.url = await self.iiif_parse(self.url)
            if not self.url:
                self.download_success = 'skip'
                print("Failed to extract image from IIIF manifest")
                return
        
        self.initial_code = 999
        while self.retries < self.max_retries:
            try:
                if self.retries == 1:
                    # First attempt with custom headers
                    headers = {
                        'User-Agent': random.choice(USER_AGENTS),
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Referer': 'https://www.google.com/',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'DNT': '1',
                        'Connection': 'keep-alive',
                    }
                    print("      Using custom headers")
                else:
                    # Second attempt without custom headers
                    headers = None
                    print(f"            Trying without custom headers for {self.url}")
                
                async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
                    # async with session.get(self.url, headers=headers, timeout=TIMEOUT) as response:
                    # async with session.get(self.url, timeout=TIMEOUT) as response:
                    if not headers:
                        async with session.get(self.url, timeout=TIMEOUT) as response:
                            print(f"      Response status: {response.status}")
                            await self.handle_response(session, response, n_queue)
                    else:
                        async with session.get(self.url, headers=headers, timeout=TIMEOUT) as response:
                            print(f"      Response status: {response.status}")
                            await self.handle_response(session, response, n_queue)
                            
                
                if (self.download_success == True) or (self.download_success == 'skip'):
                    return
                else:
                    self.retries += 1
                    self.backoff *= 2  # Exponential backoff
                    await asyncio.sleep(self.backoff)
                            
            except aiohttp.ClientError as e:
                self.retries += 1
                self.backoff *= 2  # Exponential backoff
                await asyncio.sleep(self.backoff)
                print(f"      Client error occurred: {e}. Retrying {self.retries}/{self.max_retries}...")

            except (ConnectionError, Timeout) as e:
                self.retries += 1
                self.backoff *= 2  # Exponential backoff
                await asyncio.sleep(self.backoff)
                print(f"      Connection error occurred: {e}. Retrying {self.retries}/{self.max_retries}...")

            except HTTPError as http_err:
                self.retries += 1
                self.backoff *= 2  # Exponential backoff
                await asyncio.sleep(self.backoff)
                print(f"      HTTP error occurred: {http_err}")

            except RequestException as req_err:
                self.retries += 1
                self.backoff *= 2  # Exponential backoff
                await asyncio.sleep(self.backoff)
                print(f"      Request exception occurred: {req_err}")

            except asyncio.TimeoutError:
                self.retries += 1
                self.backoff *= 2  # Exponential backoff
                await asyncio.sleep(self.backoff)
                print(f"      Connection timeout error occurred. Retrying {self.retries}/{self.max_retries}...")

            except Exception as e:
                print(f"      An unexpected error occurred: {e}")
                self.retries += 1
                self.backoff *= 2  # Exponential backoff
                await asyncio.sleep(self.backoff)
            
        # If the max retries are exhausted
        if self.download_success == False:
            self.download_success = "skip"
            # Set to skip to keep it from going to ScraperAPI
            await self.log_failed_download(f"Retries exceeded: {self.retries + 1} STATUS[{self.initial_code}] URL: {self.url}, GBIF ID: {self.taxonomic_unit}")
            print(f"{bcolors.CYELLOWBG}      Retries exceeded: {self.retries + 1} STATUS[{self.initial_code}] URL: {self.url}, GBIF ID: {self.taxonomic_unit}{bcolors.ENDC},")
            return 
        elif self.download_success == 'skip':
            await self.log_failed_download(f"Retries exceeded: {self.retries + 1} STATUS[{self.initial_code}] URL: {self.url}, GBIF ID: {self.taxonomic_unit}")
            print(f"{bcolors.CYELLOWBG}      Skipping: {self.retries + 1} STATUS[{self.initial_code}] URL: {self.url} GBIF ID: {self.taxonomic_unit}{bcolors.ENDC},")
            return 
        else:
            return 















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

    def download_base64_image_sync(self, url):
        was_saved = False
        try:
            header, encoded = url.split(",", 1)
            data = base64.b64decode(encoded)
            img = Image.open(BytesIO(data))
            dir_destination = self.cfg['dir_destination_images']
            MP_low = self.cfg['MP_low']
            MP_high = self.cfg['MP_high']
            
            was_saved, reason = self._save_matching_image_sync(img, MP_low, MP_high, dir_destination)
            self.download_success = True if was_saved else False
        except Exception as e:
            print(f"Error saving base64 image: {e}")
            self.download_success = False
        return was_saved, reason
    




    async def download_image_with_selenium(self, driver, index, n_queue, timeout_duration=TIMEOUT, max_strikes=50) -> bool:
        await asyncio.to_thread(self.download_image_with_selenium_sync, driver, index, n_queue, timeout_duration, max_strikes)

    def download_image_with_selenium_sync(self, driver, index, n_queue, timeout_duration, max_strikes=50):

        print(f"{bcolors.CWHITEBG}      RETRY [i {index}] [Q {n_queue}] {self.url} GBIF ID: {self.taxonomic_unit}{bcolors.ENDC}")
        self.download_success = False

        current_count = self.download_tracker.get(self.taxonomic_unit, 0)
        domain = self.get_domain_sync(self.url)

        if current_count >= self.n_to_download or self.taxonomic_unit in self.completed_tracker:
            print(f"Skipping download for {self.taxonomic_unit} as max number of images {current_count}/{self.n_to_download} have already been downloaded.")
            self.download_success = True
            return 
        
        if current_count < self.n_to_download:
            if ((not isinstance(self.url, str) or pd.isna(self.url)) and n_queue != 0):
                print(f"{bcolors.BOLD}      Invalid URL --> {self.url}, GBIF ID: {self.taxonomic_unit}{bcolors.ENDC}")
                self.download_success = 'skip'
                return 
            
            with self.sync_lock:
                if domain in self.banned_url_tracker:
                    print(f"{bcolors.CGREYBG}      ******URL stem is banned --> {self.url}, GBIF ID: {self.taxonomic_unit}{bcolors.ENDC}")
                    self.download_success = 'skip'
                    return 

            # Ensure the URL is a valid image URL and not a user agent string
            if "user-agent" in self.url.lower():
                print(f"{bcolors.BOLD}      Invalid URL detected (User-Agent) --> {self.url}, GBIF ID: {self.taxonomic_unit}{bcolors.ENDC}")
                self.download_success = False
                return 


            dir_destination = self.cfg['dir_destination_images']
            MP_low = self.cfg['MP_low']
            MP_high = self.cfg['MP_high']

            retries = 1
            for attempt in range(retries):
                
                time.sleep(10 * (attempt + 1))  
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

                    # Check for a 401 Unauthorized response
                    if "401" in driver.page_source or driver.current_url.endswith("401"):
                        print(f"{bcolors.WARNING}      401 Unauthorized detected for {self.url}, GBIF ID: {self.taxonomic_unit}{bcolors.ENDC}")

                        # Handle the 401 consent with Selenium
                        new_url = self.handle_consent_with_selenium_sync(self.url, driver)
                        if new_url:
                            print(f"{bcolors.CVIOLETBG}      Redirected to new URL after consent: {new_url} {bcolors.ENDC}")
                            self.url = new_url
                            driver.get(self.url)
                        else:
                            print(f"{bcolors.FAIL}      Failed to handle 401 consent for {self.url}, GBIF ID: {self.taxonomic_unit}{bcolors.ENDC}")
                            self.download_success = 'skip'
                            return

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
                        # was_saved, reason = await self.download_base64_image(img_url)
                        was_saved, reason = self.download_base64_image_sync(img_url)  # Sync version

                    else:
                        response = requests.get(img_url)
                        response.raise_for_status()
                        img = Image.open(BytesIO(response.content))

                        ### Exit early if the count has increased while working
                        current_count = self.download_tracker.get(self.taxonomic_unit, 0)
                        if current_count >= self.n_to_download:
                            print(f"Skipping urgent download for {self.taxonomic_unit} as max number of images {current_count}/{self.n_to_download} have already been downloaded.")
                            self.download_success = True
                            return 

                        was_saved, reason = self._save_matching_image_sync(img, MP_low, MP_high, dir_destination, is_retry=True)
                    
                    if reason == "too_small" and 'too_small' in self.banned_url_counts_tracker:
                        self.banned_url_counts_tracker['too_small_selenium'] += 1
                    elif reason == "too_small":
                        self.banned_url_counts_tracker['too_small_selenium'] = 1

                    if was_saved:
                        self._add_occ_and_img_data_sync() # IF saved, add row to csv
                        print(f"{bcolors.OKCYAN}      SUCCESS RETRY attempt {attempt + 1} [Q {n_queue}]{bcolors.ENDC}")
                        self.download_success = True
                        if self.download_success:
                            with self.sync_lock:
                                current_count = self.download_tracker.get(self.taxonomic_unit, 0)
                                updated_count = current_count + 1
                                self.download_tracker.update({self.taxonomic_unit: updated_count})
                                print(f"{bcolors.OKCYAN}      SUCCESS {self.taxonomic_unit} has {updated_count} images{bcolors.ENDC}")
                        return 

                    else:
                        print(f"Failed to save the image RETRY: {self.url}, GBIF ID: {self.taxonomic_unit}")


                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 503:
                        print(f"{bcolors.WARNING}      SKIP all retries due to 503 error --> {e} --> URL: {self.url}, GBIF ID: {self.taxonomic_unit}{bcolors.ENDC}")
                        self._log_failure("HTTPError503", self.url, e)
                        return   # Skip all retries if 503 error occurs

                except (WebDriverException, NoSuchElementException, TimeoutException, 
                        StaleElementReferenceException, ElementNotInteractableException, 
                        ElementClickInterceptedException, InvalidElementStateException, 
                        NoSuchFrameException, NoSuchWindowException) as e:
                    if attempt == retries - 1:
                        print(f"{bcolors.FAIL}      Selenium Exception {attempt + 1} --> {e} --> URL: {self.url}, GBIF ID: {self.taxonomic_unit}{bcolors.ENDC}")
                    else:
                        print(f"{bcolors.OKBLUE}      Selenium Exception {attempt + 1} --> {e} --> URL: {self.url}, GBIF ID: {self.taxonomic_unit}{bcolors.ENDC}")

                except ImageSaveError as e:
                    if attempt == retries - 1:
                        print(f"{bcolors.FAIL}      attempt {attempt + 1} {e} --> URL: {self.url}, GBIF ID: {self.taxonomic_unit}{bcolors.ENDC}")
                    else:
                        print(f"{bcolors.OKBLUE}      attempt {attempt + 1} {e} --> URL: {self.url}, GBIF ID: {self.taxonomic_unit}{bcolors.ENDC}")

                except Exception as e:
                    if attempt == retries - 1:
                        print(f"{bcolors.FAIL}      SKIP attempt {attempt + 1} --- No Connection or Rate Limited --> {e} --> URL: {self.url}, GBIF ID: {self.taxonomic_unit}{bcolors.ENDC}")
                    else:
                        print(f"{bcolors.OKBLUE}      SKIP attempt {attempt + 1} --- No Connection or Rate Limited --> {e} --> URL: {self.url}, GBIF ID: {self.taxonomic_unit}{bcolors.ENDC}")
            
                with self.sync_lock:
                    if domain in self.banned_url_tracker:
                        print(f"{bcolors.CGREYBG}      ***URL stem is banned --> {self.url}, GBIF ID: {self.taxonomic_unit}{bcolors.ENDC}")
                        self.download_success = 'skip'
                        return 
                
            
                # self.download_success = 'proxy'
                # return 
                # If all retries with Selenium fail, attempt to download using ScraperAPI
                print(f"{bcolors.BOLD}      All retries with Selenium failed. Attempting to download using ScraperAPI...{bcolors.ENDC}")
                MP_low = self.cfg['MP_low']
                MP_high = self.cfg['MP_high']
                dir_destination = self.cfg['dir_destination_images']

                for attempt in range(1):  # Retry up to 3 times for ScraperAPI requests
                    try:
                        
                        # async with semaphore_scraperapi:
                            # image_data = fetch_image_with_proxy_func(self.url)
                        image_data = fetch_image_with_proxy_func(self.url)
                        if image_data:
                            # Check if image_data is base64 encoded
                            if image_data.startswith(b'data:image'):
                                was_saved, reason = self.download_base64_image_sync(image_data.decode('utf-8'))
                            else:
                                try:
                                    img = Image.open(BytesIO(image_data))
                                except UnidentifiedImageError as e:
                                    print(f"{bcolors.CREDBG}      Error identifying image file with proxy: {e} URL: {self.url}{bcolors.ENDC}")
                                    self.download_success = 'skip'
                                    break  # Skip further processing for this image

                                was_saved, reason = self._save_matching_image_sync(img, MP_low, MP_high, dir_destination, is_retry=True)

                        if reason == "too_small" and 'too_small' in self.banned_url_counts_tracker:
                            self.banned_url_counts_tracker['too_small_proxy'] += 1
                        elif reason == "too_small":
                            self.banned_url_counts_tracker['too_small_proxy'] = 1
                                
                        if was_saved:
                            self._add_occ_and_img_data_sync()
                            self.download_success = True

                            with self.sync_lock:
                                current_count = self.download_tracker.get(self.taxonomic_unit, 0)
                                updated_count = current_count + 1
                                self.download_tracker.update({self.taxonomic_unit: updated_count})
                                print(f"{bcolors.CGREENBG2}      SUCCESS WITH PROXY {self.taxonomic_unit} has {updated_count} images  URL: {self.url}{bcolors.ENDC}")
                            return True
                        else:
                            print(f"{bcolors.WARNING}      No image data returned from ScraperAPI for URL: {self.url}{bcolors.ENDC}")

                    except requests.exceptions.RequestException as e:
                        print(f"{bcolors.CREDBG}      Request failed: {e} URL: {self.url}{bcolors.ENDC}")
                    except Exception as e:
                        print(f"{bcolors.CREDBG}      Error in proxy attempt {attempt + 1}: {e}{self.banned_url_counts_tracker[domain]} URL: {self.url}{bcolors.ENDC}")
                    time.sleep(2 ** attempt)  
                    


                print(f"{bcolors.CBEIGE}      ADDING TO BANNED URL LIST{bcolors.ENDC}")
                # Update banned_url_counts_tracker
                with self.sync_lock:
                    if domain in self.banned_url_counts_tracker:
                        self.banned_url_counts_tracker[domain] += 1
                    else:
                        self.banned_url_counts_tracker[domain] = 1
                    print(f"{bcolors.CREDBG}      URL stem {domain} COUNT {self.banned_url_counts_tracker[domain]} URL: {self.url}{bcolors.ENDC}")

                    # Check if the count exceeds the threshold
                    if self.banned_url_counts_tracker[domain] > max_strikes:
                        if domain not in self.banned_url_tracker:
                            self.banned_url_tracker.append(domain)
                            print(f"{bcolors.CREDBG}      URL stem {domain} added to banned_url_tracker{bcolors.ENDC}")
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
        domain = await self.get_domain(self.url)

        for attempt in range(1):  # Retry up to 1 times for ScraperAPI requests
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
                            current_count = self.download_tracker.get(self.taxonomic_unit, 0)
                            updated_count = current_count + 1
                            self.download_tracker.update({self.taxonomic_unit: updated_count})
                            print(f"{bcolors.CGREENBG2}      SUCCESS WITH PROXY {self.taxonomic_unit} has {updated_count} images{bcolors.ENDC}")
                        return True
            except requests.exceptions.RequestException as e:
                print(f"{bcolors.CREDBG}      Request failed: {e}{bcolors.ENDC}")
            except Exception as e:
                print(f"{bcolors.CREDBG}      Error in proxy attempt {attempt + 1}: {e}{self.banned_url_counts_tracker[domain]}{bcolors.ENDC}")
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

        print(f"{bcolors.CBEIGE}      ADDING TO BANNED URL LIST{bcolors.ENDC}")
        # Update banned_url_counts_tracker
        async with self.lock:
            if domain in self.banned_url_counts_tracker:
                self.banned_url_counts_tracker[domain] += 1
            else:
                self.banned_url_counts_tracker[domain] = 1
            print(f"{bcolors.CREDBG}      URL stem {domain} COUNT {self.banned_url_counts_tracker[domain]}{bcolors.ENDC}")

            # Check if the count exceeds the threshold
            if self.banned_url_counts_tracker[domain] > max_strikes:
                if domain not in self.banned_url_tracker:
                    self.banned_url_tracker.append(domain)
                    print(f"{bcolors.CREDBG}      URL stem {domain} added to banned_url_tracker{bcolors.ENDC}")
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
            img.save(image_path, quality=95)
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
            img.save(image_path, quality=95)
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

    def _save_matching_image_sync(self, img, MP_low, MP_high, dir_destination, is_retry=False):
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
            img.save(image_path, quality=95)
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
            img.save(image_path, quality=95)
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

    def _add_occ_and_img_data_sync(self) -> None:
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
        self._append_combined_occ_image_sync(combined)

    def _append_combined_occ_image_sync(self, combined) -> None:
        path_csv_combined = os.path.join(self.cfg['dir_destination_csv'], self.cfg['filename_combined'])
        with self.lock:
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
            img.save(image_path, quality=95)
            await self._add_occ_and_img_data()
            print(f"{color}                Regular MP: {img_mp}{bcolors.ENDC}")
            print(f"{color}                URL: {self.url}{bcolors.ENDC}")
            print(f"{color}                Image Saved: {image_path}{bcolors.ENDC}")
            return True

        if img_mp > MP_high and self.cfg['do_resize']:
            img_w, img_h = calc_resize(img_w, img_h)
            img = img.resize((img_w, img_h))
            image_path = os.path.join(dir_destination, self.filename_image_jpg)
            img.save(image_path, quality=95)
            await self._add_occ_and_img_data()
            print(f"{color}                {MP_high}MP+ Resize: {img_mp}{bcolors.ENDC}")
            print(f"{color}                URL: {self.url}{bcolors.ENDC}")
            print(f"{color}                Image Saved: {image_path}{bcolors.ENDC}")
            return True

        print(f"{bcolors.CVIOLETBG}                {MP_high}MP+ Resize: {img_mp}{bcolors.ENDC}")
        print(f"{bcolors.CVIOLETBG}                URL: {self.url}{bcolors.ENDC}")
        print(f"{bcolors.CVIOLETBG}                SKIP: {image_path}{bcolors.ENDC}")
        return False

