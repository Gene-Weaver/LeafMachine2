import requests, os, inspect, sys, time, logging, asyncio
import random
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup as BS
from urllib.parse import urljoin, urlparse
# need lxml



from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import undetected_chromedriver as uc
import base64
from requests.exceptions import ConnectionError, Timeout, HTTPError, RequestException
from bs4 import BeautifulSoup  

currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)

from leafmachine2.machine.general_utils import bcolors, get_cfg_from_full_path
# Initialize ScraperAPI client
# cfg_private = get_cfg_from_full_path('/media/data/Dropbox/LeafMachine2/PRIVATE_DATA.yaml')
# cfg_private = get_cfg_from_full_path('D:/Dropbox/LeafMachine2/PRIVATE_DATA.yaml') 
try:
    # Attempt to load the private data file
    cfg_private = get_cfg_from_full_path(os.path.join(parentdir, 'PRIVATE_DATA.yaml'))
except FileNotFoundError:
    # Raise an error if the file is not found
    raise FileNotFoundError("The private data file 'PRIVATE_DATA.yaml' cannot be found. This file is required for scraperAPI to function.")
except Exception as e:
    # Handle any other unexpected exceptions
    raise RuntimeError(f"An unexpected error occurred while trying to load 'PRIVATE_DATA.yaml': {e}")

# SCRAPERAPI_KEY = cfg_private['SCRAPERAPI_KEY1']
# SCRAPERAPI_KEY = cfg_private['SCRAPERAPI_KEY2']
SCRAPERAPI_KEY = cfg_private['SCRAPERAPI_KEY3']

logging.basicConfig(level=logging.INFO)

# Proxy configuration using the new format
proxies = {
    "http": f"scraperapi.premium=true.ultra_premium=true.country_code=us.device_type=desktop:{SCRAPERAPI_KEY}@proxy-server.scraperapi.com:8001"
}

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
    # 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
    # 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',
    # 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
    # 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    # 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36',
    # 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15; rv:86.0) Gecko/20100101 Firefox/86.0',
    # 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36',
    # 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36',
    # 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36',
    # 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    # 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
    # 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:90.0) Gecko/20100101 Firefox/90.0',
    # 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.192 Safari/537.36',
    # 'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; AS; rv:11.0) like Gecko',
    # 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36',
    # 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.2 Safari/605.1.15',
    # 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
    # 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36',
]

def get_driver_with_random_user_agent():
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
    driver.set_page_load_timeout(60)
    driver.set_script_timeout(60)
    return driver

# Setup the Selenium function for handling the 401 consent page
def handle_consent_with_selenium(consent_url):
    if 'arctos' in consent_url:
        try:
            driver = get_driver_with_random_user_agent()
            driver.get(consent_url)

            # Wait for the page to load (adjust as necessary)
            time.sleep(3)

            # Interact with the consent buttons
            agree_button = driver.find_element(By.XPATH, "//input[@value='I agree, continue']")
            agree_button.click()

            # Wait for the consent to process and the redirect to happen
            time.sleep(3)

            # Capture the current page URL after the redirect
            redirect_url = driver.current_url
            print(f"Redirect URL after consent: {redirect_url}")
            
            return redirect_url  # Return the redirect URL for further use

        except Exception as e:
            print(f"An error occurred while handling consent: {e}")
            return None
        finally:
            driver.quit()

def download_image_from_dynamic_page(page_url):
    """Extracts the high-resolution image or canvas image using Selenium."""
    driver = get_driver_with_random_user_agent()
    try:
        driver.get(page_url)

        # Try to extract the high-resolution image from the hidden input field
        try:
            txt_file_name = driver.find_element(By.ID, "txtFileName").get_attribute("value")
            
            # Dynamically construct the full URL based on the page_url
            high_res_image_url = urljoin(page_url, txt_file_name)
            print(f"High-resolution image URL extracted: {high_res_image_url}")

            # Download the high-resolution image using the extracted URL
            response = requests.get(high_res_image_url)
            if response.status_code == 200:
                image_data = BytesIO(response.content)
                ext = 'jpg'
                return image_data, ext, high_res_image_url
            else:
                print(f"Failed to download high-resolution image. Status code: {response.status_code}")
        except NoSuchElementException:
            print("High-resolution image not found, falling back to canvas extraction.")

        # Fall back to canvas image extraction if high-res image is not available
        try:
            canvas_element = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, 'canvas'))  # Adjust the selector if necessary
            )
            print("Canvas element found, extracting image...")

            # Use JavaScript to extract the data from the canvas
            image_url = driver.execute_script("""
                var canvas = arguments[0];
                return canvas.toDataURL('image/jpeg');  // Adjust format if needed
            """, canvas_element)

            # Handle Base64 encoded data URL
            if image_url.startswith("data:image"):
                ext = image_url.split(";")[0].split("/")[1]  # Extract extension from the data URL
                print(f"Extracted image format: {ext}")

                # Extract the base64-encoded part
                base64_data = image_url.split(",")[1]
                image_data = BytesIO(base64.b64decode(base64_data))  # Convert to a BytesIO object

                return image_data, ext, page_url  # Return image data and extension for further processing
            else:
                print("No valid image data found in the canvas element.")
                return None, None, page_url

        except TimeoutException:
            print("Timed out waiting for the canvas element to load.")
            return None, None, page_url

    except Exception as e:
        print(f"Error extracting image: {e}")
        return None, None, page_url

    finally:
        driver.quit()

def is_base64_image(image_data):
    if isinstance(image_data, str) and image_data.startswith('data:image/'):
        return True
    return False

def decode_base64_image(image_data):
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

def save_base64_image(image_data, target_path):
    """Save a decoded base64 image to a file."""
    decoded_image, ext = decode_base64_image(image_data)

    # Save the image to the specified path
    with open(target_path, 'wb') as img_file:
        img_file.write(decoded_image)
    print(f"Base64 image saved to {target_path}")

def iiif_parse(manifest_url):
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

def convert_drive_url_to_direct(drive_url):
    """Convert a Google Drive shareable URL to a direct download link."""
    file_id = drive_url.split('/d/')[1].split('/')[0]  # Extract the file ID
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download_image_directly(image_url, target_dir, fname, max_retries=2):
    """Download an image from the URL, handling base64 images and 401 redirects."""
    
    initial_code = ''
    retries = 0
    backoff = 2  # Start with a 2-second delay for exponential backoff

    # Handle Google Drive links
    if "drive.google.com" in image_url:
        print(f"Google Drive URL detected: {image_url}")
        image_url = convert_drive_url_to_direct(image_url)

    # Handle special case for URLs containing "manifest.json"
    if ("manifest.json" in image_url) or ('manifest' in image_url):
        print(f"Manifest URL detected: {image_url}")
        # Parse the IIIF manifest to get the actual image URL
        image_url = iiif_parse(image_url)
        if not image_url:
            return None, "Failed to extract image from IIIF manifest"   

    while retries < max_retries:
        try:
            with requests.Session() as session:
                headers = {
                    'User-Agent': random.choice(USER_AGENTS),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Referer': 'https://www.google.com/',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                }
                session.headers.update(headers)

                response = session.get(image_url, stream=True, timeout=80, verify=False)

                if response.status_code == 404:
                    return response.status_code, f"Error 404"
                
                # Check if we received a 401 status code and handle consent
                if response.status_code == 401:
                    initial_code = response.status_code
                    print("401 Unauthorized detected. Handling consent...")
                    if image_url:
                        redirect_url = handle_consent_with_selenium(image_url)
                        if redirect_url:
                            response = session.get(redirect_url, stream=True, timeout=80, verify=False)
                            print(f"Retrying with the redirect URL {redirect_url}")
                        else:
                            print("Failed to retrieve the redirect URL after consent.")
                            return response.status_code, f"Failed to download image. Status code: {response.status_code}"

                # Check if the image URL is base64-encoded
                if is_base64_image(image_url):
                    print("Base64-encoded image detected.")
                    ext = image_url.split(";")[0].split("/")[1]
                    image_filename = f"base64_image.{ext}"
                    target_path = os.path.join(target_dir, image_filename)
                    save_base64_image(image_url, target_path)
                    return response.status_code, f"Downloaded image. Base64-encoded image detected: {response.status_code}"

                # Get the content type to figure out the image format for non-base64 images
                content_type = response.headers.get('Content-Type')
                if not content_type:
                    content_type = 'missing'

                if 'text/html' in content_type:
                    print("HTML content detected. Parsing for image...")

                    # Try to extract the image from the dynamic page
                    image_data, ext, new_dynamic_url = download_image_from_dynamic_page(image_url)

                    if image_data:
                        ext = "jpg"  # Ensure extension is jpg
                        target_path = os.path.join(target_dir, f"{fname}.{ext}")

                        # Convert and save the image as JPEG
                        img = Image.open(image_data)
                        img = img.convert("RGB")  # Convert to RGB if needed
                        img.save(target_path, 'JPEG', quality=95)
                        print(f"Image successfully saved as JPG to {target_path}")
                        if new_dynamic_url != image_url:
                            return 200, f"Downloaded image. Extracted HIGHRES url from dynamic image viewer."
                        else:
                            return 200, f"Downloaded image. Extracted canvas LOWRES url from dynamic image viewer."

                    else:
                        return 500, "Failed to extract image from dynamic page."

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

                print(f"Image format detected: {ext}")

                # Use a placeholder filename based on the image URL and content type
                image_name = f"{fname}.jpg"
                target_path = os.path.join(target_dir, image_name)

                # Read the image content from the response
                image_data = BytesIO(response.content)
                
                try:
                    # Open the image and convert it to JPEG format if necessary
                    img = Image.open(image_data)
                    if ext != 'jpg':
                        target_path = os.path.join(target_dir, fname + f'_converted_from_{ext}' + ".jpg")
                        img = img.convert("RGB")  # Convert to RGB if needed (e.g., for PNG or TIFF)
                    
                    # Save the image as JPEG
                    img.save(target_path, 'JPEG', quality=95)
                    print(f"Image successfully saved as JPG to {target_path}")
                    return response.status_code, f"Downloaded image. Final - {response.status_code} Initial - {initial_code}"
                
                except Exception as e:
                    print(f"Error converting and saving the image: {e}")
                    return response.status_code, f"Error {e}"
        
        except (ConnectionError, Timeout) as e:
            retries += 1
            print(f"Connection error occurred: {e}. Retrying {retries}/{max_retries}...")
            time.sleep(backoff)  # Wait before retrying
            backoff *= 2  # Exponential backoff

        except HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            return False, "HTTP error occurred while downloading the image"

        except RequestException as req_err:
            print(f"Request exception occurred: {req_err}")
            return False, "A request error occurred"

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False, "An unexpected error occurred"

    # If the max retries are exhausted
    return False, "Max retries exceeded, could not download the image."

# def download_images_from_page(url, target_dir, fname):
#     with requests.Session() as session:
#         # Get the HTML content of the page
#         response = session.get(url)
#         response.raise_for_status()  # Raise an exception if the request was unsuccessful

#         # Parse the HTML to find all <img> tags
#         soup = BS(response.text, 'lxml')
#         images = soup.find_all('img')

#         # Loop over each image
#         for image in images:
#             # Extract the image source (src) attribute
#             src = image.get('src')
#             if not src:
#                 continue  # Skip if no 'src' attribute

#             # Resolve relative URLs
#             full_image_url = urljoin(url, src)

#             # Fetch the image content
#             image_response = session.get(full_image_url, stream=True)
#             image_response.raise_for_status()

#             # Save the image to the target directory
#             image_name = f"{fname}.jpg"
#             target_path = os.path.join(target_dir, image_name)


#             with open(target_path, 'wb') as file:
#                 for chunk in image_response.iter_content(chunk_size=8192):
#                     file.write(chunk)

#             print(f"Downloaded {image_name} to {target_dir}")

# Function to fetch image with ScraperAPI proxy
def fetch_image_with_scraperapi(url, retries=3, render_js=False):
    session = requests.Session()
    
    # Payload for ScraperAPI request
    payload = {
        'api_key': SCRAPERAPI_KEY,
        'url': url,
        'keep_headers': 'true',
        'premium': 'true',
        'ultra_premium': 'true',
        'render_js': str(render_js).lower(),  # Optional: use if JavaScript rendering is required
        'keep_headers': 'true',
        'device_type': 'desktop',
        'country_code': 'us',
    }

    # Exponential backoff retry logic
    for attempt in range(1, retries + 1):
        headers = {
            'User-Agent': random.choice(USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Referer': 'https://www.google.com/',
            'Accept-Language': 'en-US,en;q=0.9',
            'DNT': '1',
            'Connection': 'keep-alive',
        }
        session.headers.update(headers)

        try:
            # Random delay to avoid detection
            time.sleep(random.uniform(1, 3))

            # Request via ScraperAPI
            logging.info(f"Attempt {attempt}: Fetching URL via ScraperAPI...")
            response = session.get('http://api.scraperapi.com', params=payload, verify=False, allow_redirects=True, timeout=80)

            # Check for valid response
            if response.status_code == 200:
                # If the content type is HTML, look for redirects
                if 'text/html' in response.headers['Content-Type']:
                    logging.info("HTML content detected. Searching for redirects.")
                    # Parse the HTML content to find the redirect link
                    soup = BS(response.text, 'html.parser')
                    redirect_link = soup.find('a')  # You may refine this to find specific tags with attributes
                    
                    # Show the HTML for diagnosis
                    print("Returned HTML:")
                    print(response.text)
                    
                    if redirect_link and redirect_link.get('href'):
                        logging.info(f"Found redirect link: {redirect_link['href']}")
                        # Retry with the new URL found in the HTML
                        url = redirect_link['href']
                        continue  # Retry with the new URL
                    else:
                        logging.warning("No redirect link found in the HTML.")
                        return None

                # If the content is not HTML (likely an image), return it
                return response.content

            # Log status codes that are not 200 OK
            logging.warning(f"Attempt {attempt} failed with status code {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt} failed: {e}")

        # Backoff between retries
        backoff_delay = 2 ** attempt
        logging.info(f"Retrying in {backoff_delay} seconds...")
        time.sleep(backoff_delay)

    logging.error(f"All {retries} attempts failed to fetch the URL.")
    return None


if __name__ == "__main__":
    import logging
    import aiohttp

    logging.basicConfig(level=logging.DEBUG)
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Referer': 'https://www.google.com/',
        'Accept-Language': 'en-US,en;q=0.9',
        'DNT': '1',
        'Connection': 'keep-alive',
    }
    
    async def download_image(image_url):
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            async with session.get(image_url, timeout=80) as response:
                print(response.status)

    url = 'http://arctos.database.museum/media/10666097?open'
    asyncio.run(download_image(url))



    # import requests
    # proxies = {
    #   "http": "scraperapi.binary_target=true:96b4e61e7dee075693c1f9dfd836a044@proxy-server.scraperapi.com:8001"
    # }
    # r = requests.get('http://data.huh.harvard.edu/ac603661-d61d-4395-aa51-b2dffbf630de/image', proxies=proxies, verify=False)
    # print(r.text)

    # Test URL
    # url = 'https://sernecportal.org/imglib/seinet/sernec/BRIT_VDB/BRIT112/BRIT112025.jpg'
    url = 'http://data.huh.harvard.edu/ac603661-d61d-4395-aa51-b2dffbf630de/image' #SOLVED
    url = 'https://arctos.database.museum/media/10666097?open' #SOLVED
    url = 'https://herbarium.bgbm.org/data/iiif/B100591454/manifest.json' #SOLVED
    url = 'https://services.jacq.org/jacq-services/rest/images/europeana/563732?withredirect=1'#SOLVED
    url = 'http://www.herbariumhamburgense.de/herbarsheets/disk_batch01/medium/HBG-502503.jpg' #SOLVED malware
    url = 'https://rdu.unc.edu.ar/bitstream/handle/11086/22322/CORD00002686.jpg' #SOLVED needs headers
    url = 'http://www.jacq.org/image.php?filename=113465&method=europeana'#SOLVED is 404 but returns a literal png of 404
    url = 'https://bisque.cyverse.org/image_service/image/00-XciZf4wsmySSkRKQDY6ZuM/resize:4000/format:jpeg' #SOLVED max retries
    url = 'https://ia801503.us.archive.org/15/items/bbs-dataset202201/BBS0000925.jpg' #SOLVED max retries
    url = 'http://128.171.206.220/HAW06017.JPG' #SOLVED max retries
    url = 'https://images.chrb.njaes.rutgers.edu/2022_10_04/CHRB0001299.jpg' #SOLVED skip SSL verification
    url = 'http://imagens2.jbrj.gov.br/fsi/server?type=image&source=ian/SiBBR_Agosto_2016/IAN151635a.JPG' #SOLVED max retries
    url = 'http://www.herbarium-erlangense.nat.uni-erlangen.de/datenbank/bild/226/22687-large.jpg' #SOLVED 404
    url = 'https://specify-att.isa.ulisboa.pt/static/originals/sp67499002176161458909.att.jpg' #SOLVED timeout needed to increase
    url = 'https://ngcpr.org/magnify/smImg/FicusExasperataNgcpr01325.jpg' #SOLVED
    url = 'http://jbrj-public.s3.amazonaws.com/fsi/server?type=image&source=DZI/alcb/alcb/0/3/35/57/alcb00033557.dzi' #SOLVED no access
    url = 'https://sdplantatlas.org/HiResDisplay/HiResZoomDisplayVF.aspx?H=288321&F=20240828' #SOLVED HARD extract secret url OR from canvas
    url = 'https://iiif-manifest.oxalis.br.fgov.be/specimen/BR0000005028856/manifest' #SOLVED another manifest option
    url = 'https://sdplantatlas.org/HiResDisplay/HiResZoomDisplay.aspx?H=280778' #SOLVED HARD extract secret url OR from canvas
    url = 'https://bajaflora.org/HiResDisplay/HiResZoomDisplay.aspx?H=268674' #SOLVED HARD extract secret url OR from canvas
    url = 'https://drive.google.com/file/d/1y5_7YT1oQaB_UCuCcTbFbVIv2yjqq5UD/view?usp=sharing'
    # url = 'https://ids.si.edu/ids/deliveryService/id/ark:/65665/m38ae40750bf9b4f5595a2f413de845290' # WAY TOO SLOW LOADING
    url = 'https://image.bgbm.org/images/internal/HerbarThumbs/B101041802_1700' #SOLVED
    url = 'https://arctos.database.museum/media/10666097?open'


    fname = 'test'

    BANNED = ['www.herbariumhamburgense.de', 'imagens4.jbrj.gov.br', 'imagens1.jbrj.gov.br', 
              'arbmis.arcosnetwork.org', '128.171.206.220', 'ia801503.us.archive.org', 'procyon.acadiau.ca',
              'www.inaturalist.org']

    parsed_url = urlparse(url)
    domain = parsed_url.netloc

    if domain not in BANNED:
        status, message = download_image_directly(url, 'D:/D_Desktop', fname)

    # fname = 'test2'
    # download_images_from_page(url, 'D:/D_Desktop', fname)




    image_data = fetch_image_with_scraperapi(url)

    if image_data:
        # with open('downloaded_image.jpg', 'wb') as file:
        #     file.write(image_data)
        # img = Image.open(image_data)
        img = Image.open(BytesIO(image_data))

        # try:
            # img.show()
        # except Exception as e:
        # logging.error(f"Error displaying image: {e}")
        img.save("downloaded_image_backup.jpg", quality=95)
    else:
        print("Failed to download the image.")



    # https://api.scraperapi.com/?api_key=9d6f2497ec0e3366f93f88d1e6464e90&url=http%3A%2F%2Fdata.huh.harvard.edu%2Fac603661-d61d-4395-aa51-b2dffbf630de%2Fimage&keep_headers=true&premium=true&ultra_premium=true&render_js=true&device_type=desktop&country_code=us