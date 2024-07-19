import requests
import random
from PIL import Image
from io import BytesIO

# Initialize ScraperAPI client
SCRAPERAPI_KEY = '8d4c64c348d7013d24ace10641788ad4'

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
    # Add more user agents here...
]

def fetch_image_with_proxy(url):
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Referer': 'https://www.google.com/',
    }

    proxies = {
        "https": f"http://scraperapi:{SCRAPERAPI_KEY}@proxy-server.scraperapi.com:8001"
    }

    try:
        response = requests.get(url, headers=headers, proxies=proxies, verify=False)
        response.raise_for_status()  # Check if the request was successful
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

# Test URL
url = 'https://sernecportal.org/imglib/seinet/sernec/BRIT_VDB/BRIT112/BRIT112025.jpg'
image_data = fetch_image_with_proxy(url)

if image_data:
    with open('downloaded_image.jpg', 'wb') as file:
        file.write(image_data)
    img = Image.open(BytesIO(image_data))
    img.show()
else:
    print("Failed to download the image.")