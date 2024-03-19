import pandas as pd
import threading
import os
from dataclasses import dataclass, field
import os, inspect, sys
import requests
from PIL import Image
from io import BytesIO
import certifi
currentdir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(currentdir)
from leafmachine2.machine.utils_GBIF import ImageCandidateCustom

def process_images_from_csv(csv_file, url_column, name_column, dir_out):
    # Read the CSV file
    df = pd.read_csv(csv_file, sep=",", header=0, low_memory=False, dtype=str)

    # Configuration settings
    cfg = {
        'dir_destination_images': dir_out,
        'MP_low': 1,   # Set your own values
        'MP_high': 200, # Set your own values
        'do_resize': False # or False
    }

    # Create a lock for thread-safe operations if needed
    lock = threading.Lock()

    # Process each image
    for index, row in df.iterrows():
        try:
            image_candidate = ImageCandidateCustom(
                cfg=cfg,
                image_row=row,
                url=row[url_column],
                col_name=name_column,
                lock=lock
            )
        except Exception as e:
            print(f"Error processing row {index}: {e}")

def download_single_image(image_url, save_path):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    try:
        # response = requests.get(image_url, headers=headers, verify=certifi.where())
        response = requests.get(image_url, headers=headers, verify=False)

        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.save(save_path)
        print(f"Image successfully downloaded: {save_path}")
    except Exception as e:
        print(f"Error downloading the image: {e}")


if __name__ == "__main__":
    # test_url = "https://oregonflora.org/imglib/OSU_B/OSC-B-27/OSC-B-27513_1_lg.jpg"

    # download_single_image(test_url, "D:/Dropbox/LM2_Env/Image_Datasets/OSU_Bryophytes/img/img.jpg")

    process_images_from_csv('D:/Dropbox/LM2_Env/Image_Datasets/OSU_Bryophytes/input/multimedia.csv',
                            'accessURI',
                            'coreid',
                            "D:/Dropbox/LM2_Env/Image_Datasets/OSU_Bryophytes/img")