import os
from PIL import Image
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

# Define the directory paths
IMAGE_DIR = 'D:/T_Downloads/zips/final_dataset'
TOO_SMALL_DIR = os.path.join(IMAGE_DIR, 'too_small')

# Create the too_small directory if it doesn't exist
if not os.path.exists(TOO_SMALL_DIR):
    os.makedirs(TOO_SMALL_DIR)

def check_image_size(img):
    """Check the size of the image and return megapixels, width, and height."""
    img_w, img_h = img.size
    img_mp = round((img_w * img_h) / 1_000_000, 1)  # Convert to MP
    return img_mp, img_w, img_h

def calc_resize(img_w, img_h, max_mp=25):
    """Resize the image dimensions to keep the resolution under 25 MP."""
    # Calculate resize ratio
    img_mp = (img_w * img_h) / 1_000_000
    resize_ratio = (max_mp / img_mp) ** 0.5

    # Apply resize ratio if needed
    if resize_ratio < 1:
        img_w = round(img_w * resize_ratio)
        img_h = round(img_h * resize_ratio)

    return img_w, img_h

def process_image(filename, image_dir, too_small_dir):
    """Process a single image file."""
    if filename.endswith('.jpeg'):
        # Rename .jpeg to .jpg
        new_filename = filename.replace('.jpeg', '.jpg')
        old_filepath = os.path.join(image_dir, filename)
        new_filepath = os.path.join(image_dir, new_filename)
        os.rename(old_filepath, new_filepath)
        print(f"Renamed: {old_filepath} -> {new_filepath}")
    else:
        new_filepath = os.path.join(image_dir, filename)  # Keep the current filename

    # Open the image
    img_path = os.path.join(image_dir, new_filepath)
    img = Image.open(img_path)
    img_mp, img_w, img_h = check_image_size(img)
    img.close()  # Explicitly close the image to release the file

    # Move to too_small if the longest side is less than 1000px
    if max(img_w, img_h) < 1000:
        dest_path = os.path.join(too_small_dir, new_filename)
        shutil.move(img_path, dest_path)
        print(f"Moved too small image: {img_path} -> {dest_path}")
        return

    # Resize if image exceeds 25 MP
    if img_mp > 25:
        img = Image.open(img_path)  # Re-open the image for resizing
        img_w, img_h = calc_resize(img_w, img_h, max_mp=25)
        img = img.resize((img_w, img_h))
        img.save(img_path)  # Overwrite the resized image
        img.close()  # Ensure the image file is closed
        print(f"Resized {new_filename} to fit under 25 MP")

def process_images_in_parallel(image_dir, too_small_dir, num_workers=12):
    """Process all images in the directory in parallel using multiple cores."""
    # List all images in the directory
    files = [f for f in os.listdir(image_dir) if f.endswith(('.jpeg', '.jpg'))]

    # Use ProcessPoolExecutor to parallelize the process
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all the tasks
        futures = [executor.submit(process_image, filename, image_dir, too_small_dir) for filename in files]

        # Collect results as they complete
        for future in as_completed(futures):
            future.result()  # We can catch exceptions here if needed

if __name__ == "__main__":
    process_images_in_parallel(IMAGE_DIR, TOO_SMALL_DIR, num_workers=12)
