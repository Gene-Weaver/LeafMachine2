import os
import shutil
from math import ceil

def split_images_into_subdirs(original_dir, n_subdirs):
    """
    Splits images from 'original_dir' into 'n_subdirs' subdirectories.
    Each subdirectory will contain an approximately equal number of images.

    :param original_dir: Directory containing the original images.
    :param n_subdirs: Number of subdirectories to create.
    """
    # Check if original directory exists
    if not os.path.exists(original_dir):
        raise FileNotFoundError(f"The directory {original_dir} does not exist.")

    # Get all image files from the original directory
    image_files = [f for f in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, f))]
    n_total_images = len(image_files)

    # Calculate the number of images per subdirectory
    images_per_subdir = ceil(n_total_images / n_subdirs)

    for i in range(n_subdirs):
        # Create subdirectory
        subdir_name = f"{original_dir}_{i}"
        subdir_path = os.path.join(original_dir, subdir_name)
        os.makedirs(subdir_path, exist_ok=True)

        # Move images to the subdirectory
        for img in image_files[i * images_per_subdir:(i + 1) * images_per_subdir]:
            shutil.move(os.path.join(original_dir, img), subdir_path)

def merge_images_from_subdirs(original_dir, n_subdirs):
    """
    Merges images from subdirectories into the original directory.

    :param original_dir: Directory containing the original images.
    :param n_subdirs: Number of subdirectories to merge.
    """
    # Check if original directory exists
    if not os.path.exists(original_dir):
        raise FileNotFoundError(f"The directory {original_dir} does not exist.")

    for i in range(n_subdirs):
        # Subdirectory path
        subdir_path = os.path.join(original_dir, f"{original_dir}_{i}")

        # Check if subdirectory exists
        if not os.path.exists(subdir_path):
            print(f"Subdirectory {subdir_path} does not exist. Skipping.")
            continue

        # Move images back to the original directory
        for img in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, img)
            if os.path.isfile(img_path):
                shutil.move(img_path, original_dir)

        # Remove the empty subdirectory
        os.rmdir(subdir_path)


if __name__ == '__main__':
    # split_images_into_subdirs('/media/data/Shape_Datasets/Tropical/Sapotaceae/img', 20)
    merge_images_from_subdirs('/media/data/Shape_Datasets/Tropical/Eschweilera_Mart_ex_DC/img', 10)
