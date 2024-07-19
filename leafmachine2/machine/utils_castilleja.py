import os, shutil, random
from tqdm import tqdm

if __name__ == "__main__":

    # Directory containing the images
    source_dir = "D:/Dropbox/Castilleja/collection/img_subset"
    # Directory to store up to 5 images per species
    target_dir = "D:/Dropbox/Castilleja/collection/img_subset_15per"
    overflow_dir = "D:/Dropbox/Castilleja/collection/img_subset_15per_overflow"

    n_img = 15

    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(overflow_dir, exist_ok=True)

    # List all files in the source directory that are JPG images
    all_files = [filename for filename in os.listdir(source_dir) if filename.lower().endswith(".jpg")]
    # Randomize the list of files
    random.shuffle(all_files)

    # Dictionary to keep track of how many images per species have been moved
    species_count = {}

    # Iterate over all files in the source directory
    for filename in tqdm(all_files, desc="Processing images"):
        if filename.lower().endswith(".jpg"):  # Check if the file is a JPEG
            # Extract species name from the filename
            filename_base = filename.split(".")[0]
            filename_ext = filename.split(".")[1]
            parts = filename_base.split("_")
            if len(parts) >= 4:
                species_name = "_".join(parts[2:4])  # Join the 2nd and 3rd parts to form the species name

                # Track the count of images per species
                if species_name not in species_count:
                    species_count[species_name] = 0

                # Move the file if fewer than n_img images for this species have been moved
                if species_count[species_name] < n_img:
                    # Construct source and destination paths
                    source_path = os.path.join(source_dir, filename)
                    destination_path = os.path.join(target_dir, filename)

                    # Copy the file to the new directory
                    shutil.copy(source_path, destination_path)

                    # Increment the count for this species
                    species_count[species_name] += 1
                else:
                    source_path = os.path.join(source_dir, filename)
                    destination_path = os.path.join(overflow_dir, filename)
                    # Copy the file to the new directory
                    shutil.copy(source_path, destination_path)
