import os
import shutil
import math

def split_images_into_folders(img_folder, images_per_folder=10000):
    # Get a list of all files in the img folder
    img_files = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]
    
    total_images = len(img_files)
    print(f"Total number of images found: {total_images}")
    
    # Calculate the number of folders needed
    num_folders = math.ceil(total_images / images_per_folder)
    print(f"Creating {num_folders} folders with up to {images_per_folder} images each.")

    # Get the parent directory to create new img_{i} folders
    parent_dir = os.path.dirname(img_folder)

    # Iterate through each folder to create and copy files
    for i in range(num_folders):
        # Create a new folder (img_1, img_2, etc.)
        new_folder_name = os.path.join(parent_dir, f"img_{i+1}")
        os.makedirs(new_folder_name, exist_ok=True)
        print(f"Created folder: {new_folder_name}")
        
        # Get the start and end indices for the current batch of images
        start_idx = i * images_per_folder
        end_idx = min((i + 1) * images_per_folder, total_images)  # Handle remainder

        # Copy images into the new folder
        for img_file in img_files[start_idx:end_idx]:
            src = os.path.join(img_folder, img_file)
            dst = os.path.join(new_folder_name, img_file)
            shutil.copy2(src, dst)  # Copy the file
        print(f"Copied images {start_idx} to {end_idx - 1} into {new_folder_name}")

if __name__ == "__main__":
    # Example: Replace this with the path to your 'img' folder
    # img_folder_path = "/media/nas/GBIF_Downloads/Magnoliales/Annonaceae/img" # images_per_folder=10000 DONE
    # img_folder_path = "/media/nas/GBIF_Downloads/Fagales/Betulaceae/img" # images_per_folder=20000 DONE
    img_folder_path = "/media/nas/GBIF_Downloads/Fagales/Fagaceae/img" # images_per_folder=20000 DONE


    if os.path.exists(img_folder_path) and os.path.isdir(img_folder_path):
        split_images_into_folders(img_folder_path, images_per_folder=20000)
    else:
        print("Invalid path. Please provide a valid path to the 'img' folder.")
