import os
import shutil

def copy_images_from_subdirs(dir_start, dir_target, image_extensions=("jpg", "jpeg", "png", "gif", "tiff", "bmp")):

    if not os.path.exists(dir_target):
        os.makedirs(dir_target)

    for subdir, _, files in os.walk(dir_start):
        for file in files:
            if file.lower().endswith(image_extensions):
                source_path = os.path.join(subdir, file)
                
                parent_dir_name = os.path.basename(subdir)
                grandparent_dir_name = os.path.basename(os.path.dirname(subdir))
                
                new_filename = f"{grandparent_dir_name}_{parent_dir_name}_{file}"
                target_path = os.path.join(dir_target, new_filename)

                # Handle potential name collision in the target directory
                counter = 1
                while os.path.exists(target_path):
                    base, ext = os.path.splitext(new_filename)
                    target_path = os.path.join(dir_target, f"{base}_{counter}{ext}")
                    counter += 1

                shutil.copy(source_path, target_path)
                print(f"Copied {source_path} to {target_path}")

if __name__ == "__main__":
    dir_start = "D:/Dropbox/LM2_Env/Image_Datasets/SET_Leafsnap/dataset/images"
    dir_target = "D:/Dropbox/LM2_Env/Image_Datasets/SET_Leafsnap/dataset_extracted"
    copy_images_from_subdirs(dir_start, dir_target)
