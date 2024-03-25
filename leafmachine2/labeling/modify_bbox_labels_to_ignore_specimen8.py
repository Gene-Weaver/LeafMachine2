import os
from tqdm import tqdm

def edit_txt_files(root_dir):
    # Iterate through train, val, and test folders with tqdm
    for split in tqdm(['train', 'val', 'test'], desc="Split"):
        split_dir = os.path.join(root_dir, split)
        # Iterate through each file in the split folder with tqdm
        for file_name in tqdm(os.listdir(split_dir), desc="File", leave=False):
            file_path = os.path.join(split_dir, file_name)
            # Read the content of the file
            with open(file_path, 'r') as file:
                lines = file.readlines()
            # Check if the file contains any 0 or 1 classes
            contains_01 = any(line.split()[0] in ['0', '1'] for line in lines)
            if contains_01:
                # Filter out class 8 lines
                lines = [line for line in lines if line.split()[0] != '8']
                # Write the modified content back to the file
                with open(file_path, 'w') as file:
                    file.writelines(lines)

if __name__ == '__main__':
    # edit_txt_files('D:/Dropbox/LeafMachine2/leafmachine2/component_detector/datasets/PLANT_Group3_EverythingSomeNotReviewed/labels_test')
    edit_txt_files('D:/Dropbox/LeafMachine2/leafmachine2/component_detector/datasets/PLANT_Group3_EverythingSomeNotReviewed/labels')
