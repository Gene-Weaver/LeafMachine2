import os
import pandas as pd
from PIL import Image
import ast
import numpy as np

def merge_csv(input_dir, output_file):
    # Get a list of all the CSV files in the input directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    # Initialize an empty DataFrame to store the merged data
    merged_data = pd.DataFrame()

    # Loop over each CSV file and add it to the merged data
    for file in csv_files:
        # Read the CSV file into a DataFrame
        data = pd.read_csv(os.path.join(input_dir, file))

        # Create a new Series with the class value for each row in the DataFrame
        class_series = pd.Series(file.split('__')[0], index=data.index)

        # Add the class series as the first column of the DataFrame
        data.insert(0, "class", class_series)

        # Add the data to the merged data
        merged_data = pd.concat([merged_data, data])

    # Write the merged data to the output file
    merged_data.to_csv(output_file, index=False)

def convert_to_YOLO(input_path, image_path, output_dir):
    # Define the size of the bounding box

    classes = {
        'apex_angle': 0,
        'base_angle': 1,
        'lamina_base': 2,
        'lamina_tip': 3,
        'lamina_width': 4,
        'lobe_tip': 5,
        'midvein_trace': 6,
        'petiole_tip': 7,
        'petiole_trace': 8
    }
    # Read the train.csv file into a DataFrame
    data = pd.read_csv(input_path)

    # Get a list of all unique filenames in the DataFrame
    filenames = data['filename'].unique()

    # Loop over each unique filename and create a label file for it
    for filename in filenames:
        # Get all rows in the DataFrame that contain this filename
        rows = data[data['filename'] == filename]

        # Get the image dimensions using PIL
        img_path = os.path.join(image_path, filename)
        img = Image.open(img_path)
        width, height = img.size

        if max(width, height) >= 1000:
            box_size = 27
        elif max(width, height) >= 500:
            box_size = 15
        else: 
            box_size = 9
            
        # Convert the locations to YOLO format
        yolo_locations = []
        for row in rows.itertuples():
            class_name = row[1]
            locations = ast.literal_eval(row[4])
            for point in locations:
                # point_bbox = create_bbox_from_point(point, box_size)
                # yolo_locs = convert_locations(point_bbox, width, height, box_size)
                yolo_locs = create_bbox_from_point(point, box_size, width, height)
                for yolo_loc in yolo_locs:
                    yolo_locations.append((classes[class_name], *yolo_loc))

        # Create a string with the YOLO format labels for this image
        label_str = ""
        for loc in yolo_locations:
            label_str += f"{str(loc[0])} {str(loc[1])} {str(loc[2])} {str(loc[3])} {str(loc[4])}\n"

        # Write the label string to a text file
        label_filename = os.path.splitext(filename)[0] + '.txt'
        with open(os.path.join(output_dir, label_filename), 'w') as f:
            f.write(label_str)

# def convert_locations(bounding_box, img_width, img_height, box_size):
#     bbox_list = []
#     xmin, ymin, xmax, ymax = bounding_box
#     xmin = max(0, xmin)
#     ymin = max(0, ymin)
#     xmax = min(img_width, xmax)
#     ymax = min(img_height, ymax)
#     x_center = (xmin + xmax) / 2 / img_width
#     y_center = (ymin + ymax) / 2 / img_height
#     width = min(box_size, xmax - xmin) / img_width
#     height = min(box_size, ymax - ymin) / img_height
#     bbox_list.append((x_center, y_center, width, height))
#     if any(coord > 1 for coord in (x_center, y_center, width, height)):
#         print('here')
#     return bbox_list
def all_ints(values):
    for value in values:
        if not isinstance(value, int):
            return False
    return True

def create_bbox_from_point(point, box_size, width, height):
    valid_int = all_ints(point)
    bbox_list = []

    # The icacinaceae points have swapped x,y coordinates compared to the newer points.
    # Luckily the icacincaeae are all integers, while newer points are floats. 
    # So this swaps them, puts them all in the correct order
    if not valid_int:
        # Calculate the x and y coordinates of the top-left corner of the bounding box
        x_min = int(point[0] - (box_size - 1) / 2)
        y_min = int(point[1] - (box_size - 1) / 2)

        # Calculate the x and y coordinates of the bottom-right corner of the bounding box
        x_max = int(point[0] + (box_size - 1) / 2)
        y_max = int(point[1] + (box_size - 1) / 2)
    else:
        # Calculate the x and y coordinates of the top-left corner of the bounding box
        x_min = int(point[1] - (box_size - 1) / 2)
        y_min = int(point[0] - (box_size - 1) / 2)

        # Calculate the x and y coordinates of the bottom-right corner of the bounding box
        x_max = int(point[1] + (box_size - 1) / 2)
        y_max = int(point[0] + (box_size - 1) / 2)

    # Convert coordinates to normalized values
    x_min_norm = x_min / width
    y_min_norm = y_min / height
    x_max_norm = x_max / width
    y_max_norm = y_max / height

    # x_min_norm = max(0, x_min_norm)
    # y_min_norm = max(0, y_min_norm)
    # x_max_norm = min(1.0, x_max_norm)
    # y_max_norm = min(1.0, y_max_norm)

    # Calculate center and size of bounding box in normalized values
    x_center_norm = (x_min_norm + x_max_norm) / 2
    y_center_norm = (y_min_norm + y_max_norm) / 2
    width_norm = x_max_norm - x_min_norm
    height_norm = y_max_norm - y_min_norm

    # The adjusts the bounds when a bbox is past the edge of the image
    if y_center_norm > 1:
        y_center_norm = y_center_norm - (2 * (y_center_norm - 1))
    if x_center_norm > 1:
        x_center_norm = x_center_norm - (2 * (x_center_norm - 1))

    if y_center_norm < 0:
        y_center_norm = -y_center_norm
    if x_center_norm < 0:
        x_center_norm = -x_center_norm

    # Create a numpy array with the coordinates of the bounding box in YOLO format
    bbox_list.append((x_center_norm, y_center_norm, width_norm, height_norm))
    if any(coord > 1 for coord in (x_center_norm, y_center_norm, width_norm, height_norm)):
        print('here')
    return bbox_list

# def create_bbox_from_point(point, box_size):

#     # Calculate the x and y coordinates of the top-left corner of the bounding box
#     x_min = int(point[0] - (box_size - 1) / 2)
#     y_min = int(point[1] - (box_size - 1) / 2)

#     # Calculate the x and y coordinates of the bottom-right corner of the bounding box
#     x_max = int(point[0] + (box_size - 1) / 2)
#     y_max = int(point[1] + (box_size - 1) / 2)

#     # Create a numpy array with the coordinates of the bounding box
#     bounding_box = np.array([x_min, y_min, x_max, y_max])
#     return bounding_box

def clear_training_images_from_test_dir(dir_with_test_images, dir_with_training_labels):
    test_filenames = set(os.path.splitext(filename)[0] for filename in os.listdir(dir_with_test_images))
    label_filenames = set(os.path.splitext(filename)[0] for filename in os.listdir(dir_with_training_labels))

    matching_filenames = test_filenames.intersection(label_filenames)

    for filename in matching_filenames:
        test_path = os.path.join(dir_with_test_images, filename + ".jpg")
        if os.path.exists(test_path):
            os.remove(test_path)


if __name__ == '__main__':
    run_merge_csv = False
    run_clean_image_dirs = False
    run_convert_pts_to_bboxes = True

    # 1. Merge the csv files from labelbox into one, split by test/train
    if run_merge_csv:
        merge_csv('D:/Dropbox/LM2_Env/Image_Datasets/GroundTruth_POINTS/GroundTruth_POINTS/YOLO/test', 
                    'D:/Dropbox/LM2_Env/Image_Datasets/GroundTruth_POINTS/GroundTruth_POINTS/YOLO/test.csv')

        merge_csv('D:/Dropbox/LM2_Env/Image_Datasets/GroundTruth_POINTS/GroundTruth_POINTS/YOLO/train', 
                    'D:/Dropbox/LM2_Env/Image_Datasets/GroundTruth_POINTS/GroundTruth_POINTS/YOLO/train.csv')
    
    # 2. Clean images, so that train/test are seperate
    if run_clean_image_dirs:
        clear_training_images_from_test_dir('D:/Dropbox/LeafMachine2/leafmachine2/component_detector/datasets/Landmarks/images/test',
                                            'D:/Dropbox/LeafMachine2/leafmachine2/component_detector/datasets/Landmarks/labels/train')

        clear_training_images_from_test_dir('D:/Dropbox/LeafMachine2/leafmachine2/component_detector/datasets/Landmarks/images/train',
                                        'D:/Dropbox/LeafMachine2/leafmachine2/component_detector/datasets/Landmarks/labels/test')

    # 3. convert the points from labelbox .csv into tiny bboxes for YOLO .txt that have diff dims based on image resolution
    if run_convert_pts_to_bboxes:
        convert_to_YOLO('D:/Dropbox/LM2_Env/Image_Datasets/GroundTruth_POINTS/GroundTruth_POINTS/YOLO/test.csv', 
                    'D:/Dropbox/LM2_Env/Image_Datasets/GroundTruth_POINTS/GroundTruth_POINTS/YOLO/landmarks_YOLO/images/test',
                    'D:/Dropbox/LM2_Env/Image_Datasets/GroundTruth_POINTS/GroundTruth_POINTS/YOLO/landmarks_YOLO_expanded_bbox/labels/test')
        
        convert_to_YOLO('D:/Dropbox/LM2_Env/Image_Datasets/GroundTruth_POINTS/GroundTruth_POINTS/YOLO/train.csv', 
                    'D:/Dropbox/LM2_Env/Image_Datasets/GroundTruth_POINTS/GroundTruth_POINTS/YOLO/landmarks_YOLO/images/train',
                    'D:/Dropbox/LM2_Env/Image_Datasets/GroundTruth_POINTS/GroundTruth_POINTS/YOLO/landmarks_YOLO_expanded_bbox/labels/train')

    # NOTE: The original pass and training up to 100 epochs all had 9x9 bboxes.
    #       The 2nd time "landmarks_YOLO_expanded_bbox" had:
            # if max(width, height) >= 1000:
            #         box_size = 27
            #     elif max(width, height) >= 500:
            #         box_size = 15
            #     else: 
            #         box_size = 9
    # BUT the images are the same as the original pass