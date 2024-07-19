from PIL import Image
import os
import json

def update_annotations_and_image_dimensions(dataset_dir):
    for opt in ['train', 'val']:
        json_file_path = f'{dataset_dir}/{opt}/images/POLYGONS_{opt}_original.json'
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        unique_id_counter = 100000 if opt == 'train' else 5000000

        # Check and update annotation IDs
        updated_annotations = []
        for annotation in data.get('annotations', []):
            annotation['id'] = unique_id_counter
            updated_annotations.append(annotation)
            unique_id_counter += 1
        data['annotations'] = updated_annotations

        # Check and update image dimensions
        for image_info in data.get('images', []):
            image_path = os.path.join(dataset_dir, opt, 'images', image_info['file_name'])
            with Image.open(image_path) as img:
                actual_width, actual_height = img.size
            if image_info['width'] != actual_width or image_info['height'] != actual_height:
                print(f"Updating dimensions for {image_info['file_name']}: was ({image_info['width']}, {image_info['height']}), actual ({actual_width}, {actual_height})")
                image_info['width'] = actual_width
                image_info['height'] = actual_height

        # Save the modified dataset back to a new JSON file
        updated_json_file_path = json_file_path.replace('_original.json', '.json')
        with open(updated_json_file_path, 'w') as outfile:
            json.dump(data, outfile, indent=4)

        print(f"Updated JSON file saved for {opt} dataset: {updated_json_file_path}")

dataset_dir = '/home/brlab/Dropbox/LM2_Env/Image_Datasets/GroundTruth_SEG_Group3/GroundTruth_SEG_Group3'
update_annotations_and_image_dimensions(dataset_dir)
