
import json

def replace_id_values(file_path, current_id, anno_id):
    with open(file_path, "r") as file:
        data = json.load(file)
    
    # Keep track of the current unique ID value
    # current_id = 10000
    
    for obj in data["images"]:
        for annotation in obj["annotations"]:
            annotation["id"] = current_id
            current_id += 1

    for annotation in data['annotations']:
        annotation['id'] = anno_id
        anno_id += 1
    
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

if __name__ == '__main__':
    replace_id_values('D:/Dropbox/LeafMachine2/data/segmentation_training_data/groupB/train/images/POLYGONS_train.json', current_id=10000, anno_id=10000)
    replace_id_values('D:/Dropbox/LeafMachine2/data/segmentation_training_data/groupB/val/images/POLYGONS_val.json', current_id=50000, anno_id=50000)

