# install latest labelbox version (3.0 or above)
# !pip3 install labelbox[data] 

from labelbox import Dataset, Client
import os, sys, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from machine.general_utils import get_cfg_from_full_path, bcolors

def deleteProject(client,deleteName):
    count = 0
    projects = client.get_projects()
    for project in projects:
        if project.name == deleteName:
            count += 1
            project.delete()
            print(f"{bcolors.WARNING}      Deleted --> {project.name}{bcolors.ENDC}")
    if count > 0:
        print(f"{bcolors.HEADER}Deleted {count} project/s{bcolors.ENDC}")
    elif count == 0:
        print(f"{bcolors.FAIL}No projects deleted{bcolors.ENDC}")

def deleteDataset(client,deleteName):
    count = 0
    datasets = client.get_datasets(where=Dataset.name == deleteName)
    for dataset in datasets:
        count += 1
        dataset.delete()
        print(f"{bcolors.WARNING}      Deleted --> {dataset.name} -- Containing {dataset.row_count} images{bcolors.ENDC}")
    if count > 0:
        print(f"{bcolors.HEADER}Deleted {count} dataset/s{bcolors.ENDC}")
    elif count == 0:
        print(f"{bcolors.FAIL}No datasets deleted{bcolors.ENDC}")

def delete_from_Labelbox():
    dir_private = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path_cfg_private = os.path.join(dir_private,'PRIVATE_DATA.yaml')
    cfg_private = get_cfg_from_full_path(path_cfg_private)

    LB_API_KEY = cfg_private['labelbox']['LB_API_KEY']
    client = Client(api_key=LB_API_KEY)

    dir_labeling = os.path.dirname(__file__)
    path_cfg = os.path.join(dir_labeling,'config_delete_from_Labelbox.yaml')
    cfg = get_cfg_from_full_path(path_cfg)

    if cfg['delete_projects'] != []:
        for project in cfg['delete_projects']:
            print(f"{bcolors.HEADER}Deleting Project: '{project}'{bcolors.ENDC}")
            DELETE_PROJECT = project
            deleteProject(client,DELETE_PROJECT)

    if cfg['delete_datasets'] != []:
        for dataset in cfg['delete_datasets']:
            print(f"{bcolors.HEADER}Deleting Dataset: '{dataset}'{bcolors.ENDC}")
            DELETE_DATASET = dataset
            deleteDataset(client,DELETE_DATASET)

if __name__ == '__main__':
    delete_from_Labelbox()