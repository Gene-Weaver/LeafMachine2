# Run yolov5 on dir
import os
from os import walk
import pandas as pd
import shutil
import subprocess

from detect import *
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from machine.general_utils import make_file_names_valid

# pip install cython matplotlib tqdm scipy ipython ninja yacs opencv-python ffmpeg opencv-contrib-python Pillow scikit-image scikit-learn lmfit imutils pyyaml jupyterlab==3
# pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def validateDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def readDetailedSampleJPGs(dirDetailed):
    f = []
    nameList = []
    for (dirpath, dirnames, filenames) in walk(dirDetailed):
        f.extend(filenames)
        break
    # print(f)
    type(f)
    fileList = pd.DataFrame(f) 
    fileList.columns = ['fname']
    df = fileList['fname'].str.split('_', expand=True)
    family = df[2]
    genus = df[3]
    species = df[4]
    species = species.str.split('.', expand=True)[0]
    nameList = family + "_" + genus + '_' + species
    nameList = pd.DataFrame(nameList) 
    nameList = pd.DataFrame(nameList[0].unique())
    nameList = nameList.dropna()
    nameList.columns = ['fullname']
    return fileList, nameList

def saveDetailedBySpecies(fileList, nameList):

    for species in nameList['fullname']:
        dirSpecies = os.path.join(dirDetailedBySpecies,species)
        validateDir(dirSpecies)

        ind = fileList['fname'].str.contains(species, regex=False)
        speciesFiles = fileList['fname'][ind]
        if len(speciesFiles) == 50:
            print(f"{bcolors.BOLD}Species: {species}   >>>   Number of Images: {len(speciesFiles)}{bcolors.ENDC}")
        else:
            print(f"{bcolors.WARNING}Species: {species}   >>>   Number of Images: {len(speciesFiles)}{bcolors.ENDC}")
        
        for img in speciesFiles:
            print(f"{bcolors.BOLD}       Copied: {img}{bcolors.ENDC}")
            shutil.copy(os.path.join(dirDetailed,img), os.path.join(dirSpecies,img))

def runYOLOforDirOfFolders(dirDetailedBySpecies,dirOutBase,nosave,PROJECT,SET,ANNO,VERSION,INCLUDE_SUBDIRS,ANNO_TYPE):
    if INCLUDE_SUBDIRS:
        f = []
        for (dirpath, dirnames, filenames) in walk(dirDetailedBySpecies):
            f.extend(dirnames)
            break
        for species in f:
            print(species)
            dirSource = os.path.abspath(os.path.join(dirDetailedBySpecies,species))
            # dirYOLO = os.path.abspath(os.path.join('yolov5','detect.py'))
            dirWeights =  os.path.abspath(os.path.join('YOLOv5','yolov5',ANNO,VERSION,'weights','best.pt'))
            dirProject = os.path.abspath(os.path.join(dirOutBase,PROJECT,SET))
            run(weights=dirWeights,source=dirSource,project=dirProject,name=species,imgsz=(1280, 1280),nosave=nosave,anno_type=ANNO_TYPE)

    else:
        # f = []
        # for (dirpath, dirnames, filenames) in walk(dirDetailedBySpecies):
        #     f.extend(filenames)
        #     break
        print(SET)
        dirSource = dirDetailedBySpecies
        species = os.path.basename(dirSource)
        # dirYOLO = os.path.abspath(os.path.join('yolov5','detect.py'))
        # dirWeights =  os.path.join('yolov5','runs','train',ANNO,VERSION,'weights','best.pt')
        dirWeights =  os.path.join(dirOutBase,'runs','train','Archival_Detector','FieldPrism_Initial','FieldPrism_Initial7','weights','last.pt')
        dirProject = os.path.join(dirOutBase,'runs','detect',PROJECT,SET)
        run(weights=dirWeights,source=dirSource,project=dirProject,name=species,imgsz=(1280, 1280),nosave=nosave,anno_type=ANNO_TYPE)


def runYOLOforDirOfFolders_PLANT_Botany(dirDetailedBySpecies,dirOutBase,nosave,PROJECT,SET,ANNO,VERSION,INCLUDE_SUBDIRS):
    if INCLUDE_SUBDIRS:
        f = []
        for (dirpath, dirnames, filenames) in walk(dirDetailedBySpecies):
            f.extend(dirnames)
            break
        for species in f:
            print(species)
            dirSource = os.path.abspath(os.path.join(dirDetailedBySpecies,species))
            # dirYOLO = os.path.abspath(os.path.join('yolov5','detect.py'))
            dirWeights =  os.path.abspath(os.path.join('yolov5','runs','train',ANNO,VERSION,'weights','best.pt'))
            dirProject = os.path.abspath(os.path.join(dirOutBase,PROJECT,SET))
            run(weights=dirWeights,source=dirSource,project=dirProject,name=species,imgsz=(1280, 1280),nosave=nosave)

    else:
        # f = []
        # for (dirpath, dirnames, filenames) in walk(dirDetailedBySpecies):
        #     f.extend(filenames)
        #     break
        print(SET)
        dirSource = dirDetailedBySpecies
        species = os.path.basename(dirSource)
        # dirYOLO = os.path.abspath(os.path.join('yolov5','detect.py'))
        # dirWeights =  os.path.join('yolov5','runs','train',ANNO,VERSION,'weights','best.pt')
        # dirWeights =  os.path.abspath(os.path.join('YOLOv5','yolov5',ANNO,VERSION,'weights','best.pt'))
        dirWeights =  os.path.abspath(os.path.join('YOLOv5','yolov5',ANNO,VERSION,'weights','best.pt'))
        dirProject = os.path.abspath(os.path.join(dirOutBase,PROJECT,SET))
        run(weights=dirWeights,source=dirSource,project=dirProject,name=species,imgsz=(1280, 1280),nosave=nosave)
        
    # for species in f:
    #     print(species)
    #     dirSource = os.path.abspath(os.path.join(dirDetailedBySpecies,species))
    #     dirYOLO = os.path.abspath(os.path.join('yolov5','detect.py'))
    #     dirWeights =  os.path.join('yolov5','runs','train',ANNO,VERSION,'weights','best.pt')
    #     dirProject = os.path.join(PROJECT,SET)
    #     if INCLUDE_SUBDIRS:
    #         run(weights=dirWeights,source=dirSource,project=dirProject,name=species,imgsz=(1280, 1280))
    #     else:
    #         run(weights=dirWeights,source=dirSource,project=dirProject,name=SET,imgsz=(1280, 1280))



### Parse the aggregated DetailedSample folder that has 2,500 images
# fileList, nameList = readDetailedSampleJPGs(dirDetailed)
# saveDetailedBySpecies(fileList, nameList)

### Run detect for every folder in dirDetailedBySpecies
# PROJECT = 'MAL'
# SET = 'Detailed'
# dirDetailedBySpecies = os.path.abspath(os.path.join(os.pardir,'Image_Datasets','GBIF_DetailedSample_50Spp_Ind'))
# runYOLOforDirOfFolders(dirDetailedBySpecies,PROJECT,SET,,1)

# dirDetailed = os.path.abspath(os.path.join(os.pardir,'Image_Datasets','GBIF_DetailedSample_50Spp'))
# dirDetailedBySpecies = os.path.abspath(os.path.join(os.pardir,'Image_Datasets','GBIF_DetailedSample_50Spp_Ind'))
# dirDetailedBySpecies = os.path.abspath(os.path.join(os.pardir,'Image_Datasets','GBIF_TargetedSample_Fraxinus'))


### Run detect for TargetedSample
# PREFIX = 'DT_MAL_'
# INCLUDE_SUBDIRS = 0
# PROJECT = 'Botany'#'Cannon'#'MAL_PLANT'
# SET = 'Demo'#'Test_Sheets_PREP'#'Targeted'
# ANNO = 'PREPfull'#'PLANTfull'
# VERSION = 'baseline_all_hypEvolve'#'baseline'

# dirDetailedBySpecies = os.path.abspath(os.path.join('Image_Datasets','Botany_Test_Images'))
# dirOutBase = os.path.abspath(os.path.join('YOLOv5'))
# runYOLOforDirOfFolders(dirDetailedBySpecies,dirOutBase,False,PROJECT,SET,ANNO,VERSION,INCLUDE_SUBDIRS)
# dirDetailedBySpecies = os.path.abspath(os.path.join('Image_Datasets','Cannon','Test_Sheets'))
# dirOutBase = os.path.abspath(os.path.join('YOLOv5'))
# runYOLOforDirOfFolders(dirDetailedBySpecies,dirOutBase,False,PROJECT,SET,ANNO,VERSION,INCLUDE_SUBDIRS)
# dirDetailedBySpecies = os.path.abspath(os.path.join(os.pardir,'Image_Datasets','GBIF_TargetedSample_Quercus_havardii'))
# runYOLOforDirOfFolders(dirDetailedBySpecies,PROJECT,SET,ANNO,VERSION,INCLUDE_SUBDIRS)
# dirDetailedBySpecies = os.path.abspath(os.path.join(os.pardir,'Image_Datasets','GBIF_TargetedSample_Fraxinus'))
# runYOLOforDirOfFolders(dirDetailedBySpecies,PROJECT,SET,ANNO,VERSION,INCLUDE_SUBDIRS)
# dirDetailedBySpecies = os.path.abspath(os.path.join(os.pardir,'Image_Datasets','GBIF_TargetedSample_Juglandaceae'))
# runYOLOforDirOfFolders(dirDetailedBySpecies,PROJECT,SET,ANNO,VERSION,INCLUDE_SUBDIRS)
# dirDetailedBySpecies = os.path.abspath(os.path.join(os.pardir,'Image_Datasets','GBIF_TargetedSample_Lonicera'))
# runYOLOforDirOfFolders(dirDetailedBySpecies,PROJECT,SET,ANNO,VERSION,INCLUDE_SUBDIRS)
# dirDetailedBySpecies = os.path.abspath(os.path.join(os.pardir,'Image_Datasets','GBIF_TargetedSample_Rhus'))
# runYOLOforDirOfFolders(dirDetailedBySpecies,PROJECT,SET,ANNO,VERSION,INCLUDE_SUBDIRS)

INCLUDE_SUBDIRS = 0
PROJECT = 'MAL_FP' #'MAL_PLANT'#'Botany'#'Cannon'#'MAL_PLANT'
SET = 'FieldPrism_Initial' #'Detailed'#'Demo_Plant'#'Test_Sheets_PREP'#'Targeted'
ANNO = 'PREPfull'#'PLANT_Botany'#'PLANTfull'#'PREPfull
VERSION = 'baseline'#'Small_Adoxaceae'#'baseline'
ANNO_TYPE = 'PREP'

# # dirDetailedBySpecies = os.path.abspath(os.path.join('Image_Datasets','FieldPrism_Training_Images','FieldPrism_Training_FS-Poor'))
# dirDetailedBySpecies = 'D:/Dropbox/LM2_Env/Image_Datasets/FieldPrism_Training_Images/FieldPrism_Training_Sheets'
# # dirOutBase = os.path.abspath(os.path.join('YOLOv5'))
# dirOutBase = os.path.dirname(__file__)
# # dirOutBase > PROJECT > SET
# # ML network: ANNO > VERSION
# make_file_names_valid(dirDetailedBySpecies)
# runYOLOforDirOfFolders(dirDetailedBySpecies,dirOutBase,False,PROJECT,SET,ANNO,VERSION,INCLUDE_SUBDIRS,ANNO_TYPE)

# dirDetailedBySpecies = 'D:/Dropbox/LM2_Env/Image_Datasets/FieldPrism_Training_Images/FieldPrism_Training_Outside'
# dirOutBase = os.path.dirname(__file__)
# make_file_names_valid(dirDetailedBySpecies)
# runYOLOforDirOfFolders(dirDetailedBySpecies,dirOutBase,False,PROJECT,SET,ANNO,VERSION,INCLUDE_SUBDIRS,ANNO_TYPE)

dirDetailedBySpecies = 'D:/Dropbox/LM2_Env/Image_Datasets/FieldPrism_Training_Images/FieldPrism_Training_FS-Poor'
dirOutBase = os.path.dirname(__file__)
make_file_names_valid(dirDetailedBySpecies)
runYOLOforDirOfFolders(dirDetailedBySpecies,dirOutBase,False,PROJECT,SET,ANNO,VERSION,INCLUDE_SUBDIRS,ANNO_TYPE)

# dirDetailedBySpecies = 'D:/Dropbox/LM2_Env/Image_Datasets/FieldPrism_Training_Images/REU_Field_QR-Code-Images'
# dirOutBase = os.path.dirname(__file__)
# make_file_names_valid(dirDetailedBySpecies)
# runYOLOforDirOfFolders(dirDetailedBySpecies,dirOutBase,False,PROJECT,SET,ANNO,VERSION,INCLUDE_SUBDIRS,ANNO_TYPE)