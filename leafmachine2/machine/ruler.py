'''
Main function for analyzing the extracted ruler image

'''
import os, argparse

from utils_ruler import RulerConfig, setup_ruler, convert_pixels_to_metric

# def ruler(img_dir):
def ruler(RulerCFG,machine_or_dir):
    '''
    For "machine:
    call:
    RulerCFG = RulerConfig()
    prior to looping through images so that the ML network only loads once
    '''
    if machine_or_dir == "machine": # Process rulers as they are sent over from archival component detector
        Ruler = setup_ruler(RulerCFG,os.path.join(img_dir,ruler_fname),ruler_fname)
        # Ruler = straightenImage(Info,os.path.join(img_dir,ruler),ruler)
        convert_pixels_to_metric(RulerCFG,Ruler,ruler_fname)
    elif machine_or_dir == "dir":
        img_dir = RulerCFG.cfg['leafmachine']['images_to_process']['ruler_directory']
        img_ruler_list = os.listdir(img_dir)
        for ruler_fname in img_ruler_list:
            Ruler = setup_ruler(RulerCFG,os.path.join(img_dir,ruler_fname),ruler_fname)
            # Ruler = straighten_img(RulerCFG,os.path.join(img_dir,ruler_fname),ruler_fname)
            convert_pixels_to_metric(RulerCFG,Ruler,ruler_fname)
    

    # img_dir = os.path.abspath(os.path.join('Image_Datasets','GroundTruth_CroppedAnnotations_Group1_Partial','PREP','Ruler'))
    # ruler = 'Ruler__ALAM_18944141_Asteraceae_Grindelia_decumbens__2.jpg'

    # Strange block, whiter than normal: 'Ruler__NY_1928144823_Lecythidaceae_Lecythis_pisonis__1.jpg'
    # big problem: Ruler__DAV_2421708402_Brassicaceae_Lepidium_nitidum__1   Ruler__GH_2425426726_Rosaceae_Prunus_virginiana__1    Ruler__HXC_8576868_Asteraceae_Aster_sagittifolius__1



    # pathToModel = os.path.join(os.getcwd(),'LeafMachine2','data','Ruler_Classifier','model')
    # pathToLabelNames = os.path.join('LeafMachine2','data','ruler_classifier','ruler_classes.txt')
    # cfg = loadCFG(pathToCfg)

    # dir_ruler_correction = Info.cfg['leafmachine']['save_dirs']['ruler_correction']
    # pathToSave = Info.cfg['leafmachine']['save_dirs']['ruler_correction']
    # validateDir(pathToSave)

    # if cfg['leafmachine']['save']['ruler_correction']:
        # validateDir(os.path.join(dir_ruler_correction,'ruler_correction'))
    # if cfg['leafmachine']['save']['ruler_correction_compare']:
        # validateDir(os.path.join(dir_ruler_correction,'ruler_correction_compare'))
    # if cfg['leafmachine']['save']['ruler_type_overlay']:
    #     validateDir(os.path.join(dir_ruler_correction,'ruler_type_overlay'))
    
    # For gray tiffen rulers: use --> thresh, imgBi = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    # tempOut = os.path.abspath(os.path.join('E:','TEMP_ruler','rulerAngleCorrection'))
    
    # Ruler = setupRuler(Info,os.path.join(img_dir,ruler),ruler)
    # Ruler = straightenImage(Info,Ruler,useRegulerBinary=True,alternate_img=0)# Info,os.path.join(img_dir,ruler),ruler) # Just one image
    # convertPixelsToMetric(Info,Ruler,ruler)
    # img_ruler_list = os.listdir(os.path.abspath(os.path.join('Image_Datasets','GroundTruth_CroppedAnnotations_Group1_Partial','PREP','Ruler')))




    # img_dir = os.path.abspath(os.path.join('Image_Datasets','Cannon','REU_Cropped_Leaves','Morton','Morton_REU_Cropped_Rulers'))
    




# def main(imgLoc,printResult,saveOverlay,pathOverlay):
#     labelNames = os.path.join('LeafMachine2','data','ruler_classifier','ruler_classes.txt')
#     if saveOverlay:
#         validate_dir(pathOverlay)

#     imgSingle = True
#     try:
#         imgDir = os.listdir(imgLoc)
#         imgSingle = False
#     except:
#         imgPath = imgLoc
#         imgSingle = True

#     if imgSingle == True:
#         pred_class,percentage = detectRuler(imgPath=imgPath,
#         printResult=printResult,
#         saveOverlay=saveOverlay,
#         pathOverlay=pathOverlay,
#         modelPath=modelPath,
#         modelName=modelName)
#     elif imgSingle == False:
#         for img in imgDir:
#             imgPath = os.path.abspath(os.path.join(imgLoc,img))
#             pred_class,percentage = detectRuler(imgPath=imgPath,
#             saveOverlay=saveOverlay,
#             pathOverlay=pathOverlay,
#             printResult=True,
#             modelPath=modelPath,
#             modelName=modelName,
#             labelNames=labelNames
#             )

def parse_ruler():
    parser = argparse.ArgumentParser(
            description='Parse inputs for ruler.py',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    optional_args = parser._action_groups.pop()
    required_args = parser.add_argument_group('MANDATORY arguments')
    required_args.add_argument('--dir-images',
                                type=str,
                                required=True,
                                help='')
    parser._action_groups.append(optional_args)
    args = parser.parse_args()

    return args

def run():
    RulerCFG = RulerConfig()
    # args = parse_ruler()
    # ruler(args.dir_images)
    ruler(RulerCFG,"dir")

if __name__ == '__main__':
    run()