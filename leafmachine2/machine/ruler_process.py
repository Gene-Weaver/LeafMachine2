import os, cv2, argparse
import torch
from ruler_utils import create_overlay_bg, validate_dir
from ruler_utils import ClassifyRulerImage

'''
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
'''

'''
# imgLoc = os.path.join('Image_Datasets','GroundTruth_CroppedAnnotations_Group1_Partial','PREP','Ruler','Ruler__BHSC_2512328797_Vitaceae_Vitis_riparia__2.jpg')

# imgLoc = os.path.join('E:','TEMP_ruler','Rulers_3ofEachType')
# pathOverlay = os.path.join('E:','TEMP_ruler','Rulers_3ofEachType_Overlay')

# labelNames = os.path.join('LeafMachine2','data','ruler_classifier','ruler_classes.txt')
# modelPath = os.path.abspath(os.path.join('LeafMachine2','data','Ruler_Classifier','model'))
# modelName='model_scripted_resnet.pt'
# imgSize = 360
'''

def detect_ruler(imgPath,printResult,saveOverlay,pathOverlay,modelPath,modelName,labelNames):
    
    img = ClassifyRulerImage(img_path=imgPath)

    net = torch.jit.load(os.path.join(modelPath,modelName))
    net.eval()

    with open(os.path.abspath(labelNames)) as f:
        classes = [line.strip() for line in f.readlines()]


    out = net(img.img_tensor)
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

    _, index = torch.max(out, 1)
    percentage1 = torch.nn.functional.softmax(out, dim=1)[0] * 100
    percentage1 = round(percentage1[index[0]].item(),2)
    pred_class1 = classes[index[0]]

    if saveOverlay:
        imgBG = create_overlay_bg(img.img_sq)
        addText1 = "Class: "+str(pred_class1)
        addText2 = "Certainty: "+str(percentage1)
        newName = os.path.split(imgPath)[1]
        newName = newName.split(".")[0] + "__overlay.jpg"
        imgOverlay = cv2.putText(img=imgBG, text=addText1, org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(155, 155, 155),thickness=1)
        imgOverlay = cv2.putText(img=imgOverlay, text=addText2, org=(10, 45), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(155, 155, 155),thickness=1)
        cv2.imwrite(os.path.abspath(os.path.join(pathOverlay,newName)),imgOverlay)


    if printResult:
        # print(f'Image: {imgPath.split(sep=os.path.sep)[-1]}\n     Class: {pred_class1} Certainty: {percentage1}\n')
        print(f'Class: {pred_class1} Certainty: {percentage1}\n')

    return pred_class1,percentage1,imgOverlay



def ruler_process(imgLoc,printResult,saveOverlay,pathOverlay,modelPath,modelName):
    labelNames = os.path.join('leafMachine2','machine','ruler_classifier','ruler_classes.txt')
    if saveOverlay:
        validate_dir(pathOverlay)

    imgSingle = True
    try:
        imgDir = os.listdir(imgLoc)
        imgSingle = False
    except:
        imgPath = imgLoc
        imgSingle = True

    if imgSingle == True:
        pred_class,percentage,imgOverlay = detect_ruler(imgPath=imgPath,
        printResult=printResult,
        saveOverlay=saveOverlay,
        pathOverlay=pathOverlay,
        modelPath=modelPath,
        modelName=modelName,
        labelNames=labelNames)
    elif imgSingle == False:
        for img in imgDir:
            imgPath = os.path.abspath(os.path.join(imgLoc,img))
            pred_class,percentage,imgOverlay = detect_ruler(imgPath=imgPath,
            saveOverlay=saveOverlay,
            pathOverlay=pathOverlay,
            printResult=True,
            modelPath=modelPath,
            modelName=modelName,
            labelNames=labelNames
            )

def parse_ruler_process():
    parser = argparse.ArgumentParser(
            description='Parse inputs for ruler_process.py',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    optional_args = parser._action_groups.pop()
    required_args = parser.add_argument_group('MANDATORY arguments')
    required_args.add_argument('--dir-images',
                                type=str,
                                required=True,
                                help='')
    required_args.add_argument('--dir-overlay',
                                type=str,
                                required=True,
                                help='')
    required_args.add_argument('--print',
                                type=bool,
                                required=True,
                                help='print results: class and certainty')
    required_args.add_argument('--save-overlay',
                                type=bool,
                                required=True,
                                help='')
    required_args.add_argument('--model-path',
                                type=str,
                                required=True,
                                help='')
    required_args.add_argument('--model-name',
                                type=str,
                                required=True,
                                help='')
    parser._action_groups.append(optional_args)
    args = parser.parse_args()

    return args

def run():
    args = parse_ruler_process()
    ruler_process(args.dir_images,args.print,args.save_overlay,args.dir_overlay,args.model_path,args.model_name)

if __name__ == '__main__':
    run()