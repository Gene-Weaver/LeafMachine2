import os
import cv2
from cv2 import imwrite
from squarifyImage import squarify
from preprocessRuler import makeImgHor,validateDir

def squarifyDirs(root,dirOut,makeSquare,sz):
    for path, subdirs, files in os.walk(root):
        for subdir in subdirs:
            print(subdir)
            subPath = os.path.join(root,subdir)
            dirOutSub = os.path.abspath(os.path.join(dirOut,subdir))
            validateDir(dirOutSub)

            files = os.listdir(subPath)
            for name in files:
                imgPath=os.path.join(subPath, name)
                # print(os.path.join(subdir, name))
                img = cv2.imread(imgPath)
                img = makeImgHor(img)
                img = squarify(img,False,makeSquare,sz)
                cv2.imwrite(os.path.join(dirOutSub,name),img)

def main():
    dirToSquare = os.path.join('E:','TEMP_ruler','Rulers_ByType')
    dirOut =  os.path.join('E:','TEMP_ruler','Rulers_ByType_Squarify')
    makeSquare = True
    sz = 360
    validateDir(dirOut)
    squarifyDirs(dirToSquare,dirOut,makeSquare,sz)

main()