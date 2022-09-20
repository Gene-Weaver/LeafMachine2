import cv2
import numpy as np
import math

def createWhiteBG(img,squarifyRatio,h,w):
    w_plus = w
    # if (w_plus % squarifyRatio) != 0:
    # while (w_plus % squarifyRatio) != 0:
    #     w_plus += 1
    
    imgBG = np.zeros([h,w_plus,3], dtype=np.uint8)
    imgBG[:] = 255

    imgBG[:img.shape[0],:img.shape[1],:] = img
    # cv2.imshow('Single Channel Window', imgBG)
    # cv2.waitKey(0)
    return imgBG

def makeImgHor(img):
    # Make image horizontal
    h,w,c = img.shape
    if h > w:
        img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def stackImage(img,squarifyRatio,h,w_plus,showImg):
    # cv2.imshow('Original', img)
    wChunk = int(w_plus/squarifyRatio)
    hTotal = int(h*squarifyRatio)
    imgBG = np.zeros([hTotal,wChunk,3], dtype=np.uint8)
    imgBG[:] = 255

    wStart = 0
    wEnd = wChunk
    for i in range(1,squarifyRatio+1):
        wStartImg = (wChunk*i)-wChunk
        wEndImg =  wChunk*i
        
        hStart = (i*h)-h
        hEnd = i*h
        # imgPiece = img[:,wStartImg:wEndImg]
        # cv2.imshow('Single Channel Window', imgPiece)
        # cv2.waitKey(0)
        imgBG[hStart:hEnd,wStart:wEnd] = img[:,wStartImg:wEndImg]
    if showImg:
        cv2.imshow('Single Channel Window', imgBG)
        cv2.waitKey(0)
    return imgBG

def calcSquarifyRatio(img):
    doStack = False
    h,w,c = img.shape

    # Extend width so it's a multiple of h
    ratio = w/h
    ratio_plus = math.ceil(ratio)
    w_plus = ratio_plus*h

    ratio_go = w/h
    if ratio_go > 4:
        doStack = True

    squarifyRatio = 0
    if doStack:
        # print(f'This should equal 0 --> {w_plus % h}')
        for i in range(1,ratio_plus):
            if ((i*h) < (w_plus/i)):
                continue
            else:
                squarifyRatio = i - 1
                break
        # print(f'Optimal stack_h: {squarifyRatio}')
        while (w % squarifyRatio) != 0:
            w += 1
    return doStack,squarifyRatio,w,h

def squarify(imgSquarify,showImg,makeSquare,sz):
    imgSquarify = makeImgHor(imgSquarify)
    doStack,squarifyRatio,w_plus,h = calcSquarifyRatio(imgSquarify)

    if doStack:
        imgBG = createWhiteBG(imgSquarify,squarifyRatio,h,w_plus)
        imgSquarify = stackImage(imgBG,squarifyRatio,h,w_plus,showImg)

    if makeSquare:
        dim = (sz, sz)
        imgSquarify = cv2.resize(imgSquarify, dim, interpolation = cv2.INTER_AREA)

    return imgSquarify