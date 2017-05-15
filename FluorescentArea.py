import numpy as np
import cv2
import pylab
from matplotlib import pyplot as plt

def CleanLoad(file_name,zero_mask = np.array([])):
    rawImg = cv2.imread(file_name,cv2.IMREAD_UNCHANGED)
    
    #zero out zero_mask if present
    if zero_mask.any():
        rawImg = cv2.min(rawImg,zero_mask*np.amax(rawImg))
    
    #switch to 8bit with proper normalizing
    raw_max = np.amax(rawImg)
    raw_median = np.median(rawImg[rawImg > 0])
    
    #crush extreme outliers
    raw_norm = min(raw_max,2*raw_median)
    if raw_norm > 255:
        img = cv2.convertScaleAbs(rawImg, alpha = (255.0/raw_norm))
    else:
        img = cv2.convertScaleAbs(rawImg)
    
    return img

def Phase2Mask(imgP, circ_edge = 125, dilate_margin = 9, sp_margin = 3):
    #cancel out median glow
    img_med = cv2.medianBlur(imgP,2*circ_edge + 1)
    imgP = cv2.addWeighted(imgP,1,img_med,-1,np.median(imgP))
    
    #isolate the upper half and lower half of histogram
    ret, top_half = cv2.threshold(imgP,130,1,cv2.THRESH_BINARY)
    ret, bot_half = cv2.threshold(imgP,125,1,cv2.THRESH_BINARY_INV)
    
    #cut out the center
    zm = cv2.max(top_half,bot_half)
    
    #cut away the rim
    circ = np.zeros_like(zm)
    cv2.circle(circ,(circ.shape[0]//2,circ.shape[1]//2),circ.shape[0]//2 - circ_edge, 1,-1)
    zm = cv2.min(zm,circ)
    
    #remove salt-pepper noise
    kernel = np.ones((sp_margin,sp_margin),np.uint8)
    zm = cv2.morphologyEx(zm, cv2.MORPH_OPEN, kernel, iterations = 1)
    
    #dilate for safety margin
    kernel = np.ones((dilate_margin,dilate_margin),np.uint8)
    zm = cv2.dilate(zm,kernel,iterations = 1)
    
    return zm

def TripleShow(imgP, imgC1, imgC2):
    plt.subplot(1,3,1)
    plt.imshow(imgP)
    
    plt.subplot(1,3,2)
    plt.imshow(imgC1)
    
    plt.subplot(1,3,3)
    plt.imshow(imgC2)
    
    plt.show()
    

def FullLoad(col_let, row_num, fot_num, dirName = '', show = False, whole_well = False):
    #load the images
    head = dirName + col_let + str(row_num) + '-' + str(fot_num)
    
    imgP = cv2.imread(head + '-P.tif',cv2.IMREAD_UNCHANGED)
    
    if whole_well:
        #zm = Phase2Mask(imgP)
        #use circle mask
        zm = np.zeros_like(imgP)
        cv2.circle(zm,(zm.shape[0]//2,zm.shape[1]//2),zm.shape[0]//2 - 125, 1,-1)
        
        imgC1 = CleanLoad(head + '-C1.tif', zero_mask = zm)
        imgC2 = CleanLoad(head + '-C2.tif', zero_mask = zm)
    else:
        imgC1 = CleanLoad(head + '-C1.tif')
        imgC2 = CleanLoad(head + '-C2.tif')
    
    if show:
        TripleShow(imgP,imgC1,imgC2)
    
    return imgP, imgC1, imgC2


def CombineChannels(imgBg,imgC1,imgC2, bW = 0.85):
    return cv2.merge((
            cv2.addWeighted(imgBg,bW,imgC1,1- bW,1),
            cv2.addWeighted(imgBg,bW,imgC2,1 - bW,1),
            cv2.addWeighted(imgBg,bW,np.zeros_like(imgBg), 1 - bW, 1)))

def FluorescentAreaMark(img, gridSize = 20):
    #1: contrast limited adaptive histogram equalization to get rid of glow
    clahe = cv2.createCLAHE(clipLimit=50.0, tileGridSize=(gridSize,gridSize))

    clahe_img = clahe.apply(img)
    
    #2: threshold and clean out salt-pepper noise
    ret, thresh_img = cv2.threshold(clahe_img,127,255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3),np.uint8)

    clean_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations = 2)
    
    return clean_img

def PercentileMark(img,per_thresh = 75):
    pix_thresh = np.percentile(img[img > 0],per_thresh)
    
    ret, thresh_img = cv2.threshold(img,pix_thresh,255,cv2.THRESH_BINARY)
    
    kernel = np.ones((3,3),np.uint8)

    clean_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations = 2)
    
    return clean_img

def AreaCount(col_let, row_num, fot_num, dirName = '', whole_well = False, ign_buf = 30):
    #load the images
    grImage, rdImage8, gnImage8 = FullLoad(col_let,row_num,fot_num,dirName,whole_well = whole_well)
    
    #get the area masks
    rdFA = FluorescentAreaMark(rdImage8)
    gnFA = FluorescentAreaMark(gnImage8)
    
    rd_area = (rdFA[ign_buf:-ign_buf,ign_buf:-ign_buf] > 0)
    gn_area = (gnFA[ign_buf:-ign_buf,ign_buf:-ign_buf] > 0)
    
    #create image to save
    img_out = []
    if ret_img:
        img_out = CombineChannels(grImage,rdFa,gnFA)
    
    return [np.sum(rd_area),np.sum(gn_area)], img_out
