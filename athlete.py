import dlib
import cv2
import os
import numpy as np   
def place_eye(frame, d, my_face, scale):  
    #over laying code found at https://www.codesofinterest.com/2017/07/snapchat-like-filters-with-dlib-opencv-python.html
    #modified from the psot   
    global orig_mask
    global orig_mask_inv
   
    x1 = d.left()  #top left
    x2 = d.right()
    y1 = d.top()     
    y2 = d.bottom()
    face_h = y2 - y1
    face_w = x2 - x1
   
    h, w = frame.shape[:2]  
   
    y1 = int(y1 - face_h * 0.8)
    y2 = int(y2 + face_h * 0.2)
    x1 = int(x1 - face_w * 0.3)
    x2 = int(x2 + face_w * 0.3)
    # check for clipping  
    if x1 < 0:  
        x1 = 0  
    if y1 < 0:  
        y1 = 0  
    if x2 > w:  
        x2 = w  
    if y2 > h:  
        y2 = h  
   
    # re-calculate the size to avoid clipping  
    eyeOverlayWidth = x2 - x1  
    eyeOverlayHeight = y2 - y1  
   
    # calculate the masks for the overlay  
    eyeOverlay = cv2.resize(my_face, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
    mask = cv2.resize(orig_mask, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
    mask_inv = cv2.resize(orig_mask_inv, (eyeOverlayWidth,eyeOverlayHeight), interpolation = cv2.INTER_AREA)  
   
    # take ROI for the verlay from background, equal to size of the overlay image  
    roi = frame[y1:y2, x1:x2]  
   
    # roi_bg contains the original image only where the overlay is not, in the region that is the size of the overlay.  
    roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)  
   
    # roi_fg contains the image pixels of the overlay only where the overlay should be  
    roi_fg = cv2.bitwise_and(eyeOverlay,eyeOverlay,mask = mask)  
   
    # join the roi_bg and roi_fg  
    dst = cv2.add(roi_bg,roi_fg)  
   
    # place the joined image, saved to dst back over the original image  
    frame[y1:y2, x1:x2] = dst  



#loading in all the needed components
target_file = os.getcwd()+"/test.jpg"
my_face = cv2.imread('face.png', -1)
orig_mask = my_face[:,:,3]  
orig_mask_inv = cv2.bitwise_not(orig_mask) 
my_face = my_face[:,:,0:3]  
predictor_path = os.getcwd()+"/68point.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
img = dlib.load_grayscale_image(target_file)
img_c = cv2.imread(target_file)
#starting to process image to find faces and key points
dets = detector(img, 1)
for i, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
    place_eye(img_c, d, my_face, 1.5)
    #import numpy as np
cv2.imwrite('out.jpg',img_c)
cv2.imshow('image',img_c)
cv2.waitKey(0)

