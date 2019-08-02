import dlib
import cv2
import os
import numpy as np
def place(target):
    global my_face
    
#over laying code found at https://www.codesofinterest.com/2017/07/snapchat-like-filters-with-dlib-opencv-python.html

#loading in all the needed components
my_face = cv2.imread('me.jpg')
target = cv2.imread('test.jpg')
win = dlib.image_window()
predictor_path = os.getcwd()+"/68point.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
img = dlib.load_grayscale_image(os.getcwd()+"/me.jpg")

win.set_image(img)
dlib.hit_enter_to_continue()