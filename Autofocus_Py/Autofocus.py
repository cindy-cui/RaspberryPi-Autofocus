import cv2 #sudo apt-get install python-opencv
import numpy as np
import os
import time
import sys
from ctypes import *
import matplotlib.pyplot as plt
#load arducam shared object file
arducam_vcm= CDLL('./lib/libarducam_vcm.so')
try:
    import picamera
    from picamera.array import PiRGBArray
except:
	sys.exit(0)

def focusing(val):
    arducam_vcm.vcm_write(val)
    #print("focus value: {}".format(val))

def sobel(img):
	img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	img_sobel = cv2.Sobel(img_gray,cv2.CV_16U,1,1)
	return cv2.mean(img_sobel)[0]

def laplacian(img):
	img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	img_laplacian = cv2.Laplacian(img_gray,cv2.CV_16U)
	return img_laplacian.var()

def getBlurValue(img):
    canny = cv2.Canny(img, 50,250)
    return np.mean(canny)
	

def calculation(camera):
	rawCapture = PiRGBArray(camera) 
	camera.capture(rawCapture,format="bgr", use_video_port=True)
	image = rawCapture.array
	rawCapture.truncate(0)
	return laplacian(image)

def CMSL(img, j, k, n):
    """
    Contrast Measure based on squared Laplacian according to
    'Robust Automatic Focus Algorithm for Low Contrast Images
    Using a New Contrast Measure'
    by Xu et Al. doi:10.3390/s110908281
    """
    ky1 = np.array(([0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]))
    ky2 = np.array(([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]))
    kx1 = np.array(([0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]))
    kx2 = np.array(([0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]))
    g_img = abs(cv2.filter2D(img, cv2.CV_32F, kx1)) + \
            abs(cv2.filter2D(img, cv2.CV_32F, ky1)) + \
            abs(cv2.filter2D(img, cv2.CV_32F, kx2)) + \
            abs(cv2.filter2D(img, cv2.CV_32F, ky2))
    ret = cv2.boxFilter(
                            g_img * ((g_img + 1) ** (1.0 / n) - 1),
                            -1,
                            (j, k),
                            normalize=True)
    return ret


	

if __name__ == "__main__":

    #vcm init
    i = 0
    arducam_vcm.vcm_init()
    camera = picamera.PiCamera()
    focusing(i)
    camera.start_preview()
	#set camera resolution to 640x480(Small resolution for faster speeds.)
    camera.resolution = (640, 480)
    # time.sleep(5)
    camera.shutter_speed=30000
    calculation(camera)

    # Autofocus algorithm proposed in https://asp-eurasipjournals.springeropen.com/articles/10.1186/s13634-016-0368-5
    # Step 1. Global Search
    starting_point = 0
    ending_point = 1024
    step = 100

    loop_number = 0
    max_loop = 3
    prev_max_dist = 0

    while loop_number < max_loop:
        print("loop #" + str(loop_number))
        i = starting_point

        max_clarity = 0.0
        max_dist = 0
        intervals = (ending_point - starting_point) / step
        threshold = 0.05
        prev_clarity = 0
        decrease_counter = 0

        while i <= ending_point:
            
            focusing(i)
            clarity_score = calculation(camera)
            print("current dist: " + str(i) + ", current score: " + str(clarity_score))
            if clarity_score > max_clarity:
                max_dist = i
                max_clarity = clarity_score
            if prev_clarity - clarity_score > threshold:
                decrease_counter += 1
                if decrease_counter >= 3:
                    break
            prev_clarity = clarity_score
            i += step
        print("max dist of this loop: " + str(max_dist) + ", score = " + str(max_clarity))
        
        if loop_number != 0 and max_dist == prev_max_dist:
            break

        loop_number += 1
        starting_point = max_dist - step / 2
        ending_point = max_dist + step / 2
        step = max(1, step / 5)
        prev_max_dist = max_dist

    
    focusing(max_dist)

    print("done, dist = " + str(max_dist))

    camera.resolution = (1920,1080)
    #save image to file.
    camera.capture(sys.argv[1])


