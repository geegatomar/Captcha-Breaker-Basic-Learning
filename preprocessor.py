# takes an image as input, preprocesses it and returns contours

import numpy as np
import imutils
import cv2
import os

class Preprocessor:
    @staticmethod
    def find_contours(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        # Blurring and canny is a bad idea for this question as the images are too small. So just do thresh.
        #blurred = cv2.GaussianBlur(image, (3, 3), 0)
        #edged = cv2.Canny(blurred, 30, 180)
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[1] if imutils.is_cv3() else contours[0]
        return image, contours


