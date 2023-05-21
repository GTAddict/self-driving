import cv2
from grayscale import *
from blur import *

def get_canny(image):
    #gray = bgr_to_gray_luminosity(image)   # My own implementation for learning,
    #blur = gaussian_blur(gray, 5)          # but too slow since it doesn't use fft
    canny = cv2.Canny(image, 50, 150)
    return canny