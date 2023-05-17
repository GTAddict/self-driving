import cv2
from grayscale import *
from blur import *

image = cv2.imread('test_image.jpg')
# convert bgr to rgb
gray_image = rgb_to_gray_luminosity(image)
kernel = np.full(9 * 9, 1/(9 * 9)).reshape(3, 27)
cv2.imshow('result', convolve(gray_image, kernel))
cv2.waitKey()
cv2.imshow('result', gray_image)
cv2.waitKey()
