import cv2
from grayscale import *

images = ['test_image.jpg', 'colors.jpg']

for image in images:
    image = cv2.imread(image)
    cv2.imshow('result', rgb_to_gray_average(image))
    cv2.waitKey()
    cv2.imshow('result', rgb_to_gray_desaturation(image))
    cv2.waitKey()
    cv2.imshow('result', rgb_to_gray_luminosity(image))
    cv2.waitKey()
    cv2.imshow('result', rgb_to_gray_max_decomp(image))
    cv2.waitKey()
    cv2.imshow('result', rgb_to_gray_min_decomp(image))
    cv2.waitKey()
    cv2.imshow('result', rgb_to_gray_by_channel(image, 0))
    cv2.waitKey()
    cv2.imshow('result', rgb_to_gray_by_channel(image, 1))
    cv2.waitKey()
    cv2.imshow('result', rgb_to_gray_by_channel(image, 2))
    cv2.waitKey()