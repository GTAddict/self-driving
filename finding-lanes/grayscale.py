import numpy as np

def rgb_to_gray_average(image):
    gray_image = image // 3 # we lose some slight precision here but it's minimal
    return (gray_image[..., 0] + gray_image[..., 1] + gray_image[..., 2])

def rgb_to_gray_desaturation(image):
    return ((np.amax(image, 2) / 2) + (np.amin(image, 2) / 2)).astype('uint8')
    
def rgb_to_gray_luminosity(image):
    return (image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114).astype('uint8')

def rgb_to_gray_max_decomp(image):
    return np.amax(image, 2)

def rgb_to_gray_min_decomp(image):
    return np.amin(image, 2)

def rgb_to_gray_by_channel(image, channel):
    return image[...,channel]