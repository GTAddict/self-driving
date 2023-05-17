import numpy as np
from itertools import product

def generate_average_kernel_two_d(size):
    return np.full(size * size, 1/(size * size)).reshape(size, size)

def generate_gaussian_kernel_two_d(size):
    return

def simple_blur(image, intensity):
    return convolve(image, generate_average_kernel_two_d(intensity))

def convolve(input, kernel):
    kernel_num_rows, kernel_num_cols = kernel.shape
    padding_rows = (kernel_num_rows - 1) // 2
    padding_cols = (kernel_num_cols - 1) // 2
    kernel_extent_x = kernel_num_rows // 2
    kernel_extent_y = kernel_num_cols // 2

    padded_input = np.pad(input, ((padding_rows, padding_rows), (padding_cols, padding_cols)), 'constant', constant_values=0)
    padded_input_rows, padded_input_cols = padded_input.shape

    output = np.empty_like(input)
    flattened_kernel = kernel.flatten()
    
    for row, column in product(range(padding_rows, padded_input_rows - padding_rows), range(padding_cols, padded_input_cols - padding_cols)):
        flattened_input = padded_input[row - kernel_extent_x : row + kernel_extent_x + 1, column - kernel_extent_y : column + kernel_extent_y + 1].flatten()
        output[row - padding_rows][column - padding_cols] = np.sum(flattened_input * flattened_kernel)

    return output