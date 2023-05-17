import numpy as np
import scipy.signal
import time

arr1 = np.random.random(1000000)
arr2 = np.random.random(1000000)
start = time.time()
print (start)
convolve_one = np.convolve(arr1, arr2)
end = time.time()
print (end)
convolve_two = scipy.signal.fftconvolve(arr1, arr2)
end2 = time.time()
print (end2)
print (end - start)
print (end2 - end)
print (convolve_one in convolve_two)


