# Edson Yudi Toma - 9791305
# SCC0251 Image Processing - Prof. Moacir Ponti
# 1st Semester of 2019
# Assignment 3 : Image restoration
import numpy as np
import imageio as imio
from PIL import Image # uncomment to see the images

def iqr(arr):
    arr = np.sort(np.reshape(arr, -1)) # Reshape into 1-D array
    return arr[(len(arr)*3)//4] - arr[len(arr)//4] # Sort and subtract the lower median from the upper median to find the interquartile range

def denoising(img, size, gamma):
    mode = input()
    height, width = img.shape
    (centrality, dispertion) = (np.mean, np.std) if "average" in mode else (np.median, iqr) # Assign each function according with the {mode} input
    disp_n = dispertion(img[0:(height//6)-1, 0:(width//6)-1]) # {disp_n} established in the description
    if disp_n == 0:
        disp_n = 1 # {disp_n} cannot be 0

    span = size//2 # The amount of pixels to be gathered in a submatrix

    # The below code simply creates two lists with the centrality and dispersion measure for each submatrix in the img
    # This was done in a desperate attempt to run the test cases in time =) (but the main issue was the iqr function)
    disp_l = [[dispertion(img[i-span:i+span+1, j-span:j+span+1]) for j in range(span, width-span)] for i in range(span, height-span)]
    centr_l = [[centrality(img[i-span:i+span+1, j-span:j+span+1]) for j in range(span, width-span)] for i in range(span, height-span)]
    disp_l = [[1 if disp == 0 else disp for disp in disp_l_array] for disp_l_array in disp_l] # {disp_l} cannot be 0

    newImg = np.copy(img)
    for i in range(span, height-span):
        for j in range(span, width-span):
            newImg[i, j] = (img[i, j] - (gamma * (disp_n / disp_l[i-span][j-span]) * (img[i, j] - centr_l[i-span][j-span])))
    
    return newImg

def gaussian_filter(k=3, sigma=1.0):
    arx = np.arange((-k//2)+1.0, (k//2)+1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp(-(1/2)*(np.square(x)+np.square(y))/np.square(sigma))
    return filt/np.sum(filt)

'''
pad(img, filt):
    Pads a filter with 0 so it can be the same size as the given image
    If the height and width of the image is even,
    the left and upper pad must have and additional col/row
    If only the height is even,
    the upper pad must have and additional row
    If only the width of the image is even, 
    the left pad must have and additional col
    Args:
        * image {img}
        * filter {filt}
    Return value:
        Padded filter with the same dimensions of the given image
'''
def pad(img, filt):
    hi, wi = img.shape
    hf, wf = filt.shape

    # Both even
    if not hi%2 and not wi%2:
        ht = ((hi-hf)//2 + 1, (hi-hf)//2)
        wt = ((wi-wf)//2 + 1, (wi-wf)//2)
    # Height even
    elif not hi%2:
        ht = ((hi-hf)//2 + 1, (hi-hf)//2)
        wt = ((wi-wf)//2, (wi-wf)//2)
    # Width even
    elif not wi%2:
        ht = ((hi-hf)//2, (hi-hf)//2)
        wt = ((wi-wf)//2 + 1, (wi-wf)//2)

    return np.pad(filt, (ht, wt), 'constant')

def deblurring(img, size, gamma):
    height, width = img.shape # {img} dimensions
    laplace = pad(img, np.asarray([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])) # Laplacian kernel
    sigma = np.abs(float(input()))
    degradation_function = pad(img, gaussian_filter(size, sigma)) # Assign the degradation function according to the description

    fft = np.fft.fft2 # The fast fourier transform 
    ifft = np.fft.ifft2 # The inverse fast fourier transform
    shift = np.fft.fftshift # The shift method to recenter the image

    G = fft(img) # The fast fourier transform of the image
    P = fft(laplace) # The fast fourier transform applied to the laplacian filter
    H = fft(degradation_function) # The fast fourier transform applied to the degradation function
    A = H.conj() / ((np.square(np.abs(H)))+(gamma*np.square(np.abs(P)))) # The filter (First part of the formula presented in the description)
    
    newImg = A*G # Applying the filter

    return np.real(shift(ifft(newImg))) # Returns a recentered image

'''
normalize(matrix, normal):
    Normalizes a given 2d array with the given normal factor
    Args:
        * matrix - 2 dimentional array to be normalized
        * normal - max value to be used as reference, the min value will be set as 0
    Return value:
        * normalized 2d array
'''
def normalize(img, normal = 255):
    max = np.amax(img)
    min = np.amin(img)
    dif = max - min
    img = (img - min) * (normal / dif)
    return img

'''
rmsd(m1, m2):
    Calculates the Root-mean-square deviation between the two given 2d arrays
    Args:
        * m1 - 2d array
        * m2 - 2d array
    Return value:
        * float error
'''
def rmsd(m1, m2):
    return np.sqrt(np.mean(np.square(m1 - m2)))

if __name__ == '__main__':
    referenceFileName = str(input()).rstrip()
    degradedFileName = str(input()).rstrip()
    method = int(input())
    gamma = float(input())
    size = int(input())

    func = denoising if method == 1 else deblurring

    reference = imio.imread(referenceFileName)
    degraded = imio.imread(degradedFileName)

    degraded = normalize(func(degraded, size, gamma), np.amax(reference))
    print(round(rmsd(degraded, reference), 4))  # rounded to show only 4 decimal places

    newImg = Image.fromarray(degraded) # uncomment to see the images
    newImg.show() # uncomment to see the images