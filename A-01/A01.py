# Edson Yudi Toma - 9791305
# SCC0251 Image Processing - Prof. Moacir Ponti
# 1st Semester of 2019
# Assignment 1 : Image generation
import numpy as np
import random
from PIL import Image
'''
normalize(matrix, normal):
    Normalizes a given 2d array with the given normal factor
    Args:
        * matrix - 2 dimentional array to be normalized
        * normal - max value to be used as reference, the min value will be set as 0
    Return value:
        * normalized 2d array
'''
def normalize(matrix, normal = 65535):
    max = np.amax(matrix)
    min = np.amin(matrix)
    dif = max - min
    matrix = ((matrix - min) * normal) / dif
    return matrix

'''
downsampling(matrix, newSize, bpp):
    Reduces the pixel ratio of the image to match the given newSize
    Args:
        * matrix - 2 dimentional array to be normalized
        * newSize - the new matrix row/col size
        * bpp - number of bits per pixel (will be used to set how many bits each pixel has - number of colours)
    Return value:
        * 2d array with newSize rows and newSize cols
'''
def downsampling(matrix, newSize, bpp):
    ratio = int(len(matrix) / newSize)
    shiftRate = 16-bpp
    newMatrix = np.zeros((newSize, newSize), dtype=int)
    for x in range(0, newSize*ratio, ratio):
        for y in range(0, newSize*ratio, ratio):
            newMatrix[int(x/ratio)][int(y/ratio)] = (int(matrix[x][y]) >> shiftRate) # only using integer values between 0 and 255
    return newMatrix

'''
Functions 1 to 5 are specified in the assignment description.
'''
def function1(lSizeC):
    C = np.zeros((lSizeC, lSizeC), dtype=float)
    for x in range(lSizeC):
        for y in range(lSizeC):
            C[x][y] = ((x*y + 2*y))
    return normalize(C)

def function2(lSizeC, paramQ):
    C = np.zeros((lSizeC, lSizeC), dtype=float)
    for x in range(lSizeC):
        for y in range(lSizeC):
            C[x][y] = (np.abs(np.cos(x/paramQ) + 2*np.sin(y/paramQ)))
    return normalize(C)
    
def function3(lSizeC, paramQ):
    C = np.zeros((lSizeC, lSizeC), dtype=float)
    for x in range(lSizeC):
        for y in range(lSizeC):
            C[x][y] = (np.abs(3*(x/paramQ)-np.cbrt(y/paramQ)))
    return normalize(C)

def function4(lSizeC, seed):
    C = np.zeros((lSizeC, lSizeC), dtype=float)
    random.seed(seed)
    for x in range(lSizeC):
        for y in range(lSizeC):
            C[x][y] = (random.uniform(0, 1))
    return normalize(C)
    
def function5(lSizeC, seed): # This function isn't working properly
    C = np.zeros((lSizeC, lSizeC), dtype=int)
    random.seed(seed)
    x = 0
    y = 0
    C[x][y] = 1
    for i in range(lSizeC * int(lSizeC/2) + 1):
        x = (x + random.randint(-1, 1)) % lSizeC
        y = (y + random.randint(-1, 1)) % lSizeC
        C[x][y] = 1
    return normalize(C)
    
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
    return np.sqrt(np.sum(np.square(m1 - m2)))

if __name__ == '__main__':
    filename = str(input()).rstrip()
    lSizeC = int(input())   
    func = int(input())
    paramQ = int(input())
    lSizeN = int(input())
    bpp= int(input())
    seed = int(input())

    R = np.load(filename)

    if func == 1:
        arr = (downsampling(function1(lSizeC), lSizeN, bpp))
    elif func == 2:
        arr = (downsampling(function2(lSizeC, paramQ), lSizeN, bpp))
    elif func == 3:
        arr = (downsampling(function3(lSizeC, paramQ), lSizeN, bpp))
    elif func == 4:
        arr = (downsampling(function4(lSizeC, seed), lSizeN, bpp))
    elif func == 5:
        arr = (downsampling(function5(lSizeC, seed), lSizeN, bpp))
    else:
        arr = []

    print(round(rmsd(R, arr), 4))  # rounded to show only 4 decimal places

    '''
    Debugging print
    '''
    arr = normalize(arr, 255)
    img = Image.fromarray(arr)
    img.show()
