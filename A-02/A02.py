# Edson Yudi Toma - 9791305
# SCC0251 Image Processing - Prof. Moacir Ponti
# 1st Semester of 2019
# Assignment 2 : Image Enhancement and Filtering
import numpy as np
import imageio as imio
#from PIL import Image # uncomment to see the images

'''
    given a 2D array {a} and an index {indx} this function duplicates {num_dups} times the row in {indx}
    if {zeros}=True the inserted rows will be filled with zeroes
    the vertical size of the array is increased by {num_dups}
    auxiliar function to be used in the (filtering_2D) and (median_filter_2D) methods
'''
def dup_rows(a, indx, num_dups=1, zeros=False):
    if not zeros:
        return np.insert(a,[indx+1]*num_dups,a[indx],axis=0)
    else:
        return np.insert(a,[indx+1]*num_dups,np.zeros((1, len(a[0]))),axis=0)

'''
    given a 2D array {a} and an index {indx} this function duplicates {num_dups} times the col in {indx}
    if {zeros}=True the inserted cols will be filled with zeroes
    the horizontal size of the array is increased by {num_dups}
    auxiliar function to be used in the (filtering_2D) and (median_filter_2D) methods
'''
def dup_cols(a, indx, num_dups=1, zeros=False):
    if not zeros:
        return np.insert(a,[indx+1]*num_dups,a[:,[indx]],axis=1)
    else:
        return np.insert(a,[indx+1]*num_dups,np.zeros((len(a), 1)),axis=1)

def limiarization(img):
    T0 = int(input()) # initial threshold
    Ti = T0+5 # temporary value to initiate loop

    while np.absolute(Ti - T0) > 0.5:
        G1 = np.where(img > T0) # saves all the indexes of values that match the requirement (value > T)
        G2 = np.where(img <= T0) # saves all the indexes of values that match the requirement (value <= T)
        Ti = T0 # Ti-1 -> old value
        T0 = 1/2 * (np.mean(img[G1]) + np.mean(img[G2])) # T1 -> new value
        
    newImg = img.copy() # creation of another image
    G1 = np.where(newImg > T0) # saves all the indexes of values that match the requirement (value > T)
    G2 = np.where(newImg <= T0) # saves all the indexes of values that match the requirement (value <= T)
    newImg[G1] = 1 # for each of the indexes in {G1} set the value 1
    newImg[G2] = 0 # for each of the indexes in {G2} set the value 0

    return newImg

def filtering_1D(img):
    sizeN = int(input())
    w = np.asarray(list(map(int, input().split())))

    newImg = img.copy() # creation of another image
    newImg = np.reshape(newImg, -1, order="C") # reshape the image into a single one dimentional array
    span = int((sizeN-1)/2) # number of values before and after the center of the filter

    # the next step is used to simulate a wraped array
    tempImg = np.concatenate((newImg[span*-1:], newImg)) # adds the last values to the begining
    tempImg = np.concatenate((tempImg, newImg[:span])) # adds the first values to the end

    # convolution:
    # notice that now the {tempImage} has additional len(w)-1 entries
    # and because of that whe can index it between i and i+len(w)
    # here is an example:
    # w = [ 0, 1, 3 ]
    #        img = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
    # tempImg = [ 9[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 0 ]
    #             |  |  |
    #           [ 0, 1, 3 ]
    # when positioning the index on the first value of the {img} there is an additional value to the left,
    # the same thing happens on the end of the {img}
    for i in range(len(newImg)):
        newImg[i] = np.sum(np.asarray(tempImg[i:i+sizeN]) * w)
    
    return np.reshape(newImg, (len(img), len(img[0])), order="C") # reshape the image into a 2 dimentional array


def filtering_2D(img):
    sizeN = int(input()) # dimentions of the filter
    w = []
    for i in range(0, sizeN):
        w.append(list(map(int, input().split())))
    w = np.asarray(w) # it's important to make this conversion, otherwise the filter multiplication won't work
    
    span = int((sizeN-1)/2) # number of rows and cols that will be ignored
    newImg = np.zeros((len(img)-span*2, len(img[0])-span*2)) # new image without the {span} first and last rows and cols

    # convolution
    # the idea here is similiar to the (filtering_1D) convolution but instead of the extra values, we'll use missing values make the process easier
    # let's see with this example:
    # img = [0, 1, 2, 3, 4, 5]    w = [1, -1, 1]    span = 1    newImg = [2, 3, 4, 5]    the first and last rows were removed
    #       [1, 2, 3, 4, 5, 6]        [1, -8, 1]                         [3, 4, 5, 6]    so did the first and last cols
    #       [2, 3, 4, 5, 6, 7]        [1, -1, 1]                         [4, 5, 6, 7]
    #       [3, 4, 5, 6, 7, 8]                                           [5, 6, 7, 8]
    #       [4, 5, 6, 7, 8, 9]
    #       [5, 6, 7, 8, 9, 10]
    # now pay attention to the relative positions of the {newImg} and {img}:
    # img = [0,  1, 2,  ...]    newImg = [(2, 3), ...]   
    #       [1, (2, 3), ...]             [(3, 4), ...]    
    #       [2, (3, 4), ...]             [.,  .,  ...]
    #       [.,  ., .,  ...] 
    # by creating an image without the correct amount of rows and cols we are able to make a subarray that is easier to work with
    # note that now the index 0 of the {newImg} can easily match the values with {w} and {img}
    # [1, -1, 1]         [1, -8, 1]          [1, -1, 1]
    #  |   |  |           |   |  |            |   |  |           now we can simply multiply img[idx:span*2+1][idx:span*2+1] by {w}
    # [0, 1,  2,  ...]   [1, (2, 3), ...]    [2, (3, 4), ...]    note that we use {span}*2+1 to match the original {w} size
    for i in range(len(newImg)):
        for j in range(len(newImg[0])): # this is used to treat the case when {newImg} isn't square
            newImg[i, j] = np.sum(np.asarray(img[i:i+sizeN, j:j+sizeN]) * w)

    newImg = dup_rows(newImg, 0, span) # adds {span} times the first row
    newImg = dup_rows(newImg, len(newImg)-1, span) # adds {span} times the last row
    newImg = dup_cols(newImg, 0, span) # adds {span} times the first col
    newImg = dup_cols(newImg, len(newImg[0])-1, span) # adds {span} times the last col
    
    return limiarization(newImg)

def median_filter_2D(img):
    size = int(input())
    
    span = int((size-1)/2) # number of rows and cols that will be ignored
    newImg = np.copy(img)
    
    newImg = dup_rows(newImg, -1, span, True) # adds {span} times a blank row before the first row
    newImg = dup_rows(newImg, len(newImg)-1, span, True) # adds {span} times a blank row after the last row
    newImg = dup_cols(newImg, -1, span, True) # adds {span} times a blank col before the first col
    newImg = dup_cols(newImg, len(newImg[0])-1, span, True) # adds {span} times a blank col after the last col

    # the following loop has the same principle as the one in (filtering_2D)
    for i in range(span, len(newImg)-span):
        for j in range(span, len(newImg[0])-span): # this is used to treat the case when {newImg} isn't square
            newImg[i, j] = np.median(newImg[i-span:i+span+1, j-span:j+span+1]) # finds the median of the array
    
    newImg = np.delete(newImg, slice(0, span), axis=0) # deletes the first {span} rows
    newImg = np.delete(newImg, slice(len(newImg)-span, len(newImg)), axis=0) # deletes the last {span} rows
    newImg = np.delete(newImg, slice(0, span), axis=1) # deletes the first {span} cols
    newImg = np.delete(newImg, slice(len(newImg[0])-span, len(newImg[0])), axis=1) # deletes the last {span} cols

    return newImg

def switch_case(arg, img):
    switch = {
        1: limiarization,
        2: filtering_1D,
        3: filtering_2D,
        4: median_filter_2D,
    }
    func = switch.get(arg, lambda: "nothing")
    if (arg != 4):
        return normalize(func(normalize(img)))
    else:
        return normalize(func(img))

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
    filename = str(input()).rstrip()
    method = int(input())

    img = imio.imread(filename)

    newImg = switch_case(method, img)
    print(round(rmsd(img, newImg), 4))  # rounded to show only 4 decimal places

    #newImg = Image.fromarray(newImg) # uncomment to see the images
    #newImg.show() # uncomment to see the images