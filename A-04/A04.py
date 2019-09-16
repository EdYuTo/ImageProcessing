# Edson Yudi Toma - 9791305
# SCC0251 Image Processing - Prof. Moacir Ponti
# 1st Semester of 2019
# Assignment 4 : Colour image processing and segmentation
import numpy as np
import imageio as imio
import random
from PIL import Image # uncomment to see the images

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
    img = np.uint8((img - min) * (normal / dif))
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

'''
getValue(dataset, x, y, opt):
    Each option requires a different output, this is an auxiliary function used to
    return the date with the correct format.
    Args:
        * dataset - a 3d array containing the image data
        * x - x index of the dataset
        * y - y index of the dataset
        * opt - the given option to format the output:
            * 1 for an array with len = 3 (r, g, b)
            * 2 for an array with len = 5 (r, g, b, x, y)
            * 3 for an array with len = 1 (grayscale)
            * 4 for an array with len = 3 (grayscale, x, y)
    Return value:
        * an array formatted for the given option, with the dataset data
'''
def getValue(dataset, x, y, opt):
    value = 0 # default value
    if opt == 1:
        value = [dataset[x, y, 0], dataset[x, y, 1], dataset[x, y, 2]]
    elif opt == 2:
        value = [dataset[x, y, 0], dataset[x, y, 1], dataset[x, y, 2], x, y]
    elif opt == 3:
        value = [0.299*dataset[x, y, 0] + 0.587*dataset[x, y, 1] + 0.114*dataset[x, y, 2]]
    elif opt == 4:
        value = [0.299*dataset[x, y, 0] + 0.587*dataset[x, y, 1] + 0.114*dataset[x, y, 2], x, y]
    return value

'''
closestCentroid(centroids, value):
    Auxiliary function that retrieves the closest centroid to a given value
    Args:
        * centroids - array with all the centroids values
        * value - value to be compared with the centroids
    Return value:
        * the index of this centroid
'''
def closestCentroid(centroids, value):
    dif = np.inf
    index = 0
    for pos, cent in enumerate(centroids):
        newDif = np.linalg.norm(cent-value)
        if newDif < dif:
            dif = newDif
            index = pos
    return index

'''
kMeans(dataset, opt, klusters, niterations, seed):
    Computes the kMeans and return an image with the labels inside
    Args:
        * dataset - the image
        * opt - the option to format each value and process it
        * klusters - the k number of clusters to be generated
        * niterations - the number of times the kMeans will run before returning a result
        * seed - seed to initialize the random instance
    Return value:
        * an image with each label from each centroid inside
'''
def kMeans(dataset, opt, klusters, niterations, seed):
    try:
        height, width = dataset.shape # get a grayscale image
    except:
        height, width, depth = dataset.shape # get rgb image

    random.seed(seed) # initialize the random instance
    dataset = dataset.astype(np.float64) # convert the dataset to float for more precision
    ids = np.sort(random.sample(range(0, height*width), klusters)) # get the flatten index to the dataset to become centroids
    # convert the flatten id to a 2d id, and retrieve the value from this position on the dataset:
    centroids = np.array([getValue(dataset, i//width, i%width, opt) for i in ids]) 
    
    imageCentroids = np.zeros((height, width)) # initialize a new image (the centroids indexes image)
    
    for i in range(niterations):
        print(i)
        auxCentroids = np.copy(centroids) # cumulative centroids array
        idxCounter = np.zeros(len(centroids), dtype=int) # cumulative index array
        # both auxCentroids and idxCounter will be used to generate the new centroids array
        idxCounter += 1
        for x in range(height):
            for y in range(width):
                # gets the closest centroid value and its index
                closest = closestCentroid(centroids, getValue(dataset, x, y, opt))
                auxCentroids[closest] += getValue(dataset, x, y, opt) # adds it to the cumulative array
                idxCounter[closest] += 1 # also increment the number of centroids for the closest position
                imageCentroids[x][y] = closest # set the index to the image
        centroids = auxCentroids / idxCounter[:, None] # calculates the mean of each centroid

    # the label values will be between 0 and klusters, therefore it is necessary to normalize it:
    return normalize(imageCentroids)

if __name__ == '__main__':
    inputFileName = str(input()).rstrip()
    referenceFileName = str(input()).rstrip()
    option = int(input())
    klusters = int(input())
    niterations = int(input())
    seed = int(input())

    inputImage = imio.imread(inputFileName)
    #referenceImage = np.load(referenceFileName)

    outputImage = kMeans(inputImage, option, klusters, niterations, seed)
    #print(round(rmsd(outputImage, referenceImage), 4))  # rounded to show only 4 decimal places

    newImg = Image.fromarray(outputImage) # uncomment to see the images
    newImg.show() # uncomment to see the images