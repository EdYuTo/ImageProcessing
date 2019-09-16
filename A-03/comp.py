import numpy as np
from PIL import Image

if __name__ == '__main__':
    filename = str(input()).rstrip()
    img = (Image.open(filename)).convert('L')
    img.save("Original.jpg")