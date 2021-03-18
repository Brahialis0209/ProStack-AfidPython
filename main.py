from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import regionprops
from skimage.color import rgb2gray
from skimage.io import imsave

from afIdentifier import afid_identifier


def main():
    ch1 = io.imread('1-1.tif')
    ch2 = io.imread('1-2.tif')
    mask = io.imread('mask.tif')
    mask_gray = rgb2gray(mask)
    # plt.imshow(mask_gray,cmap=plt.cm.gray)
    # plt.savefig('mask_in.png')


    maskAF, im1AFRemoved, im2AFRemoved, kBest = afid_identifier(ch1, ch2, mask_gray, kAuto=1, k=20)
    imsave("maskAF_tif_format.tif", maskAF)
    imsave("im1AFRemoved.tif", im1AFRemoved)
    imsave("im2AFRemoved.tif", im2AFRemoved)






if __name__ == "__main__":
    main()