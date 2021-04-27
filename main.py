from skimage import data
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import regionprops
from skimage.color import rgb2gray
from skimage.io import imsave
import cv2
from afIdentifier import afid_identifier
from math import ceil
import imageio

from glowIdentifier import glow_identifier


def delete_back_noise(ch):
    plt.imshow(ch, cmap=plt.cm.gray)
    mean_value = np.mean(ch)
    std_value = np.std(ch)
    cf_value = 0.6
    deductible = mean_value + cf_value * std_value
    m = ch.shape[0]
    n = ch.shape[1]
    for i in range(m):
        for j in range(n):
            pixel = ch[i, j]
            if pixel >= deductible:
                ch[i, j] = ch[i, j] - deductible
            else:
                ch[i, j] = 0
    plt.imshow(ch, cmap=plt.cm.gray)
    return ch



def generate_mask(ch1, ch2, sigma):
    k = 2 * ceil(2 * sigma) + 1
    # k = 3
    ch1Blurred = cv2.GaussianBlur(ch1, ksize=(k, k), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    ch2Blurred = cv2.GaussianBlur(ch2, ksize=(k, k), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    ch1Blurred = delete_back_noise(ch1Blurred)
    ch2Blurred = delete_back_noise(ch2Blurred)
    ret1, th1 = cv2.threshold(ch1Blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret2, th2 = cv2.threshold(ch2Blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_res = th1 & th2
    return th_res

def main():
    examples_path = 'example/'
    m4_path = 'm4/'
    m7_path = 'm7/'

    # Example
    # ch1 = io.imread(examples_path + '1-1.tif')
    # ch2 = io.imread(examples_path + '1-2.tif')
    # mask = io.imread(examples_path + 'mask.tif')
    # mask_gray = rgb2gray(mask)
    # maskAF, im1AFRemoved, im2AFRemoved, kBest = afid_identifier(ch1, ch2, mask, kAuto=1, k=20, min_area=20, corr=0.6)
    # imsave(examples_path + "mask_example_AF.tif", maskAF)
    # imsave(examples_path + "im1_af_rem_examp_AF.tif", im1AFRemoved)
    # imsave(examples_path + "im2_af_rem_examp_AF.tif", im2AFRemoved)
    # # maskAF = io.imread(examples_path + 'mask_example_AF.tif')
    # glow1, glow2, im1_glow_removed, im2_glow_removed = glow_identifier(ch1, ch2, maskAF, trace_sensitivity=20, sigma=2)
    # imsave(examples_path + "glow1_af_example.png", glow1)
    # imsave(examples_path + "glow2_af_example.png", glow2)
    # imsave(examples_path + "im1_glow_rem_examp.tif", im1_glow_removed)
    # imsave(examples_path + "im2_glow_rem_examp.tif", im2_glow_removed)


    # plt.imshow(mask_gray,cmap=plt.cm.gray)
    # ret, images_0_ch = cv2.imreadmulti('M4-C0.tif')
    # ret, images_1_ch = cv2.imreadmulti('M4-C1.tif')
    # imsave("m4_0_58.tif", images_0_ch[58])
    # imsave("m4_1_58.tif", images_1_ch[58])

    # m7
    # ch1 = io.imread(m7_path + 'm7_0_49.tif')
    # ch2 = io.imread(m7_path + 'm7_1_49.tif')
    # # Generate mask
    # sigma = 1
    # mask = generate_mask(ch1, ch2, sigma)
    # imsave(m7_path + "mask_gen_m7_049.tif", mask)
    # maskAF, im1AFRemoved, im2AFRemoved, kBest = afid_identifier(ch1, ch2, mask, kAuto=0, k=1, min_area=7, corr=0.3)
    # imsave(m7_path + "mask_m7_AF_49.tif", maskAF)
    # imsave(m7_path + "im1_af_rem_m7_49.tif", im1AFRemoved)
    # imsave(m7_path + "im2_af_rem_m7_49.tif", im2AFRemoved)
    # glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1, ch2, maskAF,
    #                                                                                                      trace_sensitivity=20, sigma=2)
    # imsave(m7_path + "glow1_m7_49.png", glow1)
    # imsave(m7_path + "glow2_m7_49.png", glow2)
    # imsave(m7_path + "im1_glow_only_m7_49.tif", im1_glow_removed)
    # imsave(m7_path + "im2_glow_only_m7_49.tif", im2_glow_removed)
    # imsave(m7_path + "im1_res_removed_m7_49.tif", im1_res_removed)
    # imsave(m7_path + "im2_res_removed_m7_49.tif", im2_res_removed)


    # m4
    ch1 = io.imread(m4_path + 'm4_0_49.tif')
    ch2 = io.imread(m4_path + 'm4_1_49.tif')
    # Generate mask
    sigma = 1
    mask = generate_mask(ch1, ch2, sigma)
    imsave(m4_path + "mask_gen_m4_049.tif", mask)
    maskAF, im1AFRemoved, im2AFRemoved, kBest = afid_identifier(ch1, ch2, mask, kAuto=1, k=20, min_area=7, corr=0.6)
    imsave(m4_path + "mask_m4_AF_49.tif", maskAF)
    imsave(m4_path + "im1_af_rem_m4_49.tif", im1AFRemoved)
    imsave(m4_path + "im2_af_rem_m4_49.tif", im2AFRemoved)
    glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1, ch2, maskAF,
                                                                                                         trace_sensitivity=20, sigma=2)
    imsave(m4_path + "glow1_m4_49.png", glow1)
    imsave(m4_path + "glow2_m4_49.png", glow2)
    imsave(m4_path + "im1_glow_only_m4_49.tif", im1_glow_removed)
    imsave(m4_path + "im2_glow_only_m4_49.tif", im2_glow_removed)
    imsave(m4_path + "im1_res_removed_m4_49.tif", im1_res_removed)
    imsave(m4_path + "im2_res_removed_m4_49.tif", im2_res_removed)



    # ret, images_0_ch = cv2.imreadmulti('M4-C0.tif')
    # ret, images_1_ch = cv2.imreadmulti('M4-C1.tif')
    # results_af_mask = []
    # generate_masks = []
    # img_count = len(images_0_ch)
    # for index in range(img_count):
    #     print(index, end=' ')
    #     ch1 = images_0_ch[index]
    #     ch2 = images_1_ch[index]
    #     sigma = 1
    #     mask = generate_mask(ch1, ch2, sigma)
    #     generate_masks.append(cp.copy(mask))
    #     maskAF, im1AFRemoved, im2AFRemoved, kBest = afid_identifier(ch1, ch2, mask, kAuto=0, k=6, min_area=7, corr=0.2)
    #     results_af_mask.append(maskAF)
    #     # plt.imshow(maskAF, cmap=plt.cm.gray)
    # np_results = np.array(results_af_mask, np.uint8)
    # np_gen_masks = np.array(generate_masks, np.uint8)
    # imageio.mimwrite('img_parts_mask_af_m4_k6_new.tif', np_results)
    # imageio.mimwrite('gen_masks_m4_new.tif', np_gen_masks)




if __name__ == "__main__":
    main()