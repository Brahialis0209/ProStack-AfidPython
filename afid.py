import sys
from skimage.filters.rank import core_cy_3d
import numpy as np
import cv2
from afIdentifier import afid_identifier
from math import ceil
import imageio
import argparse
from skimage import io
from skimage.measure import regionprops
from skimage.color import rgb2gray
from skimage.io import imsave



from glowIdentifier import glow_identifier


def delete_back_noise(ch, noise_delete_coef):
    # plt.imshow(ch, cmap=plt.cm.gray)
    mean_value = np.mean(ch)
    std_value = np.std(ch)
    cf_value = noise_delete_coef
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
    # plt.imshow(ch, cmap=plt.cm.gray)
    return ch



def generate_mask(ch1, ch2, sigma, noise_delete_coef):
    k = 2 * ceil(2 * sigma) + 1
    # k = 3
    ch1Blurred = cv2.GaussianBlur(ch1, ksize=(k, k), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    ch2Blurred = cv2.GaussianBlur(ch2, ksize=(k, k), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    ch1Blurred = delete_back_noise(ch1Blurred, noise_delete_coef)
    ch2Blurred = delete_back_noise(ch2Blurred, noise_delete_coef)
    ret1, th1 = cv2.threshold(ch1Blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret2, th2 = cv2.threshold(ch2Blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_res = th1 & th2
    return th_res


def crop_paths(paths):
    in_paths = paths[0].split(',')
    out_paths = paths[1].split(',')
    index = 0
    new_paths = []
    new_paths.extend(in_paths)
    new_paths.extend(out_paths)
    # for path in new_paths:
    #     buf_path = path
    #     new_paths[index] = buf_path[1:-1]
    #     index += 1
    return new_paths


def main():
    examples_path = 'example/'
    # m4_path = 'm4/'
    # m4_15_path = 'm4/15/'
    m7_path = 'm7/'
    # m4_auto = 'parts_auto/'
    # m4_corr = 'parts_corr/'
    # m7_k6 = 'k6_parts/'
    # m5_path = 'm5/'
    # m5_k6 = 'parts_k6/'
    # m6_path = 'm6/'
    # m6_k6 = 'parts_k6/'
    # m1_path = 'm1/'
    # m1_k6 = 'parts_k6/'

    # Example
    # ch1 = io.imread('example/' + '1-1.tif')
    # ch2 = io.imread('example/' + '1-2.tif')
    # mask = io.imread(examples_path + 'mask.tif')
    # sigma = 1
    # # mask = generate_mask(ch1, ch2, 1, 0.8)
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





    f = open('log_af.txt', 'w')
    f.write('path1 = ' + sys.argv[-2] + '\npath2 = ' + sys.argv[-1] + '\n')
    print('path1 = ', sys.argv[-2], '\npath2 = ', sys.argv[-1])


    parser = argparse.ArgumentParser(description='AFid parametrs')
    parser.add_argument("--noise_del_cf", default=0.7, type=float, help="Factor to remove the background before applying AFid.")
    parser.add_argument("--kAuto", default=0, type=int, help="This is a flag about whether the algorithm should"
                                                             " choose the optimal number of clusters."
                                                             " (0 if not needed, 1 otherwise)")
    parser.add_argument("--k", default=6, type=int, help="The number of clusters for determining autofluorescence."
                                                         " (1 if you need a simple AFid option,"
                                                         " more if you use clustering)")
    parser.add_argument("--min_area", default=7, type=int, help="The minimum area of ​​a region that can be"
                                                                " autofluorescent.")
    parser.add_argument("--corr", default=0.3, type=float, help="The minimum area of ​​a region that can be"
                                                                "autofluorescent.")
    parser.add_argument("--trace_sensitivity", default=20, type=int, help="The number of pixels between extensions.")

    parser.add_argument('paths', type=str, nargs='+',
                        help='Input and output file paths')

    args = parser.parse_args()
    noise_del_cf = args.noise_del_cf
    k = args.k
    kAuto = args.kAuto
    trace_sensitivity = args.trace_sensitivity
    corr = args.corr
    paths = args.paths
    min_area = args.min_area
    print(args)
    paths = crop_paths(paths)
    input_path_0 = paths[0]
    input_path_1 = paths[1]
    output_path_0 = paths[2]
    output_path_1 = paths[3]
    print(input_path_0, '\n', input_path_1, '\n',output_path_0, '\n',output_path_1, '\n',)
    f.write(input_path_0 + '\n' +  input_path_1 + '\n' + output_path_0 + '\n' + output_path_1 + '\n')





    print('GG')
    # m4 prost
    ret, images_0_ch = cv2.imreadmulti(input_path_0)
    ret, images_1_ch = cv2.imreadmulti(input_path_1)
    print('readmulti ready')
    f.write('readmulti ready')
    # results_af_mask = []
    img_count = len(images_0_ch)
    ch1_result = []
    ch2_result = []
    exception_count = 0
    for index in range(img_count):
        print(index, end=' ')
        f.write(str(index) + '\n')
        ch1 = images_0_ch[index]
        ch2 = images_1_ch[index]
        sigma = 1
        mask = generate_mask(ch1, ch2, sigma, noise_del_cf)
        try:
            maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=kAuto, k=k, min_area=min_area, corr=corr)
            # results_af_mask.append(maskAF)
            glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1, ch2, maskAF,
                                                                                                                 trace_sensitivity=trace_sensitivity, sigma=2)
        except Exception:
            print("EXCEPTION!!\n")
            f.write("EXCEPTION!!\n")
            exception_count += 1
            maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=0, k=1, min_area=7,
                                                                         corr=0.3)
            # results_af_mask.append(maskAF)
            glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1,
                                                                                                                 ch2,
                                                                                                                 maskAF,
                                                                                                                 trace_sensitivity=20,
                                                                                                                 sigma=2)
        ch1_result.append(im1_res_removed)
        ch2_result.append(im2_res_removed)
    # np_results_ms = np.array(results_af_mask, np.uint8)
    np_results_ch1 = np.array(ch1_result, np.uint8)
    np_results_ch2 = np.array(ch2_result, np.uint8)
    # imageio.mimwrite(output_path, np_results_ms)
    imageio.mimwrite(output_path_0, np_results_ch1)
    imageio.mimwrite(output_path_1, np_results_ch2)














    # # M7 49
    # ch1 = io.imread(m7_path + '49/' + 'm7_0_49.tif')
    # ch2 = io.imread(m7_path + '49/' + 'm7_1_49.tif')
    # path_49 = '49/or/'
    # Generate mask
    # sigma = 1
    # noise_delete_coef = 0.8
    # mask = generate_mask(ch1, ch2, sigma, noise_delete_coef)
    # imsave(m7_path + path_49 + "mask_gen_m7_049.tif", mask)
    # maskAF, im1AFRemoved, im2AFRemoved, kBest = afid_identifier(ch1, ch2, mask, kAuto=0, k=1, min_area=7, corr=0.1)
    # imsave(m7_path + path_49 + "mask_example_AF.tif", maskAF)
    # imsave(m7_path + path_49 + "im1_af_rem_AF.tif", im1AFRemoved)
    # imsave(m7_path + path_49 + "im2_af_rem_AF.tif", im2AFRemoved)
    # glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1, ch2, maskAF,
    #                                                                                                      trace_sensitivity=20, sigma=2)
    # imsave(m7_path + path_49 + "glow1_af.png", glow1)
    # imsave(m7_path + path_49 + "glow2_af.png", glow2)
    # imsave(m7_path + path_49 + "im1_glow_rem.tif", im1_glow_removed)
    # imsave(m7_path + path_49 + "im2_glow_rem.tif", im2_glow_removed)
    # imsave(m7_path + path_49 + "im1_res_removed_m7_49.tif", im1_res_removed)
    # imsave(m7_path + path_49 + "im2_res_removed_m7_49.tif", im2_res_removed)








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
    # ret, images_0_ch = cv2.imreadmulti('m4/' + 'M4-C0.tif')
    # ret, images_1_ch = cv2.imreadmulti('m4/' + 'M4-C1.tif')
    # ch1 = images_0_ch[56]
    # ch2 = images_1_ch[56]
    # # Generate mask
    # sigma = 1
    # mask = generate_mask(ch1, ch2, sigma)
    # # imsave(m4_15_path + "mask_gen_m4_15.tif", mask)
    # maskAF, im1AFRemoved, im2AFRemoved, kBest = afid_identifier(ch1, ch2, mask, kAuto=0, k=1, min_area=7, corr=0.3)
    # imsave(m4_15_path + "mask_m4_AF_15.tif", maskAF)
    # imsave(m4_15_path + "im1_af_rem_m4_15.tif", im1AFRemoved)
    # imsave(m4_15_path + "im2_af_rem_m4_15.tif", im2AFRemoved)
    # glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1, ch2, maskAF,
    #                                                                                                      trace_sensitivity=20, sigma=2)
    # imsave(m4_15_path + "glow1_m4_15.png", glow1)
    # imsave(m4_15_path + "glow2_m4_15.png", glow2)
    # imsave(m4_15_path + "im1_glow_only_m4_15.tif", im1_glow_removed)
    # imsave(m4_15_path + "im2_glow_only_m4_15.tif", im2_glow_removed)
    # imsave(m4_15_path + "im1_res_removed_m4_15.tif", im1_res_removed)
    # imsave(m4_15_path + "im2_res_removed_m4_15.tif", im2_res_removed)



    # m4 NULTI
    # ret, images_0_ch = cv2.imreadmulti(m4_path + 'M4-C0.tif')
    # ret, images_1_ch = cv2.imreadmulti(m4_path + 'M4-C1.tif')
    # results_af_mask = []
    # generate_masks = []
    # img_count = len(images_0_ch)
    # ch1_result = []
    # ch2_result = []
    # exception_count = 0
    # for index in range(img_count):
    #     print(index, end=' ')
    #     ch1 = images_0_ch[index]
    #     ch2 = images_1_ch[index]
    #     sigma = 1
    #     mask = generate_mask(ch1, ch2, sigma)
    #     # generate_masks.append(cp.copy(mask))
    #     try:
    #         maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=0, k=1, min_area=7, corr=0.3)
    #         results_af_mask.append(maskAF)
    #         glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1, ch2, maskAF,
    #                                                                                                              trace_sensitivity=20, sigma=2)
    #     except Exception:
    #         print("EXCEPTION!!\n")
    #         exception_count += 1
    #         maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=0, k=6, min_area=7,
    #                                                                      corr=0.6)
    #         results_af_mask.append(maskAF)
    #         glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1,
    #                                                                                                              ch2,
    #                                                                                                              maskAF,
    #                                                                                                              trace_sensitivity=20,
    #                                                                                                              sigma=2)
    #     ch1_result.append(im1_res_removed)
    #     ch2_result.append(im2_res_removed)
    #     # plt.imshow(maskAF, cmap=plt.cm.gray)
    # print("exception_count = ", exception_count)
    # np_results_ms = np.array(results_af_mask, np.uint8)
    # np_results_ch1 = np.array(ch1_result, np.uint8)
    # np_results_ch2 = np.array(ch2_result, np.uint8)
    # # np_gen_masks = np.array(generate_masks, np.uint8)
    # imageio.mimwrite(m4_path + m4_corr + 'imgs_mask_af_m4_k6_corr_0.3.tif', np_results_ms)
    # imageio.mimwrite(m4_path + m4_corr + 'imgs_res_ch1_m4_k6_corr_0.3.tif', np_results_ch1)
    # imageio.mimwrite(m4_path + m4_corr + 'imgs_res_ch2_m4_k6_corr_0.3.tif', np_results_ch2)
    # # imageio.mimwrite('gen_masks_m4_new.tif', np_gen_masks)


    # # M7 multi
    # ret, images_0_ch = cv2.imreadmulti(m7_path + 'm7_0.tif')
    # ret, images_1_ch = cv2.imreadmulti(m7_path + 'm7_1.tif')
    # results_af_mask = []
    # generate_masks = []
    # img_count = len(images_0_ch)
    # ch1_result = []
    # ch2_result = []
    # exception_count = 0
    # for index in range(img_count):
    #     print(index, end=' ')
    #     ch1 = images_0_ch[index]
    #     ch2 = images_1_ch[index]
    #     sigma = 1
    #     mask = generate_mask(ch1, ch2, sigma)
    #     # generate_masks.append(cp.copy(mask))
    #     try:
    #         maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=0, k=6, min_area=7, corr=0.3)
    #         results_af_mask.append(maskAF)
    #         glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1, ch2, maskAF,
    #                                                                                                              trace_sensitivity=20, sigma=2)
    #     except Exception:
    #         print("EXCEPTION!!\n")
    #         exception_count += 1
    #         maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=0, k=1, min_area=7,
    #                                                                      corr=0.4)
    #         results_af_mask.append(maskAF)
    #         glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1,
    #                                                                                                              ch2,
    #                                                                                                              maskAF,
    #                                                                                                              trace_sensitivity=20,
    #                                                                                                              sigma=2)
    #     ch1_result.append(im1_res_removed)
    #     ch2_result.append(im2_res_removed)
    #     # plt.imshow(maskAF, cmap=plt.cm.gray)
    # print("exception_count = ", exception_count)
    # np_results_ms = np.array(results_af_mask, np.uint8)
    # np_results_ch1 = np.array(ch1_result, np.uint8)
    # np_results_ch2 = np.array(ch2_result, np.uint8)
    # # np_gen_masks = np.array(generate_masks, np.uint8)
    # imageio.mimwrite(m7_path + m7_k6 + 'imgs_mask_af_m7_k6.tif', np_results_ms)
    # imageio.mimwrite(m7_path + m7_k6 + 'imgs_res_ch1_m7_k6.tif', np_results_ch1)
    # imageio.mimwrite(m7_path + m7_k6 + 'imgs_res_ch2_m7_k6.tif', np_results_ch2)



    # # M5 multi
    # ret, images_0_ch = cv2.imreadmulti(m5_path + 'M5-C0.tif')
    # ret, images_1_ch = cv2.imreadmulti(m5_path + 'M5-C1.tif')
    # results_af_mask = []
    # generate_masks = []
    # img_count = len(images_0_ch)
    # ch1_result = []
    # ch2_result = []
    # exception_count = 0
    # for index in range(img_count):
    #     print(index, end=' ')
    #     ch1 = images_0_ch[index]
    #     ch2 = images_1_ch[index]
    #     sigma = 1
    #     mask = generate_mask(ch1, ch2, sigma)
    #     # generate_masks.append(cp.copy(mask))
    #     try:
    #         maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=0, k=6, min_area=7, corr=0.3)
    #         results_af_mask.append(maskAF)
    #         glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1, ch2, maskAF,
    #                                                                                                              trace_sensitivity=20, sigma=2)
    #     except Exception:
    #         print("EXCEPTION!!\n")
    #         exception_count += 1
    #         maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=0, k=1, min_area=7,
    #                                                                      corr=0.4)
    #         results_af_mask.append(maskAF)
    #         glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1,
    #                                                                                                              ch2,
    #                                                                                                              maskAF,
    #                                                                                                              trace_sensitivity=20,
    #                                                                                                              sigma=2)
    #     ch1_result.append(im1_res_removed)
    #     ch2_result.append(im2_res_removed)
    #     # plt.imshow(maskAF, cmap=plt.cm.gray)
    # print("exception_count = ", exception_count)
    # np_results_ms = np.array(results_af_mask, np.uint8)
    # np_results_ch1 = np.array(ch1_result, np.uint8)
    # np_results_ch2 = np.array(ch2_result, np.uint8)
    # # np_gen_masks = np.array(generate_masks, np.uint8)
    # imageio.mimwrite(m5_path + m5_k6 + 'imgs_mask_af_m5_k6.tif', np_results_ms)
    # imageio.mimwrite(m5_path + m5_k6 + 'imgs_res_ch1_m5_k6.tif', np_results_ch1)
    # imageio.mimwrite(m5_path + m5_k6 + 'imgs_res_ch2_m5_k6.tif', np_results_ch2)




    # # M6 multi
    #     # ret, images_0_ch = cv2.imreadmulti(m6_path + 'M6-C0.tif')
    #     # ret, images_1_ch = cv2.imreadmulti(m6_path + 'M6-C1.tif')
    #     # results_af_mask = []
    #     # generate_masks = []
    #     # img_count = len(images_0_ch)
    #     # ch1_result = []
    #     # ch2_result = []
    #     # exception_count = 0
    #     # for index in range(img_count):
    #     #     print(index, end=' ')
    #     #     ch1 = images_0_ch[index]
    #     #     ch2 = images_1_ch[index]
    #     #     sigma = 1
    #     #     mask = generate_mask(ch1, ch2, sigma)
    #     #     # generate_masks.append(cp.copy(mask))
    #     #     try:
    #     #         maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=0, k=6, min_area=7, corr=0.3)
    #     #         results_af_mask.append(maskAF)
    #     #         glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1, ch2, maskAF,
    #     #                                                                                                              trace_sensitivity=20, sigma=2)
    #     #     except Exception:
    #     #         print("EXCEPTION!!\n")
    #     #         exception_count += 1
    #     #         maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=0, k=1, min_area=7,
    #     #                                                                      corr=0.4)
    #     #         results_af_mask.append(maskAF)
    #     #         glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1,
    #     #                                                                                                              ch2,
    #     #                                                                                                              maskAF,
    #     #                                                                                                              trace_sensitivity=20,
    #     #                                                                                                              sigma=2)
    #     #     ch1_result.append(im1_res_removed)
    #     #     ch2_result.append(im2_res_removed)
    #     #     # plt.imshow(maskAF, cmap=plt.cm.gray)
    #     # print("exception_count = ", exception_count)
    #     # np_results_ms = np.array(results_af_mask, np.uint8)
    #     # np_results_ch1 = np.array(ch1_result, np.uint8)
    #     # np_results_ch2 = np.array(ch2_result, np.uint8)
    #     # # np_gen_masks = np.array(generate_masks, np.uint8)
    #     # imageio.mimwrite(m6_path + m6_k6 + 'imgs_mask_af_m6_k6.tif', np_results_ms)
    #     # imageio.mimwrite(m6_path + m6_k6 + 'imgs_res_ch1_m6_k6.tif', np_results_ch1)
    #     # imageio.mimwrite(m6_path + m6_k6 + 'imgs_res_ch2_m6_k6.tif', np_results_ch2)





    # # M1 multi
    # ret, images_0_ch = cv2.imreadmulti(m1_path + 'M1-C0.tif')
    # ret, images_1_ch = cv2.imreadmulti(m1_path + 'M1-C1.tif')
    # results_af_mask = []
    # generate_masks = []
    # img_count = len(images_0_ch)
    # ch1_result = []
    # ch2_result = []
    # exception_count = 0
    # for index in range(img_count):
    #     print(index, end=' ')
    #     ch1 = images_0_ch[index]
    #     ch2 = images_1_ch[index]
    #     sigma = 1
    #     mask = generate_mask(ch1, ch2, sigma)
    #     # generate_masks.append(cp.copy(mask))
    #     try:
    #         maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=0, k=6, min_area=7, corr=0.3)
    #         results_af_mask.append(maskAF)
    #         glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1, ch2, maskAF,
    #                                                                                                              trace_sensitivity=20, sigma=2)
    #     except Exception:
    #         print("EXCEPTION!!\n")
    #         exception_count += 1
    #         maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=0, k=1, min_area=7,
    #                                                                      corr=0.4)
    #         results_af_mask.append(maskAF)
    #         glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1,
    #                                                                                                              ch2,
    #                                                                                                              maskAF,
    #                                                                                                              trace_sensitivity=20,
    #                                                                                                              sigma=2)
    #     ch1_result.append(im1_res_removed)
    #     ch2_result.append(im2_res_removed)
    #     # plt.imshow(maskAF, cmap=plt.cm.gray)
    # print("exception_count = ", exception_count)
    # np_results_ms = np.array(results_af_mask, np.uint8)
    # np_results_ch1 = np.array(ch1_result, np.uint8)
    # np_results_ch2 = np.array(ch2_result, np.uint8)
    # # np_gen_masks = np.array(generate_masks, np.uint8)
    # imageio.mimwrite(m1_path + m1_k6 + 'imgs_mask_af_m1_k6.tif', np_results_ms)
    # imageio.mimwrite(m1_path + m1_k6 + 'imgs_res_ch1_m1_k6.tif', np_results_ch1)
    # imageio.mimwrite(m1_path + m1_k6 + 'imgs_res_ch2_m1_k6.tif', np_results_ch2)




    # # sz  M3 multi
    # sz_path = 'sz-139/'
    # m3_path = 'm3/'
    # m3_k6 = 'parts_k6/'
    # ret, images_0_ch = cv2.imreadmulti(sz_path + m3_path + 'C1-Sz139_M3.tif')
    # ret, images_1_ch = cv2.imreadmulti(sz_path + m3_path + 'C2-Sz139_M3.tif')
    # results_af_mask = []
    # generate_masks = []
    # img_count = len(images_0_ch)
    # ch1_result = []
    # ch2_result = []
    # exception_count = 0
    # for index in range(img_count):
    #     print(index, end=' ')
    #     ch1 = images_0_ch[index]
    #     ch2 = images_1_ch[index]
    #     sigma = 1
    #     mask = generate_mask(ch1, ch2, sigma)
    #     # generate_masks.append(cp.copy(mask))
    #     try:
    #         maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=0, k=6, min_area=7, corr=0.3)
    #         results_af_mask.append(maskAF)
    #         glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1, ch2, maskAF,
    #                                                                                                              trace_sensitivity=20, sigma=2)
    #     except Exception:
    #         print("EXCEPTION!!\n")
    #         exception_count += 1
    #         maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=0, k=1, min_area=7,
    #                                                                      corr=0.4)
    #         results_af_mask.append(maskAF)
    #         glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1,
    #                                                                                                              ch2,
    #                                                                                                              maskAF,
    #                                                                                                              trace_sensitivity=20,
    #                                                                                                              sigma=2)
    #     ch1_result.append(im1_res_removed)
    #     ch2_result.append(im2_res_removed)
    #     # plt.imshow(maskAF, cmap=plt.cm.gray)
    # print("exception_count = ", exception_count)
    # np_results_ms = np.array(results_af_mask, np.uint8)
    # np_results_ch1 = np.array(ch1_result, np.uint8)
    # np_results_ch2 = np.array(ch2_result, np.uint8)
    # # np_gen_masks = np.array(generate_masks, np.uint8)
    # imageio.mimwrite(sz_path + m3_path + m3_k6 + 'imgs_mask_af_sz_m3_k6.tif', np_results_ms)
    # imageio.mimwrite(sz_path + m3_path + m3_k6 + 'imgs_res_ch1_sz_m3_k6.tif', np_results_ch1)
    # imageio.mimwrite(sz_path + m3_path + m3_k6 + 'imgs_res_ch2_sz_m3_k6.tif', np_results_ch2)





    # # sz  M4 multi
    # sz_path = 'sz-139/'
    # m4_path = 'm4/'
    # m4_k6 = 'parts_k6/'
    # ret, images_0_ch = cv2.imreadmulti(sz_path + m4_path + 'C1-Sz139_M4.tif')
    # ret, images_1_ch = cv2.imreadmulti(sz_path + m4_path + 'C2-Sz139_M4.tif')
    # results_af_mask = []
    # generate_masks = []
    # img_count = len(images_0_ch)
    # ch1_result = []
    # ch2_result = []
    # exception_count = 0
    # for index in range(img_count):
    #     print(index, end=' ')
    #     ch1 = images_0_ch[index]
    #     ch2 = images_1_ch[index]
    #     sigma = 1
    #     mask = generate_mask(ch1, ch2, sigma)
    #     # generate_masks.append(cp.copy(mask))
    #     try:
    #         maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=0, k=6, min_area=7, corr=0.3)
    #         results_af_mask.append(maskAF)
    #         glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1, ch2, maskAF,
    #                                                                                                              trace_sensitivity=20, sigma=2)
    #     except Exception:
    #         print("EXCEPTION!!\n")
    #         exception_count += 1
    #         maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=0, k=1, min_area=7,
    #                                                                      corr=0.4)
    #         results_af_mask.append(maskAF)
    #         glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1,
    #                                                                                                              ch2,
    #                                                                                                              maskAF,
    #                                                                                                              trace_sensitivity=20,
    #                                                                                                              sigma=2)
    #     ch1_result.append(im1_res_removed)
    #     ch2_result.append(im2_res_removed)
    #     # plt.imshow(maskAF, cmap=plt.cm.gray)
    # print("exception_count = ", exception_count)
    # np_results_ms = np.array(results_af_mask, np.uint8)
    # np_results_ch1 = np.array(ch1_result, np.uint8)
    # np_results_ch2 = np.array(ch2_result, np.uint8)
    # # np_gen_masks = np.array(generate_masks, np.uint8)
    # imageio.mimwrite(sz_path + m4_path + m4_k6 + 'imgs_mask_af_sz_m4_k6.tif', np_results_ms)
    # imageio.mimwrite(sz_path + m4_path + m4_k6 + 'imgs_res_ch1_sz_m4_k6.tif', np_results_ch1)
    # imageio.mimwrite(sz_path + m4_path + m4_k6 + 'imgs_res_ch2_sz_m4_k6.tif', np_results_ch2)




    # # sz  M5 multi
    # sz_path = 'sz-139/'
    # m5_path = 'm5/'
    # m5_k6 = 'parts_k6/'
    # ret, images_0_ch = cv2.imreadmulti(sz_path + m5_path + 'C1-Sz139_M5.tif')
    # ret, images_1_ch = cv2.imreadmulti(sz_path + m5_path + 'C2-Sz139_M5.tif')
    # results_af_mask = []
    # generate_masks = []
    # img_count = len(images_0_ch)
    # ch1_result = []
    # ch2_result = []
    # exception_count = 0
    # for index in range(img_count):
    #     print(index, end=' ')
    #     ch1 = images_0_ch[index]
    #     ch2 = images_1_ch[index]
    #     sigma = 1
    #     mask = generate_mask(ch1, ch2, sigma)
    #     # generate_masks.append(cp.copy(mask))
    #     try:
    #         maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=0, k=6, min_area=7, corr=0.3)
    #         results_af_mask.append(maskAF)
    #         glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1, ch2, maskAF,
    #                                                                                                              trace_sensitivity=20, sigma=2)
    #     except Exception:
    #         print("EXCEPTION!!\n")
    #         exception_count += 1
    #         maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=0, k=1, min_area=7,
    #                                                                      corr=0.4)
    #         results_af_mask.append(maskAF)
    #         glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1,
    #                                                                                                              ch2,
    #                                                                                                              maskAF,
    #                                                                                                              trace_sensitivity=20,
    #                                                                                                              sigma=2)
    #     ch1_result.append(im1_res_removed)
    #     ch2_result.append(im2_res_removed)
    #     # plt.imshow(maskAF, cmap=plt.cm.gray)
    # print("exception_count = ", exception_count)
    # np_results_ms = np.array(results_af_mask, np.uint8)
    # np_results_ch1 = np.array(ch1_result, np.uint8)
    # np_results_ch2 = np.array(ch2_result, np.uint8)
    # # np_gen_masks = np.array(generate_masks, np.uint8)
    # imageio.mimwrite(sz_path + m5_path + m5_k6 + 'imgs_mask_af_sz_m5_k6.tif', np_results_ms)
    # imageio.mimwrite(sz_path + m5_path + m5_k6 + 'imgs_res_ch1_sz_m5_k6.tif', np_results_ch1)
    # imageio.mimwrite(sz_path + m5_path + m5_k6 + 'imgs_res_ch2_sz_m5_k6.tif', np_results_ch2)



    # sz  M6 multi
    # sz_path = 'sz-139/'
    # m6_path = 'm6/'
    # m6_k6 = 'parts_k6/'
    # ret, images_0_ch = cv2.imreadmulti(sz_path + m6_path + 'C1-Sz139_M6.tif')
    # ret, images_1_ch = cv2.imreadmulti(sz_path + m6_path + 'C2-Sz139_M6.tif')
    # results_af_mask = []
    # generate_masks = []
    # img_count = len(images_0_ch)
    # ch1_result = []
    # ch2_result = []
    # exception_count = 0
    # for index in range(img_count):
    #     print(index, end=' ')
    #     ch1 = images_0_ch[index]
    #     ch2 = images_1_ch[index]
    #     sigma = 1
    #     mask = generate_mask(ch1, ch2, sigma)
    #     # generate_masks.append(cp.copy(mask))
    #     try:
    #         maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=0, k=6, min_area=7, corr=0.3)
    #         results_af_mask.append(maskAF)
    #         glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1, ch2, maskAF,
    #                                                                                                              trace_sensitivity=20, sigma=2)
    #     except Exception:
    #         print("EXCEPTION!!\n")
    #         exception_count += 1
    #         maskAF, im1AFRemoved, im2AFRemoved, kBest, = afid_identifier(ch1, ch2, mask, kAuto=0, k=1, min_area=7,
    #                                                                      corr=0.4)
    #         results_af_mask.append(maskAF)
    #         glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed = glow_identifier(ch1,
    #                                                                                                              ch2,
    #                                                                                                              maskAF,
    #                                                                                                              trace_sensitivity=20,
    #                                                                                                              sigma=2)
    #     ch1_result.append(im1_res_removed)
    #     ch2_result.append(im2_res_removed)
    #     # plt.imshow(maskAF, cmap=plt.cm.gray)
    # print("exception_count = ", exception_count)
    # np_results_ms = np.array(results_af_mask, np.uint8)
    # np_results_ch1 = np.array(ch1_result, np.uint8)
    # np_results_ch2 = np.array(ch2_result, np.uint8)
    # # np_gen_masks = np.array(generate_masks, np.uint8)
    # imageio.mimwrite(sz_path + m6_path + m6_k6 + 'imgs_mask_af_sz_m6_k6.tif', np_results_ms)
    # imageio.mimwrite(sz_path + m6_path + m6_k6 + 'imgs_res_ch1_sz_m6_k6.tif', np_results_ch1)
    # imageio.mimwrite(sz_path + m6_path + m6_k6 + 'imgs_res_ch2_sz_m6_k6.tif', np_results_ch2)



if __name__ == "__main__":
    main()