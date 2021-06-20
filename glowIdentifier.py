import math
from skimage.morphology import skeletonize as sk
import copy as cp
import numpy as np
from skimage.measure import regionprops, label
import cv2
from math import ceil, acos, pi, floor


def glow_identifier(im1, im2, maskAF, trace_sensitivity=20, sigma=2):
    skel_af = sk(cp.copy(maskAF), method='lee')
    m, n = im1.shape
    end_nodes = [[0 for _ in range(n)] for _ in range(m)]
    end_nodes = np.array(end_nodes)
    # # imsave(examples_path + "sk_lee.tif", skel_af)
    # # binary = maskAF > filters.threshold_otsu(maskAF)
    # # skel_af = sk(binary)
    # skel_af = io.imread(examples_path + 'skelAF_matlab.png')
    # # imsave("sk_std.tif", skel_af)
    count = 0
    for r in range(m):
        for c in range(n):
            if skel_af[r][c] != 0:
                neighbours = [0 for _ in range(8)]
                r_neighbour = [r - 1, r, r + 1, r - 1, r + 1, r - 1, r, r + 1]
                c_neighbour = [c - 1, c - 1, c - 1, c, c, c + 1, c + 1, c + 1]
                for i in range(len(r_neighbour)):
                    try:
                        neighbours[i] = 1 if skel_af[r_neighbour[i]][c_neighbour[i]] != 0 else 0
                    except Exception:
                        neighbours[i] = 0
                if sum(neighbours) < 2:
                    end_nodes[r][c] = 255
                    count += 1
    # print('un_nul end_nodes elements = ', count)
    count = 0
    labels = label(skel_af, connectivity=2)
    max_intensiv_skel = []
    skel_af_pixels_struct = regionprops(labels, end_nodes)
    with_node = skel_af_pixels_struct
    for region in skel_af_pixels_struct:
        max_intensiv_skel.append(region.max_intensity)
    # print('len_max_intensiv_skel = ', len(max_intensiv_skel))
    skeleton_pixel_idx = regionprops(labels, skel_af)
    region_count = len(skeleton_pixel_idx)
    for index in range(region_count):
        if with_node[index] == 0:
            pixels = skeleton_pixel_idx[index].coords
            x = pixels[0, 0]
            y = pixels[0, 1]
            end_nodes[x][y] = 255
            count += 1
    # print(count)
    trace = cp.copy(end_nodes)
    exp_points = cp.copy(end_nodes)
    done = [[0 for _ in range(n)] for _ in range(m)]
    done = np.array(done)
    trace_count = 0
    exp_count = 0
    new_tr_count = 0

    while True:
        trace_count += 1
        new_trace = [[0 for _ in range(n)] for _ in range(m)]
        new_trace = np.array(new_trace)
        for r in range(m):
            for c in range(n):
                if trace[r][c] != 0:
                    done[r][c] = 255
                    r_neighbour = [r - 1, r, r + 1, r - 1, r + 1, r - 1, r, r + 1]
                    c_neighbour = [c - 1, c - 1, c - 1, c, c, c + 1, c + 1, c + 1]
                    for i in range(len(r_neighbour)):
                        r2 = r_neighbour[i]
                        c2 = c_neighbour[i]
                        try:
                            if 0 <= r2 < m and 0 <= c2 < n and skel_af[r2][c2] != 0 and done[r2][c2] == 0\
                                    and trace[r2][c2] == 0 and new_trace[r2][c2] == 0:
                                        new_trace[r2][c2] = 255
                                        new_tr_count += 1
                                        if trace_count % trace_sensitivity == 0:
                                            exp_points[r2][c2] = 255
                                            exp_count += 1
                        except Exception:
                            new_trace[r2][c2] = 0
        trace = cp.copy(new_trace)
        if np.sum(trace) == 0:
            break
        if trace_count > 40:
            break
    # print('exp_count = ', exp_count)
    # print('trace_count  = ', trace_count)
    # print('new_tr_count  = ', new_tr_count)
    # imsave(examples_path + "exp_points.png", exp_points)
    # exp_points = io.imread(examples_path + 'exp_points.png')
    if sigma > 0:
        k = 2 * ceil(2 * sigma) + 1
        im_1_blurred = cv2.GaussianBlur(im1, ksize=(k, k), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
        im_2_blurred = cv2.GaussianBlur(im2, ksize=(k, k), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    else:
        im_1_blurred = im1
        im_2_blurred = im2

    count1 = 0
    count2 = 0
    #  Identify body
    extend_dist = 60
    extend_width = 1
    th = acos(1 - (extend_width / extend_dist) ** 2 / 2)
    min_steps = 3
    max_steps = 30
    glow1 = [[0 for _ in range(n)] for _ in range(m)]
    glow1 = np.array(glow1)
    glow2 = [[0 for _ in range(n)] for _ in range(m)]
    glow2 = np.array(glow2)
    unnul_exp_point_elemetns_end = 0
    for r in range(m):
        for c in range(n):
            if exp_points[r][c] != 0:
                unnul_exp_point_elemetns_end += 1
                th_multiple = 1
                th_measure = th
                while th_measure < 2 * pi:
                    th_measure = th * th_multiple
                    x_step_size = math.cos(th_measure)
                    y_step_size = math.sin(th_measure)
                    extend_length = 1
                    extend_steps = 0
                    done1 = 0
                    done2 = 0
                    while True:
                        x_measure = floor(r + extend_length * x_step_size)
                        y_measure = floor(c + extend_length * y_step_size)
                        x_compare = floor(r + (extend_length - 1) * x_step_size)
                        y_compare = floor(c + (extend_length - 1) * y_step_size)
                        if x_measure < 0 or x_measure >= m or y_measure < 0 or y_measure >= n:
                            break
                        gg = maskAF[x_measure][y_measure]
                        if maskAF[x_measure][y_measure] == 0:
                            extend_steps += 1
                            im1_dif = im_1_blurred[x_measure, y_measure]
                            im2_dif = im_1_blurred[x_compare, y_compare]
                            pixel_difference1 = float(im_1_blurred[x_measure, y_measure]) - float(im_1_blurred[x_compare, y_compare])
                            pixel_difference2 = float(im_2_blurred[x_measure, y_measure]) - float(im_2_blurred[x_compare, y_compare])
                            if pixel_difference1 <= 0 or (extend_steps < min_steps and done1 == 0):
                                glow1[x_measure][y_measure] = 255
                                count1 += 1
                            else:
                                done1 = 1

                            if pixel_difference2 <= 0 or (extend_steps < min_steps and done2 == 0):
                                glow2[x_measure][y_measure] = 255
                                count2 += 1
                            else:
                                done2 = 1

                        if (done1 != 0 and done2 != 0) or extend_steps > max_steps:
                            break
                        extend_length += 1
                    th_multiple += 1

    # print('unnul_exp_point_elemetns_end = ', unnul_exp_point_elemetns_end)
    # print('count_glow1_255 = ', count1)
    # print('count2glow2_255 = ', count2)
    im1_res_removed = cp.copy(im1)
    im2_res_removed = cp.copy(im2)
    im1_glow_removed = cp.copy(im1)
    # im1_glow_removed[maskAF == 0] = 0
    im1_glow_removed[glow1 == 0] = 0
    im2_glow_removed = cp.copy(im2)
    # im2_glow_removed[maskAF == 0] = 0
    im2_glow_removed[glow2 == 0] = 0

    im1_res_removed[maskAF != 0] = 0
    im1_res_removed[glow1 != 0] = 0
    im2_res_removed[maskAF != 0] = 0
    im2_res_removed[glow2 != 0] = 0
    return glow1, glow2, im1_glow_removed, im2_glow_removed, im1_res_removed, im2_res_removed





