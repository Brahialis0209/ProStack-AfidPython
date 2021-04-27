import math
from collections import defaultdict
from scipy.stats import ttest_ind
import numpy as np
from skimage.measure import regionprops, label
from skimage.color import rgb2gray
from scipy.stats import kurtosis
from sklearn.cluster import KMeans
import copy as cp


def mean(data):
    return sum(data) / len(data)


def disp(data):
    res_mean = mean(data)
    sum = 0
    for value in data:
        sum += (value - res_mean)
    return sum


def disp_sq(data):
    res_mean = mean(data)
    sum = 0
    for value in data:
        sum += (value - res_mean) ** 2
    return sum


def pir_coef(x, y):
    cov = disp(x) * disp(y)
    sq = math.sqrt(disp_sq(x) * disp_sq(y))
    if sq == 0:
        return np.array([0])
    res = cov / sq
    if abs(res) < 1e-13:
        return np.array([0])
    return res


def update_for_log_values(old_values, kurt=False):
    new_values = []
    for old_value in old_values:
        if old_value == 0.0:
            if kurt:
                print("Warning. Kurtious vals have 0!!", end='\n')
                new_values.append(np.array([0.001]))
            else:
                print("Warning. Std vals have 0!!", end='\n')
                new_values.append(0.001)
        else:
            new_values.append(old_value)
    return new_values


def update_kr_values(kurtVals):
    new_kurtVals = []
    for kurt in kurtVals:
        new_kurtVals.append(kurt[0])
    return new_kurtVals


def splitapply(X, G):
    G = G.tolist()
    result = []
    dict = defaultdict(list)
    for id, x in enumerate(X):
        value = x[0]
        cluster_num = G[id][0]
        dict[cluster_num].append(value)
    dict_keys = list(dict.keys())
    dict_keys.sort()
    for key in dict_keys:
        dict_value = dict[key]
        mean_value = mean(dict_value)
        result.append(mean_value)
    return result


def k_best(statVals, x1, x2, y1, y2):
    k_max = 0
    max = -math.inf
    for index in range(x1, x2):
        buf = (y2 - y1) * index - 17 * statVals[index] + 20 * y1 - y2 * 3
        if buf >= max:
            k_max = index
            max = buf
    return k_max


def afid_identifier(im1, im2, bw, k=20, corr=-0.6, kAuto=1, min_area=8):
    # Classify autofluorescence
    labels = label(bw, connectivity=2)
    im1PixelsStruct = regionprops(labels, im1)
    im2PixelsStruct = regionprops(labels, im2)
    pixelsStruct = regionprops(labels, bw)
    index = 0
    # max_area = 2000
    while index != len(im1PixelsStruct):
        # filtering regions with little area
        if im1PixelsStruct[index].area < min_area or im2PixelsStruct[index].area < min_area or pixelsStruct[
            index].area < min_area:
            pixels = pixelsStruct[index].coords
            for pixel in pixels[:, :]:
                x = pixel[0]
                y = pixel[1]
                bw[x, y] = 0
            del im1PixelsStruct[index]
            del im2PixelsStruct[index]
            del pixelsStruct[index]
        else:
            index += 1

    objCount = len(im1PixelsStruct)
    corr_vals = []
    # Correlation
    for index in range(objCount):
        # delete null pexels
        im1PixelVals = im1PixelsStruct[index].intensity_image
        im1PixelVals = np.reshape(im1PixelVals, (-1, 1), order='F')
        im1PixelVals_buf = []
        for i in range(im1PixelVals.shape[0]):
            if im1PixelVals[i, :] != 0:
                im1PixelVals_buf.append(im1PixelVals[i, :][0])
        im1PixelVals = np.array(im1PixelVals_buf)
        im1PixelVals = np.reshape(im1PixelVals, (-1, 1))

        im2PixelVals = im2PixelsStruct[index].intensity_image
        im2PixelVals = np.reshape(im2PixelVals, (-1, 1), order='F')
        im2PixelVals_buf = []
        for i in range(im2PixelVals.shape[0]):
            if im2PixelVals[i, :] != 0:
                im2PixelVals_buf.append(im2PixelVals[i, :][0])
        im2PixelVals = np.array(im2PixelVals_buf)
        im2PixelVals = np.reshape(im2PixelVals, (-1, 1))
        if len(im1PixelVals) != len(im2PixelVals) :
            corr_vals.append(0)
            continue
        im1PixelVals = im1PixelVals.astype(float)
        im2PixelVals = im2PixelVals.astype(float)
        corr_val = np.corrcoef(im1PixelVals, im2PixelVals, rowvar=False)[0, 1]
        corr_vals.append(corr_val)

    # K-means
    if k > 1:
        # Additional texture measures
        std1Vals = []
        std2Vals = []
        kurt1Vals = []
        kurt2Vals = []
        for index in range(objCount):
            im1PixelVals = im1PixelsStruct[index].intensity_image
            im1PixelVals = np.reshape(im1PixelVals, (-1, 1), order='F')
            im1PixelVals_buf = []
            for i in range(im1PixelVals.shape[0]):
                if im1PixelVals[i, :] != 0:
                    im1PixelVals_buf.append(im1PixelVals[i, :][0])
            im1PixelVals = np.array(im1PixelVals_buf)
            im1PixelVals = np.reshape(im1PixelVals, (-1, 1))

            im2PixelVals = im2PixelsStruct[index].intensity_image
            im2PixelVals = np.reshape(im2PixelVals, (-1, 1), order='F')
            im2PixelVals_buf = []
            for i in range(im2PixelVals.shape[0]):
                if im2PixelVals[i, :] != 0:
                    im2PixelVals_buf.append(im2PixelVals[i, :][0])
            im2PixelVals = np.array(im2PixelVals_buf)
            im2PixelVals = np.reshape(im2PixelVals, (-1, 1))

            im1PixelVals = im1PixelVals.astype(float)
            im2PixelVals = im2PixelVals.astype(float)
            std1Vals.append(np.std(im1PixelVals))
            std2Vals.append(np.std(im2PixelVals))
            kurt1Vals.append(kurtosis(im1PixelVals, fisher=False))
            kurt2Vals.append(kurtosis(im2PixelVals, fisher=False))

        new_corr_vals = []
        for corr_val in corr_vals:
            if math.isnan(corr_val):
                print("Warning. Corr vals have Nan!!",end='\n')
                new_corr_vals.append(0)
            elif corr_val == -1:
                print("Warning. Corr vals have -1!!", end='\n')
                new_corr_vals.append(-0.999)
            elif corr_val == 1:
                print("Warning. Corr vals have 1!!", end='\n')
                new_corr_vals.append(0.999)
            else:
                new_corr_vals.append(corr_val)

        # Transform texture measures for clustering
        corrValsTform = np.arctanh(new_corr_vals)
        buf = np.std(corrValsTform)
        corrValsTform = corrValsTform / buf

        std1ValsTform = np.log(update_for_log_values(std1Vals))
        buf = np.std(std1ValsTform)
        std1ValsTform = std1ValsTform / buf

        std2ValsTform = np.log(update_for_log_values(std2Vals))
        buf = np.std(std2ValsTform)
        std2ValsTform = std2ValsTform / buf

        kurt1ValsTform = np.log(update_for_log_values(kurt1Vals, kurt=True))
        buf = np.std(kurt1ValsTform)
        kurt1ValsTform = kurt1ValsTform / buf

        kurt2ValsTform = np.log(update_for_log_values(kurt2Vals, kurt=True))
        buf = np.std(kurt2ValsTform)
        kurt2ValsTform = kurt2ValsTform / buf

        kurt1ValsTform = np.reshape(kurt1ValsTform, (1, -1))
        kurt2ValsTform = np.reshape(kurt2ValsTform, (1, -1))
        corrValsTform = np.reshape(corrValsTform, (-1, 1))
        corrValsTform = corrValsTform.tolist()
        std1ValsTform = np.reshape(std1ValsTform, (-1, 1))
        std1ValsTform = std1ValsTform.tolist()
        std2ValsTform = np.reshape(std2ValsTform, (-1, 1))
        std2ValsTform = std2ValsTform.tolist()
        kurt1ValsTform = np.reshape(kurt1ValsTform, (-1, 1))
        kurt1ValsTform = kurt1ValsTform.tolist()
        kurt2ValsTform = np.reshape(kurt2ValsTform, (-1, 1))
        kurt2ValsTform = kurt2ValsTform.tolist()
        toCluster = np.array(corrValsTform)
        toCluster = np.append(toCluster, std1ValsTform, axis=1)
        toCluster = np.append(toCluster, std2ValsTform, axis=1)
        toCluster = np.append(toCluster, kurt1ValsTform, axis=1)
        toCluster = np.append(toCluster, kurt2ValsTform, axis=1)

        # Perform k-means
        if kAuto != 0:
            kMax = k  # clusters count
            corrClust = []
            statVals = [0 for _ in range(kMax)]
            afSize = [0 for _ in range(kMax)]

            kmean = KMeans(n_clusters=1, max_iter=10000).fit(toCluster).labels_
            kmean_trans = np.reshape(kmean, (-1, 1))
            idx = np.array(kmean_trans)
            # Identify top two clusters
            corrMean = splitapply(corrValsTform, kmean_trans)
            corrClust_ind = 0
            corrClust.append(corrClust_ind)
            corrMean[corrClust_ind] = -math.inf
            secondClust = max(corrMean)
            secondClust_ind = corrMean.index(secondClust)
            corrVals1 = []
            corrVals2 = []
            index = 0
            for id in idx[:, 0]:
                if id == corrClust_ind:
                    afSize[0] += 1
                    corrVals1.append(corrValsTform[index][0])
                if id == secondClust_ind:
                    corrVals2.append(corrValsTform[index][0])
                index += 1
            # Measure t test values
            stats = ttest_ind(corrVals1, corrVals2)
            statVals[0] = stats.statistic
            for k in range(2, kMax + 1):
                kmean = KMeans(n_clusters=k, max_iter=10000).fit(toCluster).labels_
                kmean_trans = np.reshape(kmean, (-1, 1))
                idx = np.append(idx, kmean_trans, axis=1)
                corrMean = splitapply(corrValsTform, kmean_trans)
                corrClust_ind = corrMean.index(max(corrMean))
                corrClust.append(corrClust_ind)
                corrMean[corrClust_ind] = -math.inf
                secondClust = max(corrMean)
                secondClust_ind = corrMean.index(secondClust)
                index = 0
                corrVals1 = []
                corrVals2 = []
                for id in idx[:, k - 1]:
                    if id == corrClust_ind:
                        afSize[k - 1] += 1
                        corrVals1.append(corrValsTform[index][0])
                    if id == secondClust_ind:
                        corrVals2.append(corrValsTform[index][0])
                    index += 1
                stats = ttest_ind(corrVals1, corrVals2)
                statVals[k - 1] = stats.statistic
            # Perform elbow test
            x1 = 2
            x2 = kMax
            y1 = statVals[2]
            y2 = statVals[kMax - 1]
            kBest = k_best(statVals, x1, x2, y1, y2)
            if kBest < 18:
                kBest = kBest + 2
            kBest_value = corrClust[kBest]
            clustAF = []
            for id in idx[:, kBest]:
                if id == kBest_value:
                    clustAF.append(1)
                else:
                    clustAF.append(0)
        else:  # Non-automated
            kmean = KMeans(n_clusters=k, max_iter=10000).fit(toCluster).labels_
            kmean_trans = np.reshape(kmean, (-1, 1))
            idx = np.array(kmean_trans)
            corrMean = splitapply(corrValsTform, kmean_trans)
            corrClust_ind = corrMean.index(max(corrMean))
            clustAF = []
            for id in idx[:, 0]:
                if id == corrClust_ind:
                    clustAF.append(1)
                else:
                    clustAF.append(0)
            kBest = k
    else:
        clustAF = [1 for _ in range(objCount)]
        kBest = 1


    # Remove autofluorescence
    maskAF = bw
    afIdx = []
    if k > 1:
        for index, bit in enumerate(new_corr_vals):
                if bit != 0 and clustAF[index] != 0:
                    afIdx.append(1)
                else:
                    afIdx.append(0)
    else:
        for index, bit in enumerate(corr_vals):
            if bit > corr:
                afIdx.append(1)
            else:
                afIdx.append(0)



    count_changes = 0
    for index in range(objCount):
        pixels = pixelsStruct[index].coords
        if afIdx[index] == 0:
            for pixel in pixels[:, :]:
                x = pixel[0]
                y = pixel[1]
                maskAF[x, y] = 0
                count_changes += 1

    print(count_changes)
    im1AFRemoved = cp.copy(im1)
    im1AFRemoved[maskAF == 0] = 0
    im2AFRemoved = cp.copy(im2)
    im2AFRemoved[maskAF == 0] = 0
    return maskAF, im1AFRemoved, im2AFRemoved, kBest
