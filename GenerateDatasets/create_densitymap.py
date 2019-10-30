from scipy.ndimage.filters import gaussian_filter
import scipy.io as sio
import scipy.spatial
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import h5py


def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density

#
# root = 'dataset'
# train_path = os.path.join(root, 'train_data', 'images')
# test_path = os.path.join(root, 'test_data', 'images')
# path_sets = [train_path, test_path]
# # # process
# # img_paths = []
# # for path in path_sets:
# #     for img_path in glob.glob(os.path.join(path, '*.jpg')):
# #         img_paths.append(img_path)
# # print('len(img_paths):', len(img_paths))
# # print('processing...')
# # for img_path in img_paths:
# #     mat = sio.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
# #     img = plt.imread(img_path)
# #     k = np.zeros((img.shape[0], img.shape[1]))
# #     gt = mat["position"]
# #     for i in range(0, len(gt)):
# #         if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
# #             k[int(gt[i][1]), int(gt[i][0])] = 1
# #     k = gaussian_filter_density(k)
# #     with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'densitymap'), 'w') as hf:
# #         hf['density'] = k
#
# # process part_B
# img_paths = []
# for path in path_sets:
#     for img_path in glob.glob(os.path.join(path, '*.jpg')):
#         img_paths.append(img_path)
# print('len(img_paths):', len(img_paths))
# print('processing  ...')
# for img_path in img_paths:
#     mat = sio.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'ground_truth').replace('IMG_', 'GT_IMG_'))
#     img = plt.imread(img_path)
#     k = np.zeros((img.shape[0], img.shape[1]))
#     gt = mat["position"]
#     for i in range(0, len(gt)):
#         if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
#             k[int(gt[i][1]), int(gt[i][0])] = 1
#     k = gaussian_filter(k, 15)
#     with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'densitymap'), 'w') as hf:  # 存储矩阵k
#         hf['density'] = k
# print('done')

# test

ax1 = plt.subplot(1, 2, 1)
ax1.set_title('img')
img_path = 'dataset/train_data/images/IMG_1.png'
img = plt.imread(img_path)
plt.imshow(img)
ax1 = plt.subplot(1, 2, 2)
ax1.set_title('gt')
gt_file = h5py.File(img_path.replace('images', 'densitymap').replace('.png', '.h5'), 'r')
ground_truth = np.asarray(gt_file['density'])
plt.imshow(ground_truth, cmap=plt.cm.jet)
plt.show()
Gauss_sum = np.sum(ground_truth)
print('Gauss_sum:', Gauss_sum)
