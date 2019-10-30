from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
import argparse
import numpy as np
import h5py
import glob
import csv
import os
from PIL import Image

parser = argparse.ArgumentParser(description='generate densitymap')
parser.add_argument('-s', '--sigma', type=int, metavar='', required=True,
                    help='sigma of gaussian_filter for generation')
args = parser.parse_args()


def generate_densitymap(img_path):
    img = plt.imread(img_path)
    k = np.zeros((img.shape[0], img.shape[1]))
    csv_file = open(img_path.replace('image', 'location').replace('.jpg', '.csv'))
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        if int(row[1]) < img.shape[1] and int(row[0]) < img.shape[0]:
            k[int(row[0]), int(row[1])] = 1
    k = gaussian_filter(k, args.sigma)
    with h5py.File(img_path.replace('.jpg', '.h5').replace('image', 'densitymap%s' % args.sigma), 'w') as hf:
        hf['density'] = k


root = 'Dataset'
train_set = os.path.join(root, 'train', 'image')
test_set = os.path.join(root, 'test', 'image')
path_set = [train_set, test_set]


img_paths = []
for path in path_set:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)
print('len(img_paths):', len(img_paths))

for img_path in img_paths:
    densitymap_path = img_path.replace('.jpg', '.h5').replace('image', 'densitymap%s' % args.sigma)
    print('generating %s' % densitymap_path)
    generate_densitymap(img_path)
print('done')

# img_path = 'Dataset/train/image/IMG_100.jpg'
# densitymap_path = img_path.replace('image', 'densitymap').replace('.jpg', '.h5')
# GT_densitymap = h5py.File(densitymap_path, 'r')
# GT_densitymap = np.asarray(GT_densitymap['density'])

# ax1 = plt.subplot(1, 2, 1)
# ax1.set_title('img')
# img_test = Image.open(img_path).convert('RGB')
# plt.imshow(img_test)

# ax2 = plt.subplot(1, 2, 2)
# ax2.set_title('GT_densitymap')
# plt.imshow(GT_densitymap, cmap=plt.cm.jet)
# plt.show()

