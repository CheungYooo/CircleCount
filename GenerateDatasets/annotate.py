import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio
import glob
import os
import argparse
import shutil

parser = argparse.ArgumentParser(description='annote')
parser.add_argument('-r', '--root', type=str, metavar='', required=True,
                    help='data path')
args = parser.parse_args()


root = args.root
annotated_root = '%s/annotated' % root

if not os.path.exists(annotated_root):
    os.makedirs(annotated_root)

img_paths = []
for img_path in glob.glob(os.path.join(root, '*.jpg')):
    img_paths.append(img_path)
print('total number:', len(img_paths))

for i, img_path in enumerate(img_paths):
    fp = open(img_path, 'rb')
    img = Image.open(fp)
    plt.imshow(img)
    position_list = plt.ginput(n=-1, timeout=0, show_clicks=True, mouse_add=1, mouse_pop=3, mouse_stop=27)
    count = len(position_list)
    fp.close()
    mat_path = img_path.replace('.jpg', '.mat')
    sio.savemat(mat_path, {'position': position_list})
    shutil.move(img_path, annotated_root)
    print('count:', count)
