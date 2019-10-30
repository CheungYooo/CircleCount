import os
import glob
import argparse
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torchvision.transforms.functional as F

from Model import CSRNet

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sigma', type=int, metavar='', required=True,  # 必選參數
                    help='sigma of gaussian_filter to val')
args = parser.parse_args()

root = 'Dataset'
test_set = os.path.join(root, 'test', 'image')
all_test_img_paths = []
for img_path in glob.glob(os.path.join(test_set, '*.jpg')):
    all_test_img_paths.append(img_path)
model = CSRNet()
checkpoint = torch.load('Model/model%s_best.pth.tar' % args.sigma)
model.load_state_dict(checkpoint['model_state_dict'])
transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])

model.eval()
with torch.no_grad():
    mae = 0.0
    mse = 0.0
    for img_path in all_test_img_paths:
        img = Image.open(img_path).convert('RGB')
        GT_path = img_path.replace('image', 'densitymap%s' % args.sigma).replace('.jpg', '.h5')
        GT_densitymap = h5py.File(GT_path, 'r')
        GT_densitymap = np.asarray(GT_densitymap['density'])
        GT_count = np.sum(GT_densitymap)
        img = img.transform
        output = model(img)
        ae = abs(output.data.sum() - np.sum(GT_count))
        mae += ae
        se = ae * ae
        mse += se
    mae = mae / len(all_test_img_paths)
    mse = np.sqrt(mse /len(all_test_img_paths))
    print('MAE {mae:.3f}  MSE {mse:.3f} '
            .format(mae=mae, mse=mse))


