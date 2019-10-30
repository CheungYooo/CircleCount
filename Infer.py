import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
import argparse

from Model import CSRNet

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sigma', type=int, metavar='', required=True,  # 必選參數
                    help='sigma of gaussian_filter to val')
parser.add_argument('-n', '--number', type=str, metavar='', required=True,  # 必選參數
                    help='number of pic to val,from 401 to 500 ')
args = parser.parse_args()


def load_img(path):
    img = F.to_tensor(Image.open(path).convert('RGB'))
    img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    img = torch.unsqueeze(img, 0)
    return img


def predict(path):
    img = load_img(path).cuda()
    output = model(img)
    count = torch.sum(output)
    return count, img, output


model = CSRNet().cuda()
# load chaeckpoint
checkpoint = torch.load('Model/model%s_best.pth.tar' % args.sigma)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

img_path = 'Dataset/test/image/IMG_%s.jpg' % args.number
GT_path = img_path.replace('image', 'densitymap%s' % args.sigma).replace('.jpg', '.h5')
count, img, output = predict(img_path)

GT_densitymap = h5py.File(GT_path, 'r')
GT_densitymap = np.asarray(GT_densitymap['density'])
GT_count = np.sum(GT_densitymap)
print('predict count:', count.item(), 'GT_count:', GT_count)
# 绘图
ax1 = plt.subplot(1, 3, 1)
ax1.set_title('img')
img_test = Image.open(img_path).convert('RGB')
plt.imshow(img_test)

ax2 = plt.subplot(1, 3, 2)
ax2.set_title('GT_densitymap')
plt.imshow(GT_densitymap, cmap=plt.cm.jet)

ax3 = plt.subplot(1, 3, 3)
ax3.set_title('densitymap%s' % args.sigma)
densitymap = output.cpu().detach().numpy()
plt.imshow(densitymap.reshape(densitymap.shape[2], densitymap.shape[3]), cmap=plt.cm.jet)
plt.show()
