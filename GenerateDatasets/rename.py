import glob
import os

root = 'circle'
i = 1
for img_path in glob.glob(os.path.join(root, '*.JPG')):
    os.renames(img_path, 'circle_rename/%s.jpg' % i)
    i += 1
