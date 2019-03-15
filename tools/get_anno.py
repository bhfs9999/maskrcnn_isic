import cv2
import numpy as np
import os

anno_root = '/data/mxj/Skin/ISIC-2017/annotations_600'
save_root = '/data/lxc/Skin/ISIC-2017/detection_600'

pad_ratio = 0.1

for dataset in os.listdir(anno_root):
    info_dict = {}
    info_dict_ori = {}
    dataset_root = os.path.join(anno_root, dataset)

    for img in os.listdir(dataset_root):
        anno_path = os.path.join(dataset_root, img)

        img_anno = cv2.imread(anno_path)
        img_anno_r = img_anno[:, :, 0]

        rows, cols = img_anno_r.shape

        idx_zero = np.nonzero(img_anno_r)
        row_min = np.min(idx_zero[0])
        row_max = np.max(idx_zero[0])
        col_min = np.min(idx_zero[1])
        col_max = np.max(idx_zero[1])

        info_dict_ori[img] = [col_min, row_min, col_max, row_max, rows, cols]

        anno_rows = row_max - row_min + 1
        anno_cols = col_max - col_min + 1
        pad_rows = int(pad_ratio * anno_rows)
        pad_cols = int(pad_ratio * anno_cols)

        row_min = max(row_min - pad_rows, 0)
        row_max = min(row_max + pad_rows, rows)
        col_min = max(col_min - pad_cols, 0)
        col_max = min(col_max + pad_cols, cols)

        cv2.rectangle(img_anno, (col_min, row_min), (col_max, row_max), (0, 0, 255), thickness=2)

        save_path = os.path.join(save_root, dataset, img)
        cv2.imwrite(save_path, img_anno)
        print('saved detection annotation at: ', save_path)

        info_dict[img] = [col_min, row_min, col_max, row_max, rows, cols]

    with open(os.path.join(save_root, dataset, 'annotations.txt'), 'w') as f:
        for name, info in info_dict.items():
            f.write('%s\t%s %s %s %s\t%s %s\n' % (name, info[0], info[1], info[2], info[3], info[4], info[5]))

    with open(os.path.join(save_root, dataset, 'annotations_ori.txt'), 'w') as f:
        for name, info in info_dict_ori.items():
            f.write('%s\t%s %s %s %s\t%s %s\n' % (name, info[0], info[1], info[2], info[3], info[4], info[5]))

