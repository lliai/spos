import os
import numpy as np
import csv
import argparse


parser = argparse.ArgumentParser('data-split')
parser.add_argument('--data-dir', type=str, default='/home/wangguangrun/ILSVRC2012/train', help='path to the original training dataset')
parser.add_argument('--train-save-path', type=str, default='./supernet_train_data.csv', help='path to save the split training dataset')
parser.add_argument('--val-save-path', type=str, default='./supernet_val_data.csv', help='path to save the split validate dataset')
args = parser.parse_args()


def main():
    all_cls = os.listdir(args.data_dir)
    cls_dict = {k: v for (v, k) in enumerate(all_cls)}
    img_list = []
    for cls in all_cls:
        path = os.path.join(args.data_dir, cls)
        temp_list = []
        for img in os.listdir(path):
            temp_list.append(img)
        temp_list = np.sort(temp_list).tolist()
        img_list.append(temp_list)

    train_data = []
    val_data = []
    val_num = 50
    for imgs in img_list:
        for i, im in enumerate(imgs):
            cls = im.split('_')[0]
            cls_id = cls_dict[cls]
            path = os.path.join(args.data_dir, cls, im)
            if i < val_num:
                val_data.append((path, cls_id))
            else:
                train_data.append((path, cls_id))

    with open(args.train_save_path, 'w') as fout:
        csv_out = csv.writer(fout)
        csv_out.writerows(train_data)

    with open(args.val_save_path, 'w') as fout:
        csv_out = csv.writer(fout)
        csv_out.writerows(val_data)


if __name__ == '__main__':
    main()