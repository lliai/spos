import os
import numpy as np
import csv


def save_split_data(raw_data_dir, train_save_path, val_save_path):
    all_cls = os.listdir(raw_data_dir)
    cls_dict = {k: v for (v, k) in enumerate(all_cls)}
    img_list = []
    for cls in all_cls:
        path = os.path.join(raw_data_dir, cls)
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
            path = os.path.join(raw_data_dir, cls, im)
            if i < val_num:
                val_data.append((path, cls_id))
            else:
                train_data.append((path, cls_id))

    with open(train_save_path, 'w') as fout:
        csv_out = csv.writer(fout)
        csv_out.writerows(train_data)

    with open(val_save_path, 'w') as fout:
        csv_out = csv.writer(fout)
        csv_out.writerows(val_data)
