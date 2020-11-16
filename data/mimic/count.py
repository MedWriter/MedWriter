import pandas as pd
import os
from tqdm import tqdm
import random
import numpy as np
import json
import shutil
import cv2
def selet_sample(meta, report):
    base = '/home/ubuntu/yxy/mimic-cxr/'
    selected_files = []
    for index, row in tqdm(report.iterrows()):

        # Filter sample with incomplete reports
        if pd.isna(row.findings) or pd.isna(row.findings):
            continue
        study_id = int(row.study[1:])
        report_img_df = meta[meta.study_id == study_id]

        # Filter sample with less than 2 images or with one-view
        view = set(report_img_df.ViewPosition)
        if report_img_df.__len__() < 2 or ('PA' not in view and 'AP' not in view) or 'LATERAL' not in view:
            continue
        # print(view)
        if 'PA' in view:
            pa_view = report_img_df[report_img_df.ViewPosition=='PA'].iloc[0]
        else:
            pa_view = report_img_df[report_img_df.ViewPosition == 'AP'].iloc[0]
        lateral_view = report_img_df[report_img_df.ViewPosition=='LATERAL'].iloc[0]

        subject_id = str(pa_view.subject_id)
        # print(subject_id)
        subject_path = 'physionet.org/files/mimic-cxr-jpg/2.0.0/files/p{}/p{}/s{}'.format(subject_id[:2],
                                                                                                     subject_id,
                                                                                                     study_id)
        pa_path = os.path.join(subject_path, pa_view.dicom_id+'.jpg')
        lateral_path = os.path.join(subject_path, lateral_view.dicom_id+'.jpg')

        if os.path.isfile(os.path.join(base, pa_path)) and os.path.isfile(os.path.join(base, lateral_path)):
            selected_files.append(' '.join([str(study_id), pa_path, lateral_path]))
    print('Total study', len(selected_files))
    with open('mimic_seleted_list.txt', 'w') as f:
        f.write('\n'.join(selected_files))

def split(ratio = (0.7,0.1,0.2)):
    f = open('mimic_seleted_list.txt')
    lines = [line.strip() for line in f.read().split('\n')]
    random.shuffle(lines)
    data_size = len(lines)
    train_size = int(ratio[0] * data_size)
    val_size = int(ratio[1] * data_size)
    train_list = lines[:train_size]
    val_list = lines[train_size:train_size+val_size]
    test_list = lines[train_size+val_size:]
    with open('train_mimic.txt', 'w') as f:
        f.write('\n'.join(train_list))
    with open('val_mimic.txt', 'w') as f:
        f.write('\n'.join(val_list))
    with open('test_mimic.txt', 'w') as f:
        f.write('\n'.join(test_list))

def mimic_class_label(chexpert):
    # print(chexpert.head())
    f = open('mimic_seleted_list.txt')
    study_ids = [int(line.strip().split()[0]) for line in f.read().split('\n')]
    labels = {}
    for id in tqdm(study_ids):
        label = chexpert[chexpert.study_id == id].iloc[:,2:]
        label_value = np.array(label.values)
        label_value[label_value !=1]=0
        label_value[label_value ==1]=1
        labels[id] = label_value.tolist()[0]
    with open('mimic_label_dict.json','w') as f:
        json.dump(labels, f, indent=4)


def mimic_report2json(report):
    print(report.head(),report.keys())

    f = open('mimic_seleted_list.txt')
    study_ids = [line.strip().split()[0] for line in f.read().split('\n')]
    report_dict = {}
    for id in tqdm(study_ids):
        r = report[report.study == 's'+id]

        impression = r.impression.values[0]
        findings = r.findings.values[0]

        report_dict[id] = {'findings':str(findings),
                           'impression': str(impression)}
    with open('mimic_report_dict.json','w') as f:
        json.dump(report_dict, f, indent=4)

def move_img():
    f = open('mimic_seleted_list.txt')
    source_base = '/home/ubuntu/yxy/mimic-cxr/'
    target_base = '/home/ubuntu/yxy/mimic-cxr-selected/'
    lines = [line.strip() for line in f.read().split('\n')]

    for line in tqdm(lines):
        _, img1, img2 = line.split()
        source1 = os.path.join(source_base, img1)
        target1 = os.path.join(target_base, img1)
        target1_dir = os.path.split(target1)[0]
        if os.path.exists(target1_dir) == False:
            os.makedirs(target1_dir)
        shutil.copyfile(source1, target1)
        source2 = os.path.join(source_base, img2)
        target2 = os.path.join(target_base, img2)

        shutil.copyfile(source2, target2)

def move_resize_img():
    f = open('mimic_seleted_list.txt')
    source_base = '/home/ubuntu/yxy/mimic-cxr/'
    target_base = '/home/ubuntu/yxy/mimic-cxr-selected/'
    lines = [line.strip() for line in f.read().split('\n')]

    for line in tqdm(lines):
        _, img1, img2 = line.split()
        source1 = os.path.join(source_base, img1)
        target1 = os.path.join(target_base, img1)
        target1_dir = os.path.split(target1)[0]
        if os.path.exists(target1_dir) == False:
            os.makedirs(target1_dir)

        img1 = cv2.imread(source1)
        img1 = cv2.resize(img1,(512,512) )
        cv2.imwrite(target1, img1)
        # shutil.copyfile(source1, target1)
        source2 = os.path.join(source_base, img2)
        target2 = os.path.join(target_base, img2)
        img2 = cv2.imread(source2)
        img2 = cv2.resize(img2, (512, 512))
        cv2.imwrite(target2, img2)
        # shutil.copyfile(source2, target2)
if __name__ == '__main__':
    # base_path = '/home/ubuntu/yxy/mimic-cxr/physionet.org/files/mimic-cxr-jpg/2.0.0'
    # meta = pd.read_csv(os.path.join(base_path,'mimic-cxr-2.0.0-metadata.csv.gz'))
    # print(meta.keys())
    # # print(meta.head())
    # # print(meta.study_id[:10])
    # report = pd.read_csv('mimic_cxr_sectioned.csv')
    # mimic_report2json(report)
    # print(report.head())
    # selet_sample(meta, report)
    # split()
    # chexpert = pd.read_csv(os.path.join(base_path,'mimic-cxr-2.0.0-chexpert.csv.gz'))
    # mimic_class_label(chexpert)
    # move_img()
    move_resize_img()