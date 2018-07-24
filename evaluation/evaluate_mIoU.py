from argparse import ArgumentParser
from PIL import Image
import os
import glob
import time
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--gts', default="")
    parser.add_argument('--preds', default="")
    parser.add_argument('--num-workers', type=int, default=10)
    
    args = parser.parse_args()

    return args


def add_to_confusion_matrix(gt, pred, mat):
#    print(pred.shape)
#    print(pred.size[0],pred.size[1])

    if (pred.shape[0] != gt.shape[0]):
        print("Image widths of " + pred + " and " + gt + " are not equal.")
    if (pred.shape[1] != gt.shape[1]):
        print("Image heights of " + pred + " and " + gt + " are not equal.")
    if ( len(pred.shape) != 2 ):
        print("Predicted image has multiple channels.")
    W  = pred.shape[0]
    H = pred.shape[1]
    P = H*W

    pred = pred.flatten()
    gt = gt.flatten()


    for (gtp,predp) in zip(gt, pred):
        if gtp == 255:
            gtp = 26
        mat[gtp, predp] += 1

    return mat

def eval_ious(mat):
    ious = np.zeros(27)
    for l in range(26):
        tp = np.longlong(mat[l,l])
        fn = np.longlong(mat[l,:].sum()) - tp

        notIgnored = [i for i in range(26) if not i==l]
        fp = np.longlong(mat[notIgnored,l].sum())
        denom = (tp + fp + fn)
        if denom == 0:
            print('error: denom is 0')

        ious[l] =  float(tp) / denom

    return ious[:-1]

def process_pred_gt_pair(pair):
    pred, gt = pair
    # tqdm.tqdm.write(pred, gt)
    confusion_matrix = np.zeros(shape=(26+1, 26+1),dtype=np.ulonglong)
    try:
        gt = Image.open(gt)
        # print(gt.size)
        if gt.size != (1920, 1080):
            gt = gt.resize((1920, 1080), resample=Image.NEAREST)
        gt  = np.array(gt)
    except:
        print("Unable to load " + gt)
    try:
        pred = Image.open(pred)
        if pred.size != (1920, 1080):
            pred = pred.resize((1920, 1080), resample=Image.NEAREST)
        pred = np.array(pred)
    except:
        print("Unable to load " + pred)

    # plt.matshow(gt)
    # plt.show()
    # plt.matshow(pred)
    # plt.show()


    
    

    # print(pred.size,gt.size)
    
    add_to_confusion_matrix(gt, pred, confusion_matrix)

    return confusion_matrix

import tqdm

class_names = ['road', 'drivable fallback', 'sidewalk', 'non-drivable fallback', 'person', 'rider', 'motorcycle', 'bicycle', 'autorickshaw', 'car', 'truck', 'bus', 'vehicle fallback', 'curb', 'wall', 'fence', 'guard rail', 'billboard', 'traffic sign', 'traffic light', 'pole', 'obs-str-bar-fallback', 'building', 'bridge', 'vegetation', 'sky', 'misc']


def main(args):
    confusion_matrix    = np.zeros(shape=(26+1, 26+1),dtype=np.ulonglong)
    gts_folders         = glob.glob(args.gts + '/*')
    pred_folders        = [ gtf.replace(args.gts, args.preds) for gtf in gts_folders ]
    # print(gts_folders)

    gts     = []
    preds   = []
    for i, gtf in enumerate(gts_folders):
        g = glob.glob(gtf+'/*_labellevel3Ids.png')
        p = [ j.replace(gtf, pred_folders[i]) for j in g]
        gts += g
        preds += p

    pairs = [(preds[i], gts[i]) for i in range(len(gts))]

    pool = Pool(args.num_workers)
    # results = pool.map(process_pred_gt_pair, pairs)
    # results = []
    # for i,p in enumerate(pairs):
    #     results.append(process_pred_gt_pair(p))
    #     print(i)

    results = list(tqdm.tqdm(pool.imap(process_pred_gt_pair, pairs), total=len(pairs)))
    pool.close()
    pool.join()

    for i in range(len(results)):
        confusion_matrix += results[i]


    np.save('cm',confusion_matrix)

    ious = eval_ious(confusion_matrix)
    np.save('ious', np.array(ious))
    for i in range(26):
        print(f'{class_names[i]}:\t\t\t\t {ious[i]*100}')

    print(f'mIoU:\t\t\t\t{ious.mean()*100}')

        
        
if __name__ == '__main__':
    args = get_args()
    main(args)
