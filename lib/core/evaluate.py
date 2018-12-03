#prediction evaluation functions
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.inference import get_max_preds


# function to get normalized diff between prediction and target
def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    x=preds.shape[1]
    y=preds.shape[0]
    dists = np.zeros((x, y)) # to store the results
    for n in range(y):
        for c in range(x):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                diff=normed_preds - normed_targets
                dists[c, n] = np.linalg.norm(diff)
            else:
                dists[c, n] = -1
    return dists

# calculate percentage value which is below thresold. 
# Need to ignore negative terms in inputs
def dist_acc(dists, thr=0.5):
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    check= num_dist_cal>0
    res=-1
    if check==True:
        summ=np.less(dists[dist_cal], thr).sum()
        res=summ*1.0/num_dist_cal
    return res

# Function to calculate accuracy based on PCK
# Avg Accuracy with Individual Accuracy 
def accuracy(output, target, hm_type='gaussian', thr=0.5):
    list_length=range(output.shape[1])
    idx = list(list_length)
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    count= 0
    idx_length=len(idx)
    for i in range(idx_length):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            count=count+1
            avg_acc = avg_acc + acc[i + 1]
    
    if count==0:
        avg_acc=0
    else:
        avg_acc=avg_acc/count
    
    if count != 0:
        acc[0] = avg_acc
    return acc, avg_acc, count, pred
