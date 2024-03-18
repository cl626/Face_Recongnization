from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training, extract_face
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import datasets, transforms
import numpy as np
import os
import pickle,pdb

data_dir = '../data/lfw/lfw'
pairs_path = '../data/lfw/pairs.txt'

batch_size = 2
epochs = 15
workers = 0 if os.name == 'nt' else 8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('在该设备上运行: {}'.format(device))

with open('./embed_to_cls.pkl','rb') as file:
    embeddings_dict=pickle.load(file)
# print(list(embeddings_dict.keys())[-1])
print(len(list(embeddings_dict.keys())))

from sklearn.model_selection import KFold
from scipy import interpolate
import math

# 以下是从David Sandberg的FaceNet实现中提取的LFW函数
def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # 基于余弦相似度的距离
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist

# 根据识别结果计算ROC curve
def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    is_false_positive = []
    is_false_negative = []

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)

        # 寻找折叠的最佳阈值
        print(f"num of folds = {nrof_thresholds}")
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], _ ,_ = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        print(f"in fold_idx={fold_idx+1},max_accuracy={np.max(acc_train)},best_thredhold_index={np.argmax(acc_train)}")
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _, _, _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx], is_fp, is_fn = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
        is_false_positive.extend(is_fp)
        is_false_negative.extend(is_fn)

    return tpr, fpr, accuracy, is_false_positive, is_false_negative

# 根据实际标签(actual_isame)与预测标签(predict_issame)计算准确率
def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    is_fp = np.logical_and(predict_issame, np.logical_not(actual_issame))
    is_fn = np.logical_and(np.logical_not(predict_issame), actual_issame)

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc, is_fp, is_fn

def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)

        # 找到使FAR = far_target的阈值
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            # print(far_train)
            # print(len(far_train))
            # print(far_train[:5])
            # print(len(thresholds))
            # print(thresholds)
            # pdb.set_trace()
            # f = interpolate.interp1d(far_train, thresholds, kind='slinear')     # 应该是误识率到 门限的函数，但y=f(x),x不可有重复值，
            # threshold = f(far_target)
            threshold =4.0 *(np.max(far_train)-np.min(far_train))/(far_target-np.min(far_train))
        else:
            threshold = 0.0
        # print(f"test_set's len={len(test_set)},sum={sum(test_set)}")
        # print(f"sum of test_issame={sum(actual_issame[test_set])}")
        # print(f"sum of train_issame={sum(actual_issame[train_set])}")
        # 计算当前fold的正确率与误识率
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean

 # 计算正确率(val)与误识率(far,判错的占不一样的的占比)
def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)           # 第一元素<第二元素，小于门限判为同类
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))    #测为相同但实际不同的样本
    n_same = np.sum(actual_issame)                          
    n_diff = np.sum(np.logical_not(actual_issame))          
    val = (float(true_accept)+1) / (float(n_same)+10)       #加一平滑
    far = (float(false_accept)+1) / (float(n_diff)+10)   
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # 计算评估指标
    thresholds = np.arange(0, 6, 0.15)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, fp, fn  = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far, fp, fn

def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:  # pair=(str,str,str)
        # print(f"len={len(pair)}")
        path0=lfw_dir+'/'+pair[0]
        path1=lfw_dir+'/'+pair[1]
        # print(os.path.exists(path0))
        # print(type(pair[2]))      

        # if len(pair) == 3:
        if pair[2]=="1":
            # path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            # path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        # elif len(pair) == 4:
        elif pair[2]=="0":
            # path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            # path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        # print(issame)
        if os.path.exists(path0) and os.path.exists(path1):    # 仅在两个路径都存在时添加配对
            # epath_list += (path0,path1)
            path_list.append(path0)
            path_list.append(path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('跳过 %d 个图像对' % nrof_skipped_pairs)

    return path_list, issame_list

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs, dtype=object)

pairs = read_pairs(pairs_path)
path_list, issame_list = get_paths(data_dir+'_cropped', pairs)
# print(f"len of path_list={len(path_list)}")
# print(f"len of issame_list={len(issame_list)}")
# print(len(list(embeddings_dict.keys())))
# print(len(path_list))

# embeddings_dict是裁剪图像 到 嵌入式向量 的映射
# path是pair_txt中出现的裁剪图像中的图片
embeddings = np.array([embeddings_dict[path] for path in path_list if path in embeddings_dict.keys() ])
missed=[path for path in path_list if path not in embeddings_dict.keys()]
# print(missed)               # the last two is missed
img_path_list=list(embeddings_dict.keys())  
# print("../data/lfw/lfw_cropped/Yasser_Arafat/Yasser_Arafat_0003.jpg" in img_path_list)
# print("../data/lfw/lfw_cropped/Yasser_Arafat/Yasser_Arafat_0004.jpg" in img_path_list)
# print("../data/lfw/lfw_cropped/Yasser_Arafat/Yasser_Arafat_0003.jpg" in path_list)
# print("../data/lfw/lfw_cropped/Yasser_Arafat/Yasser_Arafat_0004.jpg" in path_list)
# print("../data/lfw/lfw_cropped/Yasser_Arafat/Yasser_Arafat_0005.jpg" in path_list)
# print(len(embeddings))

print(sum(issame_list))
print(len(issame_list))
tpr, fpr, accuracy, val, val_std, far, fp, fn = evaluate(embeddings, issame_list,distance_metric=1)
print(accuracy)
print(np.mean(accuracy))