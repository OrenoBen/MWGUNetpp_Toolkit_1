import torch
import numpy as np

def eval_image(pred, gt, num_classes):
    """
    计算图像分割的评估指标
    
    Args:
        pred: 预测结果，一维数组
        gt: 真实标签，一维数组
        num_classes: 类别数量
    
    Returns:
        TP: 真阳性
        FP: 假阳性
        TN: 真阴性
        FN: 假阴性
        n_valid_sample: 有效样本数
    """
    # 初始化统计变量
    TP = np.zeros((num_classes, 1))
    FP = np.zeros((num_classes, 1))
    TN = np.zeros((num_classes, 1))
    FN = np.zeros((num_classes, 1))
    
    # 对于每个类别计算TP, FP, TN, FN
    for i in range(num_classes):
        # 预测为i且真实为i
        TP[i] = np.sum((pred == i) & (gt == i))
        # 预测为i但真实不为i
        FP[i] = np.sum((pred == i) & (gt != i))
        # 预测不为i且真实不为i
        TN[i] = np.sum((pred != i) & (gt != i))
        # 预测不为i但真实为i
        FN[i] = np.sum((pred != i) & (gt == i))
    
    # 计算有效样本数（排除可能的背景值）
    n_valid_sample = np.sum(gt < num_classes)
    
    return TP, FP, TN, FN, n_valid_sample

@torch.no_grad()
def iou_score(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()
    inter = (preds * targets).sum(dim=(2,3))
    union = (preds + targets - preds*targets).sum(dim=(2,3)) + eps
    iou = (inter + eps) / union
    return iou.mean().item()

@torch.no_grad()
def dice_score(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()
    inter = (preds * targets).sum(dim=(2,3)) * 2
    denom = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + eps
    dice = (inter + eps) / denom
    return dice.mean().item()