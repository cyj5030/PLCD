import torch
import sklearn.metrics as metrics
import numpy as np

def precision_recall_f1(pred, label, num_th=10, thy=0.5):
    batch = pred.shape[0]
    pred = pred.reshape(batch, -1)
    label = label.reshape(batch, -1)

    th = torch.linspace(0, 1, num_th+2)[1:-1].view(-1, 1, 1).to(pred.device)
    pred = pred[None].repeat(num_th, 1, 1)
    label = label[None].repeat(num_th, 1, 1)
    TP = torch.logical_and(pred > th, label > thy).float().sum(2)
    FP = torch.logical_and(pred > th, label <= thy).float().sum(2)
    FN = torch.logical_and(pred <= th, label > thy).float().sum(2)
    p, r = TP / (TP + FP).clamp(1e-6), TP / (TP + FN).clamp(1e-6)
    f = 2*p*r / (p + r).clamp(1e-6)

    best_id = torch.argmax(f.mean(1))
    p_best, r_best, f_best = p[best_id].mean(), r[best_id].mean(), f[best_id].mean()

    max_id = torch.argmax(f, dim=0)
    p_max, r_max, f_max = torch.diag(p[max_id]).mean(), torch.diag(r[max_id]).mean(), torch.diag(f[max_id]).mean()
    return p_best, r_best, f_best, p_max, r_max, f_max