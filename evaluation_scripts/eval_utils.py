import os
import torch
import torch.nn.functional as F
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from pylab import mpl
fname = "/home/cyj/code/VO/debug/test-flow-PANet(full)-noise_Log/fzhei.ttf"


from core.datasets.utils import mkdir

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

def precision_recall_curve(label, pred):
    size = label.shape[0]
    num_th = 1000
    thresholds = torch.linspace(0, 1, num_th+2)[1:-1].reshape(-1, 1).to(label.device) # 1, 1000

    pred = pred[None].expand(num_th, -1)
    label = label[None].expand(num_th, -1)
    try:
        TP = torch.logical_and(pred >  thresholds, label == 1).sum(1)
        FP = torch.logical_and(pred >  thresholds, label == 0).sum(1)
        FN = torch.logical_and(pred <= thresholds, label == 1).sum(1)
    except:
        try:
            TP, FP, FN = torch.zeros(num_th), torch.zeros(num_th), torch.zeros(num_th)
            sp = torch.cat([torch.arange(0, num_th, 15), torch.tensor([num_th])], 0)
            for i in range(sp.shape[0] - 1):
                step_slice = slice(int(sp[i]), int(sp[i+1]))
                TP[step_slice] = torch.logical_and(pred[step_slice] >  thresholds[step_slice], label[step_slice] == 1).sum(1)
                FP[step_slice] = torch.logical_and(pred[step_slice] >  thresholds[step_slice], label[step_slice] == 0).sum(1)
                FN[step_slice] = torch.logical_and(pred[step_slice] <= thresholds[step_slice], label[step_slice] == 1).sum(1)
        except:
            TP, FP, FN = torch.zeros(num_th), torch.zeros(num_th), torch.zeros(num_th)
            sp = torch.cat([torch.arange(0, num_th, 1), torch.tensor([num_th])], 0)
            for i in range(sp.shape[0] - 1):
                step_slice = slice(int(sp[i]), int(sp[i+1]))
                TP[step_slice] = torch.logical_and(pred[step_slice] >  thresholds[step_slice], label[step_slice] == 1).sum(1)
                FP[step_slice] = torch.logical_and(pred[step_slice] >  thresholds[step_slice], label[step_slice] == 0).sum(1)
                FN[step_slice] = torch.logical_and(pred[step_slice] <= thresholds[step_slice], label[step_slice] == 1).sum(1)
            
    torch.cuda.empty_cache()
    p = TP / (TP + FP).clamp(1e-12)
    r = TP / (TP + FN).clamp(1e-12)
    p = torch.nan_to_num(p, nan=1.0)
    r = torch.nan_to_num(r, nan=1.0)
    
    return p.cpu().numpy(), r.cpu().numpy(), thresholds.cpu().numpy()

def nms(x, w=11):
    batch, length = x.shape
    padw = int((w - 1) / 2)

    x = F.pad(x, (padw, padw))
    for j in range(length):
        cid = j + padw
        x[:, cid] = torch.where(x[:, cid] != torch.max(x[:, cid-padw: cid+padw + 1], 1)[0], torch.zeros_like(x[:, cid]), x[:, cid] )
    x = x[:, padw:-padw]
    return x

def calc_roc(pred, label, label_threshold):
    pred = pred.view(-1).cpu().numpy()
    label = label.view(-1).cpu().numpy() > label_threshold

    fpr, tpr, threshold = metrics.roc_curve(label, pred)
    auc = metrics.auc(fpr, tpr)
    return fpr, tpr, threshold, auc

def calc_pr_cuda(pred, label, label_threshold):
    pred = pred.view(-1)
    label = label.view(-1) > label_threshold

    np_pred = pred.cpu().numpy()
    np_label = label.cpu().numpy()

    # precision, recall, threshold = metrics.precision_recall_curve(np_label, np_pred)
    precision, recall, threshold = precision_recall_curve(label.cuda(), pred.cuda())
    ap = metrics.average_precision_score(np_label, np_pred)
    return precision, recall, threshold, ap

def calc_pr(pred, label, label_threshold):
    pred = pred.view(-1)
    label = label.view(-1) > label_threshold

    np_pred = pred.cpu().numpy()
    np_label = label.cpu().numpy()

    precision, recall, threshold = metrics.precision_recall_curve(np_label, np_pred)
    # precision, recall, threshold = precision_recall_curve(label, pred)
    ap = metrics.average_precision_score(np_label, np_pred)
    return precision, recall, threshold, ap

def calc_auc(pred, label, label_threshold):
    pred = pred.view(-1).cpu().numpy()
    label = label.view(-1).cpu().numpy() > label_threshold

    fpr, tpr, thresholds = metrics.roc_curve(label, pred)
    auc = metrics.auc(fpr, tpr)
    return auc

def max_f1(p, r, th=False):
    f1 = 2*p*r / (p + r).clip(1e-6)
    idx_bestF = np.argmax(f1)
    idx_bestR = np.argmax(r)
    idx_bestP = np.argmax(p)
    ods_p = p[idx_bestF]
    ods_r = r[idx_bestF]
    ods_f1 = f1[idx_bestF]
    if th:
        return ods_p, ods_r, ods_f1, idx_bestF
    return ods_p, ods_r, ods_f1

def max_recall_wrt_precision(p, r, label, label_threshold):
    # np_label = (label > label_threshold).cpu().numpy()
    # sum_total = np_label.sum()
    # parted_sum = (np_label.sum(1) > 0.0).sum()
    try:
        p_ids = np.nonzero(p >= 0.99)[0]
        r_max = np.max(r[p_ids])
        r_id = np.argmax(r[p_ids])
    except:
        r_max = 0.0
        r_id = 0
    return r_max, r_id

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def load_txt(filename):
    with open(filename, "r") as f:
        data = f.readlines()
    return data

'''

'''
colors = [
        [1.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 1.0000],
        [0.7098, 0.2000, 0.3608],
        [0.4902, 0.0706, 0.6863],
        [0.7059, 0.5333, 0.8824],
        [0.8000, 0.8000, 0.1000],
        [0.0588, 0.6471, 0.6471],
        [0.0392, 0.4275, 0.2667],
        [0.4157, 0.5373, 0.0824],
        [1.0000, 0.0000, 1.0000],
        [0.5490, 0.5490, 0.4549],
        [0.9373, 0.6863, 0.1255],
        [0.4471, 0.3333, 0.1725],
        [0.0000, 1.0000, 1.0000],
        [0.7176, 0.5137, 0.4392],
        [1.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 1.0000],
        [0.7098, 0.2000, 0.3608],
    ]

def plot_save_lines(results, methods, save_filename, xylabel_title, 
                    x=None, xrange=None, yrange=None, linewidth=2, legend=False):
    linewidth = linewidth
    # xrange=(0.0,1.0)
    # yrange=(0.0,1.0)

    fig1 = plt.figure(1)
    if isinstance(results, list):
        length = len(results)
    else:
        length = results.shape[0]
    for i in range(length):
        if x is None:
            plt.plot(results[i], c=colors[i], linewidth=linewidth, label=methods[i])
        elif isinstance(x, tuple):
            x_r = np.linspace(x[0], x[1], results[i].shape[0])
            plt.plot(x_r, results[i], c=colors[i], linewidth=linewidth, label=methods[i])
        else:
            plt.plot(x, results[i], c=colors[i], linewidth=linewidth, label=methods[i])
    
    plt.tick_params(direction='in')
    if xrange is not None:
        plt.xlim(xrange[0],xrange[1])
        xyrange1 = np.arange(xrange[0], xrange[1]+0.01, 0.1)
        plt.xticks(xyrange1,fontsize=14,fontname='serif')

    if yrange is not None:
        plt.ylim(yrange[0],yrange[1])
        xyrange2 = np.arange(yrange[0], yrange[1]+0.01, 0.1)
        plt.yticks(xyrange2,fontsize=14,fontname='serif')
        

    plt.xlabel(xylabel_title[0],fontsize=20,fontname='serif')
    plt.ylabel(xylabel_title[1],fontsize=20,fontname='serif')
    plt.title(xylabel_title[2],fontsize=20,fontname='serif')

    font1 = {'family': 'serif',
    'weight': 'normal',
    'size': 10,
    }
    if legend:
        plt.legend(loc=legend, prop=font1)
    # plt.grid(linestyle='--')
    # plt.axis('off')
    fig1.savefig(save_filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_save_lines_text(x, y, text, methods, save_filename, xlabel, y_label, title, linewidth=2, legend=True):
    linewidth = linewidth

    fig, ax = plt.subplots()
    text_mag = [
        np.array([.0,]*10),
        np.array([1.5, 1.0, 1.5, 1.5, 2.0, 2.0, 0.5, 0.5, 1.0, 2.0]),
        np.array([.0]*10),
    ]
    length = len(methods)
    for i in range(length - 1):
        ax.plot(x[i], y[i], "-o", c=colors[i], linewidth=linewidth, label=methods[i])
        # for j, tt in enumerate(text[i]):
        #     # if not np.isnan(text[i][j]):
        #     # ax.text(x[i][j]-0.02, y[i][j]+(np.array(y).max()/50), tt, fontsize=10, va='bottom')
        #     ax.text(x[i][j]-0.02, y[i][j]+text_mag[i][j], tt, fontsize=13, va='bottom')
    ax.plot(x[i+1], y[i+1], c=colors[i+1], linewidth=linewidth, label=methods[i+1])
    
    font0 = {'family': 'serif',
        'weight': 'normal',
        'size': 20,
    }
    ax.tick_params(labelsize=20)
    ax.set_ylim(0, ymax=5 * np.ceil((np.array(y).max() + 3) / 5))

    ax.set_xlabel(xlabel,fontsize=18,fontname='serif')
    ax.set_ylabel(y_label,fontsize=18,fontname='serif')
    ax.set_title(title,fontsize=18,fontname='serif')

    font1 = {'family': 'serif',
    'weight': 'normal',
    'size': 13,
    }
    if legend:
        # ax.legend(loc=legend, prop=font1), loc='lower left'
        ax.legend(prop=font1)
    # plt.grid(linestyle='--')
    # plt.axis('off')
    fig.savefig(save_filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_save_lines_text_chinese(x, y, text, methods, save_filename, xlabel, y_label, title, linewidth=2, legend=True):
    linewidth = linewidth

    fig, ax = plt.subplots()
    text_mag = [
        np.array([.0,]*10),
        np.array([1.5, 1.0, 1.5, 1.5, 2.0, 2.0, 0.5, 0.5, 1.0, 2.0]),
        np.array([.0]*10),
    ]
    length = len(methods)
    for i in range(length - 1):
        ax.plot(x[i], y[i], "-o", c=colors[i], linewidth=linewidth, label=methods[i])
        for j, tt in enumerate(text[i]):
            # if not np.isnan(text[i][j]):
            ax.text(x[i][j]-0.02, y[i][j]+(np.array(y).max()/50), tt, fontsize=10, va='bottom')
            # ax.text(x[i][j]-0.02, y[i][j]+text_mag[i][j], tt, fontsize=10, va='bottom')
    ax.plot(x[i+1], y[i+1], c=colors[i+1], linewidth=linewidth, label=methods[i+1])
    
    zhfont = mpl.font_manager.FontProperties(fname=fname, size=13)
    ax.set_xticks(x[0], fontsize=13,fontproperties=zhfont)
    ax.set_ylim(0.0, np.array(y).max() + 3)

    zhfont = mpl.font_manager.FontProperties(fname=fname, size=18)
    ax.set_xlabel(xlabel,fontsize=18,fontproperties=zhfont)
    ax.set_ylabel(y_label,fontsize=18,fontproperties=zhfont)
    ax.set_title(title,fontsize=18,fontproperties=zhfont)

    zhfont = mpl.font_manager.FontProperties(fname=fname, size=12)
    font1 = {'family': 'serif',
    'weight': 'normal',
    'size': 10,
    }
    if legend:
        # ax.legend(loc=legend, prop=font1)
        ax.legend(prop=zhfont)
    # plt.grid(linestyle='--')
    # plt.axis('off')
    fig.savefig(save_filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_save_pr_curves(PRE, REC, method_names, title, save_filename):
    xrange=(0.0,1.0)
    yrange=(0.0,1.0)
    linewidth = 2

    fig1 = plt.figure(1)
    num = len(PRE)
    for i in range(0,num):
        if isinstance(PRE[i], float):
            plt.scatter(REC[i], PRE[i], c='k', s=30, zorder=3, label=method_names[i])
        else:
            plt.plot(REC[i], PRE[i], c=colors[i], linewidth=linewidth, label=method_names[i])
            
    plt.xlim(xrange[0],xrange[1])
    plt.ylim(yrange[0],yrange[1])

    xyrange1 = np.arange(xrange[0],xrange[1]+0.01,0.1)
    xyrange2 = np.arange(yrange[0],yrange[1]+0.01,0.1)

    plt.tick_params(direction='in')
    plt.xticks(xyrange1,fontsize=15,fontname='serif')
    plt.yticks(xyrange2,fontsize=15,fontname='serif')

    plt.xlabel('Recall',fontsize=16,fontname='serif')
    plt.ylabel('Precision',fontsize=16,fontname='serif')

    plt.title(title, fontsize=16,fontname='serif')

    font1 = {'family': 'serif',
    'weight': 'normal',
    'size': 10,
    }

    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = [len(handles)-x for x in range(1,len(handles)+1)]
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],loc='lower left', prop=font1)
    plt.legend(loc='upper right', prop=font1)
    # plt.legend(prop=font1)
    plt.grid(linestyle='--')
    fig1.savefig(save_filename, bbox_inches='tight',dpi=300)
    plt.close()

def plot_save_pr_curves_chinese(PRE, REC, method_names, title, save_filename):
    xrange=(0.0,1.0)
    yrange=(0.0,1.0)
    linewidth = 2

    fig1 = plt.figure(1)
    num = len(PRE)
    for i in range(0,num):
        if isinstance(PRE[i], float):
            plt.scatter(REC[i], PRE[i], c='k', s=30, zorder=3, label=method_names[i])
        else:
            plt.plot(REC[i], PRE[i], c=colors[i], linewidth=linewidth, label=method_names[i])
            
    plt.xlim(xrange[0],xrange[1])
    plt.ylim(yrange[0],yrange[1])

    xyrange1 = np.arange(xrange[0],xrange[1]+0.01,0.1)
    xyrange2 = np.arange(yrange[0],yrange[1]+0.01,0.1)

    plt.tick_params(direction='in')
    zhfont = mpl.font_manager.FontProperties(fname=fname, size=13)
    plt.xticks(xyrange1,fontsize=13,fontproperties=zhfont)
    plt.yticks(xyrange2,fontsize=13,fontproperties=zhfont)

    zhfont = mpl.font_manager.FontProperties(fname=fname, size=16)
    plt.xlabel('召回率',fontsize=16,fontproperties=zhfont)
    plt.ylabel('精确率',fontsize=16,fontproperties=zhfont)

    plt.title(title, fontsize=16,fontproperties=zhfont)

    font1 = {'family': 'serif',
    'weight': 'normal',
    'size': 10,
    }

    zhfont = mpl.font_manager.FontProperties(fname=fname, size=11)
    plt.legend(loc='upper right', prop=zhfont)
    # plt.legend(prop=font1)
    plt.grid(linestyle='--')
    fig1.savefig(save_filename, bbox_inches='tight',dpi=300)
    plt.close()

def plot_score(score, out_file):
    fig, ax = plt.subplots()
    ax.matshow(score)

    mkdir(os.path.dirname(out_file))
    plt.savefig(out_file, bbox_inches="tight", pad_inches=0.0)
    plt.show()
    plt.close()

def plot_traj_gt(pose, gt, save_filename):
    fig, ax = plt.subplots()

    length = pose.shape[0]
    for i in range(length):
        ax.scatter(pose[i][0], pose[i][1], s=20, c="k")
    # for i in range(length):
    #     if gt[i] == 1:
    #         ax.scatter(pose[i][0], pose[i][1], s=20, c="b")
    # for i in range(length):
    #     if gt[i] == 2:
    #         ax.scatter(pose[i][0], pose[i][1], s=20, c="r")
    for i in range(length):
        if gt[i] == 3:
            ax.scatter(pose[i][0], pose[i][1], s=20, c="orangered") # c

        if gt[i] == 1:
            ax.scatter(pose[i][0], pose[i][1], s=20, c="b")
        if gt[i] == 2:
            ax.scatter(pose[i][0], pose[i][1], s=20, c="r")

    plt.axis('off')
    fig.savefig(save_filename, bbox_inches='tight', dpi=300)
    plt.close()

import yaml
def load_args(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
    