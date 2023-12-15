import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from core.model_utils import *
from core.noise_model import NoiseModel
from evaluation_scripts.eval_utils import *
from core.network.transformer import LoopAttention, Matching_fb, Matching_full
from core.network.lstm import LoopLSTM, LoopRNN

class LoopModule(nn.Module):
    def __init__(self, args):
        super(LoopModule, self).__init__()
        self.args = args
        self.size = args.size
        input_dims = int(2 * (args.sincos_scale + 1))
        
        pc_size = args.half_pc_size
        hd_size = args.hd_size
        self.sigma_pc = 2 / (6 * args.half_pc_size)
        self.piror_pc_cells, pc_size = pc_piror([-1, 1], pc_size)
        self.piror_hd_cells = hd_piror(hd_size)

        if args.backbone == "attention":
            self.denoise_net = LoopAttention(input_dims, args.hidden_dims, pc_size, hd_size, dropout=0.5, size=self.size)
        elif args.backbone == "rnn":
            self.denoise_net = LoopRNN(input_dims, args.hidden_dims, pc_size, hd_size, dropout=0.5, size=self.size)
        elif args.backbone == "lstm":
            self.denoise_net = LoopLSTM(input_dims, args.hidden_dims, pc_size, hd_size, dropout=0.5, size=self.size)
        
        dims = 0
        if args.feature_loss == "cell":
            if "direction" in args.feature:
                dims += hd_size
            if "position" in args.feature:
                dims += pc_size
            if "hidden" in args.feature:
                dims += int(args.hidden_dims * 2)
        elif args.feature_loss == "cartesian":
            if "direction" in args.feature:
                dims += 1
            if "position" in args.feature:
                dims += 2
            if "hidden" in args.feature:
                dims += int(args.hidden_dims * 2)
        elif args.feature_loss == "None":
            dims = int(args.hidden_dims * 2)
        
        if self.args.trick == "fullatt":
            self.match_net = Matching_full(dims)
        else: #self.args.trick == "fbatt":
            self.match_net = Matching_fb(dims)

        if args.noise_model == "mix":
            self.noise_model = NoiseModel(self.size - 1, False)
        elif args.noise_model == "gaussian":
            self.noise_model = NoiseModel(self.size - 1, True)
        elif args.noise_model == "None":
            self.noise_model = None
    
    def forward(self, trajectory_input, trajectory_label, loop_label, loop_score):
        trajectory_input = interp1d(trajectory_input, self.size)
        trajectory_label = interp1d(trajectory_label, self.size)
        loop_label = interp1d(loop_label.unsqueeze(-1), self.size)
        loop_score = interp2d(loop_score, self.size)

        velo, position, direction = traj2velo(trajectory_input, self.noise_model)
        velo_input = velo2input(velo, self.args.sincos_scale)

        features = self.denoise_net(velo_input, encode_pc(position, self.piror_pc_cells, self.sigma_pc), encode_hd(direction, self.piror_hd_cells))
        feature = []
        if self.args.feature_loss == "cell":
            if "direction" in self.args.feature:
                feature.append(features["hd"])
            if "position" in self.args.feature:
                feature.append(features["pc"])
            if "hidden" in self.args.feature:
                feature.append(features["feature"])
        elif self.args.feature_loss == "cartesian":
            if "direction" in self.args.feature:
                feature.append(features["direction"])
            if "position" in self.args.feature:
                feature.append(features["position"])
            if "hidden" in self.args.feature:
                feature.append(features["feature"])
        elif self.args.feature_loss == "None":
            feature.append(features["feature"])
        score = self.match_net(torch.cat(feature, -1))
        features["similarity"] = score

        total_loss, losses = self.calc_losses(features, trajectory_label, loop_label, loop_score)
        # metrics = self.calc_metrics(outputs, loop_label)
        # metrics.update(losses)
        return total_loss, losses

    @torch.no_grad()
    def infer(self, trajectory_input):
        trajectory_input = interp1d(trajectory_input, self.size)
        
        velo, position, direction = traj2velo(trajectory_input, self.noise_model)
        # start = time.time()
        velo_input = velo2input(velo, self.args.sincos_scale)
        # torch.cuda.synchronize()
        # end = time.time()
        
        features = self.denoise_net(velo_input, encode_pc(position, self.piror_pc_cells, self.sigma_pc), encode_hd(direction, self.piror_hd_cells))
        
        # print('Time:{}ms'.format((end-start)*1000))

        feature = []
        if self.args.feature_loss == "cell":
            if "direction" in self.args.feature:
                feature.append(features["hd"])
            if "position" in self.args.feature:
                feature.append(features["pc"])
            if "hidden" in self.args.feature:
                feature.append(features["feature"])
        elif self.args.feature_loss == "cartesian":
            if "direction" in self.args.feature:
                feature.append(features["direction"])
            if "position" in self.args.feature:
                feature.append(features["position"])
            if "hidden" in self.args.feature:
                feature.append(features["feature"])
        elif self.args.feature_loss == "None":
            feature.append(features["feature"])
        
        score = self.match_net(torch.cat(feature, -1))
        
        features["similarity"] = score

        return features

    def calc_losses(self, outputs, trajectory_label, loop_label, loop_score):
        losses = {}

        # similarity loss
        score_loss = self.Sim_loss(outputs["similarity"], loop_score, self.args.threshold)

        theta = torch.arctan2(trajectory_label[:, :, 1], trajectory_label[:, :, 0])
        # cell loss
        if self.args.feature_loss == "cell":
            gt_pc = encode_pc(trajectory_label, self.piror_pc_cells, self.sigma_pc)
            gt_hd = encode_hd(theta.unsqueeze(-1), self.piror_hd_cells)
            pc_loss = self.NLL_loss(outputs["pc"], gt_pc)
            hd_loss = self.NLL_loss(outputs["hd"], gt_hd)
            
            if "hidden" in self.args.feature or ("position" in self.args.feature and "direction" in self.args.feature):
                total_loss = score_loss + pc_loss + hd_loss
            elif "position" in self.args.feature and "direction" not in self.args.feature:
                total_loss = score_loss + pc_loss
            elif "direction" in self.args.feature and "position" not in self.args.feature:
                total_loss = score_loss + hd_loss
            
            losses["L_pc"] = pc_loss.detach()
            losses["L_hd"] = hd_loss.detach()

        elif self.args.feature_loss == "cartesian":
            # abs trajectory loss
            trajectory_loss = self.L1_loss(outputs["position"], trajectory_label, beta=0.01)
            direction_loss = self.L1_loss(outputs["direction"], theta.unsqueeze(-1), beta=0.01)

            if "hidden" in self.args.feature:
                total_loss = score_loss + trajectory_loss + direction_loss
            elif "position" in self.args.feature and "direction" not in self.args.feature:
                total_loss = score_loss + trajectory_loss
            elif "direction" in self.args.feature and "position" not in self.args.feature:
                total_loss = score_loss + direction_loss
        
        
            losses["L_trajectory"] = trajectory_loss.detach()
            losses["L_direction"] = direction_loss.detach()

        elif self.args.feature_loss == "None":
            total_loss = score_loss

        losses["L_score"] = score_loss.detach()
        return total_loss, losses
    
    def NLL_loss(self, x, y, eps=1e-6):
        aN = x.shape[-1]
        x = x.clamp(eps, 1.0-eps)
        cross_entropy = -torch.sum(x.log().view(-1, aN) * y.view(-1, aN), -1).mean()
        return cross_entropy

    def L1_loss(self, x, y, beta=0.1):
        return F.smooth_l1_loss(x, y, reduction='mean', beta=beta)

    # def Sim_loss(self, pred, label, loop_label, th=0.):
    #     pred = pred.view(-1, self.size)
    #     label = label.view(-1, self.size)
    #     vaild_score_ids = (loop_label.reshape(-1) > 0).nonzero().view(-1)
    #     cross_entropy = self.CE_loss(pred[vaild_score_ids], label[vaild_score_ids], th=th)
    #     return cross_entropy

    def Sim_loss(self, pred, label, th=0.2):
        ce0 = self.CrossEntropy(pred, label, th)
        if self.args.trick == "nolms":
            return ce0
        else:
            ce1 = self.CrossEntropy(pred.max(1)[0], label.max(1)[0], th)
            ce2 = self.CrossEntropy(pred.max(2)[0], label.max(2)[0], th)
            # ce1 = self.CrossEntropy(pred.mean(1), label.mean(1), 0.0)
            # ce2 = self.CrossEntropy(pred.mean(2), label.mean(2), 0.0)
            return ce0 + ce1 + ce2

    def CrossEntropy(self, pred, label, th):
        pred = torch.clamp(pred, 1e-6, 1-1e-6)
        pos_mask = label > th
        neg_mask = label <= th

        loss_pos = - torch.log(pred[pos_mask]) * label[pos_mask]
        loss_neg = - torch.log(1 - pred[neg_mask])
        cross_entropy = loss_pos.mean() + loss_neg.mean()
        return cross_entropy

    def CE_loss(self, logits, labels, th=0.):
        labels = labels.reshape(labels.shape[0], -1)
        logits = logits.reshape(logits.shape[0], -1)
        length = labels.shape[1]

        labels = (labels > th).float() 
        N_pos = labels.sum(1, True)
        N_neg = length - N_pos

        alpha = (1 / N_pos).repeat(1, length)
        beta = (1 / N_neg).repeat(1, length)

        weight = torch.where(labels == 1, alpha, beta)
        cross_entropy = F.binary_cross_entropy(logits, labels, reduction='none', weight=weight).sum(1).mean()
        return cross_entropy

    def calc_metrics(self, outputs, loop_label):
        metrics = {}
        p_best, r_best, f_best, p_max, r_max, f_max = precision_recall_f1(outputs["loop"], loop_label, thy=self.args.threshold)
        metrics["p"] = p_best.detach()
        metrics["r"] = r_best.detach()
        metrics["f"] = f_best.detach()
        return metrics

if __name__ == "__main__":
    inputs = torch.randn((4, 1024, 6)).cuda()
    pose_label = torch.randn((4, 1024, 6)).cuda()
    loop_label = (torch.randn((4, 1024)).cuda() > 0.5).float()
    score_label = torch.randn((4, 1024, 1024)).cuda()
    module = LoopModule(1).cuda()
    module(inputs, pose_label, loop_label, score_label)