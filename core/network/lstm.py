import torch
import torch.nn as nn
import torch.nn.functional as F

class LoopLSTM(nn.Module):
    def __init__(self, inplanes, planes, pc_size, hd_size, dropout=0.5, size=1024):
        super(LoopLSTM, self).__init__()

        self.size = size
        hidden_dims = int(planes * 2)

        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(inplanes, planes, num_layers=1, batch_first=True)
        self.hidden_fc = nn.Linear(planes, hidden_dims, bias=False)

        # self.sim_head = nn.Sequential(
        #     SimHead(hidden_dims, heads=8, dim_head=hidden_dims//8, dropout=dropout),
        # )

        self.pos_head = nn.Sequential(nn.Linear(hidden_dims, 2))
        self.dir_head = nn.Sequential(nn.Linear(hidden_dims, 1))
        self.pc_head = nn.Sequential(nn.Linear(hidden_dims, pc_size, bias=False), nn.Softmax(2))
        self.hd_head = nn.Sequential(nn.Linear(hidden_dims, hd_size, bias=False), nn.Softmax(2))

        self.init_pc_0 = nn.Linear(pc_size, planes)
        self.init_hd_0 = nn.Linear(hd_size, planes)
        self.init_pc_1 = nn.Linear(pc_size, planes)
        self.init_hd_1 = nn.Linear(hd_size, planes)

    def init_hidden(self, init_p, init_d):
        hidden = []
        hidden.append((self.init_pc_0(init_p) + self.init_hd_0(init_d)).transpose(0,1))
        hidden.append((self.init_pc_1(init_p) + self.init_hd_1(init_d)).transpose(0,1))
        return hidden

    def forward(self, x, position, direction):
        hidden = self.init_hidden(position[:,0:1,:], direction[:,0:1,:])
        x, hidden = self.lstm(x, hidden)
        x = self.hidden_fc(x)
        x = self.drop(x)

        outputs = {}
        outputs["feature"] = x
        outputs["position"] = self.pos_head(x)
        outputs["direction"] = self.dir_head(x)
        # outputs["similarity"] = self.sim_head(x)
        outputs["pc"] = self.pc_head(x)
        outputs["hd"] = self.hd_head(x)
        return outputs

class LoopRNN(nn.Module):
    def __init__(self, inplanes, planes, pc_size, hd_size, dropout=0.5, size=1024):
        super(LoopRNN, self).__init__()

        self.size = size
        hidden_dims = int(planes * 2)

        self.drop = nn.Dropout(dropout)
        self.lstm = nn.RNN(inplanes, planes, num_layers=1, batch_first=True)
        self.hidden_fc = nn.Linear(planes, hidden_dims, bias=False)

        # self.sim_head = nn.Sequential(
        #     SimHead(hidden_dims, heads=8, dim_head=hidden_dims//8, dropout=dropout),
        # )

        self.pos_head = nn.Sequential(nn.Linear(hidden_dims, 2))
        self.dir_head = nn.Sequential(nn.Linear(hidden_dims, 1))
        self.pc_head = nn.Sequential(nn.Linear(hidden_dims, pc_size, bias=False), nn.Softmax(2))
        self.hd_head = nn.Sequential(nn.Linear(hidden_dims, hd_size, bias=False), nn.Softmax(2))

        self.init_pc_0 = nn.Linear(pc_size, planes)
        self.init_hd_0 = nn.Linear(hd_size, planes)
        # self.init_pc_1 = nn.Linear(pc_size, planes)
        # self.init_hd_1 = nn.Linear(hd_size, planes)

    def init_hidden(self, init_p, init_d):
        hidden = (self.init_pc_0(init_p) + self.init_hd_0(init_d)).transpose(0,1)
        # hidden.append((self.init_pc_1(init_p) + self.init_hd_1(init_d)).transpose(0,1))
        return hidden

    def forward(self, x, position, direction):
        hidden = self.init_hidden(position[:,0:1,:], direction[:,0:1,:])
        x, hidden = self.lstm(x, hidden)
        x = self.hidden_fc(x)
        x = self.drop(x)

        outputs = {}
        outputs["feature"] = x
        outputs["position"] = self.pos_head(x)
        outputs["direction"] = self.dir_head(x)
        # outputs["similarity"] = self.sim_head(x)
        outputs["pc"] = self.pc_head(x)
        outputs["hd"] = self.hd_head(x)
        return outputs

if __name__ == "__main__":
    inputs = torch.randn((4, 3, 1024)).cuda()
    module = LoopLSTM(3, 128).cuda()
    module(inputs)