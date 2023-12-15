import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class Conv1d(nn.Module):
    def __init__(self, in_planes, planes, ksize, stride=1, bias=True):
        super(Conv1d, self).__init__()
        self.pad_size = int(ksize - 1)
        self.conv = nn.Conv1d(in_planes, planes, ksize, stride, bias=bias)

    def forward(self, x):
        x = F.pad(x, (self.pad_size, 0),"constant", 0)
        x = self.conv(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, ksize, activate=nn.ReLU, norm=nn.BatchNorm1d):
        super(ConvBlock, self).__init__()

        self.conv1 = Conv1d(inplanes, planes, ksize)
        self.norm1 = norm(planes)
        self.active = activate()
        self.conv2 = Conv1d(planes, planes, ksize)
        self.norm2 = norm(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.active(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += identity
        out = self.active(out)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., activate=nn.GELU):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activate(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class AttentionLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_dropout=0.):
        super(AttentionLayer, self).__init__()
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3)

        self.heads = heads

        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, x, att_mask=None):
        b, n, d, h = *x.shape, self.heads

        # x = rearrange(x, 'b n d -> b d n')
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (d h) -> b h n d', h=h), qkv)
        q = q / math.sqrt(d)
        
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)# + att_mask 
        if att_mask is not None:
            dots = dots + att_mask

        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return out

class TransformerLayer(nn.Module):
    def __init__(self, planes, heads, dim_head, dropout):
        super(TransformerLayer, self).__init__()
        self.ff = FeedForward(planes, planes, dropout=dropout)
        self.attn = AttentionLayer(planes, heads, dim_head)

        self.norm1 = nn.LayerNorm(planes, eps=1e-5)
        self.norm2 = nn.LayerNorm(planes, eps=1e-5)

    def forward(self, x, att_mask=None):
        x = self.norm1(x + self.attn(x, att_mask))
        x = self.norm2(x + self.ff(x))
        return x

def factorization(num):
    factor = []
    while num > 1:
        for i in range(num - 1):
            k = i + 2
            if num % k == 0:
                factor.append(k)
        num = int(num / k)
        break
    return factor

def max_lower_k(factor, k=10):
    for i, v in enumerate(factor):
        if v >= k:
            break
    return factor[i-1]

class Matching_fb(nn.Module):
    def __init__(self, planes, heads=8, dropout=0.1):
        super(Matching_fb, self).__init__()
        if planes < 64:
            heads = 1
        else:
            factor = factorization(planes)
            heads = max_lower_k(factor, 10)

        self.transformer1 = TransformerLayer(planes, heads=heads, dim_head=planes//heads, dropout=dropout)
        self.transformer2 = TransformerLayer(planes, heads=heads, dim_head=planes//heads, dropout=dropout)

    def forward(self, x):
        size = x.shape[1]
        tril = torch.triu(torch.full((size, size), float('-inf')), diagonal=1).to(x.device)
        triu = torch.tril(torch.full((size, size), float('-inf')), diagonal=-1).to(x.device)
        td1 = torch.sigmoid(self.transformer1(x, att_mask=tril))
        td2 = torch.sigmoid(self.transformer2(x, att_mask=triu))
        sim = torch.matmul(td1, td2.transpose(1, 2))
        sim = sim / td1.shape[-1]
        # P = F.softmax(sim, 1) * F.softmax(sim, 2)
        return sim

class Matching_full(nn.Module):
    def __init__(self, planes, heads=8, dropout=0.1):
        super(Matching_full, self).__init__()

        self.transformer1 = TransformerLayer(planes, heads=heads, dim_head=planes//heads, dropout=dropout)
        self.transformer2 = TransformerLayer(planes, heads=heads, dim_head=planes//heads, dropout=dropout)

    def forward(self, x):
        td1 = torch.sigmoid(self.transformer1(x))
        td2 = torch.sigmoid(self.transformer2(x))
        sim = torch.matmul(td1, td2.transpose(1, 2))
        sim = sim / td1.shape[-1]
        # P = F.softmax(sim, 1) * F.softmax(sim, 2)
        return sim

class LoopAttention(nn.Module):
    def __init__(self, inplanes, planes, pc_size, hd_size, heads=8, dropout=0.1, size=1024):
        super(LoopAttention, self).__init__()
        self.size = size

        self.velocity_layer = nn.Linear(inplanes, planes)
        self.index_embed = nn.Embedding(self.size, planes)

        self.transformer1 = TransformerLayer(planes, heads=heads, dim_head=planes//heads, dropout=dropout)
        # self.transformer2 = TransformerLayer(planes, heads=heads, dim_head=planes//heads, dropout=dropout)
        # self.transformer3 = TransformerLayer(planes, heads=heads, dim_head=planes//heads, dropout=dropout)
        # self.transformer4 = TransformerLayer(planes, heads=heads, dim_head=planes//heads, dropout=dropout)

        hidden_planes = int(planes * 2)
        self.hidden_fc = nn.Linear(planes, hidden_planes, bias=False)
        self.drop = nn.Dropout(dropout)
        
        # self.sim_head = nn.Sequential(
        #     SimHead(hidden_planes, heads=heads, dim_head=hidden_planes//heads, dropout=dropout),
        # )

        self.pos_head = nn.Sequential(nn.Linear(hidden_planes, 2))
        self.dir_head = nn.Sequential(nn.Linear(hidden_planes, 1))
        self.pc_head = nn.Sequential(nn.Linear(hidden_planes, pc_size, bias=False), nn.Softmax(2))
        self.hd_head = nn.Sequential(nn.Linear(hidden_planes, hd_size, bias=False), nn.Softmax(2))

        self.init_pc = nn.Linear(pc_size, planes)
        self.init_hd = nn.Linear(hd_size, planes)

    def forward(self, x, position, direction):
        '''
        x: B N 4, position: B N NPC^2, direction: B N NHD
        '''
        init_feature = self.init_pc(position) + self.init_hd(direction)

        position_embed = self.index_embed(torch.arange(self.size).to(x.device))[None]
        x = self.velocity_layer(x) + position_embed + init_feature

        x = self.transformer1(x)
        # x = self.transformer2(x)
        # x = self.transformer3(x)
        # x = self.transformer4(x)
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
    module = LoopAttention(3, 128).cuda()
    module(inputs)
