import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
import numpy as np


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch, num_samples=32):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=num_samples)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        #               ,BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Attention(torch.nn.Module):
    def __init__(self, NN_h, NN_l, NN_g, NN_f=None):
        super(Attention, self).__init__()
        self.M_h = NN_h
        self.M_l = NN_l
        self.M_g = NN_g
        self.M_f = NN_f

    def forward(self, *inputs):
        pass


class SelfAttention(Attention):

    def __init__(self, NN_h, NN_l, NN_g, NN_f=None):
        super(SelfAttention, self).__init__(NN_h, NN_l, NN_g, NN_f)

    def forward(self, p):
        h = self.M_h(p).transpose(-1, -2)
        l = self.M_l(p)
        g = self.M_g(p)
        mm = torch.matmul(l, h)
        attn_weights = F.softmax(mm, dim=-1)
        atten_appllied = torch.bmm(attn_weights, g)
        if self.M_f is not None:
            return self.M_f(p + atten_appllied)
        else:
            return p + atten_appllied


class SkipAttention(Attention):

    def __init__(self, NN_h, NN_l, NN_g, NN_f=None):
        super(SkipAttention, self).__init__(NN_h, NN_l, NN_g, NN_f)

    def forward(self, p, r):
        h = self.M_h(p).expand(-1, -1, r.size(2), -1).unsqueeze(-2)
        l = self.M_l(r).expand(-1, h.size(1), -1, -1).unsqueeze(-1)
        g = self.M_g(r).squeeze()
        mm = torch.matmul(h, l).squeeze()
        attn_weights = F.softmax(mm, dim=-1)
        atten_appllied = torch.bmm(attn_weights, g)
        if self.M_f is not None:
            return self.M_f(p.squeeze() + atten_appllied)
        else:
            return p.squeeze() + atten_appllied


class FoldingBlock(torch.nn.Module):
    def __init__(self, input_shape, output_shape, attentions, NN_up, NN_down):
        super(FoldingBlock, self).__init__()
        self.in_shape = input_shape
        self.out_shape = output_shape
        self.self_attn = SelfAttention(*attentions)
        self.M_up = MLP(NN_up)
        self.M_down = MLP(NN_down)
        self.self_attn2 = SelfAttention(*attentions)
        self.M_up2 = MLP(NN_up)
        self.M_down2 = MLP(NN_down)

    def Up_module(self, p, m, n):

        p = p.repeat(1, int(self.out_shape / self.in_shape), 1).contiguous()
        points = SaNet.sample_2D(m, n)
        p_2d = torch.cat((p, points.unsqueeze(0).expand(p.size(0), -1, -1)), -1)
        p_2d = self.M_up(p_2d)
        p = torch.cat([p, p_2d], -1)

        return self.self_attn(p)

    def Down_module(self, p):
        return self.M_down(p.view(-1, self.in_shape, int(self.out_shape / self.in_shape) * p.size(2)))

    def forward(self, p, m, n):
        p1 = self.Up_module(p, m, n)
        p2 = self.Down_module(p1)
        p_delta = p - p2
        p2 = self.Up_module(p_delta, m, n)
        return p1 + p2


class SaNet(torch.nn.Module):
    meshgrid = [[-0.3, 0.3, 46], [-0.3, 0.3, 46]]
    x = np.linspace(*meshgrid[0])
    y = np.linspace(*meshgrid[1])

    points = torch.tensor(np.meshgrid(x, y), dtype=torch.float32)

    def __init__(self):
        super(SaNet, self).__init__()

        self.sa1_module = SAModule(0.25, 0.2, MLP([3 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.5, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 512]))

        self.skip_attn1 = SkipAttention(MLP([512 + 2, 128]), MLP([256, 128]), MLP([256, 512 + 2]), MLP([512 + 2, 512]))
        self.skip_attn2 = SkipAttention(MLP([256, 64]), MLP([128, 64]), MLP([128, 256]), MLP([256, 256]))

        self.folding1 = FoldingBlock(64, 256, [MLP([512 + 512, 256]), MLP([512 + 512, 256]), MLP([512 + 512, 512 + 512]),
                                               MLP([512 + 512, 512, 256])], [512 + 2, 512], [1024, 512])

        self.folding2 = FoldingBlock(256, 512, [MLP([256 + 256, 64]), MLP([256 + 256, 64]), MLP([256 + 256, 256 + 256]),
                                                MLP([256 + 256, 256, 128])], [256 + 2, 256], [256, 256])
        self.folding3 = FoldingBlock(512, 2048, [MLP([128 + 128, 64]), MLP([128 + 128, 64]), MLP([128 + 128, 128 + 128]),
                                                 MLP([128 + 128, 128])], [128 + 2, 128], [512, 256, 128])

        self.lin = Seq(Lin(128, 64), ReLU(), Lin(64, 3))
    @staticmethod
    def sample_2D(m, n):
        indeces_x = np.round(np.linspace(0, 45, m)).astype(int)
        indeces_y = np.round(np.linspace(0, 45, n)).astype(int)
        x, y = np.meshgrid(indeces_x, indeces_y)
        p = SaNet.points[:, x.ravel(), y.ravel()].T.contiguous()
        return p

    def Encode(self, data):
        sa1_out = self.sa1_module(data.pos, data.pos, data.batch)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        return sa1_out, sa2_out, sa3_out

    def Decode(self, encoded):
        p = SaNet.sample_2D(8, 8)
        out = encoded[2][0].contiguous()
        out = out.view(out.size(0), 1, 1, out.size(-1)).repeat(1, 64, 1, 1)
        out = torch.cat((out, p.view(1, p.size(0), 1, p.size(-1)).repeat(out.size(0), 1, 1, 1)), -1)
        out = self.skip_attn1(out, encoded[1][0].view(out.size(0), 1, 256, encoded[1][0].size(-1)))
        out = self.folding1(out, 16, 16)
        out = out.unsqueeze(-2)
        out = self.skip_attn2(out, encoded[0][0].view(out.size(0), 1, 512, encoded[0][0].size(-1)))
        out = self.folding2(out, 16, 32)
        out = self.folding3(out, 64, 32)

        return self.lin(out)

    def forward(self, data):
        encoded = self.Encode(data)

        decoded = self.Decode(encoded)

        return decoded, encoded
