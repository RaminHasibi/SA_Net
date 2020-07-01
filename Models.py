import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
import numpy as np
import itertools

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
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Attention(torch.nn.Module):
    def __init__(self, NN_h, NN_l, NN_g):
        super(Attention, self).__init__()
        self.M_h = NN_h
        self.M_l = NN_l
        self.M_g = NN_g
    def forward(self, p, r):
        attn_weights = F.softmax(
            torch.bmm(self.M_h(p).unsqueeze(0), self.M_l(r).unsqueeze(0).transpose(1,2)), dim=1)
        return p + torch.bmm(attn_weights,self.M_g(p))


class FoldingBlock(torch.nn.Module):
    def __init__(self, input, output,attentions, NN_up, NN_down):

        super(FoldingBlock, self).__init__()
        self.in_shape = input
        self.out_shape = output
        self.self_attn = Attention(*attentions)
        self.M_up = MLP(*NN_up)
        self.M_down = MLP(*NN_down)
        self.self_attn2 = Attention(*attentions)
        self.M_up2 = MLP(*NN_up)
        self.M_down2 = MLP(*NN_down)
    def Up_module(self,p, k):

        p = p.repeat(1, self.out_shape/self.in_shape, 1)
        meshgrid = [[-0.3, 0.3, 45], [-0.3, 0.3, 45]]
        x = np.linspace(*meshgrid[0])
        y = np.linspace(*meshgrid[1])
        points = torch.tensor(np.array(list(itertools.product(x, y))))

        indeces = torch.linspace(0, 45, k)
        p_2d = torch.cat([p, points[indeces,indeces]], -1)
        p_2d = self.M_up(p_2d)
        p = torch.cat([p, p_2d], -1)
        return self.self_attn(p,p)

    def Down_module(self,p):
        return self.M_down(p.view(-1, self.in_shape, -1))

    def forward(self,p, k):
        p1 = self.Up_module(p, k)
        p2 = self.Down_module(p1)
        p2 = self.Up_module(p2, k)
        return p1 + p2


class SA_Net(torch.nn.Module):
    def __init__(self):
        super(SA_Net, self).__init__()

        self.sa1_module = SAModule(0.2, 0.2, MLP([3 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.5, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 512]))

        self.skip_attn1 = Attention(MLP([512 + 3, 512]), MLP([256, 512]), MLP([512 + 3, 512]))
        self.skip_attn2 = Attention(MLP([512 + 3, 256]), MLP([256, 256]), MLP([512 + 3, 256]))

        self.folding1 = FoldingBlock(64, 256, [MLP[512 + 3, 512], MLP([256,512], MLP[512 + 3, 512])] , [256, 64], [64 , 64])
        self.folding2 = FoldingBlock(256, 512, [MLP[512 + 3, 512], MLP([256,512], MLP[512 + 3, 512])] , [256, 64], [64 , 64])
        self.folding3 = FoldingBlock(512, 2048, [MLP[512 + 3, 512], MLP([256,512], MLP[512 + 3, 512])] , [256, 64], [64 , 64])

        self.meshgrid = [[-0.3, 0.3, 45], [-0.3, 0.3, 45]]

    def Encode(self,data):
        sa1_out = self.sa1_module(data.pos, data.pos, data.batch)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        return sa1_out,sa2_out,sa3_out


    def Decode(self,encoded):
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        points = torch.tensor(np.array(list(itertools.product(x, y))))
        indeces = torch.linspace(0, 45, 64)
        out = torch.cat((encoded[3][0],points[indeces,indeces]),1)
        out = self.skip_attn1(out,encoded[1][0])
        out = self.folding1(out,256)
        out = self.skip_attn2(out,encoded[0][0])
        out = self.folding3(out,512)

        return out
    def forward(self, data):

        encoded = self.Encode(data)

        decoded = self.Decode(encoded)


        return decoded, encoded
