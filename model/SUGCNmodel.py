import torch
import torch.nn as nn

from .GraphConv import GraphConvolution
import torch.nn.functional as F





class GCN(nn.Module):
    def __init__(self, feat, feat1):
        super(GCN, self).__init__()


        self.avg1 = torch.nn.AdaptiveAvgPool1d(1)
        self.GCN_u11 = GraphConvolution(feat, feat1)
        self.GCN_u12 = GraphConvolution(feat1, feat1//2)
        self.GCN_u21 = GraphConvolution(feat1 // 2 +feat1, feat1 // 2)
        self.GCN_u22 = GraphConvolution(feat1 // 2, feat1)
        self.liner1 = nn.Sequential(nn.Linear(feat1 // 2 +feat1,feat1))


    def forward(self, x, adj):

        x11=torch.sigmoid(self.GCN_u11(x,adj))
        x12=torch.sigmoid(self.GCN_u12(x11,adj))
        x12=torch.cat((x11,x12),dim=1)
        l1 = self.liner1(x12)
        l1 = self.avg1(l1.unsqueeze(-1)).squeeze(-1)
        x21=torch.sigmoid(self.GCN_u21(x12,adj))
        x22 =torch.sigmoid( self.GCN_u22(x21, adj))
        x22=torch.sigmoid(x11+x22*l1)
        return x22


class SUGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SUGCN, self).__init__()

        self.gcn = GCN(nfeat, nhid)



    def forward(self, x1, adj1,x2,adj2):
        x1=self.gcn(x1,adj1)
        x2 = self.gcn(x2, adj2)
        return x1,x2