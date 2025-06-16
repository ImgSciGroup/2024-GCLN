import torch
import torch.nn as nn

from .GraphConv import GraphConvolution
import torch.nn.functional as F

#from .cross import SimCrossLearning1


class RestNetBasicBlock(nn.Module):
    def __init__(self, ):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4,8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        # self.conv3 = nn.Conv2d(8,16, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(16)
        # self.conv4 = nn.Conv2d(16,8, kernel_size=3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(8)
        self.conv5 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(4)
        self.conv6 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(1)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = F.relu(self.bn2(output))
        # output = self.conv3(output)
        # output=F.relu(self.bn3(output))
        # output = self.conv4(output)
        # output = F.relu(self.bn4(output))
        output = self.conv5(output)
        output = F.relu(self.bn5(output))
        output = self.conv6(output)
        output=self.bn6(output)
        return torch.sigmoid(x+output)




class GCN(nn.Module):
    def __init__(self, feat, feat1):
        super(GCN, self).__init__()


        self.avg1 = torch.nn.AdaptiveAvgPool1d(1)
        self.avg2 = torch.nn.AdaptiveAvgPool1d(1)
        self.self_GCN_norm = nn.LayerNorm(feat1)
        self.GCN_u11 = GraphConvolution(feat, feat1)
        self.GCN_u12 = GraphConvolution(feat1, feat1//2)
        self.GCN_u21 = GraphConvolution(feat1 // 2 +feat1, feat1 // 2)
        self.GCN_u22 = GraphConvolution(feat1 // 2, feat1)
        self.liner1 = nn.Sequential(nn.Linear(feat1 // 2 +feat1,feat1))
        # self.self_attention = MultiHeadAttention()

    def forward(self, x, adj):

        x11=torch.sigmoid(self.GCN_u11(x,adj))
        x12=torch.sigmoid(self.GCN_u12(x11,adj))
        x12=torch.cat((x11,x12),dim=1)
        l1 = self.liner1(x12)
        x21=torch.sigmoid(self.GCN_u21(x12,adj))
        x22 =torch.sigmoid( self.GCN_u22(x21, adj))
        x22=torch.sigmoid(x11+x22*l1)
        return x22


class SUGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(SUGCN, self).__init__()

        self.gcn = GCN(nfeat, nhid)

        # self.dropout = nn.Dropout(p=dropout)
        # self.gc3 = GraphConvolution(2 * nhid, nclass)

    def forward(self, x1, adj1,x2,adj2):
        x1=self.gcn(x1,adj1)
        x2 = self.gcn(x2, adj2)
        return x1,x2