import argparse
import time
import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


def ContrastiveLoss(adj1,feature1):
    _,index=torch.sort(adj1,1)
    N=adj1.size()[0]
    K=int(np.sqrt(N))
    loss_sum=torch.tensor(0.0).to('cpu')

    for i in range(0,N):
        target_node1=feature1[i,:]
        negative_nodes1 = feature1[index[i, 1:K + 1], :]
        positive_nodes1 = feature1[index[i, N - K:N], :]
        pos_distance1 = (torch.matmul(positive_nodes1,target_node1 ))
        neg_distance1 = (torch.matmul(negative_nodes1,target_node1 ))
        #max=torch.max(torch.matmul(positive_nodes1,target_node1 ))
        margin=torch.max(torch.matmul(positive_nodes1,target_node1 ))


        loss = torch.mean(torch.relu(neg_distance1 - pos_distance1 +margin))

        loss_sum += loss

    return loss_sum


