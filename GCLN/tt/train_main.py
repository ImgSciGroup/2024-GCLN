import argparse
import time
import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from skimage.segmentation import slic

from aux_func.acc_ass import assess_accuracy
from aux_func.clustering import otsu
from aux_func.graph_func import construct_affinity_matrix
from aux_func.graph_func import construct_affinity_matrix1
from aux_func.preprocess import preprocess_img
from model.SUGCNmodel import SUGCN
from aux_func.neighbor_node_set import getnew_node_set

from aux_func.loss import ContrastiveLoss


def load_checkpoint_for_evaluation(model, checkpoint):
    #saved_state_dict = torch.load(checkpoint, map_location='cuda:0')
    saved_state_dict = torch.load(checkpoint, map_location ='cpu')

    model.load_state_dict(saved_state_dict)
    model.cpu()
    #model.cuda()
    model.eval()


def train_model(args):
    data=""
    type=''
    if data=="Te":
        type="tif"
    else:
        type="png"


    img_t1 = imageio.imread('./data/'+data+'/T1.'+type)  # .astype(np.float32)
    img_t2 = imageio.imread('./data/'+data+'/T2.'+type)  # .astype(np.float32)
    ground_truth_changed = imageio.imread('./data/'+data+'/gt.'+type)
    if data=="uk" or data=="is"or data=="It":
        ground_truth_changed=ground_truth_changed[:,:,0]

    ground_truth_unchanged = 255 - ground_truth_changed

    height, width, channel_t1 = img_t1.shape
    _, _, channel_t2 = img_t2.shape

    #objects记录slic分割后的超像素 编号从1开始



    objects = slic(img_t2 , n_segments=args.n_seg, compactness=args.cmp)

    img_t1 = preprocess_img(img_t1, d_type='opt', norm_type='1')
    img_t2 = preprocess_img(img_t2, d_type='opt', norm_type='1')
    # objects = np.load('./object_idx.npy')
    obj_nums = np.max(objects) + 1
    min=np.min(objects)
    #将每个超像素块的内部像素值存储到node_set中
    node_set_t1 = []
    node_set_t2 = []
    for idx in range(1,obj_nums):
        obj_idx = objects == idx
        node_set_t1.append(img_t1[obj_idx])
        node_set_t2.append(img_t2[obj_idx])
    new_node_set_t1,new_node_set_t2=getnew_node_set(node_set_t1,node_set_t2,obj_nums)
    #am_set_t1 = construct_affinity_matrix(img_t1, objects, args.band_width_t1)
    #am_set_t2 = construct_affinity_matrix(img_t2, objects, args.band_width_t2)
    new_am_set_t1 = construct_affinity_matrix1(new_node_set_t1, objects, args.band_width_t1)
    new_am_set_t2 = construct_affinity_matrix1(new_node_set_t2, objects, args.band_width_t2)
    print("preprocess done")
    GCLN_model = SUGCN(nfeat=3, nhid=16, nclass=3, dropout=0.5)
    optimizer = optim.AdamW(GCLN_model.parameters(), lr=1e-4, weight_decay=1e-6)
    GCLN_model.cpu()
    #GCLN_model.cuda()
    GCLN_model.train()

    if 1:
        # Edge information reconstruction
        for _epoch in range(args.epoch):  # args.epoch-
            for _iter in range(obj_nums - 1):
                node_t1 = new_node_set_t1[_iter]  # np.expand_dims(node_set_t1[_iter], axis=0)
                adj_t1, norm_adj_t1 = new_am_set_t1[_iter]  # np.expand_dims(am_set_t1[_iter], axis=0)
                node_t1 = torch.from_numpy(node_t1).cpu().float()
                adj_t1 = torch.from_numpy(adj_t1).cpu().float()
                norm_adj_t1 = torch.from_numpy(norm_adj_t1).cpu().float()

                node_t2 = new_node_set_t2[_iter]  # np.expand_dims(node_set_t2[_iter], axis=0)
                adj_t2, norm_adj_t2 = new_am_set_t2[_iter]  # np.expand_dims(am_set_t2[_iter], axis=0)
                node_t2 = torch.from_numpy(node_t2).cpu().float()
                adj_t2 = torch.from_numpy(adj_t2).cpu().float()
                norm_adj_t2 = torch.from_numpy(norm_adj_t2).cpu().float()

                feat_t1, feat_t2 = GCLN_model(node_t1, norm_adj_t1, node_t2, norm_adj_t2)

                cnstr_loss_t1 =  ContrastiveLoss(adj_t1,feat_t1).to('cpu')
                cnstr_loss_t2 =  ContrastiveLoss(adj_t2,feat_t2).to('cpu')

                total_loss=cnstr_loss_t1+cnstr_loss_t2

                ttl_loss =total_loss

                ttl_loss.backward()
                optimizer.step()
                if (_iter + 1) % 10 == 0:
                    print(f'Epoch is {_epoch + 1}, iter is {_iter}, total loss is {total_loss.item()}')
        # torch.save(GCAE_model.state_dict(), './model_weight/' + str(time.time()) + '.pth')
        torch.save(GCLN_model.state_dict(), './model_weight_'+data+'/'+data+'_'+str(args.epoch)+'.pth')


        #Extracting deep edge representations & Change information mapping
        #Load pretrained weight
    restore_from = './model_weight_'+data+'/'+data+'_'+str(args.epoch)+'.pth'
    load_checkpoint_for_evaluation(GCLN_model, restore_from)
    GCLN_model.eval()
    diff_set_gcn_div = []
    diff_set_gcn = []



    for _iter in range(obj_nums-1):
        node_t1 = new_node_set_t1[_iter]  # np.expand_dims(node_set_t1[_iter], axis=0)
        node_t2 = new_node_set_t2[_iter]  # np.expand_dims(node_set_t2[_iter], axis=0)
        n=node_t1.shape[0]
        # node_t1 = node_set_t1[_iter]
        # node_t2 = node_set_t2[_iter]
        adj_t1, norm_adj_t1 = new_am_set_t1[_iter]  # np.expand_dims(am_set_t1[_iter], axis=0)
        adj_t2, norm_adj_t2 = new_am_set_t2[_iter]  # np.expand_dims(am_set_t2[_iter], axis=0)

        node_t1 = torch.from_numpy(node_t1).cpu().float()
        node_t2 = torch.from_numpy(node_t2).cpu().float()
        norm_adj_t1 = torch.from_numpy(norm_adj_t1).cpu().float()
        norm_adj_t2 = torch.from_numpy(norm_adj_t2).cpu().float()

        feat_t1,feat_t2 = GCLN_model(node_t1, norm_adj_t1,node_t2,norm_adj_t2)
        div=0.1

        N,D=feat_t1.size()
        d1=int(N*div)
        d2=int(N*(1-div))
        diff=torch.abs(feat_t1-feat_t2)
        diff=torch.mean(diff,1,True)
        sort_f,index_diff=torch.sort(diff,0)

        diff_set_gcn.append((torch.mean(torch.abs(feat_t1- feat_t2)).data.cpu().numpy()))

        diff_set_gcn_div.append((torch.mean(torch.abs(sort_f[d1:d2])).data.cpu().numpy()))




    diff_map_gcn_div = np.zeros((height, width))
    diff_map_gcn = np.zeros((height, width))
    for i in range(0, obj_nums-1):
        diff_map_gcn_div[objects == i+1] = diff_set_gcn_div[i]
        diff_map_gcn[objects == i + 1] = diff_set_gcn[i]
    # cnn
    diff_map_gcn_div = np.reshape(diff_map_gcn_div, (height * width, 1))

    threshold = otsu(diff_map_gcn_div)
    diff_map_cnn = np.reshape(diff_map_gcn_div, (height, width))

    bcm = np.zeros((height, width)).astype(np.uint8)
    bcm[diff_map_cnn > threshold] = 255
    bcm[diff_map_cnn <= threshold] = 0

    conf_mat, oa, f1, kappa_co = assess_accuracy(ground_truth_changed, ground_truth_unchanged, bcm)

    #imageio.imsave('./result_'+data+'/'+data+'_'+str(args.epoch)+'_'+str(args.n_seg)+'_'+str(int(div*100))+'_gcn_div.png', bcm)
    diff_map_cnn = 255 * (diff_map_gcn_div - np.min(diff_map_gcn_div)) / (np.max(diff_map_gcn_div) - np.min(diff_map_gcn_div))

    #imageio.imsave('./result_'+data+'/'+data+'_'+str(args.epoch)+'_'+str(args.n_seg)+'_'+str((div*100))+'_gcn_div_DI.png', diff_map_cnn.astype(np.uint8))
    print('gcn_div')
    print(conf_mat)
    print(oa)
    print(f1)
    print(kappa_co)
    #gcn
    diff_map_gcn = np.reshape(diff_map_gcn, (height * width, 1))

    threshold = otsu(diff_map_gcn)
    diff_map_gcn = np.reshape(diff_map_gcn, (height, width))

    bcm = np.zeros((height, width)).astype(np.uint8)
    bcm[diff_map_gcn > threshold] = 255
    bcm[diff_map_gcn <= threshold] = 0

    conf_mat, oa, f1, kappa_co = assess_accuracy(ground_truth_changed, ground_truth_unchanged, bcm)

    imageio.imsave('./result_'+data+'/'+data +'_'+str(args.n_seg)+'_'+str(args.cmp)+'_'+str(args.band_width_t1)+'_gcn.png', bcm)
    diff_map_gcn = 255 * (diff_map_gcn - np.min(diff_map_gcn)) / (np.max(diff_map_gcn) - np.min(diff_map_gcn))

    imageio.imsave('./result_'+data+'/'+data+'_'+str(args.n_seg)+'_'+str(args.cmp)+'_'+str(div)+'_'+str(args.band_width_t1)+'_gcn_DI.png',diff_map_gcn.astype(np.uint8))
    print('gcn')
    print(conf_mat)
    print(oa)
    print(f1)
    print(kappa_co)











if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detecting land-cover changes on uk1 dataset")
    parser.add_argument('--n_seg', type=int, default=2500,
                        help='Approximate number of objects obtained by the segmentation algorithm')
    parser.add_argument('--cmp', type=int, default=20
                        , help='Compectness of the obtained objects')
    parser.add_argument('--band_width_t1', type=float, default=0.1,
                        help='The bandwidth of the Gauss0ian kernel when calculating the adjacency matrix')
    parser.add_argument('--band_width_t2', type=float, default=0.1,
                        help='The bandwidth of the Gaussian kernel when calculating the adjacency matrix')
    parser.add_argument('--epoch', type=int, default=1,
                        help='Training epoch of GCLN')
    args = parser.parse_args()

    train_model(args)




