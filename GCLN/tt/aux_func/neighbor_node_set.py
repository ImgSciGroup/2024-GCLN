import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cdist

def getnew_node_set(node_set_t1,node_set_t2,obj_nums):
    new_node_set_t1 = []
    new_node_set_t2 = []
    for i in range(obj_nums-1):
        original_matrix_t1=node_set_t1[i]
        original_matrix_t2=node_set_t2[i]
        h,w=original_matrix_t1.shape
        # 计算欧式距离矩阵
        K=int(np.sqrt(h))

        distances_t1 = pairwise_distances(original_matrix_t1)
        distances_t2 = pairwise_distances(original_matrix_t2)
        # 找到每个像素点最相近的点的索引
        top_indices_t1 = np.argsort(distances_t1, axis=1)[:, 0:K]
        top_indices_t2 = np.argsort(distances_t2, axis=1)[:, 0:K]
        # 计算邻域节点的均值
        new_matrix_t1=np.zeros([h,3])
        new_matrix_t2 = np.zeros([h, 3])
        for j in range(h):
            sum_r_t1=0.0
            sum_g_t1=0.0
            sum_b_t1=0.0
            sum_r_t2 = 0.0
            sum_g_t2 = 0.0
            sum_b_t2 = 0.0
            for k in range(0,K):
                sum_r_t1+=original_matrix_t1[top_indices_t1[j,k],0]
                sum_g_t1 += original_matrix_t1[top_indices_t1[j, k], 1]
                sum_b_t1 += original_matrix_t1[top_indices_t1[j, k], 2]

                sum_r_t2 += original_matrix_t2[top_indices_t2[j, k], 0]
                sum_g_t2 += original_matrix_t2[top_indices_t2[j, k], 1]
                sum_b_t2 += original_matrix_t2[top_indices_t2[j, k], 2]
            new_matrix_t1[j,0]=sum_r_t1/(K)#+original_matrix_t1[j,0]*0.9
            new_matrix_t1[j, 1] = sum_g_t1 / (K)#+original_matrix_t1[j,1]*0.9
            new_matrix_t1[j, 2] = sum_b_t1 / (K)#+original_matrix_t1[j,2]*0.9

            new_matrix_t2[j, 0] = sum_r_t2 /(K)#+original_matrix_t2[j,0]*0.9
            new_matrix_t2[j, 1] = sum_g_t2 / (K)#+original_matrix_t2[j,1]*0.9
            new_matrix_t2[j, 2] = sum_b_t2 / (K)#+original_matrix_t2[j,2]*0.9
        new_matrix_t1=new_matrix_t1.astype('float32')
        new_matrix_t2 = new_matrix_t2.astype('float32')
        new_node_set_t1.append(new_matrix_t1)
        new_node_set_t2.append(new_matrix_t2)
    return new_node_set_t1,new_node_set_t2


