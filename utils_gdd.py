import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

import ot

from deep_robust_utils import normalize_adj_tensor


def clst_condense_feat(clusters,n_clusters, doc_raw_attrs):
    cluster_condensed_centers = []
    for cluster_id in range(n_clusters):
        doc_in_cluster = np.where(clusters == cluster_id)[0]
        if len(doc_in_cluster) == 0:
            print(f"the cluster is empty, continue")
            continue
        cluster_raw_attrs = doc_raw_attrs[doc_in_cluster] 
        mean_center = np.mean(cluster_raw_attrs, axis=0)
        cluster_condensed_centers.append(mean_center)

    attributes_condensed = torch.tensor(cluster_condensed_centers).to("cuda")
    
    return attributes_condensed

def clst_condense_adj(adj_norm, nnodes_syn, num_docs,clusters,  device):
    
    cluster_labels = clusters 
    cluster_num = cluster_labels.max()+1
    cluster_labels_mat = torch.tensor(np.eye(cluster_num)[cluster_labels]).float().to(device)
    column_sums = cluster_labels_mat.sum(dim=0, keepdim=True)
    cluster_labels_mat = cluster_labels_mat / column_sums

    adj_condense = (cluster_labels_mat.transpose(0,1) @ adj_norm) @ cluster_labels_mat
    adj_condense = (adj_condense - torch.diag(torch.diag(adj_condense))).to_sparse()

    from deep_robust_utils import normalize_adj_tensor 
    adj_condense = normalize_adj_tensor(adj_condense, sparse=True)

    cluster_edge_indices = adj_condense.coalesce().indices()  
    cluster_edge_weights = adj_condense.coalesce().values()

    return adj_condense, cluster_edge_indices, cluster_edge_weights

def clst_condense_label(n_clusters, best_logits, cluster,device):
    cluster_logits_mean = []  
    cluster_class_pred = []   

    for cluster_id in range(n_clusters):
        
        doc_indices_in_cluster = np.where(cluster == cluster_id)[0].tolist()
        
        if len(doc_indices_in_cluster) == 0:
            cluster_logits_mean.append(torch.zeros(best_logits.shape[1], device=device))
            cluster_class_pred.append(torch.tensor(0, dtype=torch.long, device=device))
            continue
        
        cluster_logits = best_logits[doc_indices_in_cluster] 
        mean_logits = torch.mean(cluster_logits, dim=0)  
        cluster_logits_mean.append(mean_logits)
        
        class_pred = torch.argmax(mean_logits)
        cluster_class_pred.append(class_pred)

    cluster_class_pred = torch.stack(cluster_class_pred)
    return cluster_class_pred


def regenerate_cluster_labels(global_cluster_ids, logits, n_global_clusters=20):
    K = n_global_clusters
    feat = logits.clone().detach()
    node_pred_classes = torch.argmax(logits, dim=1).cpu().numpy()
    all_pred_classes = np.unique(node_pred_classes)
    class_total_nodes = {}
    for cls in all_pred_classes:
        class_total_nodes[cls] = len(np.where(node_pred_classes == cls)[0])
    all_groups = []
    for cls in all_pred_classes:
        cls_total = class_total_nodes[cls]
        if cls_total == 0:
            continue
        for g_id in range(n_global_clusters):
            group_nodes = np.where((node_pred_classes == cls) & (global_cluster_ids == g_id))[0]
            group_size = len(group_nodes)
            if group_size == 0:
                continue
            group_ratio = group_size / cls_total
            group_feat = feat[group_nodes].cpu().numpy()
            group_center = np.mean(group_feat, axis=0)
            all_groups.append((cls, g_id, group_size, group_ratio, group_center))
    class_to_groups = defaultdict(list)
    for group in all_groups:
        cls = group[0]
        class_to_groups[cls].append(group)
    for cls in class_to_groups:
        class_to_groups[cls].sort(key=lambda x: (x[3], x[2]), reverse=True)
    core_groups = []
    for cls in all_pred_classes:
        if class_to_groups[cls]:
            core_group = class_to_groups[cls].pop(0)
            core_groups.append(core_group)
    remaining_quota = K - len(core_groups)
    if remaining_quota > 0:
        remaining_groups = []
        for cls in class_to_groups:
            remaining_groups.extend(class_to_groups[cls])
        remaining_groups_sorted = sorted(remaining_groups, key=lambda x: (x[3], x[2]), reverse=True)
        core_groups.extend(remaining_groups_sorted[:remaining_quota])
    select_num = len(core_groups)
    assert select_num == K, f"Core group count={select_num}, not equal to target K={K} (ensure valid pairs >= K)"
    class_to_core_groups = defaultdict(list)
    for cls, g_id, _, _, group_center in core_groups:
        class_to_core_groups[cls].append((g_id, group_center))
    
    final_clusters = np.copy(global_cluster_ids)
    core_group_to_final_id = {}
    current_final_id = 0
    for cls, g_id, _, _, _ in core_groups:
        core_group_key = (cls, g_id)
        if core_group_key in core_group_to_final_id:
            continue
        group_nodes = np.where((node_pred_classes == cls) & (global_cluster_ids == g_id))[0]
        final_clusters[group_nodes] = current_final_id
        core_group_to_final_id[core_group_key] = current_final_id
        current_final_id += 1
    core_nodes = []
    for cls, g_id, _, _, _ in core_groups:
        core_nodes.extend(np.where((node_pred_classes == cls) & (global_cluster_ids == g_id))[0])
    core_nodes = np.unique(core_nodes)
    remain_nodes = np.setdiff1d(np.arange(len(node_pred_classes)), core_nodes)
    print(f"\nTotal selected core groups: {select_num} (target K={K})")
    print(f"Number of remaining samples to merge: {len(remain_nodes)}")
    for node_idx in remain_nodes:
        node_cls = node_pred_classes[node_idx]
        node_feat = feat[node_idx].cpu().numpy()
        cls_core_groups = class_to_core_groups.get(node_cls, [])
        if not cls_core_groups:
            final_clusters[node_idx] = current_final_id
            current_final_id += 1
            continue
        distances = []
        for g_id, group_center in cls_core_groups:
            dist = np.linalg.norm(node_feat - group_center)
            core_group_key = (node_cls, g_id)
            distances.append((dist, core_group_to_final_id[core_group_key]))
        _, nearest_final_id = min(distances, key=lambda x: x[0])
        final_clusters[node_idx] = nearest_final_id
    n_final_clusters = len(np.unique(final_clusters))
    print(f"Final cluster number (Class-Global Cluster pair dimension): {n_final_clusters}")
    return final_clusters, n_final_clusters

def regenerate_cluster_labels(
    global_cluster_ids,
    logits, 
    n_global_clusters=20,  
):

    K=n_global_clusters
    feat = logits.clone().detach()
    
    node_pred_classes = torch.argmax(logits, dim=1).cpu().numpy()

    all_pred_classes = np.unique(node_pred_classes)
    
    class_total_nodes = {}
    for cls in all_pred_classes:
        class_total_nodes[cls] = len(np.where(node_pred_classes == cls)[0])
    
    all_groups = []
    for cls in all_pred_classes:
        cls_total = class_total_nodes[cls]
        if cls_total == 0:
            continue
        
        for g_id in range(n_global_clusters):
            group_nodes = np.where((node_pred_classes == cls) & (global_cluster_ids == g_id))[0]
            group_size = len(group_nodes)
            if group_size == 0:  
                continue
            group_ratio = group_size / cls_total
            group_feat = feat[group_nodes].cpu().numpy()
            group_center = np.mean(group_feat, axis=0)
            
            all_groups.append((cls, g_id, group_size, group_ratio, group_center))
    
    class_to_groups = defaultdict(list)
    for group in all_groups:
        cls = group[0]
        class_to_groups[cls].append(group)

    for cls in class_to_groups:
        class_to_groups[cls].sort(key=lambda x: (x[3], x[2]), reverse=True)
    
    core_groups = []

    for cls in all_pred_classes:
        if class_to_groups[cls]:  
            core_group = class_to_groups[cls].pop(0) 
            core_groups.append(core_group)
    
    remaining_quota = K - len(core_groups)
    if remaining_quota > 0:
        remaining_groups = []
        for cls in class_to_groups:
            remaining_groups.extend(class_to_groups[cls])
        remaining_groups_sorted = sorted(remaining_groups, key=lambda x: (x[3], x[2]), reverse=True)
        core_groups.extend(remaining_groups_sorted[:remaining_quota])
    
    select_num = len(core_groups)
    assert select_num == K, f"Core group count={select_num}, not equal to target K={K} (ensure valid pairs >= K)"
    
    class_to_core_groups = defaultdict(list)
    for cls, g_id, _, _, group_center in core_groups:
        class_to_core_groups[cls].append((g_id, group_center))
    
    final_clusters = np.copy(global_cluster_ids)
    core_group_to_final_id = {}
    current_final_id = 0

    for cls, g_id, _, _, _ in core_groups:
        core_group_key = (cls, g_id)
        if core_group_key in core_group_to_final_id:
            continue
        
        group_nodes = np.where((node_pred_classes == cls) & (global_cluster_ids == g_id))[0]
        final_clusters[group_nodes] = current_final_id
        core_group_to_final_id[core_group_key] = current_final_id
        current_final_id += 1
    
    core_nodes = []
    for cls, g_id, _, _, _ in core_groups:
        core_nodes.extend(np.where((node_pred_classes == cls) & (global_cluster_ids == g_id))[0])
    core_nodes = np.unique(core_nodes)

    remain_nodes = np.setdiff1d(np.arange(len(node_pred_classes)), core_nodes)
    
    print(f"\nTotal selected core groups: {select_num} (target K={K})")
    print(f"Number of remaining samples to merge: {len(remain_nodes)}")
    
    for node_idx in remain_nodes:
        node_cls = node_pred_classes[node_idx]  
        node_feat = feat[node_idx].cpu().numpy()
        
        cls_core_groups = class_to_core_groups.get(node_cls, [])
        if not cls_core_groups:
            final_clusters[node_idx] = current_final_id
            current_final_id += 1
            continue
        
        distances = []
        for g_id, group_center in cls_core_groups:
            dist = np.linalg.norm(node_feat - group_center)
            core_group_key = (node_cls, g_id)
            distances.append((dist, core_group_to_final_id[core_group_key]))
        
        min_dist, nearest_final_id = min(distances, key=lambda x: x[0])
        final_clusters[node_idx] = nearest_final_id
    
    n_final_clusters = len(np.unique(final_clusters))
    print(f"Final cluster number (Class-Global Cluster pair dimension): {n_final_clusters}")
    
    return final_clusters, n_final_clusters



def match_loss(gw_syn, gw_real, args, device):
    dis = torch.tensor(0.0).to(device)

    if args.dis_metric == 'ours':

        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')

    return dis

def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(gwr.shape) == 2:
        gwr = gwr.T
        gws = gws.T
    if len(shape) == 4:
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 1:
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0
    dis = torch.sum(
        1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001)
    )
    return dis

def regularization(adj, x, eig_real=None):
    loss = 0
    loss += feature_smoothing(adj, x)
    return loss

def feature_smoothing(adj, X):
    adj = (adj.t() + adj)/2
    rowsum = adj.sum(1)
    r_inv = rowsum.flatten()
    D = torch.diag(r_inv)
    L = D - adj

    r_inv = r_inv  + 1e-8
    r_inv = r_inv.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    L = r_mat_inv @ L @ r_mat_inv

    XLXT = torch.matmul(torch.matmul(X.t(), L), X)
    loss_smooth_feat = torch.trace(XLXT)
    return loss_smooth_feat


def normalize_adjacency(A: torch.sparse.Tensor):
    """A -> D^{-1/2}(A+I)D^{-1/2}"""
    A = A.coalesce()
    n = A.size(0)
    idx = torch.arange(n, device=A.device)
    I = torch.sparse_coo_tensor(
        torch.stack([idx, idx]),
        torch.ones(n, device=A.device), (n, n)
    ).coalesce()
    A_t = (A + I).coalesce()
    deg = torch.sparse.sum(A_t, dim=1).to_dense()
    deg_inv = deg.pow(-0.5)
    deg_inv[deg_inv == float('inf')] = 0.
    row, col = A_t.indices()
    val = A_t.values() * deg_inv[row] * deg_inv[col]
    return torch.sparse_coo_tensor(A_t.indices(), val, (n, n)).coalesce()

def smooth_twice(A: torch.sparse.Tensor, X: torch.Tensor):
    """Z = A^2 X"""
    hat_A = normalize_adjacency(A)
    Z = torch.sparse.mm(hat_A, X)
    Z = torch.sparse.mm(hat_A, Z)
    return Z

def subsample_sinkhorn(Z: torch.Tensor, Zp: torch.Tensor, sub: int = 8000, eps: float = 0.1):
    Z, Zp = Z.cpu(), Zp.cpu()
    n, m = Z.size(0), Zp.size(0)
    
    n_sub = min(n, sub)
    m_sub = min(m, sub)
    idx  = torch.randperm(n)[:n_sub]
    idyp = torch.randperm(m)[:m_sub]
    M    = torch.cdist(Z[idx], Zp[idyp], p=2) ** 2
    a = torch.ones(n_sub) / n_sub
    b = torch.ones(m_sub) / m_sub
    return sinkhorn_torch(a, b, M, eps, max_iter=30)

import ot 
def sinkhorn_torch(a: torch.Tensor, b: torch.Tensor, M: torch.Tensor, eps: float = 0.1, max_iter: int = 30): 

    align_score = ot.sinkhorn2(a, b, M, reg=eps)

    return float(align_score )

def using_mlp(data,device, signal=False):
    if signal:
        print("using mlp")
        adj_norm = normalize_adj_tensor(data.adj_full,sparse=True).to(device)
        num_nodes = adj_norm.size(0)
        indices = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
        values = torch.ones(num_nodes, device=device)
        adj_norm = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device)
    else:
        adj_norm = normalize_adj_tensor(data.adj_full,sparse=True).to(device)
    
    return adj_norm 


def indices_to_masks(total_nodes, train_indices, val_indices, test_indices):
    
    train_mask = torch.zeros(total_nodes, dtype=torch.bool, device=train_indices.device)
    val_mask = torch.zeros_like(train_mask)
    test_mask = torch.zeros_like(train_mask)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    
    unlabeled_mask = val_mask | test_mask 
    unlabeled_mask = unlabeled_mask & (~train_mask)  
    
    return train_mask, val_mask, test_mask, unlabeled_mask
