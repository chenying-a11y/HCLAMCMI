# import numpy as np
# import torch
# from sklearn.cluster import KMeans
#
#
# def Eu_dis(x):
#
#     x = np.mat(x)
#     aa = np.sum(np.multiply(x, x), 1)
#     ab = x * x.T
#     dist_mat = aa + aa.T - 2 * ab
#     dist_mat[dist_mat < 0] = 0
#     dist_mat = np.sqrt(dist_mat)
#     dist_mat = np.maximum(dist_mat, dist_mat.T)
#     return dist_mat
#
#
# def feature_concat(*F_list, normal_col=False):
#
#     features = None
#     for f in F_list:
#         if f is not None and f.size > 0:
#             if len(f.shape) > 2:
#                 f = f.reshape(-1, f.shape[-1])
#             if normal_col:
#                 f_max = np.max(np.abs(f), axis=0)
#                 f_max[f_max == 0] = 1
#                 f = f / f_max
#             if features is None:
#                 features = f
#             else:
#                 features = np.hstack((features, f))
#     if normal_col and features is not None:
#         features_max = np.max(np.abs(features), axis=0)
#         features_max[features_max == 0] = 1
#         features = features / features_max
#     return features
#
#
# def hyperedge_concat(*H_list):
#
#     H = None
#     for h in H_list:
#         if h is not None and len(h) != 0:
#             if H is None:
#                 H = h
#             else:
#                 if not isinstance(h, list):
#                     H = np.hstack((H, h))
#                 else:
#                     tmp = [np.hstack((a, b)) for a, b in zip(H, h)]
#                     H = tmp
#     return H
#
#
# def _generate_G_from_H(H, variable_weight=False):
#
#     H = np.array(H)
#     n_edge = H.shape[1]
#
#     W = np.ones(n_edge)
#
#     DV = np.sum(H * W, axis=1)
#     DE = np.sum(H, axis=0)
#
#     DE[DE == 0] = 1.0
#
#     invDE = np.mat(np.diag(np.power(DE, -1)))
#     DV2 = np.mat(np.diag(np.power(DV, -0.5)))
#     DV2[np.isinf(DV2)] = 0
#
#     W_mat = np.mat(np.diag(W))
#     H_mat = np.mat(H)
#     HT_mat = H_mat.T
#
#     if variable_weight:
#         DV2_H = DV2 * H_mat
#         invDE_HT_DV2 = invDE * HT_mat * DV2
#         return DV2_H, W_mat, invDE_HT_DV2
#     else:
#         G = DV2 * H_mat * W_mat * invDE * HT_mat * DV2
#         return torch.Tensor(G)
#
#
# def generate_G_from_H(H, variable_weight=False):
#
#     if not isinstance(H, list):
#         return _generate_G_from_H(H, variable_weight)
#     else:
#         return [_generate_G_from_H(sub_H, variable_weight) for sub_H in H]
import numpy as np
import torch
from sklearn.cluster import KMeans


def Eu_dis(x):

    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def feature_concat(*F_list, normal_col=False):

    features = None
    for f in F_list:
        if f is not None and f.size > 0:
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f_max[f_max == 0] = 1
                f = f / f_max
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col and features is not None:
        features_max = np.max(np.abs(features), axis=0)
        features_max[features_max == 0] = 1
        features = features / features_max
    return features


def hyperedge_concat(*H_list):

    H = None
    for h in H_list:
        if h is not None and len(h) != 0:
            if H is None:
                H = h
            else:
                if not isinstance(h, list):
                    H = np.hstack((H, h))
                else:
                    tmp = [np.hstack((a, b)) for a, b in zip(H, h)]
                    H = tmp
    return H


def _generate_G_from_H(H, variable_weight=False):

    H = np.array(H)
    n_edge = H.shape[1]

    W = np.ones(n_edge)

    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)

    DE[DE == 0] = 1.0

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    DV2[np.isinf(DV2)] = 0

    W_mat = np.mat(np.diag(W))
    H_mat = np.mat(H)
    HT_mat = H_mat.T

    if variable_weight:
        DV2_H = DV2 * H_mat
        invDE_HT_DV2 = invDE * HT_mat * DV2
        return DV2_H, W_mat, invDE_HT_DV2
    else:
        G = DV2 * H_mat * W_mat * invDE * HT_mat * DV2
        return torch.Tensor(G)


def generate_G_from_H(H, variable_weight=False):

    if not isinstance(H, list):
        return _generate_G_from_H(H, variable_weight)
    else:
        return [_generate_G_from_H(sub_H, variable_weight) for sub_H in H]

def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=False, m_prob=1):
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))

    for center_idx in range(n_obj):
        dis_vec = dis_mat[center_idx, :]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()

        if center_idx not in nearest_idx[:k_neig]:
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                avg_dis = np.average(dis_vec)

                if avg_dis == 0:
                    H[node_idx, center_idx] = 1.0
                else:

                    H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs, split_diff_scale=False, is_probH=False, m_prob=1):
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if isinstance(K_neigs, int):
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H_list = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        H_list.append(H_tmp)

    if split_diff_scale:
        return H_list
    else:
        return hyperedge_concat(*H_list)


def _construct_edge_list_from_cluster(X, n_clusters):
    N = X.shape[0]
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0, n_init=10).fit(X)
    assignment = kmeans.labels_

    H = np.zeros([N, n_clusters])
    for i in range(N):
        H[i, assignment[i]] = 1

    return H


def construct_H_with_Kmeans(X, clusters, split_diff_scale=False):
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if isinstance(clusters, int):
        clusters = [clusters]

    H_list = []
    for n_clus in clusters:
        if n_clus > X.shape[0]:
            print(f"Warning: Number of clusters ({n_clus}) exceeds number of samples ({X.shape[0]}). Skipping.")
            continue
        H_tmp = _construct_edge_list_from_cluster(X, n_clus)
        H_list.append(H_tmp)

    if not H_list:
        return None

    if split_diff_scale:
        return H_list
    else:
        return hyperedge_concat(*H_list)


def constructHW_knn(X, K_neigs, is_probH):

    H = construct_H_with_KNN(X, K_neigs, is_probH=is_probH)

    G = _generate_G_from_H(H)

    return G


def constructHW_kmean(X, clusters):

    H = construct_H_with_Kmeans(X, clusters)
    if H is None:
        raise ValueError("Failed to construct hypergraph with K-Means. Check cluster parameters.")

    G = _generate_G_from_H(H)

    return G

