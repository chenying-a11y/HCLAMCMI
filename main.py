import pandas as pd
import torch
import numpy as np
from torch import optim
from params import parameter_parser
from model import HCLAMCMI
from components import get_L2reg, Myloss
from Calculate_Metrics import Metric_fun
from data_pro import Dataset,prepare_data
import ConstructHW
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, train_data, optim, opt, test_data=None):
    model.train()
    regression_crit = Myloss()
    one_index = train_data[2][0].to(device).t().tolist()
    zero_index = train_data[2][1].to(device).t().tolist()
    circ_sim_integrate_tensor = train_data[0].to(device)
    mi_sim_integrate_tensor = train_data[1].to(device)
    mi_original_features = train_data[4].numpy()
    circ_original_features = train_data[4].numpy().T

    if test_data is not None:
        test_data_modified = test_data.copy()
        test_data_modified[4][:, :] = 0

        combined_mi_features = torch.from_numpy(np.vstack((mi_original_features, test_data_modified[4].numpy()))).to(device)
        combined_circ_features = torch.from_numpy(np.vstack((circ_original_features, test_data_modified[4].numpy().T))).to(device)
    else:
        combined_mi_features = torch.from_numpy(mi_original_features).to(device)
        combined_circ_features = torch.from_numpy(circ_original_features).to(device)

    mi_original_features = combined_mi_features
    G_mi_original_Kn = ConstructHW.constructHW_knn(mi_original_features.cpu().numpy(), K_neigs=[14], is_probH=False)
    G_mi_original_Km = ConstructHW.constructHW_kmean(mi_original_features.cpu().numpy(), clusters=[9])
    G_mi_original_Kn = G_mi_original_Kn.to(device)
    G_mi_original_Km = G_mi_original_Km.to(device)

    circ_original_features = combined_circ_features
    G_circ_original_Kn = ConstructHW.constructHW_knn(circ_original_features.cpu().numpy(), K_neigs=[14], is_probH=False)
    G_circ_original_Km = ConstructHW.constructHW_kmean(circ_original_features.cpu().numpy(), clusters=[9])
    G_circ_original_Kn = G_circ_original_Kn.to(device)
    G_circ_original_Km = G_circ_original_Km.to(device)

    mi_sim_features = mi_sim_integrate_tensor.detach().cpu().numpy()
    mi_sim_features = torch.from_numpy(mi_sim_features).to(device)
    G_mi_sim_Kn = ConstructHW.constructHW_knn(mi_sim_features.cpu().numpy(), K_neigs=[14], is_probH=False)
    G_mi_sim_Km = ConstructHW.constructHW_kmean(mi_sim_features.cpu().numpy(), clusters=[9])
    G_mi_sim_Kn = G_mi_sim_Kn.to(device)
    G_mi_sim_Km = G_mi_sim_Km.to(device)

    circ_sim_features = circ_sim_integrate_tensor.detach().cpu().numpy()
    circ_sim_features = torch.from_numpy(circ_sim_features).to(device)
    G_circ_sim_Kn = ConstructHW.constructHW_knn(circ_sim_features.cpu().numpy(), K_neigs=[14], is_probH=False)
    G_circ_sim_Km = ConstructHW.constructHW_kmean(circ_sim_features.cpu().numpy(), clusters=[9])
    G_circ_sim_Kn = G_circ_sim_Kn.to(device)
    G_circ_sim_Km = G_circ_sim_Km.to(device)

    miRNA_node2vec = pd.read_csv('data/mirna_node2vec.csv', header=0, index_col=0).values
    circRNA_node2vec = pd.read_csv('data/circrna_node2vec.csv', header=0, index_col=0).values

    miRNA_node2vec_tensor = torch.tensor(miRNA_node2vec).to(device)
    circRNA_node2vec_tensor = torch.tensor(circRNA_node2vec).to(device)

    G_mi_Kn_new = ConstructHW.constructHW_knn(miRNA_node2vec, K_neigs=[14], is_probH=False)
    G_mi_Km_new = ConstructHW.constructHW_kmean(miRNA_node2vec, clusters=[9])
    G_mi_Kn_new = G_mi_Kn_new.to(device)
    G_mi_Km_new = G_mi_Km_new.to(device)

    G_circ_Kn_new = ConstructHW.constructHW_knn(circRNA_node2vec, K_neigs=[14], is_probH=False)
    G_circ_Km_new = ConstructHW.constructHW_kmean(circRNA_node2vec, clusters=[9])
    G_circ_Kn_new = G_circ_Kn_new.to(device)
    G_circ_Km_new = G_circ_Km_new.to(device)

    for epoch in range(1, opt.epoch + 1):
        score, mi_cl_loss, circ_cl_loss = model(
            mi_original_features, circ_original_features, mi_sim_features, circ_sim_features,
            G_mi_original_Kn, G_mi_original_Km, G_circ_original_Kn, G_circ_original_Km,
            G_mi_sim_Kn, G_mi_sim_Km, G_circ_sim_Kn, G_circ_sim_Km,
            G_mi_Kn_new, G_mi_Km_new, G_circ_Kn_new, G_circ_Km_new,miRNA_node2vec_tensor,circRNA_node2vec_tensor
        )

        recover_loss = regression_crit(one_index, zero_index, train_data[4].to(device), score)
        reg_loss = get_L2reg(model.parameters())
        tol_loss = recover_loss + mi_cl_loss + circ_cl_loss + 0.00001 * reg_loss
        optim.zero_grad()
        tol_loss.backward()
        optim.step()

    true_value_one, true_value_zero, pre_value_one, pre_value_zero = test(
        model, train_data,
        mi_original_features, circ_original_features, mi_sim_features, circ_sim_features,
        G_mi_original_Kn, G_mi_original_Km, G_circ_original_Kn, G_circ_original_Km,
        G_mi_sim_Kn, G_mi_sim_Km, G_circ_sim_Kn, G_circ_sim_Km,
        G_mi_Kn_new, G_mi_Km_new, G_circ_Kn_new, G_circ_Km_new,miRNA_node2vec_tensor,circRNA_node2vec_tensor)

    return true_value_one, true_value_zero, pre_value_one, pre_value_zero


def test(model, data,mi_original_features, circ_original_features, mi_sim_features,circ_sim_features,
         G_mi_original_Kn, G_mi_original_Km, G_circ_original_Kn, G_circ_original_Km,
        G_mi_sim_Kn, G_mi_sim_Km, G_circ_sim_Kn, G_circ_sim_Km, G_mi_Kn_new, G_mi_Km_new, G_circ_Kn_new, G_circ_Km_new,
         miRNA_node2vec_tensor,circRNA_node2vec_tensor):

    model.eval()
    score,_,_ = model(mi_original_features, circ_original_features, mi_sim_features,circ_sim_features,
                     G_mi_original_Kn, G_mi_original_Km, G_circ_original_Kn, G_circ_original_Km,
                     G_mi_sim_Kn, G_mi_sim_Km, G_circ_sim_Kn, G_circ_sim_Km, G_mi_Kn_new, G_mi_Km_new, G_circ_Kn_new, G_circ_Km_new,
                      miRNA_node2vec_tensor,circRNA_node2vec_tensor)

    test_one_index = data[3][0].t().tolist()
    test_zero_index = data[3][1].t().tolist()
    true_one = data[5][test_one_index]
    true_zero = data[5][test_zero_index]

    pre_one = score[test_one_index]
    pre_zero = score[test_zero_index]

    return true_one, true_zero, pre_one, pre_zero


def evaluate(true_one, true_zero, pre_one, pre_zero):
    Metric = Metric_fun()
    metrics_tensor = np.zeros((1, 7))

    for seed in range(10):
        test_po_num = true_one.shape[0]
        test_index = np.array(np.where(true_zero == 0))
        np.random.seed(seed)
        np.random.shuffle(test_index.T)
        test_ne_index = tuple(test_index[:, :test_po_num])
        eval_true_zero = true_zero[test_ne_index]
        eval_true_data = torch.cat([true_one,eval_true_zero])
        eval_pre_zero = pre_zero[test_ne_index]
        eval_pre_data = torch.cat([pre_one,eval_pre_zero])
        metrics_tensor = metrics_tensor + Metric.cv_mat_model_evaluate(eval_true_data, eval_pre_data)

    metrics_tensor_avg = metrics_tensor / 10
    return metrics_tensor_avg

def main(opt):
    dataset = prepare_data(opt)
    train_data = Dataset(opt, dataset)
    hidden_list = [256, 256]
    num_proj_hidden = 64
    metrics_cross = np.zeros((1, 7))
    for fold_idx in range(opt.validation):
        model = HCLAMCMI(opt.mi_num, opt.circ_num, hidden_list, num_proj_hidden, opt).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        true_score_one, true_score_zero, pre_score_one, pre_score_zero = train_epoch(model, train_data[fold_idx],optimizer, opt)
        metrics_value = evaluate(true_score_one, true_score_zero, pre_score_one, pre_score_zero)
        print(f"[Fold {fold_idx + 1}] Metrics: {metrics_value}")
        metrics_cross += metrics_value
    metrics_cross_avg = np.round(metrics_cross / opt.validation, 5)
    print(f"\nCross-Validation Average Metrics: {metrics_cross_avg}")


if __name__ == '__main__':
    args = parameter_parser()
    main(args)

