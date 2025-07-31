import torch
import numpy as np
from torch import optim
from params import parameter_parser
from model import HCLAMCMI
from MetricsCalculator import MetricsCalculator
from data_pro import Dataset,prepare_data
import warnings
from train3 import evaluate, train
warnings.filterwarnings('ignore')
args = parameter_parser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_cross_validation(opt):
    dataset = prepare_data(opt)
    train_data = Dataset(opt, dataset)

    HIDDEN_LIST = [256, 256]
    NUM_PROJ_HIDDEN = 64

    all_fold_metrics= np.zeros((1, 7))

    for fold_idx in range(opt.validation):
        model = HCLAMCMI(opt.mi_num, opt.circ_num, HIDDEN_LIST,NUM_PROJ_HIDDEN, opt).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        true_score_one, true_score_zero, pre_score_one, pre_score_zero = train(model, train_data[fold_idx],optimizer,opt,fold_idx)

        fold_metrics= evaluate(true_score_one, true_score_zero, pre_score_one, pre_score_zero)

        fold_str = "  ".join([f"{x:.6f}" for x in fold_metrics[0]])

        print(f"[Fold {fold_idx + 1}] Metrics: [{fold_str}]")

        all_fold_metrics += fold_metrics

    metrics_cross_avg = np.round(all_fold_metrics / opt.validation, 5)

    print(f"(AUC, AUPR, F1, Acc, Rec, Spec, Prec):")
    print(f"{metrics_cross_avg}")

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    args = parameter_parser()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_cross_validation(args)

