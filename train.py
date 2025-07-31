import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.optim import Optimizer
from Myloss import Myloss
from params import parameter_parser
from MetricsCalculator import MetricsCalculator
import hypergraph_constructor
import warnings
from typing import List, Tuple, Optional

warnings.filterwarnings('ignore')

args = parameter_parser()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVALUATION_SEEDS = 10

class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, one_index, zero_index, input, target):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)

        return (1 - args.alpha) * loss_sum[one_index].sum() + args.alpha * loss_sum[zero_index].sum()

class FoldTrainer:

    def __init__(self, model: torch.nn.Module, optimizer: Optimizer, args):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.device = device
        self.knn_k = [13]
        self.kmeans_clusters = [11]
        self.model_inputs = {}

    def _build_hypergraphs(self, features_numpy: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        g_knn = hypergraph_constructor.constructHW_knn(features_numpy, K_neigs=self.knn_k, is_probH=False)
        g_kmeans =hypergraph_constructor.constructHW_kmean(features_numpy, clusters=self.kmeans_clusters)
        return g_knn.to(self.device), g_kmeans.to(self.device)

    def _prepare_inputs(self, train_data: list, test_data: Optional[list] = None) -> None:
        circ_sim_features = train_data[0].to(self.device).detach()
        mi_sim_features = train_data[1].to(self.device).detach()
        assoc_matrix_numpy = train_data[4].numpy()

        mi_original_numpy = assoc_matrix_numpy
        circ_original_numpy = assoc_matrix_numpy.T
        if test_data:
            test_mask = test_data[4].numpy() * 0
            mi_original_numpy = np.vstack((mi_original_numpy, test_mask))
            circ_original_numpy = np.vstack((circ_original_numpy, test_mask.T))

        mi_n2v_numpy = pd.read_csv('data/mirna_node2vec.csv', header=0, index_col=0).values
        circ_n2v_numpy = pd.read_csv('data/circrna_node2vec.csv', header=0, index_col=0).values

        g_mi_orig_kn, g_mi_orig_km = self._build_hypergraphs(mi_original_numpy)
        g_circ_orig_kn, g_circ_orig_km = self._build_hypergraphs(circ_original_numpy)
        g_mi_sim_kn, g_mi_sim_km = self._build_hypergraphs(mi_sim_features.cpu().numpy())
        g_circ_sim_kn, g_circ_sim_km = self._build_hypergraphs(circ_sim_features.cpu().numpy())
        g_mi_n2v_kn, g_mi_n2v_km = self._build_hypergraphs(mi_n2v_numpy)
        g_circ_n2v_kn, g_circ_n2v_km = self._build_hypergraphs(circ_n2v_numpy)

        self.model_inputs = {
            'mi_original_features': torch.from_numpy(mi_original_numpy).to(self.device),
            'circ_original_features': torch.from_numpy(circ_original_numpy).to(self.device),
            'mi_sim_features': mi_sim_features,
            'circ_sim_features': circ_sim_features,
            'G_mi_original_Kn': g_mi_orig_kn, 'G_mi_original_Km': g_mi_orig_km,
            'G_circ_original_Kn': g_circ_orig_kn, 'G_circ_original_Km': g_circ_orig_km,
            'G_mi_sim_Kn': g_mi_sim_kn, 'G_mi_sim_Km': g_mi_sim_km,
            'G_circ_sim_Kn': g_circ_sim_kn, 'G_circ_sim_Km': g_circ_sim_km,
            'G_mi_Kn_node2vec': g_mi_n2v_kn, 'G_mi_Km_node2vec': g_mi_n2v_km,
            'G_circ_Kn_node2vec': g_circ_n2v_kn, 'G_circ_Km_node2vec': g_circ_n2v_km,
            'miRNA_node2vec_tensor': torch.tensor(mi_n2v_numpy).to(self.device),
            'circRNA_node2vec_tensor': torch.tensor(circ_n2v_numpy).to(self.device)
        }


    def run(self, train_data: list, fold_idx: int, test_data: Optional[list] = None) -> Tuple[torch.Tensor, ...]:
        self._prepare_inputs(train_data, test_data)

        self.model.train()
        criterion = Myloss()
        one_index = train_data[2][0].to(self.device).t().tolist()
        zero_index = train_data[2][1].to(self.device).t().tolist()

        print(f"--- [Fold {fold_idx + 1}] Start training... ---")
        for epoch in range(1, self.args.epoch + 1):
            scores, mi_cl_loss, circ_cl_loss = self.model(**self.model_inputs)

            recon_loss = criterion(one_index, zero_index, train_data[4].to(self.device), scores)
            total_loss = recon_loss + mi_cl_loss + circ_cl_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if epoch % 50 == 0:
                print(f"  Epoch {epoch}/{self.args.epoch}, total loss: {total_loss.item():.4f}")

        self.model.eval()
        with torch.no_grad():
            scores, _, _ = self.model(**self.model_inputs)

        test_one_index = train_data[3][0].t().tolist()
        test_zero_index = train_data[3][1].t().tolist()

        true_one = train_data[5][test_one_index]
        true_zero = train_data[5][test_zero_index]
        pred_one = scores[test_one_index]
        pred_zero = scores[test_zero_index]

        return true_one, true_zero, pred_one, pred_zero


def train(model: torch.nn.Module, train_data: list, optimizer: Optimizer, args, fold_idx: int,
          test_data: Optional[list] = None) -> Tuple[torch.Tensor, ...]:

    trainer = FoldTrainer(model, optimizer, args)
    return trainer.run(train_data, fold_idx, test_data)


def evaluate(true_one: torch.Tensor, true_zero: torch.Tensor, pre_one: torch.Tensor,
             pre_zero: torch.Tensor) -> np.ndarray:

    metric_calculator = MetricsCalculator()
    total_metrics = np.zeros((1, 7))

    for seed in range(EVALUATION_SEEDS):
        num_positive_samples = true_one.shape[0]
        negative_indices_all = np.array(np.where(true_zero == 0))

        np.random.seed(seed)
        np.random.shuffle(negative_indices_all.T)

        sampled_negative_indices = tuple(negative_indices_all[:, :num_positive_samples])

        eval_true = torch.cat([true_one, true_zero[sampled_negative_indices]])
        eval_pred = torch.cat([pre_one, pre_zero[sampled_negative_indices]])

        total_metrics += metric_calculator.cv_mat_model_evaluate(eval_true, eval_pred)

    return total_metrics / EVALUATION_SEEDS