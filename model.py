import random
import os
import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.in_features = in_ft
        self.out_features = out_ft
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class HGCN(nn.Module):
    def __init__(self, in_dim, hidden_list, dropout = 0.5):
        super(HGCN, self).__init__()
        self.dropout = dropout
        self.hgnn1 = HGNN_conv(in_dim, hidden_list[0])

    def forward(self,x, G):
        x_embed = self.hgnn1(x, G)
        x_embed_1 = F.leaky_relu(x_embed, 0.25)
        return x_embed_1

class HGNN_conv_mi_original(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv_mi_original, self).__init__()
        self.in_features = 2346
        self.out_features = 256
        self.weight = Parameter(torch.Tensor(2346,256))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class HGCN_mi_original(nn.Module):
    def __init__(self, in_dim, hidden_list, dropout = 0.5):
        super(HGCN_mi_original, self).__init__()
        self.dropout = dropout
        self.hgnn1 = HGNN_conv_mi_original(in_dim, hidden_list[0])

    def forward(self,x, G):
        x_embed = self.hgnn1(x, G)
        x_embed_1 = F.leaky_relu(x_embed, 0.25)
        return x_embed_1

class HGNN_conv_circ_original(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv_circ_original, self).__init__()
        self.in_features = 962
        self.out_features = 256
        self.weight = Parameter(torch.Tensor(962,256))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class HGCN_circ_original(nn.Module):
    def __init__(self, in_dim, hidden_list, dropout = 0.5):
        super(HGCN_circ_original, self).__init__()
        self.dropout = dropout
        self.hgnn1 = HGNN_conv_circ_original(in_dim, hidden_list[0])

    def forward(self,x, G):
        x_embed = self.hgnn1(x, G)
        x_embed_1 = F.leaky_relu(x_embed, 0.25)
        return x_embed_1

class HGNN_conv_mi_sim(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv_mi_sim, self).__init__()
        self.in_features = 962
        self.out_features = 256
        self.weight = Parameter(torch.Tensor(962,256))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class HGCN_mi_sim(nn.Module):
    def __init__(self, in_dim, hidden_list, dropout = 0.5):
        super(HGCN_mi_sim, self).__init__()
        self.dropout = dropout
        self.hgnn2 = HGNN_conv_mi_sim(in_dim, hidden_list[0])

    def forward(self, x, G):
        x_embed = self.hgnn2(x, G)
        x_embed_1 = F.leaky_relu(x_embed, 0.25)
        return x_embed_1

class HGNN_conv_circ_sim(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv_circ_sim, self).__init__()
        self.in_features = 2346
        self.out_features = 256
        self.weight = Parameter(torch.Tensor(2346,256))

        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class HGCN_circ_sim(nn.Module):
    def __init__(self, in_dim, hidden_list, dropout = 0.5):
        super(HGCN_circ_sim, self).__init__()
        self.dropout = dropout
        self.hgnn2 = HGNN_conv_circ_sim(in_dim, hidden_list[0])

    def forward(self, x, G):
        x_embed = self.hgnn2(x, G)
        x_embed_1 = F.leaky_relu(x_embed, 0.25)
        return x_embed_1


class HGNN_conv_mi_node2vec(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv_mi_node2vec, self).__init__()
        self.in_features = 256
        self.out_features = 256
        self.weight = Parameter(torch.Tensor(256,256))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class HGCN_mi_node2vec(nn.Module):
    def __init__(self, in_dim, hidden_list, dropout = 0.5):
        super(HGCN_mi_node2vec, self).__init__()
        self.dropout = dropout
        self.hgnn1 = HGNN_conv_mi_node2vec(in_dim, hidden_list[0])

    def forward(self,x, G):
        x_embed = self.hgnn1(x, G)
        x_embed_1 = F.leaky_relu(x_embed, 0.25)
        return x_embed_1

class HGNN_conv_circ_node2vec(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv_circ_node2vec, self).__init__()
        self.in_features =256
        self.out_features = 256
        self.weight = Parameter(torch.Tensor(256,256))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class HGCN_circ_node2vec(nn.Module):
    def __init__(self, in_dim, hidden_list, dropout = 0.5):
        super(HGCN_circ_node2vec, self).__init__()
        self.dropout = dropout
        self.hgnn1 = HGNN_conv_circ_node2vec(in_dim, hidden_list[0])

    def forward(self,x, G):
        x_embed = self.hgnn1(x, G)
        x_embed_1 = F.leaky_relu(x_embed, 0.25)
        return x_embed_1

class CL_HGCN_mi_original(nn.Module):
    def __init__(self, in_size, hid_list, num_proj_hidden, out_ft,alpha = 0.5):
        super(CL_HGCN_mi_original, self).__init__()
        self.hgcn1 = HGCN_mi_original(in_size, hid_list,out_ft)
        self.hgcn2 = HGCN_mi_original(in_size, hid_list,out_ft)
        self.fc1 = torch.nn.Linear(hid_list[-1], num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, hid_list[-1])
        self.tau = 0.5
        self.alpha = alpha

    def forward(self, x1, adj1, x2, adj2):
        z1 = self.hgcn1(x1, adj1)
        h1 = self.projection(z1)
        z2 = self.hgcn2(x2, adj2)
        h2 = self.projection(z2)
        loss = self.alpha*self.sim(h1, h2) + (1-self.alpha)*self.sim(h2,h1)
        return z1, z2, loss

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def norm_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def sim(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.norm_sim(z1, z1))
        between_sim = f(self.norm_sim(z1, z2))
        loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        loss = loss.sum(dim=-1).mean()
        return loss


class CL_HGCN_circ_original(nn.Module):
    def __init__(self, in_size, hid_list, num_proj_hidden, out_ft,alpha = 0.5):
        super(CL_HGCN_circ_original, self).__init__()
        self.hgcn1 = HGCN_circ_original(in_size, hid_list,out_ft)
        self.hgcn2 = HGCN_circ_original(in_size, hid_list,out_ft)
        self.fc1 = torch.nn.Linear(hid_list[-1], num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, hid_list[-1])
        self.tau = 0.5
        self.alpha = alpha

    def forward(self, x1, adj1, x2, adj2):
        z1 = self.hgcn1(x1, adj1)
        h1 = self.projection(z1)
        z2 = self.hgcn2(x2, adj2)
        h2 = self.projection(z2)
        loss = self.alpha*self.sim(h1, h2) + (1-self.alpha)*self.sim(h2,h1)
        return z1, z2, loss

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def norm_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def sim(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.norm_sim(z1, z1))
        between_sim = f(self.norm_sim(z1, z2))
        loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        loss = loss.sum(dim=-1).mean()
        return loss


class CL_HGCN_mi_sim(nn.Module):
    def __init__(self, in_size, hid_list, num_proj_hidden, out_ft,alpha = 0.5):
        super(CL_HGCN_mi_sim, self).__init__()
        self.hgcn1 = HGCN_mi_sim(in_size, hid_list,out_ft)
        self.hgcn2 = HGCN_mi_sim(in_size, hid_list,out_ft)
        self.fc1 = torch.nn.Linear(hid_list[-1], num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, hid_list[-1])
        self.tau = 0.5
        self.alpha = alpha

    def forward(self, x1, adj1, x2, adj2):
        z1 = self.hgcn1(x1, adj1)
        h1 = self.projection(z1)
        z2 = self.hgcn2(x2, adj2)
        h2 = self.projection(z2)
        loss = self.alpha*self.sim(h1, h2) + (1-self.alpha)*self.sim(h2,h1)
        return z1, z2, loss

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def norm_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def sim(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.norm_sim(z1, z1))
        between_sim = f(self.norm_sim(z1, z2))
        loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        loss = loss.sum(dim=-1).mean()
        return loss


class CL_HGCN_circ_sim(nn.Module):
    def __init__(self, in_size, hid_list, num_proj_hidden, out_ft,alpha = 0.5):
        super(CL_HGCN_circ_sim, self).__init__()
        self.hgcn1 = HGCN_circ_sim(in_size, hid_list,out_ft)
        self.hgcn2 = HGCN_circ_sim(in_size, hid_list,out_ft)
        self.fc1 = torch.nn.Linear(hid_list[-1], num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, hid_list[-1])
        self.tau = 0.5
        self.alpha = alpha

    def forward(self, x1, adj1, x2, adj2):
        z1 = self.hgcn1(x1, adj1)
        h1 = self.projection(z1)
        z2 = self.hgcn2(x2, adj2)
        h2 = self.projection(z2)
        loss = self.alpha*self.sim(h1, h2) + (1-self.alpha)*self.sim(h2,h1)
        return z1, z2, loss

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def norm_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def sim(self, z1, z2):
        #相似度损失计算
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.norm_sim(z1, z1))
        between_sim = f(self.norm_sim(z1, z2))
        loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        loss = loss.sum(dim=-1).mean()
        return loss

class CL_HGCN_mi_node2vec(nn.Module):
    def __init__(self, in_size, hid_list, num_proj_hidden, out_ft,alpha = 0.5):
        super(CL_HGCN_mi_node2vec, self).__init__()
        self.hgcn1 = HGCN_mi_node2vec(in_size, hid_list,out_ft)
        self.hgcn2 = HGCN_mi_node2vec(in_size, hid_list,out_ft)
        self.fc1 = torch.nn.Linear(hid_list[-1], num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, hid_list[-1])
        self.tau = 0.5
        self.alpha = alpha

    def forward(self, x1, adj1, x2, adj2):
        z1 = self.hgcn1(x1, adj1)
        h1 = self.projection(z1)
        z2 = self.hgcn2(x2, adj2)
        h2 = self.projection(z2)
        loss = self.alpha*self.sim(h1, h2) + (1-self.alpha)*self.sim(h2,h1)
        return z1, z2, loss

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def norm_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def sim(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.norm_sim(z1, z1))
        between_sim = f(self.norm_sim(z1, z2))
        loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        loss = loss.sum(dim=-1).mean()
        return loss


class CL_HGCN_circ_node2vec(nn.Module):
    def __init__(self, in_size, hid_list, num_proj_hidden, out_ft,alpha = 0.5):
        super(CL_HGCN_circ_node2vec, self).__init__()
        self.hgcn1 = HGCN_circ_node2vec(in_size, hid_list,out_ft)
        self.hgcn2 = HGCN_circ_node2vec(in_size, hid_list,out_ft)
        self.fc1 = torch.nn.Linear(hid_list[-1], num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, hid_list[-1])
        self.tau = 0.5
        self.alpha = alpha

    def forward(self, x1, adj1, x2, adj2):
        z1 = self.hgcn1(x1, adj1)
        h1 = self.projection(z1)
        z2 = self.hgcn2(x2, adj2)
        h2 = self.projection(z2)
        loss = self.alpha*self.sim(h1, h2) + (1-self.alpha)*self.sim(h2,h1)

        return z1, z2, loss

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def norm_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def sim(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.norm_sim(z1, z1))
        between_sim = f(self.norm_sim(z1, z2))
        loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        loss = loss.sum(dim=-1).mean()
        return loss


class HGCN_Attention_mechanism(nn.Module):

    def __init__(self):
        super(HGCN_Attention_mechanism,self).__init__()
        self.hiddim = 64
        self.fc_x1 = nn.Linear(in_features=2, out_features=self.hiddim)
        self.fc_x2 = nn.Linear(in_features=self.hiddim, out_features=2)
        self.sigmoidx = nn.Sigmoid()

    def forward(self,input_list):
        XM = torch.cat((input_list[0], input_list[1]), 1).t()
        XM = XM.view(1, 1 * 2, input_list[0].shape[1], -1)
        globalAvgPool_x = nn.AvgPool2d((input_list[0].shape[1], input_list[0].shape[0]), (1, 1))
        x_channel_attenttion = globalAvgPool_x(XM)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), -1)
        x_channel_attenttion = self.fc_x1(x_channel_attenttion)
        x_channel_attenttion = torch.relu(x_channel_attenttion)
        x_channel_attenttion = self.fc_x2(x_channel_attenttion)
        x_channel_attenttion = self.sigmoidx(x_channel_attenttion)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1, 1)
        XM_channel_attention = x_channel_attenttion * XM
        XM_channel_attention = torch.relu(XM_channel_attention)
        return XM_channel_attention[0]


class HCLAMCMI(nn.Module):
    def __init__(self, mi_num, circ_num, hidd_list, num_proj_hidden, hyperpm):
        super(HCLAMCMI, self).__init__()
        self.CL_HGCN_mi_original = CL_HGCN_mi_original(mi_num + circ_num, hidd_list,num_proj_hidden,256)
        self.CL_HGCN_circ_original = CL_HGCN_circ_original(circ_num + mi_num, hidd_list,num_proj_hidden,256)

        self.CL_HGCN_mi_sim = CL_HGCN_mi_sim(mi_num + circ_num, hidd_list,num_proj_hidden,256 )
        self.CL_HGCN_circ_sim = CL_HGCN_circ_sim(circ_num + mi_num, hidd_list, num_proj_hidden, 256)

        self.CL_HGCN_mi_node2vec = CL_HGCN_mi_node2vec(mi_num + circ_num, hidd_list,num_proj_hidden,128 )
        self.CL_HGCN_circ_node2vec = CL_HGCN_circ_node2vec(circ_num + mi_num, hidd_list, num_proj_hidden,128)

        self.AM_mi = HGCN_Attention_mechanism()
        self.AM_circ = HGCN_Attention_mechanism()

        self.linear_x_1 = nn.Linear(1536, 1024)
        self.linear_x_2 = nn.Linear(1024, 768)

        self.linear_y_1 = nn.Linear(1536, 1024)
        self.linear_y_2 = nn.Linear(1024, 768)


    def forward(self, mi_original_features, circ_original_features, mi_sim_features, circ_sim_features,
                G_mi_original_Kn, G_mi_original_Km, G_circ_original_Kn, G_circ_original_Km,
                G_mi_sim_Kn, G_mi_sim_Km, G_circ_sim_Kn, G_circ_sim_Km,
                G_mi_Kn_new, G_mi_Km_new, G_circ_Kn_new, G_circ_Km_new, miRNA_node2vec_tensor, circRNA_node2vec_tensor,
               ):

        circ_original_features = circ_original_features.float()
        mi_original_features = mi_original_features.float()

        mi_sim_features=mi_sim_features.float()
        circ_sim_features=circ_sim_features.float()

        mi_node2vec_features = miRNA_node2vec_tensor.float()
        circ_node2vec_features  = circRNA_node2vec_tensor.float()

        mi_feature1_original, mi_feature2_original, mi_cl_loss_new = self.CL_HGCN_mi_original(mi_original_features, G_mi_original_Kn, mi_original_features,G_mi_original_Km)
        circ_feature1_original, circ_feature2_original, circ_cl_loss_new = self.CL_HGCN_circ_original(circ_original_features, G_circ_original_Kn, circ_original_features, G_circ_original_Km)

        mi_feature1_sim,mi_feature2_sim, mi_cl_loss_new = self.CL_HGCN_mi_sim(mi_sim_features,G_mi_sim_Kn, mi_sim_features,G_mi_sim_Km)
        circ_feature1_sim, circ_feature2_sim, circ_cl_loss_new = self.CL_HGCN_circ_sim(circ_sim_features,G_circ_sim_Kn,circ_sim_features, G_circ_sim_Km)

        mi_feature1_node2vec, mi_feature2_node2vec, mi_cl_loss_new = self.CL_HGCN_mi_node2vec(mi_node2vec_features, G_mi_Kn_new, mi_node2vec_features,G_mi_Km_new)
        circ_feature1_node2vec, circ_feature2_node2vec, circ_cl_loss_new = self.CL_HGCN_circ_node2vec(circ_node2vec_features , G_circ_Kn_new,circ_node2vec_features , G_circ_Km_new)

        mi_feature_att_original = self.AM_mi([mi_feature1_original, mi_feature2_original])
        circ_feature_att_original = self.AM_circ([circ_feature1_original, circ_feature2_original])
        mi_feature_att1_original = mi_feature_att_original[0].t()
        mi_feature_att2_original = mi_feature_att_original[1].t()
        mi_feature_original = torch.cat([mi_feature_att1_original, mi_feature_att2_original], dim=1)
        circ_feature_att1_original = circ_feature_att_original[0].t()
        circ_feature_att2_original = circ_feature_att_original[1].t()
        circ_feature_original = torch.cat([circ_feature_att1_original, circ_feature_att2_original], dim=1)

        mi_feature_att_sim =self.AM_mi([mi_feature1_sim, mi_feature2_sim])
        circ_feature_att_sim = self.AM_circ([circ_feature1_sim, circ_feature2_sim])
        mi_feature_att1_sim = mi_feature_att_sim[0].t()
        mi_feature_att2_sim = mi_feature_att_sim[1].t()
        mi_feature_sim= torch.cat([mi_feature_att1_sim, mi_feature_att2_sim], dim=1)
        circ_feature_att1_sim = circ_feature_att_sim[0].t()
        circ_feature_att2_sim = circ_feature_att_sim[1].t()
        circ_feature_sim = torch.cat([circ_feature_att1_sim, circ_feature_att2_sim], dim=1)

        mi_feature_att_node2vec = self.AM_mi([mi_feature1_node2vec, mi_feature2_node2vec])
        circ_feature_att_node2vec = self.AM_circ([circ_feature1_node2vec, circ_feature2_node2vec])
        mi_feature_att1_node2vec = mi_feature_att_node2vec[0].t()
        mi_feature_att2_node2vec = mi_feature_att_node2vec[1].t()
        mi_feature_node2vec = torch.cat([mi_feature_att1_node2vec, mi_feature_att2_node2vec], dim=1)
        circ_feature_att1_node2vec = circ_feature_att_node2vec[0].t()
        circ_feature_att2_node2vec = circ_feature_att_node2vec[1].t()
        circ_feature_node2vec = torch.cat([circ_feature_att1_node2vec, circ_feature_att2_node2vec], dim=1)

        mi_feature_original = F.layer_norm(mi_feature_original, mi_feature_original.shape[1:])
        mi_feature_sim = F.layer_norm(mi_feature_sim, mi_feature_sim.shape[1:])
        mi_feature_node2vec = F.layer_norm(mi_feature_node2vec, mi_feature_node2vec.shape[1:])
        circ_feature_original = F.layer_norm(circ_feature_original, circ_feature_original.shape[1:])
        circ_feature_sim = F.layer_norm(circ_feature_sim, circ_feature_sim.shape[1:])
        circ_feature_node2vec = F.layer_norm(circ_feature_node2vec, circ_feature_node2vec.shape[1:])

        mi_feature_final = torch.cat([mi_feature_original,mi_feature_sim,mi_feature_node2vec], dim=1)
        circ_feature_final = torch.cat([circ_feature_original,circ_feature_sim,circ_feature_node2vec], dim=1)

        x1 = torch.relu(self.linear_x_1(mi_feature_final))
        x2 = torch.relu(self.linear_x_2(x1))
        y1 = torch.relu(self.linear_y_1(circ_feature_final))
        y2 = torch.relu(self.linear_y_2(y1))

        score = x2.mm(y2.t())
        return score, mi_cl_loss_new, circ_cl_loss_new
