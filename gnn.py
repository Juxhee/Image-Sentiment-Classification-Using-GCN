import math
import json
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import Parameter
from node_edge import *
from util import *


# https://github.com/spmallick/learnopencv/blob/master/Graph-Convolutional-Networks-Model-Relations-In-Data/graph_convolutional_networks_model_relations_in_data.ipynb
class GraphConvolution(nn.Module):
    """
        Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # 텐서를 받아 uniform distribution 값으로 초기화
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input.float(), self.weight.float())
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'




class GCNResnext50(nn.Module):
    def __init__(self, n_classes, adj_path, in_channel=300, t=0.1, p=0.25):
        super().__init__()
        self.sigm = nn.Sigmoid()

        self.features = models.resnext50_32x4d(pretrained=True)
        # self.features = models.resnet18(pretrained=True)

        self.features.fc = nn.Identity()
        self.num_classes = n_classes

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)
        # Load data for adjacency matrix
        with open(adj_path) as fp:
            adj_data = json.load(fp)
        # Compute adjacency matrix
        adj = gen_A(n_classes, t, p, adj_data)
        self.A = Parameter(torch.from_numpy(adj).float(), requires_grad=False)

    def forward(self, imgs, inp):
        # Get visual features from image
        # resnext50_32x4d로부터 추출
        feature = self.features(imgs)
        feature = feature.view(feature.size(0), -1)

        # word embedding이랑 adj multiply
        # Get graph features from graph
        inp = inp[0].squeeze()
        adj = gen_adj(self.A).detach()
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        # We multiply the features from GСN and СNN in order to take into account
        # the contribution to the prediction of classes from both the image and the graph.
        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return self.sigm(x)
