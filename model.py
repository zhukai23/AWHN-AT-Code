import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as f
import torch
from torch_geometric.data import Data
from load_data import BuildGraph
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 64)
        self.fc1 = nn.Linear(64, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = f.relu(x)
        x = f.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = f.relu(x)
        x = f.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x1 = self.fc1(x)
        x2 = f.relu(x1)
        x3 = self.fc2(x2)
        x4 = f.relu(x3)

        return x, x4


class Gcn(nn.Module):
    def __init__(self, input_dim=116, hidden_dim=64, dropout=0.3):
        super(Gcn, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 116)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = f.relu(x)
        x = f.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        return x


def augment_graph(data):
    n_neighbors = 5
    build_graph = BuildGraph(n_neighbors)
    features, _ = build_graph.load_data()
    num_nodes = 116
    linjie = build_graph.create_knn_graph(features, num_nodes)
    gcn_model = Gcn().to(device)
    new_x = gcn_model(data)
    new_adj = linjie
    new_x = new_x.view(new_x.size(0) // 116, num_nodes, num_nodes)
    new_x_array = new_x.cpu().detach().numpy()
    new_data = []
    for i in range(len(new_x_array)):
        x = torch.as_tensor(new_x_array[i], dtype=torch.float).to(device)
        edge_index = torch.as_tensor(np.array(np.nonzero(new_adj[i])), dtype=torch.long).to(device)
        new_data.append(Data(x=x, edge_index=edge_index))

    return new_data


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU):
        super(MLP, self).__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class TrainModel(nn.Module):
    def __init__(self, input_dim=116, hidden_dim=128, out_dim=128):
        super(TrainModel, self).__init__()
        self.feature_extractor = GCN(input_dim, hidden_dim, out_dim, dropout=0.)
        self.mlp = MLP(input_dim=64, hidden_dims=[128, 64], output_dim=2, activation=nn.ReLU)

    def forward(self, data):
        z1_proj_before,  z1_proj_after = self.feature_extractor(data)
        new_data = augment_graph(data)
        z2_proj_after = [self.feature_extractor(new_data)[1] for new_data in new_data]
        z2_proj_after = torch.cat(z2_proj_after).reshape((-1, 128))
        output = self.mlp(z1_proj_before)
        x = data.x
        x_aug = torch.cat([data.x for data in new_data], dim=0)
        output = f.softmax(output, dim=1)
        return z1_proj_after, z2_proj_after, output, x, x_aug
