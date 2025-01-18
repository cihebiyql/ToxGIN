import os
import numpy as np
import pandas as pd
import torch
import dgl
import time
import csv
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter

# 定义 GraphDataset 类，主要用于数据的加载
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, scv_path, dir_path, indexes=None, add_self_loop=False):
        super(GraphDataset, self).__init__()
        self.dir_path = dir_path
        self.graphs, label_dict = dgl.load_graphs(self.dir_path+'/dgl_graph.bin')
        self.df = pd.read_csv(scv_path, index_col=0)
        self.add_self_loop = add_self_loop
        self.uni_to_index = {uni: idx for idx, uni in enumerate(self.df['sequence'])}
        if indexes is None:
            self.indexes = self.df.index
        else:
            self.indexes = indexes

    def __getitem__(self, i):
        idx = self.indexes[i]
        row = self.df.loc[idx]
        uni_id = row['sequence']
        graph_index = self.uni_to_index[uni_id]
        graph = self.graphs[graph_index].clone()
        if self.add_self_loop:
            graph = dgl.add_self_loop(graph)

        wildtype_feature = np.load(self.dir_path+'/'+row.sequence+'.npz')
        wildtype_seq = wildtype_feature['wildtype_seq']
        graph.ndata['wildtype_seq'] = torch.from_numpy(wildtype_seq)
        return graph, uni_id

    def __len__(self):
        return len(self.indexes)

class GIN(torch.nn.Module):
    def __init__(self, in_dim, layer_num=4):
        super(GIN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for i in range(layer_num):
            self.convs.append(
                dgl.nn.pytorch.GINConv(torch.nn.Sequential(
                    torch.nn.Linear(in_dim, in_dim, bias=False),
                    torch.nn.BatchNorm1d(in_dim),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(in_dim, in_dim, bias=False),
                ), learn_eps=False))
            self.activations.append(torch.nn.LeakyReLU())
            self.batch_norms.append(torch.nn.BatchNorm1d(in_dim))
        self.layer_num = layer_num
        self.out_dim = in_dim * (layer_num+1)
    def forward(self, g, h):
        hs = [h]
        for conv, batch_norm, act in zip(self.convs, self.batch_norms, self.activations):
            h = conv(g, h)
            h = batch_norm(h)
            h = act(h)
            hs.append(h)
        return torch.cat(hs, dim=-1)   
    
# 定义 GNNModel 类，用于图神经网络推理
class GNNModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout_rate=0.5):
        super(GNNModel, self).__init__()
        self.comp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.LeakyReLU()
        )
        self.gcn = GIN(hidden_dim)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(self.gcn.out_dim, self.gcn.out_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(self.gcn.out_dim, 3000),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(3000, 2000),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(2000, 1000),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(1000, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, g, wildtype_seq):
        wildtype_h = self.comp(wildtype_seq)
        gcn_h = self.gcn(g, wildtype_h)
        with g.local_scope():
            g.ndata['h'] = gcn_h
            hg = dgl.readout_nodes(g, 'h', op='sum')
        pred = self.classifier(hg)
        return pred

# 定义推理过程
def infer(model, dataloader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for graph, _ in dataloader:
            graph = graph.to(device)
            wildtype_seq = graph.ndata['wildtype_seq']
            preds = model(graph, wildtype_seq).cpu().numpy()
            all_preds.append(preds)
    return np.concatenate(all_preds)

# 定义 collate_fn，用于正确处理 DGLGraph 数据
def collate_fn(batch):
    graphs, uni_ids = zip(*batch)
    batched_graph = dgl.batch(graphs)  # 将图批处理成一个批次
    return batched_graph, uni_ids

def run_inference():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    scv_path = 'test_sequence.csv'
    # 加载数据
    df = pd.read_csv(scv_path, index_col=0)
    test_indexes = df.index  # 推理使用整个数据集

    # 创建Dataset和DataLoader
    test_dataset = GraphDataset(scv_path, 'test_data', test_indexes)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False, collate_fn=collate_fn)
    # 加载预训练的模型
    model = GNNModel(in_dim=3113, hidden_dim=1024).to(device)
    model.load_state_dict(torch.load('best_model.pth'))  # 加载已训练好的模型

    # 进行推理
    predictions = infer(model, test_dataloader, device)

    # 输出预测结果
    df['predicted_probability'] = predictions
    df.to_csv('predictions.csv')  # 将预测结果保存到CSV文件
    print("Predictions saved to 'predictions.csv'")

if __name__ == "__main__":
    run_inference()
