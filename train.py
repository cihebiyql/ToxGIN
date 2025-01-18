import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
import torch
import dgl
import time
from tqdm import tqdm
import random
import gc
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
import csv
from torch.utils.tensorboard import SummaryWriter
import dgl.nn
import dgl.nn.pytorch

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, indexes=None, add_self_loop=False):
        super(GraphDataset, self).__init__()
        self.dir_path = dir_path
        self.graphs, label_dict = dgl.load_graphs(self.dir_path+'/dgl_graph.bin')
        self.df = pd.read_csv('peptide.csv', index_col=0)
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
        sequence = row['sequence']  
        label = row['label']
        return graph,label, idx
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

def calc_metrics(y_pred, y_true):
    y_pred_labels = (y_pred > 0.5).long()
    
    accuracy = accuracy_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)
    mcc = matthews_corrcoef(y_true, y_pred_labels)
    
    return accuracy, f1, mcc
def some_function():
    local_var = 10
    a = 20
    for name in dir():
        if not name.startswith('__'):
            print(name, '=', eval(name))

def one_fold(train_indexes, val_indexes, filename, model_output_dir, device, cv_num=0):
    writer_log = SummaryWriter(f'{model_output_dir}/CV_{cv_num}')
    print(f'CV{cv_num}==================================')
    seed = 3407
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    dgl.seed(seed)
    device = device
    hidden_dim = 1024  
    epochs = 500
    batch_size = 256  
    lr = 1e-4
    weight_decay = 1e-2
    total_train_loss = 0
    total_val_loss = 0
    patience = 50
    min_val_loss = 1e5
    best_mcc = 0
    criterion = torch.nn.BCELoss(reduction='sum')
    train_dataset = GraphDataset('train_dataset_5', train_indexes)
    train_dataloader = dgl.dataloading.GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_dataset = GraphDataset('train_dataset_5', val_indexes)
    val_dataloader = dgl.dataloading.GraphDataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    graph, label, original_index = next(iter(train_dataloader))
    feature_dim = graph.ndata['wildtype_seq'].shape[-1]
    print(feature_dim)
    model = GNNModel(feature_dim, hidden_dim).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    st = time.time()

    print("start")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0  
        count = 0
        for graph, label, original_index in tqdm(train_dataloader, leave=True):
            if random.random() < 1:  
                n_nodes_to_drop = random.randint(1, max(1, graph.num_nodes() // 10))  
                nodes_to_drop = np.random.choice(graph.nodes(), n_nodes_to_drop, replace=False)
                graph = dgl.remove_nodes(graph, nodes_to_drop)

            graph = graph.to(device)
            label = label.float().to(device).unsqueeze(1)
            wildtype_seq = graph.ndata['wildtype_seq']
            pred_labels = model(graph, wildtype_seq).squeeze(1)  
            label = label.squeeze(1) 
            
            loss = criterion(pred_labels, label)
            writer_log.add_scalar('Loss/train', loss.item(), epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += label.size(0)  
            
            total_train_loss += loss.item() * label.size(0)  

        avg_train_loss = total_train_loss / count 
        count = 0
        total_val_loss = 0
        model.eval()
        y_pred, y_true = [], []
        scheduler.step()
        with torch.no_grad():
            for graph, labels, original_index in tqdm(val_dataloader, leave=True):
                graph = graph.to(device)
                labels = labels.float().to(device)
                wildtype_seq = graph.ndata['wildtype_seq']
                pred_labels = model(graph, wildtype_seq).squeeze(1)
                loss = criterion(pred_labels, labels)
                count += labels.size(0)
                total_val_loss += loss.item() * labels.size(0)
                y_pred.extend(pred_labels.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
                writer_log.add_scalar('Loss/val', total_val_loss, epoch)

        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)


        auROC = roc_auc_score(y_true_array, y_pred_array)
        auPRC = average_precision_score(y_true_array, y_pred_array)

        threshold = 0.5  
        y_pred_binary = (y_pred_array > threshold).astype(np.int32)
        tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred_binary).ravel()

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        fdr = fp / (tp + fp)  

        print(f'cv_num {cv_num}: auROC: {auROC}, auPRC: {auPRC}, Sensitivity: {sensitivity}, Specificity: {specificity}, FDR: {fdr}')

        spent_time = time.time() - st
        avg_val_loss = total_val_loss / count
        accuracy, f1, mcc = calc_metrics(torch.tensor(y_pred), torch.tensor(y_true))
        print(f'time: {np.around(spent_time/60, decimals=2)}min  Accuracy: {accuracy}, F1: {f1}, MCC: {mcc}')
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        writer_log.add_scalar('Accuracy/val', accuracy, epoch)
        writer_log.add_scalar('F1_Score/val', f1, epoch)
        writer_log.add_scalar('MCC/val', mcc, epoch)
        writer_log.add_scalar('AUROC/val', auROC, epoch)
        writer_log.add_scalar('AUPRC/val', auPRC, epoch)
        writer_log.add_scalar('Sensitivity/val', sensitivity, epoch)
        writer_log.add_scalar('Specificity/val', specificity, epoch)
        writer_log.add_scalar('FDR/val', fdr, epoch)

        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([cv_num, epoch, auROC, auPRC, accuracy, f1, mcc, avg_train_loss, avg_val_loss, sensitivity, specificity, fdr])

        if best_mcc <= mcc:
            best_mcc = mcc
            min_not_update_count = 0
            torch.save(model.state_dict(), f'{model_output_dir}/best_model{cv_num}.pth')
        else:
            min_not_update_count += 1
        if min_not_update_count > patience:
            print('Early stopping!')
            break
        if spent_time > 6000:
            print('timeover')
            break
    writer_log.close()
    print(f'CV{cv_num} finish! min_val_loss={best_mcc}')


def run():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    df = pd.read_csv('peptide.csv', index_col=0)
    labels = df['label'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3407)  
    model_output_dir = 'output_model'  
    os.makedirs(model_output_dir, exist_ok=True)  
    initial_time = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{model_output_dir}/training_log_{initial_time}.csv"

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['CV', 'Epoch', 'AUROC', 'AUPRC', 'Accuracy', 'F1', 'MCC', 'Train Loss', 'Val Loss','SN','SP',"FDR"])
        
    for cv_num, (train_idx, val_idx) in enumerate(skf.split(df, labels)):
        train_indexes = df.iloc[train_idx].index
        val_indexes = df.iloc[val_idx].index
        
        one_fold(train_indexes, val_indexes, filename, model_output_dir,device,cv_num)
        gc.collect()

    final_time = time.strftime("%Y%m%d-%H%M%S")
    final_filename = f"final_training_log_{final_time}.csv"
    os.rename(filename, final_filename)
    print(f"Training complete. Results saved to {final_filename}")
    some_function()

if __name__ == "__main__":
    run()



