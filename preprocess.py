import os
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from biopandas.mmcif import PandasMmcif
import torch
import dgl
import tokenizers
import transformers
import os
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModel, AutoConfig

def load_aaindex1(file_path):
    aaindex1_df = pd.read_csv(file_path, index_col='Description')
    aaindex_dict = {aa: aaindex1_df[aa].values for aa in aaindex1_df.columns}
    return aaindex_dict

def extract_features(sequence, aaindex_dict):
    features = []
    for aa in sequence:
        if aa in aaindex_dict:
            features.append(aaindex_dict[aa])
        else:
            features.append(np.full((len(next(iter(aaindex_dict.values()))),), np.nan))
    return np.array(features)

def generate_graph(coords, threshold=8.0):
    all_diffs = np.expand_dims(coords, axis=1) - np.expand_dims(coords, axis=0)
    distance = np.sqrt(np.sum(np.power(all_diffs, 2), axis=-1))
    adj = distance < threshold
    u, v = np.nonzero(adj)
    u, v = torch.from_numpy(u), torch.from_numpy(v)
    graph = dgl.graph((u, v), num_nodes=coords.shape[0])
    return graph

model = "facebook/esm2_t36_3B_UR50D"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

tokenizer = AutoTokenizer.from_pretrained(model)
config = AutoConfig.from_pretrained(model, output_hidden_states=True)
config.hidden_dropout = 0.
config.hidden_dropout_prob = 0.
config.attention_dropout = 0.
config.attention_probs_dropout_prob = 0.
encoder = AutoModel.from_pretrained(model, config=config).to(device).eval()
print("model loaded")

def seq_encode(seq):
    spaced_seq = " ".join(list(seq))
    inputs = tokenizer.encode_plus(
        spaced_seq, 
        return_tensors=None, 
        add_special_tokens=True,
        max_length=60,
        padding=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long).unsqueeze(0).cuda()
    with torch.no_grad():
        outputs = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    last_hidden_states = outputs[0]
    encoded_seq = last_hidden_states[inputs['attention_mask'].bool()][1:-1]
    return encoded_seq

def aggre(s):
    if type(s.values[0]) == str:
        return s.values[0]
    return np.mean(s)

aa_map = {'VAL': 'V', 'PRO': 'P', 'ASN': 'N', 'GLU': 'E', 'ASP': 'D', 'ALA': 'A', 'THR': 'T', 'SER': 'S',
          'LEU': 'L', 'LYS': 'K', 'GLY': 'G', 'GLN': 'Q', 'ILE': 'I', 'PHE': 'F', 'CYS': 'C', 'TRP': 'W',
          'ARG': 'R', 'TYR': 'Y', 'HIS': 'H', 'MET': 'M'}

df = pd.read_csv('test_sequence.csv')
root_dir = 'test_data'
os.makedirs(root_dir, exist_ok=True)
graphs = []
pdb = None

aaindex_dict = load_aaindex1('aaindex1.csv')
scaler = MinMaxScaler()
for index,row in df.iterrows():    
    atom_df = PandasPdb().read_pdb(f'test_structures/{row.sequence}.pdb')
    atom_df = atom_df.df['ATOM']
    residue_df = atom_df.groupby('residue_number', as_index=False).agg(aggre).sort_values('residue_number')
    del atom_df
    residue_df['letter'] = residue_df.residue_name.map(aa_map)
    
    graph = None
    pdb_seq = ''.join(residue_df.letter.values)

    if graph is None:
        graph = generate_graph(residue_df.loc[:, ['x_coord', 'y_coord', 'z_coord']].values)
        graphs.append(graph)
        encoded_pdb_seq = seq_encode(pdb_seq).cpu().numpy()
        features = extract_features(pdb_seq, aaindex_dict)
        # normalized_features = scaler.fit_transform(features)
        normalized_features = scaler.fit_transform(features.astype(np.float32))
        encoded_pdb_seq = np.concatenate([encoded_pdb_seq, normalized_features], axis=-1)
        path = f'{row.sequence}'
        np.savez_compressed(root_dir+'/'+path, wildtype_seq=encoded_pdb_seq)
dgl.save_graphs(root_dir+'/dgl_graph.bin', graphs)

df = pd.read_csv('train_sequence.csv')
root_dir = 'train_data'
os.makedirs(root_dir, exist_ok=True)
graphs = []
pdb = None

aaindex_dict = load_aaindex1('aaindex1.csv')
scaler = MinMaxScaler()
for index,row in df.iterrows():    
    atom_df = PandasPdb().read_pdb(f'train_structures/{row.sequence}.pdb')
    atom_df = atom_df.df['ATOM']
    residue_df = atom_df.groupby('residue_number', as_index=False).agg(aggre).sort_values('residue_number')
    del atom_df
    residue_df['letter'] = residue_df.residue_name.map(aa_map)
    
    graph = None
    pdb_seq = ''.join(residue_df.letter.values)

    if graph is None:
        graph = generate_graph(residue_df.loc[:, ['x_coord', 'y_coord', 'z_coord']].values)
        graphs.append(graph)
        encoded_pdb_seq = seq_encode(pdb_seq).cpu().numpy()
        features = extract_features(pdb_seq, aaindex_dict)
        # normalized_features = scaler.fit_transform(features)
        normalized_features = scaler.fit_transform(features.astype(np.float32))
        encoded_pdb_seq = np.concatenate([encoded_pdb_seq, normalized_features], axis=-1)
        path = f'{row.sequence}'
        np.savez_compressed(root_dir+'/'+path, wildtype_seq=encoded_pdb_seq)
dgl.save_graphs(root_dir+'/dgl_graph.bin', graphs)

