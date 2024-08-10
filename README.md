# ToxGIN
based on GIN to predict peptide toxicity
ToxGIN represented the 3D structures of peptides predicted by ColabFold as graphs, with amino acid residues and their interactions serving as nodes and edges, respectively. Next, to leverage the capabilities of the ESM2 protein language model , ToxGIN extracted deep biological features from peptide sequences and further enriched the feature representation of each amino acid node with physicochemical properties. Subsequently, GIN aggregated information from neighboring nodes to extract local and global features, followed by nonlinear transformation to output toxicity prediction probabilities.

