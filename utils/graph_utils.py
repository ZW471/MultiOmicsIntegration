import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy.sparse import csr_matrix
import torch

def calculate_modality_jaccard(edge_index):
    """Calculate self-consistency with full node coverage"""
    edges = edge_index.cpu().numpy().T
    all_nodes = np.unique(edges)
    
    adj_dict = {node: set() for node in all_nodes}
    for src, tgt in edges:
        adj_dict[src].add(tgt)
    
    jaccard_scores = []
    for node in all_nodes:
        direct_neighbors = adj_dict[node]
        if not direct_neighbors:
            jaccard_scores.append(0.0)
            continue
            
        second_order = set()
        for neighbor in direct_neighbors:
            second_order.update(adj_dict.get(neighbor, set()))
            
        intersection = len(direct_neighbors & second_order)
        union = len(direct_neighbors | second_order)
        jaccard = intersection / union if union > 0 else 0
        jaccard_scores.append(jaccard)
        
    return np.mean(jaccard_scores)

def plot_degree_distributions(hetero_data, modalities):
    """Plot degree distributions for multiple modalities"""
    fig, axes = plt.subplots(1, len(modalities), figsize=(15, 4))
    for ax, modality in zip(axes, modalities):
        edges = hetero_data['cell', modality, 'cell'].edge_index
        degrees = torch.unique(edges[0], return_counts=True)[1].cpu().numpy()
        
        sns.histplot(degrees, bins=30, ax=ax)
        ax.set_title(f"{modality} Degree Distribution\n(Mean={degrees.mean():.1f})")
        ax.set_xlabel("Number of Edges per Cell")
    plt.tight_layout()
    return fig

def plot_modality_umaps(data_dict, modalities):
    """Generate UMAP projections for multiple modalities"""
    fig, axes = plt.subplots(1, len(modalities), figsize=(15, 4))
    for ax, modality in zip(axes, modalities):
        adata = data_dict[modality]
        use_rep = 'X_pca' if 'X_pca' in adata.obsm else 'X'
        
        sc.pp.neighbors(adata, n_neighbors=10, use_rep=use_rep)
        sc.tl.umap(adata)
        sc.pl.umap(adata, ax=ax, show=False)
        ax.set_title(f"{modality} UMAP Projection")
    plt.tight_layout()
    return fig