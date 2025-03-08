import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_cluster import knn_graph
import pytorch_lightning as pl
from torch.nn import Linear
import scanpy as sc
from datasets.preprocess import remove_lsi_key, preprocess

# ---------------------------
# Define the LightningDataModule
# ---------------------------
class MultiOmicsDataModule(pl.LightningDataModule):
    def __init__(self, data, modalities, batch_size=2048, shuffle=True):
        super().__init__()
        self.data = data  # expects the noised data with original clean_x stored
        self.modalities = modalities
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Define the neighbor sampling configuration.
        self.sampler_config = {
            ('adt', 'similar_to', 'adt'): [8, 4],
            ('adt', 'belongs_to', 'cell'): [1, 0],
            ('RNA', 'similar_to', 'RNA'): [8, 4],
            ('RNA', 'belongs_to', 'cell'): [1, 0],
            ('atac', 'similar_to', 'atac'): [8, 4],
            ('atac', 'belongs_to', 'cell'): [1, 0],
            ('cell', 'similar_to', 'cell'): [8, 4],
        }

    def setup(self, stage=None):
        # No additional setup is required since the entire graph was already noised.
        pass

    def train_dataloader(self):
        sampler = NeighborLoader(
            self.data,
            num_neighbors=self.sampler_config,
            input_nodes=('cell', self.data['cell']['train_mask']),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
        return sampler

# ---------------------------
# Model Definitions (Encoder, Decoder, and Message Passing layers)
# ---------------------------
from torch_geometric.nn import MessagePassing

# --- New: Dropout-enhanced Cell-to-cell Message Passing Module ---
class CellMP(MessagePassing):
    def __init__(self, cell_dim, hidden_dim, dropout=0.1):
        super(CellMP, self).__init__(aggr='add')
        self.fc_message = nn.Sequential(
            nn.Linear(2 * cell_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.fc_update = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        m = torch.cat([x_i, x_j], dim=-1)
        return self.activation(self.fc_message(m))

    def update(self, aggr_out):
        return self.fc_update(aggr_out)

# --- Dropout-enhanced IntraModality Message Passing with optional edge gating ---
class IntraModalityMP(MessagePassing):
    def __init__(self, node_dim, edge_attr_dim, hidden_dims, dropout=0.1):
        super(IntraModalityMP, self).__init__(aggr='add')
        self.fc_message = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_attr_dim, hidden_dims),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, hidden_dims),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.fc_update = nn.Linear(hidden_dims, hidden_dims)
        self.activation = nn.SiLU()
        # Optional: edge gating if desired.
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_attr_dim, edge_attr_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_attr_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        m_ij = torch.cat([x_i, x_j, edge_attr], dim=-1)
        # Uncomment below to enable gating.
        # a_ij = self.edge_gate(edge_attr)
        # return self.activation(self.fc_message(m_ij)) * a_ij
        return self.activation(self.fc_message(m_ij))

    def update(self, aggr_out):
        return self.fc_update(aggr_out)

# --- Dropout-enhanced Modality-to-Cell Message Passing Module ---
class ModalityToCellMP(MessagePassing):
    def __init__(self, node_dim, cell_dim, hidden_cell_dim, dropout=0.1):
        super(ModalityToCellMP, self).__init__(aggr='add')
        self.cell_gate = nn.Sequential(
            nn.Linear(cell_dim, cell_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(cell_dim, 1),
            nn.Sigmoid()
        )
        self.fc_message = nn.Sequential(
            nn.Linear(node_dim + cell_dim, hidden_cell_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.fc_update = nn.Linear(hidden_cell_dim, hidden_cell_dim)
        self.activation = nn.SiLU()

    def forward(self, x, c, edge_index):
        return self.propagate(edge_index, x=x, c=c)

    def message(self, c_i, x_j):
        a = self.cell_gate(c_i)
        m_ij = self.fc_message(torch.cat([x_j, c_i], dim=-1))
        return self.activation(m_ij) * a

    def update(self, aggr_out):
        return self.fc_update(aggr_out)

# --- New: Message Passing from Cell to Modalities with Multi-head Attention ---
class CellToModalityMP(MessagePassing):
    def __init__(self, cell_dim, node_dim, hidden_node_dim, num_heads=4, dropout=0.1):
        super(CellToModalityMP, self).__init__(aggr='add')
        self.num_heads = num_heads
        self.hidden_dim_per_head = hidden_node_dim // num_heads
        # Linear projections for multi-head attention.
        self.query_lin = nn.Linear(node_dim, hidden_node_dim)
        self.key_lin = nn.Linear(cell_dim, hidden_node_dim)
        self.value_lin = nn.Linear(cell_dim, hidden_node_dim)
        self.out_lin = nn.Linear(hidden_node_dim, hidden_node_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()
        self.modality_gate = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, cell, modality, edge_index):
        return self.propagate(edge_index, c=cell, m=modality)

    def message(self, c_j, m_i):
        # Multi-head attention: compute query, key, value.
        Q = self.query_lin(m_i)  # shape: (num_edges, hidden_node_dim)
        K = self.key_lin(c_j)    # shape: (num_edges, hidden_node_dim)
        V = self.value_lin(c_j)  # shape: (num_edges, hidden_node_dim)
        # Reshape for multi-head attention.
        Q = Q.view(-1, self.num_heads, self.hidden_dim_per_head)
        K = K.view(-1, self.num_heads, self.hidden_dim_per_head)
        V = V.view(-1, self.num_heads, self.hidden_dim_per_head)
        # Compute scaled dot-product attention scores.
        attn_scores = (Q * K).sum(dim=-1) / (self.hidden_dim_per_head ** 0.5)  # (num_edges, num_heads)
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (num_edges, num_heads, 1)
        attn_output = (attn_weights * V).view(-1, self.num_heads * self.hidden_dim_per_head)
        out = self.out_lin(attn_output)
        out = self.dropout(out)
        a = self.modality_gate(m_i)
        return self.activation(out) * a

    def update(self, aggr_out):
        return aggr_out

# --- MultiOmics Embedding ---
class MultiOmicsEmbedding(nn.Module):
    def __init__(self, modality_in_dims, cell_in_dims, edge_attr_dims, modality_hidden_dims, modalities, cell_hidden_dims):
        super().__init__()
        self.activation = nn.SiLU()
        self.modality_embs = nn.ModuleDict({
            m: IntraModalityMP(
                node_dim=modality_in_dims[m],
                edge_attr_dim=edge_attr_dims[m],
                hidden_dims=modality_hidden_dims,
            ) for m in modalities
        })
        self.cell_emb = nn.Sequential(
            Linear(cell_in_dims, cell_hidden_dims),
            self.activation,
        )
        self.modalities = modalities  # Save modalities for potential use

    def forward(self, batch):
        H = {}
        for m in self.modality_embs.keys():
            H[m] = self.modality_embs[m](batch[m].x, batch[m, 'similar_to', m].edge_index, batch[m, 'similar_to', m].e)
        C = self.cell_emb(batch['cell'].x)
        return H, C

# --- MultiOmics Layer ---
class MultiOmicsLayer(nn.Module):
    def __init__(self, modality_in_dims, cell_in_dims, edge_attr_dims, modality_hidden_dims, cell_hidden_dims, modalities):
        super().__init__()
        self.activation = nn.SiLU()
        self.modalities = modalities  # store modalities for use in forward

        self.intra_modality_msg = nn.ModuleDict({
            m: IntraModalityMP(
                node_dim=modality_in_dims[m],
                edge_attr_dim=edge_attr_dims[m],
                hidden_dims=modality_hidden_dims,
            ) for m in modalities
        })
        self.modality_to_cell_msg = nn.ModuleDict({
            m: ModalityToCellMP(
                node_dim=modality_hidden_dims,
                cell_dim=cell_in_dims,
                hidden_cell_dim=cell_hidden_dims,
            ) for m in modalities
        })
        self.cell_to_modality_msg = nn.ModuleDict({
            m: CellToModalityMP(
                cell_dim=cell_in_dims,
                node_dim=modality_hidden_dims,
                hidden_node_dim=modality_hidden_dims,
            ) for m in modalities
        })
        # --- New: Cell-to-cell message passing module ---
        self.cell_mp = CellMP(cell_dim=cell_hidden_dims, hidden_dim=cell_hidden_dims)

    def forward(self, batch, H, C):
        # Store skip connections.
        H_skip = {m: H[m].clone() for m in self.modalities}
        C_skip = C.clone()

        # Intra-modality update and modality-to-cell update.
        for m in self.modalities:
            H_update = self.intra_modality_msg[m](H[m], batch[m, 'similar_to', m].edge_index, batch[m, 'similar_to', m].e)
            H[m] = H[m] + H_update
            C_update = self.modality_to_cell_msg[m](H[m], C, batch[m, 'belongs_to', 'cell'].edge_index)
            C = C + C_update

        # Cell-to-modality update.
        for m in self.modalities:
            cell_to_modality_edge_index = batch[m, "belongs_to", "cell"].edge_index[[1, 0]]
            H_update2 = self.cell_to_modality_msg[m](C, H[m], cell_to_modality_edge_index)
            H[m] = H[m] + H_update2

        # --- New: Update cell embeddings using cell-to-cell message passing ---
        C_mp = self.cell_mp(C, batch['cell', 'similar_to', 'cell'].edge_index)
        C = C + C_mp

        # Add skip connections from the input of the layer.
        for m in self.modalities:
            H[m] = H[m] + H_skip[m]
        C = C + C_skip

        return H, C

# --- MultiOmics Integration ---
class MultiOmicsIntegration(nn.Module):
    def __init__(self, modality_in_dims, cell_in_dims, edge_attr_dims, modality_hidden_dims, cell_hidden_dims,
                 modalities, layer_num=2):
        super().__init__()
        self.modality_hidden_dims = modality_hidden_dims
        self.cell_hidden_dims = cell_hidden_dims
        self.modalities = modalities
        self.layer_num = layer_num

        self.embedding = MultiOmicsEmbedding(
            modality_in_dims=modality_in_dims,
            cell_in_dims=cell_in_dims,
            edge_attr_dims=edge_attr_dims,
            modality_hidden_dims=modality_hidden_dims,
            modalities=modalities,
            cell_hidden_dims=cell_hidden_dims,
        )

        self.layers = nn.ModuleList([
            MultiOmicsLayer(
                modality_in_dims={m: modality_hidden_dims for m in modalities},
                cell_in_dims=cell_hidden_dims,
                edge_attr_dims=edge_attr_dims,
                modality_hidden_dims=modality_hidden_dims,
                cell_hidden_dims=cell_hidden_dims,
                modalities=modalities
            ) for _ in range(layer_num)
        ])

        self.modality_bn = nn.ModuleDict({
            m: nn.BatchNorm1d(modality_hidden_dims) for m in modalities
        })

        self.cell_bn = nn.BatchNorm1d(cell_hidden_dims)

    def normalize_each_modality(self, H):
        for m in self.modalities:
            H[m] = self.modality_bn[m](H[m])
        return H

    def forward(self, batch):
        H, C = self.embedding(batch)
        for layer in self.layers:
            # Save skip connections from the previous layer's outputs.
            H_layer_skip = {m: H[m].clone() for m in self.modalities}
            C_layer_skip = C.clone()

            H, C = layer(batch, H, C)

            # Add the layer-level skip connection.
            for m in self.modalities:
                H[m] = H[m] + H_layer_skip[m]
            C = C + C_layer_skip

            H = self.normalize_each_modality(H)
            C = self.cell_bn(C)
        return H, C

# --- Updated Decoder with Cell–Cell Edge Reconstruction ---
class MultiOmicsDecoder(nn.Module):
    def __init__(self, encoder, modality_in_dims, cell_in_dims, modalities):
        super().__init__()
        self.modalities = modalities
        # Increase decoder complexity with extra hidden layers.
        self.modality_decoders = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(encoder.modality_hidden_dims, encoder.modality_hidden_dims),
                nn.SiLU(),
                nn.Linear(encoder.modality_hidden_dims, modality_in_dims[m]),
                nn.SiLU(),
            )
            for m in modalities
        })
        self.cell_decoder = nn.Sequential(
            nn.Linear(encoder.cell_hidden_dims, encoder.cell_hidden_dims),
            nn.SiLU(),
            nn.Linear(encoder.cell_hidden_dims, cell_in_dims),
            nn.SiLU(),
        )
        # --- New: Decoder branch for reconstructing cell->cell edges ---
        self.cell_edge_decoder = nn.Sequential(
            nn.Linear(encoder.cell_hidden_dims * 2, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, H, C, cell_edge_index=None):
        modality_recons = {m: self.modality_decoders[m](H[m]) for m in self.modalities}
        cell_recon = self.cell_decoder(C)
        cell_edge_pred = None
        if cell_edge_index is not None:
            # Use the cell embeddings from the two nodes in each edge.
            src, tgt = cell_edge_index
            cell_src = C[src]
            cell_tgt = C[tgt]
            edge_input = torch.cat([cell_src, cell_tgt], dim=-1)
            cell_edge_pred = self.cell_edge_decoder(edge_input)
        return modality_recons, cell_recon, cell_edge_pred

# ---------------------------
# Define the LightningModule (Model + Training Step)
# ---------------------------
class MultiOmicsLitModel(pl.LightningModule):
    def __init__(self, data, modalities):
        super().__init__()
        self.modalities = modalities

        # Initialize the encoder.
        self.encoder = MultiOmicsIntegration(
            modality_in_dims={m: data[m].x.shape[1] for m in modalities},
            edge_attr_dims={m: data[m, 'similar_to', m].e.shape[1] for m in modalities},
            cell_in_dims=data['cell'].x.shape[1],
            cell_hidden_dims=256,  # latent dimension for cell embeddings
            modality_hidden_dims=128,
            modalities=modalities,
            layer_num=4,
        )

        # Initialize the decoder.
        self.decoder = MultiOmicsDecoder(
            encoder=self.encoder,
            modality_in_dims={m: data[m].x.shape[1] for m in modalities},
            cell_in_dims=data['cell'].x.shape[1],
            modalities=modalities,
        )

        self.learning_rate = 1e-3

        # ----- New: Cluster Loss parameters -----
        self.num_clusters = 10  # Adjust the number of clusters as needed.
        self.lambda_cluster = 1.0  # Weighting factor for the cluster loss.
        # Initialize cluster centers as trainable parameters.
        self.cluster_centers = nn.Parameter(torch.randn(self.num_clusters, self.encoder.cell_hidden_dims))

    def forward(self, batch):
        # Get encoder outputs: H is per-modality latent features and C is the latent cell embedding.
        H, C = self.encoder(batch)
        # The decoder returns:
        # - modality_recons: reconstruction for each modality,
        # - cell_recon: decoded cell features,
        # - pos_cell_edge_pred: positive cell–cell edge predictions (computed using C).
        modality_recons, cell_recon, pos_cell_edge_pred = self.decoder(
            H, C, batch['cell', 'similar_to', 'cell'].edge_index
        )
        # Return the decoded cell features along with the latent cell embedding C.
        return modality_recons, cell_recon, pos_cell_edge_pred, C

    def training_step(self, batch, batch_idx):
        # Forward pass: retrieve encoder outputs and predictions.
        modality_recons, cell_recon, pos_cell_edge_pred, C = self.forward(batch)

        # Reconstruction losses for modalities and cell node features.
        loss_modality = sum(F.mse_loss(modality_recons[m], batch[m].clean_x) for m in self.modalities) * 0.1
        loss_cell = F.mse_loss(cell_recon, batch['cell'].clean_x)

        # Number of positive cell–cell edges from the batch.
        num_pos = batch['cell', 'similar_to', 'cell'].edge_index.size(1)

        # ----- Advanced Negative Sampling (Hard Negatives) -----
        pos_edge_index = batch['cell', 'similar_to', 'cell'].edge_index  # shape: [2, num_pos]
        num_candidates = 5
        # For each positive edge, fix the source and sample candidate target indices.
        neg_src_candidates = pos_edge_index[0].unsqueeze(1).repeat(1, num_candidates)
        candidate_tgt = torch.randint(0, C.size(0), (num_pos, num_candidates), device=C.device)
        # Compute distances between the source and candidate targets.
        src_emb = C[neg_src_candidates]  # shape: (num_pos, num_candidates, cell_hidden_dims)
        tgt_emb_candidates = C[candidate_tgt]  # shape: (num_pos, num_candidates, cell_hidden_dims)
        dist_candidates = torch.norm(src_emb - tgt_emb_candidates, dim=-1)  # shape: (num_pos, num_candidates)
        # Choose the candidate with the minimum distance (i.e. the hardest negative).
        min_idx = dist_candidates.argmin(dim=1)  # shape: (num_pos,)
        neg_tgt = candidate_tgt[torch.arange(num_pos), min_idx]
        # Use the same source nodes as the positive edges.
        neg_edge_input = torch.cat([C[pos_edge_index[0]], C[neg_tgt]], dim=-1)
        neg_cell_edge_pred = self.decoder.cell_edge_decoder(neg_edge_input)

        # Combine positive and negative predictions and create labels.
        all_preds = torch.cat([pos_cell_edge_pred, neg_cell_edge_pred], dim=0)
        all_labels = torch.cat([torch.ones_like(pos_cell_edge_pred), torch.zeros_like(neg_cell_edge_pred)], dim=0)
        loss_cell_edges = F.binary_cross_entropy(all_preds, all_labels) * 10

        # ----- New: Cluster Loss on C using Soft Assignments -----
        distances = torch.cdist(C, self.cluster_centers, p=2)
        soft_assignments = F.softmax(-distances, dim=1)  # Closer centers get higher weights.
        loss_cluster = (soft_assignments * distances).sum(dim=1).mean() * self.lambda_cluster

        # Total loss.
        loss = loss_modality + loss_cell + loss_cell_edges + loss_cluster

        # Explicitly provide the batch size from the cell nodes to avoid iteration over batch.
        batch_size = batch['cell'].x.size(0)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("cell_loss", loss_cell, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("modality_loss", loss_modality, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("cell_edge_loss", loss_cell_edges, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("cluster_loss", loss_cluster, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()) + [self.cluster_centers],
            lr=self.learning_rate,
            weight_decay=1e-4  # Added weight decay for regularization.
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'frequency': 1
            }
        }

# ---------------------------
# Main execution: instantiate DataModule, model and start training
# ---------------------------
if __name__ == '__main__':
    # ---------------------------
    # Data preparation and graph building
    # ---------------------------
    modalities = ["adt", "RNA", "atac"]
    # (The following data loading and processing code remains unchanged.)
    XEC = torch.load("../datasets/data/processed/PBMC-DOGMA/XEC.pt", weights_only=False)
    X = XEC['X']
    E = XEC['E']
    C = XEC['C']

    edge_indices = {}
    for m in modalities:
        edge_indices[m] = knn_graph(
            X[m],
            k=min(X[m].shape[0] // 20, 50),
            cosine=True,
            num_workers=16
        )

    data = HeteroData()
    # Cell nodes.
    data['cell'].x = C
    data['cell', 'similar_to', 'cell'].edge_index = knn_graph(
        C,
        k=min(100, C.shape[0] // 200),
        cosine=True,
        num_workers=16
    )

    for m in modalities:
        data[m].x = X[m]
        data[m, "similar_to", m].edge_index = edge_indices[m]
        data[m, "belongs_to", "cell"].edge_index = torch.stack([
            torch.arange(C.shape[0]),
            torch.arange(C.shape[0])
        ], dim=0)
        data[m, "similar_to", m].e = E[m] if len(E[m].shape) > 1 else E[m].unsqueeze(-1)

    if 'train_mask' not in data['cell']:
        data['cell']['train_mask'] = torch.ones(data['cell'].x.size(0), dtype=torch.bool)

    # Save original (clean) features.
    data['cell'].clean_x = data['cell'].x.clone()
    for m in modalities:
        data[m].clean_x = data[m].x.clone()

    def add_noise(graph: HeteroData, modalities, noise_std=0.1, drop_modality_prob=0.5):
        noisy_graph = copy.deepcopy(graph)
        noise_mask = torch.rand(noisy_graph['cell'].x.shape[0]) > 0.1
        for m in modalities:
            if 'x' in noisy_graph[m]:
                original = noisy_graph[m].x[noise_mask]
                if torch.rand(1).item() < drop_modality_prob:
                    noisy_graph[m].x[noise_mask] = noise_std * torch.randn_like(original)
                else:
                    noisy_graph[m].x[noise_mask] = original + noise_std * torch.randn_like(original)
        if 'x' in noisy_graph['cell']:
            original = noisy_graph['cell'].x[noise_mask]
            noisy_graph['cell'].x[noise_mask] = original + noise_std * torch.randn_like(original)
        return noisy_graph

    # Apply noise.
    noisy_data = add_noise(data, modalities, noise_std=0.1, drop_modality_prob=0.5)

    # For training, you can switch between noisy_data and data as needed.
    dm = MultiOmicsDataModule(data, modalities, batch_size=2048, shuffle=True)
    model = MultiOmicsLitModel(data, modalities)

    early_stop_callback = EarlyStopping(
        monitor='train_loss',
        min_delta=0.001,
        patience=10,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=300,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=1,
        callbacks=[early_stop_callback]
    )
    trainer.fit(model, dm)

    # ---------------------------
    # Final evaluation: Output cell embeddings and cell->cell edge predictions.
    # ---------------------------
    clean_data = data.to(model.device)
    model.eval()
    with torch.no_grad():
        modality_recons, cell_recon, cell_edge_pred, C = model(clean_data)
        print("Cell embeddings shape:", C.shape)
        print("Cell->cell edge predictions:", cell_edge_pred.shape)
        # Save the cell embeddings and edge predictions.
        torch.save(cell_recon, "cell_embeddings.pt")
        torch.save(cell_edge_pred, "cell_edge_predictions.pt")

    # ---------------------------
    # Downstream analysis with Scanpy (unchanged)
    # ---------------------------
    protein_data = sc.read_h5ad("../datasets/data/processed/PBMC-DOGMA_adt.h5ad")
    protein_data.obsm["emb"] = C.detach().cpu().numpy()
    sc.pp.neighbors(protein_data, use_rep='emb')
    sc.tl.louvain(protein_data, resolution=0.5)
    sc.tl.umap(protein_data)
    sc.pl.embedding(protein_data, color='louvain', basis='umap')

    gt = protein_data.obs['celltype'].tolist()
    pred = protein_data.obs['louvain'].tolist()

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    ari = adjusted_rand_score(gt, pred)
    nmi = normalized_mutual_info_score(gt, pred)

    print("Adjusted Rand Index:", ari)
    print("Normalized Mutual Information:", nmi)
