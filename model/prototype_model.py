#%%
import scanpy as sc
# from datasets.preprocess import preprocess_scRNA, preprocess_ADT
#
# # Read the h5ad file created from Seurat
# protein_data = preprocess_scRNA("./datasets/data/processed/LUNG-CITE_RNA.h5ad", h5ad_out="./datasets/data/processed/LUNG-CITE/RNA.h5ad")
# rna_data = preprocess_ADT("./datasets/data/processed/LUNG-CITE_ADT.h5ad", h5ad_out="./datasets/data/processed/LUNG-CITE/ADT.h5ad")
#
# print(protein_data)
# print(rna_data)
#%%

import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import scanpy as sc

modalities = ["ATAC", "RNA"]

data = {}

for modality in modalities:
    # data[modality] = sc.read_h5ad(f"./datasets/data/processed/LUNG-CITE_{modality}.h5ad")
    # data[modality] = sc.read_h5ad(f"./datasets/data/processed/LUNG-CITE/{modality}.h5ad")
    data[modality] = sc.read_h5ad(f"./datasets/data/processed/{modality.lower()}-match.h5ad")
data
#%%
import torch

processed = {m: {'x': torch.tensor(data[m].obsm['X_glue'], dtype=torch.float)} for m in modalities}
processed

#%%
from torch_geometric.data import HeteroData

# (1) Assign attributes after initialization,
data = HeteroData(processed)
data['cell'].x = torch.cat([data[m].x for m in modalities], dim=1)
data
#%%
from torch_geometric.nn import knn_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

for m in modalities:
    data['cell', m, 'cell'].edge_index = knn_graph(
        data[m].x,
        k=10,
        cosine=True,
        num_workers=16
    )
#%%
import torch
from torch_geometric.loader import NeighborLoader

# Assume 'data' is your HeteroData instance and the number of cell nodes is available via:
num_cells = data['cell'].x.size(0)  # or data['cell'].num_nodes if set

# Create indices for all cells:
cell_idx = torch.arange(num_cells)

# Create a NeighborLoader that samples neighborhoods for 'cell' nodes.
# Here, we specify how many neighbors to sample for each edge type.
neighbor_loader = NeighborLoader(
    data,
    num_neighbors={
        ('cell', m, 'cell'): [5, 5] for m in modalities
    },
    input_nodes=('cell', cell_idx),
    batch_size=2048  # choose an appropriate batch size for your memory constraints
)

for batch in neighbor_loader:
    print(batch)
#%%
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import HeteroConv, GCNConv, GATv2Conv
from torch_geometric.utils import negative_sampling

# Heterogeneous Graph Autoencoder model (without variational reparameterization).
class HeteroGraphAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, modalities, num_layers=2, **kwargs):
        super().__init__()
        # First heterogeneous convolution layer.
        self.conv1 = HeteroConv({
            ('cell', m, 'cell'): GCNConv(in_channels, hidden_channels) for m in modalities
        }, aggr='sum')
        self.bn1 = nn.ModuleDict({'cell': nn.BatchNorm1d(hidden_channels)})

        # Intermediate layers (if num_layers > 1).
        self.layers = nn.ModuleList([
            HeteroConv({
                ('cell', m, 'cell'): GATv2Conv(hidden_channels, hidden_channels // 8, heads=8) for m in modalities
            }, aggr='sum')
            for _ in range(num_layers - 1)
        ])
        self.bn_layers = nn.ModuleList([
            nn.ModuleDict({'cell': nn.BatchNorm1d(hidden_channels)}) for _ in range(num_layers - 1)
        ])

        # Single heterogeneous convolution to compute the latent representation.
        self.z_conv = HeteroConv({
            ('cell', m, 'cell'): GCNConv(hidden_channels, latent_channels) for m in modalities
        }, aggr='sum')


    def encode(self, data):
        x_dict = {'cell': data['cell'].x}
        # First layer: conv -> batch norm -> activation.
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {k: self.bn1[k](x) for k, x in x_dict.items()}
        x_dict = {k: F.silu(x) for k, x in x_dict.items()}

        # Intermediate layers.
        for bn_layer, layer in zip(self.bn_layers, self.layers):
            x_dict = layer(x_dict, data.edge_index_dict)
            x_dict = {k: bn_layer[k](x) for k, x in x_dict.items()}
            x_dict = {k: F.silu(x) for k, x in x_dict.items()}

        # Compute the latent representation.
        z_dict = self.z_conv(x_dict, data.edge_index_dict)
        return z_dict

    def decode(self, z, edge_index):
        # Dot-product decoder: compute similarity scores for given edges.
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        return torch.sigmoid((z_src * z_dst).sum(dim=1))

    def forward(self, data):
        z_dict = self.encode(data)
        z = z_dict['cell']
        return z

class GraphAELightningModule(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, latent_channels, modalities, num_layers, learning_rate, total_epochs, num_clusters, clustering_weight, warmup_epochs=5):
        super().__init__()
        self.save_hyperparameters(ignore=['modalities'])
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

        self.model = HeteroGraphAE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels,
            modalities=modalities,
            num_layers=num_layers
        )
        self.learning_rate = learning_rate

        # Initialize clustering parameters
        self.num_clusters = num_clusters
        self.clustering_weight = clustering_weight
        # Learnable cluster centers (initialized randomly)
        self.cluster_centers = torch.nn.Parameter(torch.randn(num_clusters, latent_channels))

    def forward(self, data):
        return self.model(data)

    def compute_clustering_loss(self, z):
        """
        Compute clustering loss as the mean L2 distance from each latent embedding
        to its nearest cluster center.
        z: Tensor of shape [num_nodes, latent_channels]
        """
        # Compute pairwise Euclidean distances between embeddings and cluster centers.
        distances = torch.cdist(z, self.cluster_centers, p=2)  # shape: [num_nodes, num_clusters]
        # For each embedding, take the distance to the closest center.
        min_distances, _ = torch.min(distances, dim=1)
        # Clustering loss is the average of these minimum distances.
        return torch.mean(min_distances)

    def training_step(self, batch, batch_idx):
        # Obtain latent embeddings.
        z = self.model(batch)

        # Aggregate positive edges from all 'cell'–to–'cell' relations.
        pos_edge_list = []
        for key, edge_index in batch.edge_index_dict.items():
            if key[0] == 'cell' and key[2] == 'cell':
                pos_edge_list.append(edge_index)
        if len(pos_edge_list) == 0:
            raise ValueError("No 'cell'-to-'cell' edges found in data.edge_index_dict.")
        pos_edge_index = torch.cat(pos_edge_list, dim=1)

        # Sample negative edges.
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=z.size(0),
            num_neg_samples=pos_edge_index.size(1)
        )

        # Compute predictions for positive and negative edges.
        pos_pred = self.model.decode(z, pos_edge_index)
        neg_pred = self.model.decode(z, neg_edge_index)
        preds = torch.cat([pos_pred, neg_pred], dim=0)
        labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)], dim=0)
        recon_loss = F.binary_cross_entropy(preds, labels)

        # Compute clustering loss.
        cluster_loss = self.compute_clustering_loss(z)

        # Total loss: reconstruction loss + weighted clustering loss.
        loss = recon_loss + self.clustering_weight * cluster_loss

        batch_size = batch['cell'].x.size(0)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("cluster_loss", cluster_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        def lr_lambda(current_epoch):
            if current_epoch < self.warmup_epochs:
                # Linear warm-up.
                return float(current_epoch) / float(max(1, self.warmup_epochs))
            else:
                # Cosine decay.
                progress = (current_epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    #
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer,
    #         mode='min',         # Assumes lower validation loss is better.
    #         factor=0.1,         # Factor by which the learning rate will be reduced.
    #         patience=10,        # Number of epochs with no improvement after which LR will be reduced.
    #         verbose=True
    #     )
    #
    #     return {
    #         'optimizer': optimizer,
    #         'lr_scheduler': {
    #             'scheduler': scheduler,
    #             'monitor': 'recon_loss',  # Metric to be monitored for plateau.
    #             'interval': 'epoch',
    #             'frequency': 1,
    #         }
    #     }


# Hyperparameters.
in_channels = 512
hidden_channels = 512
latent_channels = 256   # Dimensionality of the latent space.
num_layers = 2
learning_rate = 1e-3
n_epochs = 500

# Ensure that `modalities` and your dataloader (e.g., neighbor_loader) are defined.
# For example:
# modalities = ['modality1', 'modality2']
# neighbor_loader = your_dataloader_object

# Instantiate the Lightning module.
model = GraphAELightningModule(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    latent_channels=latent_channels,
    modalities=modalities,
    num_layers=num_layers,
    learning_rate=learning_rate,
    total_epochs=n_epochs,
    warmup_epochs=10,
    num_clusters=20,
    clustering_weight=.1
)

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

early_stop_callback = EarlyStopping(
    monitor='train_loss',
    min_delta=0.001,
    patience=3,
    verbose=True,
    mode='min'
)

trainer = Trainer(
    max_epochs=n_epochs,
    accelerator="gpu",
    devices=1,
    callbacks=[early_stop_callback]
)
trainer.fit(model, train_dataloaders=neighbor_loader)


# Inference on full data:
model.eval()
with torch.no_grad():
    # Move data to the same device as the model.
    data = data.to(model.device)
    z = model(data)
    # For example, reconstruct edge probabilities using one set of edges.
    pos_edge_index = list(data.edge_index_dict.values())[0]
    pred_edge_probs = model.model.decode(z, pos_edge_index)
    print(f"nde_embedding: {z}")
    print("Predicted edge probabilities:", pred_edge_probs)

#%%
protein_data = sc.read_h5ad("./datasets/data/processed/rna-match.h5ad")
protein_data.obsm["emb"] = z.detach().cpu().numpy()
# protein_data.obsm["emb"] = data['cell'].x.detach().cpu().numpy()
sc.pp.neighbors(protein_data, use_rep='emb')
sc.tl.louvain(protein_data, resolution=0.5)
sc.tl.umap(protein_data)
sc.pl.embedding(protein_data, color='louvain', basis='umap')

#%%
gt = protein_data.obs['cell_type'].tolist()
pred = protein_data.obs['louvain'].tolist()

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari = adjusted_rand_score(gt, pred)
nmi = normalized_mutual_info_score(gt, pred)

print("Adjusted Rand Index:", ari)
print("Normalized Mutual Information:", nmi)
