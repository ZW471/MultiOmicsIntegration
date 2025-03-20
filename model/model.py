import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import FeatureStore
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
                ('cell', m, 'cell'): GATv2Conv(hidden_channels, hidden_channels // 8, heads=8)
                for m in modalities
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

        # Save modalities and create decoders for each modality.
        self.modalities = modalities
        # Here we use a simple nn.Sigmoid as the modality-specific decoder.
        # In practice you might replace this with a more complex module.
        self.decoder_dict = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(latent_channels, 1),
                nn.Sigmoid()
            ) for m in modalities
        })

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
        """
        Compute dot-product scores and pass them through a modality-specific decoder.
        Returns a dictionary mapping each modality to its predicted edge probabilities.
        """
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        # Compute the dot product similarity.
        dot_product = z_src * z_dst
        predicted_edges = {}
        for m in self.modalities:
            # Each modality’s decoder can learn its own transformation.
            predicted_edges[m] = self.decoder_dict[m](dot_product)
        return predicted_edges

    def forward(self, data):
        z_dict = self.encode(data)
        # We assume that we are working with 'cell' nodes.
        z = z_dict['cell']
        return z


class GraphAELightningModule(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, latent_channels, modalities, num_layers,
                 learning_rate, total_epochs, num_clusters, clustering_weight, warmup_epochs=5):
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

        # Initialize clustering parameters.
        self.num_clusters = num_clusters
        self.clustering_weight = clustering_weight
        # Learnable cluster centers (initialized randomly).
        self.cluster_centers = torch.nn.Parameter(torch.randn(num_clusters, latent_channels))

    def forward(self, data):
        return self.model(data)

    def compute_clustering_loss(self, z):
        """
        Compute clustering loss as the mean L2 distance from each latent embedding
        to its nearest cluster center.
        z: Tensor of shape [num_nodes, latent_channels]
        """
        distances = torch.cdist(z, self.cluster_centers, p=2)  # shape: [num_nodes, num_clusters]
        min_distances, _ = torch.min(distances, dim=1)
        return torch.mean(min_distances)

    def training_step(self, batch, batch_idx):
        # Obtain latent embeddings.
        z = self.model(batch)

        # Calculate reconstruction loss per modality.
        total_recon_loss = 0.0
        modality_losses = {}
        # Iterate over each edge type that connects 'cell' to 'cell'.
        for key, pos_edge_index in batch.edge_index_dict.items():
            if key[0] == 'cell' and key[2] == 'cell':
                modality = key[1]  # extract modality from the edge key
                # Sample negative edges for this modality.
                neg_edge_index = negative_sampling(
                    edge_index=pos_edge_index,
                    num_nodes=z.size(0),
                    num_neg_samples=pos_edge_index.size(1)
                )
                # Use the decoder to get predictions.
                # Note: decode returns a dict, so we extract the modality-specific output.
                pos_pred = self.model.decode(z, pos_edge_index)[modality]
                neg_pred = self.model.decode(z, neg_edge_index)[modality]
                preds = torch.cat([pos_pred, neg_pred], dim=0)
                labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)], dim=0)
                modality_loss = F.binary_cross_entropy(preds, labels)
                modality_losses[modality] = modality_loss
                total_recon_loss += modality_loss

        # Compute clustering loss.
        cluster_loss = self.compute_clustering_loss(z)

        # Total loss: reconstruction loss (summed over modalities) + weighted clustering loss.
        loss = total_recon_loss + self.clustering_weight * cluster_loss

        batch_size = batch['cell'].x.size(0)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("cluster_loss", cluster_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        for m, l in modality_losses.items():
            self.log(f"{m}_recon_loss", l, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        def lr_lambda(current_epoch):
            if current_epoch < self.warmup_epochs:
                return float(current_epoch) / float(max(1, self.warmup_epochs))
            else:
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



if __name__ == '__main__':
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    import os
    DATASET_NAME = "LUNG-CITE"
    BASE_DATA_DIR = os.path.join("..", "datasets", "data", "processed")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_path = os.path.join(os.path.join(BASE_DATA_DIR, DATASET_NAME), f"{DATASET_NAME}_processed.pt")
    loaded_data = torch.load(output_path)
    hetero_data = loaded_data.to(DEVICE)  # Move back to GPU if needed

    import torch
    from torch_geometric.loader import NeighborLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hetero_data = hetero_data.to(device)

    num_cells = hetero_data['cell'].x.size(0)
    cell_idx = torch.arange(num_cells, device=device)

    modalities = ['ADT', 'RNA']

    neighbor_loader = NeighborLoader(
        hetero_data,
        num_neighbors={
            ('cell', m, 'cell'): [5, 5] for m in modalities
        },
        input_nodes=('cell', cell_idx),
        batch_size=2048  # choose an appropriate batch size for your memory constraints
    )

    for batch in neighbor_loader:
        print(batch)

    # Hyperparameters.
    in_channels = hetero_data['cell'].x.size(1)
    hidden_channels = 512
    latent_channels = 512   # Dimensionality of the latent space.
    num_layers = 2
    learning_rate = 1e-3
    n_epochs = 500 # change to 500 for full training

    # Instantiate the Lightning module.
    model = GraphAELightningModule(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        latent_channels=latent_channels,
        modalities=modalities,
        num_layers=num_layers,
        learning_rate=learning_rate,
        total_epochs=n_epochs,
        warmup_epochs=3,
        num_clusters=20,
        clustering_weight=.01
    )

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',         # monitor your training loss
        dirpath='checkpoints',        # directory to save checkpoints
        filename='graph_ae-{epoch:02d}-{train_loss:.2f}',
        save_top_k=1,                 # save the best model
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='train_loss',
        min_delta=0.001,
        patience=5,
        verbose=True,
        mode='min'
    )

    trainer = Trainer(
        max_epochs=n_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[early_stop_callback, checkpoint_callback]
    )
    trainer.fit(model, train_dataloaders=neighbor_loader)

    # Inference on full data:
    model.eval()
    with torch.no_grad():
        # Move data to the same device as the model.
        hetero_data = hetero_data.to(model.device)
        z = model(hetero_data)
        # For example, reconstruct edge probabilities using one set of edges.
        pos_edge_index = list(hetero_data.edge_index_dict.values())[0]
        pred_edge_probs = model.model.decode(z, pos_edge_index)
        print(f"nde_embedding: {z}")
        print("Predicted edge probabilities:", pred_edge_probs)

    import scanpy as sc
    latent_embedding = z.to('cpu').detach().numpy()
    adata_eval = sc.AnnData(X=latent_embedding, obs=hetero_data['cell'].metadata.copy())
    adata_eval.obsm["emb"] = latent_embedding


    sc.pp.neighbors(adata_eval, use_rep='emb')         # Build neighbor graph using the latent embedding.
    sc.tl.louvain(adata_eval, resolution=0.5)            # Run Louvain clustering.
    sc.tl.umap(adata_eval)                               # Compute UMAP coordinates.
    sc.pl.embedding(adata_eval, color='louvain', basis='umap')  # Visualize the UMAP colored by Louvain clusters.

    gt = adata_eval.obs['celltype'].tolist()   # True labels.
    pred = adata_eval.obs['louvain'].tolist()     # Louvain cluster labels.

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    ari = adjusted_rand_score(gt, pred)
    nmi = normalized_mutual_info_score(gt, pred)

    print("Adjusted Rand Index:", ari)
    print("Normalized Mutual Information:", nmi)
