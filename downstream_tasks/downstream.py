import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class CrossModalPredictor(nn.Module):
    """
    An MLP head that takes the shared latent embedding and predicts
    the features of a target modality.
    """
    def __init__(self, latent_channels, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_channels, latent_channels // 2),
            nn.ReLU(),
            nn.Linear(latent_channels // 2, output_dim)
        )
        
    def forward(self, z):
        return self.mlp(z)

class DownstreamLightningModule(pl.LightningModule):
    """
    A Lightning module for training the cross-modal predictor.
    The goal is to predict the features of one modality (e.g. ADT)
    from the shared latent embedding.
    """
    def __init__(self, latent_channels, output_dim, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.predictor = CrossModalPredictor(latent_channels, output_dim)
        self.learning_rate = learning_rate

    def forward(self, z):
        return self.predictor(z)

    def training_step(self, batch, batch_idx):
        # Batch is a tuple: (latent_embedding, target_features)
        z, target = batch
        pred = self(z)
        loss = F.mse_loss(pred, target)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        z, target = batch
        pred = self(z)
        loss = F.mse_loss(pred, target)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer