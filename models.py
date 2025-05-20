import torch.nn as nn
import torch

import torch.nn.functional as F

def vae_loss(recon, x, mu, logvar):
    # Reconstruction loss + KL divergence
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def contrastive_loss(z, handcrafted_proj, temperature=0.1):
    # Cosine similarity-based InfoNCE loss
    z = F.normalize(z, dim=-1)
    handcrafted_proj = F.normalize(handcrafted_proj, dim=-1)
    logits = torch.matmul(z, handcrafted_proj.T) / temperature
    labels = torch.arange(z.size(0)).to(z.device)
    return F.cross_entropy(logits, labels)

class GridMLMMH(nn.Module):
    def __init__(self, 
                 chord_vocab_size,  # V
                 d_model=512, 
                 nhead=8, 
                 num_layers=8, 
                 dim_feedforward=2048,
                 conditioning_dim=16,
                 pianoroll_dim=100,
                 grid_length=256,
                 dropout=0.3,
                 max_stages=10,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.seq_len = 1 + grid_length + grid_length # condition + melody + harmony
        self.grid_length = grid_length
        # Embedding for condition vector (e.g., style, time sig)
        self.condition_proj = nn.Linear(conditioning_dim, d_model, device=self.device)
        # Melody projection: pianoroll_dim binary -> d_model
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=self.device)
        # Harmony token embedding: V -> d_model
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=self.device)
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, d_model))
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # embedding for curriculum stage
        self.max_stages = max_stages
        self.stage_embedding_dim = 64
        self.stage_embedding = nn.Embedding(self.max_stages, self.stage_embedding_dim, device=self.device)
        # New projection layer to go from (d_model + stage_embedding_dim) → d_model
        self.stage_proj = nn.Linear(self.d_model + self.stage_embedding_dim, self.d_model, device=self.device)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead, 
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(
                        encoder_layer,
                        num_layers=num_layers)
        # Optional: output head for harmonies
        self.output_head = nn.Linear(d_model, chord_vocab_size, device=self.device)
        # Layer norm at input and output
        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.to(device)
    # end init

    def forward(self, conditioning_vec, melody_grid, harmony_tokens=None, stage_indices=None):
        """
        conditioning_vec: (B, C)
        melody_grid: (B, grid_length, pianoroll_dim)
        harmony_tokens: (B, grid_length) - optional for training or inference
        """
        B = conditioning_vec.size(0)

        # Project condition: (B, d_model) → (B, 1, d_model)
        cond_emb = self.condition_proj(conditioning_vec).unsqueeze(1)

        # Project melody: (B, grid_length, pianoroll_dim) → (B, grid_length, d_model)
        melody_emb = self.melody_proj(melody_grid)

        # Harmony token embedding (optional for training): (B, grid_length) → (B, grid_length, d_model)
        if harmony_tokens is not None:
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            # Placeholder (zeros) if not provided
            harmony_emb = torch.zeros(B, self.grid_length, self.d_model, device=self.device)

        # Concatenate full input: (B, 1 + grid_length + grid_length, d_model)
        full_seq = torch.cat([cond_emb, melody_emb, harmony_emb], dim=1)

        # Add positional encoding
        full_seq = full_seq + self.pos_embedding[:, :self.seq_len, :]
        if stage_indices is not None:
            stage_emb = self.stage_embedding(stage_indices)  # (B, stage_embedding_dim)
            stage_emb = stage_emb.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, seq_len, stage_embedding_dim)
            # Concatenate along the feature dimension
            full_seq = torch.cat([full_seq, stage_emb], dim=-1)  # (B, seq_len, d_model + stage_embedding_dim)
            # Project back to d_model
            full_seq = self.stage_proj(full_seq)  # (B, seq_len, d_model)

        full_seq = self.input_norm(full_seq)
        full_seq = self.dropout(full_seq)

        # Transformer encode
        encoded = self.encoder(full_seq)
        encoded = self.output_norm(encoded)

        # Optionally decode harmony logits (only last grid_length tokens)
        harmony_output = self.output_head(encoded[:, -self.grid_length:, :])  # (B, grid_length, V)

        return harmony_output
    # end forward
# end class GridMLMMH

class GuidanceVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dim=256, feature_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.proj_features = nn.Linear(feature_dim, latent_dim)  # For contrastive supervision
    # end init

    def forward(self, x, handcrafted_features):
        # Encode
        h = self.encoder(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)

        # Sample
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Decode
        recon = self.decoder(z)

        # Contrastive projection
        contrast_proj = self.proj_features(handcrafted_features)

        return {
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'recon': recon,
            'features_proj': contrast_proj
        }
    # ebd forward
# end class GuidanceVAE

class GuidedGridMLMMH(GridMLMMH):
    def __init__(self, *args, latent_dim=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.z_proj = nn.Linear(latent_dim, self.d_model)
    # end init

    def forward(self, conditioning_vec, melody_grid, harmony_tokens=None, stage_indices=None, z=None):
        B = conditioning_vec.size(0)

        # Embed inputs as usual
        cond_emb = self.condition_proj(conditioning_vec).unsqueeze(1)
        melody_emb = self.melody_proj(melody_grid)
        if harmony_tokens is not None:
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            harmony_emb = torch.zeros(B, self.grid_length, self.d_model, device=self.device)

        full_seq = torch.cat([cond_emb, melody_emb, harmony_emb], dim=1)

        # Add positional encoding
        full_seq = full_seq + self.pos_embedding[:, :self.seq_len, :]

        # Inject latent vector (broadcasted)
        if z is not None:
            z_proj = self.z_proj(z).unsqueeze(1).repeat(1, self.seq_len, 1)
            full_seq = full_seq + z_proj

        if stage_indices is not None:
            stage_emb = self.stage_embedding(stage_indices).unsqueeze(1).repeat(1, self.seq_len, 1)
            full_seq = torch.cat([full_seq, stage_emb], dim=-1)
            full_seq = self.stage_proj(full_seq)

        full_seq = self.input_norm(full_seq)
        full_seq = self.dropout(full_seq)
        encoded = self.encoder(full_seq)
        encoded = self.output_norm(encoded)
        return self.output_head(encoded[:, -self.grid_length:, :])
    # end forward
# end class GuidedGridMLMMH