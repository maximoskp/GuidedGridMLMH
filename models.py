import torch
import torch.nn as nn
import torch.nn.functional as F

class GuidanceVAE(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            latent_dim,
            embedding_dim,
            seq_len,
            feature_dim,
            chord_vocab_size,
            device='cpu'
        ):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self.embedding = nn.Embedding(chord_vocab_size, embedding_dim, device=device)
        self.encoder_rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

        self.decoder_rnn = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.recon_proj = nn.Linear(hidden_dim, chord_vocab_size)

        self.feature_proj = nn.Linear(latent_dim, feature_dim)
    # end init

    def encode(self, harmony_tokens):
        emb = self.embedding(harmony_tokens)  # (B, L, E)
        _, (h_n, _) = self.encoder_rnn(emb)
        h = h_n.squeeze(0)
        mu = self.mu_proj(h)
        logvar = self.logvar_proj(h)
        return mu, logvar
    # end encode

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    # end parametrize

    def decode(self, z):
        z_seq = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        output, _ = self.decoder_rnn(z_seq)
        return self.recon_proj(output)
    # end decode
    def forward(self, harmony_tokens):
        mu, logvar = self.encode(harmony_tokens)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        z_proj = self.feature_proj(z)
        return z, mu, logvar, recon_x, z_proj
    # end forward
# end class GuidanceVAE

class GridMLMMelHarmEncoder(nn.Module):
    def __init__(self, chord_vocab_size, d_model, nhead, num_layers,
                 stage_embedding_dim, max_stages, device='cpu'):
        super().__init__()
        self.device = device

        self.stage_embedding = nn.Embedding(max_stages, stage_embedding_dim, device=device)
        self.stage_proj = nn.Linear(d_model + stage_embedding_dim, d_model, device=device)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=4*d_model, dropout=0.1,
                                                   activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, chord_vocab_size, device=device)
    # end init

    def forward(self, full_seq, stage_indices):
        if stage_indices is not None:
            stage_emb = self.stage_embedding(stage_indices).unsqueeze(1).repeat(1, full_seq.size(1), 1)
            full_seq = torch.cat([full_seq, stage_emb], dim=-1)
            full_seq = self.stage_proj(full_seq)

        full_seq = self.input_norm(full_seq)
        encoded = self.encoder(full_seq)
        encoded = self.output_norm(encoded)
        return self.output_head(encoded[:, -256:, :])
    # end forward
# end class GridMLMMelHarmEncoder

class GuidedMLMH(nn.Module):
    def __init__(self, vae_cfg, encoder_cfg,
                 chord_vocab_size, d_model,
                 conditioning_dim, pianoroll_dim, grid_length,
                 guidance_dim, unfold_latent=True, device='cpu'):
        super().__init__()
        self.device = device
        self.grid_length = grid_length
        self.seq_len = 1 + 2 * grid_length
        self.unfold_latent = unfold_latent

        self.condition_proj = nn.Linear(conditioning_dim, d_model, device=device)
        self.melody_proj = nn.Linear(pianoroll_dim, d_model, device=device)
        self.harmony_embedding = nn.Embedding(chord_vocab_size, d_model, device=device)
        self.guidance_to_dmodel = nn.Linear(guidance_dim, d_model, device=device)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, d_model))

        self.vae = GuidanceVAE(**vae_cfg, chord_vocab_size=chord_vocab_size, device=device)
        self.encoder = GridMLMMelHarmEncoder(chord_vocab_size=chord_vocab_size, d_model=d_model,
                                             **encoder_cfg, device=device)

        self.recon_loss_fn = nn.CrossEntropyLoss()
        self.contrastive_margin = 1.0
        self.guidance_dim = guidance_dim
    # end init

    def compute_losses(self, harmony_tokens, recon_seq, mu, logvar,
                       z_proj, handcrafted_features):
        # Input: (B, L) target tokens
        # Output: (B, L, V) predictions
        recon_seq = recon_seq.permute(0, 2, 1)  # (B, V, L) for CrossEntropyLoss
        recon_loss = self.recon_loss_fn(recon_seq, harmony_tokens)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        z_norm = F.normalize(z_proj, dim=-1)
        feats_norm = F.normalize(handcrafted_features, dim=-1)
        similarity = torch.mm(z_norm, feats_norm.T)
        labels = torch.arange(similarity.size(0), device=similarity.device)
        contrastive_loss = F.cross_entropy(similarity, labels)
        return recon_loss, kl_loss, contrastive_loss
    # end compute_losses

    def forward(self, conditioning_vec, melody_grid, harmony_tokens, guiding_harmony,
                stage_indices, handcrafted_features=None):
        B = conditioning_vec.size(0)

        cond_emb = self.condition_proj(conditioning_vec).unsqueeze(1)
        melody_emb = self.melody_proj(melody_grid)

        if harmony_tokens is not None:
            harmony_emb = self.harmony_embedding(harmony_tokens)
        else:
            harmony_emb = torch.zeros(B, self.grid_length, melody_emb.size(-1), device=self.device)

        input_seq = torch.cat([cond_emb, melody_emb, harmony_emb], dim=1)
        input_seq += self.pos_embedding[:, :input_seq.size(1), :]

        z, mu, logvar, recon_seq, z_proj = self.vae(guiding_harmony) # TODO: do we need to detach?

        z_dmodel = self.guidance_to_dmodel(z).unsqueeze(1)  # (B, 1, D)
        if self.unfold_latent:
            z_seq = z_dmodel.repeat(1, self.seq_len, 1)
        else:
            z_seq = torch.zeros_like(input_seq)
            z_seq[:, 0:1, :] = z_dmodel


        guided_seq = input_seq + z_seq

        harmony_output = self.encoder(guided_seq, stage_indices)
        if handcrafted_features is not None:
            recon_loss, kl_loss, contrastive_loss = self.compute_losses(
                harmony_tokens, recon_seq, mu, logvar, z_proj, handcrafted_features)

            return harmony_output, {
                'recon_loss': recon_loss,
                'kl_loss': kl_loss,
                'contrastive_loss': contrastive_loss
            }
        else:
            return harmony_output
    # end forward

    def get_z_from_harmony(self, full_harmony):
        z, _, _, _, _ = self.vae(full_harmony)
        return z
    # end get_z_from_harmony
# end class GuidedMLMH