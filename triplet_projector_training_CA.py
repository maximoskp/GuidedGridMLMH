import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from data_utils import GuidedGridMLMDataset, GuidedGridMLM_collate_fn
from GridMLM_tokenizers import GuidedGridMLMTokenizer
from models import GuidedMLMH
from tqdm import tqdm

class TripletProjector(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)
# end TripletProjector

class TripletDataset(Dataset):
    def __init__(self, z_vectors, top_k=5, bottom_k=5):
        self.z = F.normalize(torch.FloatTensor(z_vectors), dim=1)  # L2 normalization
        self.sim_matrix = torch.matmul(self.z, self.z.T)  # cosine sim
        self.top_k = top_k
        self.bottom_k = bottom_k
        self.indices = list(range(len(self.z)))

    def __len__(self):
        return len(self.z)

    def __getitem__(self, anchor_idx):
        # Exclude self
        sim_row = self.sim_matrix[anchor_idx].clone()
        sim_row[anchor_idx] = -1.0

        # Positive: highest similarity
        pos_candidates = torch.topk(sim_row, self.top_k).indices
        pos_idx = pos_candidates[torch.randint(0, self.top_k, (1,)).item()]

        # Negative: lowest similarity
        neg_candidates = torch.topk(sim_row, self.bottom_k, largest=False).indices
        neg_idx = neg_candidates[torch.randint(0, self.bottom_k, (1,)).item()]

        return self.z[anchor_idx], self.z[pos_idx], self.z[neg_idx]
# end TripletDataset

def train_triplet_model(z_array, latent_dim, save_dir="saved_models/triplet_models", epochs=200, lr=1e-4, margin=0.2):
    os.makedirs(save_dir, exist_ok=True)

    model = TripletProjector(latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TripletDataset(z_array, top_k=5, bottom_k=5)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    best_loss = 1000

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for anchor, pos, neg in dataloader:
            optimizer.zero_grad()

            a = F.normalize(model(anchor), dim=1)
            p = F.normalize(model(pos), dim=1)
            n = F.normalize(model(neg), dim=1)

            # Triplet loss using cosine similarity
            sim_ap = F.cosine_similarity(a, p, dim=1)
            sim_an = F.cosine_similarity(a, n, dim=1)
            loss = F.relu(sim_an - sim_ap + margin).mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss: {total_loss / len(dataloader):.4f}")
        if best_loss > total_loss:
            best_loss = total_loss
            # Save model
            print('saving...')
            torch.save(model.state_dict(), f"{save_dir}/triplet_epoch_CA.pt")

    return model
# end train_triplet_model

device_name = 'cuda:2'
val_dir = '/media/maindisk/data/hooktheory_hr/hooktheory_CA_test'
jazz_dir = '/media/maindisk/data/gjt_melodies/gjt_CA'
subfolder = 'CA'
curriculum_type='random'
ablation = 'all'


model_path = 'saved_models/' + subfolder + '/' + curriculum_type + '_' + ablation + '.pt'

if device_name == 'cpu':
        device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        print('Selected device not available: ' + device_name)

def initialize_data():
    print('FUN initialize_data')
    tokenizer = GuidedGridMLMTokenizer(fixed_length=256)
    val_dataset = GuidedGridMLMDataset(val_dir, tokenizer, 512, frontloading=True)
    jazz_dataset = GuidedGridMLMDataset(jazz_dir, tokenizer, 512, frontloading=True)
    return tokenizer, val_dataset, jazz_dataset
# end initiailze_data

def initialize_model(tokenizer):
    print('FUN initialize_model')
    vae_cfg = {
        'input_dim': 512,
        'hidden_dim': 256,
        'latent_dim': 128,
        'embedding_dim': 64,
        'seq_len': 256,
        'feature_dim': 37,
    }
    encoder_cfg = {
        'nhead': 8,
        'num_layers': 8,
        'stage_embedding_dim': 64,
        'max_stages': 10
    }
    model = GuidedMLMH(
        vae_cfg=vae_cfg,
        encoder_cfg=encoder_cfg,
        chord_vocab_size=len(tokenizer.vocab),
        d_model=512,
        conditioning_dim=16,
        pianoroll_dim=100,
        grid_length=256,
        guidance_dim=128,
        unfold_latent=True,
        device=device,
    )
    checkpoint = torch.load(model_path, map_location=device_name)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model
# end initialize_model

def make_z_array(model, tokenizer, val_dataset, jazz_dataset):
    zs = []
    for d in tqdm(val_dataset):
        full_harmony = torch.tensor(d['input_ids']).reshape(1, len(d['input_ids']))
        z = model.get_z_from_harmony(full_harmony.to(device)).detach().cpu()[0].tolist()
        zs.append(z)
    # for d in tqdm(jazz_dataset):
    #     full_harmony = torch.tensor(d['input_ids']).reshape(1, len(d['input_ids']))
    #     z = model.get_z_from_harmony(full_harmony.to(device)).detach().cpu()[0].tolist()
    #     zs.append(z)

    z_np = np.array( zs )
    return z_np
# end apply_pca

if __name__ == '__main__':
    # load data and model
    tokenizer, val_dataset, jazz_dataset = initialize_data()
    model = initialize_model(tokenizer)
    z_array = make_z_array(model, tokenizer, val_dataset, jazz_dataset)
    latent_dim = z_array.shape[1]
    trained_model = train_triplet_model(z_array, latent_dim=latent_dim)