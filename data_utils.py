import torch
from torch.utils.data import Dataset
from GridMLM_tokenizers import GuidedGridMLMTokenizer
import os
import numpy as np
from music21 import converter, note, chord, harmony, meter, stream
import torch.nn.functional as F
from tqdm import tqdm
import pickle
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

def extract_lead_sheet_info(xml_path, quantization='16th', fixed_length=None):
    # Load the score and flatten
    score = converter.parse(xml_path)
    melody_part = score.parts[0].flat

    # Define 16th note length
    ql_per_16th = 0.25 / 4

    # Get pitch range: MIDI 21 (A0) to 108 (C8) -> 88 notes
    pitch_range = list(range(21, 109))
    n_pitch = len(pitch_range)

    # Step 1: Find first chord symbol and bar to trim before it
    first_chord_offset = None
    for el in melody_part.recurse().getElementsByClass(harmony.ChordSymbol):
        first_chord_offset = el.offset
        break

    measure_start_offset = 0.0
    if first_chord_offset is not None:
        for meas in melody_part.getElementsByClass(stream.Measure):
            if meas.offset <= first_chord_offset < meas.offset + meas.duration.quarterLength:
                measure_start_offset = meas.offset
                break

    skip_steps = int(np.round(measure_start_offset / ql_per_16th))

    # Determine total length in 16th notes
    total_duration_q = melody_part.highestTime
    total_steps = int(np.ceil(total_duration_q / ql_per_16th))

    # Allocate raw matrices (we will trim/pad later)
    raw_pianoroll = np.zeros((total_steps, n_pitch), dtype=np.uint8)
    raw_chords = [None] * total_steps

    # Fill pianoroll
    for el in melody_part.notesAndRests:
        start = int(np.round(el.offset / ql_per_16th))
        dur_steps = int(np.round(el.quarterLength / ql_per_16th))

        if isinstance(el, note.Note):
            midi = el.pitch.midi
            if midi in pitch_range:
                idx = pitch_range.index(midi)
                raw_pianoroll[start:start+dur_steps, idx] = 1

        elif isinstance(el, chord.Chord):  # Just in case
            for pitch in el.pitches:
                midi = pitch.midi
                if midi in pitch_range:
                    idx = pitch_range.index(midi)
                    raw_pianoroll[start:start+dur_steps, idx] = 1

    # Fill chord grid
    for el in melody_part.recurse().getElementsByClass(harmony.ChordSymbol):
        start = int(np.round(el.offset / ql_per_16th))
        if 0 <= start < len(raw_chords):
            raw_chords[start] = el.figure

    # Propagate chord forward
    for i in range(1, len(raw_chords)):
        if raw_chords[i] is None:
            raw_chords[i] = raw_chords[i-1]

    # Fill missing with <pad> or <nc>
    for i in range(len(raw_chords)):
        if raw_chords[i] is None:
            raw_chords[i] = "<nc>"  # Or use "<pad>" if preferred

    # Trim to start at first chord bar
    raw_pianoroll = raw_pianoroll[skip_steps:]
    raw_chords = raw_chords[skip_steps:]

    # Add pitch class profile (top 12 dims)
    n_steps = len(raw_pianoroll)
    pitch_classes = np.zeros((n_steps, 12), dtype=np.uint8)
    for i in range(n_steps):
        pitch_indices = np.where(raw_pianoroll[i] > 0)[0]
        for idx in pitch_indices:
            midi = pitch_range[idx]
            pitch_classes[i, midi % 12] = 1
    full_pianoroll = np.hstack([pitch_classes, raw_pianoroll])  # Shape: (T, 12 + 88)

    # Apply fixed length (pad or trim)
    if fixed_length is not None:
        if n_steps > fixed_length:
            full_pianoroll = full_pianoroll[:fixed_length]
            raw_chords = raw_chords[:fixed_length]
        elif n_steps < fixed_length:
            pad_len = fixed_length - n_steps
            pad_pr = np.zeros((pad_len, full_pianoroll.shape[1]), dtype=np.uint8)
            pad_ch = ["<pad>"] * pad_len
            full_pianoroll = np.vstack([full_pianoroll, pad_pr])
            raw_chords += pad_ch

    return full_pianoroll, raw_chords
#  end extract_lead_sheet_info

def compute_normalized_token_entropy(logits, target_ids, pad_token_id=None):
    """
    Computes Expected Bits per Token (Token Entropy) for a batch.
    
    Args:
        logits (torch.Tensor): Model logits of shape (batch_size, seq_len, vocab_size).
        target_ids (torch.Tensor): Target token IDs of shape (batch_size, seq_len).
        pad_token_id (int, optional): Token ID for padding. If provided, masked out in computation.
        
    Returns:
        entropy_per_token (torch.Tensor): Average entropy per token for each sequence.
        entropy_per_batch (float): Average entropy per token across the batch.
    """
    # Infer vocabulary size from logits shape
    vocab_size = logits.shape[-1]
    # Compute max possible entropy for normalization
    max_entropy = torch.log2(torch.tensor(vocab_size, dtype=torch.float32)).item()

    # Compute probabilities with softmax
    probs = F.softmax(logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)
    
    # Compute log probabilities (base 2)
    log_probs = torch.log2(probs + 1e-9)  # Avoid log(0) errors

    # Compute entropy: H(x) = - sum(P(x) * log2(P(x)))
    entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: (batch_size, seq_len)

    # Mask out padding tokens if provided
    if pad_token_id is not None:
        mask = (target_ids != pad_token_id).float()  # 1 for valid tokens, 0 for padding
        entropy = entropy * mask  # Zero out entropy for padding
        entropy_per_token = entropy.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)  # Normalize per valid token
    else:
        entropy_per_token = entropy.mean(dim=-1)  # Average over sequence length

    # Compute overall batch entropy
    entropy_per_batch = entropy_per_token.mean().item()

    return entropy_per_token/max_entropy, entropy_per_batch/max_entropy
# end compute_token_entropy



class GuidedGridMLMDataset(Dataset):
    def __init__(self, root_dir, tokenizer, fixed_length=512, frontloading=True, refrontload=False):
        self.data_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.xml') or file.endswith('.mxl') or file.endswith('.musicxml'):
                    full_path = os.path.join(dirpath, file)
                    self.data_files.append(full_path)
        self.tokenizer = tokenizer
        self.fixed_length = fixed_length
        self.frontloading = frontloading
        if self.frontloading:
            # check if file exists and load it
            root_dir = root_dir[:-1] if root_dir[-1] == '/' else root_dir
            frontloaded_file = root_dir + '.pickle'
            if refrontload or not os.path.isfile(frontloaded_file):
                print('Frontloading data.')
                self.encoded = []
                for data_file in tqdm(self.data_files):
                    try:
                        self.encoded.append( self.tokenizer.encode( data_file ) )
                    except Exception as e: 
                        print('Problem in:', data_file)
                        print(e)
                if frontloaded_file is not None:
                    with open(frontloaded_file, 'wb') as f:
                        pickle.dump(self.encoded, f)
            else:
                print('Loading data file.')
                with open(frontloaded_file, 'rb') as f:
                    self.encoded = pickle.load(f)
    # end init

    def __len__(self):
        if self.frontloading:
            return len(self.encoded)
        else:
            return len(self.data_files)
    # end len

    def __getitem__(self, idx):
        if self.frontloading:
            encoded = self.encoded[idx]
        else:
            data_file = self.data_files[idx]
            encoded = self.tokenizer.encode( data_file )
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'pianoroll': encoded['pianoroll'],
            'time_signature': encoded['time_signature'],
            'features': encoded['features']
        }
    # end getitem
# end class dataset

def GuidedGridMLM_collate_fn(batch):
    """
    batch: list of dataset items, each one like:
        {
            'input_ids': List[int],
            'attention_mask': List[int],
            'time_sig': List[int],
            'pianoroll': np.ndarray of shape (140, fixed_length)
            'features': List[float]
        }
    """
    input_ids = [torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch]
    time_signature = [torch.tensor(item['time_signature'], dtype=torch.float) for item in batch]
    pianorolls = [torch.tensor(item['pianoroll'], dtype=torch.float) for item in batch]
    features = [torch.tensor(item['features'], dtype=torch.float) for item in batch]

    return {
        'input_ids': torch.stack(input_ids),  # shape: (B, L)
        'attention_mask': torch.stack(attention_mask),  # shape: (B, L)
        'time_signature': torch.stack(time_signature),  # shape: (B, whatever dim)
        'pianoroll': torch.stack(pianorolls),  # shape: (B, 140, T)
        'features': torch.stack(features) # shape: (B, F)
    }
# end GuidedGridMLM_collate_fn:

class CosineMDS:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.X_fit = None
        self.embedding_ = None
        self.eigvecs_ = None
        self.eigvals_ = None
        self.mean_distance_row_ = None
        self.mean_distance_total_ = None
    # end init

    def fit_transform(self, X):
        """
        Compute 2D classical MDS embedding using cosine distance.
        """
        self.X_fit = X
        N = X.shape[0]

        # Step 1: Compute cosine distance matrix
        D = pairwise_distances(X, metric='cosine')  # shape: NxN
        D2 = D ** 2  # squared distances

        # Step 2: Double-center the distance matrix
        J = np.eye(N) - np.ones((N, N)) / N
        B = -0.5 * J @ D2 @ J

        # Step 3: Eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(B)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Take top components
        L = np.diag(np.sqrt(eigvals[:self.n_components]))
        V = eigvecs[:, :self.n_components]
        Y = V @ L  # Final embedding

        # Store for transform
        self.embedding_ = Y
        self.eigvecs_ = V
        self.eigvals_ = eigvals[:self.n_components]

        # Precompute mean distances for new point projection
        self.mean_distance_row_ = np.mean(D2, axis=1)
        self.mean_distance_total_ = np.mean(D2)

        return Y
    # end fit_transform

    def transform(self, x_new):
        """
        Project a new point (1xM) into the existing MDS space using Nystrom method.
        """
        if self.X_fit is None:
            raise ValueError("The model must be fitted with `fit_transform` before calling `transform`.")

        x_new = x_new.reshape(1, -1)
        N = self.X_fit.shape[0]

        # Step 1: Compute squared cosine distances from x_new to each point in the original data
        dists = pairwise_distances(self.X_fit, x_new, metric='cosine').flatten()
        d2 = dists ** 2  # shape: (N,)

        # Step 2: Apply Nystrom formula
        d_bar = np.mean(d2)
        b = -0.5 * (d2 - self.mean_distance_row_ - d_bar + self.mean_distance_total_)

        # Step 3: Project using the stored eigendecomposition
        y_new = b @ self.eigvecs_ / np.sqrt(self.eigvals_)
        return y_new.reshape(1, -1)
    # end transform
# end class CosineMDS