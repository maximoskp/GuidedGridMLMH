import torch
import torch.nn.functional as F
from data_utils import GuidedGridMLMDataset, GuidedGridMLM_collate_fn
from torch.utils.data import DataLoader
from GridMLM_tokenizers import GuidedGridMLMTokenizer
from models import GuidedMLMH
from tqdm import tqdm
import numpy as np
from generate_utils import load_model, structured_progressive_generate, random_progressive_generate
from copy import deepcopy
import pickle
import multiprocessing

subfolder = 'unf_CA'
train_dir = '/media/maindisk/data/hooktheory_hr/hooktheory_CA_train'

tokenizer = None

train_dataset = None

save_folder = None

def init_worker(td, tok, sf):
    global train_dataset, tokenizer, save_folder
    train_dataset = td
    tokenizer = tok
    save_folder = sf
# end init_worker

def generate_disentanglement_data(
        tokenizer,
        train_dataset,
        save_folder,
        device_name = 'cuda:2',
        subfolder = 'unf_CA',
        curriculum_type='random',
        ablation = 'all',
        num_melodies = 1000,
        num_guides = 200,
    ):
    # load model
    model = load_model( curriculum_type, subfolder, ablation, device_name, tokenizer )
    # generation function
    if curriculum_type == 'random':
        gen_fun = random_progressive_generate
    else:
        gen_fun = structured_progressive_generate
    # train dataset has to come from outside, to load it once for multiple simultaneous runs
    # # load training data
    # train_dataset = GuidedGridMLMDataset(train_dir, tokenizer, 512, frontloading=True)
    # permutation of melody indices
    input_idxs = np.random.permutation( len(train_dataset) )
    # initialize new dataset
    new_dataset = []
    for i_input in tqdm(input_idxs[:num_melodies]):
        # permutation of guide indices
        # exclude the input idx
        all_indexes = np.delete(np.arange(len(train_dataset)), i_input)
        # permutation of the remaining indices
        guide_idxs = np.random.permutation( all_indexes )
        for i_guide in guide_idxs[:num_guides]:
            input_encoded = train_dataset[i_input]
            guide_encoded = train_dataset[i_guide]
            harmony_guide = torch.LongTensor(guide_encoded['input_ids']).reshape(1, len(guide_encoded['input_ids']))
            # harmony_real = torch.LongTensor(input_encoded['input_ids']).reshape(1, len(input_encoded['input_ids']))
            melody_grid = torch.FloatTensor( input_encoded['pianoroll'] ).reshape( 1, input_encoded['pianoroll'].shape[0], input_encoded['pianoroll'].shape[1] )
            conditioning_vec = torch.FloatTensor( input_encoded['time_signature'] ).reshape( 1, len(input_encoded['time_signature']) )
            generated_harmony = gen_fun(
                model=model,
                melody_grid=melody_grid.to(model.device),
                conditioning_vec=conditioning_vec.to(model.device),
                guiding_harmony=harmony_guide.to(model.device),
                num_stages=10,
                mask_token_id=tokenizer.mask_token_id,
                temperature=1.0,
                strategy='sample',
                pad_token_id=tokenizer.pad_token_id,      # token ID for <pad>
                nc_token_id=tokenizer.nc_token_id,       # token ID for <nc>
                force_fill=True,         # disallow <pad>/<nc> before melody ends
                chord_constraints = None
            )
            new_attention = torch.logical_not( generated_harmony[0] == tokenizer.pad_token_id ).long().tolist()
            # we don't need to put -100 to pad, we need to learn it...
            # generated_harmony[ generated_harmony == tokenizer.pad_token_id ] = -100
            new_input_ids = generated_harmony[0].tolist()
            new_d = deepcopy( input_encoded )
            new_d['input_ids'] = new_input_ids
            new_d['attention_mask'] = new_attention
            new_dataset.append( new_d )
    save_folder += subfolder + '_' + curriculum_type + '_' + ablation + '.pickle'
    with open(save_folder, 'wb') as f:
        pickle.dump(new_dataset, f)
    return new_dataset
# end generate_disentanglement_data

def train_wrapper(kwargs):
    return generate_disentanglement_data(
        tokenizer,
        train_dataset,
        save_folder,
        **kwargs
    )
# end train_wrapper

if __name__ == "__main__":
    # Load heavy objects ONCE
    tokenizer = GuidedGridMLMTokenizer(fixed_length=256)

    train_dataset = GuidedGridMLMDataset(train_dir, tokenizer, 512, frontloading=True)

    save_folder = '/media/maindisk/data/hooktheory_hr/guidance_disentanglement_data/'

    task_args = [
        {
            'device_name': 'cuda:2',
            'subfolder': subfolder,
            'curriculum_type': 'random',
            'ablation': 'all',
            'num_melodies': 1000,
            'num_guides': 20
        },
        {
            'device_name': 'cuda:2',
            'subfolder': subfolder,
            'curriculum_type': 'random',
            'ablation': 'con',
            'num_melodies': 1000,
            'num_guides': 20
        },
        {
            'device_name': 'cuda:2',
            'subfolder': subfolder,
            'curriculum_type': 'random',
            'ablation': 'kl',
            'num_melodies': 1000,
            'num_guides': 20
        },
        {
            'device_name': 'cuda:2',
            'subfolder': subfolder,
            'curriculum_type': 'random',
            'ablation': 'rec',
            'num_melodies': 1000,
            'num_guides': 20
        },
        {
            'device_name': 'cuda:2',
            'subfolder': subfolder,
            'curriculum_type': 'base2',
            'ablation': 'all',
            'num_melodies': 1000,
            'num_guides': 20
        },
        {
            'device_name': 'cuda:2',
            'subfolder': subfolder,
            'curriculum_type': 'base2',
            'ablation': 'con',
            'num_melodies': 1000,
            'num_guides': 20
        },
        {
            'device_name': 'cuda:2',
            'subfolder': subfolder,
            'curriculum_type': 'base2',
            'ablation': 'kl',
            'num_melodies': 1000,
            'num_guides': 20
        },
        {
            'device_name': 'cuda:2',
            'subfolder': subfolder,
            'curriculum_type': 'base2',
            'ablation': 'rec',
            'num_melodies': 1000,
            'num_guides': 20
        },
    ]

    # Use "fork" for memory-efficient sharing (if on Unix)
    with multiprocessing.get_context("fork").Pool(
        processes=len(task_args),
        initializer=init_worker,
        initargs=(train_dataset, tokenizer, save_folder)
    ) as pool:
        results = pool.map(train_wrapper, task_args)

    print("All finished:", results)
# end main