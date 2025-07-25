import os
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from models import GuidedMLMH, CrossGuidedMLMH
from train_utils import train_with_curriculum, train_cross_with_curriculum
from generate_utils import load_model, load_cross_model

def train_gmlmh(
        trainloader=None,
        valloader=None,
        tokenizer=None,
        subfolder='',
        epochs= 10,
        lr=5e-5,
        curriculum_type='random',
        device_name='cpu',
        tqdm_position=0,
        validations_per_epoch=1,
        loss_scheme='all'
    ):

    loss_fn=CrossEntropyLoss(ignore_index=-100)

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    vae_cfg = {
        'input_dim': 512,
        'hidden_dim': 256,
        'latent_dim': 128,
        'embedding_dim': 64,
        'seq_len': 256,
        'feature_dim': 348,
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
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # save results
    os.makedirs('results/', exist_ok=True)
    os.makedirs('results/' + subfolder + '/', exist_ok=True)
    results_path = 'results/' + subfolder + '/' + curriculum_type + '_' + loss_scheme + '.csv'
    
    os.makedirs('saved_models/', exist_ok=True)
    os.makedirs('saved_models/' + subfolder + '/', exist_ok=True)
    save_dir = 'saved_models/' + subfolder + '/'
    transformer_path = save_dir + curriculum_type + '_' + loss_scheme + '.pt'

    train_with_curriculum(
        model, optimizer, trainloader, valloader, loss_fn, tokenizer.mask_token_id,
        epochs=epochs,
        curriculum_type=curriculum_type,  # 'random', 'base2'
        results_path=results_path,
        transformer_path=transformer_path,
        tqdm_position=tqdm_position,
        validations_per_epoch=validations_per_epoch,
        loss_scheme=loss_scheme
    )
# end train_gmlmh

def train_disentangle_gmlmh(
        trainloader=None,
        valloader=None,
        tokenizer=None,
        subfolder='',
        epochs= 10,
        lr=5e-5,
        curriculum_type='random',
        device_name='cpu',
        tqdm_position=0,
        validations_per_epoch=1,
        loss_scheme='all'
    ):

    loss_fn=CrossEntropyLoss(ignore_index=-100)

    model = load_model(curriculum_type = curriculum_type, subfolder=subfolder, ablation=loss_scheme, \
        device_name = device_name, tokenizer=tokenizer)
    
    optimizer = AdamW(model.parameters(), lr=lr)

    # save results
    os.makedirs('results/disentangle/', exist_ok=True)
    os.makedirs('results/disentangle/' + subfolder + '/', exist_ok=True)
    results_path = 'results/disentangle/' + subfolder + '/' + curriculum_type + '_' + loss_scheme + '.csv'
    
    os.makedirs('saved_models/disentangle/', exist_ok=True)
    os.makedirs('saved_models/disentangle/' + subfolder + '/', exist_ok=True)
    save_dir = 'saved_models/disentangle/' + subfolder + '/'
    transformer_path = save_dir + curriculum_type + '_' + loss_scheme + '.pt'

    train_with_curriculum(
        model, optimizer, trainloader, valloader, loss_fn, tokenizer.mask_token_id,
        epochs=epochs,
        curriculum_type=curriculum_type,  # 'random', 'base2'
        results_path=results_path,
        transformer_path=transformer_path,
        tqdm_position=tqdm_position,
        validations_per_epoch=validations_per_epoch,
        loss_scheme=loss_scheme
    )
# end train_disentangle_gmlmh

# -----------------------------------------------------------------
# CROSS ATTENTION
# -----------------------------------------------------------------

def train_cross_gmlmh(
        trainloader=None,
        valloader=None,
        tokenizer=None,
        subfolder='',
        epochs=10,
        lr=5e-5,
        curriculum_type='random',
        device_name='cpu',
        tqdm_position=0,
        validations_per_epoch=1
    ):

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Device setup
    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(device_name if torch.cuda.is_available() else 'cpu')

    # Configuration for encoder and guidance BiLSTM
    encoder_cfg = {
        'nhead': 8,
        'num_layers': 8,
        'stage_embedding_dim': 64,
        'max_stages': 10,
    }

    guidance_lstm_cfg = {
        'vocab_size': len(tokenizer.vocab),
        'embedding_dim': 64,
        'hidden_dim': 128,
        'num_layers': 1,
        'bidirectional': True
    }

    model = CrossGuidedMLMH(
        encoder_cfg=encoder_cfg,
        chord_vocab_size=len(tokenizer.vocab),
        d_model=512,
        conditioning_dim=16,
        pianoroll_dim=100,
        grid_length=256,
        guidance_lstm_cfg=guidance_lstm_cfg,
        device=device,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Output paths
    os.makedirs('results/cross/', exist_ok=True)
    os.makedirs(f'results/cross/{subfolder}/', exist_ok=True)
    results_path = f'results/cross/{subfolder}/{curriculum_type}.csv'

    os.makedirs('saved_models/cross/', exist_ok=True)
    os.makedirs(f'saved_models/cross/{subfolder}/', exist_ok=True)
    transformer_path = f'saved_models/cross/{subfolder}/{curriculum_type}.pt'

    train_cross_with_curriculum(
        model, optimizer, trainloader, valloader, loss_fn, tokenizer.mask_token_id,
        epochs=epochs,
        curriculum_type=curriculum_type,
        results_path=results_path,
        transformer_path=transformer_path,
        tqdm_position=tqdm_position,
        validations_per_epoch=validations_per_epoch
    )
# end train_cross_gmlmh

def train_cross_disentangle_gmlmh(
        trainloader=None,
        valloader=None,
        tokenizer=None,
        subfolder='',
        epochs= 10,
        lr=5e-5,
        curriculum_type='random',
        device_name='cpu',
        tqdm_position=0,
        validations_per_epoch=1,
    ):

    loss_fn=CrossEntropyLoss(ignore_index=-100)

    model = load_cross_model(curriculum_type = curriculum_type, subfolder=subfolder, \
        device_name = device_name, tokenizer=tokenizer)
    
    optimizer = AdamW(model.parameters(), lr=lr)

    # save results
    os.makedirs('results/cross/disentangle/', exist_ok=True)
    os.makedirs('results/cross/disentangle/' + subfolder + '/', exist_ok=True)
    results_path = 'results/cross/disentangle/' + subfolder + '/' + curriculum_type + '.csv'
    
    os.makedirs('saved_models/cross/disentangle/', exist_ok=True)
    os.makedirs('saved_models/cross/disentangle/' + subfolder + '/', exist_ok=True)
    save_dir = 'saved_models/cross/disentangle/' + subfolder + '/'
    transformer_path = save_dir + curriculum_type + '.pt'

    train_cross_with_curriculum(
        model, optimizer, trainloader, valloader, loss_fn, tokenizer.mask_token_id,
        epochs=epochs,
        curriculum_type=curriculum_type,  # 'random', 'base2'
        results_path=results_path,
        transformer_path=transformer_path,
        tqdm_position=tqdm_position,
        validations_per_epoch=validations_per_epoch
    )
# end train_cross_disentangle_gmlmh