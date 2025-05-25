import os
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from models import GuidedMLMH
from train_utils import train_with_curriculum

def train_gmlmh(
        trainloader=None,
        valloader=None,
        tokenizer=None,
        subfolder='',
        epochs= 10,
        lr=5e-5,
        curriculum_type='base2',
        device_name='cpu'
    ):

    loss_fn=CrossEntropyLoss(ignore_index=-100)

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)
    model = GuidedMLMH(
        chord_vocab_size=len(tokenizer.vocab),
        device=device,
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # save results
    os.makedirs('results/', exist_ok=True)
    os.makedirs('results/' + subfolder + '/', exist_ok=True)
    results_path = 'results/' + subfolder + '/' + curriculum_type + '.csv'
    
    os.makedirs('saved_models/', exist_ok=True)
    os.makedirs('saved_models/' + subfolder + '/', exist_ok=True)
    save_dir = 'saved_models/' + subfolder + '/'
    transformer_path = save_dir + curriculum_type + '.pt'

    train_with_curriculum(
        model, optimizer, trainloader, valloader, loss_fn, tokenizer.mask_token_id,
        epochs=epochs,
        curriculum_type=curriculum_type,  # 'random', 'base2'
        results_path=results_path,
        transformer_path=transformer_path,
    )
# end train_gmlmh