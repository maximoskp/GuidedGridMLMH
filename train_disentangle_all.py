from train_gmlmh_fn import train_disentangle_gmlmh
from data_utils import GuidedGridMLMDataset, GuidedGridMLM_collate_fn
from torch.utils.data import DataLoader
from GridMLM_tokenizers import GuidedGridMLMTokenizer
import multiprocessing

# train_dir = '/media/maindisk/data/hooktheory/hooktheory_train'
# val_dir = '/media/maindisk/data/hooktheory/hooktheory_test'
# train_dir = '/media/maindisk/data/gjt_melodies/gjt'
# val_dir = '/media/maindisk/data/gjt_melodies/gjt'

# TODO: implement argument forwarding of unfold=True/False in models.py
subfolder = 'unf_CA'
epochs = 10
validations_per_epoch = 1
# train_dir = '/media/maindisk/data/hooktheory_hr/hooktheory_CA_train'
# val_dir = '/media/maindisk/data/hooktheory_hr/hooktheory_CA_test'

# subfolder = 'unf_all12'
# epochs = 5
# validations_per_epoch = 10
# train_dir = '/media/maindisk/data/hooktheory_hr/hooktheory_all12_train'
# val_dir = '/media/maindisk/data/hooktheory_hr/hooktheory_all12_test'

batchsize = 8

tokenizer = None

# train_dataset = None
# val_dataset = None

# trainloader = None
# valloader = None

def init_worker(x, tok):
    global tokenizer
    x = 1
    tokenizer = tok
# end init_worker

def train_wrapper(kwargs):
    train_dir = '/media/maindisk/data/hooktheory_hr/guidance_disentanglement_data/' + \
        kwargs['subfolder'] + '_' + kwargs['curriculum_type'] + '_' + kwargs['loss_scheme']
    val_dir = train_dir

    train_dataset = GuidedGridMLMDataset(train_dir, tokenizer, 512, frontloading=True)
    val_dataset = GuidedGridMLMDataset(val_dir, tokenizer, 512, frontloading=True)
    trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, collate_fn=GuidedGridMLM_collate_fn)
    valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, collate_fn=GuidedGridMLM_collate_fn)
    return train_disentangle_gmlmh(
        trainloader=trainloader,
        valloader=valloader,
        tokenizer=tokenizer,
        **kwargs
    )
# end train_wrapper

if __name__ == "__main__":
    # Load your heavy objects ONCE
    tokenizer = GuidedGridMLMTokenizer(fixed_length=256)

    # train_dataset = GuidedGridMLMDataset(train_dir, tokenizer, 512, frontloading=True)
    # val_dataset = GuidedGridMLMDataset(val_dir, tokenizer, 512, frontloading=True)

    # trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, collate_fn=GuidedGridMLM_collate_fn)
    # valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, collate_fn=GuidedGridMLM_collate_fn)

    task_args = [
        {
            'subfolder': subfolder,
            'epochs': epochs,
            'lr': 5e-5,
            'curriculum_type': 'random',
            'device_name': 'cuda:2',
            'tqdm_position': 0,
            'validations_per_epoch': validations_per_epoch,
            'loss_scheme': 'all'
        },
        {
            'subfolder': subfolder,
            'epochs': epochs,
            'lr': 5e-5,
            'curriculum_type': 'random',
            'device_name': 'cuda:2',
            'tqdm_position': 1,
            'validations_per_epoch': validations_per_epoch,
            'loss_scheme': 'kl'
        },
        {
            'subfolder': subfolder,
            'epochs': epochs,
            'lr': 5e-5,
            'curriculum_type': 'random',
            'device_name': 'cuda:2',
            'tqdm_position': 2,
            'validations_per_epoch': validations_per_epoch,
            'loss_scheme': 'rec'
        },
        {
            'subfolder': subfolder,
            'epochs': epochs,
            'lr': 5e-5,
            'curriculum_type': 'random',
            'device_name': 'cuda:2',
            'tqdm_position': 3,
            'validations_per_epoch': validations_per_epoch,
            'loss_scheme': 'con'
        },
        {
            'subfolder': subfolder,
            'epochs': epochs,
            'lr': 5e-5,
            'curriculum_type': 'base2',
            'device_name': 'cuda:1',
            'tqdm_position': 4,
            'validations_per_epoch': validations_per_epoch,
            'loss_scheme': 'all'
        },
        {
            'subfolder': subfolder,
            'epochs': epochs,
            'lr': 5e-5,
            'curriculum_type': 'base2',
            'device_name': 'cuda:1',
            'tqdm_position': 5,
            'validations_per_epoch': validations_per_epoch,
            'loss_scheme': 'kl'
        },
        {
            'subfolder': subfolder,
            'epochs': epochs,
            'lr': 5e-5,
            'curriculum_type': 'base2',
            'device_name': 'cuda:1',
            'tqdm_position': 6,
            'validations_per_epoch': validations_per_epoch,
            'loss_scheme': 'rec'
        },
        {
            'subfolder': subfolder,
            'epochs': epochs,
            'lr': 5e-5,
            'curriculum_type': 'base2',
            'device_name': 'cuda:1',
            'tqdm_position': 7,
            'validations_per_epoch': validations_per_epoch,
            'loss_scheme': 'con'
        },
    ]

    # Use "fork" for memory-efficient sharing (if on Unix)
    with multiprocessing.get_context("fork").Pool(
        processes=len(task_args),
        initializer=init_worker,
        initargs=(0, tokenizer)
    ) as pool:
        results = pool.map(train_wrapper, task_args)

    print("All finished:", results)
# end main