from train_gmlmh_fn import train_gmlmh
from data_utils import GuidedGridMLMDataset, GuidedGridMLM_collate_fn
from torch.utils.data import DataLoader
from GridMLM_tokenizers import GuidedGridMLMTokenizer
import multiprocessing

# train_dir = '/media/maindisk/maximos/data/hooktheory_train'
# val_dir = '/media/maindisk/maximos/data/hooktheory_test'
train_dir = '/media/maindisk/maximos/data/gjt'
val_dir = '/media/maindisk/maximos/data/gjt'

batchsize = 32

tokenizer = None

train_dataset = None
val_dataset = None

trainloader = None
valloader = None

def init_worker(td, vd, tl, vl, tok):
    global train_dataset, val_dataset, trainloader, valloader, tokenizer
    train_dataset = td
    val_dataset = vd
    trainloader = tl
    valloader = vl
    tokenizer = tok
# end init_worker

def train_wrapper(kwargs):
    return train_gmlmh(
        trainloader=trainloader,
        valloader=valloader,
        tokenizer=tokenizer,
        **kwargs
    )
# end train_wrapper

if __name__ == "__main__":
    # Load your heavy objects ONCE
    tokenizer = GuidedGridMLMTokenizer(fixed_length=256)

    train_dataset = GuidedGridMLMDataset(train_dir, tokenizer, 512, frontloaded_file='data/' + train_dir.split('/')[-1] + '.pickle')
    val_dataset = GuidedGridMLMDataset(val_dir, tokenizer, 512, frontloaded_file='data/' + val_dir.split('/')[-1] + '.pickle')

    trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, collate_fn=GuidedGridMLM_collate_fn)
    valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, collate_fn=GuidedGridMLM_collate_fn)

    task_args = [
        {
            'subfolder': 'CA',
            'epochs': 50,
            'lr': 5e-5,
            'curriculum_type': 'base2',
            'device_name': 'cuda:0'
        },
        {
            'subfolder': 'CA',
            'epochs': 50,
            'lr': 5e-5,
            'curriculum_type': 'random',
            'device_name': 'cuda:1'
        },
    ]

    # Use "fork" for memory-efficient sharing (if on Unix)
    with multiprocessing.get_context("fork").Pool(
        processes=len(task_args),
        initializer=init_worker,
        initargs=(train_dataset, val_dataset, trainloader, valloader, tokenizer)
    ) as pool:
        results = pool.map(train_wrapper, task_args)

    print("All finished:", results)
# end main