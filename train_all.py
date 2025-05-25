from train_gmlmh_fn import train_gmlmh
from data_utils import GuidedGridMLMDataset, GuidedGridMLM_collate_fn
from torch.utils.data import DataLoader
from GridMLM_tokenizers import GuidedGridMLMTokenizer
import multiprocessing

train_dir = '/media/maindisk/maximos/data/hooktheory_train'
val_dir = '/media/maindisk/maximos/data/hooktheory_test'

batchsize = 32

tokenizer = GuidedGridMLMTokenizer(fixed_length=256)

train_dataset = GuidedGridMLMDataset(train_dir, tokenizer, 512, frontloaded_file='data/' + train_dir.split('/')[-1] + '.pickle')
val_dataset = GuidedGridMLMDataset(val_dir, tokenizer, 512, frontloaded_file='data/' + val_dir.split('/')[-1] + '.pickle')

trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, collate_fn=GuidedGridMLM_collate_fn)
valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, collate_fn=GuidedGridMLM_collate_fn)

def init_worker(tl, vl, tok):
    global trainloader, valloader, tokenizer
    trainloader = tl
    valloader = vl
    tokenizer = tok
# end init_worker

def train_wrapper(args):
    subfolder, curriculum_type, device_name = args
    return train_gmlmh(
        trainloader=trainloader,
        valloader=valloader,
        tokenizer=tokenizer,
        subfolder=subfolder,
        curriculum_type=curriculum_type,
        device_name=device_name
    )
# end train_wrapper

if __name__ == "__main__":
    # Load your heavy objects ONCE
    data = ...  # however you load your trainloader
    val = ...
    tok = ...

    task_args = [
        ("run1", "base2", "cuda:0"),
        ("run2", "linear", "cuda:1"),
        ("run3", "none", "cuda:2"),
        ("run4", "step", "cuda:3"),
    ]

    # Use "fork" for memory-efficient sharing (if on Unix)
    with multiprocessing.get_context("fork").Pool(
        processes=4,
        initializer=init_worker,
        initargs=(data, val, tok)
    ) as pool:
        results = pool.map(train_wrapper, task_args)

    print("All finished:", results)
# end main