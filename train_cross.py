from train_gmlmh_fn import train_cross_gmlmh
from data_utils import GuidedGridMLMDataset, GuidedGridMLM_collate_fn
from torch.utils.data import DataLoader
from GridMLM_tokenizers import GuidedGridMLMTokenizer

subfolder = 'unf_CA'
epochs = 50
validations_per_epoch = 1
train_dir = '/media/maindisk/data/hooktheory_hr/hooktheory_CA_train'
val_dir = '/media/maindisk/data/hooktheory_hr/hooktheory_CA_test'

batchsize = 16

tokenizer = GuidedGridMLMTokenizer(fixed_length=256)

train_dataset = GuidedGridMLMDataset(train_dir, tokenizer, 512, frontloading=True)
val_dataset = GuidedGridMLMDataset(val_dir, tokenizer, 512, frontloading=True)

trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, collate_fn=GuidedGridMLM_collate_fn)
valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, collate_fn=GuidedGridMLM_collate_fn)

kwargs = {
    'subfolder': subfolder,
    'epochs': epochs,
    'lr': 5e-5,
    'curriculum_type': 'random',
    'device_name': 'cuda:2',
    'tqdm_position': 0,
    'validations_per_epoch': validations_per_epoch
}

train_cross_gmlmh(
    trainloader=trainloader,
    valloader=valloader,
    tokenizer=tokenizer,
    **kwargs
)