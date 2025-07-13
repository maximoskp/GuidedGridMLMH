import torch.nn.functional as F
import torch
from GridMLM_tokenizers import GuidedGridMLMTokenizer
from models import GuidedMLMH
from generate_utils import load_model
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import numpy as np
from sklearn.metrics import jaccard_score
import os
import csv

results_folder = 'results/experiments/'

pickles_folder = '/media/maindisk/data/hooktheory_hr/guidance_experiment_data/'
files = os.listdir(pickles_folder)
print(files)

curricula = ['random', 'base2']
subfolders = ['unf_CA', 'disentangle/unf_CA']
ablations = ['all', 'kl', 'rec', 'con']

device_name = 'cuda:2'

for curriculum_type in curricula:
    print(curriculum_type)
    for subfolder in subfolders:
        print(subfolder)
        for ablation in ablations:
            print(ablation)
            tokenizer = GuidedGridMLMTokenizer(fixed_length=256)
            model = load_model(
                    curriculum_type=curriculum_type,
                    subfolder=subfolder,
                    ablation=ablation,
                    device_name=device_name,
                    tokenizer=tokenizer
            )
            device = model.device

            for input_dataset in ['test', 'jazz']:
                for guide_dataset in ['test', 'jazz']:
                    print(input_dataset, guide_dataset)
                    base_name = '_'.join( [subfolder.replace('/','_'), curriculum_type, ablation, input_dataset, guide_dataset] )

                    csv_file = results_folder + base_name + '.csv'
                    result_fields = ['z_inp_gen', 'z_gui_gen', 'z_inp_gui', \
                                    'f_inp_gen', 'f_gui_gen', 'f_inp_gui']
                    with open(csv_file, 'w') as f:
                        writer = csv.writer(f)
                        writer.writerow(result_fields)

                    pickle_file = base_name + '.pickle'
                    with open(pickles_folder+pickle_file, 'rb') as f:
                        print(pickles_folder+pickle_file)
                        ds = pickle.load(f)

                    for d in ds:
                        original_ids = d['input_encoded']['input_ids']
                        original_fs = d['input_encoded']['features']
                        results = d['results']
                        for r in results:
                            guiding_ids = r['guide_encoded']['input_ids']
                            guiding_fs = r['guide_encoded']['features']
                            gen_ids = r['generated']['input_ids']
                            gen_fs = r['generated']['features']

                            z_original = model.get_z_from_harmony(torch.LongTensor([original_ids]).to(device)).detach().cpu()[0].tolist()
                            z_guiding = model.get_z_from_harmony(torch.LongTensor([guiding_ids]).to(device)).detach().cpu()[0].tolist()
                            z_gen = model.get_z_from_harmony(torch.LongTensor([gen_ids]).to(device)).detach().cpu()[0].tolist()

                            z_inp_gen = F.cosine_similarity(torch.FloatTensor(z_original), torch.FloatTensor(z_gen), dim=-1).item()
                            z_gui_gen = F.cosine_similarity(torch.FloatTensor(z_guiding), torch.FloatTensor(z_gen), dim=-1).item()
                            z_inp_gui = F.cosine_similarity(torch.FloatTensor(z_guiding), torch.FloatTensor(z_original), dim=-1).item()

                            f_inp_gen = jaccard_score( np.array(original_fs) > 0 , np.array( gen_fs ) > 0 )
                            f_gui_gen = jaccard_score( np.array(guiding_fs) > 0 , np.array( gen_fs ) > 0 )
                            f_inp_gui = jaccard_score( np.array(original_fs) > 0 , np.array( guiding_fs ) > 0 )

                            with open( csv_file, 'a' ) as f:
                                writer = csv.writer(f)
                                writer.writerow( [z_inp_gen, z_gui_gen, z_inp_gui, \
                                                f_inp_gen, f_gui_gen, f_inp_gui] )

                    res = pd.read_csv( csv_file )
                    stat_z, p_z = mannwhitneyu(res['z_inp_gen'], res['z_gui_gen'])
                    print(f"U-statistic={stat_z}, p-value={p_z}")
                    stat_f, p_f = mannwhitneyu(res['f_inp_gen'], res['f_gui_gen'])
                    print(f"U-statistic={stat_f}, p-value={p_f}")

                    ax = res.plot(kind='box')
                    plt.savefig(results_folder + base_name + '_' + str(p_z < 0.01) + '_' + str(p_f < 0.01) + '.png', dpi=300)