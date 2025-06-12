import torch
from data_utils import GuidedGridMLMDataset, GuidedGridMLM_collate_fn
from torch.utils.data import DataLoader
from GridMLM_tokenizers import GuidedGridMLMTokenizer
from models import GuidedMLMH
from tqdm import tqdm
import numpy as np
# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import dash
from dash import dcc, html, Input, Output

device_name = 'cuda:0'
val_dir = '/media/maindisk/data/hooktheory_hr/hooktheory_CA_test'
jazz_dir = '/media/maindisk/data/gjt_melodies/gjt_CA'
subfolder = 'CA'
curriculum_type='random'
ablation = 'all'

model_path = 'saved_models/' + subfolder + '/' + curriculum_type + '_' + ablation + '.pt'
tokenizer = GuidedGridMLMTokenizer(fixed_length=256)
val_dataset = GuidedGridMLMDataset(val_dir, tokenizer, 512, frontloading=True)
jazz_dataset = GuidedGridMLMDataset(jazz_dir, tokenizer, 512, frontloading=True)

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

def condenced_str_from_token_ids(inp_ids, tokenizer):
    tmp_str = ''
    tmp_count = 0
    prev_id = -1
    num_chords = 0
    for t in inp_ids:
        if prev_id == t:
            tmp_count += 1
        else:
            if prev_id != -1:
                tmp_str += f'{tmp_count}x{tokenizer.ids_to_tokens[prev_id]}'
                num_chords += 1
                if num_chords == 4:
                    tmp_str += '\n'
                    num_chords = 0
                else:
                    tmp_str += '_'
            tmp_count = 1
            prev_id = t
    tmp_str += f'{tmp_count}x{tokenizer.ids_to_tokens[prev_id]}'
    return tmp_str
# end condenced_str_from_token_ids

zs = []
z_idxs = []
z_tokens = []
data_all = []
for d in tqdm(val_dataset):
    full_harmony = torch.tensor(d['input_ids']).reshape(1, len(d['input_ids']))
    z_tokens.append(condenced_str_from_token_ids(d['input_ids'], tokenizer))
    data_all.append( d )
    z = model.get_z_from_harmony(full_harmony.to(device)).detach().cpu()[0].tolist()
    zs.append(z)
    z_idxs.append(0)
for d in tqdm(jazz_dataset):
    full_harmony = torch.tensor(d['input_ids']).reshape(1, len(d['input_ids']))
    z_tokens.append(condenced_str_from_token_ids(d['input_ids'], tokenizer))
    data_all.append( d )
    z = model.get_z_from_harmony(full_harmony.to(device)).detach().cpu()[0].tolist()
    zs.append(z)
    z_idxs.append(1)

z_np = np.array( zs )

# y = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3, verbose=2).fit_transform(z_np)
pca = PCA(n_components=2)
y = pca.fit_transform( z_np )

# Combine into a DataFrame for easy Plotly integration
df = pd.DataFrame({
    'x': y[:, 0],
    'y': y[:, 1],
    'class': z_idxs,
    'token': z_tokens
})

df['hover_text'] = df['token'].str.replace('\n', '<br>')

# Create interactive scatter plot
fig = px.scatter(
    df,
    x='x',
    y='y',
    color='class',
    hover_data=None
    # hover_name='token',  # This will show on hover
    # title='2D Visualization',
)

fig.update_layout(
    xaxis=dict(scaleanchor='y', scaleratio=1),  # Equal aspect ratio
)

fig.update_traces(
    hovertemplate=df['hover_text']
)

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='scatter-plot', figure=fig),
    html.Div(id='click-output')  # area to display info on click
])

@app.callback(
    Output('click-output', 'children'),
    Input('scatter-plot', 'clickData')
)
def display_click_data(clickData):
    if clickData is None:
        return "Click on a point to see details."
    point = clickData['points'][0]
    print(point)
    d = data_all[point['pointNumber']]
    return f"You clicked on: {d['input_ids']}"

if __name__ == '__main__':
    app.run(debug=True, port=3052)