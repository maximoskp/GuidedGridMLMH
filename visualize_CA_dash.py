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
from dash import dcc, html, Input, Output, State

# global
df = None
data_all = None

device_name = 'cuda:0'
val_dir = '/media/maindisk/data/hooktheory_hr/hooktheory_CA_test'
jazz_dir = '/media/maindisk/data/gjt_melodies/gjt_CA'
subfolder = 'CA'
curriculum_type='random'
ablation = 'all'

model_path = 'saved_models/' + subfolder + '/' + curriculum_type + '_' + ablation + '.pt'

custom_colors = ['#1f77b4', '#ff7f0e']  # blue and orange

if device_name == 'cpu':
        device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        print('Selected device not available: ' + device_name)

def initialize_data():
    print('FUN initialize_data')
    tokenizer = GuidedGridMLMTokenizer(fixed_length=256)
    val_dataset = GuidedGridMLMDataset(val_dir, tokenizer, 512, frontloading=True)
    jazz_dataset = GuidedGridMLMDataset(jazz_dir, tokenizer, 512, frontloading=True)
    return tokenizer, val_dataset, jazz_dataset
# end initiailze_data

def initialize_model(tokenizer):
    print('FUN initialize_model')
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
    return model
# end initialize_model

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

def apply_pca(model, tokenizer, val_dataset, jazz_dataset):
    print('FUN apply_pca')
    zs = []
    z_idxs = []
    z_tokens = []
    global data_all, df
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
    df['class_str'] = df['class'].astype(str)
# end apply_pca

def make_figure(selected):
    print('FUN make_figure')
    global df
    # Create interactive scatter plot
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='class_str',
        hover_data=None,
        color_discrete_sequence=custom_colors
    )

    fig.update_layout(
        xaxis=dict(scaleanchor='y', scaleratio=1),  # Equal aspect ratio
    )

    fig.update_traces(
        hovertemplate=df['hover_text']
    )
    print(selected)
    if selected:
        if selected['first'] is not None:
            row = df.iloc[selected['first']]
            fig.add_scatter(
                x=[row['x']],
                y=[row['y']],
                mode='markers',
                marker=dict(
                    color='red',
                    size=16,
                    symbol='star',
                    line=dict(color='black', width=2),
                    opacity=1.0
                ),
                name='First',
                showlegend=True
            )

        if selected['second'] is not None:
            row = df.iloc[selected['second']]
            fig.add_scatter(
                x=[row['x']],
                y=[row['y']],
                mode='markers',
                marker=dict(
                    color='green',
                    size=16,
                    symbol='diamond',
                    line=dict(color='black', width=2),
                    opacity=1.0
                ),
                name='Second',
                showlegend=True
            )
    return fig
# end make_figure

tokenizer, val_dataset, jazz_dataset = initialize_data()
model = initialize_model(tokenizer)
apply_pca(model, tokenizer, val_dataset, jazz_dataset)

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='scatter-plot', figure=make_figure(None)),
    dcc.Store(id='selected-points', data={'first': None, 'second': None}),
    html.Div(id='click-output')  # area to display info on click
])

@app.callback(
    Output('scatter-plot', 'figure'),
    Output('click-output', 'children'),
    Output('selected-points', 'data'),
    Input('scatter-plot', 'clickData'),
    State('selected-points', 'data'),
    prevent_initial_call=True
)

def handle_click(clickData, selected):
    if clickData is None:
        return dash.no_update
    point = clickData['points'][0]
    point_index = point['pointIndex']
    curve_number = point['curveNumber']

    # Find the global index in the DataFrame
    # Get trace class based on curveNumber
    clicked_class = df['class'].unique()[curve_number]

    # Get the row indices in df for that class
    matching_indices = df[df['class'] == clicked_class].index.to_list()
    
    # Use pointIndex within that group to get the global index
    idx = matching_indices[point_index]

    if selected:
        if selected['first'] is None or selected['first'] == idx:
            selected['first'] = idx
        elif selected['second'] is None or selected['second'] == idx:
            selected['second'] = idx
        else:
            # Rotate selection
            selected['first'], selected['second'] = selected['second'], idx

    token1 = df.iloc[selected['first']]['token'].split('\n') if selected['first'] is not None else ["None"]
    token2 = df.iloc[selected['second']]['token'].split('\n') if selected['second'] is not None else ["None"]

    text = html.Div([
        html.Strong("First:"),
        *[html.Div(line) for line in token1],
        html.Br(),
        html.Strong("Second:"),
        *[html.Div(line) for line in token2],
    ])

    return make_figure(selected), text, selected

if __name__ == '__main__':
    print('FUN main')
    app.run(debug=True, port=3052)