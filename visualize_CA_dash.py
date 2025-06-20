import torch
import torch.nn.functional as F
from data_utils import GuidedGridMLMDataset, GuidedGridMLM_collate_fn
from torch.utils.data import DataLoader
from GridMLM_tokenizers import GuidedGridMLMTokenizer
from models import GuidedMLMH
from tqdm import tqdm
import numpy as np
# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
import plotly.express as px
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
from generate_utils import load_model, structured_progressive_generate
import os
from triplet_projector_training_CA import TripletProjector

# global
df = None
data_all = None
projector_model = None

device_name = 'cuda:0'
val_dir = '/media/maindisk/data/hooktheory_hr/hooktheory_CA_test'
jazz_dir = '/media/maindisk/data/gjt_melodies/gjt_CA'
subfolder = 'CA'
curriculum_type='random'
ablation = 'all'

mxl_folder = 'examples_musicXML/' + subfolder + '/' + curriculum_type + '/' + ablation + '/'
midi_folder = 'examples_MIDI/' + subfolder + '/' + curriculum_type + '/' + ablation + '/'
os.makedirs(mxl_folder, exist_ok=True)
os.makedirs(midi_folder, exist_ok=True)

model_path = 'saved_models/' + subfolder + '/' + curriculum_type + '_' + ablation + '.pt'

custom_colors = ["#42b41f", "#0e2eff", '#d62728']  # blue, orange, red
symbol_map = {
    '0': 'circle',
    '1': 'circle',
    '2': 'diamond'  # for harmonized points
}
size_map = {
    '0': 10,
    '1': 10,
    '2': 15  # larger for harmonized points
}

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

def apply_projection(model, tokenizer, val_dataset, jazz_dataset):
    global projector_model
    print('FUN apply_projection')
    zs = []
    z_idxs = []
    z_tokens = []
    global data_all, df
    data_all = []
    for d in tqdm(val_dataset):
        full_harmony = torch.tensor(d['input_ids']).reshape(1, len(d['input_ids']))
        z_tokens.append(condenced_str_from_token_ids(d['input_ids'], tokenizer))
        z = model.get_z_from_harmony(full_harmony.to(device)).detach().cpu()[0].tolist()
        d['z'] = z
        data_all.append( d )
        zs.append(z)
        z_idxs.append(0)
    # for d in tqdm(jazz_dataset):
    #     full_harmony = torch.tensor(d['input_ids']).reshape(1, len(d['input_ids']))
    #     z_tokens.append(condenced_str_from_token_ids(d['input_ids'], tokenizer))
    #     data_all.append( d )
    #     z = model.get_z_from_harmony(full_harmony.to(device)).detach().cpu()[0].tolist()
    #     zs.append(z)
    #     z_idxs.append(1)

    z_np = np.array( zs )

    projector_model = TripletProjector(z_np.shape[1])

    checkpoint = torch.load('saved_models/triplet_models/triplet_epoch_CA.pt', map_location=device_name)
    projector_model.load_state_dict(checkpoint)
    projector_model.eval()
    projector_model.to(device)

    projection = projector_model( torch.FloatTensor( z_np ).to(device) ).detach().cpu().numpy()
    
    # Combine into a DataFrame for easy Plotly integration
    df = pd.DataFrame({
        'x': projection[:, 0],
        'y': projection[:, 1],
        'class': z_idxs,
        'token': z_tokens
    })

    df['hover_text'] = df['token'].str.replace('\n', '<br>')
    df['class_str'] = df['class'].astype(str)
    df['symbol'] = df['class_str'].map(symbol_map)
    df['size'] = df['class_str'].map(size_map)
# end apply_projection

def make_figure(selected):
    print('FUN make_figure')
    global df
    # Create interactive scatter plot
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='class_str',
        symbol='symbol',
        # size='size',
        # size_max=10,
        hover_data=None,
        color_discrete_sequence=custom_colors,
        symbol_sequence=list(symbol_map.values())
    )

    fig.update_layout(
        xaxis=dict(scaleanchor='y', scaleratio=1),  # Equal aspect ratio
    )

    fig.update_traces(
        hovertemplate=df['hover_text']
    )
    print(selected)
    if selected:
        if selected['melody'] is not None:
            row = df.iloc[selected['melody']]
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
                name='Melody',
                showlegend=True
            )

        if selected['guide'] is not None:
            row = df.iloc[selected['guide']]
            fig.add_scatter(
                x=[row['x']],
                y=[row['y']],
                mode='markers',
                marker=dict(
                    color='green',
                    size=16,
                    symbol='star',
                    line=dict(color='black', width=2),
                    opacity=1.0
                ),
                name='Guide',
                showlegend=True
            )
    return fig
# end make_figure

tokenizer, val_dataset, jazz_dataset = initialize_data()
model = initialize_model(tokenizer)
apply_projection(model, tokenizer, val_dataset, jazz_dataset)

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='scatter-plot', figure=make_figure(None)),
    dcc.Store(id='selected-points', data={'melody': None, 'guide': None}),
    html.Div([
        # Left side: Selections info
        html.Div(id='click-output', style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Right side: Harmonize button + result
        html.Div([
            html.Button("Harmonize", id='harmonize-button', n_clicks=0),
            html.Div(id='harmonize-output', style={'marginTop': '10px'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '20px'})
    ], style={'display': 'flex', 'justifyContent': 'space-between'})
])

@app.callback(
    Output('scatter-plot', 'figure', allow_duplicate=True),
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
        if selected['melody'] is None or selected['melody'] == idx:
            selected['melody'] = idx
        elif selected['guide'] is None or selected['guide'] == idx:
            selected['guide'] = idx
        else:
            # Rotate selection
            selected['melody'], selected['guide'] = selected['guide'], idx

    token1 = df.iloc[selected['melody']]['token'].split('\n') if selected['melody'] is not None else ["None"]
    token2 = df.iloc[selected['guide']]['token'].split('\n') if selected['guide'] is not None else ["None"]

    text = html.Div([
        html.Strong("Melody:"),
        *[html.Div(line) for line in token1],
        html.Br(),
        html.Strong("Guide:"),
        *[html.Div(line) for line in token2],
    ])

    return make_figure(selected), text, selected

@app.callback(
    Output('scatter-plot', 'figure', allow_duplicate=True),
    Output('harmonize-output', 'children'),
    Input('harmonize-button', 'n_clicks'),
    State('selected-points', 'data'),
    prevent_initial_call=True,
)
def run_harmonization(n_clicks, selected):
    global df
    if not selected or selected['melody'] is None or selected['guide'] is None:
        return "Please select both a melody and a guide first."
    guide_encoded = data_all[selected['melody']]
    input_encoded = data_all[selected['guide']]
    harmony_guide = torch.LongTensor(guide_encoded['input_ids']).reshape(1, len(guide_encoded['input_ids']))
    harmony_real = torch.LongTensor(input_encoded['input_ids']).reshape(1, len(input_encoded['input_ids']))
    melody_grid = torch.FloatTensor( input_encoded['pianoroll'] ).reshape( 1, input_encoded['pianoroll'].shape[0], input_encoded['pianoroll'].shape[1] )
    conditioning_vec = torch.FloatTensor( input_encoded['time_signature'] ).reshape( 1, len(input_encoded['time_signature']) )
    pad_token_id = tokenizer.pad_token_id
    nc_token_id = tokenizer.nc_token_id
    use_constraints = False
    base2_generated_harmony = structured_progressive_generate(
        model=model,
        melody_grid=melody_grid.to(model.device),
        conditioning_vec=conditioning_vec.to(model.device),
        guiding_harmony=harmony_guide.to(model.device),
        num_stages=10,
        mask_token_id=tokenizer.mask_token_id,
        temperature=1.0,
        strategy='sample',
        pad_token_id=pad_token_id,      # token ID for <pad>
        nc_token_id=nc_token_id,       # token ID for <nc>
        force_fill=True,         # disallow <pad>/<nc> before melody ends
        chord_constraints = harmony_real.to(model.device) if use_constraints else None
    )
    gen_output_tokens = []
    for t in base2_generated_harmony[0].tolist():
        gen_output_tokens.append( tokenizer.ids_to_tokens[t] )
    # text to present to html
    gen_output_html = condenced_str_from_token_ids(base2_generated_harmony[0].tolist(), tokenizer).split('\n')
    txt = html.Div([
        html.Strong("Harmonized:"),
        *[html.Div(line) for line in gen_output_html],
    ])
    # embedding to apply triplet model transformation to
    z = model.get_z_from_harmony(base2_generated_harmony.to(device)).detach().cpu()[0].tolist()
    print('guide-z: ', F.cosine_similarity(torch.FloatTensor(z), torch.FloatTensor(guide_encoded['z']), dim=-1))
    print('input-z: ', F.cosine_similarity(torch.FloatTensor(z), torch.FloatTensor(input_encoded['z']), dim=-1))
    # appy projection
    z_proj = projector_model(torch.FloatTensor([z]).to(device)).detach().cpu().numpy()[0]
    # append new point to df
    new_point = {
        'x': z_proj[0],
        'y': z_proj[1],
        'class': 2,
        'token': gen_output_tokens,
        'hover_text': txt,
        'class_str': '2',
        'symbol': symbol_map['2'],
        'size': size_map['2']
    }
    print(z_proj)
    guide_z = [ df.iloc[selected['guide']]['x'] , df.iloc[selected['guide']]['y'] ]
    input_z = [ df.iloc[selected['melody']]['x'] , df.iloc[selected['melody']]['y'] ]
    print('proj guide-z', F.cosine_similarity(torch.FloatTensor(z_proj), torch.FloatTensor(guide_z), dim=-1))
    print('proj input-z', F.cosine_similarity(torch.FloatTensor(z_proj), torch.FloatTensor(input_z), dim=-1))
    df = pd.concat([df, pd.DataFrame([new_point])], ignore_index=True)
    return make_figure(selected), txt

if __name__ == '__main__':
    print('FUN main')
    app.run(debug=True, port=3052)