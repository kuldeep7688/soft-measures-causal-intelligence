import dash
from dash import dcc, html, ctx, dash_table, callback, ClientsideFunction
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import json
import math
import base64
from dash_selectable import DashSelectable
import io
import random
from striprtf.striprtf import rtf_to_text
from datetime import date
import itertools
"""
A tip for anyone working with this Dash app and you want to know what a specific object is for better
debugging, raise an error and call that object with it
Ex: raise ValueError(data)
Otherwise dash will only error on what the callback returns, or raise an error with no context of what part of the
function caused it (dash only says what callback triggered the error, and what the error was)
"""

#### LABELED DATA LOAD ####

llama2 = json.load(open("assets/llama2.json"))
llama3 = json.load(open("assets/llama3.json"))
mistral = json.load(open("assets/mistral.json"))

splits = json.load(open("assets/inter-rater_splits.json", "r"))

############################
#### FILE PROCESSING #######
labelers = [f'labeler{i+1}' for i in range(7)]
labeler_to_data = {
    'llama2': llama2,
    'llama3': llama3,
    'mistral': mistral
}

json_files = [llama2, llama3, mistral]


def remove_labeler_entries(data, labeler):
    # Create a new dictionary to store the filtered results
    filtered_data = {}
    # Create a list to store the texts in order
    texts_in_order = []
    #raise ValueError(data)
    #raise ValueError(data)
    # Iterate over the items in the original dictionary
    for key, value in data[labeler].items():
        filtered_data[key] = value
        texts_in_order.append(value[0])
    #raise ValueError(filtered_data)
    return filtered_data, texts_in_order



metadata_prompt = html.Div(hidden=False,children=[
    html.P(id="metadata-prompt-text",title="Please Select Your Name."),
    dcc.Dropdown(
        options=list(labelers),
        #value='Menu',
        id='dropdown-menu'),
    html.Br(),
    html.P("If your name is not in the options, enter it below, then select it from the dropdown menu:"),
    dbc.Input(id="name-input", value="Enter new name here", type="text",debounce=True),
    dbc.Button("Finished", id='metadata-finish-button'),
])

metric_dropdown = dcc.Dropdown(
    id="metric-dropdown",
    placeholder="Select LLM",
    clearable=False,
    multi=True,
    options=["All"],
    style={'width': '400px'},
)

inverse_in = html.Div(id="inverse-div", hidden=True,children=[
    dbc.Input(id='inverse-in', value='text', type='text'),
    dbc.Button("Submit", color="success", id='submit-inverse', className="me-2", n_clicks=0),
    html.Br(),
    html.Br(),
    dbc.Button("Cancel", color="danger",id='cancel-inverse',n_clicks=0),
])

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.CYBORG], meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],)

buttons = {#"source": dbc.Button('Source', id='source-btn', outline=True, color="primary", className="me-2", n_clicks=0),
           #"target": dbc.Button('Target', id='target-btn', outline=True, color="primary", className="me-3", n_clicks=0),
           "back": dbc.Button('Back', id='back-btn', outline=True, color="primary",  className="me-3", n_clicks=0),
           "next": dbc.Button('Next', id='next-btn', outline=True, color="primary",  n_clicks=0),
           "download": dbc.Button('Download JSON', id='download-btn', n_clicks=0),
           "download-llm": dbc.Button('Download JSON', id='download-btn-llm', n_clicks=0)}

datatables = {
    "labeler1": dash_table.DataTable(id="datatable-labeler1",
                                     style_cell={
                                         'height': 'auto',
                                         # all three widths are needed
                                         'minWidth': '200px', 'width': '300px', 'maxWidth': '300px',
                                         'whiteSpace': 'normal'
                                     },
                                     style_table={'height': '225px', 'overflowY': 'auto'},
                                     style_header={
                                         'backgroundColor': 'rgb(30, 30, 30)',
                                         'color': 'white'
                                     },
                                     style_data={
                                         'backgroundColor': 'rgb(50, 50, 50)',
                                         'color': 'white'
                                     },
                                     columns=[{
                                         'name': 'src',
                                         'id': "1"
                                     },
                                         {
                                             'name': 'tgt',
                                             'id': "2"
                                         },
                                         {
                                             'name': 'direction',
                                             'id': "3"
                                         }
                                     ],
                                     data=[],
                                     sort_action='native',
                                     sort_mode='single'),
    "labeler2": dash_table.DataTable(id="datatable-labeler2",
                                     style_cell={
                                         'height': 'auto',
                                         # all three widths are needed
                                         'minWidth': '200px', 'width': '300px', 'maxWidth': '300px',
                                         'whiteSpace': 'normal'
                                     },
                                     style_table={'height': '225px', 'overflowY': 'auto'},
                                     style_header={
                                         'backgroundColor': 'rgb(30, 30, 30)',
                                         'color': 'white'
                                     },
                                     style_data={
                                         'backgroundColor': 'rgb(50, 50, 50)',
                                         'color': 'white'
                                     },
                                     columns=[{
                                         'name': 'src',
                                         'id': "1"
                                     },
                                         {
                                             'name': 'tgt',
                                             'id': "2"
                                         },
                                         {
                                             'name': 'direction',
                                             'id': "3"
                                         }
                                     ],
                                     data=[],
                                     sort_action='native',
                                     sort_mode='single'),
}


llm_comparison_html = html.Div(
    [
        metadata_prompt,
        html.H5(id="sentence",className="d-grid gap-2 d-md-flex justify-content-md-center"),
        html.Br(),
        html.P(id="output2"), # output for some key functions that don't actually need an output, but Dash needs an output
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        dbc.Row([
            # dbc.Col([]),
            dbc.Col([
                datatables["labeler1"],
                dbc.Button("Left Labels", id="left-btn", n_clicks=0)
            ], className="d-grid gap-2 d-md-flex justify-content-md-center"
            ),
            dbc.Col([
                datatables["labeler2"],
                dbc.Button("Right Labels", id="right-btn", n_clicks=0)
            ], className="d-grid gap-2 d-md-flex justify-content-md-center"),
        ],
            justify='center'),
        html.Br(),
        dbc.Row(
            [
                dbc.Col([]),
                dbc.Col(
                    [
                        dbc.Button('Tie', id='tie-btn', className="me-3", n_clicks=0),
                    ],
                    className="d-grid gap-2 d-md-flex justify-content-md-center"),
                dbc.Col([]),
            ]
        ),
        html.Br(),
        html.Div([
            dbc.Button('Back', id='back-btn', outline=True, color="primary", className="me-3", n_clicks=0),
            dbc.Button('Next', id='next-btn', outline=True, color="primary", n_clicks=0),
        ],
        className="d-grid gap-2 d-md-flex justify-content-md-center"),
        html.Br(),
        html.Div(id="prev-data"),
        html.Div(id="next-data"),
        html.Br(),
        html.Br(),
        html.Br(),
        dbc.Row([
            dbc.Col([]),
            dbc.Col([]),
            dbc.Col([dcc.Upload(
                id='upload-data',
                children=html.Div([
                    dbc.Button('Select Files')
                ]),
                multiple=True),
                buttons["download"], ],
                className="d-grid gap-2 d-md-flex justify-content-end"),
        ]),

        html.Br(),
        html.Br(),
        html.Div(id="output-data-upload"),
    ]
)

llm_comparison_card = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                [
                    llm_comparison_html
                ]
            )
        ]
    )
)

tabs = dbc.Tabs(
    id="tabs",
    children =
    [
        dbc.Tab(llm_comparison_card, label="ELO Comparison", tab_id = "ELO-comparison-tab", id='ELO-comparisons')
    ]
)
app.layout = html.Div([

    html.Div([
        tabs,
        dcc.Store(id='input-sentences', data=[], storage_type='memory'),

        dcc.Store(id='all-relation-store', data=[], storage_type='memory'),
        # CHANGE BACK TO SESSION OR LOCAL FOR RESEARCHER RELEASE

        dcc.Store(id='current-relation-store',data={"src":"","tgt":"","direction":""},storage_type='memory'),
        dcc.Store(id='meta-data',data={"title": "", "authors": "", "year": ""},storage_type='memory'),
        dcc.Store(id='labelers-data',data=labeler_to_data, storage_type='memory'),
        dcc.Store(id='random-data', data=splits, storage_type='memory'),  # should be local/session
        dcc.Download(id="download-json"),  # this is needed as this is the functionality behind the download
    ],
    style={'overflow-x':'hidden'})
])


@app.callback(
    [Output('labelers-data', 'data',allow_duplicate=True),
     Output('sentence','children'),
     Output('next-btn', 'n_clicks',allow_duplicate=True),
     Output('back-btn', 'n_clicks',allow_duplicate=True)],
    [Input('next-btn', 'n_clicks'),
     Input('back-btn', 'n_clicks')],
    [State('sentence', 'children'),
     State('labelers-data', 'data'),
     State('input-sentences','data'),
     State('random-data','data'),
     State('dropdown-menu','value')],
    prevent_initial_call='initial_duplicate',
)
def next_sentence(n_clicks, back_clicks, current_text, all_data, sentences, out_data, comparator):
    current_sentence_index = int(n_clicks) - int(back_clicks)
    button_id = ctx.triggered_id if not None else False
    if not all_data:  # Prevents moving the amount of clicks, and thus the index of sentences
        # , when there is no file [On start, and after download]
        return all_data, "", 0, 0
    if current_sentence_index <= 0: # if we've gone negative, we can just reset the clicks and return default sentence
        return all_data, "", 0, 0
    key_list = list(out_data.keys())
    if len(out_data)+1 <= current_sentence_index: # This case is used when arrow keys are used instead of buttons
        # At max array size due to javascript reading button presses faster than python code can handle them
        #all_data = saving_relation(-1, all_data, curr_relation)
        return dash.no_update, out_data[key_list[current_sentence_index-2]][0], n_clicks-1, dash.no_update
    elif current_sentence_index == 1:
        return all_data, out_data[key_list[current_sentence_index-1]][0], n_clicks, back_clicks
    elif current_sentence_index < len(sentences):
        # Handling case where current relation is not filled out enough to be usable
        if button_id == "back-btn":
            index = current_sentence_index
        else:
            index = current_sentence_index - 2  # -1 because of starter sentence,-1 again because next button makes index + 1
            # of where we are saving, so -2

        #all_data = saving_relation
        return all_data, out_data[key_list[current_sentence_index-1]][0], n_clicks, back_clicks
    elif out_data[key_list[-1]][0] == current_text:
        # This case is hit when the user hits the final sentence of a paper, and hits next 1 additional time
        # This makes sure that the last sentence is saved.
        # The following code in this elif could be made into a function as it is now repeated.
        #all_data = saving_relation(-1,all_data,curr_relation)
        if button_id == "back_btn":
            return dash.no_update, out_data[key_list[current_sentence_index-2]][0], n_clicks, dash.no_update
        else:
            return dash.no_update, out_data[key_list[current_sentence_index-1]][0], n_clicks, dash.no_update
    else:
        #raise ValueError(sentences)
        return all_data, out_data[key_list[current_sentence_index-1]][0], n_clicks, back_clicks


def saving_relation(index,all_data,curr_relation):
    if curr_relation["src"] == '' or curr_relation["tgt"] == '':
        pass
    else:
        if len(all_data[index]["causal relations"]):
            check = False
            for relation in all_data[index]["causal relations"]:
                if relation == curr_relation:
                    check = True
            if not check:  # checking if it's a duplicate
                all_data[index]["causal relations"].append(curr_relation)
        else:
            all_data[index]["causal relations"].append(curr_relation)
    return all_data




@app.callback(
    [Output('datatable-labeler1', 'data'),
     Output('datatable-labeler2', 'data'),
     Output('next-data', 'children'),
     Output('prev-data', 'children')],
    Input('labelers-data', 'data'),
    [State('next-btn', 'n_clicks'),
     State('back-btn', 'n_clicks'),
     State('datatable-labeler1', 'data'),
     State('datatable-labeler1', 'columns'),
     State('datatable-labeler2', 'data'),
     State('datatable-labeler2', 'columns'),
     State('random-data','data')
     ],
)
def currentStorage(data, for_index, back_index, rows1,columns1, rows2, columns2,out_data):
    if not data:  # If there is no input file
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    index = int(for_index)-int(back_index)
    if index == 0:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    #raise ValueError(out_data)
    key_list = list(out_data.keys())
    labeler1 = out_data[key_list[index-1]][1][0]
    labeler2 = out_data[key_list[index - 1]][1][1]
    real_ind = out_data[key_list[index - 1]][2]-1
    if index <= 0:  # If we're at the starter sentence
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    elif index == 1:  # If at first sentence of paper, there is no previous sentence
        rows1 = []
        rows2 = []
        #raise ValueError(out_data[key_list[index - 1]], out_data[key_list[index - 1]][2])
        #raise ValueError(labeler1,labeler2,data[labeler1][index-1])
        for relation in data[labeler1][real_ind]['causal relations']:
            temp_dict = {}
            for c, val in zip(columns1, relation):
                if val == "direction":
                    if relation[val].lower() == "positive":
                        relation[val] = "increase"
                    if relation[val].lower() == "negative":
                        relation[val] = "decrease"
                    relation[val] = relation[val].lower()
                temp_dict[c['id']] = relation[val]
            rows1.append(temp_dict)
            #rows1.append({c['id']: relation[val] for c, val in zip(columns1,relation)})
        for relation in data[labeler2][real_ind]['causal relations']:
            temp_dict = {}
            for c, val in zip(columns1, relation):
                if val == "direction":
                    if relation[val].lower() == "positive":
                        relation[val] = "increase"
                    if relation[val].lower() == "negative":
                        relation[val] = "decrease"
                    relation[val] = relation[val].lower()
                temp_dict[c['id']] = relation[val]
            rows2.append(temp_dict)
        if len(data)>1:
            return rows1, rows2, f"Next Passage: {data[labeler1][real_ind]['text']}", "Previous Passage: []"
        else:
            return rows1, rows2, f"Next Passage: []", "Previous Passage: []"
    elif len(data[labeler1]) <= index:  # If we're at EOF, there is no next sentence
        rows1 = []
        rows2 = []
        #index = len(data[labeler1])
        for relation in data[labeler1][real_ind]['causal relations']:
            temp_dict = {}
            for c, val in zip(columns1, relation):
                if val == "direction":
                    if relation[val].lower() == "positive":
                        relation[val] = "increase"
                    if relation[val].lower() == "negative":
                        relation[val] = "decrease"
                    relation[val] = relation[val].lower()
                temp_dict[c['id']] = relation[val]
            rows1.append(temp_dict)
            # rows1.append({c['id']: relation[val] for c, val in zip(columns1,relation)})
        for relation in data[labeler2][real_ind]['causal relations']:
            temp_dict = {}
            for c, val in zip(columns1, relation):
                if val == "direction":
                    if relation[val].lower() == "positive":
                        relation[val] = "increase"
                    if relation[val].lower() == "negative":
                        relation[val] = "decrease"
                    relation[val] = relation[val].lower()
                temp_dict[c['id']] = relation[val]
            rows2.append(temp_dict)
        return rows1,rows2, f"Next Passage: []", f"Previous Passage: {data[labeler1][real_ind]['text']}"
    else:
        rows1 = []
        rows2 = []
        for relation in data[labeler1][real_ind]['causal relations']:
            temp_dict = {}
            for c, val in zip(columns1, relation):
                if val == "direction":
                    if relation[val].lower() == "positive":
                        relation[val] = "increase"
                    if relation[val].lower() == "negative":
                        relation[val] = "decrease"
                    relation[val] = relation[val].lower()
                temp_dict[c['id']] = relation[val]
            rows1.append(temp_dict)
            # rows1.append({c['id']: relation[val] for c, val in zip(columns1,relation)})
        for relation in data[labeler2][real_ind]['causal relations']:
            temp_dict = {}
            for c, val in zip(columns1, relation):
                if val == "direction":
                    if relation[val].lower() == "positive":
                        relation[val] = "increase"
                    if relation[val].lower() == "negative":
                        relation[val] = "decrease"
                    relation[val] = relation[val].lower()
                temp_dict[c['id']] = relation[val]
            rows2.append(temp_dict)
        return rows1, rows2, f"Next Passage: {data[labeler1][real_ind]['text']}", f"Previous Passage: {data[labeler1][real_ind]['text']}"


@app.callback(
    [Output("download-json", "data"),
     Output('labelers-data','data'),
     Output('input-sentences','data', allow_duplicate=True),
     Output('next-btn','n_clicks'),],
    [Input("download-btn", "n_clicks"),],
    [State('labelers-data','data'),
     State('random-data','data'),
     State('next-btn','n_clicks'),
     State('input-sentences','data'),
     State('upload-data', 'filename'),
     State('dropdown-menu','value')
     ],
    prevent_initial_call=True,
)
def download(n_clicks, data, out_data, curr_sen_index, inp_sentences,file, name):
    # In current implementation, only required variables are the input (download-btn)
    # and the state of all-relation-store
    """

    :param file:
    :param n_clicks2:
    :param n_clicks:
    :param data:
    :param curr_sen_index:
    :param inp_sentences:
    :return: json, relational storage, input_sentences, next btn n_clicks
    """
    # WHEN YOU HIT SAVE, YOU ARE DONE WITH THAT SESSION, ALL REMAINING SENTENCES ARE REMOVED, AND THE PROGRAM IS
    # BASICALLY RESET
    if not data:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    fileData = json.dumps(out_data, indent=2)
    today = date.today()
    file_out = name + "_elo_comparison_" + f"{today}.json"
    return dict(content=fileData, filename=file_out), [], ["Please Insert RTF or JSON File"], 0


# This callback also activates on download, and updates the text on screen.


@app.callback(
    Output('output-data-upload', 'children', allow_duplicate=True),
    Input('input-sentences','data'),
    prevent_initial_call='initial_duplicate',
)
def refresh(inp_sentences):
    return f"Current Sentences: {inp_sentences}" + f" Length: {len(inp_sentences)}"


@app.callback([Output('input-sentences','data'),
               Output('labelers-data','data', allow_duplicate=True),
               Output('random-data', 'data'),],
              Input('upload-data', 'contents'),
              [State('upload-data', 'filename'),
               State('input-sentences','data'),
               State('random-data','data'),],
              prevent_initial_call="initial_duplicate"
)
def upload(list_of_contents, list_of_names,inp_sentences,random_list):
    if list_of_contents is not None:
        if len(list_of_contents) != 2:
            pass
        children = [
            parse_contents(c, n) for c, n in
            zip(list_of_contents, list_of_names)]

        for obj in children[0][0]:
            inp_sentences.append(obj['text'])
            text = obj['text']
            if random.random() > 0.5:
                if 'labeler' in children[0][0][0]['meta_data'].keys() and 'labeler' in children[1][0][0]['meta_data'].keys():
                    random_list[text] = {"labelers":[children[0][0][0]["meta_data"]["labeler"],children[1][0][0]['meta_data']['labeler']]}
                elif 'labeler' in children[0][0][0]['meta_data'].keys():
                    random_list[text] = {"labelers":[children[0][0][0]['meta_data']['labeler'],'User2']}
                elif 'labeler' in children[1][0][0]['meta_data'].keys():
                    random_list[text] = {"labelers":['User1',children[1][0][0]['meta_data']['labeler']]}
                else:
                    random_list[text] = {"labelers":['User1', 'User2']}
            else:
                if 'labeler' in children[0][0][0]['meta_data'].keys() and 'labeler' in children[1][0][0]['meta_data'].keys():
                    random_list[text] = {"labelers":[children[1][0][0]["meta_data"]["labeler"], children[0][0][0]['meta_data']['labeler']]}
                elif 'labeler' in children[0][0][0]['meta_data'].keys():
                    random_list[text] = {"labelers":['User2', children[0][0][0]['meta_data']['labeler']]}
                elif 'labeler' in children[1][0][0]['meta_data'].keys():
                    random_list[text] = {"labelers":[children[1][0][0]['meta_data']['labeler'], 'User1']}
                else:
                    random_list[text] = {"labelers": ['User2', 'User1']}
        return inp_sentences, [children[0][0], children[1][0]], random_list
    return dash.no_update, dash.no_update, dash.no_update


def parse_contents(contents,filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if ".json" in filename:
            data = json.loads(decoded)
            return data, filename
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing the file."])


@app.callback([Output('random-data', 'data',allow_duplicate=True),
               Output('next-btn','n_clicks',allow_duplicate=True)],
              [Input('left-btn', 'n_clicks'),
               Input('right-btn', 'n_clicks'),
               Input('tie-btn','n_clicks')],
              [State('next-btn','n_clicks'),
               State('back-btn','n_clicks'),
               State('sentence','children'),
               State('random-data','data'),],
              prevent_initial_call=True
)
def elo_update(left, right, tie, next_click, back, text, elo_data):
    index = int(next_click)-int(back)
    key_list = list(elo_data.keys())
    location = key_list[index-1]
    labeler1 = elo_data[key_list[index - 1]][1][0]
    labeler2 = elo_data[key_list[index - 1]][1][1]
    if ctx.triggered_id == 'left-btn':
        if len(elo_data[location]) == 3:
            elo_data[location].append(labeler1)
        else:
            elo_data[location][3] = labeler1
        return elo_data,next_click+1
    elif ctx.triggered_id == 'right-btn':
        if len(elo_data[location]) == 3:
            elo_data[location].append(labeler2)
        else:
            elo_data[location][3] = labeler2
        return elo_data,next_click+1
    else:
        if len(elo_data[location]) == 3:
            elo_data[location].append("Tie")
        else:
            elo_data[location][3] = "Tie"
        return elo_data, next_click + 1

# Arrow key controls
# event.key == 37 is for left arrow
# event.key == 39 is for right arrow
app.clientside_callback(
    """
        function(id) {
            document.addEventListener("keydown", function(event) {
                if (event.keyCode == '37') {
                    document.getElementById('back-btn').click()
                    event.stopPropogation()
                }
                if (event.keyCode == '39') {
                    document.getElementById('next-btn').click()
                    event.stopPropogation()
                }
            });
            return window.dash_clientside.no_update       
        }
    """,
    Output("back-btn", "id"),
    Input("back-btn", "id"),

)

@app.callback(
    Output("output2", "children", allow_duplicate=True),
    Input("back-btn", "n_clicks"),
    Input("next-btn", "n_clicks"),
    State("input-sentences","data"),
    prevent_initial_call=True
)
def show_value(n1, n2,data):
    index = int(n2)-int(n1)
    if len(data) == 0:
        return f"Index: {index}, Total Passages: 0"
    if ctx.triggered_id == 'back-btn':
        return f"Index: {index}, Total Passages: {len(data)}"
    if index == 0: # without this, get caught in case 3 EOF at 0 index
        if ctx.triggered_id == 'next-btn':
            return f"Index: {index}, Total Passages: {len(data)}"
    if len(data[0]) == index:
        return f"Index: {index}, Total Passages: {len(data)}, EOF"
    elif len(data[0]) < index:
        return f"Index: {index}, Total Passages: {len(data)}, Past EOF"
    return f"Index: {index}, Total Passages: {len(data)}"


@app.callback(
    Output("dropdown-menu", "options"),
    [Input("name-input", "value")],
    State("dropdown-menu", "options"),
    prevent_initial_call=True
)
def output_text(value, options):
    options.append(value)
    return options


@app.callback([Output(metadata_prompt,'hidden',allow_duplicate=True),
               Output('random-data','data', allow_duplicate=True),
               Output('input-sentences','data', allow_duplicate=True)],
              Input('metadata-finish-button', 'n_clicks'),
              [State('random-data','data'),
               State('dropdown-menu','value')],
              prevent_initial_call=True
)
def metadata(n_clicks, data, comparator):
    data, texts = remove_labeler_entries(data, comparator)
    return True, data, texts


if __name__ == '__main__':
    app.run_server(debug=False)
