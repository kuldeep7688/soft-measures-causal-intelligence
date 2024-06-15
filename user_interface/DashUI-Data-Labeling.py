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
from striprtf.striprtf import rtf_to_text
from datetime import date
"""
Functionality ideas:
- Could write "helper" functions for callbacks to increase readability of callbacks

Functionality to be added:
- Ability to read in files (besides RTF) and be added to sentences for data labeling (Look at: Dash upload component)

Functionality to be updated:
- (Not Required) Being able to choose the file name for the download

Unexpected (or frustrating) Behavior:
- Clicking anywhere on the same "y" as the upload button opens the file menu
- After saving a json, the input sentences are removed and the program is basically reset
-- However, even though it is reset, you cannot upload the same file consecutively.
-- You CAN upload 1 paper, then upload a second paper, and they will combine in the storage.
--- This problem likely occurs based on how dash is handling uploads, and may not be fixable. 
--- Also, this issue may not be relevant as why would you upload the same thing multiple times consecutively.
--- This issue is probably due to the filename not changing within the app, thus not invoking the callback.
Errors in Functionality:
- If you hold down the right arrow key, it will out of index on the function for the datatable, this affects nothing for
- the program as far as I can tell, and the error it throws does not crash the program.
- Weird infrequent error where next/back count increases by an additional 1 every live update from dash
-- There should be no live update for users as this occurs when the code or css is edited
-- A fix could be to put the JS code into its own asset file, and maybe it won't dupe the event reader
- UI is best on and designed for a 1920x1080 monitor, and needs a way to scale to other sizes
"""

metadata_prompt = html.Div(hidden=True,children=[
    html.P(id="metadata-prompt-text",title="Please enter the following metadata."),
    html.Div([
        dbc.Input(id='title', value='Title', type='text'),
        dbc.Input(id='author', value='Author(s)', type='text'),
        dbc.Input(id='year', value='Year', type='text'),
    ]),
    html.Br(),
    dbc.Button("Finished",id='metadata-finish-button'),
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

app.layout = html.Div([

    html.Div([
        metadata_prompt,
        inverse_in,
        html.Div([
        DashSelectable(
            id="dash-selectable",
            children=[html.H5(id="sentence",className="d-grid gap-2 d-md-flex justify-content-md-center"), html.P(id="output")],
        ),],),
        html.Br(),
        html.P(id="output2"), # output for some key functions that don't actually need an output, but Dash needs an output
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div(id='my-source')
                ]),
                dbc.Col([
                    dbc.Button('Source', id='source-btn', outline=True, color="primary", className="me-2", n_clicks=0),
                    dbc.Button('+', id='increase-btn', outline=True, color="primary", className="me-2", n_clicks=0),
                ],
                    className="d-grid gap-2 d-md-flex justify-content-md-center", ),
                dbc.Col([]),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id='my-target'),
                ]),
                dbc.Col([
                    dbc.Button('Target', id='target-btn', outline=True, color="primary", className="me-3", n_clicks=0),
                    dbc.Button('-', id='decrease-btn', outline=True, color="primary", className="me-2", n_clicks=0),
                ],
                    className="d-grid gap-2 d-md-flex justify-content-md-center", ),
                dbc.Col([])

            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id='my-direction'),
                ]),
                dbc.Col([]),
                dbc.Col([]),
                dbc.Col([]),
                dbc.Col([]),
                dbc.Col([dash_table.DataTable(id="datatable-metrics",
                                                    style_cell={
                                                        'height': 'auto',
                                                        # all three widths are needed
                                                        'minWidth': '10px', 'width': '10px', 'maxWidth': '10px',
                                                        'whiteSpace': 'normal'
                                                    },
                                                    # style_table={'height': '225px', 'overflowY': 'auto'},
                                                    style_header={
                                                        'backgroundColor': 'rgb(30, 30, 30)',
                                                        'color': 'white'
                                                    },
                                                    style_data={
                                                        'backgroundColor': 'rgb(50, 50, 50)',
                                                        'color': 'white'
                                                    },

                                                    merge_duplicate_headers=True,
                                                    )],
                        width=3,
                        align="center"),
                dbc.Col([]), ],
            ),
            #dbc.Row([
            #    dbc.Col([]),
            #    dbc.Col([]),
            #    dbc.Col([]),
            #], justify="evenly")
        ], className="pad-row"),
        html.Br(),
        html.Br(),
        dbc.Button('Save Relation', id='save-btn', outline=True, color="success", className="me-5", n_clicks=0),
        dbc.Button('Reset', id='reset-btn', outline=True, color="warning", n_clicks=0),
        html.Br(),
        html.Br(),
        dbc.Row([
            # dbc.Col([]),
            dbc.Col([
                dash_table.DataTable(id="datatable-current",
                                     style_cell={
                                         'height': 'auto',
                                         # all three widths are needed
                                         'minWidth': '120px', 'width': '120px', 'maxWidth': '120px',
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
                                     editable=True,
                                     row_deletable=True,
                                     ),
            ], width=9,
            ),
            # dbc.Col([]),
        ],
            justify='center'),
        html.Br(),
        html.Div([
            dbc.Button('Back', id='back-btn', outline=True, color="primary",  className="me-3", n_clicks=0),
            dbc.Button('Next', id='next-btn', outline=True, color="primary",  n_clicks=0),
        ],
        className="d-grid gap-2 d-md-flex justify-content-md-center"),
        html.Br(),
        html.Div(id="prev-data"),
        html.Div(id="next-data"),
        html.Br(),
        html.Br(),
        dbc.Row([
            dbc.Col([dbc.Button('Discard Current Sentence', outline=True, color="danger", id="discard-btn")],),
            dbc.Col([]),
            dbc.Col([dbc.Button('Display LLM Outputs',id='llm-output-btn')],
                    className="d-grid gap-2 d-md-flex justify-content-end"),
        ]),
        #dbc.Button('Discard Current Sentence', outline=True, color="danger", id="discard-btn"),
        html.Br(),
        html.Br(),
        html.Br(),
        dbc.Row([
            dbc.Col([dbc.Button('Modify and add new sentence', outline=True, color="info", id="inverse-btn")],),
            dbc.Col([]),
            dbc.Col([dcc.Upload(
                id='upload-data',
                children=html.Div([
                    dbc.Button('Select Files')
            ]),),
            dbc.Button('Download JSON', id='download-btn', n_clicks=0),],
            className="d-grid gap-2 d-md-flex justify-content-end"),
        ]),
        #dbc.Button('Modify and add new sentence', outline=True, color="info", id="inverse-btn"),
        html.Br(),
        html.Br(),

        html.Br(),
        html.Br(),
        html.Div(id="output-data-upload"),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("LLM Output Comparison")),
                dbc.ModalBody(children=[
                    dash_table.DataTable(id="datatable-llm-output",
                                        style_cell={
                                            'height': 'auto',
                                            # all three widths are needed
                                            'minWidth': '10px', 'width': '10px', 'maxWidth': '10px',
                                            'whiteSpace': 'normal'
                                        },
                                        # style_table={'height': '225px', 'overflowY': 'auto'},
                                        style_header={
                                            'backgroundColor': 'rgb(30, 30, 30)',
                                            'color': 'white'
                                        },
                                        style_data={
                                            'backgroundColor': 'rgb(50, 50, 50)',
                                            'color': 'white'
                                        },

                                        merge_duplicate_headers=True,
                    )
                ]),
            ],
            id="modal-xl",
            size="xl",
            is_open=False,
        ),
        dcc.Store(id='input-sentences', data=["Please Insert RTF or JSON File"], storage_type='local'),

        dcc.Store(id='all-relation-store', data=[], storage_type='local'),
        # CHANGE BACK TO SESSION OR LOCAL FOR RESEARCHER RELEASE

        dcc.Store(id='current-relation-store',data={"src":"","tgt":"","direction":""},storage_type='memory'),
        dcc.Store(id='meta-data',data={"title": "", "authors": "", "year": ""},storage_type='memory'),
        dcc.Store(id='llm-metrics',data={}, storage_type='local'),
        dcc.Store(id='llm-scores',data={}, storage_type='local'),
        #dcc.Store(id='index-store',data=0, storage_type='memory'),
        dcc.Download(id="download-json"),
    ],
    style={'overflow-x':'hidden'})
])


@app.callback(Output("output", "children"),
              [Input("dash-selectable", "selectedValue")])
def display_output(value):
    text = ""
    if value:
        text = value

    return "Currently selected: {}".format(text)


@app.callback(
    [Output('all-relation-store', 'data',allow_duplicate=True),
     Output('sentence','children'),
     Output('current-relation-store', 'data',allow_duplicate=True),
     Output('next-btn', 'n_clicks',allow_duplicate=True),
     Output('back-btn', 'n_clicks',allow_duplicate=True)],
    [Input('next-btn', 'n_clicks'),
     Input('back-btn', 'n_clicks')],
    [State('sentence', 'children'),
     State('all-relation-store', 'data'),
     State('current-relation-store', 'data'),
     State('input-sentences','data'),],
    prevent_initial_call='initial_duplicate',
)
def next_sentence(n_clicks, back_clicks, current_text, all_data,curr_relation,sentences):
    current_sentence_index = int(n_clicks) - int(back_clicks)
    button_id = ctx.triggered_id if not None else False
    if len(sentences) == 1:  # Prevents moving the amount of clicks, and thus the index of sentences
        # , when there is no file [On start, and after download]
        return all_data, "Please Insert RTF or JSON File", curr_relation, 0, 0
    if current_sentence_index < 0: # if we've gone negative, we can just reset the clicks and return default sentence
        return all_data, "Please Insert RTF or JSON File", curr_relation, 0, 0
    if len(all_data) <= current_sentence_index: # This case is used when arrow keys are used instead of buttons
        # At max array size
        if curr_relation["src"] == '' or curr_relation["tgt"] == '':
            return dash.no_update, all_data[-1]["text"], curr_relation, len(all_data), 0
        all_data = saving_relation(-1, all_data, curr_relation)
        curr_relation = {'src': "", 'tgt': '', 'direction': ''}
        return all_data, all_data[-1]["text"], curr_relation, len(all_data), 0
    if current_sentence_index == 0:
        return all_data, sentences[current_sentence_index], curr_relation, 0, 0
    elif current_sentence_index == 1:
        return all_data, all_data[current_sentence_index-1]["text"], curr_relation, n_clicks, back_clicks
    elif current_sentence_index < len(sentences):
        # Handling case where current relation is not filled out enough to be usable
        if button_id == "back-btn":
            index = current_sentence_index
        else:
            index = current_sentence_index - 2  # -1 because of starter sentence,-1 again because next button makes index + 1
            # of where we are saving, so -2
        all_data = saving_relation(index,all_data,curr_relation)
        curr_relation = {'src':"",'tgt':'','direction':''}
        return all_data, all_data[current_sentence_index-1]["text"], curr_relation, n_clicks, back_clicks
    elif all_data[-1]["text"] == current_text:
        # This case is hit when the user hits the final sentence of a paper, and hits next 1 additional time
        # This makes sure that the last sentence is saved.
        # The following code in this elif could be made into a function as it is now repeated.
        all_data = saving_relation(-1,all_data,curr_relation)
        curr_relation = {'src': "", 'tgt': '', 'direction': ''}
        if button_id == "back_btn":
            return all_data, all_data[-2]["text"], curr_relation, n_clicks, back_clicks
        else:
            return all_data, all_data[-1]["text"], curr_relation, n_clicks-1, back_clicks
    else:
        return all_data, current_text, curr_relation, n_clicks, back_clicks

# Callback for increase, decrease, source,target, save, and reset in the following


@app.callback(
    [Output('my-direction', 'children'),
     Output('my-source', 'children'),
     Output('my-target', 'children'),
     Output("current-relation-store", "data")],
    [Input('increase-btn', 'n_clicks'),
     Input('decrease-btn', 'n_clicks'),
     Input('source-btn', 'n_clicks'),
     Input('target-btn', 'n_clicks'),
     Input('next-btn', 'n_clicks'),
     Input('reset-btn', 'n_clicks')],
    [State("dash-selectable", "selectedValue"),
     State("current-relation-store", "data"),
     State("inverse-div",'hidden')
     ],
)
def allLabel(inc, dec, src, tgt, next, reset, selected_data, relation, modifier_block):
    """
    Function that handles all relation button data
    :param inc: Increase button
    :param dec: Decrease button
    :param src: Source button
    :param tgt: Target button
    :param next: Next button - allows next button to access function
    :param reset: Reset button - allows reset button to access function
    :param selected_data: User-selected data
    :param relation: Relation data storage
    :return: [Direction text, ]
    """
    button_id = ctx.triggered_id if not None else False
    direcText = f"Direction: "
    srcText = f"Source: "
    tgtText = f"Target: "
    if button_id == "reset-btn":
        relation = {'src': "", 'tgt': '', 'direction': ''}
        return direcText, srcText, tgtText, relation
    if not modifier_block: # not modifier_block because this is visibility, when it is visible, we do not allow
        return dash.no_update, dash.no_update,dash.no_update, dash.no_update
    if button_id == "increase-btn":
        relation["direction"] = "Increase"
        return f"Direction: Increase",dash.no_update, dash.no_update,relation
    elif button_id == "decrease-btn":
        relation["direction"] = "Decrease"
        return f"Direction: Decrease",dash.no_update, dash.no_update,relation
    elif button_id == "source-btn":
        relation["src"] = selected_data
        return dash.no_update, f"Source: {selected_data}", dash.no_update,relation
    elif button_id == "target-btn":
        relation["tgt"] = selected_data
        return dash.no_update, dash.no_update, f"Target: {selected_data}",relation
    else:  # This else corresponds to initial call (program start) and when the next button is hit
        # Have not tried multiple changes to one output from one button,
        # and it probably isn't a good idea, so don't change this
        return direcText,srcText,tgtText,relation


@app.callback(
    [Output('all-relation-store','data',allow_duplicate=True),
     Output('current-relation-store','data',allow_duplicate=True)],
    [Input('save-btn', 'n_clicks')],
    [State('current-relation-store','data'),
     State('all-relation-store', 'data'),
     State('next-btn', 'n_clicks'),
     State('back-btn', 'n_clicks'),
     State("inverse-div",'hidden')],
    prevent_initial_call=True,
)
def save_relation(n_clicks,curr_relation,all_data,for_index,back_index, modifier_block):
    index = int(for_index)-int(back_index)
    if not modifier_block: # not modifier_block because this is visibility, when it is visible, we do not allow
       return dash.no_update, dash.no_update
    if index <= 0:
        return all_data,dash.no_update
    all_data = saving_relation(index-1,all_data,curr_relation)
    return all_data,dash.no_update


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
    [Output('datatable-current', 'data'),
     Output('next-data', 'children'),
     Output('prev-data', 'children')],
    Input('all-relation-store', 'data'),
    [State('next-btn', 'n_clicks'),
     State('back-btn', 'n_clicks'),
     State('datatable-current', 'data'),
     State('datatable-current', 'columns')]
)
def currentStorage(data, for_index, back_index, rows,columns):
    if not data:  # If there is no input file
        return [], dash.no_update, dash.no_update
    index = int(for_index)-int(back_index)
    if index <= 0:  # If we're at the starter sentence
        return dash.no_update, dash.no_update, dash.no_update
    elif index == 1:  # If at first sentence of paper, there is no previous sentence
        rows = []
        for relation in data[index-1]['causal relations']:
            rows.append({c['id']: relation[val] for c, val in zip(columns,relation)})
        if len(data)>1:
            return rows, f"Next Passage: {data[index]['text']}", "Previous Passage: []"
        else:
            return rows, f"Next Passage: []", "Previous Passage: []"
    elif len(data) <= index:  # If we're at EOF, there is no next sentence
        rows = []
        index = len(data)
        for relation in data[index - 1]['causal relations']:
            rows.append({c['id']: relation[val] for c, val in zip(columns, relation)})
        return rows, f"Next Passage: []", f"Previous Passage: {data[index - 2]['text']}"
    else:
        rows = []
        for relation in data[index - 1]['causal relations']:
            rows.append({c['id']: relation[val] for c, val in zip(columns, relation)})
        return rows, f"Next Passage: {data[index]['text']}", f"Previous Passage: {data[index - 2]['text']}"


@app.callback(
    [Output('all-relation-store', 'data', allow_duplicate=True),
     Output('datatable-current', 'data', allow_duplicate=True)],
    Input('datatable-current', 'data'),
    [State('all-relation-store', 'data'),
     State('next-btn', 'n_clicks'),
     State('back-btn', 'n_clicks'),],
    prevent_initial_call=True
)
def updating_json(rows,data,next_index,back_index):
    """
    This function updates the JSON after the editable dash datatable has been changed.
    :param value:
    :return:
    """
    if len(data)==0:
        raise PreventUpdate
    index = int(next_index)-int(back_index)
    conv = []
    for row, i in zip(rows,range(len(rows))):  # row is a singular relation
        temp = {}
        temp["src"] = row["1"]
        temp["tgt"] = row["2"]
        temp["direction"] = row["3"]
        if temp["direction"] == "+":
            temp["direction"] = 'increase'
        if temp["direction"] == "-":
            temp["direction"] = 'decrease'
        if temp["direction"] != "increase" and temp["direction"] != "decrease":
            temp["direction"] = data[index-1]['causal relations'][i]['direction']
        if temp["src"] == "":  # if any parameters are empty, restore that part of the relation
            temp["src"] = data[index - 1]['causal relations'][i]['src']
        if temp["tgt"] == "":
            temp["tgt"] = data[index - 1]['causal relations'][i]['tgt']
        conv.append(temp)

    data[index-1]['causal relations'] = conv
    return data, rows

@app.callback(
    [Output("download-json", "data"),
     Output('all-relation-store','data'),
     Output('input-sentences','data', allow_duplicate=True),
     Output('next-btn','n_clicks'),
     Output('llm-metrics','data', allow_duplicate=True),
     Output('llm-scores','data', allow_duplicate=True),
     ],
    Input("download-btn", "n_clicks"),
    [State('all-relation-store','data'),
     State('next-btn','n_clicks'),
     State('input-sentences','data'),
     State('upload-data', 'filename'),
     ],
    prevent_initial_call=True,
)
def download(n_clicks,data,curr_sen_index, inp_sentences,file):
    # In current implementation, only required variables are the input (download-btn)
    # and the state of all-relation-store
    """

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
    fileData = json.dumps(data, indent=2)
    today = date.today()
    if file is None:
        return dict(content=fileData, filename=f"Labeled_Data-{today}.json"), [], ["Please Insert RTF or JSON File"], 0, {}, {}
    file = file.replace(".rtf",f"-{today}.json")
    return dict(content=fileData, filename=file), [], ["Please Insert RTF or JSON File"], 0, {}, {}


# This callback also activates on download, and updates the text on screen.


@app.callback(
    Output('output-data-upload', 'children', allow_duplicate=True),
    Input('input-sentences','data'),
    prevent_initial_call='initial_duplicate',
)
def refresh(inp_sentences):
    return f"Current Sentences: {inp_sentences}" + f" Length: {len(inp_sentences)}"


def abbreviation_handler(sentences):
    # File handler helper function
    sentences_to_add = []
    temp = sentences[0]
    for i in range(len(sentences) - 1):
        if sentences[i] == '':
            continue
        if not (sentences[i + 1].strip())[0].isupper():
            temp = temp + '. ' + sentences[i + 1]
        else:
            sentences_to_add.append(temp)
            temp = sentences[i + 1]
    sentences_to_add.append(temp)
    return sentences_to_add


@app.callback([Output('input-sentences','data'),
               Output('all-relation-store','data', allow_duplicate=True),
               Output(metadata_prompt,'hidden'),
               Output('llm-metrics','data'),
               Output('llm-scores','data')],
              Input('upload-data', 'contents'),
              [State('upload-data', 'filename'),
               State('input-sentences','data'),
               State('all-relation-store','data'),
               State('llm-metrics','data'),
               State('llm-scores','data')],
              prevent_initial_call="initial_duplicate"
)
def upload(list_of_contents, list_of_names,inp_sentences,data,LLM_metrics,LLM_scores):
    if list_of_contents is None:
        if len(inp_sentences) > 1:
            return inp_sentences, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    content_type, content_string = list_of_contents.split(',')
    decoded = base64.b64decode(content_string)
    if ".json" in list_of_names:
        data = json.loads(decoded)
        for sentence in data:
            inp_sentences.append(sentence["text"])
        if 'LLM' in data[0].keys():
            for LLM in data[0]['LLM']:
                LLM_scores[LLM] = {"TP":0, "FP":0, "TN":0, "FN": 0}
                LLM_metrics[LLM] = {}
            for sentence in data:
                for LLM in sentence['LLM'].keys():  # LLM is a list of relations
                    for relation in sentence['causal relations']:
                        if relation not in sentence['LLM'][LLM]:
                            LLM_scores[LLM]['FN'] += 1
                        else:
                            LLM_scores[LLM]['TP'] += 1
                    if len(LLM) == 0:
                        if len(sentence['causal relations']) == 0:
                            LLM_scores[LLM]['TN'] += 1
                    for relation in sentence['LLM'][LLM]:
                        if relation not in sentence['causal relations']:
                            LLM_scores[LLM]['FP'] += 1
                        # Don't need an else here, as that'd be a true positive and is already added
            for LLM in LLM_scores:
                LLM_metrics[LLM]['precision'] = LLM_scores[LLM]['TP'] / (LLM_scores[LLM]['TP'] + LLM_scores[LLM]['FP'])
                LLM_metrics[LLM]['recall'] = LLM_scores[LLM]['TP'] / (LLM_scores[LLM]['TP'] + LLM_scores[LLM]['FN'])
                LLM_metrics[LLM]['F1'] = (2 * LLM_metrics[LLM]['precision'] * LLM_metrics[LLM]['recall']) / (LLM_metrics[LLM]['precision'] + LLM_metrics[LLM]['recall'])
                LLM_metrics[LLM]['accuracy'] = ((LLM_scores[LLM]['TP'] + LLM_scores[LLM]['TN']) /
                                                (LLM_scores[LLM]['TP'] + LLM_scores[LLM]['TN'] + LLM_scores[LLM]['FP'] + LLM_scores[LLM]['FN']))
            return inp_sentences, data, dash.no_update, LLM_metrics, LLM_scores
        return inp_sentences, data, dash.no_update, dash.no_update, dash.no_update
    if ".rtf" in list_of_names:
        temp = io.StringIO(decoded.decode('utf-8')).getvalue()
        text = rtf_to_text(temp)
        period_split = text.split(". ")
        sentences = []
        for sentence in period_split:
            temp = sentence.split(".\n")
            if type(temp) is str:
                sentences.append(sentence)
            else:
                for sen in temp:
                    if sen == "":
                        continue
                    sentences.append(sen)
        sentences = abbreviation_handler(sentences)
        for sentence in sentences:
            if sentence == '':
                continue
            sentence = sentence.replace("\n", "")
            sentence = sentence + "."
            inp_sentences.append(sentence)
            template = {"text": sentence,
                        "causal relations": [],
                        "meta_data": {"title": "", "authors": "", "year": ""}}
            data.append(template)
    if ".txt" in list_of_names:
        text = io.StringIO(decoded.decode('utf-8')).getvalue()
        newline_split = text.split("\n")
        for sentence in newline_split:
            if sentence == '':
                continue
            sentence = sentence.replace("\r", "")
            inp_sentences.append(sentence)
            template = {"text": sentence,
                        "causal relations": [],
                        "meta_data": {"title": "", "authors": "", "year": ""}}
            data.append(template)
    return inp_sentences, data, False, dash.no_update, dash.no_update


@app.callback([Output(metadata_prompt,'hidden',allow_duplicate=True),
               Output('all-relation-store','data', allow_duplicate=True)],
              Input('metadata-finish-button', 'n_clicks'),
              [State('title', 'value'),
               State('author','value'),
               State('year','value'),
               State('all-relation-store','data'),],
              prevent_initial_call="initial_duplicate"
)
def metadata(n_clicks, title, author, year, data):
    meta_dict = {"title": title, "authors": author, "year": year}
    for i in range(len(data)):
        if data[i]["meta_data"] == {"title": "", "authors": "", "year": ""}:
            data[i]["meta_data"] = meta_dict
    return True, data


@app.callback(
               Output("inverse-div",'hidden',allow_duplicate=True),
              Input('inverse-btn', 'n_clicks'),
              [State("inverse-div",'hidden'),
               State('all-relation-store','data'),
               State('next-btn', 'n_clicks'),
               State('back-btn', 'n_clicks'),
               State('inverse-in', 'value')],
              prevent_initial_call=True
)
def modify(n_clicks, editable, data,for_index,back_index,input_val):
    if not data:
        return dash.no_update
    index = int(for_index)-int(back_index)
    if index == 0:
        return dash.no_update
    if editable:

        return False
    else:
        return dash.no_update


@app.callback([
               Output("inverse-div",'hidden',allow_duplicate=True),
               Output('all-relation-store','data', allow_duplicate=True),
               Output('sentence','children', allow_duplicate=True),
               Output('input-sentences','data', allow_duplicate=True)],
              [Input('submit-inverse', 'n_clicks'),
               Input('cancel-inverse', 'n_clicks')],
              [State("inverse-div",'hidden'),
               State('sentence','children'),
               State('all-relation-store','data'),
               State('next-btn', 'n_clicks'),
               State('back-btn', 'n_clicks'),
               State('inverse-in', 'value'),
               State('input-sentences', 'data')],
              prevent_initial_call=True
)
def save_inverse(n_clicks, n_clicks2, visible, sen, data,for_index,back_index,input_val,sentence_list):
    trigger = ctx.triggered_id
    if trigger == "cancel-inverse":
        return True, dash.no_update, dash.no_update, dash.no_update
    index = int(for_index) - int(back_index)
    relations = []
    for relation in data[index - 1]["causal relations"]:  # -1 because data does not have starter sentence
        temp = dict(relation)
        if temp["direction"] == "increase":
            temp["direction"] = "decrease"
        else:
            temp["direction"] = "increase"
        relations.append(temp)
    template = {"text": input_val,
                "causal relations": relations,
                "meta_data": data[index]["meta_data"]}
    data.insert(index, template)
    sentence_list.insert(index + 1, input_val)
    return True, data, dash.no_update, sentence_list


@app.callback([
               Output('sentence','children', allow_duplicate=True),
               Output('inverse-in', 'value',allow_duplicate=True),],
              Input('inverse-div', 'hidden'),
              [State('sentence','children'),
               ],
              prevent_initial_call=True
)
def inverse_pt2(hidden,sen):
    return sen, sen


@app.callback([
               Output('input-sentences','data', allow_duplicate=True),
               Output('all-relation-store','data',allow_duplicate=True),
               Output('next-btn', 'n_clicks',allow_duplicate=True),
               Output('sentence', 'children',allow_duplicate=True),],
              Input('discard-btn', 'n_clicks'),
              [State('input-sentences','data'),
               State('all-relation-store','data'),
               State('next-btn', 'n_clicks'),
               State('back-btn', 'n_clicks')
               ],
              prevent_initial_call=True
)
def discard(n_clicks,sentence_storage,data,for_index,back_index):
    if len(sentence_storage) == 1:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    index = int(for_index)-int(back_index)
    if index == 0:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    if index == len(sentence_storage):
        for_index -= 1
    sentence_storage.pop(index)
    data.pop(index-1)
    if index == len(sentence_storage):
        return sentence_storage,data, for_index-1, dash.no_update
    return sentence_storage,data, dash.no_update, sentence_storage[index]

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
    State("all-relation-store","data"),
    prevent_initial_call=True
)
def show_value(n1, n2,data):
    index = int(n2)-int(n1)
    if ctx.triggered_id == 'back-btn':
        return f"Index: {index}, Total Passages: {len(data)}"
    if index == 0: # without this, get caught in case 3 EOF at 0 index
        if ctx.triggered_id == 'next-btn':
            return f"Index: {index}, Total Passages: {len(data)}"
    if len(data) == index:
        return f"Index: {index}, Total Passages: {len(data)}, EOF"
    elif len(data) < index:
        return f"Index: {index}, Total Passages: {len(data)}, Past EOF"
    return f"Index: {index}, Total Passages: {len(data)}"



# Following function is used for up-arrow and down-arrow binding to increase and decrease for the current relation
app.clientside_callback(
    """
        function(id) {
            document.addEventListener("keydown", function(event) {
                if (event.keyCode == '38' || event.keyCode == '107') {
                    document.getElementById('increase-btn').click()
                    event.stopPropogation()
                }
                if (event.shiftKey){
                    if (event.keyCode == '61') {
                        document.getElementById('increase-btn').click()
                        event.stopPropogation()
                    }
                    if (event.keyCode == '173') {
                        document.getElementById('decrease-btn').click()
                        event.stopPropogation()
                    }
                }
                if (event.keyCode == '40' || event.keyCode == '109') {
                    document.getElementById('decrease-btn').click()
                    event.stopPropogation()
                }
            });
            return window.dash_clientside.no_update       
        }
    """,
    Output("increase-btn", "id"),
    Input("increase-btn", "id"),

)

@app.callback(
    [Output("output2", "children", allow_duplicate=True)],
    Input("increase-btn", "n_clicks"),
    Input("decrease-btn", "n_clicks"),
    State("all-relation-store","data"),
    prevent_initial_call=True
)
def increase_decrease_keys(n1, n2,data): # don't know why we need an additional function and callback here,
    # but it doesn't seem to work without it
    return dash.no_update

app.clientside_callback(
    """
        function(id) {
            document.addEventListener("keydown", function(event) {
                if (event.shiftKey){
                    if (event.keyCode == '83'){
                        document.getElementById('save-btn').click()
                        event.stopPropogation()
                    }
                }
                if (event.key == 's') {
                    document.getElementById('source-btn').click()
                    event.stopPropogation()
                } 
            });
            return window.dash_clientside.no_update
        }
    """,
    Output("source-btn", "id"),
    Output("save-btn", "id"),
    Input("source-btn", "id"),
    Input("save-btn", "id"),

)

@app.callback(
    [Output("output2", "children", allow_duplicate=True)],
    Input("source-btn", "n_clicks"),
    Input("save-btn", "n_clicks"),
    State("all-relation-store","data"),
    prevent_initial_call=True
)
def source_keybind(n1, n2, data): # don't know why we need an additional function and callback here,
    # but it doesn't seem to work without it
    return dash.no_update


app.clientside_callback(
    """
        function(id) {
            document.addEventListener("keydown", function(event) {
                if (event.key == 't') {
                    document.getElementById('target-btn').click()
                    event.stopPropogation()
                }
            });
            return window.dash_clientside.no_update
        }
    """,
    Output("target-btn", "id"),
    Input("target-btn", "id"),

)

@app.callback(
    [Output("output2", "children", allow_duplicate=True)],
    Input("target-btn", "n_clicks"),
    State("all-relation-store","data"),
    prevent_initial_call=True
)
def target_keybind(n1, data): # don't know why we need an additional function and callback here,
    # but it doesn't seem to work without it
    return dash.no_update

@app.callback(
    [Output('datatable-metrics', 'data'),
     Output('datatable-metrics', 'columns'),],
    Input('llm-metrics', 'data'),
    [State('datatable-metrics', 'columns'),
     State('all-relation-store', 'data')]
)
def update_metrics(llmMetrics, cols, data):
    """
    This function is for updating the metrics table immediately after a file upload.
    As a byproduct, it also updates the dropdown menu next to the metrics table.
    :param llmMetrics:
    :param cols:
    :param backPass:
    :return:
    """
    cols = []
    row = {}
    rows = []
    i = 0
    if data is None:
        return [], []
    for llm in llmMetrics.keys():
        cols.append({'name': [f'{llm}','F1'], 'id': f"{i}", 'hideable':'first'})
        row[i] = f"{round(llmMetrics[llm]['F1'],4)}"
        i += 1

        cols.append({'name': [f'{llm}', 'Accuracy'], 'id': f"{i}"})
        row[i] = f"{round(llmMetrics[llm]['accuracy'], 4)}"
        i += 1

        cols.append({'name': [f'{llm}', 'Recall'], 'id': f"{i}"})
        row[i] = f"{round(llmMetrics[llm]['recall'], 4)}"
        i += 1
    rows.append(row)
    return rows, cols


def toggle_modal(n1, is_open):
    if n1:
        return not is_open
    return is_open

app.callback(Output("modal-xl","is_open"),
              Input("llm-output-btn","n_clicks"),
              State("modal-xl","is_open")
)(toggle_modal)


if __name__ == '__main__':
    app.run_server(debug=True)
