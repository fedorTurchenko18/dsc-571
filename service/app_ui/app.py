import dash_bootstrap_components as dbc, dash, plotly
from dash import html, dcc, Input, Output, callback
from utils import *
from sidebar import *

app = app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
app.config['suppress_callback_exceptions'] = True
server = app.server

input_form = dbc.Form([
    dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label(
                        "Excel Upload",
                        html_for="upload-data"
                    ),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div(
                            [
                                'Drag and Drop or Select Files'
                            ],
                            id='file-name'
                        ),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'groove',
                            'borderRadius': '5px',
                            'textAlign': 'center'
                        },
                        accept='.xlsx'
                    ),
                    dbc.FormText(
                        [
                            'Upload the spreadsheet with ',
                            html.Span(
                                "smile_sales",
                                id='sales-tooltip',
                                style={
                                    "textDecoration": "underline",
                                    "cursor": "pointer"
                                }
                            ),
                            ' and ',
                            html.Span(
                                "smile_customers",
                                id='customers-tooltip',
                                style={
                                    "textDecoration": "underline",
                                    "cursor": "pointer"
                                }
                            ),
                            ' sheets'
                        ]
                    ),
                    dbc.Tooltip(
                        html.P(
                            [
                                'Columns:', html.Br(),
                                '- ciid', html.Br(),
                                '- receiptdate', html.Br(),
                                '- qty', html.Br(),
                                '- receiptid', html.Br(),
                                '- proddesc', html.Br(),
                                '- prodcategoryname', html.Br(),
                                '- station_mapping (optional)'
                            ]
                        ),
                        target='sales-tooltip'
                    ),
                    dbc.Tooltip(
                        html.P(
                            [
                                'Columns:', html.Br(),
                                '- ciid', html.Br(),
                                '- accreccreateddate', html.Br(),
                                '- lifecyclestate (optional)', html.Br(),
                                '- cigender (optional)', html.Br(),
                                '- ciyearofbirth (optional)'
                            ]
                        ),
                        target='customers-tooltip'
                    ),
                    html.Div(id='file-upload-error-alert'),
                    dbc.Spinner(
                        html.Div(
                            id="loading-output-content",
                            style={
                                'margin-top': '20px'
                            }
                        ),
                        show_initially=False
                    )
                ],
                width=6,
            ),
            dbc.Col(
                [
                    dbc.Label(
                        "User Selection",
                        html_for="dropdown-selection"
                    ),
                    dcc.Dropdown(
                        ['Waiting for the spreadsheet...'],
                        'Waiting for the spreadsheet...',
                        id='dropdown-selection',
                        className="dash-bootstrap"
                    ),
                    dbc.FormText('Select the user for prediction'),
                    dcc.Store(id='file-input-store'),
                ],
                width=6,
            ),
        ]
    ),
    dbc.Row(
        dbc.Col([
            dbc.Button(
                'Get Prediction',
                type='submit',
                id='form-submit-button',
                disabled=True,
                value=''
            ),
            dbc.Spinner(
                html.Div(
                    id="loading-form-submit"
                ),
                show_initially=False
            )
        ], width=6)
    ),
])

api_output = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Store(id='api-output-store'),
                        html.Div(id='card-div')
                    ]
                ),
                dbc.Col(
                    [
                        html.Div(id='waterfall-chart-slider-div'),
                        html.Div(id='waterfall-chart-div')
                    ]
                )
            ]
        ),
        html.Div(id='clusters-desc-div')
    ],
    style={
        'margin-top': '20px'
    }
)
    
app.layout = dbc.Container(
    [
        dcc.Location(id="url"),
        sidebar,
        html.Div(id='page-content')
    ],
    style={
        "margin-left": "18rem"
    }
)

@callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def render_page_content(pathname):
    if pathname == "/prediction-app":
        return [
            html.H1(
                "Churn Prediction App",
                style={
                    'margin-top': '10px'
                }
            ),
            html.Hr(
                style={
                    "borderWidth": "0.5vh",
                    "width": "100%",
                    "borderColor": "#000000",
                    "opacity": "unset",
                }
            ),
            input_form,
            api_output
        ]
    elif pathname == "/dashboard":
        monthly_visits_line_chart = plotly.io.read_json('static/monthly_visits.json')
        funnel_chart = plotly.io.read_json('static/funnel.json')
        max_breaks_chart = plotly.io.read_json('static/max_breaks.json')
        max_breaks__horizontal_chart = plotly.io.read_json('static/max_breaks_horizontal.json')

        funnel_chart.update_layout(
            height=600
        )

        return [
            html.H1(
                "Historical Data Dashboard",
                style={
                    'margin-top': '10px'
                }
            ),
            html.Hr(
                style={
                    "borderWidth": "0.5vh",
                    "width": "100%",
                    "borderColor": "#000000",
                    "opacity": "unset",
                }
            ),
            dbc.Card(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dcc.Graph(
                                        figure=monthly_visits_line_chart,
                                        id='monthly-visits-line-chart'
                                    )
                                ]
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            # dbc.Col(
                            #     [
                            #         dcc.Graph(
                            #             figure=max_breaks_chart,
                            #             id='max-breaks-chart'
                            #         )
                            #     ]
                            # )
                            dbc.Col(
                                [
                                    dcc.Graph(
                                        figure=max_breaks__horizontal_chart,
                                        id='max-breaks-horizontal-chart'
                                    )
                                ]
                            )
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dcc.Graph(
                                        figure=funnel_chart,
                                        id='funnel-chart'
                                    )
                                ]
                            )
                        ]
                    ),
                ]
            )
        ]
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

if __name__ == '__main__':
    app.run(debug=True)
    # app.run(debug=False,dev_tools_ui=False,dev_tools_props_check=False)