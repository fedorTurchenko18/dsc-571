import pandas as pd, base64, requests, warnings, numpy as np, shap
warnings.filterwarnings('ignore')
from io import BytesIO
from loguru import logger
from requests.exceptions import ConnectionError

# import sys, os
# sys.path.append(os.path.abspath('../..'))
from shap_plots.waterfall import waterfall

import matplotlib.pyplot
matplotlib.pyplot.switch_backend('Agg') 

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, html, dcc


def _make_prediction_request(sales, customers, ciid):
    sales = pd.DataFrame(sales)
    customers = pd.DataFrame(customers)

    sales_cut = sales[sales['ciid']==ciid]
    sales_cut.drop('ciid', axis=1, inplace=True)
    sales_cut['receiptdate'] = sales_cut['receiptdate'].astype(str)
    sales_cut = sales_cut.to_dict(orient='list')

    customers_cut = customers[customers['ciid']==ciid]
    customers_cut[['accreccreateddate', 'ciyearofbirth']] = customers_cut[['accreccreateddate', 'ciyearofbirth']].astype(str)
    customers_cut.drop('ciid', axis=1, inplace=True)
    customers_cut.dropna(axis=1, inplace=True)
    customers_cut = customers_cut.to_dict(orient='list')

    request_pred_json = {
        'user': {'ciid': ciid},
        'sales': sales_cut,
        'customers': customers_cut,
        'request_fields': {'fields': ['prediction', 'confidence', 'shapley_values']}
    }
    request_clust_json = {
        'user': {'ciid': ciid},
        'sales': sales_cut,
        'customers': customers_cut
    }
    logger.info(request_pred_json)
    try:
        request_pred = requests.post(
            'http://0.0.0.0:8000/predict',
            json=request_pred_json
        )
        request_clust = requests.post(
            'http://0.0.0.0:8000/predict_cluster',
            json=request_pred_json
        )
    except ConnectionError:
        request_pred = requests.post(
            'http://app-api:8000/predict',
            json=request_pred_json
        )
        request_clust = requests.post(
            'http://app-api:8000/predict_cluster',
            json=request_pred_json
        )
    response = {
        'prediction': request_pred.json(),
        'clustering': request_clust.json()
    }
    logger.info(response)
    return response

def _process_prediction_response(response, ciid):
    if 'detail' in response.keys():
        return {
            'form-submit-button': ciid,
            'prediction-decision': response['detail'],
            'prediction-confidence': '',
            'waterfall-chart': None,
            'loading-form-submit': None,
        }
    else:
        prediction_mapping = {
            0: ['The user is ', html.B("unlikely"), ' to make ', html.B("any"), f' purchases in the {response["target_month"]} month after first purchase'],
            1: ['The user is ', html.B("likely"), ' to make at least ', html.B(response["n_purchases"]), f' purchase(s) in the {response["target_month"]} month after first purchase'],
        }
        waterfall_chart = _waterfall_plot(response)
        return {
            'form-submit-button': ciid,
            'prediction-decision': prediction_mapping[response['prediction']],
            'prediction-confidence': f"Prediction confidence: {response['confidence'][0] * 100 :.2f}%" if response['prediction'] == 0 else f"Prediction confidence: {response['confidence'][1] * 100 :.2f}%",
            'waterfall-chart': waterfall_chart,
            'loading-form-submit': None,
        }

def _process_clustering_response(response):
    if 'detail' in response.keys():
        return {
            'clustering-decision': None,
            'clustering-sim-table': pd.DataFrame()
        }
    else:
        cluster = response['cluster']
        predicted_cluster = response['label'].replace('_', ' ').title()

        similarities = response['similarities'][0]
        clusters_mapping = response['clusters_mapping']
        
        # drop predicted cluster
        similarities = [i for i in similarities if similarities.index(i) != cluster]
        clusters_mapping = {str(i): clusters_mapping[i] for i in clusters_mapping.keys() if i != str(cluster)}

        df_sim = pd.DataFrame({
            'Segment': list(clusters_mapping.values()),
            'Affiliation Likelihood': similarities
        })
        df_sim.sort_values('Affiliation Likelihood', ascending=False, inplace=True)
        df_sim['Segment'] = df_sim['Segment'].apply(lambda x: x.replace('_', ' ').title())
        df_sim['Affiliation Likelihood'] = df_sim['Affiliation Likelihood'].apply(lambda x: f'{x * 100: .2f}%')

        return {
            'clustering-decision': [
                'The user is predicted to be in a segment of ',
                html.B(predicted_cluster),
                ' according to ',
                html.Span(
                    html.U('RFM'),
                    id='rfm-tooltip',
                    style={
                        'cursor': 'pointer'
                    }
                ),
                ' segmentation'
            ],
            'clustering-sim-table': df_sim
        }

def _waterfall_plot(response, max_display=10):
    shapley_values = np.array(response['shapley_values']['shapley_values'])
    X = pd.DataFrame(response['shapley_values']['X'])
    y = pd.DataFrame(response['shapley_values']['y'])
    ev = response['shapley_values']['shapley_expected_value']
    pred_proba = response['confidence']

    shapley_values_explainer = shap.Explanation(
        values=shapley_values,
        data=X.values,
        base_values=np.array([ev]*X.shape[0], dtype=np.float32),
        feature_names=X.columns
    )
    fig = waterfall(shapley_values_explainer[0], link='logit', predicted_probability=pred_proba[1], show=False, max_display=max_display)
    fig = fig.update_layout(
        width=700,
        height=500,
        legend={
            'yanchor': 'bottom',
            'xanchor': 'right'
        },
        xaxis = {
            'tickmode': 'linear',
            'tick0': 0.0,
            'dtick': 0.05
        }
    )
    return fig

@callback(
    Output('file-upload-error-alert', 'children'),
    Output('file-input-store', 'data'),
    Output('loading-output-content', 'children'),
    Output('form-submit-button', 'disabled'),
    Output('file-name', 'children'),
    Output('dropdown-selection', 'options'),
    Output('dropdown-selection', 'value'),
    Output('form-submit-button', 'n_clicks'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
)
def process_file_input(contents, filename):
    if contents is None:
        return (
            None,
            {},
            None,
            True,
            'Drag and Drop or Select Files',
            ['Waiting for the spreadsheet...'],
            'Waiting for the spreadsheet...',
            None
        )
    else:
        decoded_contents = base64.b64decode(contents.split(',')[1])
        df = pd.ExcelFile(BytesIO(decoded_contents))
        sheet_names = df.sheet_names
        sheet_names.sort()
        if sheet_names != ['smile_customers', 'smile_sales']:
            return(
                dbc.Alert(
                    ['Sheet names are incorrect. Make sure the Excel file contains 2 sheets, ', html.B("smile_customers"),' and ', html.B("smile_sales")],
                    color='danger'
                ),
                {},
                None,
                True,
                'Drag and Drop or Select Files',
                ['Waiting for the spreadsheet...'],
                'Waiting for the spreadsheet...',
                None
            )

        customers = df.parse(sheet_name='smile_customers')
        users = customers['ciid'].to_list()
        customers['accreccreateddate'] = customers['accreccreateddate'].astype(str)
        customers = customers.to_dict(orient='list')

        sales = df.parse(sheet_name='smile_sales')
        sales['receiptdate'] = sales['receiptdate'].astype(str)
        sales = sales.to_dict(orient='list')

        return (
            dbc.Alert(
                    'Success! You can now select the user and click the button below to get the prediction:',
                    color='success'
                ),
            {'sales': sales, 'customers': customers},
            None,
            False,
            filename,
            users,
            users[0],
            None
        )

@callback(
    Output('card-div', 'children'),
    Output('waterfall-chart-div', 'children'),
    Output('waterfall-chart-slider-div', 'children'),
    Output('clusters-desc-div', 'children'),
    Output('loading-form-submit', 'children'),
    Output('form-submit-button', 'value'),
    Output('api-output-store', 'data'),
    Input('form-submit-button', 'n_clicks'),
    Input('file-input-store', 'data'),
    State('dropdown-selection', 'value'),
    State('form-submit-button', 'value'),
    State('api-output-store', 'data')
)
def get_prediction(click, data, ciid, ciid_value, current_response):
    if click is None:
        return None, None, None, None, None, '', {}
    else:
        if ciid == ciid_value:
            parsed_pred_response = _process_prediction_response(current_response['prediction'], ciid)
            parsed_clust_response = _process_clustering_response(current_response['clustering'])
        else:
            response = _make_prediction_request(data['sales'], data['customers'], ciid)
            parsed_pred_response = _process_prediction_response(response['prediction'], ciid)
            parsed_clust_response = _process_clustering_response(response['clustering'])
        
        if 'outdated' not in parsed_pred_response['prediction-decision']:
            clustering_layout = [
                html.H4('Clustering Result'),
                html.P(
                    parsed_clust_response['clustering-decision'],
                    id='clustering-decision'
                ),
                html.P('Affiliation likelihood with 3 other segments:'),
                dbc.Tooltip(
                    [
                        'R – Recency', html.Br(),
                        'F – Frequency', html.Br(),
                        'M – Monetary'
                    ],
                    target='rfm-tooltip'
                ),
                dbc.Table.from_dataframe(
                    parsed_clust_response['clustering-sim-table'],
                    striped=True,
                    bordered=True,
                    hover=True,
                    id='clustering-sim-table'
                )
            ]
            clusters_desc_layout = dbc.Accordion(
                dbc.AccordionItem(
                    [
                        html.P(
                            [
                                html.P(
                                    [
                                        html.B('Regular Drivers: '),
                                        html.Span('The customers in this cluster are quite loyal but, perhaps, do not drive much, hence they do not need to visit gas stations often and spend a lot')
                                    ]
                                ),
                                html.P(
                                    [
                                        html.B('Passerbys: '),
                                        html.Span('The customers in this cluster made one-two visits and most likely have left')
                                    ]
                                ),
                                html.P(
                                    [
                                        html.B('Frequent Drivers: '),
                                        html.Span('The customers in this cluster frequently visit gas stations, spending a lot. Perhaps, these are the most loyal customers')
                                    ]
                                ),
                                html.P(
                                    [
                                        html.B('At Churn Risk: '),
                                        html.Span('The customers in this cluster visit gas stations from time to time. They do not spend much, do not make their visits often, hence could be considered to be at risk of churn')
                                    ]
                                ),
                            ]
                        ),
                    ],
                    title='Click to find more about the clusters'
                ),
                start_collapsed=True
            )
        else:
            clustering_layout, clusters_desc_layout = [], None

        return (
            dbc.CardBody(
                [
                    html.H4('Prediction Result'),
                    html.P(
                        parsed_pred_response['prediction-decision'],
                        id='prediction-decision'
                    ),
                    html.P(
                        parsed_pred_response['prediction-confidence'],
                        id='prediction-confidence'
                    ),
                    
                ] + clustering_layout
            ),
            dcc.Graph(
                figure=parsed_pred_response['waterfall-chart'],
                id='waterfall-chart'
            ) if parsed_pred_response['waterfall-chart'] is not None else None,
            html.Div(
                [
                    dbc.Label(
                        "Select the number of predictors to display on the y-axis:",
                        html_for="waterfall-chart-slider",
                        style={
                            'margin-top': '20px',
                            'margin-left': '19px'
                        }
                    ),
                    dcc.Slider(
                        min=1, 
                        max=20,
                        value=10,
                        step=1,
                        marks=dict(zip([1]+[x for x in range(1, 21) if x % 5 == 0], ['1']+[str(x) for x in range(1, 21) if x % 5 == 0])),
                        id='waterfall-chart-slider'
                    ),
                    dbc.FormText(
                        [
                            'Now displaying ',
                            html.Span('10', id='waterfall-chart-slider-current-value'),
                            ' features'
                        ],
                        style={
                            'margin-left': '19px'
                        }
                    ),
                ]
            ) if parsed_pred_response['waterfall-chart'] is not None else None,
            clusters_desc_layout,
            parsed_pred_response['loading-form-submit'],
            parsed_pred_response['form-submit-button'],
            response if ciid != ciid_value else current_response
        )

@callback(
    Output('waterfall-chart', 'figure'),
    Output('waterfall-chart-slider-current-value', 'children'),
    Input('waterfall-chart-slider', 'value'),
    State('waterfall-chart-div', 'children'),
    State('api-output-store', 'data')
)
def waterfall_y_axis(n_features, chart, response):
    if chart != []:
        fig = _waterfall_plot(response['prediction'], max_display=n_features)
        return fig, str(n_features)