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

    request_json = {
            'user': {'ciid': ciid},
            'sales': sales_cut,
            'customers': customers_cut,
            'request_fields': {'fields': ['prediction', 'confidence', 'shapley_values']}
        }
    logger.info(request_json)
    try:
        request = requests.post(
            'http://0.0.0.0:8000/predict',
            json=request_json
        )
    except ConnectionError:
        request = requests.post(
            'http://app-api:8000/predict',
            json=request_json
        )
    response = request.json()
    return response

def _process_prediction_request(response, ciid):
    if 'detail' in response.keys():
        return {
            'form-submit-button': ciid,
            'prediction-decision': response['detail'],
            'prediction-confidence': '',
            'waterfall-chart': None,
            'loading-form-submit': None,
            'api-output-store': response
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
            'api-output-store': response
        }

def _waterfall_plot(response):
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
    fig = waterfall(shapley_values_explainer[0], link='logit', predicted_probability=pred_proba[1], show=False)
    fig = fig.update_layout(
        width=700,
        height=500,
        legend={
            'yanchor': 'bottom',
            'xanchor': 'right'
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
        return None, None, None, '', {}
    else:
        if ciid == ciid_value:
            parsed_response = _process_prediction_request(current_response, ciid)
        else:
            response = _make_prediction_request(data['sales'], data['customers'], ciid)
            parsed_response = _process_prediction_request(response, ciid)
        return (
            dbc.CardBody(
                [
                    html.H4('Prediction Result'),
                    html.P(
                        parsed_response['prediction-decision'],
                        id='prediction-decision'
                    ),
                    html.P(
                        parsed_response['prediction-confidence'],
                        id='prediction-confidence'
                    )
                ]
            ),
            dcc.Graph(
                figure=parsed_response['waterfall-chart'],
                id='waterfall-chart'
            ) if parsed_response['waterfall-chart'] is not None else None,
            parsed_response['loading-form-submit'],
            parsed_response['form-submit-button'],
            parsed_response['api-output-store']
        )