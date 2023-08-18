import streamsync as ss, pandas as pd, requests, numpy as np, shap

from bs4 import BeautifulSoup as bs
from loguru import logger
from dash import html
from datetime import datetime
from io import BytesIO


def parse_file_input(state):
    try:
        excel_file = BytesIO(state['users_xlsx'][0]['data'])

        sales = pd.read_excel(excel_file, sheet_name='smile_sales')
        customers = pd.read_excel(excel_file, sheet_name='smile_customers')
        users = customers['ciid'].to_list()

        state['ciid'] = dict(zip(users, users))
        state['sales'] = sales
        state['customers'] = customers

        state['file_upload_message']['text'] = '+Data successfully processed! Choose the user for prediction'
    except:
        state['users_xlsx'] = None
        state['ciid'] = {'': 'Error generating the list of users'}
        state['file_upload_message']['text'] = '-An error occured! Make sure you have uploaded the correct spreadsheet'

def selected_user(state, payload):
    state['selected_ciid'] = payload

def predict(state):
    sales_cut = pd.DataFrame(state['sales'])
    sales_cut = sales_cut[sales_cut['ciid']==state['selected_ciid']]
    sales_cut.drop('ciid', axis=1, inplace=True)
    sales_cut['receiptdate'] = sales_cut['receiptdate'].astype(str)
    sales_cut = sales_cut.to_dict(orient='list')

    customers_cut = pd.DataFrame(state['customers'])
    customers_cut = customers_cut[customers_cut['ciid']==state['selected_ciid']]
    customers_cut.drop('ciid', axis=1, inplace=True)
    customers_cut.dropna(axis=1, inplace=True)
    customers_cut['accreccreateddate'] = customers_cut['accreccreateddate'].astype(str)
    customers_cut = customers_cut.to_dict(orient='list')
    request = requests.post(
        'http://0.0.0.0:8000/predict',
        json={
            'user': {'ciid': state['selected_ciid']},
            'sales': sales_cut,
            'customers': customers_cut,
            'request_fields': {'fields': ['prediction', 'confidence', 'shapley_values']}
        }
    )
    response = request.json()

    state['prediction'] = response['prediction']
    state['results']['visible'] = True
    state['results']['prediction_text'] = f"Predicted value is: {state['prediction']}"

    shapley_values = np.array(response['shapley_values']['shapley_values'])
    X = pd.DataFrame(response['shapley_values']['X'])
    y = pd.DataFrame(response['shapley_values']['y'])
    ev = response['shapley_values']['shapley_expected_value']

    force_plot = shap.force_plot(ev, shapley_values, X, link='logit').html()
    force_plot_func = bs(force_plot).select_one('script').get_text()
    # with open('app_ui/static/force_plot.js', 'w') as f:
    #     f.write(force_plot_func)

    state.import_frontend_module('general_shap', 'app_ui/static/general_shap.js')
    a = state.call_frontend_function('general_shap')
    print(a)
    state.import_frontend_module('force_plot', 'app_ui/static/force_plot.js')
    b = state.call_frontend_function('force_plot')
    print(b)



initial_state = ss.init_state({
    "my_app": {
        "title": "Churn Prediction App"
    },
    "users_xlsx": None,
    'ciid': {'': 'The user list is not composed yet. Waiting for the input to be processed'},
    'selected_ciid': None,
    'sales': None,
    'customers': None,
    'file_upload_message': {
        'text': 'Upload the spreadsheet to begin. Once you upload the file, it will take some time to process. You will be notified by the success/error message',
        'visible': True
    },
    'prediction': None,
    'results': {
        'visible': True,
        'prediction_text': '',
        'force_plot': None
    }
})

print(f'Started at {str(datetime.now())}')