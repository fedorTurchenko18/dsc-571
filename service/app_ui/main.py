import streamsync as ss, pandas as pd, requests

from datetime import datetime

from io import BytesIO


def parse_file_input(state):
    try:
        excel_file = BytesIO(state['users_xlsx'][0]['data'])

        sales = pd.read_excel(excel_file, sheet_name='smile_sales')
        customers = pd.read_excel(excel_file, sheet_name='smile_customers')
        users = customers['ciid'].to_list()

        state['ciid'] = dict(zip(users, users))
        state['sales'] = sales.to_json(orient='records', date_format='iso')
        state['customers'] = customers.to_json(orient='records', date_format='iso')
        state['file_upload_message']['text'] = '+Data successfully processed! Choose the user for prediction'
    except:
        state['users_xlsx'] = None
        state['ciid'] = {'': 'Error generating the list of users'}
        state['file_upload_message']['text'] = '-An error occured! Make sure you have uploaded the correct spreadsheet'

def selected_user(state, payload):
    state['selected_ciid'] = payload

def predict(state):
    r = requests.post(
        'http://0.0.0.0:8000/predict',
        json={
            'user': {'ciid': state['selected_ciid']},
            'sales': {'sales': state['sales']},
            'customers': {'customers': state['customers']}
        }
    )
    state['prediction'] = r.json()['prediction']
    state['results']['visible'] = True
    state['results']['prediction_text'] = f"Predicted value is: {state['prediction']}"

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
        'visible': False,
        'prediction_text': ''
    }
})

print(f'Started at {str(datetime.now())}')