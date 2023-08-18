import pandas as pd, numpy as np, yaml, inflect, shap, json

import sys, os
sys.path.append(os.path.abspath('../..'))

from fastapi import FastAPI, HTTPException
import service.app_api.schemas as schemas

from service.app_api.features.extractor import FeatureExtractor
from service.app_api.configs import utils

from xgboost import XGBClassifier

app = FastAPI()

with open('service/app_api/api_config.yaml', 'rb') as f:
    config = yaml.safe_load(f)

# Declare constants = default values of `FeatureExtractor` constructor arguments
TARGET_MONTH = config['target_month']
N_PURCHASES = config['n_purchases']
PERFORM_SPLIT = config['perform_split']
GENERATION_TYPE = config['generation_type']
FILTERING_SET = config['filtering_set']
PERIOD = config['period']
SUBPERIOD = config['subperiod']

# Get model from Weights&Biases
model_artifact = utils.get_artifact(config['wandb_model_project'], f"{config['wandb_model_id']}_model.json")
model_artifact.download(config['model_path'])
model = XGBClassifier()
model.load_model(config['model_path']+f"/{config['wandb_model_id']}_model.json")
model_features = model.get_booster().feature_names
# Utility for number-to-string conversion
num_convert = inflect.engine()

@app.post('/predict', response_model=schemas.PredictionFields)
def predict(
    user: schemas.EkoUser,
    sales: schemas.EkoSales,
    customers: schemas.EkoCustomers,
    request_fields: schemas.RequestFields
):
    user = user.model_dump()

    sales = sales.model_dump()
    sales_df = pd.DataFrame(
        data={
            'receiptdate': pd.Series(sales['receiptdate'], dtype='datetime64[ns]'),
            'qty': sales['qty'],
            'receiptid': sales['receiptid'],
            'proddesc': sales['proddesc'],
            'prodcategoryname': sales['prodcategoryname'],
            'station_mapping': sales['station_mapping']
        }
    )
    sales_df['ciid'] = user['ciid']

    customers = customers.model_dump()
    customers_df = pd.DataFrame(
        data={
            'ciid': user['ciid'],
            'accreccreateddate': pd.Series(customers['accreccreateddate'], dtype='datetime64[ns]'),
            'lifecyclestate': customers['lifecyclestate'],
            'cigender': customers['cigender'],
            'ciyearofbirth': customers['ciyearofbirth']
        }
    )

    user_customers = customers_df[customers_df['ciid'] == user['ciid']]
    user_sales = sales_df[sales_df['ciid'] == user['ciid']]

    user_lifetime_days = (user_sales['receiptdate'].max() - user_sales['receiptdate'].min()).days
    if user_lifetime_days < 60:
        raise HTTPException(
            400,
            f'At least 60 days are required to make a prediction. The user {user.ciid} has been a customer for only {user_lifetime_days} days.'
        )

    fe = FeatureExtractor(
        target_month=TARGET_MONTH,
        n_purchases=N_PURCHASES,
        perform_split=PERFORM_SPLIT,
        generation_type=GENERATION_TYPE,
        filtering_set=FILTERING_SET,
        period=PERIOD,
        subperiod=SUBPERIOD
    )
    X, y = fe.transform(sales=user_sales, customers=user_customers)
    if X is None and y is None:
        # Both `X` and `y` dataframes are returned as None
        # It means that filtering removed all rows
        raise HTTPException(
            400,
            f"Unfortunately, this customer is outdated. Model was trained on all customers who registered at {config['customers_set_filtering_thresholds']} or later"
        )
    X_features = set(X.columns)
    missing_features = set(model_features).difference(X_features)
    X = X.assign(
        **dict(
            zip(
                missing_features,
                [0 for i in range(len(missing_features))]
            )
        )
    )
    X = X[model_features]


    prediction_mapping = {
        0: f'The user is unlikely to make any purchases in the {num_convert.ordinal(TARGET_MONTH)} month after first purchase',
        1: f'The user is likely to make at least {N_PURCHASES} purchase(s) in the {num_convert.ordinal(TARGET_MONTH)} month after first purchase',
    }
    prediction = int(model.predict(X)[0])
    output = schemas.PredictionFields(prediction=prediction_mapping[prediction])

    request_fields = request_fields.model_dump()

    if 'confidence' in request_fields['fields']:
        probas = model.predict_proba(X)
        confidence = probas[0][0] if prediction == 0 else probas[0][1]
        output.confidence = confidence.astype(np.float64)

    if 'shapley_values' in request_fields['fields']:
        explainer = shap.TreeExplainer(model)
        shapley_output = {}
        shapley_output['shapley_values'] = explainer.shap_values(X, y).tolist()
        shapley_output['X'] = X.to_dict(orient='list')
        shapley_output['y'] = {'target': y.tolist()}
        shapley_output['shapley_expected_value'] = explainer.expected_value.astype(np.float64)
        output.shapley_values = shapley_output

    return output