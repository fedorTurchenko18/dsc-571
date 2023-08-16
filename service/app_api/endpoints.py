import pandas as pd, service.app_api.schemas as schemas, numpy as np, joblib

import sys, os
sys.path.append(os.path.abspath('../..'))

from fastapi import FastAPI, HTTPException
from sklearn.preprocessing import RobustScaler

from service.app_api.features.extractor import FeatureExtractor
from service.app_api.features.final_processing import CustomColumnTransformer


app = FastAPI()

# model = joblib.load('resource/rf.joblib')
# model_features = set(model.feature_names_in_)

@app.post('/predict')
def predict(user: schemas.EkoUser, sales: schemas.EkoSales, customers: schemas.EkoCustomers):
    sales = pd.read_json(sales.sales, orient='records', dtype={'receiptdate': 'datetime64[ns]'})
    customers = pd.read_json(customers.customers, orient='records', dtype={'accreccreateddate': 'datetime64[ns]'})

    user_customers = customers[customers['ciid'] == user.ciid]
    user_sales = sales[sales['ciid'] == user.ciid]

    user_lifetime_days = (user_sales['receiptdate'].max() - user_sales['receiptdate'].min()).days
    if user_lifetime_days < 60:
        raise HTTPException(
            400, f'At least 60 days are required to make a prediction. The user {user.ciid} has been a customer for only {user_lifetime_days} days.'
        )

    fe = FeatureExtractor(target_month=3, perform_split=False, generation_type='continuous', filtering_set='sales', period=60, subperiod=15)
    X, y = fe.transform(sales=user_sales, customers=user_customers)

    qty_cols = [col for col in X.columns if 'qty' in col]
    col_transform = CustomColumnTransformer(
        cols_for_scaling=qty_cols,
        scaling_algo=RobustScaler(),
        cols_for_ohe=None,
        cols_for_winsor=None,
        cols_to_skip=None
    )
    X = col_transform.fit_transform(X, y)
    
    # X_features = set(X.columns)
    # missing_features = model_features.difference(X_features)
    # X = X.assign(**dict(zip(
    #     missing_features, [0 for i in range(len(missing_features))]
    # )))
    # X = X[model.feature_names_in_]

    # prediction_mapping = {
    #     0: 'The user is unlikely to make any purchases in the 3rd month after first purchase',
    #     1: 'The user is likely to make at least a purchase in the 3rd month after first purchase',
    # }
    # prediction = int(model.predict(X)[0])
    return {'prediction': int(np.random.randint(2, size=1)[0])}