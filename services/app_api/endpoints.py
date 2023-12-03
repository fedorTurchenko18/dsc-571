import pandas as pd, numpy as np, yaml, inflect, shap, json, logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import services.app_api.schemas as schemas

from services.app_api.features.extractor import FeatureExtractor
from services.app_api.configs import utils

from xgboost import XGBClassifier
# Clustering dependencies
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

app = FastAPI()

with open('services/app_api/api_config.yaml', 'rb') as f:
    config = yaml.safe_load(f)

# Declare constants = default values of `FeatureExtractor` constructor arguments
TARGET_MONTH = config['target_month']
N_PURCHASES = config['n_purchases']
PERFORM_SPLIT = config['perform_split']
GENERATION_TYPE = config['generation_type']
FILTERING_SET = config['filtering_set']
PERIOD = config['period']
SUBPERIOD = config['subperiod']

# Get classification model from Weights&Biases
utils.login_wandb()
classification_model_artifact = utils.get_artifact(config['wandb_classification_model_project'], f"{config['wandb_classification_model_id']}_model.json")
classification_model_artifact.download(config['model_path'])
classification_model = XGBClassifier()
classification_model.load_model(config['model_path']+f"/{config['wandb_classification_model_id']}_model.json")
model_features = classification_model.get_booster().feature_names

# Get clustering model from Weights&Biases
clustering_model_artifact = utils.get_artifact(config['wandb_clustering_model_project'], f"{config['wandb_clustering_model_id']}")
clustering_model_artifact.download(config['model_path'])
with open('services/app_api/configs/centroids_table.table.json', 'r') as f:
    centroids_table = json.load(f)
centroids = centroids_table['data']
with open('services/app_api/configs/monetary_winsorization_threshold.table.json', 'r') as f:
    monetary_threshold_table = json.load(f)
monetary_threshold = monetary_threshold_table['data'][0][0]
clusters_mapping = config['clusters_mapping']

# Utility for number-to-string conversion
num_convert = inflect.engine()


# Extended logging for 422 Unprocessable Entity
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
	logging.error(f'{request}: {exc_str}')
	content = {'status_code': 10422, 'message': exc_str, 'data': None}
	return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


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

    prediction = int(classification_model.predict(X)[0])
    output = schemas.PredictionFields(prediction=prediction, target_month=TARGET_MONTH, n_purchases=N_PURCHASES)

    request_fields = request_fields.model_dump()

    if 'confidence' in request_fields['fields']:
        probas = classification_model.predict_proba(X)
        confidence = probas[0]
        output.confidence = confidence.astype(np.float64).tolist()

    if 'shapley_values' in request_fields['fields']:
        explainer = shap.TreeExplainer(classification_model)
        shapley_output = {}
        shapley_output['shapley_values'] = explainer.shap_values(X, y).tolist()
        shapley_output['X'] = X.to_dict(orient='list')
        shapley_output['y'] = {'target': y.tolist()}
        shapley_output['shapley_expected_value'] = explainer.expected_value.astype(np.float64)
        output.shapley_values = shapley_output

    return output


@app.post('/predict_cluster', response_model=schemas.ClusterPredictionFields)
def predict_cluster(
    user: schemas.EkoUser,
    sales: schemas.EkoSales,
    customers: schemas.EkoCustomers
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
    fe = FeatureExtractor(generation_type='continuous', filtering_set='customers', period=60, subperiod=60, perform_split=False)
    X, _ = fe.transform(sales_df, customers_df)
    if X is None:
        raise HTTPException(
            400,
            f"Unfortunately, this customer is outdated. Model was trained on all customers who registered at {config['customers_set_filtering_thresholds']} or later"
        )
    X.columns = [f'{col[:col.find("_1-60")]}' for col in X.columns]
    X = X[['monetary', 'recency', 'average_days_between_visits']]
    # outliers winsorization
    X.loc[X['monetary'] > monetary_threshold, 'monetary'] = monetary_threshold
    # Assure proper scaling (ref: https://stackoverflow.com/questions/46555820/sklearn-standardscaler-returns-all-zeros)
    X = X.to_numpy()[0]
    X = StandardScaler().fit_transform(X[:, np.newaxis])
    # Reshape: 1 col 4 rows => 4 cols 1 row
    # to assure proper distance computation
    X = X.reshape(1, -1)
    euclidean_distances = cdist(X, centroids, 'euclidean')
    similarities = np.exp(-euclidean_distances).astype(np.float64)
    total_score = similarities.sum()
    # convert smoothed (exponentiated) Euclidean distances to percentages
    similarities = similarities/total_score
    # similarities object is a numpy array of shape (1, 4)
    cluster = similarities[0].tolist().index(similarities.max())
    label = clusters_mapping[cluster]
    output = schemas.ClusterPredictionFields(cluster=cluster, label=label, similarities=similarities, clusters_mapping=clusters_mapping)
    return output