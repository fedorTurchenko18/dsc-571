import yaml

from datetime import datetime, date
from pydantic import BaseModel, validator, conint, confloat
from typing import List, Optional, Union, Literal, Dict

with open('service/app_api/api_config.yaml', 'rb') as f:
    config = yaml.safe_load(f)

clusters = list(config['clusters_mapping'].keys())
FIRST_CLUSTER = min(clusters)
LAST_CLUSTER = max(clusters)

MIN_CUSTOMER_REGISTRATION_DATE = config['customers_set_filtering_thresholds']


class EkoUser(BaseModel):
    ciid: str


class EkoSales(BaseModel):
    receiptdate: List[
        confloat(
            ge=datetime.timestamp(
                datetime.strptime(MIN_CUSTOMER_REGISTRATION_DATE, '%Y-%m-%d %H:%M:%S')
            )
        )
    ]
    qty: List[confloat(allow_inf_nan=False)]
    receiptid: List[str]
    proddesc: List[str]
    prodcategoryname: List[str]
    station_mapping: Optional[List[int]] = None


class EkoCustomers(BaseModel):
    accreccreateddate: List[
        confloat(
            ge=datetime.timestamp(
                datetime.strptime(MIN_CUSTOMER_REGISTRATION_DATE, '%Y-%m-%d %H:%M:%S')
            )
        )
    ]
    # accreccreateddate: List[date]
    lifecyclestate: Optional[List[str]] = None
    cigender: Optional[List[str]] = None
    ciyearofbirth: Optional[List[str]] = None


class RequestFields(BaseModel):
    fields: Union[Literal['prediction'], List[Literal['prediction', 'confidence', 'shapley_values']]]

    @validator('fields', pre=True, always=True)
    def validate_custom_field(cls, value):
        if isinstance(value, str):
            if value != 'prediction':
                raise ValueError("Only 'prediction' could be passed as a string, not list")
            return [value]
        elif isinstance(value, list):
            allowed_values = ['prediction', 'shapley_values', 'confidence']
            if 'prediction' not in value:
                raise ValueError("'prediction' must be present in the list of fields")
            for v in value:
                if v not in allowed_values:
                    raise ValueError(f"Invalid value '{v}' in list")
            return value
        else:
            raise ValueError("Invalid type for 'custom_field'")


class PredictionFields(BaseModel):
    # predicted class could be either 0 or 1
    prediction: conint(ge=0, le=1)
    # prediction could be made for the second month of activity at minimum
    # first month of activity is always used as training data
    target_month: conint(ge=2)
    # target variable is constructed from at least one purchase made at `target_month`
    n_purchases: conint(ge=1)
    # confidence is a probability, limited to the range of [0; 1]
    confidence: Optional[List[confloat(gt=0.0, le=1.0)]] = None
    shapley_values: Optional[
        Dict[
            # key
            str,
            # possible values
            Union[
                # 2D numpy array converted to list (np.array.tolist())
                # represents shapley values
                List[List[float]],
                # pandas dataframe converted to dictionary (`pd.DataFrame.to_dict(orient='list')`)
                # represents features set after all transformations
                Dict[
                    # key
                    str,
                    # values
                    # List[List[float]]
                    List[float]
                ],
                # a single value
                # represents shapley expected value
                float
            ]
        ]
    ] = None


class EkoRFM(BaseModel):
    # pandas dataframe converted to dictionary (`pd.DataFrame.to_dict(orient='list')`)
    # represents rfm features dataframe
    rfm_features: Dict[
        # key
        str,
        # values
        List[float]
    ]


class ClusterPredictionFields(BaseModel):
    cluster: conint(ge=FIRST_CLUSTER, le=LAST_CLUSTER)
    label: str
    similarities: List[List[float]]
    clusters_mapping: Dict[conint(ge=FIRST_CLUSTER, le=LAST_CLUSTER), str]
