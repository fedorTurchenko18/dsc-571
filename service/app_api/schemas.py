from pydantic import BaseModel, validator
from datetime import datetime
from typing import List, Optional, Union, Literal, Dict


class EkoUser(BaseModel):
    ciid: str


class EkoSales(BaseModel):
    receiptdate: List[datetime]
    qty: List[float]
    receiptid: List[str]
    proddesc: List[str]
    prodcategoryname: List[str]
    station_mapping: Optional[List[int]] = None


class EkoCustomers(BaseModel):
    accreccreateddate: List[datetime]
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
    prediction: int
    target_month: int
    n_purchases: int
    confidence: Optional[List[float]] = None
    # shapley_values: Optional[Union[List[List[float]], Any]] = None
    shapley_values: Optional[
        Dict[
            # key
            str,
            # possible values
            Union[
                # 2D numpy array converted to list (np.array.tolist())
                List[List[float]],
                # pandas dataframe converted to dictionary (`pd.DataFrame.to_dict(orient='list')`)
                Dict[
                    # key
                    str,
                    # values
                    List[List[float]]
                ],
                # a single value
                float
            ]
        ]
    ] = None
