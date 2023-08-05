from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from scipy.stats.mstats import winsorize
from typing import List, Optional, Literal, Union
import pandas as pd

class CustomColumnTransformer(BaseEstimator, TransformerMixin):
    '''
    Class which implements final data transformations like scaling, categorical encoding, etc.
    Inherits from `sklearn` abstractions, i.e. `sklearn.pipeline.Pipeline` compatible
    '''
    def __init__(
            self,
            cols_for_scaling: Union[List[str], None],
            cols_for_ohe: Union[List[str], None],
            scaling_algo: Literal['sklearn-scaling-algorithm'],
            cols_for_winsor: Union[List[str], None],
            cols_to_skip: Union[List[str], None]
    ):
        '''
        `cols_for_scaling` - numeric columns to pass to a scaling algorithm \n

        `cols_for_ohe` - categorical columns to pass to a one-hot encoding algorithm \n

        `scaling_algo` - scaling algorithm from `sklearn.preprocessing` module \n

        `cols_for_scaling` - numeric columns to winsorize for outliers handling \n

        `cols_to_skip` - columns for which no transformation is needed
        '''
        self.cols_for_scaling = cols_for_scaling
        self.cols_for_ohe = cols_for_ohe
        self.scaling_algo = scaling_algo
        self.cols_for_winsor = cols_for_winsor
        self.cols_to_skip = cols_to_skip


    def fit(self, X, y=None):
        '''
        Dummy method to replicate `sklearn`'s "fit-transform" mechanics
        '''
        return self


    def transform(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None):
        '''
        Method to apply all necessary transformations
        '''
        dataframes = []

        if self.cols_for_winsor:
            for col in self.cols_for_winsor:
                X[col] = winsorize(X[col], limits=(0.0, 0.01))

        if self.cols_for_ohe:
            ohe = OneHotEncoder(handle_unknown='ignore', min_frequency=0.001)
            dataframes.append(
                pd.DataFrame(
                    ohe.fit_transform(X[self.cols_for_ohe]).toarray(),
                    columns=ohe.get_feature_names_out(self.cols_for_ohe)
                ).reset_index(drop=True)
            )
        
        if self.cols_for_scaling:
            dataframes.append(
                pd.DataFrame(
                    self.scaling_algo.fit_transform(X[self.cols_for_scaling]),
                    columns=self.scaling_algo.get_feature_names_out(self.cols_for_scaling)
                ).reset_index(drop=True)
            )

        if self.cols_to_skip:
            dataframes.append(X[self.cols_to_skip].reset_index(drop=True))

        return pd.concat(dataframes, axis=1)