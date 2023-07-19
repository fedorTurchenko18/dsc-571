import pandas as pd
import numpy as np
import warnings
import yaml

from datetime import datetime, timedelta
from typing import Annotated, Callable, Union, List
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

###--- Unused ---###
# from bertopic import BERTopic
# from huggingface_hub import login, logout
###--------------###


with open('features/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


@dataclass
class PeriodValueRange:
    min: int
    max: float


class FeatureExtractor:

    LAMBDA_FEATURES: List[dict] = config['lambda_features']
    CUSTOMER_LEVEL_FEATURES: List[dict] = config['customer_level_features']
    CURRENT_FEATURES: List[str] = config['current_features']
    TRAIN_TEST_SPLIT_PARAMS: dict = config['train_test_split_params']
    HUGGINGFACE_TOKEN: str = config['huggingface']['token']
    HUGGINGFACE_MODEL_REPO: str = config['huggingface']['model_repo']

    def __init__(
            self,
            sales: pd.DataFrame,
            # `customers` dataset is currently considered to be useless for feature generation
            customers: pd.DataFrame = None,
            period: Annotated[int, PeriodValueRange(7, float('inf'))] = 30,
            target_month: Annotated[int, PeriodValueRange(1, float('inf'))] = 3,
            lambda_features = LAMBDA_FEATURES,
            customer_level_features = CUSTOMER_LEVEL_FEATURES,
            current_features: List[str] = CURRENT_FEATURES,
            train_test_split_params = TRAIN_TEST_SPLIT_PARAMS,
            huggingface_token = HUGGINGFACE_TOKEN,
            huggingface_model_repo = HUGGINGFACE_MODEL_REPO
        ):
        '''
        `sales` - dataframe with transactional data \n

        `customers` - [UNUSED] dataframe with customers-related data \n

        `period` - period for feature extraction, 7 days at minimum; defaults to 30 days, starting from the first visit
            Some features are extracted weekly. For example if `period` = 30, these features will be extracted within sub-periods:
                `[ [1st-7th], [8th-14th], [15th-21st]), [22nd-28th] ]` days

        `target_month` - consecutive month for which the prediction should be performed; defaults to the 3rd month of user activity \n

        `lambda_features` - dictionary of features to be extracted from the given dataset using `.apply` => `.lambda` method
            Each item of this dictionary is of a form:
                `{'given_dataset_column_name': {'new_feature_name': 'some_name', 'lambda_func': 'some_func'}}`

        `customer_level_features` - list of dictionaries, which represent features to be extracted from the given dataset at customer level through `self.pivot_table` method
            Each dictionary stores keyword arguments of `pd.pivot_table` method and optional one called `prefix`
                `prefix` is a prefix which will be applied to each column, generated from `pd.pivot_table` \n
                Must-use when generating feature from `'columns': 'breaks'` so that resulting X dataframe columns are named not as:
                - `['1-7', '8-14', ...]` (same column names for all variables generated from 'breaks' column)
                But:
                - `['prefix1_1-7', 'prefix1_8-14', ...]`
                - `['prefix2_1-7', 'prefix2_8-14', ...]`

        `current_features` - list of predictors (applies to X dataframe only) to keep for modelling after all transformations are performed

        `train_test_split_params` - dictionary of keyword arguments to pass to `sklearn.model_selection.train_test_split` method
            This is to ensure homogeneity of experiments in terms of shuffling data, using the same test sample size, etc.

        `huggingface_token` - [UNUSED] access token to login and download model for clustering `sales['prodcategoryname']` categories into broad ones \n
        
        `huggingface_model_repo` - [UNUSED] repository of the model for clustering `sales['prodcategoryname']` categories into broad ones
        '''
        self.sales = sales
        self.customers = customers
        self.period = period
        self.target_month = target_month
        self.lambda_features = lambda_features
        self.customer_level_features = customer_level_features
        self.current_features = current_features
        self.train_test_split_params = train_test_split_params
        self.huggingface_token = huggingface_token
        self.huggingface_model_repo = huggingface_model_repo

    
    def transform(self):
        '''
        Main method, which wraps all the filterings, transformations, joins, extractions, etc.
        Returns:
            X: pd.DataFrame - predictors
            y: pd.DataFrame - target variable
        '''
        # Remove data before June 2021
        self.sales = self.filter_sales()

        # Extract target first, since then the data will only be limited to a first month of activity for each user
        # but target refers to the activity during `target_month` month
        self.sales = self.extract_target()

        # Leave only the first month of activity data for each user
        self.sales = self.extract_training_period()

        # Define weekly sub-periods for each user
        self.sales = self.extract_subperiods()

        # These features have unique extraction algorithms, so they are generated independently
        self.sales = self.extract_days_between_visits()
        self.sales = self.extract_peak_hours()

        # Extract "lambda" features
        for feature in self.lambda_features:
            self.sales = self.extract_feature_lambda(
                initial_col=feature,
                feature_name=self.lambda_features[feature]['new_feature_name'],
                # lambda expression is stored as a string in config => use `eval`
                lambda_func=eval(self.lambda_features[feature]['lambda_func'])
            )

        # Depends on `prodcatbroad` column, thus, executed after extraction of "lambda" features
        self.sales = self.extract_fuel_type()

        # Perform transition to customer level
        pivot_tables = []
        for feature in self.customer_level_features:
            pivot_tables.append(
                self.pivot_table(
                    self.sales,
                    **feature
                )
            )
        df_customer_level = pd.concat(pivot_tables, axis=1).reset_index()
        try:
            X = df_customer_level[self.current_features].fillna(0)
        except KeyError:
            warnings.warn('Certain columns, specified in `current_features` list of class constructor, do not exist. Full dataframe will be returned')
            X = df_customer_level.fillna(0)
        y = df_customer_level['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            **self.train_test_split_params
        )
        return X_train, X_test, y_train, y_test


    def filter_sales(self):
        '''
        Method to filter the transactional training data
        Data before June 2021 is irrelevant due to Covid restrictions, etc.
        '''
        self.sales['receiptdate'] = pd.to_datetime(self.sales['receiptdate'])
        
        # if clause is needed to ensure the transformation is performed only on train set
        if self.sales['receiptdate'].min() < datetime(2021, 6, 1, 0, 0, 0):
            return self.sales[
                self.sales['receiptdate'] >= datetime(2021, 6, 1, 0, 0, 0)
            ]


    def extract_target(self):
        '''
        Creates binary target variable, indicating if at least one transaction was recorded (`target = 1`) for a user
        at `self.target_month` month, starting from the first transaction
        Returns:
            `1 | 0`
        '''
        self.sales = self.sales.sort_values(['ciid', 'receiptdate'])

        self.sales = pd.concat(
            [
                self.sales.reset_index(drop=True),
                self.sales.groupby('ciid')['receiptdate'].apply(self.cut_series, 30).to_frame('months_enum').explode('months_enum').reset_index(drop=True)
            ],
            axis=1
        )

        def months_from_breaks(s):
            return np.int32(s.split('-')[1])/30
        
        self.sales['months_enum'] = self.sales['months_enum'].apply(months_from_breaks)
        mask = self.sales['ciid'].isin(self.sales[self.sales['months_enum']==self.target_month]['ciid'].unique())
        mask.replace(True, 1, inplace=True)
        mask.replace(False, 0, inplace=True)
        self.sales['target'] = mask

        return self.sales


    def extract_training_period(self):
        '''
        Method to filter transactional data with respect to the first `self.period` days for each user
        '''
        def filter_customer_transactions(s, period=self.period):
            return s <= (s.min()+timedelta(days=period))
        mask = self.sales.groupby('ciid')['receiptdate'].apply(filter_customer_transactions)
        self.sales = self.sales[mask]
        return self.sales


    def extract_subperiods(self):
        '''
        Method to calculate sub-periods ranges for each user, append them as a column to the transactional dataset
        '''
        self.sales = pd.concat(
            [
                self.sales.reset_index(drop=True),
                self.sales.groupby('ciid')['receiptdate'].apply(self.cut_series, 7).to_frame('breaks').explode('breaks').reset_index(drop=True)
            ],
            axis=1
        )
        return self.sales
    

    def cut_series(self, s: pd.Series, days: int):
        '''
        Method to be used as an aggregation function after `groupby`
        
        s - represents the series, containing all records of a user
        days - days per period

        Returns: pandas.Series
            breaks, representing subperiods
        '''
        s = s.astype('datetime64[D]')
        min_date = s.min()
        max_date = s.max()
        # Compute total number of periods - number of weeks which fit in the date range for a user
        dates_diff = (max_date-min_date).days
        if dates_diff <= days:
            # If only less or equal a week remained after filtering, only 1 week fits in the date range for a user
            total_periods = 1
        else:
            total_periods = np.floor(dates_diff / days)
        interval_idx = pd.interval_range(start=s.min(), end=s.max(), periods=total_periods, closed='left')

        date_ranges = []
        for idx, date_range in enumerate(interval_idx):
            if idx==0:
                # First period should be unchanged
                date_ranges.append(
                    (
                        pd.Timestamp(date_range.left.date()),
                        pd.Timestamp(date_range.right.date())
                    )
                )
            else:
                # Pandas overlaps periods by default
                # Therefore, extend each other one by its order number
                date_ranges.append(
                    (
                        pd.Timestamp(
                            (date_range.left + timedelta(days=idx)).date()    
                        ),
                        pd.Timestamp(
                            (date_range.right + timedelta(days=idx)).date()
                        )
                    )
                )
        new_series = pd.cut(s.tolist(), pd.IntervalIndex.from_tuples(date_ranges, closed='both'))

        # Constants:
        labels = [f'1-{days}'] # whatever `period` value is, first one will always be 1-`days`
        lower = 1
        upper = days
        for _ in range(1, len(new_series.categories)):
            lower += days
            upper += days
            labels.append(f'{lower}-{upper}')

        return new_series.rename_categories({i: j for i, j in zip(new_series.categories, labels)})
    

    def extract_days_between_visits(self):
        '''
        Method to extract the average number of days between visits feature
        '''
        def compute_average(s):
            # NaT occur if customer made only one visit
            if type(s) == pd._libs.tslibs.nattype.NaTType:
                return 0
            else:
                return s[~pd.isnull(s)].days.sum()/s.shape[0]

        # Visit - unique `receiptdate` entry
        # Therefore, drop duplicates of this column for each user
        tmp = self.sales.drop_duplicates(['ciid', 'receiptdate'])\
                .sort_values(['ciid', 'receiptdate'])\
                    .groupby('ciid')\
                        .agg(days_between_visits = pd.NamedAgg('receiptdate', pd.Series.diff))\
                            .reset_index()
        tmp['average_days_between_visits'] = tmp['days_between_visits'].apply(compute_average)

        self.sales = pd.merge(
            self.sales,
            tmp[['ciid', 'average_days_between_visits']],
            how='left',
            on='ciid'
        )
        return self.sales
    
    
    def extract_peak_hours(self, hours_start: List[str] = ['07:00', '13:00', '17:30'], hours_end: List[str] = ['08:15', '14:00', '19:30']):
        '''
        Method to extract the feature, indicating if the transaction was made during the peak hours or regular hours
        '''
        peak_hours = pd.concat(
            [
                self.sales.set_index(['receiptdate']).between_time(s, e) for s, e in zip(hours_start, hours_end)
            ]
        ).reset_index()['receiptdate'].unique()
        mask = self.sales['receiptdate'].isin(peak_hours)
        mask.replace(True, 'peak_hours_qty', inplace=True)
        mask.replace(False, 'usual_hours_qty', inplace=True)
        self.sales['peak_hour'] = mask
        return self.sales


    def extract_feature_lambda(self, feature_name: str, initial_col: str, lambda_func: Callable):
        '''
        Method to extract feature from the given dataset using `.apply` => `.lambda` method
        Arguments replicate structure of a `self.lambda_features`
        '''
        self.sales[feature_name] = self.sales[initial_col].apply(lambda_func)
        return self.sales
    

    def extract_fuel_type(self):
        '''
        Method to extract fuel type feature
        Should be executed in main method `transform` after extraction of feature `prodcatbroad` (from `extract_feature_lambda` method)
        '''
        self.sales['fuel_type'] = np.where(
            self.sales['prodcatbroad'] == 'fuel_qty',
            self.sales['proddesc'].str.lower().str.replace(' ', '_') + '_qty', 'other'
        )
        return self.sales
    

    def to_customer_level(self, values: str, index: str, columns: str, aggfunc: Union[str, Callable]):
        '''
        Method to transform features into a format of final dataframe by aggregating data on customer level
        '''
        return pd.pivot_table(self.sales, values, index, columns, aggfunc).reset_index()


    # def extract_topic_modelling_features(self):
    #     '''
    #     Method to cluster categories of `sales['prodcategoryname']` through pre-trained topic modelling model
    #     Currently unused, since features turned out to be unrepresentative in terms of variety of relationship between classes of target variable
    #     '''
    #     login(token=self.huggingface_token)
    #     topic_model = BERTopic.load(self.huggingface_model_repo)
    #     logout()
    #     unique_categories = self.sales[~self.sales['prodcategoryname'].isin(['FUELS', 'CAR WASH'])]['prodcategoryname'].unique()
    #     print(unique_categories.shape[0])
    #     topics, probs = topic_model.transform(unique_categories)
    #     topic_model_df = topic_model.get_topic_info().set_index('Topic')
    #     mapping = {
    #         cat: topic_model_df.loc[topic, 'Name'][topic_model_df.loc[topic, 'Name'].find('_')+len('_'):] for cat, topic in zip(unique_categories, topics)
    #     }
    #     self.sales['prodcatbroad'] = self.sales['prodcategoryname'].apply(lambda x: 'fuel' if x == 'FUELS' else 'car_wash' if x == 'CAR WASH' else mapping[x])
    #     return self.sales


    def merge_dataframes(self):
        '''
        Method to merge two dataframes and ensure only customers with at least one transaction are remained
        as well as appropriate time period is selected for training data
        Currently unused, since `customers` dataset is not useful for feature generation
        '''
        sales_unique_customers = self.sales['ciid'].unique()
        self.customers = self.customers[
            self.customers['ciid'].isin(sales_unique_customers)
        ]
        df = pd.merge(
            self.sales,
            self.customers,
            on='ciid',
            how='left'
        )
        return df
    

    def pivot_table(self, data: pd.DataFrame, values: str, index: str, columns: str, aggfunc: str, prefix: str = None):
        '''
        Implements the functionality of `pandas.pivot_table` but also allows to rename columns of transformed dataframe with desired `prefix`
        Inputs to these method are passed from `customer_level_features` array, which is passed to the constructor
        '''
        pivot = pd.pivot_table(
            data,
            values,
            index,
            columns,
            aggfunc
        )
        if prefix:
            pivot.columns = [f'{prefix}_{col}' for col in pivot.columns]
        return pivot
