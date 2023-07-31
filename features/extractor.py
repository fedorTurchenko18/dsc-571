import pandas as pd
import numpy as np
import warnings
import yaml
import pickle

from configs import settings
from datetime import datetime, timedelta
from typing import Annotated, Callable, Union, List, Literal
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bertopic import BERTopic
from huggingface_hub import login, logout


with open('features/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


@dataclass
class PeriodValueRange:
    min: int
    max: float


class FeatureExtractor:

    GROUPBY_FEATURES: List[dict] = config['groupby_features']
    LAMBDA_FEATURES: List[dict] = config['lambda_features']
    CUSTOMER_LEVEL_FEATURES: List[dict] = config['customer_level_features']
    CURRENT_FEATURES: List[str] = config['current_features']
    TRAIN_TEST_SPLIT_PARAMS: dict = config['train_test_split_params']
    HUGGINGFACE_TOKEN: str = settings.SETTINGS['HUGGINGFACE_TOKEN']
    HUGGINGFACE_MODEL_REPO: str = settings.SETTINGS['HUGGINGFACE_MODEL_REPO']

    def __init__(
            self,
            sales: pd.DataFrame,
            customers: pd.DataFrame,
            generation_type: Union[Literal['continuous'], Literal['categorical']],
            filtering_set: Union[Literal['customers'], Literal['sales']],
            period: Annotated[int, PeriodValueRange(7, float('inf'))] = 30,
            subperiod: Annotated[int, PeriodValueRange(7, float('inf'))] = None,
            target_month: Annotated[int, PeriodValueRange(1, float('inf'))] = 3,
            perform_split: bool = True,
            groupby_features = GROUPBY_FEATURES,
            lambda_features = LAMBDA_FEATURES,
            customer_level_features = CUSTOMER_LEVEL_FEATURES,
            current_features: List[str] = CURRENT_FEATURES,
            train_test_split_params = TRAIN_TEST_SPLIT_PARAMS,
            huggingface_token = HUGGINGFACE_TOKEN,
            huggingface_model_repo = HUGGINGFACE_MODEL_REPO
        ):
        '''
        `sales` - dataframe with transactional data \n

        `customers` - dataframe with customers-related data \n

        `generation_type` - TODO: add docstring \n

        `filtering_set` - TODO: add docstring \n

        `period` - period for feature extraction, 7 days at minimum; defaults to 30 days, starting from the first visit
            Some features are extracted weekly. For example if `period` = 30, these features will be extracted within sub-periods:
                `[ [1st-7th], [8th-14th], [15th-21st]), [22nd-28th] ]` days

        `subperiod` - TODO: add docstring

        `target_month` - consecutive month for which the prediction should be performed; defaults to the 3rd month of user activity \n

        `perform_split` - if `sklearn.model_selection.train_test_split` should be performed \n

        `groupby_features` - TODO: add docstring \n

        `lambda_features` - dictionary of features to be extracted from the given dataset using `.apply` => `.lambda` method
            TODO: change docstring
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
        self.generation_type = generation_type
        self.filtering_set = filtering_set
        self.period = period
        self.subperiod = subperiod
        self.target_month = target_month
        self.perform_split = perform_split
        self.groupby_features = groupby_features
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
        if self.subperiod:
            self.sales = self.extract_subperiods()

        # These features have unique extraction algorithms, so they are generated independently
        self.sales = self.extract_days_between_visits()
        self.sales = self.extract_peak_hours()
        self.sales = self.extract_last_purchase_share()
        self.sales = self.extract_topic_modelling_features()

        #Extract "groupby" features
        for feature in self.groupby_features:
            self.sales = self.extract_feature_groupby(
                **feature
            )

        # Extract "lambda" features
        for feature in self.lambda_features:
            self.sales = self.extract_feature_lambda(
                **feature
            )

        # Depends on `prodcatbroad` column, thus, executed after extraction of "lambda" features
        self.sales = self.extract_fuel_type()

        # Perform transition to customer level
        pivot_tables = []
        for feature in self.customer_level_features[self.generation_type]:
            if self.subperiod:
                # Breakdown data by `self.subperiod`
                current_column = feature['columns']
                if current_column:
                    # Check if `columns` argument value is not `None`
                    # to assure pivoting with 'breaks'
                    feature['columns'] = [current_column, 'breaks']
                    pt = self.pivot_table(
                        self.sales,
                        **feature
                    )
                    pt.columns = [f'{feature_col}_{breakdown_col}' for feature_col, breakdown_col in zip(pt.columns.get_level_values(0), pt.columns.get_level_values(1))]
                
                else:
                    current_value = feature['values']
                    if current_value == 'target':
                        # target variable cannot be broken down
                        pt = self.pivot_table(
                            self.sales,
                            **feature
                        )
                    else:
                        feature['columns'] = 'breaks'
                        feature['prefix'] = current_value
                        pt = self.pivot_table(
                            self.sales,
                            **feature
                        )

                pivot_tables.append(pt)
            else:
                pivot_tables.append(
                    self.pivot_table(
                        self.sales,
                        **feature
                    )
                )
        df_customer_level = pd.concat(pivot_tables, axis=1).reset_index()
        if not self.subperiod:
            # User segment cannot be interpreted at monthly level
            df_customer_level = self.extract_clustering_feature(df_customer_level)
        else:
            # Add RFM features instead of segments
            if self.generation_type=='continuous':
                self.current_features[self.generation_type].extend(['monetary', 'recency', 'average_days_between_visits'])
        df_customer_level = pd.concat(
            [
                df_customer_level.select_dtypes(exclude='category').fillna(0),
                df_customer_level.select_dtypes(include='category')
            ],
            axis=1
        )
        try:
            final_cols = []
            for col in df_customer_level.columns:
                for comp_col in self.current_features[self.generation_type]:
                    if comp_col in col:
                        final_cols.append(col)
            X = df_customer_level[final_cols]
        except KeyError:
            warnings.warn('Certain columns, specified in `current_features` list of class constructor, do not exist. Full dataframe will be returned')
            X = df_customer_level
        y = df_customer_level['target']

        if self.generation_type == 'categorical':
            cat_cols = X.select_dtypes(include=['int64', 'object']).columns
            X[cat_cols] = X.select_dtypes(include=['int64', 'object']).astype('category')
        
        if self.perform_split:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                **self.train_test_split_params
            )
            return X_train, X_test, y_train, y_test
        else:
            return X, y


    def filter_sales(self):
        '''
        Method to filter the transactional training data
        Data before June 2021 is irrelevant due to Covid restrictions, etc.
        '''
        self.sales['receiptdate'] = pd.to_datetime(self.sales['receiptdate'])
        
        # if clause is needed to ensure the transformation is performed only on train set
        if self.sales['receiptdate'].min() < datetime(2021, 6, 1, 0, 0, 0):
            if self.filtering_set == 'sales':
                return self.sales[
                    self.sales['receiptdate'] >= datetime(2021, 6, 1, 0, 0, 0)
                ]
            elif self.filtering_set == 'customers':
                return self.sales[
                    self.sales['ciid'].isin(
                        self.customers[
                            self.customers['accreccreateddate'] >= datetime(2021, 6, 1, 0, 0, 0)
                        ]['ciid']
                    )
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
        if self.subperiod:
            self.sales = pd.concat(
                [
                    self.sales.reset_index(drop=True),
                    self.sales.groupby('ciid')['receiptdate'].apply(self.cut_series, self.subperiod).to_frame('breaks').explode('breaks').reset_index(drop=True)
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
    

    def extract_last_purchase_share(self):
        '''
        TODO: add docstring
        '''
        tmp = self.sales.groupby(['ciid', 'receiptdate']).agg({'qty': 'sum'}).reset_index()
        tmp = pd.merge(
            tmp.groupby('ciid').agg(amount_spent = pd.NamedAgg('qty', 'sum')),
            tmp.loc[tmp.groupby('ciid')['receiptdate'].apply(pd.Series.idxmax).values, :][['ciid', 'qty']],
            how='left',
            on='ciid'
        )
        tmp['last_purchase_share'] = tmp['qty']/tmp['amount_spent']
        self.sales = pd.merge(
            self.sales,
            tmp[['ciid', 'last_purchase_share']],
            on='ciid',
            how='left'
        )
        return self.sales


    def extract_feature_groupby(self, groupcol, aggcol, aggfunc, to_frame_name):
        '''
        TODO: add docstring
        '''
        aggfunc = eval(aggfunc)
        tmp = self.sales.groupby(groupcol)[aggcol].apply(aggfunc).to_frame(to_frame_name).reset_index()
        self.sales = pd.merge(
            self.sales,
            tmp,
            how='left',
            on=groupcol
        )
        return self.sales


    def extract_feature_lambda(self, feature_name: str, initial_col: str, lambda_func: Callable):
        '''
        Method to extract feature from the given dataset using `.apply` => `.lambda` method
        Arguments replicate structure of a `self.lambda_features`
        '''
        lambda_func = eval(lambda_func)
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
    

    def extract_topic_modelling_features(self):
        '''
        Method to cluster categories of `sales['prodcategoryname']` through pre-trained topic modelling model
        Currently unused, since features turned out to be unrepresentative in terms of variety of relationship between classes of target variable
        '''
        login(token=self.huggingface_token)
        topic_model = BERTopic.load(self.huggingface_model_repo)
        logout()
        unique_categories = self.sales[~self.sales['prodcategoryname'].isin(['FUELS', 'CAR WASH'])]['prodcategoryname'].unique()
        topics, probs = topic_model.transform(unique_categories)
        topic_model_df = topic_model.get_topic_info().set_index('Topic')
        mapping = {
            cat: topic_model_df.loc[topic, 'Name'][topic_model_df.loc[topic, 'Name'].find('_')+len('_'):] for cat, topic in zip(unique_categories, topics)
        }
        self.sales['prodcatbroad'] = self.sales['prodcategoryname'].apply(
            lambda x: 'fuel_qty' if x == 'FUELS' else 'car_wash_qty' if x == 'CAR WASH' else f'{mapping[x]}_qty'
        )
        return self.sales
    

    def extract_clustering_feature(self, df_customer_level: pd.DataFrame):
        '''
        Method to extract clusters (i.e. customer segments) based on RFM variables
        TODO: add extended docstring
        '''
        # Load clustering model
        with open('./features/clustering_model.pkl', 'rb') as f:
            model = pickle.load(f)
        # Load `scipy.stats.mstats.winsorize` output object to define threshold for the `monetary` variable
        with open('./features/winsorizing_object_for_threshold.pkl', 'rb') as f:
            winsor = pickle.load(f)
        X_clust = df_customer_level[['monetary', 'recency', 'average_days_between_visits']]
        monetary_threshold = winsor.max()
        # Perform winsorization
        X_clust.loc[X_clust['monetary'] > monetary_threshold, 'monetary'] = monetary_threshold
        scaler = StandardScaler()
        labels = pd.Categorical(
            model.predict(
                scaler.fit_transform(X_clust)
            )
        )
        df_customer_level['segments'] = labels
        df_customer_level['segments'] = df_customer_level['segments'].cat.rename_categories({2: 'frequent_drivers', 1: 'passerbys', 0: 'regular_drivers'})
        return df_customer_level
    

    def perform_train_test_split(self, X: pd.DataFrame, y: pd.DataFrame):
        '''
        Separate method to implement `sklearn.train_test_split` with pre-selected params
        if `FeatureExctractor(..., perform_split=False)` was called

        X, y - outputs of `FeatureExctractor(..., perform_split=False).transform()` method
        '''
        X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                **self.train_test_split_params
            )
        return X_train, X_test, y_train, y_test


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
        try:
            # Aggregate function is not a keyword (e.g. string `'pd.Series.nunique'`)
            aggfunc = eval(aggfunc)
        except NameError:
            # Aggregate function is a keyword (e.g. 'mean')
            pass
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
