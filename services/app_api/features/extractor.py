import pandas as pd
import numpy as np
import warnings
import yaml
import pickle
import copy

from services.app_api.configs import settings
from datetime import datetime, timedelta
from typing import Annotated, Callable, Union, List, Literal
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bertopic import BERTopic
from huggingface_hub import login, logout


with open('services/app_api/features/config.yaml', 'r') as f:
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
            # sales: pd.DataFrame,
            # customers: pd.DataFrame,
            generation_type: Union[Literal['continuous'], Literal['categorical']],
            filtering_set: Union[Literal['customers'], Literal['sales']],
            period: Annotated[int, PeriodValueRange(7, float('inf'))] = 30,
            subperiod: Annotated[int, PeriodValueRange(7, float('inf'))] = None,
            target_month: Annotated[int, PeriodValueRange(1, float('inf'))] = 3,
            n_purchases: Annotated[int, PeriodValueRange(1, float('inf'))] = 3,
            perform_split: bool = True,
            groupby_features = GROUPBY_FEATURES,
            lambda_features = LAMBDA_FEATURES,
            customer_level_features = CUSTOMER_LEVEL_FEATURES,
            current_features: List[str] = CURRENT_FEATURES,
            train_test_split_params = TRAIN_TEST_SPLIT_PARAMS,
            huggingface_token = HUGGINGFACE_TOKEN,
            huggingface_model_repo = HUGGINGFACE_MODEL_REPO,
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
        # self.sales = sales
        # self.customers = customers
        self.generation_type = generation_type
        self.filtering_set = filtering_set
        self.period = period
        self.subperiod = subperiod
        self.target_month = target_month
        self.n_purchases = n_purchases
        self.perform_split = perform_split
        self.groupby_features = groupby_features
        self.lambda_features = lambda_features
        self.customer_level_features = customer_level_features
        self.current_features = current_features
        self.train_test_split_params = train_test_split_params
        self.huggingface_token = huggingface_token
        self.huggingface_model_repo = huggingface_model_repo

    
    def transform(self, sales, customers):
        '''
        Main method, which wraps all the filterings, transformations, joins, extractions, etc.
        Returns:
            X: pd.DataFrame - predictors
            y: pd.DataFrame - target variable
        '''
        # Remove data before June 2021
        sales = self.filter_sales(sales, customers)
        if sales.shape[0] == 0:
            # Filtering removed all rows
            # Could be the case, e.g., if sales dataset is being cut to a single customer
            # with outdated records
            if self.perform_split:
                X_train, X_test, y_train, y_test = None, None, None, None
                return X_train, X_test, y_train, y_test
            else:
                X, y = None, None
                return X, y

        # Extract target first, since then the data will only be limited to a first month of activity for each user
        # but target refers to the activity during `target_month` month
        customer_level_features = copy.deepcopy(self.customer_level_features[self.generation_type])
        current_features = copy.deepcopy(self.current_features[self.generation_type])
        if self.target_month:
            sales = self.extract_target(sales)
        else:
            for d in customer_level_features:
                if d['values'] == 'target':
                    customer_level_features.remove(d)

        # Leave only the first month of activity data for each user
        sales = self.extract_training_period(sales)

        # Define weekly sub-periods for each user
        if self.subperiod:
            sales = self.extract_subperiods(sales)

        # These features have unique extraction algorithms, so they are generated independently
        sales = self.extract_days_between_visits(sales)
        sales = self.extract_peak_hours(sales)
        sales = self.extract_last_purchase_share(sales)
        sales = self.extract_topic_modelling_features(sales)
        sales = self.extract_recency(sales)

        #Extract "groupby" features
        for feature in self.groupby_features:
            sales = self.extract_feature_groupby(
                sales,
                **feature
            )
        # This feature is generally extracted through "groupby" methodology
        # However, needs additional oneline tweak to be extracted
        if self.subperiod:
            sales['days_of_inactivity'] = sales['days_of_inactivity'].apply(lambda x: self.subperiod-x)
        else:
            sales['days_of_inactivity'] = sales['days_of_inactivity'].apply(lambda x: self.period-x)

        # Extract "lambda" features
        for feature in self.lambda_features:
            sales = self.extract_feature_lambda(
                sales,
                **feature
            )

        # Depends on `prodcatbroad` column, thus, executed after extraction of "lambda" features
        sales = self.extract_fuel_type(sales)
        sales['ciid_copy'] = sales['ciid']
        # Perform transition to customer level
        pivot_tables = []
        for feature in copy.copy(customer_level_features):
            if self.subperiod:
                # Breakdown data by `self.subperiod`
                current_column = feature['columns']
                if current_column:
                    # Check if `columns` argument value is not `None`
                    # to assure pivoting with 'breaks'
                    feature['columns'] = [current_column, 'breaks']
                    pt = self.pivot_table(
                        sales,
                        **feature
                    )
                    pt.columns = [f'{feature_col}_{breakdown_col}' for feature_col, breakdown_col in zip(pt.columns.get_level_values(0), pt.columns.get_level_values(1))]
                
                else:
                    # `columns` argument value is `None`
                    current_value = feature['values']
                    if (
                        current_value == 'target'
                        or
                        (
                            current_value in ['recency', 'average_days_between_visits', 'monetary']
                            and
                            self.generation_type == 'categorical'
                        )
                    ):
                        # target variable cannot be broken down
                            # AND #
                        # RFM features should be extracted in their initial form
                        # to assure appropriate segments extraction
                        # when calling clustering model
                        pt = self.pivot_table(
                            sales,
                            **feature
                        )
                    else:
                        # RFM features are being broken down for `'continuous'` `generation_type`
                        # as well as other features with `columns` argument value equal to `None`
                        feature['columns'] = 'breaks'
                        feature['prefix'] = current_value
                        pt = self.pivot_table(
                            sales,
                            **feature
                        )
                pivot_tables.append(pt)
            else:
                # Do not breakdown data by subperiod
                pivot_tables.append(
                    self.pivot_table(
                        sales,
                        **feature
                    )
                )
        df_customer_level = pd.concat(pivot_tables, axis=1).reset_index()
        if self.generation_type=='categorical':
            # Segments are generated for categorical version of model only
            df_customer_level = self.extract_clustering_feature(df_customer_level)
        else:
            # Add RFM features to the output feature list instead of segments for continuous features
            current_features.extend(['monetary', 'recency', 'average_days_between_visits'])
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
                for comp_col in current_features:
                    if comp_col in col:
                        final_cols.append(col)
            # Compute intraperiod difference for feature, broken down by subperiod
            X = df_customer_level[final_cols]
            X = self.extract_intraperiod_difference(X)
        except KeyError:
            warnings.warn('Certain columns, specified in `current_features` list of class constructor, do not exist. Full dataframe will be returned')
            X = df_customer_level

        if self.generation_type == 'categorical':
            # Change type of last_purchase_qty_share to set it aside from breakdown columns
            last_purchase_share_cols = [col for col in X.columns if 'last_purchase_qty_share' in col]
            X[last_purchase_share_cols] = X[last_purchase_share_cols].astype('float32')
            # Convert breakdown columns to integer
            float_cols = X.select_dtypes(include=['float64']).columns
            X[float_cols] = X[float_cols].astype('int16')
            # Convert to categorical format for `catboost` proper interpetation of these
            cat_cols = X.select_dtypes(include=['int16', 'object']).columns
            X[cat_cols] = X[cat_cols].astype('category')
        
        if self.perform_split:
            y = df_customer_level['target']
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                **self.train_test_split_params
            )
            return X_train, X_test, y_train, y_test
        else:
            if self.target_month:
                y = df_customer_level['target']
                return X, y
            else:
                return X


    def filter_sales(self, sales, customers):
        '''
        Method to filter the transactional training data
        Data before June 2021 is irrelevant due to Covid restrictions, etc.
        '''
        sales['receiptdate'] = pd.to_datetime(sales['receiptdate'])
        customers['accreccreateddate'] = pd.to_datetime(customers['accreccreateddate'])
        
        # if clause is needed to ensure the transformation is performed only on train set
        if sales['receiptdate'].min() < datetime(2021, 6, 1, 0, 0, 0):
            if self.filtering_set == 'sales':
                return sales[
                    sales['receiptdate'] >= datetime(2021, 6, 1, 0, 0, 0)
                ]
            elif self.filtering_set == 'customers':
                return sales[
                    sales['ciid'].isin(
                        customers[
                            customers['accreccreateddate'] >= datetime(2021, 6, 1, 0, 0, 0)
                        ]['ciid']
                    )
                ]
        else:
            return sales


    def extract_target(self, sales):
        '''
        Creates binary target variable, indicating if at least one transaction was recorded (`target = 1`) for a user
        at `self.target_month` month, starting from the first transaction
        Returns:
            `1 | 0`
        '''
        sales = sales.sort_values(['ciid', 'receiptdate'])
        # sales = pd.concat(
        #     [
        #         sales.reset_index(drop=True),
        #         sales.groupby('ciid')['receiptdate'].apply(self.cut_series, 30).to_frame('months_enum').explode('months_enum').reset_index(drop=True)
        #     ],
        #     axis=1
        # )breaks
        tmp = sales.groupby('ciid')['receiptdate'].apply(self.cut_series, 30).to_frame('months_enum')
        tmp['receiptdate_date'] = tmp['months_enum'].apply(lambda x: x[1].tolist())
        tmp['months_enum'] = tmp['months_enum'].apply(lambda x: x[0])
        tmp = tmp.explode(['months_enum', 'receiptdate_date']).reset_index()
        tmp.drop_duplicates(inplace=True)
        sales['receiptdate_date'] = sales['receiptdate'].dt.date.astype('datetime64')
        sales = pd.merge(
            sales,
            tmp,
            how='left',
            on=['ciid', 'receiptdate_date']
        )
        def months_from_breaks(s):
            return np.int32(s.split('-')[1])/30
        
        sales['months_enum'] = sales['months_enum'].apply(months_from_breaks)
        tmp = sales[sales['months_enum'] == self.target_month].groupby('ciid', as_index=False).agg(purchases_count = pd.NamedAgg('receiptid', pd.Series.nunique))
        tmp = tmp[tmp['purchases_count'] >= self.n_purchases]
        mask = sales['ciid'].isin(tmp['ciid'].unique())
        mask.replace(True, 1, inplace=True)
        mask.replace(False, 0, inplace=True)
        sales['target'] = mask

        return sales


    def extract_training_period(self, sales):
        '''
        Method to filter transactional data with respect to the first `self.period` days for each user
        '''
        def filter_customer_transactions(s, period=self.period):
            return s <= (s.min()+timedelta(days=period))
        mask = sales.groupby('ciid')['receiptdate'].apply(filter_customer_transactions)
        sales = sales[mask]
        return sales


    def extract_subperiods(self, sales):
        '''
        Method to calculate sub-periods ranges for each user, append them as a column to the transactional dataset
        '''
        sales.sort_values(['ciid', 'receiptdate'])
        # sales = pd.concat(
        #     [
        #         sales.reset_index(drop=True),
        #         sales.groupby('ciid')['receiptdate'].apply(self.cut_series, self.subperiod).to_frame('breaks').explode('breaks').reset_index(drop=True)
        #     ],
        #     axis=1
        # )
        tmp = sales.groupby('ciid')['receiptdate'].apply(self.cut_series, self.subperiod).to_frame('breaks')
        tmp['receiptdate_date'] = tmp['breaks'].apply(lambda x: x[1].tolist())
        tmp['breaks'] = tmp['breaks'].apply(lambda x: x[0])
        tmp = tmp.explode(['breaks', 'receiptdate_date']).reset_index()
        tmp.drop_duplicates(inplace=True)
        sales['receiptdate_date'] = sales['receiptdate'].dt.date.astype('datetime64')
        sales = pd.merge(
            sales,
            tmp,
            how='left',
            on=['ciid', 'receiptdate_date']
        )
        return sales
    

    def cut_series(self, s: pd.Series, days: int):
        '''
        Method to be used as an aggregation function after `groupby`
        
        s - represents the series, containing all records of a user
        days - days per period

        Returns: pandas.Series
            breaks, representing subperiods
        '''
        s = pd.Series(s.values.astype('datetime64[D]'))
        min_date = s.min()
        max_date = s.max()
        # Compute total number of periods - number of weeks which fit in the date range for a user
        dates_diff = (max_date-min_date).days
        if dates_diff <= days:
            # If only less or equal a week remained after filtering, only 1 week fits in the date range for a user
            total_periods = 1
        else:
            total_periods = np.ceil(dates_diff / days)
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

        return new_series.rename_categories({i: j for i, j in zip(new_series.categories, labels)}), s
    

    def extract_days_between_visits(self, sales):
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
        if self.subperiod:
            tmp = sales.drop_duplicates(['ciid', 'receiptdate'])\
                    .sort_values(['ciid', 'receiptdate'])\
                        .groupby(['ciid', 'breaks'])\
                            .agg(days_between_visits = pd.NamedAgg('receiptdate', pd.Series.diff))\
                                .reset_index()
        else:
            tmp = sales.drop_duplicates(['ciid', 'receiptdate'])\
                    .sort_values(['ciid', 'receiptdate'])\
                        .groupby('ciid')\
                            .agg(days_between_visits = pd.NamedAgg('receiptdate', pd.Series.diff))\
                                .reset_index()
        tmp['average_days_between_visits'] = tmp['days_between_visits'].apply(compute_average)

        sales = pd.merge(
            sales,
            tmp[['ciid', 'average_days_between_visits']],
            how='left',
            on='ciid'
        )
        return sales
    
    
    def extract_peak_hours(self, sales, hours_start: List[str] = ['07:00', '13:00', '17:30'], hours_end: List[str] = ['08:15', '14:00', '19:30']):
        '''
        Method to extract the feature, indicating if the transaction was made during the peak hours or regular hours
        '''
        peak_hours = pd.concat(
            [
                sales.set_index(['receiptdate']).between_time(s, e) for s, e in zip(hours_start, hours_end)
            ]
        ).reset_index()['receiptdate'].unique()
        mask = sales['receiptdate'].isin(peak_hours)
        mask.replace(True, 'peak_hours_qty', inplace=True)
        mask.replace(False, 'usual_hours_qty', inplace=True)
        sales['peak_hour'] = mask
        return sales
    

    def extract_last_purchase_share(self, sales):
        '''
        TODO: add docstring
        '''
        if self.subperiod:
            tmp = sales.groupby(['ciid', 'receiptdate', 'breaks']).agg({'qty': 'sum'}).reset_index()
            tmp = pd.merge(
                tmp.groupby(['ciid', 'breaks']).agg(amount_spent = pd.NamedAgg('qty', 'sum')),
                tmp.loc[tmp.groupby(['ciid', 'breaks'])['receiptdate'].apply(pd.Series.idxmax).values, :][['ciid', 'qty', 'breaks']],
                how='left',
                on=['ciid', 'breaks']
            )
            tmp['last_purchase_qty_share'] = tmp['qty']/tmp['amount_spent']
            sales = pd.merge(
                sales,
                tmp[['ciid', 'last_purchase_qty_share', 'breaks']],
                on=['ciid', 'breaks'],
                how='left'
            )
        else:
            tmp = sales.groupby(['ciid', 'receiptdate']).agg({'qty': 'sum'}).reset_index()
            tmp = pd.merge(
                tmp.groupby('ciid').agg(amount_spent = pd.NamedAgg('qty', 'sum')),
                tmp.loc[tmp.groupby('ciid')['receiptdate'].apply(pd.Series.idxmax).values, :][['ciid', 'qty']],
                how='left',
                on='ciid'
            )
            tmp['last_purchase_qty_share'] = tmp['qty']/tmp['amount_spent']
            sales = pd.merge(
                sales,
                tmp[['ciid', 'last_purchase_qty_share']],
                on='ciid',
                how='left'
            )
        return sales


    def extract_feature_groupby(self, sales: pd.DataFrame, groupcol, aggcol, aggfunc, to_frame_name):
        '''
        TODO: add docstring
        '''
        aggfunc = eval(aggfunc)
        if self.subperiod:
            tmp = sales.groupby([groupcol, 'breaks'])[aggcol].apply(aggfunc).to_frame(to_frame_name).reset_index()
            sales = pd.merge(
                sales,
                tmp,
                how='left',
                on=[groupcol, 'breaks']
            )
        else:
            tmp = sales.groupby(groupcol)[aggcol].apply(aggfunc).to_frame(to_frame_name).reset_index()
            sales = pd.merge(
                sales,
                tmp,
                how='left',
                on=groupcol
            )
        return sales


    def extract_feature_lambda(self, sales, feature_name: str, initial_col: str, lambda_func: Callable):
        '''
        Method to extract feature from the given dataset using `.apply` => `.lambda` method
        Arguments replicate structure of a `self.lambda_features`
        '''
        lambda_func = eval(lambda_func)
        sales[feature_name] = sales[initial_col].apply(lambda_func)
        return sales
    

    def extract_fuel_type(self, sales):
        '''
        Method to extract fuel type feature
        Should be executed in main method `transform` after extraction of feature `prodcatbroad` (from `extract_feature_lambda` method)
        '''
        sales['fuel_type'] = np.where(
            sales['prodcatbroad'] == 'fuel_qty',
            sales['proddesc'].str.lower().str.replace(' ', '_') + '_qty', 'other'
        )
        return sales
    

    def extract_recency(self, sales):
        '''
        TODO: add docstring
        '''
        if self.subperiod:
            tmp = sales.groupby(['ciid', 'breaks'])['receiptdate']\
                .apply(lambda x: ((x.min()+timedelta(days=self.subperiod))-x.max()).days)\
                    .to_frame('recency').reset_index()
            sales = pd.merge(
                sales,
                tmp,
                how='left',
                on=['ciid', 'breaks']
            )
        else:
            tmp = sales.groupby('ciid')['receiptdate']\
                .apply(lambda x: ((x.min()+timedelta(days=self.period))-x.max()).days)\
                    .to_frame('recency').reset_index()
            sales = pd.merge(
                sales,
                tmp,
                how='left',
                on='ciid'
            )
        sales['recency'] = sales['recency'].replace(-1, 0)
        return sales
    

    def extract_topic_modelling_features(self, sales):
        '''
        Method to cluster categories of `sales['prodcategoryname']` through pre-trained topic modelling model
        Currently unused, since features turned out to be unrepresentative in terms of variety of relationship between classes of target variable
        '''
        unique_categories = sales[~sales['prodcategoryname'].isin(['FUELS', 'CAR WASH'])]['prodcategoryname'].unique()
        if unique_categories.shape[0] > 0:
            login(token=self.huggingface_token)
            topic_model = BERTopic.load(self.huggingface_model_repo)
            logout()
            topics, probs = topic_model.transform(unique_categories)
            topic_model_df = topic_model.get_topic_info().set_index('Topic')
            mapping = {
                cat: topic_model_df.loc[topic, 'Name'][topic_model_df.loc[topic, 'Name'].find('_')+len('_'):] for cat, topic in zip(unique_categories, topics)
            }
            sales['prodcatbroad'] = sales['prodcategoryname'].apply(
                lambda x: 'fuel_qty' if x == 'FUELS' else 'car_wash_qty' if x == 'CAR WASH' else f'{mapping[x]}_qty'
            )
        else:
            sales['prodcatbroad'] = sales['prodcategoryname']
        return sales
    

    def extract_clustering_feature(self, df_customer_level: pd.DataFrame):
        '''
        Method to extract clusters (i.e. customer segments) based on RFM variables
        TODO: add extended docstring
        '''
        # Load clustering model
        with open('services/app_api/features/clustering_model.pkl', 'rb') as f:
            model = pickle.load(f)
        # Load `scipy.stats.mstats.winsorize` output object to define threshold for the `monetary` variable
        with open('services/app_api/features/winsorizing_object_for_threshold.pkl', 'rb') as f:
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


    def extract_intraperiod_difference(self, df_customer_level):
        '''
        TODO: add docstring
        '''
        cols_with_breaks = []
        for col in df_customer_level.columns:
            # Dash exists only in subperiods naming
            if '-' in col \
                and col[:col.rindex('_')] not in cols_with_breaks \
                    and 'qty' in col:
                # Extract only general column name (without subperiod)
                cols_with_breaks.append(col[:col.rindex('_')])
        # gen_col - general column name (without subperiod)
        for gen_col in cols_with_breaks:
            df_customer_level_diff = df_customer_level.loc[
                :, # all rows
                [col for col in df_customer_level.columns if gen_col in col]
            ]
            if df_customer_level_diff.iloc[:, 0].nunique() > 2:
                df_customer_level_diff = df_customer_level_diff.diff(axis=1) # compute difference
                df_customer_level_diff = df_customer_level_diff.iloc[:, 1:] # first subperiod column will always be full NaN
                df_customer_level_diff = df_customer_level_diff.rename({col: f'{col}_previous_period_diff' for col in df_customer_level_diff.columns}, axis=1)
                df_customer_level = pd.concat(
                    [df_customer_level, df_customer_level_diff],
                    axis=1
                )
        return df_customer_level