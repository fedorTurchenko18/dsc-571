import pandas as pd, os, warnings
warnings.filterwarnings('ignore')

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA

from typing import Tuple


class SparkPCATransformer:
    def __init__(self, X_train, X_test, **spark_env_vars) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.spark_env_vars = spark_env_vars

    def get_pca_features(self, k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        for env_var in self.spark_env_vars.keys():
            os.environ[env_var] = self.spark_env_vars[env_var]
        spark = SparkSession.builder.appName('pca').master('local[*]').getOrCreate()
        
        X = spark.createDataFrame(pd.concat([self.X_train, self.X_test], axis=0))

        assembler = VectorAssembler(inputCols=[col_name for col_name, dtype in X.dtypes if dtype == 'double'], outputCol="features")
        X = assembler.transform(X)

        scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
        scaler_model = scaler.fit(X)
        X_scaled = scaler_model.transform(X)

        pca = PCA(k=k, inputCol="scaled_features", outputCol="pca_features")
        pca_model = pca.fit(X_scaled)
        X_pca = pca_model.transform(X_scaled).select('pca_features').toPandas()
        self.explained_variance_ = sum(pca_model.explainedVariance.toArray().tolist())

        pca_cols = [f"comp_{i}" for i in range(1, k+1)]
        X_pca[pca_cols] = X_pca['pca_features'].apply(pd.Series)
        X_pca.drop('pca_features', axis=1, inplace=True)

        X_train_pca = X_pca.loc[self.X_train.index]
        X_test_pca = X_pca.loc[self.X_test.index]

        spark.stop()
        return X_train_pca, X_test_pca