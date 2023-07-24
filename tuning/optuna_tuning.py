import optuna, numpy as np
from typing import Literal, Union, Callable


class OptunaTuner:
    def __init__(
            self,
            model: Literal['sklearn-model-object'],
            error_metric: Callable,
            direction: Union[Literal['maximize'], Literal['minimize']]
    ) -> None:
        """
        Class constructor

        ## Arguments
        `model`: Literal['sklearn-model-object']
            Object of `sklearn` ML model or `sklearn.pipeline.Pipeline`

        `error_metric`: Callable
            Function to measure model performance

        `direction`: str
            Either to maximize or minimize performance metric
        """
        self.model = model
        self.error_metric = error_metric
        self.direction = direction
        optuna.logging.set_verbosity(0)

    def fit(self, n_trials: int, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, *args) -> None:
        """
        Find optimal hyperparameters and fit the model

        ## Arguments
        `n_trials` : int
            Number of optimization iterations

        `X_train`, `y_train`, `X_test`, `y_test` : np.ndarray
            Training and test datasets
        
        `*args` : tuple
            Tuples containing hyperparameters and their values. If a hyperparameter is:
            - Numeric, then the tuple passed is (hyperparameter_name, min, max)
            - Categorical, then the tuple passed is (hyperparameter_name, [val_1, val_2, ..., val_n])
        """
        def objective(trial, *args):
            """
            Define objective for `optuna.study`
            """
            params = {}
            for arg in args:
                if len(arg) > 2:
                    if type(arg[1]) == int:
                        params[arg[0]] = trial.suggest_int(arg[0], arg[1], arg[2])
                    else:
                        params[arg[0]] = trial.suggest_float(arg[0], arg[1], arg[2])
                else:
                    params[arg[0]] = trial.suggest_categorical(arg[0], arg[1])

            try:
                model = self.model()
            except TypeError:
                # Account for the case when the object passed is not callable
                # e.g. `sklearn.pipeline.Pipeline`
                model = self.model
            model.set_params(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            error = self.error_metric(y_test, y_pred)
            return error

        study = optuna.create_study(direction=self.direction)
        study.optimize(lambda trial: objective(trial, *args), n_trials, n_jobs=7)

        try:
            model = self.model()
        except TypeError:
            model = self.model
        model.set_params(**study.best_params)
        model.fit(X_train, y_train)

        self.model = model
        self.study = study