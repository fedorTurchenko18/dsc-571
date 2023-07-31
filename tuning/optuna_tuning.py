import optuna, numpy as np
from typing import Literal, Union, Callable


class OptunaTuner:
    def __init__(
            self,
            model: Literal['sklearn-model-object'],
            error_metric: Callable,
            direction: Union[Literal['maximize'], Literal['minimize']],
            **model_params
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
        
        `model_params`: kwargs
            Keyword arguments to be passed as default parameters of the specified model
        """
        self.model = model(**model_params)
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
            Tuples containing hyperparameters and their values. Pass as `('hyperparameter_name', 'value_type', value_args)`

            Note that value_args depend on the 'value_type':
            - If `'int'`, then pass as `('hyperparameter_name', 'int', low: int, high: int, Optional[{'step': int = 1, log: bool = False}])`
            - If `'float'`, then pass as `('hyperparameter_name', 'float', low: float, high: float, Optional[{'step': int = None, log: bool = False}])`
            - If `'categorical'`, then pass as `('hyperparameter_name', 'categorical', [list_of_suggested_values])`
            - If `'uniform'`, then pass as `('hyperparameter_name', 'uniform', low: float, high: float)`
            - If `'discrete_uniform'`, then pass as `('hyperparameter_name', 'discrete_uniform', low: float, high: float, q: float)`
            - If `'loguniform'`, then pass as `('hyperparameter_name', 'loguniform', low: float, high: float)`
            
            For int and float one of 'step' and 'log' can be passed as well, the other will be set to default then
        """
        def objective(trial, *args):
            """
            Define objective for `optuna.study`
            """
            params = {}
            for arg in args:
                if arg[1] == 'int':
                    step = 1
                    log = False
                    if len(arg) > 4:
                        step = arg[4]['step'] if 'step' in arg[4].keys() else 1
                        log = arg[4]['log'] if 'log' in arg[4].keys() else False
                    params[arg[0]] = trial.suggest_int(arg[0], arg[2], arg[3], step=step, log=log)
                
                elif arg[1] == 'float':
                    step = None
                    log = False
                    if len(arg) > 4:
                        step = arg[4]['step'] if 'step' in arg[4].keys() else None
                        log = arg[4]['log'] if 'log' in arg[4].keys() else False
                    params[arg[0]] = trial.suggest_float(arg[0], arg[2], arg[3], step=step, log=log)
                
                elif arg[1] == 'categorical':
                    params[arg[0]] = trial.suggest_categorical(arg[0], arg[2])
                
                elif arg[1] == 'uniform':
                    params[arg[0]] = trial.suggest_uniform(arg[0], arg[2], arg[3])

                elif arg[1] == 'discrete_uniform':
                    params[arg[0]] = trial.suggest_discrete_uniform(arg[0], arg[2], arg[3], arg[4])
                
                elif arg[1] == 'loguniform':
                    params[arg[0]] = trial.suggest_loguniform(arg[0], arg[2], arg[3])

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