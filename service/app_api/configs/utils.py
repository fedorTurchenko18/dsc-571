import wandb
import subprocess
import re
import sys

from service.app_api.configs import settings
from datetime import datetime
from typing import Literal, Union


def login_wandb():
    '''
    Performs login to Weights & Biases
    Pre-setting routine is described in `./README.md`
    '''
    try:
        subprocess.check_call(['wandb', 'login', settings.SETTINGS['WANDB_KEY']])
    except:
        # TODO: change to proper logging
        print('Login failed')


def generate_wandb_project_name(model: Literal['sklearn-model']):
    '''
    Generates project name from sklearn model name
    
    `model` - sklearn model object
        Usage example:
            ```
            from sklearn.neighbors import KNeighborsClassifier
            project = generate_wanb_project_name(KNeighborsClassifier)
            ```
    '''
    model_str = model.__name__
    uppercase_letters = re.findall(r'[A-Z]', model_str)
    for letter in uppercase_letters:
        if model_str.find(letter) != 0:
            model_str = model_str[:model_str.find(letter)]+'-'+model_str[model_str.find(letter):]
    return model_str


def init_wandb_run(
    name: str,
    model: Literal['sklearn-model'],
    config: dict,
    target_month: Literal['FeatureExtractor.target_month'],
    group: Literal[
        'default_parameters',
        'parameters_tuning',
        'dimensionality_reduction_default_parameters',
        'dimensionality_reduction_parameters_tuning',
        'resampling_default_parameters',
        'resampling_parameters_tuning'
    ],
    job_type: Literal[
        'train',
        'test',
        'tuning_train',
        'tuning_test'
    ],
    add_timestamp_to_name: bool = True,
    resume: str = None,
    reinit: bool = False,
    entity: str = settings.SETTINGS['WANDB_ENTITY'],
    **kwargs
):
    '''
    Wrapper around the wandb.init function. Re-implemented to provide handy type hints and implement slightly extended functionality
    
    `name` - name of the run \n
    `model` - sklearn model object, see `utils.generate_wandb_project_name` for details \n
    `config` - inputs to your job, like hyperparameters for a model \n
    `target_month` - the month for which prediction is made; pass exactly `FeatureExtractor.target_month` object \n
    `project` - Weights & Biases project to which the run should be published
        Each project refers to a certain model \n
    `group` - the group to which the run belongs \n
    `job_type` - type of the run in terms of whether the is done on training or test set
        - train: performance on train set
        - test: performance on test set
        - tuning_train: parameters tuning performance on train set
        - tuning_test: parameters tuning performance on test set \n
    `add_timestamp_to_name` - if adding timestamp of the run is required \n
    `resume` - if resuming run is allowed \n
    `reinit` - if calling `wandb.init` multiple times within a run is allowed \n
    `entity` - Weights & Biases team to which the run should be published
    '''
    project = generate_wandb_project_name(model)

    run_id = wandb.util.generate_id()
    
    name = f'{name}_month{target_month}'
    
    if add_timestamp_to_name:
        name = f"{name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=config,
        group=group,
        job_type=job_type,
        id=run_id,
        reinit=reinit,
        resume=resume,
        **kwargs
    )
    return run


def check_if_artifact_exists(
    project: str,
    artifact_name: str,
    entity: str = settings.SETTINGS['WANDB_ENTITY'],
) -> bool:
    '''
    Utiliy function that checks if a Weights & Biases artifact exists
    '''
    try:
        get_artifact(artifact_name, project, entity)
        return True
    except wandb.errors.CommError:
        return False


def get_artifact(
    project: str,
    artifact_name: str,
    entity: str = settings.SETTINGS['WANDB_ENTITY'],
) -> wandb.Artifact:
    '''
    Get the latest version of a W&B artifact
    '''
    api = wandb.Api()
    artifact = api.artifact(f"{entity}/{project}/{artifact_name}:latest")
    return artifact


def parse_classification_report(report: Union[Literal['sklearn.metrics.classification_report'], dict]) -> dict:
    '''
    Extract required metrics from `sklearn.metrics.classification_report`
    and transform it into `wandb.Artefact` friendly format

    report - `sklearn.metrics.classifcation_report(..., output_dict=True)`; report as dictionary
    '''
    new_dict = {
        'accuracy': None,
        'precision': [],
        'recall': [],
        'f1-score': []
    }
    new_dict['accuracy'] = report['accuracy']
    for k in (['0', '1', 'macro avg']):
        for metric in ['precision', 'recall', 'f1-score']:
            new_dict[metric].append({f'{k}_{metric}': report[k][metric]})

    return new_dict


# function from shap package
def format_value(s, format_str):
    """ Strips trailing zeros and uses a unicode minus sign.
    """

    if not issubclass(type(s), str):
        s = format_str % s
    s = re.sub(r'\.?0+$', '', s)
    if s[0] == "-":
        s = "\u2212" + s[1:]
    return s


# function from shap package
def safe_isinstance(obj, class_path_str):
    """
    Acts as a safe version of isinstance without having to explicitly
    import packages which may not exist in the users environment.

    Checks if obj is an instance of type specified by class_path_str.

    Parameters
    ----------
    obj: Any
        Some object you want to test against
    class_path_str: str or list
        A string or list of strings specifying full class paths
        Example: `sklearn.ensemble.RandomForestRegressor`

    Returns
    --------
    bool: True if isinstance is true and the package exists, False otherwise
    """
    if isinstance(class_path_str, str):
        class_path_strs = [class_path_str]
    elif isinstance(class_path_str, list) or isinstance(class_path_str, tuple):
        class_path_strs = class_path_str
    else:
        class_path_strs = ['']

    # try each module path in order
    for class_path_str in class_path_strs:
        if "." not in class_path_str:
            raise ValueError("class_path_str must be a string or list of strings specifying a full \
                module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'")

        # Splits on last occurrence of "."
        module_name, class_name = class_path_str.rsplit(".", 1)

        # here we don't check further if the model is not imported, since we shouldn't have
        # an object of that types passed to us if the model the type is from has never been
        # imported. (and we don't want to import lots of new modules for no reason)
        if module_name not in sys.modules:
            continue

        module = sys.modules[module_name]

        #Get class
        _class = getattr(module, class_name, None)

        if _class is None:
            continue

        if isinstance(obj, _class):
            return True

    return False