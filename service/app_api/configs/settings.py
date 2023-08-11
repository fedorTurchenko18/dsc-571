import os

from typing import Union
from pathlib import Path
from dotenv import load_dotenv

def load_env_vars(root_dir: Union[str, Path] = './service/app_api/configs') -> dict:
    '''
    Load environment variables from .env.default and .env files
    
    root_dir - root directory of the .env files; `configs` directory of the project by default

    Returns:
        Dictionary with the environment variables
    '''
    if isinstance(root_dir, str):
        root_dir = Path(root_dir)
    load_dotenv(dotenv_path=root_dir / '.env.default')
    load_dotenv(dotenv_path=root_dir / '.env', override=True)
    return dict(os.environ)

SETTINGS = load_env_vars()