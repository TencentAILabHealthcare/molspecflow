# utils/config_loader.py
import yaml
import os
from easydict import EasyDict as edict

def load_config(config_path: str) -> edict:
    """
    Loads a YAML configuration file, handling inheritance from a 'base' config.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        edict: An EasyDict object containing the final configuration.
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # If a 'base' config is specified, load it first and merge
    if 'base' in config_dict:
        base_path = os.path.join(os.path.dirname(config_path), config_dict['base'])
        base_config = load_config(base_path) # Recursive call
        
        # The update_dict function handles nested dictionary updates
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        base_config = update_dict(base_config, config_dict)
        config_dict = base_config
        del config_dict['base'] # Clean up the 'base' key

    return edict(config_dict)
