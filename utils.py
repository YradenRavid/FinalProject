import json


def import_json_config(config_path):
    """Import json configurations.
    
    Args:
        config_path: Absolute path to configuration .json file.
    
    Returns:
        A configuration dictionary.

    Raises:
        NameError: If the file does not exist.
    
    """
    try:
        with open(config_path) as config_json:
            config = json.load(config_json)
            config_json.close()
        return config
    except NameError as ex:
        print("Read Error: no file named %s" % config_path)
        raise ex
