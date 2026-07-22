import yaml

def read_yaml(path):
    with open(path) as file:
        content = yaml.safe_load(file)

    return content


import dill

def load_object(file_path):

    with open(
        file_path,
        "rb"
    ) as file_obj:
        
        return dill.load(
            file_obj
        )