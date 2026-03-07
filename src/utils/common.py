import yaml


def read_yaml_file(file_path: str):

    with open(file_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file)