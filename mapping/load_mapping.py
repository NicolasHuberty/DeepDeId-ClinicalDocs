import json
import os

def load_mapping(dataset_name):
    """
    Retrieve a dataset mapping from a JSON file
    Parameters:
    - dataset_name (str): Name of the dataset. The mapping file must be named
      as `[dataset_name]_mapping.json` and located in the `./mapping/` directory.
    Returns:
    - dict: Mapping from the specified JSON file, or None if the file does not exist.
    """
    dataset_mapping = "./mapping/"+dataset_name+"_mapping.json"
    if not os.path.exists(dataset_mapping):
        print(f"Error: The mapping file '{dataset_mapping}' does not exist.")
        return None
    with open(dataset_mapping, 'r') as file:
        mapping = json.load(file)
    return mapping