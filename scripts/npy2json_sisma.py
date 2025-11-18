import json
import numpy as np
import os


def convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, bool):
        return bool(obj)
    else:
        return obj

def convert_npy_to_json():
    for model in range(15):
        folder_name = f'../models/PINN_ext_model_{model}'
        try:
            # if os.path.exists(folder_name+'/model_params.json'):
            #     print(f'Model {model} JSON already exists, skipping.')
            #     continue  # JSON already exists, skip conversion
            model_params = np.load(folder_name+'/model_params.npy', allow_pickle=True).item()
            model_params = convert(model_params)
            model_params_json = os.path.join(folder_name, 'model_params.json')
            with open(model_params_json, 'w') as f:
                json.dump(model_params, f)
            print(f'Model {model} converted to JSON')
        except:
            print(f'Error converting model {model} or file not found.')
            continue

if __name__ == "__main__":

    convert_npy_to_json()