import json
import numpy as np
import os

for model in range(15):
    folder_name = f'../models/PINN_ext_model_{model}'
    try:
        if os.path.exists(folder_name+'/model_params.json'):
            print(f'Model {model} JSON already exists, skipping.')
            continue  # JSON already exists, skip conversion
        model_params = np.load(folder_name+'/model_params.npy', allow_pickle=True).item()
        model_params_json = os.path.join(folder_name, 'model_params.json')
        with open(model_params_json, 'w') as f:
            json.dump(model_params, f)
        print(f'Model {model} converted to JSON')
    except:
        continue