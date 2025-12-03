import torch
import torch.utils
import torch.utils.data
import configs.config_prediction as config
import numpy as np
import utils
import os
import pandas as pd
import warnings
from curl_function import curl_differentiable
import importlib.util
import json
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = 'cpu'
print('Working on device: ', device)

# while os.path.basename(os.getcwd()) != 'project':
#     os.chdir('../')

def generate_input_fibonacci():
    n = config.prediction_config['num_samples']
    df, input_tensor = utils.fibonacci_sphere(samples = n,   alt = config.prediction_config['alt'])
    alt = torch.ones(len(df))*config.prediction_config['alt']
    alt = alt.unsqueeze(1)
    input_tensor = torch.concatenate((input_tensor, alt), dim=1)
    return df, input_tensor

def generate_input_profiles():
    n = int(np.sqrt(config.prediction_config['num_samples']))
    lon_value = np.deg2rad(config.prediction_config['lon'])

    colat_i = torch.linspace(start = 0, end = torch.pi, steps = n, dtype=torch.float32)
    alt_max = config.prediction_config['alt_max_profile']
    r_i = torch.linspace(start = 3393.5, end = 3393.5+alt_max, steps = n, dtype=torch.float32)
    colat, r = torch.meshgrid(colat_i, r_i)
    colat = colat.flatten()
    r = r.flatten()
    lon = torch.ones_like(colat) * lon_value
    input_tensor = utils.spherical_to_cartesian_torch(r, colat, lon)

    lat_deg = torch.rad2deg(torch.pi/2 - colat)
    lon_deg = torch.rad2deg(lon)
    df = pd.DataFrame({'alt':r-3393.5, 'lat':lat_deg.numpy(), 'lon':lon_deg.numpy()})
    return df, input_tensor

def generate_input_data():
    season = config.prediction_config['season']
    if season == 'summer':
        target_ls = 90
    elif season == 'winter':
        target_ls = 270
    elif season == 'spring':
        target_ls = 0
    elif season  == 'autumn':
        target_ls = 180
    elif season == 'summer_autumn':
        target_ls = 135
    elif season == 'autumn_winter':
        target_ls = 225
    elif season == 'winter_spring':
        target_ls = 315
    elif season == 'spring_summer':
        target_ls = 45
    else:
        raise ValueError('season_filter must be "summer", "winter", "spring", "autumn" or None')
    angle_half_band = 30
    ls = torch.load('data/Ls_series.pt')
    lower_bound = target_ls - angle_half_band
    upper_bound = target_ls + angle_half_band
    lower_bound = lower_bound % 360
    upper_bound = upper_bound % 360
    if lower_bound > upper_bound:
        condition1 = (ls >= lower_bound) | (ls <= upper_bound)
    else:
        condition1 = (ls <= upper_bound) & (ls >= lower_bound)

    input_sph = torch.load('data/position_mso_spherical.pt')
    condition2 = (input_sph[:,0] <= config.prediction_config['alt_max_data'])
    del input_sph
    condition = condition1 & condition2
    input_xyz = torch.load('data/position_mso.pt')[condition]
    df = pd.read_parquet('data/MAVEN_MSO_data.parquet', columns=['alt', 'lat', 'lon'])[condition.numpy()]
    # print(input_xyz.shape)
    # print(df.shape)
    # print(df)
    return df, input_xyz
 
def choose_input_type():
    input_type_str = config.prediction_config['input_type']
    if input_type_str == 'fibonacci':
        df, input_tensor = generate_input_fibonacci()
    elif input_type_str == 'profile':
        df, input_tensor = generate_input_profiles()
    elif input_type_str == 'data':
        df, input_tensor = generate_input_data()
    else:
        print('Please select a valid input type in config_prediction.py')
        return
    return df, input_tensor, input_type_str

def predict(input, k , minibatch=config.prediction_config['minibatch'],models_dir = config.prediction_config['models_dir'],verbose=True):
    '''
    Input must be a torch tensor of shape (n,4) with columns: X,Y,Z [km], alt [km]
    k: int, number of the bootstrap model to use.
    Output: tuple of (1) torch tensor of shape (n,3) with columns: Bx, By, Bz [nT]; (2) torch tensor of shape (n,3) with columns: Jx, Jy, Jz [mA/m2]
    device = GPU if minibatch = 0, else CPU.
    '''
    # Load model -----------------------------------------------
 
    
    folder_name = 'models/'+models_dir+str(k)
    # folder_name = f'models/2500km/PINN_ext_all_data_model_'+str(k)
    # folder_name = 'models/PINN_ext_smoothness_reg_'+f'{config.prediction_config["reg_nb"]:.0e}'

    try:
        model_params = np.load(folder_name+'/model_params.npy', allow_pickle=True).item()
    except:
        with open(folder_name+'/model_params.json', 'r') as f:
            model_params = json.load(f)

    if model_params['num_inputs'] == 3:
        input = input[:, :3]

    # Load the script as a module
    spec = importlib.util.spec_from_file_location("neuralnets", folder_name+"/neuralnets.py")
    neuralnets_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(neuralnets_module)
    NeuralNet = neuralnets_module.NeuralNet_indep
    if verbose:
        print('Imported neuralnets from', folder_name)
    spec_config = importlib.util.spec_from_file_location("config_training", folder_name+"/config_training.py")
    config_training_module = importlib.util.module_from_spec(spec_config)
    spec_config.loader.exec_module(config_training_module)
    training_config = config_training_module.training_config

    # troubleshoot -------------------------------------------------------------------
    # (that's because the randomly selected num_neurons_per_layer were not recorded)
    # for num_neurons_per_layer in range(7,11):
    #     try:
    model = NeuralNet(
                num_hidden_layers=training_config['num_hidden_layers'],
                num_neurons_per_layer=model_params['num_neurons_per_layer'],#training_config['num_neurons_per_layer'],
                xyz_mean=model_params['xyz_mean'],
                xyz_std=model_params['xyz_std'],
                alt_mean=model_params['alt_mean'],
                alt_std=model_params['alt_std'],
                num_inputs=model_params['num_inputs'],
                activation=training_config['activation'])
    file_name = folder_name+"/models/model_val_min.pt"
    network = torch.load(file_name, map_location=device)
    model.load_state_dict(network)
        #     break
        # except:
        #     continue
    # ---------------------------------------------------------------------------------

    # epoch_nb = config.prediction_config['epoch_nb']
    # if epoch_nb == None:
    #     file_name = folder_name+"/models/model.pt"
    # else:
    #     file_name = folder_name+f"/models/model{epoch_nb}.pt"
    # network = torch.load(file_name, map_location=device)
    # model.load_state_dict(network)

    # Predict --------------------------------------------------
    if minibatch == 0:
        input = input.to(device).requires_grad_(True)
        A_pred = model(input)
        B_pred = curl_differentiable(input, A_pred)
        J_pred = ( 8.8541878188*1e-12) * (299792458**2) * curl_differentiable(input, B_pred)
        
    elif minibatch == 1:
        num_workers = config.prediction_config['num_workers']
        batch_size = config.prediction_config['batch_size']
        n = len(input)
        B_pred = torch.zeros(n,3)
        J_pred = torch.zeros(n,3)
        dataloader = torch.utils.data.DataLoader(input, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        indexloader = torch.utils.data.DataLoader(torch.arange(n), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        if verbose:
            print('Starting minibatch prediction')
        for batch, index in zip(dataloader, indexloader):
            input_batch = batch.to(device).requires_grad_(True)
            A_pred_batch = model(input_batch)
            B_pred_batch = curl_differentiable(input_batch, A_pred_batch)
            J_pred_batch = ( 8.8541878188*1e-12) * (299792458**2) * curl_differentiable(input_batch, B_pred_batch)
            B_pred[index] = B_pred_batch.to('cpu').detach()
            J_pred[index] = J_pred_batch.to('cpu').detach()

            del input_batch, A_pred_batch, B_pred_batch, J_pred_batch
            # percent = index[-1]*100/n
            # if percent % 10 < 0.1:
            #     print(f'Samples {percent} % done', end='\r')
        
        J_pred /= 1000 # because distance is in km, initial result has units of 1e-3*nA/m2

    return (B_pred, J_pred)
    



def predict_ensemble():

    df, input_tensor, input_type_str = choose_input_type()


    k_start = config.prediction_config['models_start_stop'][0]
    k_stop  = config.prediction_config['models_start_stop'][1]

    B_sum = None
    B_sum_sq = None
    J_sum = None  
    J_sum_sq = None
    n_models = 0
    from tqdm import tqdm
    models = tqdm(range(k_start, k_stop+1))
    for i, model in enumerate(models):
    # for i, model in enumerate(range(k_start, k_stop+1)):
        try:
            B, J = predict(input_tensor, model,verbose=False)
            if B_sum is None:
                B_sum = B.clone()
                B_sum_sq = B.pow(2)
                J_sum = J.clone()
                J_sum_sq = J.pow(2)
            else:
                B_sum += B
                B_sum_sq += B.pow(2)
                J_sum += J
                J_sum_sq += J.pow(2)
            n_models += 1
            models.set_postfix_str(f'Model {model} successfully computed')
        except:
            # print('failed')
            models.set_postfix_str(f'Model {model} failed')
            continue
        

    # Calculate statistics
    B_mean = B_sum / n_models
    B_std = torch.sqrt((B_sum_sq / n_models) - B_mean.pow(2))
    J_mean = J_sum / n_models  
    J_std = torch.sqrt((J_sum_sq / n_models) - J_mean.pow(2))

    # df['Bx'] = B_mean[:,0]
    # df['By'] = B_mean[:,1]
    # df['Bz'] = B_mean[:,2]
    # df['Jx'] = J_mean[:,0]
    # df['Jy'] = J_mean[:,1]
    # df['Jz'] = J_mean[:,2]

    # df['Bx_std'] = B_std[:,0]
    # df['By_std'] = B_std[:,1]
    # df['Bz_std'] = B_std[:,2]
    # df['Jx_std'] = J_std[:,0]
    # df['Jy_std'] = J_std[:,1]
    # df['Jz_std'] = J_std[:,2]

    Br, Bt, Bp = utils.field_cart_to_spher(B_mean[:,0], B_mean[:,1], B_mean[:,2],
                                        lat_deg = df['lat'].values, lon_deg = df['lon'].values, device = device)
    df['Br'] = Br
    df['Bt'] = Bt
    df['Bp'] = Bp
    del Br, Bt, Bp
    
    Jr, Jt, Jp = utils.field_cart_to_spher(J_mean[:,0], J_mean[:,1], J_mean[:,2],
                                        lat_deg = df['lat'].values, lon_deg = df['lon'].values, device = device)
    df['Jr'] = Jr
    df['Jt'] = Jt
    df['Jp'] = Jp
    del Jr, Jt, Jp

    Br_std, Bt_std, Bp_std = utils.field_cart_to_spher(B_std[:,0], B_std[:,1], B_std[:,2],
                                        lat_deg = df['lat'].values, lon_deg = df['lon'].values, device = device)
    df['Br_std'] = Br_std
    df['Bt_std'] = Bt_std
    df['Bp_std'] = Bp_std
    del Br_std, Bt_std, Bp_std
    
    Jr_std, Jt_std, Jp_std = utils.field_cart_to_spher(J_std[:,0], J_std[:,1], J_std[:,2],
                                        lat_deg = df['lat'].values, lon_deg = df['lon'].values, device = device)
    df['Jr_std'] = Jr_std
    df['Jt_std'] = Jt_std
    df['Jp_std'] = Jp_std
    del Jr_std, Jt_std, Jp_std

    add_str = config.prediction_config['add_str']
    if add_str != '':
        add_str = '_'+add_str
    if input_type_str == 'fibonacci':
        df.to_csv(f"predictions/PINN_MSO_ensemble_models_{k_start}to{n_models}_{config.prediction_config['alt']}km_fibonacci{add_str}.csv", index=False)
    elif input_type_str == 'profile':
        alt_max = config.prediction_config['alt_max_profile']
        df.to_csv(f"predictions/PINN_MSO_ensemble_models_{k_start}to{n_models}_lon_{config.prediction_config['lon']}deg_profile{add_str}_{alt_max}km.csv", index=False)
    elif input_type_str == 'data':
        alt_max = config.prediction_config['alt_max_data']
        df.to_csv(f"predictions/data/PINN_MSO_ensemble_models_{k_start}to{n_models}{add_str}_data_{alt_max}km.csv", index=False)
    print(df)


def predict_single():
    df, input_tensor, input_type_str = choose_input_type()

    B, J = predict(input_tensor, k = config.prediction_config['model_nb'])
    # df['Bx'] = B[:,0].to('cpu').detach()
    # df['By'] = B[:,1].to('cpu').detach()
    # df['Bz'] = B[:,2].to('cpu').detach()
    # df['Jx'] = J[:,0].to('cpu').detach()
    # df['Jy'] = J[:,1].to('cpu').detach()
    # df['Jz'] = J[:,2].to('cpu').detach()
    Br, Bt, Bp = utils.field_cart_to_spher(B[:,0], B[:,1], B[:,2],
                                        lat_deg = df['lat'].values, lon_deg = df['lon'].values, device = device)
    Jr, Jt, Jp = utils.field_cart_to_spher(J[:,0], J[:,1], J[:,2],
                                        lat_deg = df['lat'].values, lon_deg = df['lon'].values, device = device)
    df['Br'] = Br.to('cpu').detach()
    df['Bt'] = Bt.to('cpu').detach()
    df['Bp'] = Bp.to('cpu').detach()
    df['Jr'] = Jr.to('cpu').detach()
    df['Jt'] = Jt.to('cpu').detach()
    df['Jp'] = Jp.to('cpu').detach()
    
    # epoch_nb = config.prediction_config['epoch_nb']
    # if epoch_nb == None:
    #     epoch_nb = 'last'

    add_str = config.prediction_config['add_str']
    if add_str != '':
        add_str = '_'+add_str
    df.to_csv(f"predictions/PINN_MSO_model{config.prediction_config['model_nb']}_{config.prediction_config['alt']}km_fibonacci{add_str}.csv", index=False)
    # df.to_csv(f"predictions/PINN_MSO_reg{config.prediction_config['reg_nb']:.0e}_{config.prediction_config['alt']}km_fibonacci.csv", index=False)
    
    print(df)


def predict_first_neuron():
    df, input, _ = choose_input_type()
    k = config.prediction_config['model_nb']
    minibatch=config.prediction_config['minibatch']

    # Load model -----------------------------------------------
    models_dir = config.prediction_config['models_dir']
    folder_name = 'models/'+models_dir+str(k)
    # folder_name = f'models/2500km/PINN_ext_all_data_model_'+str(k)
    # folder_name = 'models/PINN_ext_smoothness_reg_'+f'{config.prediction_config["reg_nb"]:.0e}'

    try:
        model_params = np.load(folder_name+'/model_params.npy', allow_pickle=True).item()
    except:
        with open(folder_name+'/model_params.json', 'r') as f:
            model_params = json.load(f)

    if model_params['num_inputs'] == 3:
        input = input[:, :3]

    # Load the script as a module
    spec = importlib.util.spec_from_file_location("neuralnets", folder_name+"/neuralnets.py")
    neuralnets_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(neuralnets_module)
    NeuralNet = neuralnets_module.NeuralNet_indep
    print('Imported neuralnets from', folder_name)
    spec_config = importlib.util.spec_from_file_location("config_training", folder_name+"/config_training.py")
    config_training_module = importlib.util.module_from_spec(spec_config)
    spec_config.loader.exec_module(config_training_module)
    training_config = config_training_module.training_config

    model = NeuralNet(
                num_hidden_layers=training_config['num_hidden_layers'],
                num_neurons_per_layer=model_params['num_neurons_per_layer'],#training_config['num_neurons_per_layer'],
                xyz_mean=model_params['xyz_mean'],
                xyz_std=model_params['xyz_std'],
                alt_mean=model_params['alt_mean'],
                alt_std=model_params['alt_std'],
                num_inputs=model_params['num_inputs'],
                activation=training_config['activation'])
    file_name = folder_name+"/models/model_val_min.pt"
    network = torch.load(file_name, map_location=device)
    model.load_state_dict(network)

    # hook first neuron of first layer -------------------------
    first_layer_output = {}
    def hook_fn(_module, _input, output):
        first_layer_output['act'] = output
    handle = model.network_x[0].register_forward_hook(hook_fn)



    # Predict --------------------------------------------------
    if minibatch == 0:
        input = input.to(device).requires_grad_(True)
        _ = model(input)
        first_neuron_values = first_layer_output['act'][:, :8]
        
        
    elif minibatch == 1:
        num_workers = config.prediction_config['num_workers']
        batch_size = config.prediction_config['batch_size']
        n = len(input)
        first_neuron_values = torch.zeros(n,8)
        dataloader = torch.utils.data.DataLoader(input, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        indexloader = torch.utils.data.DataLoader(torch.arange(n), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        print('Starting minibatch prediction')
        for batch, index in zip(dataloader, indexloader):
            input_batch = batch.to(device).requires_grad_(True)
            _ = model(input_batch)
            first_neuron_values_batch = first_layer_output['act'][:, :8]
            first_neuron_values[index] = first_neuron_values_batch.to('cpu').detach()

            del input_batch, first_neuron_values_batch

    # print(first_neuron_values)
    # print(first_neuron_values.shape)
    for i in range(8):
        df['neuron'+str(i)] = first_neuron_values[:,i]
    # print(f'Min : {torch.min(first_neuron_values):1f}, max : {torch.max(first_neuron_values):1f}')
    print(df)
    add_str = config.prediction_config['add_str']
    if add_str != '':
        add_str = '_'+add_str
    df.to_csv(f"predictions/PINN_MSO_model{config.prediction_config['model_nb']}_{config.prediction_config['alt']}km_fibonacci{add_str}_neuronoutput.csv", index=False)
            
    




if __name__ == '__main__':

  
    if config.predict_ensemble:
        predict_ensemble()

    if config.predict_single_model:
        predict_single()

    if config.predict_single_neuron:
        predict_first_neuron()


