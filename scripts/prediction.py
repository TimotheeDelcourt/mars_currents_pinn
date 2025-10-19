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
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = 'cpu'
# print('Working on device: ', device)

# while os.path.basename(os.getcwd()) != 'project':
#     os.chdir('../')

def generate_input():
    n = config.prediction_config['num_samples']
    df, input_tensor = utils.fibonacci_sphere(samples = n,   alt = config.prediction_config['alt'])
    alt = torch.ones(len(df))*config.prediction_config['alt']
    alt = alt.unsqueeze(1)
    input_tensor = torch.concatenate((input_tensor, alt), dim=1)
    return df, input_tensor
 

def predict(input, k , minibatch=config.prediction_config['minibatch']):
    '''
    Input must be a torch tensor of shape (n,4) with columns: X,Y,Z [km], alt [km]
    k: int, number of the bootstrap model to use.
    Output: tuple of (1) torch tensor of shape (n,3) with columns: Bx, By, Bz [nT]; (2) torch tensor of shape (n,3) with columns: Jx, Jy, Jz [mA/m2]
    device = GPU if minibatch = 0, else CPU.
    '''
     
    # Load model -----------------------------------------------
    # if config.prediction_config['bootstrap_nb'] is None:
    #     k = config.prediction_config['model_nb']
    folder_name = 'models/PINN_ext_model_'+str(k)
    #     print('Starting prediction, model', k)
    # else:
    #     k = config.prediction_config['bootstrap_nb']
    #     folder_name = 'models/PINN_ext_bootstrap_'+str(k)
    #     print('Starting prediction, bootstrap model', k)

    model_params = np.load(folder_name+'/model_params.npy', allow_pickle=True).item()
    if model_params['num_inputs'] == 3:
        input = input[:, :3]

    # Load the script as a module
    # try:
    spec = importlib.util.spec_from_file_location("neuralnets", folder_name+"/neuralnets.py")
    neuralnets_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(neuralnets_module)
    NeuralNet = neuralnets_module.NeuralNet_indep
    print('Imported neuralnets from', folder_name)
    spec_config = importlib.util.spec_from_file_location("config_training", folder_name+"/config_training.py")
    config_training_module = importlib.util.module_from_spec(spec_config)
    spec_config.loader.exec_module(config_training_module)
    training_config = config_training_module.training_config

    # std_params = torch.load(folder_name+'/std_params.pt')
    # troubleshoot -------------------------------------------------------------------
    # (that's because the randomly selected num_neurons_per_layer were not recorded)
    # for num_neurons_per_layer in range(7,11):
    #     try:
    model = NeuralNet(
                num_hidden_layers=1,#training_config['num_hidden_layers'],
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
        print('Starting minibatch prediction')
        for batch, index in zip(dataloader, indexloader):
            input_batch = batch.to(device).requires_grad_(True)
            A_pred_batch = model(input_batch)
            B_pred_batch = curl_differentiable(input_batch, A_pred_batch)
            J_pred_batch = ( 8.8541878188*1e-12) * (299792458**2) * curl_differentiable(input_batch, B_pred_batch)
            B_pred[index] = B_pred_batch.to('cpu').detach()
            J_pred[index] = J_pred_batch.to('cpu').detach()

            del input_batch, A_pred_batch, B_pred_batch, J_pred_batch
            percent = index[-1]*100/n
            if percent % 10 < 0.1:
                print(f'Samples {percent} % done', end='\r')
        
        J_pred /= 1000 # because distance is in km, initial result has units of 1e-3*nA/m2

    return (B_pred, J_pred)
    
def predict_ensemble():
    df, input_tensor = generate_input()

    k_start = config.prediction_config['models_start_stop'][0]
    k_stop  = config.prediction_config['models_start_stop'][1]

    B_sum = None
    B_sum_sq = None
    J_sum = None  
    J_sum_sq = None
    n_models = 0

    for i, model in enumerate(range(k_start, k_stop+1)):
        try:
            B, J = predict(input_tensor, model)
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
        except:
            continue

    # Calculate statistics
    B_mean = B_sum / n_models
    B_std = torch.sqrt((B_sum_sq / n_models) - B_mean.pow(2))
    J_mean = J_sum / n_models  
    J_std = torch.sqrt((J_sum_sq / n_models) - J_mean.pow(2))

    df['Bx'] = B_mean[:,0]
    df['By'] = B_mean[:,1]
    df['Bz'] = B_mean[:,2]
    df['Jx'] = J_mean[:,0]
    df['Jy'] = J_mean[:,1]
    df['Jz'] = J_mean[:,2]

    df['Bx_std'] = B_std[:,0]
    df['By_std'] = B_std[:,1]
    df['Bz_std'] = B_std[:,2]
    df['Jx_std'] = J_std[:,0]
    df['Jy_std'] = J_std[:,1]
    df['Jz_std'] = J_std[:,2]

    Br, Bt, Bp = utils.field_cart_to_spher(B_mean[:,0], B_mean[:,1], B_mean[:,2],
                                        lat_deg = df['lat'], lon_deg = df['lon'], device = device)
    df['Br'] = Br
    df['Bt'] = Bt
    df['Bp'] = Bp
    del Br, Bt, Bp
    
    Jr, Jt, Jp = utils.field_cart_to_spher(J_mean[:,0], J_mean[:,1], J_mean[:,2],
                                        lat_deg = df['lat'], lon_deg = df['lon'], device = device)
    df['Jr'] = Jr
    df['Jt'] = Jt
    df['Jp'] = Jp
    del Jr, Jt, Jp

    Br_std, Bt_std, Bp_std = utils.field_cart_to_spher(B_std[:,0], B_std[:,1], B_std[:,2],
                                        lat_deg = df['lat'], lon_deg = df['lon'], device = device)
    df['Br_std'] = Br_std
    df['Bt_std'] = Bt_std
    df['Bp_std'] = Bp_std
    del Br_std, Bt_std, Bp_std
    
    Jr_std, Jt_std, Jp_std = utils.field_cart_to_spher(J_std[:,0], J_std[:,1], J_std[:,2],
                                        lat_deg = df['lat'], lon_deg = df['lon'], device = device)
    df['Jr_std'] = Jr_std
    df['Jt_std'] = Jt_std
    df['Jp_std'] = Jp_std
    del Jr_std, Jt_std, Jp_std

    df.to_csv(f"predictions/PINN_MSO_ensemble_model{k_start}to{k_stop}_{config.prediction_config['alt']}km_fibonacci.csv", index=False)

    print(df)

def predict_single():
    df, input_tensor = generate_input()

    B, J = predict(input_tensor, k = config.prediction_config['model_nb'])
    df['Bx'] = B[:,0].to('cpu').detach()
    df['By'] = B[:,1].to('cpu').detach()
    df['Bz'] = B[:,2].to('cpu').detach()
    df['Jx'] = J[:,0].to('cpu').detach()
    df['Jy'] = J[:,1].to('cpu').detach()
    df['Jz'] = J[:,2].to('cpu').detach()
    Br, Bt, Bp = utils.field_cart_to_spher(B[:,0], B[:,1], B[:,2],
                                        lat_deg = df['lat'], lon_deg = df['lon'], device = device)
    Jr, Jt, Jp = utils.field_cart_to_spher(J[:,0], J[:,1], J[:,2],
                                        lat_deg = df['lat'], lon_deg = df['lon'], device = device)
    df['Br'] = Br.to('cpu').detach()
    df['Bt'] = Bt.to('cpu').detach()
    df['Bp'] = Bp.to('cpu').detach()
    df['Jr'] = Jr.to('cpu').detach()
    df['Jt'] = Jt.to('cpu').detach()
    df['Jp'] = Jp.to('cpu').detach()
    
    epoch_nb = config.prediction_config['epoch_nb']
    if epoch_nb == None:
        epoch_nb = 'last'

    df.to_csv(f"predictions/PINN_MSO_model{config.prediction_config['model_nb']}_epoch{epoch_nb}_{config.prediction_config['alt']}km_fibonacci.csv", index=False)
    
    print(df)







if __name__ == '__main__':

  
    if config.predict_ensemble:
        predict_ensemble()

    if config.predict_single_model:
        predict_single()