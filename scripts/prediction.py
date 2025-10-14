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
print('Working on device: ', device)

# while os.path.basename(os.getcwd()) != 'project':
#     os.chdir('../')
 

def predict(input, minibatch=config.prediction_config['minibatch']):
    '''
    Input must be a torch tensor of shape (n,4) with columns: X,Y,Z [km], alt [km]
    k: int, number of the bootstrap model to use.
    Output: tuple of (1) torch tensor of shape (n,3) with columns: Bx, By, Bz [nT]; (2) torch tensor of shape (n,3) with columns: Jx, Jy, Jz [mA/m2]
    device = GPU if minibatch = 0, else CPU.
    '''
    
    
 
    # Load model -----------------------------------------------
    if config.prediction_config['bootstrap_nb'] is None:
        k = config.prediction_config['model_nb']
        folder_name = 'models/PINN_ext_model_'+str(k)
        print('Starting prediction, model', k)
    else:
        k = config.prediction_config['bootstrap_nb']
        folder_name = 'models/PINN_ext_bootstrap_'+str(k)
        print('Starting prediction, bootstrap model', k)

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
    std_params = torch.load(folder_name+'/std_params.pt')
    model = NeuralNet(
                num_hidden_layers=training_config['num_hidden_layers'],
                num_neurons_per_layer=training_config['num_neurons_per_layer'],
                xyz_mean=std_params[0],
                xyz_std=std_params[1],
                # alt_mean=675.7549,
                # alt_std=411.4479,
                activation=training_config['activation'])
    # except:
    #     from neuralnets import NeuralNet
    #     model = NeuralNet().to(device)

    epoch_nb = config.prediction_config['epoch_nb']
    if epoch_nb == None:
        file_name = folder_name+"/models/model.pt"
    else:
        file_name = folder_name+f"/models/model{epoch_nb}.pt"
    network = torch.load(file_name, map_location=device)
    # network = {k: v.to(device) for k, v in network.items()}
    model.load_state_dict(network)

    # l1 = sum(p.abs().sum() for p in model.parameters()).item()
    # print(l1)
    # assert 1 == 0

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
    





if __name__ == '__main__':

  
    # if config.predict_ensemble:
    #     ensemble_predict()
    
    # if config.predict_single_model:
    if True:
        n = config.prediction_config['num_samples']
        df, input_tensor = utils.fibonacci_sphere(samples = n,   alt = config.prediction_config['alt'])
        # alt = torch.ones(len(df))*config.prediction_config['alt']
        # alt = alt.unsqueeze(1)
        # input_tensor = torch.concatenate((input_tensor, alt), dim=1)
        B, J = predict(input_tensor)
        # df.drop(columns=['sin_colat','cos_colat','sin_lon','cos_lon','colat'], inplace=True)
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