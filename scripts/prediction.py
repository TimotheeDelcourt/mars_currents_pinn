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
try:
    import torch_directml
    # device = torch_directml.device()
    device = torch.device('privateuseone:0')
except:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = 'cpu'
print('Working on device: ', device)

# while os.path.basename(os.getcwd()) != 'project':
#     os.chdir('../')
 

def predict(input, k, minibatch=config.prediction_config['minibatch']):
    '''
    Input must be a torch tensor of shape (n,4) with columns: X,Y,Z [km], alt [km]
    k: int, number of the bootstrap model to use.
    Output: tuple of (1) torch tensor of shape (n,3) with columns: Bx, By, Bz [nT]; (2) torch tensor of shape (n,3) with columns: Jx, Jy, Jz [mA/m2]
    device = GPU if minibatch = 0, else CPU.
    '''
    print('Starting prediction, bootstrap model', k)
    
 
    # Load model -----------------------------------------------
    folder_name = 'models/PINN_ext_bootstrap_'+str(k)
    # Load the script as a module
    try:
        spec = importlib.util.spec_from_file_location("neuralnets", folder_name+"/neuralnets.py")
        neuralnets_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(neuralnets_module)
        NeuralNet = neuralnets_module.NeuralNet
        print('Imported neuralnets from', folder_name)
    except:
        from neuralnets import NeuralNet

    model = NeuralNet().to(device)
    file_name = folder_name+f"/model{config.prediction_config['epoch_nb']}.pt"
    network = torch.load(file_name, map_location=torch.device("cpu"))
    # network = {k: v.to(device) for k, v in network.items()}
    model.load_state_dict(network)

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
    


def ensemble_predict(input_type=config.prediction_config['input_type'],nb_bootstraps = config.prediction_config['bootstrap_max']):
    
    '''
    input_type: str, 'fibonacci_sphere' or 'meshgrid'
    '''

    if input_type == 'fibonacci_sphere':
        n = config.prediction_config['num_samples']
        if config.prediction_config['regional']:
            df, tensor_cart, n = utils.fibonacci_sphere_restricted(samples = n,
                                                alt = config.prediction_config['alt'],
                                                lat_deg = config.prediction_config['lat_deg'],
                                                lon_deg = config.prediction_config['lon_deg'])
        else:
            df, tensor_cart = utils.fibonacci_sphere(samples = n,
                                                alt = config.prediction_config['alt'])
        
    elif input_type == 'data_input':
        if config.prediction_config['data_input_file'] == 'MGS':
            file_name = 'data/MGS_MO_obs_input.pt'
        elif config.prediction_config['data_input_file'] == 'ABSPOday':
            file_name = 'data/ABSPOday_input.pt'
        elif config.prediction_config['data_input_file'] == 'MAV':
            file_name = 'data/MAV_obs_input_2025.pt'
        else:
            file_name = 'data/'+config.prediction_config['data_input_file']+'_obs_cleaned_input.pt'
        tensor_sph = torch.load(file_name)
        tensor_cart = utils.spherical_to_cartesian_torch(tensor_sph[:,0], tensor_sph[:,1], tensor_sph[:,2])
        n = len(tensor_cart)

    elif input_type == 'other':
        try:
            df = pd.read_pickle(config.prediction_config['other_path'])
        except:
            df = pd.read_csv(config.prediction_config['other_path'], header = 0)
        df['alt'] = config.prediction_config['alt']
        r = torch.tensor(df['alt'].values + 3390.0, dtype=torch.float32)
        lat_deg = torch.tensor(df['lat'].values, dtype=torch.float32)
        lon_deg = torch.tensor(df['lon'].values, dtype=torch.float32)
        lon_rad = torch.deg2rad(lon_deg)
        colat_rad = torch.deg2rad(90 - lat_deg)
        tensor_cart = utils.spherical_to_cartesian_torch(r, colat_rad, lon_rad)
        n = len(tensor_cart)

    elif input_type == 'grid':
        lmax = 161
        n_grid_SH = (lmax+1)*2
        df, lmax2 = utils.generate_input_grid(n = n_grid_SH, save = 0)
        lmax2 = int(lmax2)
        assert lmax == lmax2, 'lmax should be the same as in the input grid generation'
        df['alt'] = config.prediction_config['alt']
        r = torch.tensor(df['alt'].values + 3393.5, dtype=torch.float32)
        lat_deg = torch.tensor(df['lat'].values, dtype=torch.float32)
        lon_deg = torch.tensor(df['lon'].values, dtype=torch.float32)
        lon_rad = torch.deg2rad(lon_deg)
        colat_rad = torch.deg2rad(90 - lat_deg)
        tensor_cart = utils.spherical_to_cartesian_torch(r, colat_rad, lon_rad)
        n = len(tensor_cart)
        
    if config.prediction_config['V only']:
        running_V = torch.zeros(size=(n,))
        count = 0
        for k in range(nb_bootstraps+1):
            if config.prediction_config['data'] == 'all':
                folder_name = 'outputs/models/PINN_bootstrap_'+str(k)
            elif config.prediction_config['data'] == 2017:
                folder_name = 'outputs/models/PINN_L19data_bootstrap_'+str(k)
            else:
                print('Please provide a valid data option, i.e. "all" or 2017')
                return
            
            if os.path.exists(folder_name):
                count += 1
                result_V = predict(tensor_cart, k)
                running_V += result_V.to('cpu').detach()
                print(f'Bootstrap {k} done')
                # clear gpu memory
                del result_V
        
        V_final = running_V/count

        if config.prediction_config['data'] == 'all':
            model_name = 'PINN_2025'
        elif config.prediction_config['data'] == 2017:
            model_name = 'PINN17'

       
        df['V'] = V_final
        n_grid = len(np.unique(df['lat']))

        output_name = 'outputs/predictions/synthetic/'+model_name+'_ensemble_'+str(count)+'models_'+str(config.prediction_config['alt'])+'km'
        if input_type == 'other':
            output_name += '_' + config.prediction_config['other_name']
        output_name += f'_n{n_grid}_V'
        
        df = df.drop(columns=['alt'])
        df.to_csv(output_name+'.csv', index=False)
    
    else:
        prediction_ensemble = torch.zeros(size=(n,4))
        std_ensemble = torch.zeros(size=(n,4))
        count = 0
        for k in range(nb_bootstraps+1):
            if config.prediction_config['data'] == 'all':
                folder_name = 'outputs/models/PINN_2025_bootstrap_'+str(k)
            elif config.prediction_config['data'] == 2017:
                folder_name = 'outputs/models/PINN_L19data_bootstrap_'+str(k)
            else:
                print('Please provide a valid data option, i.e. "all" or 2017')
                return
            
            if os.path.exists(folder_name):
                count += 1

                if config.prediction_config['minibatch'] == 0:
                    dev0 = device
                elif config.prediction_config['minibatch'] == 1:
                    dev0 = 'cpu'

                if (input_type == 'fibonacci_sphere') | (input_type == 'other') | (input_type == 'grid'):
                    result_vector = predict(tensor_cart, k)
                    result_scalar = torch.sqrt(result_vector[:,0]**2 + result_vector[:,1]**2 + result_vector[:,2]**2)
                    Br, Bt, Bp = utils.field_cart_to_spher(result_vector[:,0], result_vector[:,1], result_vector[:,2],
                                                        lat_deg = df['lat'], lon_deg = df['lon'], device = dev0)
                    
                elif (input_type == 'data_input'):
                    result_vector = predict(tensor_cart, k)
                    result_scalar = torch.sqrt(result_vector[:,0]**2 + result_vector[:,1]**2 + result_vector[:,2]**2)
                    Br, Bt, Bp = utils.field_cart_to_spher(result_vector[:,0], result_vector[:,1], result_vector[:,2],
                                                        colat_rad= tensor_sph[:,1], lon_rad = tensor_sph[:,2], device = dev0)
                    
                result = torch.stack((Br, Bt, Bp, result_scalar), dim=1)
                if config.prediction_config['minibatch'] == 0:
                    result = result.to('cpu').detach()
                # clear gpu memory
                del result_vector, result_scalar, Br, Bt, Bp
                prediction_ensemble += result
                std_ensemble += result**2
                print(f'Bootstrap {k} done')
        
        prediction_ensemble /= count
        std_ensemble = torch.sqrt(std_ensemble/count - prediction_ensemble**2)

        if config.prediction_config['data'] == 'all':
            model_name = 'PINN_2025'
        elif config.prediction_config['data'] == 2017:
            model_name = 'PINN17'

        if (input_type == 'fibonacci_sphere') or (input_type == 'other'):
            df['Br'] = prediction_ensemble[:,0]
            df['Bt'] = prediction_ensemble[:,1]
            df['Bp'] = prediction_ensemble[:,2]
            df['B'] = prediction_ensemble[:,3]
            df['Br_std'] = std_ensemble[:,0]
            df['Bt_std'] = std_ensemble[:,1]
            df['Bp_std'] = std_ensemble[:,2]
            df['B_std'] = std_ensemble[:,3]

            output_name = 'outputs/predictions/synthetic/'+model_name+'_ensemble_'+str(count)+'models_'+str(config.prediction_config['alt'])+'km'
            if input_type == 'fibonacci_sphere':
                output_name += '_fibonacci'
            elif input_type == 'other':
                output_name += '_' + config.prediction_config['other_name']
            
            if config.prediction_config['regional']:
                output_name += '_regional'

            # pd.to_pickle(df, output_name+'.pkl')
            df = df.drop(columns=['alt'])
            df.to_csv(output_name+'.csv', index=False)

        elif input_type == 'data_input':
            torch.save(prediction_ensemble, 'outputs/predictions/data_input/'+model_name+'_ensemble_'+str(count)+'models_'+config.prediction_config['data_input_file']+'.pt')
            torch.save(std_ensemble, 'outputs/predictions/data_input/'+model_name+'_ensemble_'+str(count)+'models_'+config.prediction_config['data_input_file']+'_std.pt')
        
        elif input_type == 'grid':
            grid_Br = prediction_ensemble[:,0].reshape(n_grid_SH, 2*n_grid_SH)
            grid_Bt = prediction_ensemble[:,1].reshape(n_grid_SH, 2*n_grid_SH)
            grid_Bp = prediction_ensemble[:,2].reshape(n_grid_SH, 2*n_grid_SH)

            np.save('outputs/predictions/synthetic/PINN_2025_grid_130km_Br_lmax161.npy', grid_Br)
            np.save('outputs/predictions/synthetic/PINN_2025_grid_130km_Bt_lmax161.npy', grid_Bt)
            np.save('outputs/predictions/synthetic/PINN_2025_grid_130km_Bp_lmax161.npy', grid_Bp)


if __name__ == '__main__':

  
    if config.predict_ensemble:
        ensemble_predict()
    
    if config.predict_single_model:
        n = config.prediction_config['num_samples']
        df, input_tensor = utils.fibonacci_sphere(samples = n,   alt = config.prediction_config['alt'])
        alt = torch.ones(len(df))*config.prediction_config['alt']
        alt = alt.unsqueeze(1)
        input_tensor = torch.concatenate((input_tensor, alt), dim=1)
        B, J = predict(input_tensor, config.prediction_config['bootstrap_nb'])
        df['Bx'] = B[:,0].to('cpu').detach()
        df['By'] = B[:,1].to('cpu').detach()
        df['Bz'] = B[:,2].to('cpu').detach()
        df['Jx'] = J[:,0].to('cpu').detach()
        df['Jy'] = J[:,1].to('cpu').detach()
        df['Jz'] = J[:,2].to('cpu').detach()
        df.to_csv(f"predictions/PINN_MSO_model{config.prediction_config['bootstrap_nb']}_epoch{config.prediction_config['epoch_nb']}_{config.prediction_config['alt']}km_fibonacci.csv", index=False)
        print(df)