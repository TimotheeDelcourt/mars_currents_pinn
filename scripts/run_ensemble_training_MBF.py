import configs.config_training_MBF as config
import os
import torch
import bootstrap_sampling
import shutil
from torch import optim
from training import train, train_noval
from neuralnets import NeuralNet_indep
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
# import utils.utils as utils
import utils.npy2json_sisma as npy2json
import json

while (os.path.basename(os.getcwd()) != 'mars_currents_pinn'):
    os.chdir('../')
    


def run_ensemble_training():

  

       
       
    ensemble_size = config.training_config['ensemble_size']

    for _ in range(ensemble_size):
   
        print('Making folder')
        # Make folder-------------------------------------------------
        counter = config.training_config['bootstrap_counter_start']
        add_str  = config.training_config['add_folder_str']
        base_folder_name = f'models/MBF/PINN_ext_model_'
        
        # Keep creating new folders with incremented names until one with a unique name is found
        while True:
            folder_name = base_folder_name+str(counter)
            # Check if the folder exists
            if not os.path.exists(folder_name):
                # If the folder does not exist, create it
                os.makedirs(folder_name)
                break
            else:
                # If the folder exists, increment the counter and try again
                counter += 1
        print(f'Training model {counter}...')

        # Save the current config file in the folder----------------
        
        # and training script
        shutil.copyfile('scripts/training.py', folder_name+'/training.py')
        shutil.copyfile('scripts/neuralnets.py', folder_name+'/neuralnets.py')
        shutil.copyfile('scripts/run_ensemble_training_MBF.py', folder_name+'/run_ensemble_training.py')
        shutil.copyfile('scripts/configs/config_training_MBF.py', folder_name+'/config_training_MBF.py')
        os.makedirs(folder_name+'/models/')

        # Load and sample datasets ----------------------------------
        print('Loading data')
        input_xyz = torch.load('data/position_pc_xyz.pt')
        # input_sph = torch.load('data/position_pc_spherical.pt')
        input_sph = torch.load('data/position_mso_spherical.pt')
        alt = input_sph[:,0]
        
        crustal_field_mbf = torch.load('data/crustal_field_pc_xyz.pt')
        observation_mbf = torch.load('data/observation_pc.pt')
        target = observation_mbf - crustal_field_mbf

        if config.training_config['random_parameters'] == False:
            num_neurons_per_layer=config.training_config['num_neurons_per_layer']
            alt_max = config.training_config['altitude_max']
            # l1_lambda = config.training_config['l1_lambda']
            include_alt = config.training_config['include_alt']
            lim = 1000
        elif config.training_config['random_parameters'] == True:
            include_alt = False #np.random.choice([True,False])
            alt_max_list = config.training_config['altitudes_max']
            if len(alt_max_list) == 1:
                alt_max = alt_max_list[0]
            else:
                alt_max = np.random.randint(alt_max_list[0], alt_max_list[1]+1)
            # l1_lambda = np.random.choice(config.training_config['l1_lambdas'])
            num_neurons_per_layer = np.random.randint(config.training_config['nums_neurons_per_layer'][0], config.training_config['nums_neurons_per_layer'][1]+1)
            lim = np.random.choice(config.training_config['crop_outlier'])

        smoothness_lambda = 1e10
        l1_lambda = 0
        print(f'Altitude max: {alt_max} km')
        print(f'Number of neurons per layer: {num_neurons_per_layer}')
        # print(f'L1 lambda: {l1_lambda}')
        print(f'Include altitude: {include_alt}')
        print(f'Smoothness lambda: {smoothness_lambda}')
        print(f'Crop limit: {lim} nT')

        condition = (alt <= alt_max) & torch.all((target <= lim) & (target >= -lim), dim=1)

        # select data when subsolar point is at longitude 180 +/- 15° ------------------------
        subsolar = torch.load('data/subsolar_lat_lon.pt')
        condition2 = (subsolar[:,1] >= 165) | (subsolar[:,1] <= -165)
        condition = condition & condition2
        # ------------------------------------------------------------------------------------


        target = target[condition]
        input = input_xyz[condition]
        num_inputs = 3
        alt_mean = None
        alt_std = None


        xyz_mean = 0#torch.mean(input[:, :3]).item()
        xyz_std = torch.std(input[:, :3]).item()
        
        model_params = {
            'xyz_mean': xyz_mean,
            'xyz_std': xyz_std,
            'alt_mean': alt_mean,
            'alt_std': alt_std,
            'num_inputs': num_inputs,
            'l1_lambda': l1_lambda,
            'num_neurons_per_layer': num_neurons_per_layer,
            'alt_max': alt_max,
            'crop_limit': lim,
            'smoothness_lambda': smoothness_lambda,
            'include_alt': include_alt,
        }
        np.save(folder_name+'/model_params.npy', model_params)

        model_params_json = npy2json.convert(model_params)
        with open(folder_name+'/model_params.json', 'w') as f:
            # f.write(model_params_json)
            json.dump(model_params_json, f)

        print('Input shape: ', input.shape)

        # Device ---------------------------------------------------
        if torch.cuda.is_available():
            DEVICE = torch.device('cuda')
        else:
            DEVICE = 'cpu'
        print(f'''Device: {DEVICE}''')

        # Load network ---------------------------------------------
        model = NeuralNet_indep(
            num_hidden_layers=config.training_config['num_hidden_layers'],
            num_neurons_per_layer=num_neurons_per_layer,
            xyz_mean=xyz_mean,
            xyz_std=xyz_std,
            num_inputs=num_inputs,
            alt_mean=alt_mean,
            alt_std=alt_std,
            activation=config.training_config['activation']
        ).to(DEVICE)

        # parameters -------------------------------------------------
        num_epochs = config.training_config['num_epochs']
        optimizer = optim.Adam(model.parameters(), lr=config.training_config['learning_rate'])
        n_cpus = config.training_config['n_cpus']
        lossfn = config.training_config['lossfn']
        batch_size = config.training_config['batch_size']

        if config.training_config['validation']:
            orbit_nb = torch.load('data/orbit_nb.pt')
            orbit_nb = orbit_nb[condition]
            train_loader, val_loader = bootstrap_sampling.prepare_bootstrap_dataloaders(input, target, orbit_nb, 
                                                                                    batch_size,
                                                                                    n_cpus,
                                                                                    replacement=config.training_config['sample_with_replacement'],
                                                                                    )
            
            train(model,train_loader,val_loader, num_epochs, optimizer, DEVICE,
                folder_name, n_cpus, lossfn, l1_lambda=l1_lambda, smoothness_lambda=smoothness_lambda)
        else:
            train_dataset = TensorDataset(input, target)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpus)

            train_noval(model, train_loader, num_epochs, optimizer, DEVICE,
                        folder_name, n_cpus, lossfn, l1_lambda = config.training_config['l1_lambda'])
            
        print('')
        del model

if __name__ == "__main__":
    
    run_ensemble_training()


        



